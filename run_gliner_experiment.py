"""
run_gliner_experiment.py
========================
Runs the V2 pipeline end-to-end using the GLiNER-biomed classifier as the
semantic-classification stage (Methodology §4.3.6), in place of the embedding
classifier used by run_embedding_only_experiment.py. Everything downstream of
classification — corrector V2, enricher, graph V2, association V2, paragraph
fallback, and the evaluator — is identical to that runner.

This is the ablation-3 sibling of run_embedding_only_experiment.py: same
cascade, same images, same gold, same graph/association; only the nutrient
node differs (GLiNER span acceptance instead of a cosine threshold).

CLASSIFIER
----------
  src/classification/gliner_classifier.py :: GLiNERSemanticClassifier
    - mode gliner_only : GLiNER decides every nutrient candidate (lexicon ignored)
    - mode hybrid      : lexicon NUTRIENT calls kept; GLiNER rescues UNKNOWN only
  Single span-acceptance threshold (NO margin). The default below is the value
  picked by the token-level threshold sweep (gliner_threshold_sweep.py):
      803 candidates (438 nutrient / 365 non-nutrient) -> F1-optimal tau* = 0.18
      (F1 0.891, precision 0.908, recall 0.874). See gliner_node_calibration_evidence.md.

CHANGES (normalised GT variant)
-------------------------------
  - Stage 2 uses paddleOCR_corrector_v2 (C15 canonical name normalisation)
  - GT loaded from test_set_normalized.csv instead of per-image JSON files

CHANGES (exp39 edge modes + overlap default)
--------------------------------------------
  - --row-edge-mode {overlap|cy|rank|role_rank},
    --col-edge-mode {overlap|cx|column_id} and
    --context-scope-mode {overlap|cy} are forwarded into GraphConstructorV2.
    Defaults are the tuned OVERLAP (intersection) pipeline:
    row=overlap col=overlap ctx=overlap at the graph's tuned thresholds
    (row 0.30 / col 0.20 / context 0.10; coordinate-descent sweep,
    cy/cx 40.7% -> overlap 42.3% 4F-F1). To recover the previous geometric-
    centroid behaviour pass: --row-edge-mode cy --col-edge-mode cx
    --context-scope-mode cy.

CHANGES (late-fusion merge)
---------------------------
  - --merge-modes runs DEFAULT (cy) first, then STRUCTURAL (role_rank), and
    adds ONLY role_rank's UNIQUE tuples to the default set. Implemented via
    src/matching/merged_graph_associator.py (MergedModeAssociator). The chosen
    col/context modes are forwarded to both passes.

CHANGES (stage audit)
---------------------
  - --audit writes, per image, the OUTPUT of every stage to its own file under
    outputs/{EXPERIMENT}/audit/{image}/ plus a _STAGE_AUDIT.md. Each stage's
    output IS the next stage's input.

PIPELINE
--------
  Stage 1   — PaddleOCR
  Stage 2   — PaddleOCR Corrector V2  (+ C15 canonical name normalisation)
  Stage 3   — GLiNERSemanticClassifier  (mode=gliner_only | hybrid)
  Stage 3.5 — Token Enricher
  Stage 4   — Graph V2  (row/col/context edge mode configurable; dual when --merge-modes)
  Stage 5   — Association V2 (+ paragraph fallback)
  Stage 6   — LLM-Assisted Tuple Evaluator

USAGE
-----
    python run_gliner_experiment.py --experiment exp41_gliner_tuned
    python run_gliner_experiment.py --experiment exp41_quick --no-llm

    # override the calibrated threshold or use the hybrid node:
    python run_gliner_experiment.py --experiment exp41_t030 --threshold 0.30
    python run_gliner_experiment.py --experiment exp41_hybrid --mode hybrid

    # recover the old geometric-centroid graph (cy/cx/cy):
    python run_gliner_experiment.py --experiment exp41_cycx \
        --row-edge-mode cy --col-edge-mode cx --context-scope-mode cy

    # exp39 edge-mode experiments:
    python run_gliner_experiment.py --experiment exp39a_gliner_colid \
        --col-edge-mode column_id

    # late-fusion merge (default cy + role_rank unique-add):
    python run_gliner_experiment.py --experiment exp39f_gliner_merged \
        --merge-modes --no-llm --compare exp41_gliner_tuned

    # with per-stage audit dumps (scope to a few images for inspection):
    python run_gliner_experiment.py --experiment exp41_audit \
        --audit --images 1,108,118
"""

import sys, csv, json, argparse, os
import importlib.util
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

from src.ocr.paddleocr_runner          import run_ocr_on_image
from src.utils.paddleOCR_corrector_v2  import correct_tokens            # ← V2
from src.utils.token_enricher          import TokenEnricher
from src.utils.paragraph_extractor     import (extract_from_paragraph,
                                               should_use_paragraph_mode)
from src.utils.sentence_extractor      import extract_from_sentences
from src.graph.graph_constructor_v2    import GraphConstructorV2
from src.matching.association_v2       import TupleAssociatorV2
from src.matching.merged_graph_associator import MergedModeAssociator   # ← late-fusion merge
from src.evaluation.llm_evaluator      import LLMTupleEvaluator


# ── Dynamic loader (mirrors run_embedding_only_experiment.py pattern) ─────────

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = os.path.dirname(__file__)
_gln_module = _load_module(
    'gliner_classifier',
    os.path.join(_base, 'src', 'classification', 'gliner_classifier.py'),
)
GLiNERSemanticClassifier = _gln_module.GLiNERSemanticClassifier


# ── Normalised CSV GT loader ─────────────────────────────────────────────────

def load_gt_from_csv(csv_path: str) -> list:
    """
    Load ground-truth tuples from the normalised test_set CSV.

    Expected columns:
        image_id, nutrient, quantity, unit, context, nrv_percent, serving_size

    Context is built as "context (serving_size)" when serving_size is
    present, matching the format the evaluator expects.
    """
    rows = []
    path = Path(csv_path)
    if not path.exists():
        print(f"[load_gt] ERROR: GT file not found: {csv_path}")
        return rows

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            context      = str(row.get("context",      "") or "").strip()
            serving_size = str(row.get("serving_size",  "") or "").strip()
            context_full = f"{context} ({serving_size})" if serving_size else context

            # Normalise image_id extension to lowercase
            raw_id   = str(row.get("image_id", "")).strip()
            image_id = Path(raw_id).stem + Path(raw_id).suffix.lower()

            rows.append({
                "image_id": image_id,
                "nutrient": str(row.get("nutrient", "") or "").strip(),
                "quantity": str(row.get("quantity", "") or "").strip(),
                "unit":     str(row.get("unit",     "") or "").strip(),
                "context":  context_full,
            })

    return rows


# ── Stage-audit helpers ───────────────────────────────────────────────────────

def _json_safe(obj):
    """Recursively convert numpy arrays/scalars to JSON-native types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy ndarray or numpy scalar
        return obj.tolist()
    return obj


def _dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)


def _sample(records, drop=(), n=1):
    """Pretty-print the first n records, dropping bulky keys for readability."""
    out = []
    for rec in records[:n]:
        if isinstance(rec, dict):
            rec = {k: v for k, v in rec.items() if k not in drop}
        out.append(_json_safe(rec))
    payload = out if n > 1 else (out[0] if out else {})
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _added_fields(in_records, out_records):
    """Keys present in the first OUTPUT record but not the first INPUT record."""
    if not in_records or not out_records:
        return []
    a = in_records[0] if isinstance(in_records[0], dict) else {}
    b = out_records[0] if isinstance(out_records[0], dict) else {}
    return sorted(set(b.keys()) - set(a.keys()))


def write_image_audit(audit_root, image_key, img_path,
                      tokens, corrected, correction_log, labeled,
                      enriched, enr_diag, graph, assoc_tuples, tuples,
                      used_paragraph, row_mode, col_mode):
    """
    Write per-stage OUTPUT dumps + a human-readable _STAGE_AUDIT.md
    documenting each stage's input and output for one image.

    Each stage's output is the next stage's input:
        image -> 01 ocr -> 02 corrected -> 03 classified
              -> 04 enriched -> 05 graph -> 06 tuples
    """
    d = Path(audit_root) / image_key
    d.mkdir(parents=True, exist_ok=True)

    # ── per-stage output dumps ────────────────────────────────────────
    _dump_json(d / "01_ocr_tokens.json",        tokens)
    _dump_json(d / "02_corrected_tokens.json",  corrected)
    _dump_json(d / "02_correction_log.json",    correction_log or [])
    _dump_json(d / "03_classified_tokens.json", labeled)
    _dump_json(d / "04_enriched_tokens.json",   enriched)
    _dump_json(d / "05_graph.json",             graph)
    _dump_json(d / "06_tuples.json",            tuples)            # eval input
    if used_paragraph:
        _dump_json(d / "06b_association_tuples.json", assoc_tuples)

    # ── derived summaries ─────────────────────────────────────────────
    label_dist = Counter(t.get("label", "?") for t in labeled)
    edge_dist  = Counter(e.get("type", "?") for e in graph.get("edges", []))
    enr_active = [t for t in enriched if t.get("is_enriched")]
    enr_sample = enr_active[0] if enr_active else (enriched[0] if enriched else {})
    n_noise    = sum(1 for t in enriched if not t.get("is_enriched"))
    n_assoc    = len(assoc_tuples)
    no_qty     = sum(1 for t in tuples if not t.get("quantity"))
    no_unit    = sum(1 for t in tuples if not t.get("unit"))
    no_ctx     = sum(1 for t in tuples if not t.get("context"))

    HEAVY = ('embedding_scores', 'direction', 'normal', 'center')

    md = []
    md.append(f"# Stage I/O Audit — `{image_key}`")
    md.append(f"\nGraph edge modes: **row_mode = `{row_mode}`**, "
              f"**col_mode = `{col_mode}`**")
    md.append("\nEach stage's OUTPUT is the next stage's INPUT. "
              "Full per-stage dumps are in this folder (`01_…` → `06_…`).\n")

    md.append("## Stage 1 — OCR  ·  `src/ocr/paddleocr_runner.py`")
    md.append(f"- **Input:**  image file `{img_path}`")
    md.append(f"- **Output:** {len(tokens)} raw tokens → `01_ocr_tokens.json`")
    md.append("- **Token schema:** `token, x1, y1, x2, y2, cx, cy, conf`")
    md.append(f"- **Sample:**\n```json\n{_sample(tokens)}\n```\n")

    md.append("## Stage 2 — OCR Correction  ·  `paddleOCR_corrector_v2.py`")
    md.append(f"- **Input:**  {len(tokens)} raw tokens")
    md.append(f"- **Output:** {len(corrected)} corrected tokens "
              f"({len(correction_log or [])} corrections) → "
              f"`02_corrected_tokens.json` (+ `02_correction_log.json`)")
    md.append("- **Schema:** unchanged; `token` text cleaned/normalised, "
              "tokens may be split (e.g. energy `kJ/kcal`), names canonicalised (C15)")
    md.append(f"- **Sample correction log:**\n```json\n{_sample(correction_log or [], n=3)}\n```\n")

    md.append("## Stage 3 — Semantic Classification  ·  `gliner_classifier.py` (GLiNER node)")
    md.append(f"- **Input:**  {len(corrected)} corrected tokens")
    md.append(f"- **Output:** {len(labeled)} labelled tokens → `03_classified_tokens.json`")
    md.append(f"- **Fields added:** `{', '.join(_added_fields(corrected, labeled)) or '—'}`")
    md.append("- **Distribution:** "
              + ", ".join(f"{k}={v}" for k, v in label_dist.most_common()))
    md.append("- **Decision:** cascade owns QUANTITY/UNIT/CONTEXT/NOISE; "
              "GLiNER decides NUTRIENT vs UNKNOWN for each candidate "
              "(`classification_method`, `gliner_score` recorded per token)")
    md.append(f"- **Sample:**\n```json\n{_sample(labeled, drop=HEAVY)}\n```\n")

    md.append("## Stage 3.5 — Token Enrichment  ·  `token_enricher.py`")
    md.append(f"- **Input:**  {len(labeled)} labelled tokens")
    md.append(f"- **Output:** {len(enr_active)} enriched tokens "
              f"({n_noise} NOISE preserved with `is_enriched=False`) → `04_enriched_tokens.json`")
    md.append(f"- **Fields added:** `{', '.join(_added_fields(labeled, [enr_sample])) or '—'}`")
    md.append(f"- **Structure:** rows={enr_diag.get('num_rows','?')}, "
              f"columns={enr_diag.get('num_columns','?')}, "
              f"dosage_streams={enr_diag.get('dosage_streams','?')}, "
              f"headers={enr_diag.get('headers_detected','?')}, "
              f"rank_consistent={enr_diag.get('rank_consistent','?')}")
    md.append(f"- **Sample (enriched):**\n```json\n{_sample([enr_sample], drop=HEAVY)}\n```\n")

    md.append("## Stage 4 — Semantic Graph  ·  `graph_constructor_v2.py`")
    md.append(f"- **Input:**  {len(enr_active)} enriched tokens")
    md.append(f"- **Output:** graph — {graph.get('num_nodes',0)} nodes, "
              f"{graph.get('num_edges',0)} edges → `05_graph.json`")
    md.append("- **Node schema:** all enriched fields + `id`")
    md.append("- **Edge schema:** `{src, dst, type, weight}`")
    md.append(f"- **Edge construction:** row_mode=`{row_mode}`, col_mode=`{col_mode}`")
    md.append("- **Edge types:** "
              + ", ".join(f"{k}={v}" for k, v in edge_dist.most_common()))
    md.append("")

    md.append("## Stage 5 — Tuple Association  ·  `association_v2.py`")
    md.append(f"- **Input:**  graph ({graph.get('num_nodes',0)} nodes / "
              f"{graph.get('num_edges',0)} edges)")
    md.append(f"- **Output:** {n_assoc} tuples → "
              + ("`06b_association_tuples.json`" if used_paragraph else "`06_tuples.json`"))
    md.append("- **Tuple schema:** `image_id, nutrient, quantity, unit, context` "
              "(+ internal `_score, _stream`; `source` when merged)")
    md.append(f"- **Empty fields:** no_quantity={no_qty}, no_unit={no_unit}, "
              f"no_context={no_ctx}")
    md.append(f"- **Sample tuples:**\n```json\n{_sample(assoc_tuples, drop=('_score','_stream'), n=3)}\n```\n")

    if used_paragraph:
        md.append("## Stage 5b — Paragraph Fallback  ·  `paragraph_extractor.py`  (TRIGGERED)")
        md.append(f"- Replaced association output ({n_assoc} tuples) with "
                  f"paragraph extraction ({len(tuples)} tuples → `06_tuples.json`).")
        md.append("")

    md.append("## Stage 6 — Evaluation  ·  `llm_evaluator.py`")
    md.append("- **Input:**  final tuples (`06_tuples.json`) + ground truth")
    md.append("- **Output:** per-experiment metrics (4F-F1, 3F-F1, per-field) — "
              "written at run level, not per image")
    md.append("")

    (d / "_STAGE_AUDIT.md").write_text("\n".join(md), encoding="utf-8")


# ── Config ────────────────────────────────────────────────────────────────────

GT_CSV     = 'test_set_normalized.csv'                                  # ← CSV
RAW_DIR    = 'data/raw'
CONF_THRESH = 0.30
FIELDNAMES  = ['image_id', 'nutrient', 'quantity', 'unit', 'context']

# Calibrated via the token-level threshold sweep (gliner_threshold_sweep.py):
#   803 candidates (438 nutrient / 365 non-nutrient) -> F1-optimal tau* = 0.18
#   (F1 0.891, P 0.908, R 0.874). See gliner_node_calibration_evidence.md.
GLINER_THRESHOLD = 0.18
DEFAULT_MODE     = 'gliner_only'


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
parser.add_argument('--images',    default=None,
                    help="Comma-separated image stems (e.g. '1,101,118')")
parser.add_argument('--compare',   default=None,
                    help="Prior experiment folder name to diff against")
parser.add_argument('--notes',     default='')
parser.add_argument('--mode',      default=DEFAULT_MODE,
                    choices=['gliner_only', 'hybrid'],
                    help="gliner_only (lexicon ignored) | hybrid (lexicon NUTRIENT kept, "
                         "GLiNER rescues UNKNOWN)")
parser.add_argument('--threshold', type=float, default=GLINER_THRESHOLD,
                    help=f"GLiNER span-acceptance threshold (calibrated default {GLINER_THRESHOLD})")
parser.add_argument('--model-id',  default=None,
                    help="Override the GLiNER model id (default: bi-large variant)")
parser.add_argument('--device',    default=None,
                    help="cpu | cuda | (auto-detect)")
parser.add_argument('--no-llm',    action='store_true',
                    help='Skip LLM evaluation — fast rule-based pass only')
parser.add_argument('--gt-csv',    default=GT_CSV,
                    help=f"Path to normalised GT CSV (default {GT_CSV})")
# ── exp39 edge-mode switches (forwarded to GraphConstructorV2) ──────────────
# Defaults = tuned OVERLAP (intersection) pipeline. Pass cy/cx/cy to recover the
# geometric-centroid baseline. Overlap thresholds inherit the graph defaults
# (row 0.30 / col 0.20 / ctx 0.10); threshold tuning lives in the sweep cell.
parser.add_argument('--row-edge-mode', default='overlap',
                    choices=['overlap', 'cy', 'rank', 'role_rank'],
                    help="ROW_COMPAT build: 'overlap' vertical bbox-extent (default) | "
                         "'cy' geometric centroid | 'rank' structural | "
                         "'role_rank' label-specific rank")
parser.add_argument('--col-edge-mode', default='overlap',
                    choices=['overlap', 'cx', 'column_id'],
                    help="COL_COMPAT build: 'overlap' horizontal bbox-extent (default) | "
                         "'cx' geometric centroid | 'column_id' structural")
parser.add_argument('--context-scope-mode', default='overlap',
                    choices=['overlap', 'cy'],
                    help="CONTEXT_SCOPE lateral gate: 'overlap' horizontal bbox-extent (default) | "
                         "'cy' no lateral gate (V1 behaviour)")
# ── late-fusion merge ────────────────────────────────────────────────────────
parser.add_argument('--merge-modes', action='store_true',
                    help="Run DEFAULT (cy) then STRUCTURAL (role_rank) and ADD ONLY "
                         "role_rank's UNIQUE tuples to the default set. Overrides "
                         "--row-edge-mode. col/context modes still apply to both passes.")
# ── per-stage audit ─────────────────────────────────────────────────────────
parser.add_argument('--audit', action='store_true',
                    help='Write per-stage I/O dumps + _STAGE_AUDIT.md for each image')
args = parser.parse_args()

EXPERIMENT = args.experiment
USE_LLM    = not args.no_llm
MODE       = args.mode
THRESHOLD  = args.threshold
MODEL_ID   = args.model_id
DEVICE     = args.device
GT_CSV     = args.gt_csv
ROW_EDGE_MODE  = args.row_edge_mode
COL_EDGE_MODE  = args.col_edge_mode
CTX_SCOPE_MODE = args.context_scope_mode
MERGE_MODES    = args.merge_modes
AUDIT      = args.audit

OUT_DIR    = Path(f'outputs/{EXPERIMENT}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TUPLES_CSV = OUT_DIR / 'tuples.csv'
LOG_FILE   = OUT_DIR / 'run_log.txt'
AUDIT_DIR  = OUT_DIR / 'audit'
if AUDIT:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ── Image discovery ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".tif"}

if args.images:
    wanted = {s.strip() for s in args.images.split(',') if s.strip()}
    IMAGE_FILES = [f for f in Path(RAW_DIR).iterdir()
                   if f.stem in wanted and f.suffix.lower() in IMAGE_EXTENSIONS]
else:
    IMAGE_FILES = [f for f in Path(RAW_DIR).iterdir()
                   if f.suffix.lower() in IMAGE_EXTENSIONS]
IMAGE_FILES.sort(key=lambda f: (int(f.stem) if f.stem.isdigit() else 1e9, f.name))


# ── Pipeline instances ────────────────────────────────────────────────────────
# Build the classifier FIRST (loads the GLiNER model once, then reused per image).

_clf_kwargs = dict(
    mode                 = MODE,
    threshold            = THRESHOLD,
    confidence_threshold = CONF_THRESH,
    device               = DEVICE,
)
if MODEL_ID:
    _clf_kwargs["model_id"] = MODEL_ID
classifier = GLiNERSemanticClassifier(**_clf_kwargs)

enricher    = TokenEnricher()
constructor = GraphConstructorV2({
    "row_edge_mode":      ROW_EDGE_MODE,
    "col_edge_mode":      COL_EDGE_MODE,
    "context_scope_mode": CTX_SCOPE_MODE,
    # Overlap thresholds inherit GraphConstructorV2 defaults (row 0.30 / col 0.20 /
    # ctx 0.10 — the tuned sweep values). Threshold tuning lives in the sweep cell;
    # this runner ablates MODES. Effective values are read back below for logging.
})
associator  = TupleAssociatorV2()

# Effective overlap thresholds (from the graph defaults) — for banner + log only.
_OVL = {k: constructor.config[k]
        for k in ("row_overlap_min", "col_overlap_min", "context_overlap_min")}

# Late-fusion merger (only used when --merge-modes). Runs cy then role_rank
# internally, both at the chosen col/context modes, and keeps default +
# role_rank uniques. Overlap thresholds inherit the graph defaults.
merger = (MergedModeAssociator(
    base_config={"col_edge_mode":      COL_EDGE_MODE,
                 "context_scope_mode": CTX_SCOPE_MODE},
) if MERGE_MODES else None)


# ── Banner ────────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  EXPERIMENT : {EXPERIMENT}")
print(f"  Pipeline   : V2 (PaddleOCR + Corrector V2 + GLiNER Clf + Graph V2)")
print(f"  Corrector  : paddleOCR_corrector_v2 (C15 canonical normalisation)")
print(f"  Classifier : GLiNER {MODE}  (gliner-biomed, t={THRESHOLD})")
if MERGE_MODES:
    print(f"  Graph edges: MERGE (default cy + structural role_rank, add unique) "
          f"| col={COL_EDGE_MODE} ctx={CTX_SCOPE_MODE}")
else:
    print(f"  Graph edges: row={ROW_EDGE_MODE} | col={COL_EDGE_MODE} | ctx={CTX_SCOPE_MODE}")
    print(f"  Overlap min: row={_OVL['row_overlap_min']} col={_OVL['col_overlap_min']} "
          f"ctx={_OVL['context_overlap_min']}  (active where mode=overlap)")
print(f"  GT source  : {GT_CSV}")
print(f"  Schema     : nutrient | quantity | unit | context")
print(f"  Evaluator  : {'LLM (qwen2.5:3b)' if USE_LLM else 'fast rule-based'}")
print(f"  Stage audit: {'ON  -> ' + str(AUDIT_DIR) if AUDIT else 'off'}")
print(f"  Images     : {len(IMAGE_FILES)} from {RAW_DIR}/")
print(f"  Output     : outputs/{EXPERIMENT}/")
if args.notes:
    print(f"  Notes      : {args.notes}")
print(f"{'='*65}\n")


# ── Per-image loop ────────────────────────────────────────────────────────────

all_tuples  = []
diagnostics = {}
ok = err = 0

for img_path in IMAGE_FILES:
    image_key = img_path.stem + img_path.suffix.lower()
    try:
        # 1. OCR
        tokens       = run_ocr_on_image(str(img_path))
        ocr_total    = len(tokens)
        ocr_low_conf = sum(1 for t in tokens if t.get('conf', 1.0) < CONF_THRESH)

        # 2. Corrector V2 (includes C15 canonical normalisation)
        corrected, correction_log = correct_tokens(tokens, return_log=True)

        # 3. GLiNER classifier
        labeled    = classifier.classify_all(corrected)
        clf_counts = Counter(t['label'] for t in labeled)
        clf_gliner = sum(1 for t in labeled
                         if t.get('classification_method') == 'gliner')

        # 3.5 Enricher
        enriched = enricher.enrich(labeled)
        enr_diag = enricher.diagnostics

        # 4 + 5. Graph + Association  (single mode, or merged dual-mode)
        if MERGE_MODES:
            # DEFAULT (cy) first, then STRUCTURAL (role_rank); keep every
            # default tuple, add only role_rank's UNIQUE tuples.
            tuples        = merger.extract(enriched, image_id=image_key)
            graph         = merger.default_graph          # cy graph, for logging/audit
            merge_diag    = dict(merger.diagnostics)
            assoc_matches = merge_diag.get('default_matches', 0)
        else:
            graph         = constructor.build(enriched)
            tuples        = associator.extract(graph, image_id=image_key)
            merge_diag    = {}
            assoc_matches = (associator.diagnostics or {}).get('matches', 0)

        edge_counts  = Counter(e['type'] for e in graph.get('edges', []))
        assoc_tuples = list(tuples)            # snapshot before paragraph fallback

        # 5b. Paragraph-mode fallback (wraps the merged tuples in merge mode)
        used_paragraph = False
        if should_use_paragraph_mode(labeled, len(tuples)):
            para = extract_from_paragraph(labeled, image_key)
            if len(para) > len(tuples):
                print(f"  [paragraph] {image_key}: {len(tuples)} -> {len(para)}")
                tuples = para
                used_paragraph = True

        all_tuples.extend(tuples)

        diagnostics[image_key] = {
            'ocr_total':           ocr_total,
            'ocr_low_conf':        ocr_low_conf,
            'clf_nutrient':        clf_counts.get('NUTRIENT', 0),
            'clf_quantity':        clf_counts.get('QUANTITY', 0),
            'clf_unit':            clf_counts.get('UNIT',     0),
            'clf_context':         clf_counts.get('CONTEXT',  0),
            'clf_noise':           clf_counts.get('NOISE',    0),
            'clf_unknown':         clf_counts.get('UNKNOWN',  0),
            'clf_via_gliner':      clf_gliner,
            'enr_rows':            enr_diag.get('num_rows',         0),
            'enr_columns':         enr_diag.get('num_columns',      0),
            'enr_dosage_streams':  enr_diag.get('dosage_streams',   0),
            'enr_headers':         enr_diag.get('headers_detected', 0),
            'enr_rank_consistent': enr_diag.get('rank_consistent',  False),
            'graph_row_compat':    edge_counts.get('ROW_COMPAT',       0),
            'graph_col_compat':    edge_counts.get('COL_COMPAT',       0),
            'graph_dir_adj':       edge_counts.get('DIRECTIONAL_ADJ',  0),
            'graph_header_scope':  edge_counts.get('HEADER_SCOPE',     0),
            'assoc_tuples':        len(tuples),
            'assoc_matches':       assoc_matches,
            'assoc_no_qty':        sum(1 for t in tuples if not t.get('quantity')),
            'assoc_no_unit':       sum(1 for t in tuples if not t.get('unit')),
            'assoc_no_ctx':        sum(1 for t in tuples if not t.get('context')),
            # late-fusion merge bookkeeping (0 when not merging)
            'merge_default_tuples':  merge_diag.get('default_tuples',      0),
            'merge_rolerank_tuples': merge_diag.get('rolerank_tuples',     0),
            'merge_added_unique':    merge_diag.get('added_from_rolerank', 0),
            'merge_consensus':       merge_diag.get('consensus',           0),
        }

        # 5c. Stage audit dumps (optional)
        if AUDIT:
            write_image_audit(
                AUDIT_DIR, image_key, str(img_path),
                tokens, corrected, correction_log, labeled,
                enriched, enr_diag, graph, assoc_tuples, tuples,
                used_paragraph,
                ("merge:cy+role_rank" if MERGE_MODES else ROW_EDGE_MODE),
                COL_EDGE_MODE,
            )

        extra = (f"  | +{merge_diag.get('added_from_rolerank', 0)} rr-uniq"
                 if MERGE_MODES else "")
        print(f"OK    {image_key}  -> {len(tuples):>3} tuples  "
              f"| NUTR:{clf_counts.get('NUTRIENT',0):>3}  "
              f"GLN:{clf_gliner:>3}  "
              f"ROWS:{enr_diag.get('num_rows',0):>3}  "
              f"STREAMS:{enr_diag.get('dosage_streams',0)}{extra}")
        ok += 1

    except Exception as e:
        import traceback
        print(f"ERR   {image_key}  -> {e}")
        traceback.print_exc()
        err += 1


# ── Save tuples ───────────────────────────────────────────────────────────────

with open(TUPLES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    writer.writerows([{k: t.get(k, '') for k in FIELDNAMES} for t in all_tuples])

print(f"\n{'='*65}")
print(f"  Processed: {ok}  Errors: {err}  Total tuples: {len(all_tuples)}")
if MERGE_MODES:
    tot_added = sum(d.get('merge_added_unique', 0) for d in diagnostics.values())
    print(f"  Merge: role_rank uniquely added {tot_added} tuples on top of the default (cy) set")
if AUDIT:
    print(f"  Stage audits: {AUDIT_DIR}/<image>/_STAGE_AUDIT.md")
print(f"{'='*65}\n")


# ── Evaluation ────────────────────────────────────────────────────────────────

print("Running evaluation...\n")
gt_rows = load_gt_from_csv(GT_CSV)                                      # ← CSV
print(f"GT loaded: {len(gt_rows)} tuples from {GT_CSV}\n")

evaluator = LLMTupleEvaluator(
    gt_rows = gt_rows,
    use_llm = USE_LLM,
    model   = "qwen2.5:3b",
)
metrics = evaluator.run(
    predictions = all_tuples,
    experiment  = EXPERIMENT,
    out_dir     = OUT_DIR,
    notes       = args.notes,
    diagnostics = diagnostics,
)


# ── Run log ───────────────────────────────────────────────────────────────────

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Experiment   : {EXPERIMENT}\n")
    f.write(f"Pipeline     : V2 + Corrector V2 + GLiNERSemanticClassifier ({MODE})\n")
    f.write(f"Classifier   : GLiNER ({MODE})  model={MODEL_ID or 'default bi-large'}\n")
    f.write(f"Threshold    : {THRESHOLD}\n")
    if MERGE_MODES:
        f.write(f"Row edge mode: MERGE (default cy + structural role_rank, add unique)\n")
    else:
        f.write(f"Row edge mode: {ROW_EDGE_MODE}\n")
    f.write(f"Col edge mode: {COL_EDGE_MODE}\n")
    f.write(f"Context scope: {CTX_SCOPE_MODE}\n")
    f.write(f"Overlap min  : row={_OVL['row_overlap_min']} col={_OVL['col_overlap_min']} "
            f"ctx={_OVL['context_overlap_min']}\n")
    f.write(f"Merge modes  : {'ON' if MERGE_MODES else 'off'}\n")
    f.write(f"Stage audit  : {'ON' if AUDIT else 'off'}\n")
    f.write(f"GT source    : {GT_CSV}\n")
    f.write(f"Evaluator    : {'LLM (qwen2.5:3b)' if USE_LLM else 'fast rule-based'}\n")
    f.write(f"Timestamp    : {metrics['timestamp']}\n")
    f.write(f"Notes        : {args.notes}\n\n")
    f.write(f"GT tuples            : {metrics['gt_tuples']}\n")
    f.write(f"Predicted tuples     : {metrics['predicted_tuples']}\n")
    f.write(f"Matched pairs        : {metrics['matched_pairs']}\n")
    f.write(f"Full 4F correct      : {metrics.get('full4f_correct', 0)}\n")
    f.write(f"Full Tuple Precision : {metrics['full_tuple_precision']*100:.1f}%\n")
    f.write(f"Full Tuple Recall    : {metrics['full_tuple_recall']*100:.1f}%\n")
    f.write(f"Full Tuple F1        : {metrics['full_tuple_f1']*100:.1f}%\n")
    f.write(f"Nutrient F1          : {metrics['nutrient_f1']:.3f}\n")
    f.write(f"Unit Acc             : {metrics['unit_acc']*100:.1f}%\n")
    f.write(f"Context Acc          : {metrics['context_acc']*100:.1f}%\n")


# ── Headline (the number you actually care about) ─────────────────────────────

print(f"\n{'='*65}")
print(f"  HEADLINE — true hits (full 4-field matches)")
print(f"{'='*65}")
print(f"  GT tuples           : {metrics['gt_tuples']}")
print(f"  Predicted tuples    : {metrics['predicted_tuples']}")
print(f"  TRUE HITS (4F corr) : {metrics.get('full4f_correct', '?')}")
print(f"  Full Tuple F1       : {metrics['full_tuple_f1']*100:.2f}%")
print(f"  Nutrient F1         : {metrics['nutrient_f1']:.3f}")
print(f"{'='*65}\n")
print(f"Saved: {TUPLES_CSV}")
print(f"Saved: {LOG_FILE}")
print(f"Saved: outputs/{EXPERIMENT}/evaluation_summary.csv")


# ── Optional comparison ───────────────────────────────────────────────────────

if args.compare:
    cmp_path = Path(f'outputs/{args.compare}/evaluation_results.json')
    if cmp_path.exists():
        prev = json.loads(cmp_path.read_text(encoding='utf-8'))
        print(f"\n{'='*65}")
        print(f"  COMPARISON : {args.compare}  ->  {EXPERIMENT}")
        print(f"{'='*65}")
        for label, key in [
            ('TRUE HITS (4F correct)', 'full4f_correct'),
            ('Full Tuple Precision',   'full_tuple_precision'),
            ('Full Tuple Recall',      'full_tuple_recall'),
            ('Full Tuple F1',          'full_tuple_f1'),
            ('Nutrient F1',            'nutrient_f1'),
            ('Unit Match Acc',         'unit_acc'),
            ('Quantity Match Acc',     'quantity_acc'),
            ('Context Match Acc',      'context_acc'),
            ('Matched Pairs',          'matched_pairs'),
            ('Predicted Tuples',       'predicted_tuples'),
        ]:
            old = prev.get(key, 0)
            new = metrics.get(key, 0)
            try:
                d    = float(new) - float(old)
                sign = '+' if d >= 0 else ''
                arr  = '↑' if d > 0 else ('↓' if d < 0 else '=')
                print(f"  {label:<26}  {old:.3f}  ->  {new:.3f}  "
                      f"({sign}{d:.3f}) {arr}")
            except Exception:
                print(f"  {label:<26}  {old}  ->  {new}")
        print(f"{'='*65}\n")
    else:
        print(f"\n  [compare] Not found: {cmp_path}")