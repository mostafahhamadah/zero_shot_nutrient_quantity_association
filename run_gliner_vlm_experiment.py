"""
run_gliner_vlm_experiment.py
=====================
Runs the V2 pipeline end-to-end using the **VLM** (vlm_association.py) as the
tuple-association stage, in place of Graph V2 + Association V2. This is the
association-axis sibling of run_gliner_experiment.py: same OCR, same Corrector
V2, same GLiNER classifier, same Token Enricher, same gold, same paragraph
fallback, same evaluator — only the association engine differs.

  run_gliner_experiment.py :  OCR -> Corr V2 -> GLiNER -> Enrich -> Graph V2 -> Assoc V2 -> eval
  run_vlm_experiment.py    :  OCR -> Corr V2 -> GLiNER -> Enrich -> VLM association          -> eval

The classifier is intentionally still GLiNER (this mirrors the GLiNER runner so
the only changed variable is the associator). To swap the classifier as well,
that is a separate edit.

ASSOCIATOR
----------
  src/matching/vlm_association.py :: VLMAssociator
    .extract(enriched_tokens, image_path, image_id) -> List[{image_id, nutrient,
                                                              quantity, unit, context}]
  The VLM is given the SAME enriched tokens the graph would have received, as a
  structured token table (the ONLY text source), plus the image (used only to
  verify spatial layout). Default backend is the LM Studio OpenAI-compatible
  endpoint with google/gemma-3-4b; all of that is configurable from the CLI and
  otherwise falls back to vlm_association.DEFAULT_CONFIG.

NOTE on --no-llm
----------------
  --no-llm governs ONLY the tuple EVALUATOR (qwen2.5:3b judge). The VLM
  associator ALWAYS calls its backend (LM Studio / Gemma) — that is the
  association engine, not the evaluator.

CHANGES vs run_gliner_experiment.py
-----------------------------------
  - Stage 4/5 (Graph V2 + Association V2) replaced by a single VLM call.
  - Removed graph-only switches: --row-edge-mode, --col-edge-mode, --merge-modes.
  - Removed graph/merge diagnostics; added VLM diagnostics
    (backend, model, parse_mode, salvaged, attempts, image_max_side, elapsed_s).
  - Added VLM knobs: --backend, --vlm-model, --temperature, --max-tokens,
    --image-max-side, --include-structure, --retry-max, --no-postprocess,
    --save-prompts. Each is a pass-through into VLMAssociator's config; unset
    flags keep vlm_association.DEFAULT_CONFIG.
  - --audit now dumps the exact VLM token table, user prompt, and raw model
    response for each image (instead of a graph).

Everything else (normalised CSV GT loader, paragraph fallback, evaluator,
headline, --compare) is identical to run_gliner_experiment.py.

PIPELINE
--------
  Stage 1   — PaddleOCR
  Stage 2   — PaddleOCR Corrector V2  (+ C15 canonical name normalisation)
  Stage 3   — GLiNERSemanticClassifier  (mode=gliner_only | hybrid)
  Stage 3.5 — Token Enricher
  Stage 4   — VLM Association  (vlm_association.py; replaces Graph V2 + Assoc V2)
  Stage 5   — Association V2 paragraph fallback (operates on classified tokens)
  Stage 6   — LLM-Assisted Tuple Evaluator

USAGE
-----
    # Make sure LM Studio is running with the VLM loaded at
    # http://127.0.0.1:1234 (or pass --backend / --vlm-model accordingly).

    python run_vlm_experiment.py --experiment exp42_vlm
    python run_vlm_experiment.py --experiment exp42_quick --no-llm

    # scope to a few images (VLM is a network call per image — start small):
    python run_vlm_experiment.py --experiment exp42_smoke --images 1,108,118 --no-llm

    # override backend / model / generation:
    python run_vlm_experiment.py --experiment exp42_hf \
        --backend hf --vlm-model zai-org/GLM-4.5V:fastest
    python run_vlm_experiment.py --experiment exp42_struct \
        --include-structure --image-max-side 768

    # feed the VLM the hybrid GLiNER node instead of gliner_only:
    python run_vlm_experiment.py --experiment exp42_hybrid --mode hybrid

    # with per-stage audit dumps (incl. exact prompt + raw VLM response):
    python run_vlm_experiment.py --experiment exp42_audit \
        --audit --images 1,108,118 --no-llm

    # diff against a prior experiment:
    python run_vlm_experiment.py --experiment exp42_vlm --compare exp41_gliner_tuned
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
from src.matching                      import vlm_association as vlm_mod  # ← VLM associator
from src.evaluation.llm_evaluator      import LLMTupleEvaluator

VLMAssociator = vlm_mod.VLMAssociator


# ── Dynamic loader (mirrors run_gliner_experiment.py pattern) ─────────────────

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


# ── Normalised CSV GT loader (identical to run_gliner_experiment.py) ──────────

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


def _dump_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text is not None else "")


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
                      enriched, enr_diag,
                      vlm_token_table, vlm_user_prompt, vlm_raw_response, vlm_diag,
                      assoc_tuples, tuples, used_paragraph):
    """
    Write per-stage OUTPUT dumps + a human-readable _STAGE_AUDIT.md
    documenting each stage's input and output for one image.

    VLM variant: Stage 4 is the VLM association — its INPUT (token table that
    is sent as the only text source, plus the full user prompt) and its raw
    model OUTPUT are dumped verbatim, instead of a graph.

        image -> 01 ocr -> 02 corrected -> 03 classified
              -> 04 enriched -> 05 vlm (table/prompt/raw) -> 06 tuples
    """
    d = Path(audit_root) / image_key
    d.mkdir(parents=True, exist_ok=True)

    # ── per-stage output dumps ────────────────────────────────────────
    _dump_json(d / "01_ocr_tokens.json",        tokens)
    _dump_json(d / "02_corrected_tokens.json",  corrected)
    _dump_json(d / "02_correction_log.json",    correction_log or [])
    _dump_json(d / "03_classified_tokens.json", labeled)
    _dump_json(d / "04_enriched_tokens.json",   enriched)
    _dump_text(d / "05_vlm_token_table.txt",    vlm_token_table)        # VLM input (text source)
    _dump_text(d / "05_vlm_user_prompt.txt",    vlm_user_prompt)        # full user prompt
    _dump_text(d / "05_vlm_raw_response.txt",   vlm_raw_response)       # raw model output
    _dump_json(d / "06_tuples.json",            tuples)                 # eval input
    if used_paragraph:
        _dump_json(d / "06b_association_tuples.json", assoc_tuples)

    # ── derived summaries ─────────────────────────────────────────────
    label_dist = Counter(t.get("label", "?") for t in labeled)
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
    md.append("\nAssociation engine: **VLM** "
              f"(`{vlm_diag.get('backend','?')}` / `{vlm_diag.get('model','?')}`)")
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

    md.append("## Stage 4 — VLM Association  ·  `vlm_association.py`  (replaces Graph V2 + Association V2)")
    md.append(f"- **Input:**  {len(enr_active)} enriched tokens → token table (the ONLY text "
              "source) + the label image (used only for spatial verification)")
    md.append(f"- **Output:** {n_assoc} tuples → "
              + ("`06b_association_tuples.json`" if used_paragraph else "`06_tuples.json`"))
    md.append("- **Tuple schema:** `image_id, nutrient, quantity, unit, context`")
    md.append(f"- **Call:** backend=`{vlm_diag.get('backend','?')}`, "
              f"model=`{vlm_diag.get('model','?')}`, "
              f"img_max=`{vlm_diag.get('image_max_side','?')}px`, "
              f"attempts=`{vlm_diag.get('attempts','?')}`, "
              f"parse_mode=`{vlm_diag.get('parse_mode','?')}`, "
              f"salvaged=`{vlm_diag.get('salvaged_objects','?')}`, "
              f"elapsed=`{vlm_diag.get('elapsed_s','?')}s`, "
              f"last_error=`{vlm_diag.get('last_error', None)}`")
    md.append("- **Verbatim dumps:** token table → `05_vlm_token_table.txt`; "
              "full user prompt → `05_vlm_user_prompt.txt`; "
              "raw model response → `05_vlm_raw_response.txt`")
    md.append(f"- **Empty fields (final):** no_quantity={no_qty}, no_unit={no_unit}, "
              f"no_context={no_ctx}")
    md.append(f"- **Sample tuples:**\n```json\n{_sample(assoc_tuples, n=3)}\n```\n")

    if used_paragraph:
        md.append("## Stage 5 — Paragraph Fallback  ·  `paragraph_extractor.py`  (TRIGGERED)")
        md.append(f"- Replaced VLM output ({n_assoc} tuples) with "
                  f"paragraph extraction ({len(tuples)} tuples → `06_tuples.json`).")
        md.append("")

    md.append("## Stage 6 — Evaluation  ·  `llm_evaluator.py`")
    md.append("- **Input:**  final tuples (`06_tuples.json`) + ground truth")
    md.append("- **Output:** per-experiment metrics (4F-F1, 3F-F1, per-field) — "
              "written at run level, not per image")
    md.append("")

    (d / "_STAGE_AUDIT.md").write_text("\n".join(md), encoding="utf-8")


# ── Config ────────────────────────────────────────────────────────────────────

GT_CSV      = 'test_set_normalized.csv'                                 # ← CSV
RAW_DIR     = 'data/raw'
CONF_THRESH = 0.0
FIELDNAMES  = ['image_id', 'nutrient', 'quantity', 'unit', 'context']

# GLiNER node: calibrated via the token-level threshold sweep
#   (gliner_threshold_sweep.py) -> F1-optimal tau* = 0.18. Classifier is kept
# identical to run_gliner_experiment.py so the only changed variable is the
# associator (VLM instead of graph).
GLINER_THRESHOLD = 0.18
DEFAULT_MODE     = 'gliner_only'


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
parser.add_argument('--images',    default=None,
                    help="Comma-separated image stems (e.g. '1,108,118')")
parser.add_argument('--compare',   default=None,
                    help="Prior experiment folder name to diff against")
parser.add_argument('--notes',     default='')
parser.add_argument('--no-llm',    action='store_true',
                    help='Skip LLM EVALUATION only — the VLM associator still runs')
parser.add_argument('--gt-csv',    default=GT_CSV,
                    help=f"Path to normalised GT CSV (default {GT_CSV})")
parser.add_argument('--audit', action='store_true',
                    help='Write per-stage I/O dumps + _STAGE_AUDIT.md for each image '
                         '(includes the exact VLM token table, prompt, and raw response)')

# ── GLiNER classifier (unchanged from run_gliner_experiment.py) ──────────────
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

# ── VLM association (pass-throughs into VLMAssociator config) ─────────────────
# Unset flags keep vlm_association.DEFAULT_CONFIG.
parser.add_argument('--backend',   default=None,
                    choices=['openai_compat', 'lmstudio', 'hf', 'ollama'],
                    help="VLM backend (default: vlm_association DEFAULT_CONFIG = openai_compat / LM Studio)")
parser.add_argument('--vlm-model', default=None,
                    help="Override the VLM model id for the chosen backend "
                         "(e.g. 'google/gemma-3-4b', 'zai-org/GLM-4.5V:fastest')")
parser.add_argument('--temperature', type=float, default=None,
                    help="VLM sampling temperature (default config 0.1)")
parser.add_argument('--max-tokens',  type=int, default=None,
                    help="VLM max output tokens (default config 4096)")
parser.add_argument('--image-max-side', type=int, default=None,
                    help="Resize images so the longest side <= this many px (default config 896)")
parser.add_argument('--include-structure', action='store_true',
                    help="Add structural metadata (row/col/role/context/stream) to the VLM token table")
parser.add_argument('--retry-max', type=int, default=None,
                    help="VLM retries-on-fail with progressive image shrink (default config 0)")
parser.add_argument('--no-postprocess', action='store_true',
                    help="Disable the conservative VLM output post-processing "
                         "(quantity/unit/nutrient cleanup, context canonicalisation, dedupe)")
parser.add_argument('--save-prompts', action='store_true',
                    help="Have the VLM associator dump each prompt to outputs/debug_vlm_prompts/")

args = parser.parse_args()

EXPERIMENT      = args.experiment
USE_LLM         = not args.no_llm
MODE            = args.mode
THRESHOLD       = args.threshold
MODEL_ID        = args.model_id
DEVICE          = args.device
GT_CSV          = args.gt_csv
AUDIT           = args.audit

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

enricher = TokenEnricher()

# VLM associator — build config from CLI, otherwise keep vlm_association DEFAULT_CONFIG.
_vlm_cfg = {}
if args.backend:        _vlm_cfg["backend"]            = args.backend
if args.temperature is not None:    _vlm_cfg["temperature"]    = args.temperature
if args.max_tokens is not None:     _vlm_cfg["max_tokens"]     = args.max_tokens
if args.image_max_side is not None: _vlm_cfg["image_max_side"] = args.image_max_side
if args.include_structure:          _vlm_cfg["include_structure"] = True
if args.retry_max is not None:      _vlm_cfg["retry_max"]      = args.retry_max
if args.no_postprocess:             _vlm_cfg["postprocess_output"] = False
if args.save_prompts:               _vlm_cfg["save_prompts"]   = True
if args.vlm_model:
    _b = _vlm_cfg.get("backend", vlm_mod.DEFAULT_CONFIG["backend"])
    if _b in ("openai_compat", "lmstudio"):
        _vlm_cfg["openai_model"] = args.vlm_model
    elif _b == "hf":
        _vlm_cfg["hf_model"] = args.vlm_model
    else:  # ollama
        _vlm_cfg["model"] = args.vlm_model

vlm_associator     = VLMAssociator(_vlm_cfg)
VLM_BACKEND_ACTIVE = vlm_associator.config["backend"]
VLM_MODEL_ACTIVE   = vlm_associator._active_model_name()


# ── Banner ────────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  EXPERIMENT : {EXPERIMENT}")
print(f"  Pipeline   : V2 (PaddleOCR + Corrector V2 + GLiNER Clf + VLM Assoc)")
print(f"  Corrector  : paddleOCR_corrector_v2 (C15 canonical normalisation)")
print(f"  Classifier : GLiNER {MODE}  (gliner-biomed, t={THRESHOLD})")
print(f"  Associator : VLM  ({VLM_BACKEND_ACTIVE} / {VLM_MODEL_ACTIVE})")
print(f"               T={vlm_associator.config['temperature']}  "
      f"max_tokens={vlm_associator.config['max_tokens']}  "
      f"img_max={vlm_associator.config['image_max_side']}px  "
      f"structure={vlm_associator.config['include_structure']}  "
      f"postprocess={vlm_associator.config['postprocess_output']}  "
      f"retry_max={vlm_associator.config['retry_max']}")
print(f"  GT source  : {GT_CSV}")
print(f"  Schema     : nutrient | quantity | unit | context")
print(f"  Evaluator  : {'LLM (qwen2.5:3b)' if USE_LLM else 'fast rule-based'}  "
      f"(VLM association runs regardless of --no-llm)")
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

        # 3.5 Enricher  (VLM receives the SAME enriched tokens the graph would have)
        enriched = enricher.enrich(labeled)
        enr_diag = enricher.diagnostics

        # 4. VLM association  (replaces Graph V2 + Association V2)
        tuples   = vlm_associator.extract(
            enriched,
            image_path = str(img_path),
            image_id   = image_key,
        )
        vlm_diag     = dict(vlm_associator.diagnostics)
        vlm_raw      = vlm_associator._last_raw_response
        assoc_tuples = list(tuples)            # snapshot before paragraph fallback

        # 5. Paragraph-mode fallback (operates on classified tokens, as in the graph runner)
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
            # VLM association bookkeeping (replaces graph_* and merge_*)
            'vlm_backend':         vlm_diag.get('backend',          ''),
            'vlm_model':           vlm_diag.get('model',            ''),
            'vlm_active_tokens':   vlm_diag.get('active_tokens',    0),
            'vlm_nutrients':       vlm_diag.get('nutrients',        0),
            'vlm_quantities':      vlm_diag.get('quantities',       0),
            'vlm_attempts':        vlm_diag.get('attempts',         0),
            'vlm_parse_mode':      vlm_diag.get('parse_mode',       ''),
            'vlm_salvaged':        vlm_diag.get('salvaged_objects', 0),
            'vlm_image_max_side':  vlm_diag.get('image_max_side',   0),
            'vlm_elapsed_s':       vlm_diag.get('elapsed_s',        0),
            'vlm_last_error':      vlm_diag.get('last_error',       None),
            'assoc_tuples':        len(tuples),
            'assoc_no_qty':        sum(1 for t in tuples if not t.get('quantity')),
            'assoc_no_unit':       sum(1 for t in tuples if not t.get('unit')),
            'assoc_no_ctx':        sum(1 for t in tuples if not t.get('context')),
        }

        # 5b. Stage audit dumps (optional) — incl. exact VLM table/prompt/raw response
        if AUDIT:
            try:
                vlm_token_table = vlm_mod._build_token_table(enriched, vlm_associator.config)
                vlm_user_prompt = vlm_mod._build_user_prompt(vlm_token_table, image_key)
            except Exception:
                vlm_token_table = "(token table unavailable)"
                vlm_user_prompt = "(prompt unavailable)"
            write_image_audit(
                AUDIT_DIR, image_key, str(img_path),
                tokens, corrected, correction_log, labeled,
                enriched, enr_diag,
                vlm_token_table, vlm_user_prompt, vlm_raw, vlm_diag,
                assoc_tuples, tuples, used_paragraph,
            )

        flag = "" if not vlm_diag.get('last_error') else f"  !{vlm_diag.get('last_error')}"
        print(f"OK    {image_key}  -> {len(tuples):>3} tuples  "
              f"| NUTR:{clf_counts.get('NUTRIENT',0):>3}  "
              f"GLN:{clf_gliner:>3}  "
              f"ROWS:{enr_diag.get('num_rows',0):>3}  "
              f"VLM:{vlm_diag.get('parse_mode','?'):<8}  "
              f"{vlm_diag.get('elapsed_s',0)}s{flag}")
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
n_vlm_fail = sum(1 for d in diagnostics.values() if d.get('vlm_last_error'))
if n_vlm_fail:
    print(f"  VLM: {n_vlm_fail} image(s) had a backend/parse error "
          f"(see vlm_last_error per image; check LM Studio is running)")
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
    f.write(f"Pipeline     : V2 + Corrector V2 + GLiNER ({MODE}) + VLM association\n")
    f.write(f"Classifier   : GLiNER ({MODE})  model={MODEL_ID or 'default bi-large'}  t={THRESHOLD}\n")
    f.write(f"Associator   : VLM  backend={VLM_BACKEND_ACTIVE}  model={VLM_MODEL_ACTIVE}\n")
    f.write(f"VLM params   : T={vlm_associator.config['temperature']}  "
            f"max_tokens={vlm_associator.config['max_tokens']}  "
            f"img_max={vlm_associator.config['image_max_side']}px  "
            f"include_structure={vlm_associator.config['include_structure']}  "
            f"postprocess={vlm_associator.config['postprocess_output']}  "
            f"retry_max={vlm_associator.config['retry_max']}\n")
    f.write(f"Stage audit  : {'ON' if AUDIT else 'off'}\n")
    f.write(f"GT source    : {GT_CSV}\n")
    f.write(f"Evaluator    : {'LLM (qwen2.5:3b)' if USE_LLM else 'fast rule-based'} "
            f"(VLM association always runs)\n")
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