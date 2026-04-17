"""
run_graph_v2_experiment.py
==========================
Geometry-Aware Pipeline V2 — Experiment Runner
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PIPELINE
--------
  Stage 1   — OCR              (EasyOCR)
  Stage 2   — OCR Corrector
  Stage 3   — Semantic Classifier
  Stage 3.5 — Token Enricher              ← NEW
  Stage 4   — Graph Construction V2        ← REDESIGNED
  Stage 5   — Association V2               ← REDESIGNED
  Stage 6   — LLM-Assisted Evaluation

Usage:
    python run_graph_v2_experiment.py --experiment graph_v2_exp01
    python run_graph_v2_experiment.py --experiment graph_v2_exp01 --images 1,101,102
    python run_graph_v2_experiment.py --experiment graph_v2_exp01 --compare experiment_08_v3_evaluator
    python run_graph_v2_experiment.py --experiment graph_v2_exp01 --no-llm
"""

import sys, csv, json, argparse
from pathlib import Path
from collections import Counter
from src.utils.paragraph_extractor import extract_from_paragraph, should_use_paragraph_mode
sys.path.insert(0, '.')
from src.utils.sentence_extractor import extract_from_sentences

# ── Stage imports ─────────────────────────────────────────────────────────────

from src.ocr.paddleocr_runner               import run_ocr_on_image
from src.utils.paddleocr_corrector          import correct_tokens
from src.utils.token_enricher              import TokenEnricher
from src.graph.graph_constructor_v2        import GraphConstructorV2
from src.matching.association_v2           import TupleAssociatorV2
from src.evaluation.llm_evaluator          import LLMTupleEvaluator

import importlib.util, os

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_base = os.path.dirname(__file__)
_clf_module = _load_module('exp01_classifier',
    os.path.join(_base, 'src', 'classification',
                 'experiment_01_final_semantic_classifier.py'))

SemanticClassifier = _clf_module.SemanticClassifier


# ── JSON annotation loader ────────────────────────────────────────────────────

def load_gt_from_json(annotations_dir: str) -> list:
    """Load all per-image JSON annotation files."""
    rows = []
    ann_path = Path(annotations_dir)
    for jf in sorted(ann_path.glob("*.json")):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            raw_id   = Path(data.get("image_id", jf.stem + ".jpeg")).name
            image_id = Path(raw_id).stem + Path(raw_id).suffix.lower()
            for n in data.get("nutrients", []):
                context      = str(n.get("context",      "")).strip()
                serving_size = str(n.get("serving_size", "") or "").strip()
                context_full = f"{context} ({serving_size})" if serving_size else context
                rows.append({
                    "image_id": image_id,
                    "nutrient": str(n.get("nutrient", "")).strip(),
                    "quantity": str(n.get("quantity", "")).strip(),
                    "unit":     str(n.get("unit",     "")).strip(),
                    "context":  context_full,
                })
        except Exception as e:
            print(f"[load_gt] Failed to load {jf.name}: {e}")
    return rows


# ── Config ────────────────────────────────────────────────────────────────────

GT_ANNOTATIONS = 'data/annotations'
RAW_DIR        = 'data/raw'
CONF_THRESH    = 0.30
FIELDNAMES     = ['image_id', 'nutrient', 'quantity', 'unit', 'context']

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
parser.add_argument('--images',  default=None)
parser.add_argument('--compare', default=None)
parser.add_argument('--notes',   default='')
parser.add_argument('--no-llm',  action='store_true',
                    help='Skip LLM evaluation — use fast rule-based pass only')
parser.add_argument('--classifier', default='rule',
                    choices=['rule', 'embedding_only', 'hybrid'],
                    help='Classifier mode: rule (lexicon), embedding_only, or hybrid')
args = parser.parse_args()

EXPERIMENT = args.experiment
USE_LLM    = not args.no_llm
CLF_MODE   = args.classifier
OUT_DIR    = Path(f'outputs/{EXPERIMENT}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TUPLES_CSV = OUT_DIR / 'tuples.csv'
LOG_FILE   = OUT_DIR / 'run_log.txt'

# ── Discover images ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

if args.images:
    # Find actual files matching the given stems
    IMAGE_FILES = []
    for stem in (i.strip() for i in args.images.split(',')):
        found = [f for f in Path(RAW_DIR).iterdir()
                 if f.stem == stem and f.suffix.lower() in IMAGE_EXTENSIONS]
        if found:
            IMAGE_FILES.append(found[0])
        else:
            print(f'WARN  No image found for stem: {stem}')
    IMAGE_FILES.sort(key=lambda f: (int(f.stem) if f.stem.isdigit() else float('inf'), f.name))
else:
    IMAGE_FILES = sorted(
        [f for f in Path(RAW_DIR).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float('inf'), f.name)
    )

print(f"\n{'='*65}")
print(f"  EXPERIMENT : {EXPERIMENT}")
print(f"  Pipeline   : Geometry-Aware V2 (PaddleOCR + Token Enricher + Graph V2)")
print(f"  Classifier : {CLF_MODE}")
print(f"  Schema     : nutrient | quantity | unit | context")
print(f"  Evaluator  : {'LLM-Assisted (Qwen 2.5:7b)' if USE_LLM else 'Fast rule-based only'}")
print(f"  Images     : {len(IMAGE_FILES)} discovered in {RAW_DIR}/")
print(f"  Output     : outputs/{EXPERIMENT}/")
if args.notes:
    print(f"  Notes      : {args.notes}")
print(f"{'='*65}\n")

# ── Pipeline instances ────────────────────────────────────────────────────────

if CLF_MODE == 'rule':
    classifier = SemanticClassifier(confidence_threshold=CONF_THRESH)
else:
    # Load embedding classifier
    _emb_module = _load_module('emb_classifier',
        os.path.join(_base, 'src', 'classification',
                     'embedding_semantic_classifier.py'))
    EmbeddingSemanticClassifier = _emb_module.EmbeddingSemanticClassifier
    classifier = EmbeddingSemanticClassifier(
        mode=CLF_MODE,
        confidence_threshold=CONF_THRESH,
    )
enricher    = TokenEnricher()         # ← NEW Stage 3.5
constructor = GraphConstructorV2()    # ← REDESIGNED Stage 4
associator  = TupleAssociatorV2()     # ← REDESIGNED Stage 5

all_tuples   = []
diagnostics  = {}
ok = skip = err = 0

for img_path in IMAGE_FILES:
    image_key = img_path.stem + img_path.suffix.lower()
    try:
        # Stage 1 — OCR
        tokens       = run_ocr_on_image(str(img_path))
        ocr_total    = len(tokens)
        ocr_low_conf = sum(1 for t in tokens if t.get('conf', 1.0) < CONF_THRESH)

        # Stage 2 — PaddleOCR Corrector
        corrected, _ = correct_tokens(tokens, return_log=True)

        # Stage 3 — Classifier
        labeled    = classifier.classify_all(corrected)
        clf_counts = Counter(t['label'] for t in labeled)

        # Stage 3.5 — Token Enricher (NEW)
        enriched = enricher.enrich(labeled)
        enr_diag = enricher.diagnostics

        # Stage 4 — Graph V2 (REDESIGNED)
        graph       = constructor.build(enriched)
        edge_counts = Counter(e['type'] for e in graph.get('edges', []))

        # Stage 5 — Association V2 (REDESIGNED)
        tuples     = associator.extract(graph, image_id=image_key)
        assoc_diag = associator.diagnostics
        # Stage 5b — Paragraph mode fallback
        if should_use_paragraph_mode(labeled, len(tuples)):
            para_tuples = extract_from_paragraph(labeled, image_key)
            if len(para_tuples) > len(tuples):
                print(f"  [Paragraph mode] {image_key}: "
                      f"replacing {len(tuples)} -> {len(para_tuples)} tuples")
                tuples = para_tuples
        assoc_no_qty  = sum(1 for t in tuples if not t.get('quantity'))
        assoc_no_unit = sum(1 for t in tuples if not t.get('unit'))
        assoc_no_ctx  = sum(1 for t in tuples if not t.get('context'))

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
            # V2 enricher diagnostics
            'enr_rows':            enr_diag.get('num_rows',        0),
            'enr_columns':         enr_diag.get('num_columns',     0),
            'enr_dosage_streams':  enr_diag.get('dosage_streams',  0),
            'enr_headers':         enr_diag.get('headers_detected', 0),
            'enr_rank_consistent': enr_diag.get('rank_consistent', False),
            # V2 graph diagnostics
            'graph_row_compat':    edge_counts.get('ROW_COMPAT',       0),
            'graph_col_compat':    edge_counts.get('COL_COMPAT',       0),
            'graph_dir_adj':       edge_counts.get('DIRECTIONAL_ADJ',  0),
            'graph_header_scope':  edge_counts.get('HEADER_SCOPE',     0),
            # V2 association diagnostics
            'assoc_tuples':        len(tuples),
            'assoc_matches':       assoc_diag.get('matches', 0),
            'assoc_no_qty':        assoc_no_qty,
            'assoc_no_unit':       assoc_no_unit,
            'assoc_no_ctx':        assoc_no_ctx,
        }

        print(f'OK    {image_key}  → {len(tuples):>3} tuples  '
              f'| NUTR:{clf_counts.get("NUTRIENT",0):>3}  '
              f'QTY:{clf_counts.get("QUANTITY",0):>3}  '
              f'ROWS:{enr_diag.get("num_rows",0):>3}  '
              f'COLS:{enr_diag.get("num_columns",0):>3}  '
              f'STREAMS:{enr_diag.get("dosage_streams",0)}')
        ok += 1

    except Exception as e:
        import traceback
        print(f'ERR   {image_key}  → {e}')
        traceback.print_exc()
        err += 1

# ── Save tuples CSV ───────────────────────────────────────────────────────────

with open(TUPLES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    writer.writerows([{k: t.get(k, '') for k in FIELDNAMES} for t in all_tuples])

print(f'\n{"="*65}')
print(f'  Processed: {ok}  |  Skipped: {skip}  |  Errors: {err}')
print(f'  Total tuples: {len(all_tuples)}')
print(f'{"="*65}\n')

# ── Evaluation ────────────────────────────────────────────────────────────────

print("Running evaluation...\n")

gt_rows = load_gt_from_json(GT_ANNOTATIONS)
print(f"GT loaded: {len(gt_rows)} tuples from {GT_ANNOTATIONS}/*.json\n")

evaluator = LLMTupleEvaluator(
    gt_rows = gt_rows,
    use_llm = USE_LLM,
    model   = "qwen2.5:7b",
)

metrics = evaluator.run(
    predictions = all_tuples,
    experiment  = EXPERIMENT,
    out_dir     = OUT_DIR,
    notes       = args.notes,
    diagnostics = diagnostics,
)

# ── Save run log ──────────────────────────────────────────────────────────────

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Experiment  : {EXPERIMENT}\n")
    f.write(f"Pipeline    : Geometry-Aware V2 (PaddleOCR + Token Enricher + Graph V2)\n")
    f.write(f"Classifier  : {CLF_MODE}\n")
    f.write(f"Evaluator   : {'LLM-Assisted (qwen2.5:7b)' if USE_LLM else 'Fast rule-based'}\n")
    f.write(f"Schema      : nutrient|quantity|unit|context\n")
    f.write(f"Timestamp   : {metrics['timestamp']}\n")
    f.write(f"Notes       : {args.notes}\n\n")
    f.write(f"Full Tuple Precision : {metrics['full_tuple_precision']*100:.1f}%\n")
    f.write(f"Full Tuple Recall    : {metrics['full_tuple_recall']*100:.1f}%\n")
    f.write(f"Full Tuple F1        : {metrics['full_tuple_f1']*100:.1f}%\n")
    f.write(f"Nutrient F1          : {metrics['nutrient_f1']:.3f}\n")
    f.write(f"Unit Acc             : {metrics['unit_acc']*100:.1f}%\n")
    f.write(f"Context Acc          : {metrics['context_acc']*100:.1f}%\n")
    if USE_LLM:
        f.write(f"\nLLM calls    : {metrics.get('llm_calls', 0)}\n")
        f.write(f"Fast hits    : {metrics.get('fast_hits', 0)}\n")
        f.write(f"Eval time    : {metrics.get('eval_time_s', 0)}s\n")

print(f"    {TUPLES_CSV}")
print(f"    {LOG_FILE}")

# ── Comparison ────────────────────────────────────────────────────────────────

if args.compare:
    cmp_path = Path(f'outputs/{args.compare}/evaluation_results.json')
    if cmp_path.exists():
        with open(cmp_path, encoding='utf-8') as f:
            prev = json.load(f)
        print(f"\n{'='*65}")
        print(f"  COMPARISON : {args.compare}  →  {EXPERIMENT}")
        print(f"{'='*65}")
        for label, key, up_good in [
            ('Full Tuple Precision', 'full_tuple_precision', True),
            ('Full Tuple Recall',    'full_tuple_recall',    True),
            ('Full Tuple F1',        'full_tuple_f1',        True),
            ('Nutrient F1',          'nutrient_f1',          True),
            ('Unit Match Acc',       'unit_acc',             True),
            ('Quantity Match Acc',   'quantity_acc',         True),
            ('Context Match Acc',    'context_acc',          True),
            ('Matched Pairs',        'matched_pairs',        True),
            ('Predicted Tuples',     'predicted_tuples',     False),
        ]:
            old = prev.get(key, 0)
            new = metrics.get(key, 0)
            try:
                d    = float(new) - float(old)
                sign = '+' if d >= 0 else ''
                arr  = ('↑' if (d > 0) == up_good and d != 0
                        else ('↓' if d != 0 else '='))
                print(f"  {label:<26}  {old:.3f}  →  {new:.3f}"
                      f"  ({sign}{d:.3f}) {arr}")
            except Exception:
                print(f"  {label:<26}  {old}  →  {new}")
        print(f"{'='*65}\n")
    else:
        print(f"\n  [compare] Not found: {cmp_path}")