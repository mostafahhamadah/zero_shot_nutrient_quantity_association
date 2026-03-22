"""
run_experiment.py  v4
=====================
Pure orchestrator — pipeline runner only.

CHANGE IN v4:
  Collects per-image pipeline diagnostics during the pipeline loop
  and passes them to TupleEvaluator.run() for deep stage analysis.

  Diagnostics collected per image:
    OCR stage       : total tokens, tokens below conf threshold
    Classifier stage: token counts per label (NUTRIENT/QTY/UNIT/CONTEXT/NOISE/UNKNOWN)
    Graph stage     : edge counts per type (SAME_ROW/SAME_COL/ADJACENT/CONTEXT_SCOPE)
    Association stage: tuples produced, nutrients with no qty/unit/ctx found

  These are saved to pipeline_diagnostics.csv by the evaluator.

CHANGE IN v3:
  All evaluation logic moved to evaluator.py (TupleEvaluator class).
  norm_qty, norm_unit, norm_context, CONTEXT_MAP removed from this file.
"""

import sys, csv, json, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

from src.ocr.ocr_runner          import run_ocr_on_image
from src.utils.ocr_corrector     import OCRCorrector
from src.graph.graph_constructor import GraphConstructor
from src.evaluation.evaluator    import TupleEvaluator

import importlib.util, os

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_base = os.path.dirname(__file__)
_clf_module   = _load_module('exp01_classifier',
    os.path.join(_base, 'src', 'classification',
                 'experiment_01_final_semantic_classifier.py'))
_assoc_module = _load_module('exp01_association',
    os.path.join(_base, 'src', 'matching',
                 'experiment_01_final_association.py'))

SemanticClassifier = _clf_module.SemanticClassifier
TupleAssociator    = _assoc_module.TupleAssociator

# ── Config ────────────────────────────────────────────────────────────────────

GT_CSV      = 'data/annotations/gold_annotations_4field.csv'
RAW_DIR     = 'data/raw'
CONF_THRESH = 0.30
FIELDNAMES  = ['image_id', 'nutrient', 'quantity', 'unit', 'context']

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
parser.add_argument('--images',  default=None)
parser.add_argument('--compare', default=None)
parser.add_argument('--notes',   default='')
args = parser.parse_args()

EXPERIMENT = args.experiment
OUT_DIR    = Path(f'results/{EXPERIMENT}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TUPLES_CSV = OUT_DIR / 'tuples.csv'
LOG_FILE   = OUT_DIR / 'run_log.txt'

# ── Discover images ───────────────────────────────────────────────────────────

if args.images:
    IMAGE_IDS = [i.strip() for i in args.images.split(',')]
else:
    IMAGE_IDS = sorted(
        [f.stem for f in Path(RAW_DIR).glob('*.jpeg')],
        key=lambda x: int(x) if x.isdigit() else float('inf')
    )

print(f"\n{'='*65}")
print(f"  EXPERIMENT : {EXPERIMENT}")
print(f"  Schema     : nutrient | quantity | unit | context")
print(f"  Images     : {len(IMAGE_IDS)} discovered in {RAW_DIR}/")
print(f"  Output     : results/{EXPERIMENT}/")
if args.notes:
    print(f"  Notes      : {args.notes}")
print(f"{'='*65}\n")

# ── Pipeline ──────────────────────────────────────────────────────────────────

corrector   = OCRCorrector()
classifier  = SemanticClassifier(confidence_threshold=CONF_THRESH,
                                  split_fused_tokens=True)
constructor = GraphConstructor()
associator  = TupleAssociator()

all_tuples   = []
diagnostics  = {}   # keyed by image_id, populated per image below
ok = skip = err = 0

for img_id in IMAGE_IDS:
    img_path = Path(RAW_DIR) / f'{img_id}.jpeg'
    if not img_path.exists():
        print(f'SKIP  {img_id}.jpeg')
        skip += 1
        continue
    try:
        image_key = f'{img_id}.jpeg'

        # Stage 1 — OCR
        tokens    = run_ocr_on_image(str(img_path))
        ocr_total    = len(tokens)
        ocr_low_conf = sum(1 for t in tokens if t.get('conf', 1.0) < CONF_THRESH)

        # Stage 2.5 — Corrector
        corrected = corrector.correct_all(tokens)

        # Stage 3 — Classifier
        labeled   = classifier.classify_all(corrected)
        clf_counts = Counter(t['label'] for t in labeled)

        # Stage 4 — Graph
        graph     = constructor.build(labeled)
        edge_counts = Counter(e['type'] for e in graph.get('edges', []))

        # Stage 5 — Association
        # Instrument the associator output to count per-tuple gaps
        tuples    = associator.extract(graph, image_id=image_key)
        assoc_no_qty  = sum(1 for t in tuples if not t.get('quantity'))
        assoc_no_unit = sum(1 for t in tuples if not t.get('unit'))
        assoc_no_ctx  = sum(1 for t in tuples if not t.get('context'))

        all_tuples.extend(tuples)

        # Record diagnostics for this image
        diagnostics[image_key] = {
            'ocr_total':        ocr_total,
            'ocr_low_conf':     ocr_low_conf,
            'clf_nutrient':     clf_counts.get('NUTRIENT', 0),
            'clf_quantity':     clf_counts.get('QUANTITY', 0),
            'clf_unit':         clf_counts.get('UNIT',     0),
            'clf_context':      clf_counts.get('CONTEXT',  0),
            'clf_noise':        clf_counts.get('NOISE',    0),
            'clf_unknown':      clf_counts.get('UNKNOWN',  0),
            'graph_same_row':   edge_counts.get('SAME_ROW',      0),
            'graph_same_col':   edge_counts.get('SAME_COL',      0),
            'graph_adjacent':   edge_counts.get('ADJACENT',      0),
            'graph_ctx_scope':  edge_counts.get('CONTEXT_SCOPE', 0),
            'assoc_tuples':     len(tuples),
            'assoc_no_qty':     assoc_no_qty,
            'assoc_no_unit':    assoc_no_unit,
            'assoc_no_ctx':     assoc_no_ctx,
        }

        print(f'OK    {img_id}.jpeg  → {len(tuples):>3} tuples  '
              f'| NUTR:{clf_counts.get("NUTRIENT",0):>3}  '
              f'QTY:{clf_counts.get("QUANTITY",0):>3}  '
              f'UNIT:{clf_counts.get("UNIT",0):>3}  '
              f'NOISE:{clf_counts.get("NOISE",0)/max(len(labeled),1)*100:>4.0f}%')
        ok += 1

    except Exception as e:
        print(f'ERR   {img_id}.jpeg  → {e}')
        err += 1

# ── Save tuples CSV ───────────────────────────────────────────────────────────

with open(TUPLES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows([{k: t.get(k, '') for k in FIELDNAMES} for t in all_tuples])

print(f'\n{"="*65}')
print(f'  Processed: {ok}  |  Skipped: {skip}  |  Errors: {err}')
print(f'  Total tuples: {len(all_tuples)}')
print(f'{"="*65}\n')

# ── Evaluation ────────────────────────────────────────────────────────────────

print("Running evaluation...\n")

evaluator = TupleEvaluator(gt_csv=GT_CSV)
metrics   = evaluator.run(
    predictions = all_tuples,
    experiment  = EXPERIMENT,
    out_dir     = OUT_DIR,
    notes       = args.notes,
    diagnostics = diagnostics,
)

# ── Save run log ──────────────────────────────────────────────────────────────

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Experiment : {EXPERIMENT}\n")
    f.write(f"Schema     : nutrient|quantity|unit|context\n")
    f.write(f"Timestamp  : {metrics['timestamp']}\n")
    f.write(f"Notes      : {args.notes}\n\n")
    f.write(f"Full Tuple Precision : {metrics['full_tuple_precision']*100:.1f}%\n")
    f.write(f"Full Tuple Recall    : {metrics['full_tuple_recall']*100:.1f}%\n")
    f.write(f"Full Tuple F1        : {metrics['full_tuple_f1']*100:.1f}%\n")
    f.write(f"Nutrient F1          : {metrics['nutrient_f1']:.3f}\n")
    f.write(f"Unit Acc             : {metrics['unit_acc']*100:.1f}%\n")
    f.write(f"Context Acc          : {metrics['context_acc']*100:.1f}%\n")

print(f"    {TUPLES_CSV}")
print(f"    {LOG_FILE}")

# ── Comparison ────────────────────────────────────────────────────────────────

if args.compare:
    cmp_path = Path(f'results/{args.compare}/evaluation_results.json')
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