"""
run_experiment.py  v2
=====================
Experiment-aware pipeline runner — 4-field schema.

CHANGE IN v2:
  Output schema: image_id, nutrient, quantity, unit, context
  Removed: nrv_percent, serving_size

  Evaluation scores:
    - Nutrient Precision / Recall / F1
    - Quantity Match Accuracy
    - Unit Match Accuracy
    - Context Match Accuracy  (new)
    - Full Tuple Accuracy (nutrient + qty + unit correct)

Usage:
    python run_experiment.py --experiment experiment_02 --notes "description"
    python run_experiment.py --experiment experiment_02 --compare experiment_01_extended
"""

import sys, csv, json, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, '.')

from src.ocr.ocr_runner import run_ocr_on_image
from src.utils.ocr_corrector import OCRCorrector
from src.graph.graph_constructor import GraphConstructor

# ── Experiment 01 Final — load directly from named module files ───────────────
import importlib.util, os

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_clf_path  = os.path.join(os.path.dirname(__file__),
                          'src', 'classification', 'experiment_01_final_semantic_classifier.py')
_assoc_path = os.path.join(os.path.dirname(__file__),
                           'src', 'matching', 'experiment_01_final_association.py')

_clf_module   = _load_module('exp01_classifier',  _clf_path)
_assoc_module = _load_module('exp01_association', _assoc_path)

SemanticClassifier = _clf_module.SemanticClassifier
TupleAssociator    = _assoc_module.TupleAssociator

# ── Config ────────────────────────────────────────────────────────────────────

GT_CSV      = 'data/annotations/gold_annotations.csv'
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

TUPLES_CSV   = OUT_DIR / 'tuples.csv'
METRICS_JSON = OUT_DIR / 'evaluation_results.json'
SUMMARY_CSV  = OUT_DIR / 'evaluation_summary.csv'
PER_IMG_CSV  = OUT_DIR / 'per_image_results.csv'
PAIRS_CSV    = OUT_DIR / 'pair_details.csv'
LOG_FILE     = OUT_DIR / 'run_log.txt'

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
classifier  = SemanticClassifier(confidence_threshold=CONF_THRESH, split_fused_tokens=True)
constructor = GraphConstructor()
associator  = TupleAssociator()

all_tuples = []
ok = skip = err = 0

for img_id in IMAGE_IDS:
    img_path = Path(RAW_DIR) / f'{img_id}.jpeg'
    if not img_path.exists():
        print(f'SKIP  {img_id}.jpeg')
        skip += 1
        continue
    try:
        tokens    = run_ocr_on_image(str(img_path))
        corrected = corrector.correct_all(tokens)
        labeled   = classifier.classify_all(corrected)
        graph     = constructor.build(labeled)
        tuples    = associator.extract(graph, image_id=f'{img_id}.jpeg')
        all_tuples.extend(tuples)

        counts    = Counter(t['label'] for t in labeled)
        print(f'OK    {img_id}.jpeg  → {len(tuples):>3} tuples  '
              f'| NUTR:{counts.get("NUTRIENT",0):>3}  '
              f'QTY:{counts.get("QUANTITY",0):>3}  '
              f'UNIT:{counts.get("UNIT",0):>3}  '
              f'NOISE:{counts.get("NOISE",0)/max(len(labeled),1)*100:>4.0f}%')
        ok += 1
    except Exception as e:
        print(f'ERR   {img_id}.jpeg  → {e}')
        err += 1

# ── Save tuples ───────────────────────────────────────────────────────────────

with open(TUPLES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows([{k: t.get(k, '') for k in FIELDNAMES} for t in all_tuples])

print(f'\n{"="*65}')
print(f'  Processed: {ok}  |  Skipped: {skip}  |  Errors: {err}')
print(f'  Total tuples: {len(all_tuples)}')
print(f'{"="*65}\n')

# ── Normalisation helpers ─────────────────────────────────────────────────────

def norm_qty(v):
    s = str(v or '').strip().replace(',', '.')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f'{f:.10g}'
    except ValueError:
        return s.lower()

def norm_str(v):
    return str(v or '').strip().lower()

# ── Evaluation ────────────────────────────────────────────────────────────────

print("Running evaluation...\n")

gt_rows = []
with open(GT_CSV, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        gt_rows.append({
            'image_id': row.get('image_id','').strip(),
            'nutrient': row.get('nutrient','').strip(),
            'quantity': row.get('quantity','').strip(),
            'unit':     row.get('unit','').strip(),
            'context':  row.get('context','').strip(),
        })

gt_by_img   = {}
pred_by_img = {}
for row in gt_rows:
    gt_by_img.setdefault(row['image_id'], []).append(row)
for t in all_tuples:
    pred_by_img.setdefault(t['image_id'], []).append(t)

all_images = sorted(set(list(gt_by_img.keys()) + list(pred_by_img.keys())))

total_gt = len(gt_rows)
tp = fp = fn = 0
full_count = 0
per_image  = []
pair_rows  = []

for img in all_images:
    gt_list   = gt_by_img.get(img, [])
    pred_list = pred_by_img.get(img, [])
    matched = qty_m = unit_m = ctx_m = full_m = 0
    used    = set()

    for gt in gt_list:
        gt_n = norm_str(gt['nutrient'])
        best = None
        for i, pred in enumerate(pred_list):
            if i in used:
                continue
            pn = norm_str(pred.get('nutrient',''))
            if gt_n == pn or gt_n in pn or pn in gt_n:
                best = (i, pred)
                break

        if best is None:
            fn += 1
            continue

        idx, pred = best
        used.add(idx)
        matched += 1
        tp      += 1

        gq = norm_qty(gt['quantity'])
        gu = norm_str(gt['unit'])
        gc = norm_str(gt['context'])
        pq = norm_qty(pred.get('quantity'))
        pu = norm_str(pred.get('unit'))
        pc = norm_str(pred.get('context'))

        qok  = bool(gq) and gq == pq
        uok  = bool(gu) and gu == pu
        cok  = bool(gc) and gc == pc
        fok  = qok and uok

        if qok: qty_m  += 1
        if uok: unit_m += 1
        if cok: ctx_m  += 1
        if fok:
            full_m     += 1
            full_count += 1

        pair_rows.append({
            'image_id':     img,
            'gt_nutrient':  gt['nutrient'],
            'pred_nutrient':pred.get('nutrient',''),
            'gt_qty':       gt['quantity'],
            'pred_qty':     pred.get('quantity',''),
            'gt_unit':      gt['unit'],
            'pred_unit':    pred.get('unit',''),
            'gt_context':   gt['context'],
            'pred_context': pred.get('context',''),
            'qty_match':    qok,
            'unit_match':   uok,
            'ctx_match':    cok,
            'full_match':   fok,
        })

    fp += len(pred_list) - len(used)
    nm = matched
    per_image.append({
        'image_id': img,
        'gt':       len(gt_list),
        'pred':     len(pred_list),
        'match':    nm,
        'qty_pct':  round(qty_m  / nm * 100) if nm else 0,
        'unit_pct': round(unit_m / nm * 100) if nm else 0,
        'ctx_pct':  round(ctx_m  / nm * 100) if nm else 0,
        'full_pct': round(full_m / nm * 100) if nm else 0,
    })

precision = tp / (tp+fp) if (tp+fp) > 0 else 0
recall    = tp / (tp+fn) if (tp+fn) > 0 else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
np_       = len(pair_rows)
qty_acc   = sum(1 for r in pair_rows if r['qty_match'])  / max(np_, 1)
unit_acc  = sum(1 for r in pair_rows if r['unit_match']) / max(np_, 1)
ctx_acc   = sum(1 for r in pair_rows if r['ctx_match'])  / max(np_, 1)
full_acc  = full_count / total_gt if total_gt > 0 else 0

# ── Print ─────────────────────────────────────────────────────────────────────

print(f"{'='*65}")
print(f"  EXPERIMENT RESULTS : {EXPERIMENT}")
print(f"{'='*65}")
print(f"  Images evaluated  : {len(all_images)}")
print(f"  GT tuples         : {total_gt}")
print(f"  Predicted tuples  : {len(all_tuples)}")
print(f"  Matched pairs     : {np_}")
print(f"{'─'*65}")
print(f"  Nutrient Precision : {precision:.3f}")
print(f"  Nutrient Recall    : {recall:.3f}")
print(f"  Nutrient F1        : {f1:.3f}")
print(f"{'─'*65}")
print(f"  Quantity Match Acc : {qty_acc*100:.1f}%")
print(f"  Unit Match Acc     : {unit_acc*100:.1f}%")
print(f"  Context Match Acc  : {ctx_acc*100:.1f}%")
print(f"{'─'*65}")
print(f"  ★ Full Tuple Acc   : {full_acc*100:.1f}%")
print(f"{'='*65}")
print()
print(f"  {'IMAGE':<17} {'GT':>4}  {'PRED':>5}  {'MATCH':>6}  "
      f"{'QTY%':>5}  {'UNIT%':>6}  {'CTX%':>5}  {'FULL%':>6}")
print(f"  {'─'*65}")
for row in per_image:
    print(f"  {row['image_id']:<17} {row['gt']:>4}  {row['pred']:>5}  "
          f"{row['match']:>6}  {row['qty_pct']:>4}%  "
          f"{row['unit_pct']:>5}%  {row['ctx_pct']:>4}%  "
          f"{row['full_pct']:>5}%")

# ── Save outputs ──────────────────────────────────────────────────────────────

metrics = {
    'experiment':         EXPERIMENT,
    'timestamp':          datetime.now().isoformat(),
    'notes':              args.notes,
    'schema':             'nutrient|quantity|unit|context',
    'images_evaluated':   len(all_images),
    'gt_tuples':          total_gt,
    'predicted_tuples':   len(all_tuples),
    'matched_pairs':      np_,
    'nutrient_precision': round(precision, 4),
    'nutrient_recall':    round(recall,    4),
    'nutrient_f1':        round(f1,        4),
    'quantity_acc':       round(qty_acc,   4),
    'unit_acc':           round(unit_acc,  4),
    'context_acc':        round(ctx_acc,   4),
    'full_tuple_acc':     round(full_acc,  4),
}

with open(METRICS_JSON, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
    w.writeheader(); w.writerow(metrics)

with open(PER_IMG_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=[
        'image_id','gt','pred','match','qty_pct','unit_pct','ctx_pct','full_pct'])
    w.writeheader(); w.writerows(per_image)

with open(PAIRS_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=[
        'image_id','gt_nutrient','pred_nutrient',
        'gt_qty','pred_qty','gt_unit','pred_unit',
        'gt_context','pred_context',
        'qty_match','unit_match','ctx_match','full_match'])
    w.writeheader(); w.writerows(pair_rows)

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Experiment : {EXPERIMENT}\n")
    f.write(f"Schema     : nutrient|quantity|unit|context\n")
    f.write(f"Timestamp  : {datetime.now().isoformat()}\n")
    f.write(f"Notes      : {args.notes}\n\n")
    f.write(f"Full Tuple Acc  : {full_acc*100:.1f}%\n")
    f.write(f"Nutrient F1     : {f1:.3f}\n")
    f.write(f"Unit Acc        : {unit_acc*100:.1f}%\n")
    f.write(f"Context Acc     : {ctx_acc*100:.1f}%\n")

print(f"\n  Saved:")
print(f"    {TUPLES_CSV}")
print(f"    {METRICS_JSON}")
print(f"    {SUMMARY_CSV}")
print(f"    {PER_IMG_CSV}")
print(f"    {PAIRS_CSV}")
print(f"    {LOG_FILE}")

# ── Comparison ────────────────────────────────────────────────────────────────

if args.compare:
    cmp = Path(f'results/{args.compare}/evaluation_results.json')
    if cmp.exists():
        with open(cmp, encoding='utf-8') as f:
            prev = json.load(f)
        print(f"\n{'='*65}")
        print(f"  COMPARISON : {args.compare}  →  {EXPERIMENT}")
        print(f"{'='*65}")
        for label, key, up_good in [
            ('Full Tuple Accuracy',  'full_tuple_acc',  True),
            ('Nutrient F1',          'nutrient_f1',     True),
            ('Unit Match Acc',       'unit_acc',        True),
            ('Quantity Match Acc',   'quantity_acc',    True),
            ('Context Match Acc',    'context_acc',     True),
            ('Matched Pairs',        'matched_pairs',   True),
            ('Predicted Tuples',     'predicted_tuples',False),
        ]:
            old = prev.get(key, 0)
            new = metrics.get(key, 0)
            try:
                d    = float(new) - float(old)
                sign = '+' if d >= 0 else ''
                arr  = '↑' if (d>0)==up_good and d!=0 else ('↓' if d!=0 else '=')
                print(f"  {label:<26}  {old:.3f}  →  {new:.3f}  ({sign}{d:.3f}) {arr}")
            except Exception:
                print(f"  {label:<26}  {old}  →  {new}")
        print(f"{'='*65}\n")
    else:
        print(f"\n  [compare] Not found: {cmp}")