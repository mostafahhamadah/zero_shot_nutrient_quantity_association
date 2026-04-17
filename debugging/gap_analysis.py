"""
Comprehensive gap analysis for exp29.
Identifies the biggest remaining opportunities to improve 4F F1.

Current: 32.9% 4F F1, 547 matched pairs, 291 correct 4F, 866 GT tuples.
Gap: 575 GT tuples NOT correctly extracted as 4F matches.
"""
import csv, json, os
from collections import Counter, defaultdict

PAIR_FILE = "outputs/graph_v2_exp29/pair_details.csv"
PRED_FILE = "outputs/graph_v2_exp29/tuples.csv"
ANN_DIR = "data/annotations"

rows = list(csv.DictReader(open(PAIR_FILE, encoding='utf-8')))
preds = list(csv.DictReader(open(PRED_FILE, encoding='utf-8')))

print(f"={'='*70}")
print(f"GAP ANALYSIS — exp29")
print(f"={'='*70}")
print(f"Matched pairs: {len(rows)}")
print(f"Predicted tuples: {len(preds)}")

# ── 1. Failure breakdown among matched pairs ─────────────────────
matched_4f = [r for r in rows if r['full_match_4f'] == 'True']
matched_3f = [r for r in rows if r['full_match_3f'] == 'True']
wrong_qty = [r for r in rows if r['qty_match'] == 'False']
wrong_unit = [r for r in rows if r['unit_match'] == 'False']
wrong_ctx = [r for r in rows if r['ctx_match'] == 'False']

print(f"\n── MATCHED PAIR FAILURES ({len(rows)} pairs) ──")
print(f"  4F correct:  {len(matched_4f)} ({100*len(matched_4f)//len(rows)}%)")
print(f"  3F correct:  {len(matched_3f)} ({100*len(matched_3f)//len(rows)}%)")
print(f"  Wrong qty:   {len(wrong_qty)} ({100*len(wrong_qty)//len(rows)}%)")
print(f"  Wrong unit:  {len(wrong_unit)} ({100*len(wrong_unit)//len(rows)}%)")
print(f"  Wrong ctx:   {len(wrong_ctx)} ({100*len(wrong_ctx)//len(rows)}%)")

# Failure combos
combos = Counter()
for r in rows:
    if r['full_match_4f'] == 'True':
        continue
    key = []
    if r['qty_match'] == 'False': key.append('qty')
    if r['unit_match'] == 'False': key.append('unit')
    if r['ctx_match'] == 'False': key.append('ctx')
    combos['+'.join(key) if key else 'nutrient_only'] += 1

print(f"\n── FAILURE COMBINATIONS ──")
for combo, cnt in combos.most_common():
    print(f"  {combo:<25} {cnt:>4}")

# ── 2. Wrong qty sub-analysis ─────────────────────────────────────
print(f"\n── WRONG QTY ANALYSIS ({len(wrong_qty)} pairs) ──")
qty_cats = Counter()
for r in wrong_qty:
    gt_q = r['gt_qty'].strip()
    pred_q = r['pred_qty'].strip()
    if not pred_q:
        qty_cats['pred_empty'] += 1
    else:
        try:
            gt_f = float(gt_q.replace(',', '.'))
            pred_f = float(pred_q.replace(',', '.'))
            if gt_f == 0 and pred_f != 0:
                qty_cats['gt=0_pred!=0'] += 1
            elif pred_f == 0 and gt_f != 0:
                qty_cats['pred=0_gt!=0'] += 1
            else:
                # Check if it's a value from the same image (swap)
                img_gt_qtys = set(row['gt_qty'].strip() for row in rows if row['image_id'] == r['image_id'])
                if pred_q in img_gt_qtys:
                    qty_cats['swap_with_other_row'] += 1
                else:
                    qty_cats['unrelated_value'] += 1
        except:
            qty_cats['non_numeric'] += 1

for cat, cnt in qty_cats.most_common():
    print(f"  {cat:<25} {cnt:>4}")

# ── 3. Wrong unit sub-analysis ────────────────────────────────────
print(f"\n── WRONG UNIT ANALYSIS ({len(wrong_unit)} pairs) ──")
unit_cats = Counter()
for r in wrong_unit:
    gt_u = r['gt_unit'].strip().lower()
    pred_u = r['pred_unit'].strip().lower()
    if not pred_u:
        unit_cats[f'{gt_u}→empty'] += 1
    elif gt_u == 'µg' and pred_u == 'g':
        unit_cats['µg→g'] += 1
    elif gt_u == 'g' and pred_u == '':
        unit_cats['g→empty'] += 1
    elif gt_u == 'kj' and pred_u != 'kj':
        unit_cats[f'kJ→{pred_u}'] += 1
    elif gt_u == 'kcal' and pred_u != 'kcal':
        unit_cats[f'kcal→{pred_u}'] += 1
    else:
        unit_cats[f'{gt_u}→{pred_u}'] += 1

for cat, cnt in unit_cats.most_common(15):
    print(f"  {cat:<25} {cnt:>4}")

# ── 4. Unmatched GT tuples (nutrient recall gap) ─────────────────
print(f"\n── UNMATCHED GT TUPLES ──")
total_gt = 0
matched_gt_by_img = defaultdict(set)
for r in rows:
    matched_gt_by_img[r['image_id']].add(r['gt_nutrient'])

unmatched_by_img = Counter()
unmatched_examples = defaultdict(list)

for f in os.listdir(ANN_DIR):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(ANN_DIR, f), encoding='utf-8') as fh:
        gt_data = json.load(fh)
    gt_rows = gt_data if isinstance(gt_data, list) else gt_data.get("nutrients", [])
    img_id = gt_data.get("image_id", f.replace('.json', '.jpeg')) if isinstance(gt_data, dict) else f.replace('.json', '.jpeg')
    img_id = img_id.replace('.PNG', '.jpeg').replace('.png', '.jpeg')
    
    total_gt += len(gt_rows)
    matched_nuts = matched_gt_by_img.get(img_id, set())
    
    for gt in gt_rows:
        gt_nut = gt.get("nutrient", "")
        # Check if this nutrient was matched
        found = False
        for mn in matched_nuts:
            if gt_nut.lower()[:12] in mn.lower() or mn.lower()[:12] in gt_nut.lower():
                found = True
                break
        if not found:
            unmatched_by_img[img_id] += 1
            if len(unmatched_examples[img_id]) < 3:
                unmatched_examples[img_id].append(gt_nut[:40])

print(f"Total GT tuples: {total_gt}")
print(f"Matched: {len(rows)}")
print(f"Unmatched: ~{total_gt - len(rows)}")

print(f"\nTop 15 images by unmatched GT:")
for img, cnt in unmatched_by_img.most_common(15):
    examples = ", ".join(unmatched_examples[img])
    print(f"  {img:<18} {cnt:>3} unmatched  e.g. {examples[:60]}")

# ── 5. Per-image 4F score distribution ────────────────────────────
print(f"\n── 4F SCORE DISTRIBUTION ──")
img_scores = defaultdict(lambda: {"gt": 0, "correct": 0})
for r in rows:
    img_scores[r['image_id']]["gt"] += 1
    if r['full_match_4f'] == 'True':
        img_scores[r['image_id']]["correct"] += 1

# Load actual GT counts
for f in os.listdir(ANN_DIR):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(ANN_DIR, f), encoding='utf-8') as fh:
        gt_data = json.load(fh)
    gt_rows = gt_data if isinstance(gt_data, list) else gt_data.get("nutrients", [])
    img_id = gt_data.get("image_id", f.replace('.json', '.jpeg')) if isinstance(gt_data, dict) else f.replace('.json', '.jpeg')
    img_id = img_id.replace('.PNG', '.jpeg').replace('.png', '.jpeg')
    img_scores[img_id]["total_gt"] = len(gt_rows)

buckets = Counter()
zero_images = []
for img, s in img_scores.items():
    total_gt = s.get("total_gt", s["gt"])
    if total_gt == 0:
        continue
    pct = 100 * s["correct"] // total_gt
    bucket = f"{(pct // 20) * 20}-{(pct // 20) * 20 + 19}%"
    if pct == 100:
        bucket = "100%"
    buckets[bucket] += 1
    if s["correct"] == 0:
        zero_images.append((img, total_gt))

print(f"Images at 0% 4F ({len(zero_images)}):")
for img, gt in sorted(zero_images, key=lambda x: -x[1]):
    print(f"  {img:<18} {gt:>3} GT tuples")

# ── 6. Potential gains summary ────────────────────────────────────
print(f"\n{'='*70}")
print(f"POTENTIAL GAINS SUMMARY")
print(f"{'='*70}")
print(f"Current 4F correct: {len(matched_4f)}/{total_gt} = {100*len(matched_4f)/total_gt:.1f}%")
print(f"\nIf we fix:")
print(f"  All wrong qty ({len(wrong_qty)}):     → +{len(wrong_qty) - len([r for r in wrong_qty if r['unit_match']=='False' or r['ctx_match']=='False'])} potential 4F gains")
print(f"  All wrong unit ({len(wrong_unit)}):    → +{len([r for r in wrong_unit if r['qty_match']=='True' and r['ctx_match']=='True'])} potential 4F gains")
print(f"  All wrong ctx ({len(wrong_ctx)}):      → +{len([r for r in wrong_ctx if r['qty_match']=='True' and r['unit_match']=='True'])} potential 4F gains")
qty_only_fail = len([r for r in rows if r['qty_match']=='False' and r['unit_match']=='True' and r['ctx_match']=='True'])
unit_only_fail = len([r for r in rows if r['unit_match']=='False' and r['qty_match']=='True' and r['ctx_match']=='True'])
ctx_only_fail = len([r for r in rows if r['ctx_match']=='False' and r['qty_match']=='True' and r['unit_match']=='True'])
print(f"\n  Qty-only failures (unit+ctx correct): {qty_only_fail}")
print(f"  Unit-only failures (qty+ctx correct): {unit_only_fail}")
print(f"  Ctx-only failures (qty+unit correct): {ctx_only_fail}")
print(f"\nBiggest single lever: nutrient recall — {total_gt - len(rows)} GT tuples never matched")