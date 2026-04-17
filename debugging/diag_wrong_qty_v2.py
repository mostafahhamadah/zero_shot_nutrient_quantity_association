"""Categorize wrong-qty failures into actionable patterns"""
import csv
from collections import Counter, defaultdict

rows = list(csv.DictReader(open('outputs/graph_v2_exp26/pair_details.csv', encoding='utf-8')))
wrong = [r for r in rows if r['qty_match'] == 'False' and r['pred_qty'].strip() != '']

print(f"=== WRONG-QTY CATEGORIZATION ({len(wrong)} pairs) ===\n")

# Category 1: Column swap — pred_qty matches a DIFFERENT gt_qty for same image
# This means the pipeline grabbed the right number but assigned it to wrong nutrient
cat = Counter()
details = defaultdict(list)

for r in wrong:
    img = r['image_id']
    gt_q = r['gt_qty'].strip()
    pred_q = r['pred_qty'].strip()
    gt_nut = r['gt_nutrient']
    pred_nut = r['pred_nutrient']

    # Get all GT quantities for this image
    img_gt_qtys = set(row['gt_qty'].strip() for row in rows if row['image_id'] == img)
    # Get all pred quantities for this image (from the tuples file if available)
    img_pred_qtys = set(row['pred_qty'].strip() for row in rows if row['image_id'] == img)

    try:
        gt_f = float(gt_q.replace(',', '.'))
        pred_f = float(pred_q.replace(',', '.'))
    except:
        cat['non-numeric'] += 1
        details['non-numeric'].append(r)
        continue

    # Check if pred_qty exists as another GT qty in the same image
    if pred_q in img_gt_qtys and pred_q != gt_q:
        # The predicted qty is correct for a DIFFERENT nutrient — column/row swap
        cat['row/column swap'] += 1
        details['row/column swap'].append(r)
    elif gt_f != 0 and abs(pred_f / gt_f - 1) < 0.15:
        cat['close match (<15%)'] += 1
        details['close match (<15%)'].append(r)
    elif pred_q == '0' or pred_q == '0.0':
        cat['pred=0 (missed real value)'] += 1
        details['pred=0 (missed real value)'].append(r)
    elif gt_q == '0' or gt_q == '0.0':
        cat['gt=0 pred!=0 (false qty)'] += 1
        details['gt=0 pred!=0 (false qty)'].append(r)
    else:
        cat['unrelated value'] += 1
        details['unrelated value'].append(r)

print("Category breakdown:")
for c, n in cat.most_common():
    print(f"  {c:<35} {n:>4} ({100*n//len(wrong)}%)")

# Deep dive into row/column swaps
print(f"\n{'='*80}")
print(f"ROW/COLUMN SWAP detail ({cat.get('row/column swap', 0)} pairs)")
print(f"{'='*80}")
swaps = details.get('row/column swap', [])
swap_imgs = Counter(r['image_id'] for r in swaps)
for img, cnt in swap_imgs.most_common(10):
    print(f"\n  {img} ({cnt} swaps):")
    for r in [s for s in swaps if s['image_id'] == img][:4]:
        print(f"    {r['gt_nutrient'][:35]:<37} gt={r['gt_qty']:<8} pred={r['pred_qty']:<8} ctx={r['gt_context']}")

# Deep dive into unrelated values
print(f"\n{'='*80}")
print(f"UNRELATED VALUE detail ({cat.get('unrelated value', 0)} pairs)")
print(f"{'='*80}")
unrel = details.get('unrelated value', [])
unrel_imgs = Counter(r['image_id'] for r in unrel)
for img, cnt in unrel_imgs.most_common(10):
    print(f"\n  {img} ({cnt} unrelated):")
    for r in [s for s in unrel if s['image_id'] == img][:4]:
        print(f"    {r['gt_nutrient'][:35]:<37} gt={r['gt_qty']:<8} pred={r['pred_qty']:<8} ctx={r['gt_context']}")