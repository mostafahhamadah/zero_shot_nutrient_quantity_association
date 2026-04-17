"""Analyze wrong-qty matches from exp26 pair_details.csv"""
import csv
from collections import Counter

rows = list(csv.DictReader(open('outputs/graph_v2_exp26/pair_details.csv', encoding='utf-8')))
wrong_qty = [r for r in rows if r['qty_match'] == 'False' and r['pred_qty'].strip() != '']
no_pred   = [r for r in rows if r['qty_match'] == 'False' and r['pred_qty'].strip() == '']
correct   = [r for r in rows if r['qty_match'] == 'True']

print(f"Total matched pairs: {len(rows)}")
print(f"Correct qty:         {len(correct)}")
print(f"Wrong qty (has pred): {len(wrong_qty)}")
print(f"Wrong qty (no pred):  {len(no_pred)}")

# Top images by wrong-qty
imgs = Counter(r['image_id'] for r in wrong_qty)
print(f"\nTop 15 images by wrong-qty (has pred):")
for img, cnt in imgs.most_common(15):
    total = sum(1 for r in rows if r['image_id'] == img)
    samples = [r for r in wrong_qty if r['image_id'] == img][:3]
    print(f"  {img:<16} {cnt:>3}/{total:>3}")
    for s in samples:
        print(f"    {s['gt_nutrient'][:30]:<32} gt={s['gt_qty']:<8} pred={s['pred_qty']:<8} unit_gt={s['gt_unit']:<6} ctx_gt={s['gt_context']}")

# Top images by no-pred qty
imgs2 = Counter(r['image_id'] for r in no_pred)
print(f"\nTop 10 images by missing qty (pred empty):")
for img, cnt in imgs2.most_common(10):
    total = sum(1 for r in rows if r['image_id'] == img)
    samples = [r for r in no_pred if r['image_id'] == img][:3]
    print(f"  {img:<16} {cnt:>3}/{total:>3}")
    for s in samples:
        print(f"    {s['gt_nutrient'][:30]:<32} gt_qty={s['gt_qty']}")

# Pattern analysis: what's the delta between gt_qty and pred_qty?
print(f"\nWrong-qty pattern analysis:")
patterns = Counter()
for r in wrong_qty:
    gt = r['gt_qty']
    pred = r['pred_qty']
    try:
        gt_f = float(gt.replace(',', '.'))
        pred_f = float(pred.replace(',', '.'))
        ratio = pred_f / gt_f if gt_f != 0 else 999
        if abs(ratio - 1) < 0.01:
            patterns['~equal (rounding)'] += 1
        elif 0.9 < ratio < 1.1:
            patterns['within 10%'] += 1
        elif pred_f > gt_f:
            patterns[f'pred > gt'] += 1
        else:
            patterns[f'pred < gt'] += 1
    except:
        patterns['non-numeric'] += 1

for p, c in patterns.most_common():
    print(f"  {p:<25} {c:>4}")