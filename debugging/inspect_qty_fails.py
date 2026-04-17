"""Analyze quantity mismatch patterns."""
import csv
from collections import Counter

EXP = 'graph_v2_exp21'

with open(f'outputs/{EXP}/pair_details.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

qty_fails = [r for r in pairs if r['qty_match'] != 'True']

print(f"Total qty failures: {len(qty_fails)}")

# Categorize failures
empty_pred = [r for r in qty_fails if not r['pred_qty'].strip()]
empty_gt = [r for r in qty_fails if not r['gt_qty'].strip()]
both_have = [r for r in qty_fails if r['pred_qty'].strip() and r['gt_qty'].strip()]

print(f"\n  PRED qty empty (no qty found):     {len(empty_pred)}")
print(f"  GT qty empty:                      {len(empty_gt)}")
print(f"  Both have values but mismatch:     {len(both_have)}")

# Show images with most qty failures
print(f"\n  IMAGES WITH MOST QTY FAILURES:")
by_img = Counter(r['image_id'] for r in qty_fails)
for img, count in by_img.most_common(12):
    print(f"    {img:<15} {count} qty failures")

# Show sample mismatches
print(f"\n  SAMPLE QTY MISMATCHES (both have values):")
print(f"  {'IMAGE':<12} {'NUTRIENT':<25} {'GT_QTY':<10} {'PRED_QTY':<10} {'RATIO':<8}")
for r in both_have[:25]:
    try:
        gt_v = float(r['gt_qty'].replace(',','.').lstrip('<>~'))
        pr_v = float(r['pred_qty'].replace(',','.').lstrip('<>~'))
        ratio = pr_v / gt_v if gt_v != 0 else 999
    except:
        ratio = -1
    print(f"  {r['image_id']:<12} {r['gt_nutrient'][:24]:<25} {r['gt_qty']:<10} {r['pred_qty']:<10} {ratio:<8.2f}")