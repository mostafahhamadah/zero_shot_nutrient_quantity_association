"""Fresh failure analysis for exp23 — find next highest-impact targets."""
import csv
from collections import Counter

EXP = 'graph_v2_exp23'

with open(f'outputs/{EXP}/pair_details.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

total = len(pairs)
f3_ok = sum(1 for r in pairs if r['full_match_3f'] == 'True')
f4_ok = sum(1 for r in pairs if r['full_match_4f'] == 'True')

print(f"Matched: {total}, 3F correct: {f3_ok} ({f3_ok/total*100:.0f}%), 4F correct: {f4_ok} ({f4_ok/total*100:.0f}%)")

# Unit-only failures (qty correct, unit wrong — easy wins)
unit_only = [r for r in pairs if r['qty_match']=='True' and r['unit_match']!='True']
print(f"\n=== UNIT-ONLY FAILURES (qty correct, unit wrong): {len(unit_only)} ===")
print(f"  GT_UNIT → PRED_UNIT patterns:")
unit_patterns = Counter((r['gt_unit'].strip(), r.get('pred_unit','').strip()) for r in unit_only)
for (gt, pred), count in unit_patterns.most_common(15):
    print(f"    {gt:<10} → {pred or '(empty)':<15} x{count}")

# Context-only failures (3F correct but 4F wrong)
ctx_only = [r for r in pairs if r['full_match_3f']=='True' and r['full_match_4f']!='True']
print(f"\n=== CONTEXT-ONLY FAILURES (3F ok, ctx wrong): {len(ctx_only)} ===")
ctx_patterns = Counter((r['gt_context'][:30], r.get('pred_context','')[:30]) for r in ctx_only)
for (gt, pred), count in ctx_patterns.most_common(10):
    print(f"    {gt:<32} → {pred or '(empty)':<20} x{count}")

# Images with most unit-only failures
print(f"\n=== IMAGES WITH MOST UNIT-ONLY FAILS ===")
by_img = Counter(r['image_id'] for r in unit_only)
for img, count in by_img.most_common(10):
    print(f"    {img:<15} {count} unit-only fails")

# Sample unit mismatches
print(f"\n=== SAMPLE UNIT MISMATCHES ===")
print(f"  {'IMAGE':<12} {'NUTRIENT':<22} {'GT_U':<8} {'PRED_U':<10} {'GT_Q':<8}")
for r in unit_only[:20]:
    print(f"  {r['image_id']:<12} {r['gt_nutrient'][:21]:<22} {r['gt_unit']:<8} {r.get('pred_unit',''):<10} {r['gt_qty']:<8}")