"""Systematic analysis of remaining failures to prioritize next fixes."""
import csv, json
from pathlib import Path
from collections import Counter

EXP = 'graph_v2_exp21'

with open(f'outputs/{EXP}/pair_details.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

with open(f'outputs/{EXP}/per_image_results.csv', encoding='utf-8') as f:
    per_img = list(csv.DictReader(f))

# 1. Overall failure breakdown
total = len(pairs)
qty_ok = sum(1 for r in pairs if r['qty_match'] == 'True')
unit_ok = sum(1 for r in pairs if r['unit_match'] == 'True')
ctx_ok = sum(1 for r in pairs if r['ctx_match'] == 'True')
f3_ok = sum(1 for r in pairs if r['full_match_3f'] == 'True')
f4_ok = sum(1 for r in pairs if r['full_match_4f'] == 'True')

print(f"{'='*70}")
print(f"  SYSTEMATIC FAILURE ANALYSIS — {EXP}")
print(f"{'='*70}")
print(f"\n  Matched pairs: {total}")
print(f"  Qty correct:   {qty_ok} ({qty_ok/total*100:.0f}%)  — {total-qty_ok} WRONG")
print(f"  Unit correct:  {unit_ok} ({unit_ok/total*100:.0f}%)  — {total-unit_ok} WRONG")
print(f"  Ctx correct:   {ctx_ok} ({ctx_ok/total*100:.0f}%)  — {total-ctx_ok} WRONG")
print(f"  3F correct:    {f3_ok} ({f3_ok/total*100:.0f}%)")
print(f"  4F correct:    {f4_ok} ({f4_ok/total*100:.0f}%)")

# 2. What's killing 3F? (need all 3 fields correct)
qty_only_fail = sum(1 for r in pairs if r['qty_match']!='True' and r['unit_match']=='True')
unit_only_fail = sum(1 for r in pairs if r['unit_match']!='True' and r['qty_match']=='True')
both_fail = sum(1 for r in pairs if r['qty_match']!='True' and r['unit_match']!='True')

print(f"\n  3F FAILURE BREAKDOWN (of {total-f3_ok} non-3F pairs):")
print(f"    Qty wrong only:  {qty_only_fail}")
print(f"    Unit wrong only: {unit_only_fail}")
print(f"    Both wrong:      {both_fail}")

# 3. Images with most absolute 4F potential (high match count, low 4F)
print(f"\n  TOP IMPROVEMENT TARGETS (by absolute 4F potential):")
print(f"  {'IMAGE':<15} {'GT':>4} {'MATCH':>5} {'4F%':>5} {'CORRECT':>7} {'POSSIBLE':>8} {'GAP':>5}")
targets = []
for r in per_img:
    match = int(r['match'])
    f4pct = int(r['full4f_pct'])
    correct = round(match * f4pct / 100)
    gap = match - correct
    if gap >= 3:
        targets.append((r['image_id'], int(r['gt']), match, f4pct, correct, gap))
targets.sort(key=lambda x: -x[5])
for img, gt, match, f4, correct, gap in targets[:15]:
    print(f"  {img:<15} {gt:4} {match:5} {f4:4}% {correct:7} {match:8} {gap:5}")

# 4. Unmatched GT tuples (nutrient recall gap)
total_gt = sum(int(r['gt']) for r in per_img)
total_match = sum(int(r['match']) for r in per_img)
print(f"\n  NUTRIENT RECALL GAP:")
print(f"    Total GT: {total_gt}, Matched: {total_match}, Unmatched: {total_gt - total_match}")
print(f"\n  Images with most unmatched GT:")
for r in sorted(per_img, key=lambda x: int(x['gt'])-int(x['match']), reverse=True)[:10]:
    unmatched = int(r['gt']) - int(r['match'])
    if unmatched > 0:
        print(f"    {r['image_id']:<15} GT={r['gt']:>3}  matched={r['match']:>3}  unmatched={unmatched}")