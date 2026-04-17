"""Find images where 3F > 4F — pure context failures."""
import csv

with open('results/graph_v2_exp17/pair_details.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

# Find rows where 3F correct but 4F wrong (context is sole failure)
ctx_only_fails = [r for r in pairs 
                  if r['full_match_3f'] == 'True' and r['full_match_4f'] == 'False']

print(f"Total 3F-correct but 4F-wrong: {len(ctx_only_fails)} tuples")
print(f"\nContext mismatch patterns:")
print(f"  {'GT_CONTEXT':<25} {'PRED_CONTEXT':<25} {'COUNT':>5}")
print(f"  {'-'*25} {'-'*25} {'-'*5}")

from collections import Counter
patterns = Counter((r['gt_context'][:24], r['pred_context'][:24]) for r in ctx_only_fails)
for (gt, pred), count in patterns.most_common(20):
    print(f"  {gt:<25} {pred:<25} {count:>5}")

print(f"\nImages with most context-only failures:")
by_img = Counter(r['image_id'] for r in ctx_only_fails)
for img, count in by_img.most_common(15):
    print(f"  {img:<15} {count} tuples losing context")