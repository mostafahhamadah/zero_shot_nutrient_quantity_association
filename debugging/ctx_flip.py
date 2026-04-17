import csv
from collections import Counter

with open('outputs/graph_v2_exp23/pair_details.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

# per_100g GT but per_serving pred
flips = [r for r in pairs if r['full_match_3f']=='True' and r['full_match_4f']!='True'
         and 'per_100' in r.get('gt_context','') and 'per_serving' in r.get('pred_context','')]

print(f"per_100g -> per_serving flips: {len(flips)}")
by_img = Counter(r['image_id'] for r in flips)
for img, count in by_img.most_common():
    print(f"  {img:<15} {count} flips")
    for r in flips:
        if r['image_id'] == img:
            print(f"    {r['gt_nutrient'][:30]:<32} pred_ctx={r.get('pred_context','')}")