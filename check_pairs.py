import csv

with open('results/metrics/pair_details.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f'Total pairs: {len(rows)}')
print(f'Full matches: {sum(1 for r in rows if r["full_match"] == "True")}')
print()
print(f'{"IMAGE":<12} {"GT NUTRIENT":<30} {"PRED QTY":<10} {"GT QTY":<10} {"UNIT_OK":<8} {"FULL"}')
print('-' * 80)
for r in rows:
    print(
        f'{r["image_id"]:<12} '
        f'{r["gt_nutrient"][:28]:<30} '
        f'{str(r["pred_qty"]):<10} '
        f'{str(r["gt_qty"]):<10} '
        f'{str(r["unit_match"] if "unit_match" in r else ""):<8} '
        f'{r["full_match"]}'
    )