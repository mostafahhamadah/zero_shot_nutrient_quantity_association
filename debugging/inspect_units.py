"""Inspect unit matching failures — images with high qty but 0% unit."""
import csv

IMAGES = ['30.png', '8.png', '42.png', '45_1.png']

for exp in ['graph_v2_exp13']:
    path = f'outputs/{exp}/pair_details.csv'
    with open(path) as f:
        rows = list(csv.DictReader(f))

    for img in IMAGES:
        img_rows = [r for r in rows if r['image_id'] == img]
        if not img_rows:
            continue
        print(f"\n{'='*90}")
        print(f"  {img} — {len(img_rows)} matched pairs")
        print(f"{'='*90}")
        print(f"  {'GT_NUTRIENT':<25} {'GT_QTY':<8} {'GT_UNIT':<8} {'PRED_QTY':<10} {'PRED_UNIT':<10} {'Q':3} {'U':3}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*3} {'-'*3}")
        for r in img_rows:
            qm = 'Y' if r['qty_match'] == 'True' else ' '
            um = 'Y' if r['unit_match'] == 'True' else ' '
            print(f"  {r['gt_nutrient'][:24]:<25} {r['gt_qty']:<8} {r['gt_unit']:<8} "
                  f"{r['pred_qty']:<10} {r['pred_unit']:<10} {qm:3} {um:3}")