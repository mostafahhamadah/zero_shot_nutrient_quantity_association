import csv, json
from pathlib import Path

EXP = 'graph_v2_exp15'

with open(f'outputs/{EXP}/pair_details.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

gt_all = []
for jf in sorted(Path('data/annotations').glob('*.json')):
    with open(jf, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'nutrients' in data:
        img_id = data.get('image_id', jf.stem)
        for t in data['nutrients']:
            t['image_id'] = img_id
            gt_all.append(t)

with open(f'outputs/{EXP}/tuples.csv', encoding='utf-8') as f:
    preds = list(csv.DictReader(f))

print("=" * 90)
print("  CATEGORY A: 51.png / 53.png -- WHY 0 MATCHES?")
print("=" * 90)
for img in ['51.png', '53.png']:
    stem = img.split('.')[0]
    gts = [r for r in gt_all if stem in r.get('image_id','')]
    print(f"\n  {img}: GT found = {len(gts)}")
    for g in gts[:5]:
        print(f"    GT:   id={g.get('image_id','')} | {str(g.get('nutrient',''))[:60]}")
    img_preds = [r for r in preds if r['image_id'] == img]
    print(f"  {img}: PRED = {len(img_preds)}")
    for p in img_preds[:5]:
        print(f"    PRED: {p['nutrient'][:60]}")

print(f"\n{'=' * 90}")
print("  CATEGORY B: 118/119 -- QTY FAILURES")
print("=" * 90)
for img in ['118.jpeg', '119.jpeg']:
    img_pairs = [r for r in pairs if r['image_id'] == img]
    print(f"\n  {img}: {len(img_pairs)} pairs")
    for r in img_pairs[:8]:
        qm = 'Y' if r['qty_match']=='True' else ' '
        print(f"    {r['gt_nutrient'][:25]:<27} GT={r['gt_qty']:<8} PRED={r['pred_qty']:<8} {qm}")

print(f"\n{'=' * 90}")
print("  CATEGORY C: 120.jpeg -- CONTEXT FAILURES")
print("=" * 90)
img_pairs = [r for r in pairs if r['image_id'] == '120.jpeg']
print(f"  120.jpeg: {len(img_pairs)} pairs")
for r in img_pairs[:10]:
    cm = 'Y' if r['ctx_match']=='True' else ' '
    print(f"    {r['gt_nutrient'][:25]:<27} GT_CTX={r['gt_context'][:20]:<22} PRED_CTX={r['pred_context'][:20]:<22} {cm}")