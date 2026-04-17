"""Inspect zero-match images: compare predicted nutrients vs GT nutrients."""
import csv, json
from pathlib import Path

IMAGES = ['68.jpeg', '61.png', '63.png', '51.png', '53.png', '201.jpeg', '12.png']
EXP = 'graph_v2_exp14'

# Load predicted tuples
with open(f'results/{EXP}/tuples.csv') as f:
    pred_rows = list(csv.DictReader(f))

# Load GT - unpack nutrients list
gt_all = []
for jf in sorted(Path('data/annotations').glob('*.json')):
    with open(jf, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'nutrients' in data:
        img_id = data.get('image_id', jf.stem)
        for t in data['nutrients']:
            t['image_id'] = img_id
            gt_all.append(t)
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and 'nutrients' in entry:
                img_id = entry.get('image_id', jf.stem)
                for t in entry['nutrients']:
                    t['image_id'] = img_id
                    gt_all.append(t)
            else:
                gt_all.append(entry)

print(f"Total GT tuples: {len(gt_all)}\n")

for img in IMAGES:
    preds = [r for r in pred_rows if r['image_id'] == img]
    gts = [r for r in gt_all if r.get('image_id') == img]

    print(f"\n{'='*90}")
    print(f"  {img} -- GT: {len(gts)}, PRED: {len(preds)}")
    print(f"{'='*90}")

    print(f"\n  GT NUTRIENTS (first 12):")
    for g in gts[:12]:
        n = str(g.get('nutrient',''))[:40]
        q = str(g.get('quantity',''))[:8]
        u = str(g.get('unit',''))[:6]
        c = str(g.get('context',''))[:20]
        print(f"    {n:<42} {q:<8} {u:<6} {c}")
    if len(gts) > 12:
        print(f"    ... and {len(gts)-12} more")

    print(f"\n  PRED NUTRIENTS (first 12):")
    for p in preds[:12]:
        n = str(p.get('nutrient',''))[:40]
        q = str(p.get('quantity',''))[:8]
        u = str(p.get('unit',''))[:6]
        c = str(p.get('context',''))[:20]
        print(f"    {n:<42} {q:<8} {u:<6} {c}")
    if len(preds) > 12:
        print(f"    ... and {len(preds)-12} more")