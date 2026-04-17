"""Inspect top unmatched images — compare GT vs PRED nutrient names."""
import csv, json
from pathlib import Path

EXP = 'graph_v2_exp22'

with open(f'outputs/{EXP}/tuples.csv', encoding='utf-8') as f:
    preds = list(csv.DictReader(f))

# Load GT
gt_all = []
for jf in sorted(Path('data/annotations').glob('*.json')):
    with open(jf, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'nutrients' in data:
        img_id = data.get('image_id', jf.stem)
        for t in data['nutrients']:
            t['image_id'] = img_id
            gt_all.append(t)

# Also load analysis CSV for "missing" tags
try:
    with open(f'outputs/{EXP}/graph_v2_exp22_analysis.csv', encoding='utf-8') as f:
        analysis = list(csv.DictReader(f))
except:
    analysis = []

IMAGES = ['15.png', '12.png', '68.jpeg', '119.jpeg', '120.jpeg']

for img in IMAGES:
    img_preds = [r for r in preds if r['image_id'] == img]
    # Case-insensitive GT match
    stem = img.split('.')[0]
    img_gts = [r for r in gt_all if stem in r.get('image_id', '')]
    img_missing = [r for r in analysis if r.get('image_id','') == img and r.get('tag','') == 'missing']

    print(f"\n{'='*90}")
    print(f"  {img} — GT: {len(img_gts)}, PRED: {len(img_preds)}, Missing: {len(img_missing)}")
    print(f"{'='*90}")

    # Show GT nutrients that are missing
    if img_missing:
        print(f"\n  MISSING GT (not matched to any prediction):")
        for r in img_missing[:15]:
            print(f"    {r.get('gt_nutrient','')[:50]:<52} qty={r.get('gt_quantity',''):<8} unit={r.get('gt_unit',''):<5}")
    elif img_gts:
        print(f"\n  GT nutrients (first 10):")
        for g in img_gts[:10]:
            print(f"    {str(g.get('nutrient',''))[:50]:<52} qty={str(g.get('quantity','')):<8} unit={str(g.get('unit','')):<5}")

    print(f"\n  PRED nutrients (first 10):")
    for p in img_preds[:10]:
        print(f"    {p['nutrient'][:50]:<52} qty={str(p.get('quantity','')):<8} unit={str(p.get('unit','')):<5}")
    if not img_preds:
        print(f"    (NONE)")