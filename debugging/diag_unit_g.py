"""Investigate why pred_unit=g when gt_unit is kJ/µg/mg/kcal"""
import csv
from collections import Counter, defaultdict

rows = list(csv.DictReader(open('outputs/graph_v2_exp29/pair_details.csv', encoding='utf-8')))

# All cases where pred_unit=g but gt_unit != g
g_contam = [r for r in rows
            if r['unit_match'] == 'False'
            and r['pred_unit'].strip().lower() == 'g'
            and r['gt_unit'].strip().lower() != 'g']

print(f"Unit 'g' contamination: {len(g_contam)} cases\n")

# Group by gt_unit
by_gt_unit = defaultdict(list)
for r in g_contam:
    by_gt_unit[r['gt_unit'].strip().lower()].append(r)

for gt_u, cases in sorted(by_gt_unit.items(), key=lambda x: -len(x[1])):
    print(f"\n{'='*70}")
    print(f"GT_UNIT = '{gt_u}' → pred='g'  ({len(cases)} cases)")
    print(f"{'='*70}")
    for r in cases[:5]:
        print(f"  {r['image_id']:<14} {r['gt_nutrient'][:35]:<37} "
              f"gt_q={r['gt_qty']:<8} pred_q={r['pred_qty']:<8} "
              f"pred_ctx={r['pred_context'][:15]}")

# Also check: what pred tuples have unit=g? Are they from fused tokens?
pred_rows = list(csv.DictReader(open('outputs/graph_v2_exp29/tuples.csv', encoding='utf-8')))
g_preds = [r for r in pred_rows if r.get('unit', '').strip().lower() == 'g']
non_g_preds = [r for r in pred_rows if r.get('unit', '').strip().lower() not in ('g', '')]

print(f"\n{'='*70}")
print(f"PREDICTED TUPLES WITH UNIT='g': {len(g_preds)} / {len(pred_rows)}")
print(f"{'='*70}")

# Check if the qty field has fused "Xg" pattern
fused_count = 0
for r in g_preds:
    q = r.get('quantity', '')
    # The unit=g likely came from inline parse of "0g", "42g" etc.
    # Or from embedded unit extraction
    pass

# Show the g→empty cases too
g_empty = [r for r in rows
           if r['unit_match'] == 'False'
           and r['pred_unit'].strip() == ''
           and r['gt_unit'].strip().lower() == 'g']

print(f"\n{'='*70}")
print(f"GT_UNIT='g' but pred_unit=EMPTY: {len(g_empty)} cases")
print(f"{'='*70}")
for r in g_empty[:10]:
    print(f"  {r['image_id']:<14} {r['gt_nutrient'][:35]:<37} "
          f"gt_q={r['gt_qty']:<8} pred_q={r['pred_qty']:<8}")

# Now check: for the contaminated cases, what does the PREDICTED tuple look like?
# Cross-reference by image_id + pred_nutrient
print(f"\n{'='*70}")
print(f"TRACING: Where does 'g' come from for kJ/kcal nutrients?")
print(f"{'='*70}")
energy_contam = [r for r in g_contam if r['gt_unit'].strip().lower() in ('kj', 'kcal')]
for r in energy_contam[:10]:
    # Find the actual predicted tuple
    matching_preds = [p for p in pred_rows
                      if p['image_id'] == r['image_id']
                      and r['pred_nutrient'][:15].lower() in p.get('nutrient', '').lower()]
    print(f"\n  GT: {r['gt_nutrient'][:35]} qty={r['gt_qty']} unit={r['gt_unit']}")
    print(f"  PRED matched: {r['pred_nutrient'][:35]} qty={r['pred_qty']} unit={r['pred_unit']}")
    for p in matching_preds[:3]:
        print(f"    pred tuple: {p.get('nutrient','')[:35]} qty={p.get('quantity','')} unit={p.get('unit','')} ctx={p.get('context','')}")