"""
create_gold_4field.py
=====================
Creates a reduced gold annotations CSV with exactly 5 columns:
    image_id, nutrient, quantity, unit, context

Reads:  data/annotations/gold_annotations.csv  (full schema)
Writes: data/annotations/gold_annotations_4field.csv  (reduced schema)

This 4-field CSV is the reference file used for all performance
measurement in experiment_01_final and onwards.

Usage:
    python create_gold_4field.py
"""

import csv
from pathlib import Path

INPUT_CSV  = 'data/annotations/gold_annotations_updated.csv'
OUTPUT_CSV = 'data/annotations/gold_annotations_4field.csv'
FIELDS_OUT = ['image_id', 'nutrient', 'quantity', 'unit', 'context']

# ── Read source ───────────────────────────────────────────────────────────────

input_path = Path(INPUT_CSV)
if not input_path.exists():
    raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

rows_in = []
with open(input_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    available_fields = reader.fieldnames
    print(f"Source columns : {available_fields}")
    for row in reader:
        rows_in.append(row)

print(f"Source rows    : {len(rows_in)}")

# ── Check required columns exist ─────────────────────────────────────────────

missing = [f for f in FIELDS_OUT if f not in available_fields]
if missing:
    raise ValueError(f"Missing columns in source CSV: {missing}")

# ── Write reduced CSV ─────────────────────────────────────────────────────────

rows_out = []
for row in rows_in:
    rows_out.append({
        'image_id': row.get('image_id', '').strip(),
        'nutrient': row.get('nutrient', '').strip(),
        'quantity': row.get('quantity', '').strip(),
        'unit':     row.get('unit',     '').strip(),
        'context':  row.get('context',  '').strip(),
    })

output_path = Path(OUTPUT_CSV)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS_OUT)
    writer.writeheader()
    writer.writerows(rows_out)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"Output columns : {FIELDS_OUT}")
print(f"Output rows    : {len(rows_out)}")
print(f"Saved to       : {OUTPUT_CSV}")
print()

# Quick sanity check — show first 5 rows
print("First 5 rows:")
print(f"  {'IMAGE_ID':<12} {'NUTRIENT':<35} {'QTY':<8} {'UNIT':<6} CONTEXT")
print("  " + "-"*75)
for row in rows_out[:5]:
    print(f"  {row['image_id']:<12} {row['nutrient'][:33]:<35} "
          f"{row['quantity']:<8} {row['unit']:<6} {row['context']}")

# Count unique images
unique_images = sorted(set(r['image_id'] for r in rows_out))
print(f"\nUnique images  : {len(unique_images)}")
print(f"Image IDs      : {', '.join(unique_images)}")