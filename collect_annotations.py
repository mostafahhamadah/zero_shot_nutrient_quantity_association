"""
Collect all annotation JSON files into a single test_set.csv (UTF-8).
Usage:
    python collect_annotations.py
"""

import json
import csv
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATIONS_DIR = Path(r"C:\Users\MSAEI\OneDrive\Desktop\zero_shot_nutrient_association\data\annotations")
OUTPUT_CSV      = ANNOTATIONS_DIR.parent.parent / "test_set.csv"

COLUMNS = ["image_id", "nutrient", "quantity", "unit", "context", "nrv_percent", "serving_size"]

# ── Collect ───────────────────────────────────────────────────────────────────
rows = []
json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))

print(f"Found {len(json_files)} JSON files in {ANNOTATIONS_DIR}")

for jf in json_files:
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        # image_id from the JSON field, fallback to filename stem
        raw_id   = data.get("image_id", jf.stem)
        image_id = raw_id  # keep as-is (e.g. "8.PNG")

        for n in data.get("nutrients", []):
            rows.append({
                "image_id":     image_id,
                "nutrient":     str(n.get("nutrient",     "") or "").strip(),
                "quantity":     str(n.get("quantity",     "") or "").strip(),
                "unit":         str(n.get("unit",         "") or "").strip(),
                "context":      str(n.get("context",      "") or "").strip(),
                "nrv_percent":  str(n.get("nrv_percent",  "") or "").strip(),
                "serving_size": str(n.get("serving_size", "") or "").strip(),
            })
    except Exception as e:
        print(f"  SKIP {jf.name}: {e}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} tuples from {len(json_files)} images → {OUTPUT_CSV}")