
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── PATH RESOLUTION ──────────────────────────────────────────────────────────
CANDIDATE_PATHS = [

    "data/annotations/gold_annotations.csv"
]

CSV_PATH = None
for p in CANDIDATE_PATHS:
    if Path(p).exists():
        CSV_PATH = p
        break

if CSV_PATH is None:
    ann_dir = Path("data/annotations")
    print("ERROR: Could not locate master CSV. Files found in data/annotations/:")
    if ann_dir.exists():
        for f in sorted(ann_dir.iterdir()):
            print(f"  {f.name}")
    else:
        print("  data/annotations/ directory does not exist.")
    print("\nEdit CANDIDATE_PATHS at the top of this script to match your filename.")
    sys.exit(1)

print(f"Loading: {CSV_PATH}")

# ── ROBUST CSV READER ────────────────────────────────────────────────────────
# Handles: trailing commas (>7 fields), short rows (<7 fields after decimal
# comma fix), BOM, and any remaining bad lines.
COLUMNS = ["image_id", "nutrient", "quantity", "unit", "context",
           "nrv_percent", "serving_size"]

import io, csv as _csv

rows = []
with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
    reader = _csv.reader(fh)
    next(reader)  # skip header
    for line in reader:
        # Pad short rows with empty strings, truncate long rows to 7
        line = (line + [""] * 7)[:7]
        rows.append(line)

df = pd.DataFrame(rows, columns=COLUMNS)

# Replace empty strings with NaN for numeric columns
df.replace("", np.nan, inplace=True)

# ── COLUMN NAME NORMALISATION ─────────────────────────────────────────────────
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

COL_IMAGE   = next(c for c in df.columns if "image" in c)
COL_NUT     = next(c for c in df.columns if "nutri" in c)
COL_QTY     = next(c for c in df.columns if "quant" in c)
COL_UNIT    = next(c for c in df.columns if "unit" in c)
COL_CTX     = next(c for c in df.columns if "context" in c)
COL_NRV     = next((c for c in df.columns if "nrv" in c or "percent" in c), None)
COL_SERVING = next((c for c in df.columns if "serving" in c), None)

SEP  = "=" * 65
SEP2 = "-" * 65

# ── 1. HIGH-LEVEL COUNTS ──────────────────────────────────────────────────────
n_images    = df[COL_IMAGE].nunique()
n_tuples    = len(df)
n_nutrients = df[COL_NUT].nunique()
per_image   = df.groupby(COL_IMAGE).size()

print(f"\n{SEP}")
print("  1. HIGH-LEVEL COUNTS")
print(SEP)
print(f"  {'Total images':<40} : {n_images}")
print(f"  {'Total ground-truth tuples':<40} : {n_tuples}")
print(f"  {'Unique nutrient name strings':<40} : {n_nutrients}")
print(SEP2)
print(f"  {'Tuples/image  min':<40} : {per_image.min()}")
print(f"  {'Tuples/image  max':<40} : {per_image.max()}")
print(f"  {'Tuples/image  mean':<40} : {per_image.mean():.1f}")
print(f"  {'Tuples/image  median':<40} : {per_image.median():.1f}")
print(f"  {'Tuples/image  std':<40} : {per_image.std():.1f}")

# ── 2. CONTEXT DISTRIBUTION ───────────────────────────────────────────────────
ctx_counts = df[COL_CTX].value_counts(dropna=False)
multi_ctx  = (df.groupby(COL_IMAGE)[COL_CTX].nunique() > 1).sum()
single_ctx = n_images - multi_ctx

print(f"\n{SEP}")
print("  2. CONTEXT DISTRIBUTION")
print(SEP)
for ctx, cnt in ctx_counts.items():
    bar = "#" * int(cnt / n_tuples * 50)
    print(f"  {str(ctx):<25} : {cnt:>4}  ({cnt/n_tuples*100:5.1f}%)  {bar}")
print(SEP2)
print(f"  {'Single-context images':<40} : {single_ctx} / {n_images}  ({single_ctx/n_images*100:.1f}%)")
print(f"  {'Multi-context images':<40} : {multi_ctx} / {n_images}  ({multi_ctx/n_images*100:.1f}%)")

# ── 3. UNIT DISTRIBUTION ──────────────────────────────────────────────────────
unit_counts = df[COL_UNIT].value_counts(dropna=False)

print(f"\n{SEP}")
print("  3. UNIT DISTRIBUTION")
print(SEP)
for unit, cnt in unit_counts.items():
    bar = "#" * int(cnt / n_tuples * 50)
    print(f"  {str(unit):<20} : {cnt:>4}  ({cnt/n_tuples*100:5.1f}%)  {bar}")

# ── 4. QUANTITY STATISTICS ────────────────────────────────────────────────────
qty_numeric   = pd.to_numeric(df[COL_QTY], errors="coerce")
n_parseable   = qty_numeric.notna().sum()
n_unparseable = qty_numeric.isna().sum()
non_numeric_vals = list(df[qty_numeric.isna()][COL_QTY].unique())

print(f"\n{SEP}")
print("  4. QUANTITY STATISTICS")
print(SEP)
print(f"  {'Numeric (float-parseable)':<40} : {n_parseable:>4}  ({n_parseable/n_tuples*100:.1f}%)")
print(f"  {'Non-numeric (threshold notation)':<40} : {n_unparseable:>4}  ({n_unparseable/n_tuples*100:.1f}%)")
print(f"  {'Non-numeric values found':<40} : {non_numeric_vals}")
print(SEP2)
print(f"  {'Min':<40} : {qty_numeric.min()}")
print(f"  {'Max':<40} : {qty_numeric.max()}")
print(f"  {'Mean':<40} : {qty_numeric.mean():.3f}")
print(f"  {'Median':<40} : {qty_numeric.median():.3f}")
print(f"  {'Std':<40} : {qty_numeric.std():.3f}")
print(SEP2)
print("  Magnitude buckets (numeric only):")
bins = [0, 1, 10, 100, 1000, 10000, float("inf")]
lbls = ["< 1", "1 - 10", "10 - 100", "100 - 1000", "1000 - 10000", "> 10000"]
bk = pd.cut(qty_numeric.dropna(), bins=bins, labels=lbls)
total_numeric = len(qty_numeric.dropna())
for label, cnt in bk.value_counts().sort_index().items():
    bar = "#" * int(cnt / total_numeric * 40)
    print(f"    {label:<20} : {cnt:>4}  ({cnt/total_numeric*100:5.1f}%)  {bar}")

# ── 5. NRV PERCENT STATISTICS ─────────────────────────────────────────────────
if COL_NRV:
    nrv = pd.to_numeric(df[COL_NRV], errors="coerce")
    n_nrv_present = nrv.notna().sum()
    n_nrv_missing = nrv.isna().sum()

    print(f"\n{SEP}")
    print("  5. NRV% (NUTRIENT REFERENCE VALUE) STATISTICS")
    print(SEP)
    print(f"  {'Tuples with NRV present':<40} : {n_nrv_present:>4}  ({n_nrv_present/n_tuples*100:.1f}%)")
    print(f"  {'Tuples with NRV absent':<40} : {n_nrv_missing:>4}  ({n_nrv_missing/n_tuples*100:.1f}%)")
    print(SEP2)
    if n_nrv_present > 0:
        print(f"  {'NRV min':<40} : {nrv.min():.1f}%")
        print(f"  {'NRV max':<40} : {nrv.max():.1f}%")
        print(f"  {'NRV mean':<40} : {nrv.mean():.1f}%")
        print(f"  {'NRV median':<40} : {nrv.median():.1f}%")
        print(f"  {'NRV std':<40} : {nrv.std():.1f}%")
        print(f"  {'Tuples with NRV > 100%':<40} : {(nrv > 100).sum()}")
        print(f"  {'Tuples with NRV = 100%':<40} : {(nrv == 100).sum()}")
    print(SEP2)
    print("  NRV coverage by context:")
    for ctx in df[COL_CTX].dropna().unique():
        sub    = df[df[COL_CTX] == ctx]
        n_with = pd.to_numeric(sub[COL_NRV], errors="coerce").notna().sum()
        n_tot  = len(sub)
        bar    = "#" * int(n_with / n_tot * 30)
        print(f"    {str(ctx):<25} : {n_with:>4} / {n_tot:<4}  ({n_with/n_tot*100:5.1f}%)  {bar}")

# ── 6. SERVING SIZE STATISTICS ────────────────────────────────────────────────
if COL_SERVING:
    srv = df[COL_SERVING]
    n_srv_present = srv.notna().sum()
    n_srv_missing = srv.isna().sum()

    print(f"\n{SEP}")
    print("  6. SERVING SIZE STATISTICS")
    print(SEP)
    print(f"  {'Tuples with serving size present':<40} : {n_srv_present:>4}  ({n_srv_present/n_tuples*100:.1f}%)")
    print(f"  {'Tuples with serving size absent':<40} : {n_srv_missing:>4}  ({n_srv_missing/n_tuples*100:.1f}%)")
    print(SEP2)
    print("  Top 15 most frequent serving size values:")
    for sv, cnt in srv.value_counts(dropna=True).head(15).items():
        bar = "#" * cnt
        print(f"    {str(sv):<25} : {cnt:>3}  {bar}")

# ── 7. LAYOUT COMPLEXITY ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  7. LAYOUT COMPLEXITY (tuples per image distribution)")
print(SEP)
bins2 = [0, 2, 5, 10, 15, 20, 30, 50, 999]
lbls2 = ["1-2", "3-5", "6-10", "11-15", "16-20", "21-30", "31-50", "50+"]
bk2   = pd.cut(per_image, bins=bins2, labels=lbls2)
for label, cnt in bk2.value_counts().sort_index().items():
    bar = "#" * (cnt * 3)
    print(f"  {label:<10} tuples/image : {cnt:>2} images  {bar}")
print(SEP2)
print("  Most complex images (top 10 by tuple count):")
for img, cnt in per_image.sort_values(ascending=False).head(10).items():
    print(f"    {img:<20} : {cnt} tuples")

# ── 8. TOP-20 NUTRIENT NAMES ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("  8. TOP-20 MOST FREQUENT NUTRIENT NAME STRINGS")
print(SEP)
for i, (name, cnt) in enumerate(df[COL_NUT].value_counts().head(20).items(), 1):
    bar = "#" * cnt
    print(f"  {i:>2}. {str(name)[:52]:<52} : {cnt}  {bar}")

# ── 9. NULL / MISSING VALUE AUDIT ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  9. NULL / MISSING VALUE AUDIT (per field)")
print(SEP)
all_cols = [COL_IMAGE, COL_NUT, COL_QTY, COL_UNIT, COL_CTX]
if COL_NRV:     all_cols.append(COL_NRV)
if COL_SERVING: all_cols.append(COL_SERVING)

for col in all_cols:
    nulls   = df[col].isna().sum()
    empties = (df[col].astype(str).str.strip() == "").sum()
    total   = nulls + empties
    status  = "COMPLETE" if total == 0 else f"MISSING: {total} ({total/n_tuples*100:.1f}%)"
    print(f"  {col:<20} : {status}")

# ── 10. PER-IMAGE TUPLE COUNTS (full table) ───────────────────────────────────
print(f"\n{SEP}")
print("  10. PER-IMAGE TUPLE COUNT (all images, sorted descending)")
print(SEP)
for img, cnt in per_image.sort_values(ascending=False).items():
    bar = "#" * cnt
    print(f"  {img:<20} : {cnt:>3}  {bar}")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
nrv_str = (f"{n_nrv_present} / {n_tuples} ({n_nrv_present/n_tuples*100:.1f}%)"
           if COL_NRV else "N/A")
srv_str = (f"{n_srv_present} / {n_tuples} ({n_srv_present/n_tuples*100:.1f}%)"
           if COL_SERVING else "N/A")

print(f"\n{SEP}")
print("  THESIS SECTION 2.3 — SUMMARY TABLE")
print(SEP)
rows = [
    ("Total images",                          str(n_images)),
    ("Total ground-truth tuples",             str(n_tuples)),
    ("Unique nutrient name strings",          str(n_nutrients)),
    ("Tuples/image  (min / mean / max)",      f"{per_image.min()} / {per_image.mean():.1f} / {per_image.max()}"),
    ("Tuples/image  std",                     f"{per_image.std():.1f}"),
    ("Multi-context images",                  f"{multi_ctx} / {n_images} ({multi_ctx/n_images*100:.1f}%)"),
    ("Context: per_serving",                  f"{ctx_counts.get('per_serving', 0)} tuples ({ctx_counts.get('per_serving',0)/n_tuples*100:.1f}%)"),
    ("Context: per_100g",                     f"{ctx_counts.get('per_100g', 0)} tuples ({ctx_counts.get('per_100g',0)/n_tuples*100:.1f}%)"),
    ("Context: per_100ml",                    f"{ctx_counts.get('per_100ml', 0)} tuples"),
    ("Context: per_daily_dose",               f"{ctx_counts.get('per_daily_dose', 0)} tuples"),
    ("Most frequent unit",                    f"{unit_counts.index[0]} ({unit_counts.iloc[0]} tuples, {unit_counts.iloc[0]/n_tuples*100:.1f}%)"),
    ("Numeric quantity values",               f"{n_parseable} / {n_tuples} ({n_parseable/n_tuples*100:.1f}%)"),
    ("Non-numeric quantities (<x.xx)",        f"{n_unparseable}"),
    ("Quantity range",                        f"{qty_numeric.min()} – {qty_numeric.max()}"),
    ("Tuples with NRV% present",              nrv_str),
    ("Tuples with serving size present",      srv_str),
]
for label, value in rows:
    print(f"  {label:<42} | {value}")

print(f"\n  Analysis complete. Tuples analysed: {n_tuples}\n")