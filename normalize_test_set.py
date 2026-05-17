"""
normalize_test_set.py
─────────────────────
Reads  test_set.csv  (UTF-8) and normalises every nutrient name
to a canonical English form.  Writes  test_set_normalized.csv.

Usage:
    python normalize_test_set.py
"""

import csv, re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\MSAEI\OneDrive\Desktop\zero_shot_nutrient_association")
INPUT_CSV = BASE / "test_set.csv"
OUT_CSV   = BASE / "test_set_normalized.csv"

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — exact-match dictionary  (checked BEFORE pattern rules)
#  Keys are STRIPPED + CASE-FOLDED internally; originals listed for clarity.
# ═════════════════════════════════════════════════════════════════════════════

_EXACT_MAP_RAW = {
    # ── Energy ────────────────────────────────────────────────────────────
    "Energie":                      "Energy",
    "ENERGIE":                      "Energy",
    "ENERGIE kJ":                   "Energy",
    "ENERGIE kcal":                 "Energy",
    "Brennwert":                    "Energy",
    "Energy":                       "Energy",
    "ENERGY":                       "Energy",

    # ── Fat ───────────────────────────────────────────────────────────────
    "Fett":                         "Fat",
    "FETT":                         "Fat",
    "FETT:":                        "Fat",
    "Fat":                          "Fat",
    "FAT":                          "Fat",

    # ── Saturated Fats ────────────────────────────────────────────────────
    "- davon gesättigte Fettsäuren":            "Saturated Fats",
    "davon gesättigte Fettsäuren":              "Saturated Fats",
    "DAVON GESÄTTIGTE FETTSÄUREN":              "Saturated Fats",
    "- DAVON GESÄTTIGTE FETTSÄUREN":            "Saturated Fats",
    "OF WHICH SATURATES":                       "Saturated Fats",
    "SATURATED FAT":                            "Saturated Fats",
    "Saturated Fat":                            "Saturated Fats",

    # ── Carbohydrate ──────────────────────────────────────────────────────
    "Kohlenhydrate":                "Carbohydrate",
    "KOHLENHYDRATE":                "Carbohydrate",
    "KOHLENHYDRATE:":               "Carbohydrate",
    "Carbohydrate":                 "Carbohydrate",
    "Carbohydrates":                "Carbohydrate",
    "CARBOHYDRATE":                 "Carbohydrate",

    # ── Sugars ────────────────────────────────────────────────────────────
    "- davon Zucker":               "Sugars",
    "davon Zucker":                 "Sugars",
    "DAVON ZUCKER":                 "Sugars",
    "- DAVON ZUCKER":               "Sugars",
    "OF WHICH SUGARS":              "Sugars",
    "SUGARS":                       "Sugars",
    "Sugars":                       "Sugars",
    "of which sugars":              "Sugars",

    # ── Sugar Alcohols ────────────────────────────────────────────────────
    "- DAVON MEHRWERTIGE ALKOHOLE": "Sugar Alcohols",

    # ── Fructose ──────────────────────────────────────────────────────────
    # (handled by pattern rule too, but exact match is faster)

    # ── Protein ───────────────────────────────────────────────────────────
    "Eiweiß":                       "Protein",
    "EIWEIẞ":                       "Protein",
    "EIWEISS":                      "Protein",
    "Eiweiss":                      "Protein",
    "Protein":                      "Protein",
    "PROTEIN":                      "Protein",

    # ── Fibre ─────────────────────────────────────────────────────────────
    "Ballaststoffe":                "Fibre",
    "BALLASTSTOFFE":                "Fibre",
    "Fibre":                        "Fibre",
    "DIETARY FIBRE":                "Fibre",

    # ── Salt ──────────────────────────────────────────────────────────────
    "Salz":                         "Salt",
    "SALZ":                         "Salt",
    "Salt":                         "Salt",
    "SALT":                         "Salt",

    # ── Sodium ────────────────────────────────────────────────────────────
    "Natrium":                      "Sodium",
    "Sodium":                       "Sodium",
    "SODIUM":                       "Sodium",

    # ── Calcium ───────────────────────────────────────────────────────────
    "Calcium":                      "Calcium",
    "CALCIUM":                      "Calcium",
    "Kalzium":                      "Calcium",

    # ── Magnesium ─────────────────────────────────────────────────────────
    "Magnesium":                    "Magnesium",
    "MAGNESIUM":                    "Magnesium",

    # ── Potassium ─────────────────────────────────────────────────────────
    "Potassium":                    "Potassium",
    "Kalium":                       "Potassium",

    # ── Chloride ──────────────────────────────────────────────────────────
    "Chlorid":                      "Chloride",
    "Chlorid(e)":                   "Chloride",
    "Chloride":                     "Chloride",

    # ── Phosphorus ────────────────────────────────────────────────────────
    "Phosphorus":                   "Phosphorus",

    # ── Zinc ──────────────────────────────────────────────────────────────
    "Zink":                         "Zinc",
    "ZINK":                         "Zinc",
    "Zinc":                         "Zinc",

    # ── Selenium ──────────────────────────────────────────────────────────
    "Selen":                        "Selenium",
    "SELEN":                        "Selenium",
    "Selenium":                     "Selenium",

    # ── Copper ────────────────────────────────────────────────────────────
    "Kupfer":                       "Copper",
    "KUPFER":                       "Copper",
    "Copper":                       "Copper",

    # ── Manganese ─────────────────────────────────────────────────────────
    "Mangan":                       "Manganese",
    "MANGAN":                       "Manganese",
    "Manganese":                    "Manganese",

    # ── Chromium ──────────────────────────────────────────────────────────
    "Chrom":                        "Chromium",
    "CHROM":                        "Chromium",
    "Chromium":                     "Chromium",

    # ── Iodine ────────────────────────────────────────────────────────────
    "JOD":                          "Iodine",
    "Jod":                          "Iodine",

    # ── Molybdenum ────────────────────────────────────────────────────────
    "MOLYBDÄN":                     "Molybdenum",
    "Molybdän":                     "Molybdenum",

    # ── Vitamins ──────────────────────────────────────────────────────────
    "Vitamin A":                    "Vitamin A",
    "Vitamin B1":                   "Vitamin B1",
    "VITAMIN B1":                   "Vitamin B1",
    "B1":                           "Vitamin B1",
    "Thiamin":                      "Vitamin B1",
    "Vitamin B2":                   "Vitamin B2",
    "VITAMIN B2":                   "Vitamin B2",
    "B2":                           "Vitamin B2",
    "Vitamin B3":                   "Niacin",
    "B3":                           "Niacin",
    "Niacin":                       "Niacin",
    "NIACIN":                       "Niacin",
    "Niacin (NE)":                  "Niacin",
    "B5":                           "Pantothenic Acid",
    "Pantothensäure":               "Pantothenic Acid",
    "PANTOTHENSÄURE":               "Pantothenic Acid",
    "Vitamin B6":                   "Vitamin B6",
    "VITAMIN B6":                   "Vitamin B6",
    "B6":                           "Vitamin B6",
    "Vitamin B12":                  "Vitamin B12",
    "VITAMIN B12":                  "Vitamin B12",
    "B12":                          "Vitamin B12",
    "Vitamin C":                    "Vitamin C",
    "VITAMIN C":                    "Vitamin C",
    "C":                            "Vitamin C",
    "Vitamin D":                    "Vitamin D",
    "Vitamin D3":                   "Vitamin D3",
    "VITAMIN D3":                   "Vitamin D3",
    "Vitamin E":                    "Vitamin E",
    "VITAMIN E":                    "Vitamin E",
    "Vitamin E (α-TE)":             "Vitamin E",
    "Vitamin K":                    "Vitamin K",
    "VITAMIN K":                    "Vitamin K",
    "Biotin":                       "Biotin",
    "BIOTIN":                       "Biotin",
    "Folsäure":                     "Folic Acid",
    "FOLSÄURE":                     "Folic Acid",
    "Folic acid":                   "Folic Acid",
    "Folic Acid":                   "Folic Acid",
    "Caffeine":                     "Caffeine",
    "KOFFEIN":                      "Caffeine",

    # ── Amino acids ───────────────────────────────────────────────────────
    # (handled mainly by pattern rule — first segment before /)

    # ── Specialty (keep as-is) ────────────────────────────────────────────
    "Magnesium (gesamt)":           "Magnesium (gesamt)",
    "Magnesiumsalze der Citronensäure": "Magnesiumsalze der Citronensäure",
    "Magnesiumoxid":                "Magnesiumoxid",
    "Magnesiumbisglycinat":         "Magnesiumbisglycinat",
    "Magnesiummalat":               "Magnesiummalat",
    "Melissen Extrakt":             "Melissen Extrakt",
    "Colecalciferol":               "Colecalciferol",
    "Creatine monohydrate":         "Creatine monohydrate",

    # ── Specialty (title-case) ────────────────────────────────────────────
    "GREEN TEA":                    "Green Tea",
    "INOSITOL":                     "Inositol",
    "ALPHA-LIPONSÄURE":             "Alpha-Liponsäure",
    "CHOLIN":                       "Choline",
    "RUTIN":                        "Rutin",
    "LACTOBACILLUS ACIDOPHILUS":     "Lactobacillus Acidophilus",
    "BIFIDOBACTERIUM BIFIDUM":       "Bifidobacterium Bifidum",
}

# Build case-sensitive lookup (exact match first, then stripped)
EXACT_MAP = {}
for k, v in _EXACT_MAP_RAW.items():
    EXACT_MAP[k.strip()] = v


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — pattern-based rules  (applied when exact match fails)
# ═════════════════════════════════════════════════════════════════════════════

# After extracting the first segment of a multilingual name (split on / or |),
# we try to map the cleaned segment to a canonical name.
_FIRST_SEGMENT_MAP = {
    # Energy
    "energie":              "Energy",
    "energy":               "Energy",
    "brennwert":            "Energy",
    "energy | energie":     "Energy",  # pipe-separated short

    # Fat
    "fett":                 "Fat",
    "fat":                  "Fat",
    "vetten":               "Fat",

    # Saturated Fats
    "davon gesättigte fettsäuren":  "Saturated Fats",
    "- davon gesättigte fettsäuren":"Saturated Fats",
    "of which saturates":           "Saturated Fats",
    "of which saturated":           "Saturated Fats",
    "- of which saturates":         "Saturated Fats",
    "waarvan verzadigde vetzuren":  "Saturated Fats",
    "- waarvan verzadigde vetzuren":"Saturated Fats",
    "saturated fat":                "Saturated Fats",

    # Carbohydrate
    "kohlenhydrate":        "Carbohydrate",
    "carbohydrate":         "Carbohydrate",
    "carbohydrates":        "Carbohydrate",
    "koolhydraten":         "Carbohydrate",

    # Sugars
    "davon zucker":         "Sugars",
    "- davon zucker":       "Sugars",
    "davon zuckerarten":    "Sugars",
    "of which sugars":      "Sugars",
    "- of which sugars":    "Sugars",
    "waarvan suikers":      "Sugars",
    "sugars":               "Sugars",

    # Fructose
    "davon fructose":       "Fructose",
    "davon fructose ":      "Fructose",

    # Protein
    "eiweiß":               "Protein",
    "eiweiss":              "Protein",
    "protein":              "Protein",
    "protein(e)":           "Protein",
    "eiwitten":             "Protein",

    # Fibre
    "ballaststoffe":        "Fibre",
    "fibre":                "Fibre",

    # Salt
    "salz":                 "Salt",
    "salt":                 "Salt",
    "zout":                 "Salt",

    # Sodium
    "natrium":              "Sodium",
    "sodium":               "Sodium",

    # Calcium
    "calcium":              "Calcium",
    "kalzium":              "Calcium",

    # Magnesium
    "magnesium":            "Magnesium",

    # Potassium
    "kalium":               "Potassium",
    "potassium":            "Potassium",

    # Chloride
    "chlorid":              "Chloride",
    "chlorid(e)":           "Chloride",

    # Phosphorus
    "phosphor":             "Phosphorus",

    # Selenium
    "selen":                "Selenium",
    "selenium":             "Selenium",

    # Zinc
    "zink":                 "Zinc",
    "zinc":                 "Zinc",

    # Copper
    "kupfer":               "Copper",
    "copper":               "Copper",

    # Manganese
    "mangan":               "Manganese",
    "manganese":            "Manganese",

    # Chromium
    "chrom":                "Chromium",
    "chromium":             "Chromium",

    # Vitamins
    "vitamin a":            "Vitamin A",
    "vitamin a (aus":       "Vitamin A",
    "vitamin b1":           "Vitamin B1",
    "vitamin b2":           "Vitamin B2",
    "vitamin(e) b2":        "Vitamin B2",
    "vitamin b6":           "Vitamin B6",
    "vitamin(e) b6":        "Vitamin B6",
    "vitamin b6 | vitamine b6": "Vitamin B6",
    "vitamin b12":          "Vitamin B12",
    "vitamin(e) b12":       "Vitamin B12",
    "vitamin c":            "Vitamin C",
    "vitamin(e) c":         "Vitamin C",
    "vitamin d":            "Vitamin D",
    "vitamin d3":           "Vitamin D3",
    "vitamin e":            "Vitamin E",
    "vitamin e | vitamine e": "Vitamin E",
    "vitamin k":            "Vitamin K",
    "niacin":               "Niacin",
    "niacin(e)":            "Niacin",
    "pantothensäure":       "Pantothenic Acid",
    "folsäure":             "Folic Acid",
    "folic acid":           "Folic Acid",
    "biotin":               "Biotin",
    "koffein":              "Caffeine",
    "caffeine":             "Caffeine",

    # Amino acids
    "l-valine":             "L-Valine",
    "l-leucine":            "L-Leucine",
    "l-isoleucine":         "L-Isoleucine",
    "l-lysine":             "L-Lysine",
    "l-histidine":          "L-Histidine",
    "l-methionine":         "L-Methionine",
    "l-phenylalanine":      "L-Phenylalanine",
    "l-threonine":          "L-Threonine",
    "l-tryptophan":         "L-Tryptophan",

    # Misc
    "melissen extrakt":     "Melissen Extrakt",
    "creatine monohydrate": "Creatine monohydrate",
}


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — normalisation function
# ═════════════════════════════════════════════════════════════════════════════

def _clean(s: str) -> str:
    """Strip leading dash/bullet, trailing colon, collapse whitespace."""
    s = s.strip()
    s = re.sub(r'^[-–—•]\s*', '', s)      # leading dash
    s = s.rstrip(':')                       # trailing colon
    s = re.sub(r'\s*\.{2,}\s*', ' / ', s)  # dots → slash (Brennwert ...... Energie)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _first_segment(name: str, sep: str) -> str:
    """Return first segment before separator, stripped and lowered."""
    return name.split(sep)[0].strip().lower()


def normalise_nutrient(raw: str) -> str:
    """Map a raw nutrient name to its canonical form."""
    stripped = raw.strip()

    # 1) Exact match (case-sensitive)
    if stripped in EXACT_MAP:
        return EXACT_MAP[stripped]

    # 2) Clean (remove dash prefix, colon suffix, dots)
    cleaned = _clean(stripped)
    if cleaned in EXACT_MAP:
        return EXACT_MAP[cleaned]

    # 3) Split on multilingual separators and try first segment
    #    Separators: " / ", " | ", "/"  (in priority order)
    for sep in [' | ', ' / ', '/ ', '/']:
        if sep in cleaned:
            first = cleaned.split(sep)[0].strip()
            # exact match on first segment
            if first in EXACT_MAP:
                return EXACT_MAP[first]
            # lower-case lookup
            fl = first.lower()
            if fl in _FIRST_SEGMENT_MAP:
                return _FIRST_SEGMENT_MAP[fl]
            # also try cleaned first segment (remove dash again)
            fl_clean = re.sub(r'^[-–—•]\s*', '', fl).strip()
            if fl_clean in _FIRST_SEGMENT_MAP:
                return _FIRST_SEGMENT_MAP[fl_clean]

    # 4) Lower-case lookup on full cleaned name
    cl = cleaned.lower()
    if cl in _FIRST_SEGMENT_MAP:
        return _FIRST_SEGMENT_MAP[cl]

    # 5) Remove trailing asterisks/stars
    cl_nostar = re.sub(r'\*+$', '', cleaned).strip()
    if cl_nostar in EXACT_MAP:
        return EXACT_MAP[cl_nostar]
    cl_nostar_l = cl_nostar.lower()
    if cl_nostar_l in _FIRST_SEGMENT_MAP:
        return _FIRST_SEGMENT_MAP[cl_nostar_l]

    # 6) Pattern: "Koffein (aus …) / Caffeine (from …)" → Caffeine
    if re.match(r'(?i)koffein\b', cleaned):
        return "Caffeine"
    if re.match(r'(?i)vitamin\s*a\s*\(', cleaned):
        return "Vitamin A"

    # 7) If nothing matched, return cleaned version (preserves original)
    return cleaned


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — read → normalise → write
# ═════════════════════════════════════════════════════════════════════════════

COLUMNS = ["image_id", "nutrient", "quantity", "unit", "context", "nrv_percent", "serving_size"]

rows = []
unmapped = set()

with open(INPUT_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        original  = row["nutrient"]
        canonical = normalise_nutrient(original)

        # Track if anything fell through to fallback
        if canonical == _clean(original.strip()) and canonical != original.strip():
            unmapped.add((original.strip(), canonical))
        elif canonical not in set(_EXACT_MAP_RAW.values()) | set(_FIRST_SEGMENT_MAP.values()):
            unmapped.add((original.strip(), canonical))

        row["nutrient"] = canonical
        rows.append(row)

# Write output
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in COLUMNS})

print(f"✅ Wrote {len(rows)} rows → {OUT_CSV}")

# Report
originals = set()
canonicals = set()
with open(INPUT_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        originals.add(row["nutrient"].strip())
with open(OUT_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        canonicals.add(row["nutrient"].strip())

print(f"\n   Original unique names:   {len(originals)}")
print(f"   Normalised unique names: {len(canonicals)}")
print(f"   Reduction:               {len(originals) - len(canonicals)} names merged")

if unmapped:
    print(f"\n⚠  {len(unmapped)} name(s) fell through to fallback (not in canonical set):")
    for orig, canon in sorted(unmapped):
        print(f"   '{orig}'  →  '{canon}'")
else:
    print(f"\n✓  All names mapped to a known canonical form.")