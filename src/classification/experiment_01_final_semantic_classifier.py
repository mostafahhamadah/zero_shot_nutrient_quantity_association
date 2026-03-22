"""
semantic_classifier.py  v5
==========================
Stage 3 — Zero-Shot Semantic Token Classifier

CHANGE IN v5 (Fix 1 — NUTRIENT Lexicon Expansion):
  Root cause addressed: 390 missing tuples. Three sub-causes:
    (a) English-only labels (26, 30, 8, 34) — ENERGY, FAT, PROTEIN,
        DIETARY FIBRE, SODIUM, SATURATED FAT, SUGARS, FIBRE, FOLIC ACID,
        POTASSIUM, ZINC not in lexicon.
    (b) Shorthand vitamin names (34) — B1, B2, B3, B5, B6, B12, C
        not matched. Added special shorthand regex rule.
    (c) Multilingual slash-variants (2, 12, 15, 20, 33, 201) — tokens
        like 'Fett/fat', 'Kohlenhydrate/Carbohydrate', 'Energy/Energie'
        not matched because the full slash-string is not in the lexicon.
        Fix: _is_nutrient() now splits on '/' and checks each part
        independently. Any part matching the lexicon makes the whole
        token NUTRIENT.
    (d) German variants missing: 'brennwert', 'mehrwertige alkohole',
        Dutch variants: 'koolhydraten', 'eiwitten', 'vetten', 'zout'.
    (e) Sub-nutrient dash-prefix stripping: tokens like '- davon zucker'
        or '- SATURATED FAT' now strip the leading dash before matching.

  No changes to priority rule chain, graph, or association modules.

CHANGE IN v4:
  CONTEXT_MAP moved from run_experiment.py into classifier.

CHANGE IN v3:
  NRV% and serving_size classified as NOISE.
  Active labels: NUTRIENT, QUANTITY, UNIT, CONTEXT, NOISE, UNKNOWN.
"""

import re


# ── Lexicons ──────────────────────────────────────────────────────────────────

NUTRIENT_LEXICON = {
    # ── English — standard EU label fields ───────────────────────────────────
    "energy",
    "fat", "fats",
    "saturates", "saturated", "saturated fat", "saturated fats",
    "of which saturates", "of which saturated",
    "carbohydrate", "carbohydrates",
    "sugars", "sugar", "of which sugars",
    "fibre", "fiber", "fibres", "dietary fibre", "dietary fiber",
    "dietary fibres",
    "protein", "proteins",
    "salt",
    "sodium",
    "potassium",
    "chloride", "chlorides",
    "fluoride",
    "calcium",
    "magnesium",
    "iron",
    "zinc",
    "iodine",
    "selenium",
    "copper",
    "manganese",
    "chromium",
    "molybdenum",
    "phosphorus",
    "folic acid",

    # ── English — sub-nutrient dash-prefix variants ───────────────────────────
    "- saturated fat", "- saturated fats",
    "- of which saturates",
    "- sugars", "- of which sugars",
    "- dietary fibre", "- dietary fiber",

    # ── German — standard ─────────────────────────────────────────────────────
    "energie", "brennwert",
    "fett", "fette",
    "fettsauren", "fettsäuren",
    "gesattigte fettsauren", "gesättigte fettsäuren",
    "kohlenhydrate",
    "zuckerarten",
    "ballaststoffe",
    "eiweiss", "eiweiß",
    "salz",
    "natrium",
    "kalium",
    "kalzium",
    "eisen",
    "zink",
    "jod", "jodid",
    "selen",
    "kupfer",
    "mangan",
    "chrom",
    "molybdän",
    "phosphor",
    "chlorid",
    "fluorid",
    "folsäure", "folsaure",

    # ── German — sub-nutrient ─────────────────────────────────────────────────
    "davon gesattigte", "davon gesättigte",
    "davon zucker",
    "davon mehrwertige alkohole",
    "mehrwertige alkohole",

    # ── French ────────────────────────────────────────────────────────────────
    "graisses", "glucides", "dont sucres", "fibres", "sel",
    "proteines", "protéines",
    "sodium",

    # ── Dutch ─────────────────────────────────────────────────────────────────
    "koolhydraten",
    "eiwitten",
    "vetten",
    "zout",
    "vezels",
    "natrium",

    # ── Vitamins — full names ─────────────────────────────────────────────────
    "vitamin", "vitamine",
    "vitamin a", "vitamin b",
    "vitamin b1", "vitamin b2", "vitamin b3", "vitamin b5",
    "vitamin b6", "vitamin b7", "vitamin b9", "vitamin b12",
    "vitamin c", "vitamin d", "vitamin d3",
    "vitamin e", "vitamin k", "vitamin k1", "vitamin k2",
    "thiamin", "thiamine",
    "riboflavin",
    "niacin",
    "pantothenic", "pantothensäure", "pantothensaure",
    "pyridoxin",
    "biotin",
    "folat", "folate", "folic", "cobalamin",
    "ascorbic", "ascorbinsäure", "ascorbinsaure",
    "tocopherol",
    "retinol",
    "cholecalciferol",
    "menaquinon", "phylloquinon",

    # ── Fatty acids / lipids ──────────────────────────────────────────────────
    "omega", "dha", "epa", "ala",
    "linolsäure", "linolsaure",
    "linolensäure", "linolensaure",
    "monounsaturated", "polyunsaturated",
    "mehrfach ungesattigte", "einfach ungesattigte",

    # ── Amino acids ───────────────────────────────────────────────────────────
    "leucin", "isoleucin", "valin", "lysin",
    "methionin", "phenylalanin",
    "threonin", "tryptophan", "histidin",

    # ── Speciality compounds ──────────────────────────────────────────────────
    "coenzym", "coenzyme",
    "l-carnitin", "carnitin", "kreatin", "creatine",
    "inositol",
    "cholin", "choline",
    "lutein", "zeaxanthin", "lycopin",
    "astaxanthin", "resveratrol", "quercetin", "rutin",
    "koffein", "caffeine",
    "alpha-liponsäure", "alpha-liponsaure", "liponsäure", "liponsaure",

    # ── Probiotics ────────────────────────────────────────────────────────────
    "lactobacillus", "lactobacillus acidophilus",
    "bifidobacterium", "bifidobacterium bifidum",

    # ── Bilingual slash-anchors (common first-part forms) ─────────────────────
    # The _is_nutrient() slash-split handles full slash strings dynamically.
    # These anchors help when the slash-string is too long for substring match.
    "kalium /potassium", "natrium /sodium", "kalzium/calcium",
}

# ── Vitamin shorthand pattern ─────────────────────────────────────────────────
# Matches standalone shorthand vitamin/mineral labels found on English labels:
# B1, B2, B3, B5, B6, B12, C, D, E, K
# Must be used ONLY after NOISE, UNIT, CONTEXT, QUANTITY rules fail.
VITAMIN_SHORTHAND_RE = re.compile(
    r'^(b\d{1,2}|vitamin\s+[a-z]\d*|[cdek])$',
    re.IGNORECASE
)

UNIT_LEXICON = {
    "mg", "g", "kg", "mcg", "µg", "ug", "μg",
    "kj", "kcal", "cal",
    "ie", "iu", "µg re", "µg ne", "µg te",
    "mg/tag", "µg/tag", "g/tag",
    "mg/day", "µg/day", "g/day",
    "mg ne", "mg α-te",
    "kcal", "kbe",
}

CONTEXT_LEXICON = {
    "per 100g", "per 100 g", "per 100ml", "je 100g", "je 100 g",
    "pro 100g", "pro 100 g", "per 100",
    "per serving", "per portion", "pro portion",
    "per tube", "per sachet", "per capsule", "per tablet",
    "je kapsel", "je tablette", "je tagesdosis",
    "pro kapsel", "pro tablette", "pro tagesdosis",
    "per daily dose", "per tagesdosis",
    "1tube", "2 kapseln", "1 kapsel", "per 1 tube",
    "100g", "100 g",
    "pro100g", "je 1009",
    "pro 2 sticks", "pro 3 tabletten", "je 700g",
}

# ── Context normalisation map ─────────────────────────────────────────────────
CONTEXT_MAP = {
    # per_100g
    "per 100g":           "per_100g",
    "per 100 g":          "per_100g",
    "per 100":            "per_100g",
    "per100g":            "per_100g",
    "je 100g":            "per_100g",
    "je 100 g":           "per_100g",
    "je100g":             "per_100g",
    "pro 100g":           "per_100g",
    "pro 100 g":          "per_100g",
    "pro100g":            "per_100g",
    "100g":               "per_100g",
    "100 g":              "per_100g",
    "je 1009":            "per_100g",
    "je1009":             "per_100g",
    # per_100ml
    "per 100ml":          "per_100ml",
    "je 100ml":           "per_100ml",
    "pro 100ml":          "per_100ml",
    "per 100 ml":         "per_100ml",
    # per_serving
    "per serving":        "per_serving",
    "per portion":        "per_serving",
    "pro portion":        "per_serving",
    "pro portionl":       "per_serving",
    "pro portionll":      "per_serving",
    "pro Portion":        "per_serving",
    "PRO PORTION":        "per_serving",
    "per sachet":         "per_serving",
    "per tube":           "per_serving",
    "per capsule":        "per_serving",
    "per tablet":         "per_serving",
    "je kapsel":          "per_serving",
    "je tablette":        "per_serving",
    "pro kapsel":         "per_serving",
    "pro tablette":       "per_serving",
    "1 kapsel":           "per_serving",
    "2 kapseln":          "per_serving",
    "3 kapseln":          "per_serving",
    "per 1 tube":         "per_serving",
    "1tube":              "per_serving",
    "pro 2 sticks":       "per_serving",
    "pro 3 tabletten":    "per_serving",
    "pro 2 tabletten":    "per_serving",
    "je 700g":            "per_serving",
    # per_daily_dose
    "per daily dose":     "per_daily_dose",
    "per tagesdosis":     "per_daily_dose",
    "je tagesdosis":      "per_daily_dose",
    "pro tagesdosis":     "per_daily_dose",
    "per tagesration":    "per_daily_dose",
    "je tagesration":     "per_daily_dose",
}

NOISE_SUBSTRINGS = [
    "mindestens", "best before", "haltbar bis", "lot nr", "lot-nr",
    "charge", "batch", "mhd",
    "hergestellt", "manufactured", "vertrieb", "distributor",
    "verantwortlich", "responsible", "kontakt", "contact",
    "gmbh", "ag ", " kg ", "ltd", "inc.", "s.a.",
    "iso ", "haccp", "bio-siegel", "organic certified",
    "fsc", "eu-bio", "eg-öko",
    "kuehl lagern", "trocken lagern", "store in", "keep away",
    "ausserhalb der reichweite", "aus der reichweite",
    "nicht fuer kinder", "not for children",
    "verschlossen halten", "keep closed",
    "nach oeffnung", "after opening",
    "siehe siegelrand", "see seal",
    "nutrition facts", "naehrwertangaben",
    "naehrwerttabelle", "nutritional information",
    "zutaten", "ingredients",
    "de:", "en:", "fr:", "nl:", "es:", "it:", "pl:",
    "nrv", "nutrient reference", "naehrstoffbezugswert",
    "referenzwert", "empfohlene tagesdosis", "% der empfohlenen",
    "bezugswert",
]

NRV_PATTERNS = [
    re.compile(r'^\d+\s*%$'),
    re.compile(r'^\d+\s*%\s*\*?$'),
    re.compile(r'^\*?\s*\d+\s*%'),
]

SERVING_PATTERNS_RE = [
    re.compile(r'^\d+\s*(kapseln?|tabletten?|softgels?|tubes?|sachets?|stueck)$', re.IGNORECASE),
    re.compile(r'^\d+\s*(capsules?|tablets?|gummies?|drops?)$', re.IGNORECASE),
    re.compile(r'^\d+\s*(tube|sachet)\s*\([\d.,]+\s*g\)$', re.IGNORECASE),
    re.compile(r'^serving\s+size[:\s]', re.IGNORECASE),
    re.compile(r'^portionsgroesse[:\s]', re.IGNORECASE),
    re.compile(r'^verzehrempfehlung[:\s]', re.IGNORECASE),
    re.compile(r'^\d+\s*x\s*\d+', re.IGNORECASE),
]

QUANTITY_PATTERN = re.compile(
    r'^[\*\'\"]?(\d+[.,]?\d*)\s*(mg|g|kg|µg|mcg|ug|kj|kcal|ie|iu|%|ml|l)?[\*\'\"\s]*$',
    re.IGNORECASE
)

FUSED_TOKEN_RE = re.compile(
    r'^([\*\'\"]?\d+[.,]?\d*)\s*'
    r'(mg|g|kg|µg|μg|mcg|ug|kj|kcal|cal|ie|iu|ml|l)'
    r'[\*\'\"\s.,;]*$',
    re.IGNORECASE
)


def split_fused_token(token: dict) -> list:
    """
    Split fused quantity+unit token into two child tokens.
    '400mg' -> QUANTITY:'400' + UNIT:'mg'
    """
    text  = token.get("token", "").strip()
    clean = text.strip("*'\".,; ")

    if not clean:
        return [token]
    if len(clean.split()) > 2:
        return [token]
    if clean[0].isalpha():
        return [token]

    m = FUSED_TOKEN_RE.match(clean)
    if not m:
        return [token]

    qty_str  = m.group(1).strip("*'\"").strip()
    unit_str = m.group(2)

    try:
        float(qty_str.replace(',', '.'))
    except ValueError:
        return [token]

    x1      = token.get("x1", 0)
    x2      = token.get("x2", 0)
    split_x = x1 + int((x2 - x1) * 0.6)

    qty_token  = {**token, "token": qty_str,  "label": "QUANTITY",
                  "norm": qty_str.lower(),  "x2": split_x, "split_from": text}
    unit_token = {**token, "token": unit_str, "label": "UNIT",
                  "norm": unit_str.lower(), "x1": split_x, "split_from": text}
    return [qty_token, unit_token]


# ── Classifier ────────────────────────────────────────────────────────────────

class SemanticClassifier:
    """
    Zero-shot semantic classifier - v5.

    v5: NUTRIENT_LEXICON expanded. _is_nutrient() improved with:
      - Slash-variant splitting: 'Fett/fat' -> check 'fett' and 'fat' separately
      - Dash-prefix stripping: '- SATURATED FAT' -> check 'saturated fat'
      - Vitamin shorthand rule: 'B1', 'B12', 'C' -> NUTRIENT
    """

    def __init__(self, confidence_threshold: float = 0.30,
                 split_fused_tokens: bool = True):
        self.confidence_threshold = confidence_threshold
        self.split_fused_tokens   = split_fused_tokens

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"['\"`]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(".,!?;:-")

    def _normalise_context(self, raw_text: str) -> str:
        """Translate raw OCR context to canonical underscore form."""
        key = raw_text.lower().strip()
        canonical = CONTEXT_MAP.get(key)
        if canonical:
            return canonical
        key_clean = key.strip(".,;:*'\"")
        return CONTEXT_MAP.get(key_clean, key_clean)

    def _is_noise(self, norm: str) -> bool:
        if len(norm) <= 1:
            return True
        return any(s in norm for s in NOISE_SUBSTRINGS)

    def _is_nrv_pattern(self, norm: str) -> bool:
        return any(p.search(norm) for p in NRV_PATTERNS)

    def _is_serving_pattern(self, norm: str) -> bool:
        return any(p.match(norm) for p in SERVING_PATTERNS_RE)

    def _is_unit(self, norm: str) -> bool:
        return norm.strip(".,*()[]") in UNIT_LEXICON

    def _is_context(self, norm: str) -> bool:
        if any(ctx in norm for ctx in CONTEXT_LEXICON):
            return True
        if re.match(r'^(per|je|pro|fuer)\s+\d', norm):
            return True
        if re.match(r'^\d+\s*(tube|sachet|kapsel|tablette)', norm):
            return True
        return False

    def _is_quantity(self, norm: str) -> bool:
        clean = norm.strip("*'\"()[]. ")
        if not clean:
            return False
        if QUANTITY_PATTERN.match(clean):
            return True
        if re.match(r'^\d+[.,]?\d*\s*(mg|g|kg|µg|mcg|ug|kj|kcal|ml|%)$',
                    clean, re.IGNORECASE):
            return True
        return False

    def _is_nutrient(self, norm: str) -> bool:
        """
        Check if a normalised token is a nutrient name.

        v5 improvements:
          1. Dash-prefix strip: '- saturated fat' -> 'saturated fat'
          2. Slash-variant split: 'fett/fat' -> check 'fett' and 'fat'
          3. Vitamin shorthand: 'b1', 'b12', 'c' -> match via regex
          4. Standard lexicon exact + substring (unchanged from v4)
        """
        # 1. Strip leading dash/bullet prefix (e.g. '- davon zucker')
        stripped = re.sub(r'^[-–•]\s*', '', norm).strip()

        # 2. Check the slash-split variants first
        #    'kohlenhydrate/carbohydrate' -> check each part
        parts = [p.strip() for p in re.split(r'\s*/\s*', stripped) if p.strip()]
        if len(parts) > 1:
            for part in parts:
                if self._is_nutrient_single(part):
                    return True
            return False

        return self._is_nutrient_single(stripped)

    def _is_nutrient_single(self, norm: str) -> bool:
        """
        Check a single (non-slash) token against the NUTRIENT_LEXICON
        and vitamin shorthand rule.
        """
        # Exact match
        if norm in NUTRIENT_LEXICON:
            return True
        # Substring match (lexicon entry appears inside token)
        for nutrient in NUTRIENT_LEXICON:
            if len(nutrient) > 4 and nutrient in norm:
                return True
        # Starts with 'vitamin'
        if norm.startswith("vitamin"):
            return True
        # Vitamin shorthand: 'b1', 'b2', 'b12', 'c', 'd', 'e', 'k'
        if VITAMIN_SHORTHAND_RE.match(norm):
            return True
        return False

    def classify_token(self, token: dict) -> dict:
        """
        Classify one token.

        Priority order (v5 — unchanged from v4):
          1. NOISE     - irrelevant, short, NRV%, SERVING patterns
          2. UNIT      - exact lexicon match
          3. CONTEXT   - context patterns -> norm = canonical form
          4. QUANTITY  - numeric values
          5. NUTRIENT  - lexicon + slash-split + shorthand (v5)
          6. UNKNOWN   - fallback
        """
        result = token.copy()
        text   = token.get("token", "")
        conf   = token.get("conf", 0.0)
        norm   = self._normalize(text)

        if conf < self.confidence_threshold:
            result["label"] = "NOISE"
            result["norm"]  = norm
            return result

        if   self._is_noise(norm):           label = "NOISE"
        elif self._is_nrv_pattern(norm):     label = "NOISE"
        elif self._is_serving_pattern(norm): label = "NOISE"
        elif self._is_unit(norm):            label = "UNIT"
        elif self._is_context(norm):         label = "CONTEXT"
        elif self._is_quantity(norm):        label = "QUANTITY"
        elif self._is_nutrient(norm):        label = "NUTRIENT"
        else:                                label = "UNKNOWN"

        result["label"] = label

        if label == "CONTEXT":
            result["norm"] = self._normalise_context(text)
        else:
            result["norm"] = norm

        return result

    def classify_all(self, tokens: list) -> list:
        """Classify all tokens with optional fused-token splitting."""
        result = []
        for token in tokens:
            conf = token.get("conf", 0.0)
            if self.split_fused_tokens and conf >= self.confidence_threshold:
                parts = split_fused_token(token)
                if len(parts) > 1:
                    result.extend(parts)
                    continue
            result.append(self.classify_token(token))
        return result

    def summary(self, labeled_tokens: list) -> dict:
        """Print classification summary."""
        from collections import Counter
        counts = Counter(t["label"] for t in labeled_tokens)
        total  = len(labeled_tokens)
        print(f"\n{'='*50}")
        print("SEMANTIC CLASSIFICATION SUMMARY (v5)")
        print(f"{'='*50}")
        print(f"Total tokens: {total}")
        for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE", "UNKNOWN"]:
            count = counts.get(label, 0)
            pct   = count / total * 100 if total else 0
            print(f"  {label:<10} {count:>3} ({pct:>5.1f}%)  {'#'*int(pct/3)}")
        print(f"{'='*50}\n")
        ctx_tokens = [t for t in labeled_tokens if t["label"] == "CONTEXT"]
        if ctx_tokens:
            print("  CONTEXT tokens extracted (raw -> canonical):")
            for t in ctx_tokens:
                print(f"    '{t['token']}'  ->  '{t['norm']}'")
            print()
        return dict(counts)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ocr.ocr_runner import run_ocr_on_image
    from src.utils.ocr_corrector import OCRCorrector

    tokens    = run_ocr_on_image("data/raw/1.jpeg")
    corrected = OCRCorrector().correct_all(tokens)
    classifier = SemanticClassifier(0.30, split_fused_tokens=True)
    labeled    = classifier.classify_all(corrected)

    print(f"\n{'TOKEN':<45} {'LABEL':<12} {'NORM':<25} {'CONF'}")
    print("-"*95)
    for t in sorted(labeled, key=lambda x: x["y1"]):
        note = f"<- split from '{t['split_from']}'" if t.get("split_from") else ""
        print(f"{t['token']:<45} {t['label']:<12} {t['norm']:<25} {t['conf']:.3f}  {note}")
    classifier.summary(labeled)