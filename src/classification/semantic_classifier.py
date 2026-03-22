"""
semantic_classifier.py  v4
==========================
Stage 3 — Zero-Shot Semantic Token Classifier

CHANGE IN v4:
  CONTEXT_MAP moved from run_experiment.py (evaluator) into the classifier.

  Rationale:
    The classifier already identifies that a token IS a CONTEXT token.
    The next logical step — what canonical value it represents — belongs
    in the same place. Previously norm_context() lived in run_experiment.py
    as an evaluator patch added during Fix 1, before the graph construction
    was fixed. Now that CONTEXT tokens are correctly extracted via
    CONTEXT_SCOPE edges, canonical normalisation belongs at classification
    time, not at evaluation time.

  Effect:
    - tuples.csv now contains 'per_100g', 'per_serving', 'per_daily_dose'
      instead of raw OCR strings like 'Je 100g', 'PRO PORTION'.
    - Evaluator no longer needs norm_context() — plain string comparison.
    - Experiment 2/3 OCR engines get canonical context for free.
    - Single source of truth: CONTEXT_MAP lives here only.

CHANGE IN v3:
  Target output schema reduced to 4 fields:
    <nutrient, quantity, unit, context>

  NRV% and serving_size are no longer extraction targets.
  NRV patterns  -> classified as NOISE
  SERVING patterns -> classified as NOISE

  Active labels:
    NUTRIENT  - nutrient name
    QUANTITY  - numeric value
    UNIT      - measurement unit
    CONTEXT   - serving context (canonical: per_100g, per_serving, per_daily_dose)
    NOISE     - irrelevant (includes NRV and SERVING)
    UNKNOWN   - unmatched fallback
"""

import re


# ── Lexicons ──────────────────────────────────────────────────────────────────

NUTRIENT_LEXICON = {
    "energy", "fat", "saturates", "saturated", "carbohydrate", "carbohydrates",
    "sugars", "sugar", "fibre", "fiber", "protein", "salt",
    "energie", "fett", "fettsauren", "fettsäuren", "kohlenhydrate",
    "zuckerarten", "ballaststoffe", "eiweiss", "eiweiß", "salz",
    "graisses", "glucides", "dont sucres", "fibres", "sel",
    "vitamin", "vitamine", "thiamin", "thiamine", "riboflavin", "niacin",
    "pantothenic", "pantothensäure", "pyridoxin", "biotin", "folat",
    "folate", "folic", "cobalamin",
    "vitamin a", "vitamin b", "vitamin b1", "vitamin b2", "vitamin b3",
    "vitamin b5", "vitamin b6", "vitamin b7", "vitamin b9", "vitamin b12",
    "vitamin c", "vitamin d", "vitamin d3", "vitamin e", "vitamin k",
    "vitamin k1", "vitamin k2",
    "ascorbic", "ascorbinsäure", "tocopherol", "retinol", "cholecalciferol",
    "menaquinon", "phylloquinon",
    "calcium", "magnesium", "iron", "zinc", "iodine", "selenium",
    "copper", "manganese", "chromium", "molybdenum", "phosphorus",
    "potassium", "sodium", "chloride", "fluoride",
    "kalzium", "eisen", "zink", "jod", "jodid", "selen", "kupfer",
    "mangan", "chrom", "molybdän", "phosphor", "kalium", "natrium",
    "chlorid", "fluorid",
    "kalium /potassium", "natrium /sodium", "kalzium/calcium",
    "omega", "dha", "epa", "ala", "linolsäure", "linolensäure",
    "monounsaturated", "polyunsaturated", "mehrfach ungesattigte",
    "einfach ungesattigte",
    "davon gesattigte", "davon zucker",
    "leucin", "isoleucin", "valin", "lysin", "methionin", "phenylalanin",
    "threonin", "tryptophan", "histidin",
    "coenzym", "coenzyme", "l-carnitin", "carnitin", "kreatin", "creatine",
    "inositol", "cholin", "choline", "lutein", "zeaxanthin", "lycopin",
    "astaxanthin", "resveratrol", "quercetin", "rutin",
    "koffein", "caffeine",
}

UNIT_LEXICON = {
    "mg", "g", "kg", "mcg", "µg", "ug", "μg",
    "kj", "kcal", "cal",
    "ie", "iu", "µg re", "µg ne", "µg te",
    "mg/tag", "µg/tag", "g/tag",
    "mg/day", "µg/day", "g/day",
    "mg ne", "mg α-te",
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
# Moved from run_experiment.py evaluator into classifier (v4).
# Maps raw OCR context strings (lowercased) to canonical underscore form.
# Canonical values must match the ground truth annotation schema exactly:
#   per_100g | per_100ml | per_serving | per_daily_dose
#
# Add new variants here as they are discovered from analysis CSVs.

CONTEXT_MAP = {
    # per_100g variants
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
    # OCR corruption of "je 100g" — '0' misread as '9' at low resolution
    "je 1009":            "per_100g",
    "je1009":             "per_100g",

    # per_100ml variants
    "per 100ml":          "per_100ml",
    "je 100ml":           "per_100ml",
    "pro 100ml":          "per_100ml",
    "per 100 ml":         "per_100ml",

    # per_serving variants
    "per serving":        "per_serving",
    "per portion":        "per_serving",
    "pro portion":        "per_serving",
    "pro portionl":       "per_serving",
    "pro portionll":      "per_serving",
    "pRo portion":        "per_serving",
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

    # per_daily_dose variants
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
    "siehe siegelrand", "see seal", "sieh rueckseite",
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
    Zero-shot semantic classifier - v4.

    v4: CONTEXT tokens are normalised to canonical form at classification
    time via CONTEXT_MAP. The 'norm' field on a CONTEXT node contains
    'per_100g', 'per_serving', or 'per_daily_dose' — not the raw OCR string.
    The evaluator no longer needs norm_context() — plain string comparison.

    NRV and SERVING are classified as NOISE (v3 behaviour retained).
    Active targets: NUTRIENT, QUANTITY, UNIT, CONTEXT.
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
        """
        Translate raw OCR context text to canonical underscore form.

        Lookup is case-insensitive. If no map entry exists, the lowercased
        text is returned unchanged — novel variants are preserved for
        CONTEXT_MAP extension.

        Examples:
          'Je 100g'         -> 'per_100g'
          'PRO PORTION'     -> 'per_serving'
          'pRo portion'     -> 'per_serving'
          'Je 1009'         -> 'per_100g'  (OCR corruption of Je 100g)
          'pro 3 Tabletten' -> 'per_serving'
          'unknown ctx'     -> 'unknown ctx'  (passthrough)
        """
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
        if norm in NUTRIENT_LEXICON:
            return True
        for nutrient in NUTRIENT_LEXICON:
            if len(nutrient) > 4 and nutrient in norm:
                return True
        return norm.startswith("vitamin")

    def classify_token(self, token: dict) -> dict:
        """
        Classify one token.

        Priority order (v4):
          1. NOISE     - irrelevant, short, NRV%, SERVING patterns
          2. UNIT      - exact lexicon match
          3. CONTEXT   - context patterns -> norm set to canonical form via CONTEXT_MAP
          4. QUANTITY  - numeric values
          5. NUTRIENT  - nutrient lexicon
          6. UNKNOWN   - fallback

        v4 change: CONTEXT label -> norm = _normalise_context(raw_text)
                   All other labels -> norm = _normalize(raw_text)
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

        # v4: CONTEXT norm -> canonical form via CONTEXT_MAP
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
        """Print classification summary with label distribution and context mappings."""
        from collections import Counter
        counts = Counter(t["label"] for t in labeled_tokens)
        total  = len(labeled_tokens)
        print(f"\n{'='*50}")
        print("SEMANTIC CLASSIFICATION SUMMARY (v4)")
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