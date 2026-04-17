"""
semantic_classifier.py
======================
Stage 3 — Semantic Token Classification
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Assign a semantic label and canonical norm to every OCR token produced
by Stage 2 (corrector).  No tokens are split here — fused token splits
are Stage 2's responsibility (corrector rules C2/C3/C6).

INPUT / OUTPUT SCHEMA
---------------------
Input  : List[Dict]  — token, x1, y1, x2, y2, cx, cy, conf
Output : List[Dict]  — same fields + label, norm

LABELS
------
  NUTRIENT  — nutrient name (Magnesium, Kohlenhydrate, Vitamin B12 …)
  QUANTITY  — numeric value (400, 0.8, <0.1 …)
  UNIT      — measurement unit (mg, µg, kcal, g …)
  CONTEXT   — column/serving context (per_100g, per_serving, per_daily_dose)
  NOISE     — low-confidence, NRV%, serving descriptors, non-nutritional text
  UNKNOWN   — token not matched by any rule

PRIORITY CHAIN (highest → lowest)
----------------------------------
  1. conf < threshold           → NOISE
  2. norm in _SINGLE_CHAR_UNITS → UNIT   (g, l — before noise check)
  3. NRV percentage pattern     → NOISE
  4. Serving-size descriptor    → NOISE
  5. General noise substrings   → NOISE
  6. UNIT lexicon               → UNIT
  7. _resolve_context()         → CONTEXT  (single source of truth)
  8. QUANTITY pattern           → QUANTITY
  9. NUTRIENT lexicon/heuristic → NUTRIENT
 10. fallthrough                → UNKNOWN

CONFIDENCE THRESHOLD
--------------------
Stage 1 returns all tokens without filtering.  This stage is the single
place that decides whether a low-confidence token becomes NOISE.
Default threshold: 0.30.
"""

import re
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLE-CHARACTER UNIT WHITELIST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Checked BEFORE the noise rule.  Without this, bare 'g' tokens produced
# by the corrector's C2/C3 splits (e.g. '7,7g' → ['7.7', 'g']) hit
# len(norm) <= 1 → NOISE and are destroyed, collapsing Unit Acc across
# all German-format nutrition labels.
_SINGLE_CHAR_UNITS: frozenset = frozenset({'g', 'l'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIT LEXICON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIT_LEXICON: frozenset = frozenset({
    # Mass
    "mg", "g", "kg",
    "µg", "mcg", "ug", "μg",
    # Energy
    "kj", "kcal", "cal",
    "kj/kcal", "kj kcal", "kj/kca", "kcal/kj",
    # Volume
    "ml", "dl", "cl", "l",
    # International units
    "ie", "i.e.", "iu", "i.u.",
    # Microbiology
    "kbe", "cfu",
    # Compound nutrition units
    "mg ne", "mg α-te", "mg a-te",
    "µg re", "µg ne", "µg te",
    "mg/tag", "µg/tag", "g/tag",
    "mg/day", "µg/day", "g/day",
    # Uppercase variants (all-caps labels, images 113–121)
    "KJ", "KCAL", "MG", "G", "KG", "ML",
    "KJ/KCAL", "KJ/KCA",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NUTRIENT LEXICON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUTRIENT_LEXICON: frozenset = frozenset({

    # ── German ──────────────────────────────────────────────────────
    "energie", "brennwert",
    "fett", "fette",
    "fettsauren", "fettsäuren",
    "gesattigte fettsauren", "gesättigte fettsäuren",
    "davon gesattigte", "davon gesättigte",
    "davon gesattigte fettsauren", "davon gesättigte fettsäuren",
    "kohlenhydrate",
    "davon zucker", "davon zuckerarten",
    "davon mehrwertige alkohole", "mehrwertige alkohole",
    "ballaststoffe",
    "eiweiss", "eiweiß",
    "salz",
    "natrium", "kalium", "kalzium", "calcium",
    "magnesium", "phosphor", "chlorid", "chloride", "fluorid",
    "eisen", "zink", "jod", "jodid",
    "selen", "kupfer", "mangan",
    "chrom", "molybdän", "molybdan",
    "folsäure", "folsaure",
    "pantothensäure", "pantothensaure",
    "thiamin", "riboflavin",
    "niacin", "biotin",
    "koffein",
    "inositol", "cholin",
    "kreatin",
    "alpha-liponsäure", "alpha-liponsaure",
    "liponsäure", "liponsaure",
    "rutin",

    # ── English ─────────────────────────────────────────────────────
    "energy", "energy value",
    "fat", "fats",
    "saturated fat", "saturated fats", "saturates", "saturated",
    "of which saturates", "of which saturated",
    "carbohydrate", "carbohydrates",
    "sugars", "sugar", "of which sugars",
    "fibre", "fiber", "fibres", "dietary fibre", "dietary fiber",
    "protein", "proteins",
    "salt", "sodium", "potassium", "calcium", "magnesium",
    "iron", "zinc", "iodine", "selenium", "copper", "manganese",
    "chromium", "molybdenum", "phosphorus",
    "chloride", "fluoride",
    "folic acid",
    "pantothenic acid", "pantothenic",
    "thiamine", "riboflavin", "niacin", "biotin",
    "caffeine",
    "inositol", "choline",
    "creatine", "creatine monohydrate",

    # ── French ──────────────────────────────────────────────────────
    "énergie",
    "graisses", "matières grasses", "lipides",
    "dont acides gras saturés", "dont acides gras satures",
    "glucides", "dont sucres",
    "fibres",
    "protéines", "proteines",
    "sel",
    "valeur énergétique", "valeur energetique",

    # ── Dutch ────────────────────────────────────────────────────────
    "koolhydraten", "eiwitten", "vetten", "zout", "vezels",
    "verzadigde vetzuren",
    "waarvan verzadigde vetzuren", "waarvan suikers",

    # ── Italian ──────────────────────────────────────────────────────
    "grassi", "grassi saturi",
    "di cui acidi grassi saturi",
    "carboidrati", "di cui zuccheri",
    "proteine",
    "sale",
    "valore energetico",

    # ── Spanish ──────────────────────────────────────────────────────
    "grasas", "grasas saturadas",
    "de las cuales saturadas",
    "hidratos de carbono",
    "de los cuales azúcares", "de los cuales azucares",
    "proteínas", "proteinas",
    "sal",
    "valor energético", "valor energetico",

    # ── Compound energy forms ────────────────────────────────────────
    "energie kj", "energie kcal", "energie kj kcal",
    "brennwert kj", "brennwert kcal",
    "brennwert/energy value", "energie/energy",

    # ── Vitamins ─────────────────────────────────────────────────────
    "vitamin", "vitamine",
    "vitamin a", "vitamin b", "vitamin c", "vitamin d", "vitamin d3",
    "vitamin e", "vitamin k", "vitamin k1", "vitamin k2",
    "vitamin b1", "vitamin b2", "vitamin b3", "vitamin b5",
    "vitamin b6", "vitamin b7", "vitamin b9", "vitamin b12",
    "folat", "folate", "cobalamin",
    "ascorbic", "ascorbinsäure", "ascorbinsaure",
    "tocopherol", "retinol", "cholecalciferol", "colecalciferol",
    "menaquinon", "phylloquinon",
    "pyridoxin",

    # ── Fatty acids ──────────────────────────────────────────────────
    "omega", "dha", "epa", "ala",
    "linolsäure", "linolsaure", "linolensäure", "linolensaure",
    "monounsaturated", "polyunsaturated",

    # ── Amino acids ──────────────────────────────────────────────────
    "leucin", "isoleucin", "valin", "lysin",
    "methionin", "phenylalanin", "threonin", "tryptophan", "histidin",
    "l-valine", "l-valin", "l-valina",
    "l-leucine", "l-leucin", "l-leucina",
    "l-isoleucine", "l-isoleucin", "l-isoleucina",
    "l-lysine", "l-lysin", "l-lisina",
    "l-histidine", "l-histidin", "l-histidina", "l-istidina",
    "l-methionine", "l-methionin",
    "l-méthionine", "l-metionina", "l-metionin",     # multilingual
    "l-phenylalanine", "l-phenylalanin",
    "l-fenilalanina", "l-fenylalanin",                # IT/NL
    "l-threonine", "l-threonin", "l-thréonine",
    "l-treonina", "l-treonin",                        # ES/IT
    "l-tryptophan", "l-tryptofaan",                   # NL
    "l-triptófano", "l-triptofano", "l-tryptofan",   # ES/IT/CZ

    # ── Specialty compounds ──────────────────────────────────────────
    "coenzym", "coenzyme", "coq10",
    "l-carnitin", "carnitin",
    "lutein", "zeaxanthin", "lycopin",
    "astaxanthin", "resveratrol", "quercetin",

    # ── Probiotics ───────────────────────────────────────────────────
    "lactobacillus", "lactobacillus acidophilus",
    "acidophilus",
    "bifidobacterium", "bifidobacterium bifidum",
    "bifidum",
    "eiweib", "eweib",              # PaddleOCR corruption: EiweiB, EwebB, EsemB
    
    # ── Plant extracts / other ────────────────────────────────────────
    "green tea", "greentea", "green tea extract",
    "melissen extrakt", "melissenextrakt", "melissa extract",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTEXT LEXICON  (substring / prefix matching — fallback detector)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT_LEXICON: frozenset = frozenset({
    "per 100g", "per 100 g", "per 100ml", "per 100 ml",
    "je 100g", "je 100 g", "je 100ml", "je 100 ml",
    "pro 100g", "pro 100 g", "pro 100ml", "pro 100 ml",
    "per 100", "per100g", "per100ml",
    "je100g", "pro100g",
    "100g", "100 g", "100ml", "100 ml",
    "je 1009", "je1009",
    "per serving", "per portion", "pro portion",
    "per tube", "per sachet", "per capsule", "per tablet",
    "je kapsel", "je tablette", "je tagesdosis",
    "pro kapsel", "pro tablette", "pro tagesdosis",
    "per daily dose", "per tagesdosis",
    "pro 2 sticks", "pro 3 tabletten",
    "pr.100g", "pr0100g",
    "pour 100g", "pour 100 g",
    "por 100g", "por 100 g",
    "per 100g powder", "per 100 g powder",
    "per 25g",
    "per 10g", "je 10g", "pro 10g",
    "pro port.", "pro port",
    "proportion",
    "perdrink", "per drink",
    "per piece", "per bar",
    "tagesration", "tagesdosis",
    "per stück", "pro stück",
    "per daily", "daily diet",
    "shot", "shots",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTEXT MAP  (OCR variant → canonical form)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Keys: lowercase. Values: per_100g | per_100ml | per_serving | per_daily_dose
CONTEXT_MAP: dict = {

    # ── per_100g ────────────────────────────────────────────────────
    "per 100g":  "per_100g",    "per 100 g":  "per_100g",
    "per 100":   "per_100g",    "per100g":    "per_100g",
    "je 100g":   "per_100g",    "je 100 g":   "per_100g",
    "je100g":    "per_100g",    "pro 100g":   "per_100g",
    "pro 100 g": "per_100g",    "pro100g":    "per_100g",
    "100g":      "per_100g",    "100 g":      "per_100g",
    "je 1009":   "per_100g",    "je1009":     "per_100g",
    "1009":      "per_100g",
    "pr.100g":   "per_100g",    "pr0100g":    "per_100g",
    "pour 100g": "per_100g",    "pour 100 g": "per_100g",
    "por 100g":  "per_100g",    "por 100 g":  "per_100g",
    "por/na100g":"per_100g",    "pot/na100g": "per_100g",
    "per 100g powder":   "per_100g",
    "per 100 g powder":  "per_100g",
    "je100g1portion12g": "per_100g",
    "100g1tube70g":      "per_100g",

    # ── per_100ml ───────────────────────────────────────────────────
    "per 100ml": "per_100ml",   "per 100 ml": "per_100ml",
    "je 100ml":  "per_100ml",   "je 100 ml":  "per_100ml",
    "pro 100ml": "per_100ml",   "pro 100 ml": "per_100ml",
    "per100ml":  "per_100ml",   "100ml":      "per_100ml",

    # ── per_serving ─────────────────────────────────────────────────
    "per serving":     "per_serving",
    "per portion":     "per_serving",
    "pro portion":     "per_serving",
    "pro portionl":    "per_serving",
    "pro portionll":   "per_serving",
    "per sachet":      "per_serving",
    "per tube":        "per_serving",
    "per capsule":     "per_serving",
    "per tablet":      "per_serving",
    "je kapsel":       "per_serving",
    "je tablette":     "per_serving",
    "pro kapsel":      "per_serving",
    "pro tablette":    "per_serving",
    "per 1 tube":      "per_serving",
    "1tube":           "per_serving",
    "pro 2 sticks":    "per_serving",
    "per 25g":         "per_serving",
    "pro 2 tabletten": "per_serving",
    "je 700g":         "per_serving",
    "1portion":        "per_serving",
    "pro 1 portion":   "per_serving",
    "pro esslöffel":   "per_serving",
    "pro essloffel":   "per_serving",
    "per scoop":       "per_serving",
    "pro schuss":      "per_serving",
    "per stick":       "per_serving",
    "pro stick":       "per_serving",
    "per serve":       "per_serving",
    "per bar":         "per_serving",
    "portion**":       "per_serving",
    "pro port.":       "per_serving",
    "pro port":        "per_serving",
    "proportion":      "per_serving",
    "perdrink":        "per_serving",
    "per drink":       "per_serving",
    "per piece":       "per_serving",
    "piece":           "per_serving",
    "pro stück":       "per_serving",
    "pro stueck":      "per_serving",
    "per stück":       "per_serving",
    "per stueck":      "per_serving",
    "je stück":        "per_serving",
    "stück":           "per_serving",
    "stueck":          "per_serving",

    # ── per_daily_dose ──────────────────────────────────────────────
    "per daily dose":    "per_daily_dose",
    "per tagesdosis":    "per_daily_dose",
    "je tagesdosis":     "per_daily_dose",
    "pro tagesdosis":    "per_daily_dose",
    "per tagesration":   "per_daily_dose",
    "je tagesration":    "per_daily_dose",
    "1tablette":         "per_daily_dose",
    "1 tablette":        "per_daily_dose",
    "2 tabletten":       "per_daily_dose",
    "3 tabletten":       "per_daily_dose",
    "pro 3 tabletten":   "per_daily_dose",
    "pro 10g":           "per_daily_dose",
    "je 10g":            "per_daily_dose",
    "per 10g":           "per_daily_dose",
    "tagesration":       "per_daily_dose",
    "1 tagesration":     "per_daily_dose",
    "per daily":         "per_daily_dose",
    "per daily dose":    "per_daily_dose",
    "per daily diet":    "per_daily_dose",
    "daily diet":        "per_daily_dose",
    "shot":              "per_daily_dose",
    "shots":             "per_daily_dose",
    "75 g (300 ml)":  "per_serving",
    "75 g(300 ml)":   "per_serving",
    "75g (300ml)":    "per_serving",
    "75g(300ml)":     "per_serving",
    "75 g":           "per_serving",
    "75g":            "per_serving",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOISE SUBSTRINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOISE_SUBSTRINGS: tuple = (
    "mindestens", "best before", "haltbar bis", "lot nr", "lot-nr",
    "charge", "batch", "mhd",
    "hergestellt", "manufactured", "vertrieb", "distributor",
    "verantwortlich", "responsible", "kontakt", "contact",
    "gmbh", " ag ", " kg ", "ltd", "inc.", "s.a.",
    "iso ", "haccp", "bio-siegel", "organic certified",
    "kuehl lagern", "trocken lagern", "store in", "keep away",
    "ausserhalb der reichweite",
    "nicht fuer kinder", "not for children",
    "verschlossen halten", "keep closed",
    "nach oeffnung", "after opening",
    "nutrition facts", "naehrwertangaben",
    "naehrwerttabelle", "nutritional information",
    "zutaten", "ingredients",
    "nrv", "nutrient reference", "naehrstoffbezugswert",
    "referenzwert", "empfohlene tagesdosis", "% der empfohlenen",
    "bezugswert",
    "daily value", "daily values", "percent daily",
    "based on a 2",
    "empfehlung", "verzehrempfehlung",
    "enthält", "contains", "zubereitung", "preparation",
    "anwendung", "dosierung", "dosage",
    "tabletten täglich", "kapseln täglich",
    "protein-riegel",
    "risiko einer", "entsprechend",
    "tragt bei zur", "beitragen zur",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPILED PATTERNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VITAMIN_SHORTHAND_RE = re.compile(
    r"^(b\d{1,2}|vitamin\s+[a-z]\d*|[cdek]\d?)$",
    re.IGNORECASE,
)

_NRV_RE = re.compile(r"^\*?\s*\d+[\s,.]?\d*\s*%\s*\*?$")

_SERVING_RE = re.compile(
    r"^\d+\s*(kapseln?|tabletten?|softgels?|tubes?|sachets?|"
    r"capsules?|tablets?|gummies?|drops?|stueck)$",
    re.IGNORECASE,
)

_QUANTITY_RE = re.compile(
    r"^[<>]?\s*[\*\'\"]?(\d+[.,]?\d*(?:[xX×]\s*10\d*)?)\s*"
    r"(mg|g|kg|µg|mcg|ug|kj|kcal|cal|ie|iu|%|ml|l)?[\*\'\"\s]*$",
    re.IGNORECASE,
)

_UNIT_SUFFIX_RE = re.compile(
    r"\s*(kj|kcal|cal|mg|g|kg|µg|mcg|ug|μg|ml|l|ie|iu|ne|re|%)[/\s].*$",
    re.IGNORECASE,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SemanticClassifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SemanticClassifier:
    """
    Zero-shot semantic token classifier.

    Parameters
    ----------
    confidence_threshold : float
        Tokens with OCR confidence below this value are labelled NOISE.
        Stage 1 returns all tokens; this is the single place the threshold
        is applied.  Default: 0.30.
    """

    def __init__(self, confidence_threshold: float = 0.30):
        self.confidence_threshold = confidence_threshold

    # ── Normalisation ────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """
        Lowercase + light cleanup.
        Removes parenthetical qualifiers (e), (ne), (a-te) and strips
        trailing colons/punctuation so 'FETT:' and 'FETT' match equally.
        """
        s = text.lower().strip()
        s = re.sub(r"['\"`]", "", s)
        s = re.sub(r"\([a-zα-ω0-9\-]+\)", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip(".,!?;:- ")

    # ── Context — single source of truth ────────────────────────────

    def _resolve_context(self, norm: str, raw_text: str) -> Optional[str]:
        """
        If this token is a context token, return its canonical form.
        Returns None if the token is not a context token.

        Replaces the original _is_context() + _normalise_context() pair.
        Having one function eliminates the duplicated regex patterns and
        ensures that detection and canonicalisation are always consistent.

        Lookup order:
          1. Exact CONTEXT_MAP lookup (covers the vast majority of cases)
          2. Strip punctuation + exact CONTEXT_MAP lookup
          3. CONTEXT_LEXICON substring match (safety net for OCR noise)
          4. Regex patterns for dynamic forms not in the static maps
        """
        key = raw_text.lower().strip()

        # Pass 1: exact match
        canonical = CONTEXT_MAP.get(key)
        if canonical:
            return canonical

        # Pass 2: strip punctuation + exact match
        key_clean = key.strip(".,;:*'\"()")
        canonical = CONTEXT_MAP.get(key_clean)
        if canonical:
            return canonical

        # Pass 3: substring membership in CONTEXT_LEXICON
        # Catches tokens that contain a known context string but are not
        # exactly in CONTEXT_MAP (e.g. "Typical values per piece").
        # Extract the MATCHING substring and look it up in CONTEXT_MAP.
        for ctx in CONTEXT_LEXICON:
            if ctx in norm:
                canonical = CONTEXT_MAP.get(ctx)
                if canonical:
                    return canonical
                return CONTEXT_MAP.get(norm) or norm

        # Pass 4: regex patterns for variable forms

        # "1tablette (4.3g)", "2 kapseln", "1 ampulle", "2 shots", "1 tube(70g)" → per_daily_dose
        if re.match(
            r"^\d+\s*(tablette|tabletten|kapsel|kapseln|softgel|softgels|"
            r"ampulle|ampullen|flasche|flaschen|flask|flasks|"
            r"stick|sticks|beutel|sachet|sachets|"
            r"tropfen|drops|dragee|dragees|"
            r"tagesration|tagesdosis|tagesration/daily|"
            r"shot|shots|tube|tubes|riegel|"
            r"ampulle/flask|caps\.?)",
            key_clean, re.IGNORECASE,
        ):
            return "per_daily_dose"

        # Embedded dosage form after prefix noise: "ration 1ampulle", "xyz 2kapseln"
        if re.search(
            r"\d+\s*(ampulle|ampullen|kapsel|kapseln|tablette|tabletten|"
            r"flask|caps\.?|stick|sticks)",
            key_clean, re.IGNORECASE,
        ):
            return "per_daily_dose"

        # "per piece", "pro stück" → per_serving
        if re.match(
            r"^(pro|per|je)\s+(stück|stueck|piece|riegel|bar)",
            key_clean, re.IGNORECASE,
        ):
            return "per_serving"

        # Embedded "per piece", "per bar" etc. inside longer tokens
        if re.search(
            r"(per|pro|je)\s+(piece|stück|stueck|bar|riegel|serve|serving|portion)",
            key_clean, re.IGNORECASE,
        ):
            return "per_serving"

        # "pro esslöffel", "per scoop" → per_serving
        if re.match(
            r"^(pro|per|je)\s+(esslöffel|essloffel|scoop|schuss|löffel)",
            key_clean, re.IGNORECASE,
        ):
            return "per_serving"

        # "per/je/pro/pour/por + number" → variable per-N-g context
        if re.match(r"^(per|je|pro|fuer|pour|por|for)\s+\d", norm):
            return CONTEXT_MAP.get(norm) or norm

        # "pr.100g", "pr0100g" → per_100g
        if re.match(r"^pr[o0.]?\s*100", norm):
            return "per_100ml" if "ml" in norm else "per_100g"

        # "pr.4g", "pr.25g" → per_serving (non-100g amounts)
        if re.match(r"^pr[o0.]?\s*\d+\s*(g|ml)\b", norm):
            return "per_serving"

        # "pot/na100g", "por/na100g" → per_100g
        if re.match(r"^p[oa][rt]/na", norm):
            return "per_100g"

        # "pour 100g", "por 100g" → per_100g
        if re.match(r"^(pour|por)\s+\d", norm):
            return "per_100g"

        # "per 100g powder", "je 100ml" etc. with optional suffix
        if re.match(
            r"^(per|je|pro)\s+100\s*(g|ml)\s*(powder|pulver)?$",
            norm, re.IGNORECASE,
        ):
            return "per_100ml" if "ml" in norm else "per_100g"

        return None  # not a context token

    # ── Predicates ──────────────────────────────────────────────────

    def _is_noise(self, norm: str) -> bool:
        """
        True for tokens with no nutritional content.
        NOTE: _SINGLE_CHAR_UNITS is checked in classify_token() before
        this method — bare 'g' tokens must never reach this check.
        Single digits (0-9) are exempt — they are valid quantities
        (e.g. Calcium 9 mg, Green Tea 1 mg).
        """
        if len(norm) <= 1:
            if norm.isdigit():
                return False   # single digits are valid quantities
            return True
        return any(s in norm for s in NOISE_SUBSTRINGS)

    def _is_nrv_pattern(self, norm: str) -> bool:
        return bool(_NRV_RE.match(norm))

    def _is_serving_pattern(self, norm: str) -> bool:
        return bool(_SERVING_RE.match(norm))

    def _is_unit(self, norm: str) -> bool:
        candidate = norm.strip(".,*()[] ")
        return candidate in UNIT_LEXICON or candidate.lower() in UNIT_LEXICON

    def _is_quantity(self, norm: str) -> bool:
        clean = norm.strip("*'\"()[]. ")
        return bool(clean and _QUANTITY_RE.match(clean))

    def _strip_unit_suffix(self, norm: str) -> str:
        """Strip trailing unit from compound names: 'energie kj' → 'energie'."""
        return _UNIT_SUFFIX_RE.sub("", norm).strip(".,!?;:- ").strip()

    def _is_nutrient_single(self, norm: str) -> bool:
        if norm in NUTRIENT_LEXICON:
            return True
        for entry in NUTRIENT_LEXICON:
            if len(entry) > 4 and entry in norm:
                return True
        if norm.startswith("vitamin"):
            return True
        if VITAMIN_SHORTHAND_RE.match(norm):
            return True
        return False

    def _is_nutrient(self, norm: str) -> bool:
        """
        Full NUTRIENT check:
          1. Strip leading dash/bullet
          2. Build unit-stripped variant
          3. Slash-variant split (Fett/fat → ['fett', 'fat'])
          4. Vitamin shorthand regex
        """
        stripped    = re.sub(r"^[-–•]\s*", "", norm).strip()
        stripped_nu = self._strip_unit_suffix(stripped)
        candidates  = list(dict.fromkeys([stripped, stripped_nu]))

        for candidate in candidates:
            if not candidate:
                continue
            parts = [p.strip() for p in re.split(r"\s*/\s*", candidate) if p.strip()]
            if len(parts) > 1:
                for part in parts:
                    for check in dict.fromkeys([part, self._strip_unit_suffix(part)]):
                        if check and self._is_nutrient_single(check):
                            return True
            else:
                if self._is_nutrient_single(candidate):
                    return True
        return False

    # ── Classification ───────────────────────────────────────────────

    def classify_token(self, token: dict) -> dict:
        """
        Classify a single token dict and return it with label and norm set.

        Priority chain — first match wins:
          1. conf < threshold           → NOISE
          2. norm in _SINGLE_CHAR_UNITS → UNIT
          3. NRV percentage             → NOISE
          4. Serving-size descriptor    → NOISE
          5. Noise substrings / len<=1  → NOISE
          6. UNIT lexicon               → UNIT
          7. _resolve_context()         → CONTEXT
          8. QUANTITY pattern           → QUANTITY
          9. NUTRIENT lexicon           → NUTRIENT
         10. fallthrough                → UNKNOWN
        """
        result = token.copy()
        text   = token.get("token", "")
        conf   = token.get("conf",  0.0)
        norm   = self._normalize(text)

        # 1. Low confidence
        if conf < self.confidence_threshold:
            result.update(label="NOISE", norm=norm)
            return result

        # 2. Single-char unit whitelist (before noise — see module docstring)
        if norm in _SINGLE_CHAR_UNITS:
            result.update(label="UNIT", norm=norm)
            return result

        # 3. NRV percentage
        if self._is_nrv_pattern(norm):
            result.update(label="NOISE", norm=norm)
            return result

        # 4. CONTEXT — MUST run before serving/noise checks!
        #    Tokens like "1 tablette", "2 kapseln", "1 Ampulle" are
        #    valid CONTEXT tokens (per_daily_dose) but also match
        #    _SERVING_RE which would label them NOISE.
        canonical = self._resolve_context(norm, text)
        if canonical is not None:
            result.update(label="CONTEXT", norm=canonical)
            return result

        # 5. Serving-size descriptor (only reaches here if NOT a context)
        if self._is_serving_pattern(norm):
            result.update(label="NOISE", norm=norm)
            return result

        # 6. General noise
        if self._is_noise(norm):
            result.update(label="NOISE", norm=norm)
            return result

        # 7. UNIT lexicon
        if self._is_unit(norm):
            result.update(label="UNIT", norm=norm.strip(".,*()[] ").lower())
            return result

        # 8. QUANTITY
        if self._is_quantity(norm):
            result.update(label="QUANTITY", norm=norm)
            return result

        # 9. NUTRIENT
        if self._is_nutrient(norm):
            result.update(label="NUTRIENT", norm=norm)
            return result

        # 10. UNKNOWN
        result.update(label="UNKNOWN", norm=norm)
        return result

    def classify_all(self, tokens: list) -> list:
        """Classify a list of token dicts. Returns the list with label and norm added."""
        return [self.classify_token(t) for t in tokens]

    def summary(self, labeled_tokens: list) -> dict:
        """Print and return a label-count summary."""
        from collections import Counter
        counts = Counter(t.get("label", "UNKNOWN") for t in labeled_tokens)
        total  = len(labeled_tokens)
        print(f"\n{'='*50}")
        print("  SEMANTIC CLASSIFICATION SUMMARY")
        print(f"{'='*50}")
        print(f"  Total tokens : {total}")
        for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE", "UNKNOWN"]:
            n   = counts.get(label, 0)
            pct = n / total * 100 if total else 0
            print(f"  {label:<10}  {n:>4}  ({pct:>5.1f}%)")
        print(f"{'='*50}\n")
        return dict(counts)