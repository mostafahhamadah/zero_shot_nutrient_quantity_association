"""
ocr_corrector.py
================
Stage 2.5 — OCR Post-Correction Layer

Applies two levels of correction to raw OCR tokens before
semantic classification:

Level 1 — Rule-based character correction:
  - Fix known OCR character confusions (0↔O, 1↔I, rn→m, etc.)
  - Strip leading/trailing OCR artifacts (' " | } ] [ )
  - Normalize numeric formats (1.195 → 1195, 40Omg → 400mg)
  - Normalize whitespace and punctuation

Level 2 — Lexicon-guided fuzzy snap:
  - If a token is within edit distance of a known nutrient → correct it
  - Uses a curated nutrient + unit lexicon
  - Snaps only when confidence is high enough to avoid false corrections

Zero-shot: no training data required.

Thesis positioning:
  This module is documented as "OCR Post-Correction Layer" and
  represents a pipeline contribution that improves downstream
  graph-based association by reducing token noise.

Usage:
    from ocr_corrector import OCRCorrector

    corrector = OCRCorrector()
    tokens = [...]  # from ocr_runner.py
    corrected = corrector.correct_all(tokens)
"""

import re
from difflib import SequenceMatcher


# ── Known OCR character confusion map ────────────────────────────────────────

# Applied at character level inside tokens
CHAR_CONFUSION_MAP = [
    # Common OCR misreads in nutrition context
    ("0", "O"),   # zero vs letter O — context-dependent
    ("1", "I"),   # one vs letter I — context-dependent
    ("rn", "m"),  # classic OCR split
    ("vv", "w"),  # double-v for w
    ("li", "li"), # already correct but normalizes font variants
]

# Direct token-level substitutions (after normalization)
# Format: (wrong_pattern, correct_replacement)
# These are applied as regex substitutions on the normalized token
TOKEN_CORRECTIONS = [
    # Magnesium variants
    (r'\bHagnesium\b', 'Magnesium'),
    (r'\bMagnesiunn\b', 'Magnesium'),
    (r'\bMagnes[il]um\b', 'Magnesium'),
    (r'\bMagnesiu[mn]\b', 'Magnesium'),

    # Kalium / Potassium
    (r'\bKalumciuat\b', 'Kaliumcitrat'),
    (r'\bKalumc[il]uat\b', 'Kaliumcitrat'),

    # Calcium variants
    (r'\bCalc[il]um\b', 'Calcium'),
    (r'\bKalz[il]um\b', 'Kalzium'),

    # Vitamin common corruptions
    (r'\bV[il]tam[il]n\b', 'Vitamin'),
    (r'\bV[il]tamin\b', 'Vitamin'),

    # Energie / Energy
    (r'\bEnerg[il]e[il]\b', 'Energie'),
    (r'\bEnerg[il]e\b', 'Energie'),

    # Eiweiß / Protein
    (r'\bE[il]we[il][sß][sß]\b', 'Eiweiß'),
    (r'\bE[il]we[il]ß\b', 'Eiweiß'),

    # Kohlenhydrate
    (r'\bKoh[il]enhydrate\b', 'Kohlenhydrate'),
    (r'\bKoh[il]enhydrat\b', 'Kohlenhydrate'),

    # Fettsäuren
    (r'\bFettsauren\b', 'Fettsäuren'),
    (r'\bFettsäure[mn]\b', 'Fettsäuren'),

    # Ballaststoffe
    (r'\bBa[il][il]aststoffe\b', 'Ballaststoffe'),

    # Salz / Salt
    (r'\bSa[il]z[Il|][sS]a[il]t\b', 'Salz/Salt'),
    (r'\bSa[il]z[Il]\b', 'Salz'),

    # Natrium / Sodium
    (r'\bNatr[il]um\b', 'Natrium'),

    # Phosphor
    (r'\bPhosphor\b', 'Phosphor'),
    (r'\bPhosph0r\b', 'Phosphor'),

    # Niacin
    (r'\bN[il]ac[il]n\b', 'Niacin'),

    # Biotin
    (r'\bB[il]ot[il]n\b', 'Biotin'),

    # Folsäure
    (r'\bFo[il]säure\b', 'Folsäure'),
    (r'\bFo[il]saure\b', 'Folsäure'),

    # Pantothensäure
    (r'\bPantothensaure\b', 'Pantothensäure'),

    # Numeric corruption patterns
    (r'\b4O(\d)', r'40\1'),   # 4O → 40 (letter O as zero)
    (r'\bO(\d{2,})', r'0\1'), # Leading O before digits → 0
    (r'(\d)O(\d)', r'\g<1>0\2'),  # digit-O-digit → digit-0-digit
    (r'(\d)l(\d)', r'\g<1>1\2'),  # digit-l-digit → digit-1-digit (lowercase L as 1)

    # Unit corruptions
    (r'\bmg\s*[|}\]]+', 'mg'),
    (r'\bg\s*[|}\]]+', 'g'),
    (r'\bk[Jj]\b', 'kJ'),
    (r'\bKca[il]\b', 'kcal'),
    (r'\bKCa[il]\b', 'kcal'),
    (r'\bUg\b', 'µg'),
    (r'\bug\b', 'µg'),
    (r'\bMcg\b', 'µg'),
    (r'\bmcg\b', 'µg'),

    # Artifact removal at word boundaries
    (r"^['\"'`]+", ''),   # leading quote artifacts
    (r"['\"'`]+$", ''),   # trailing quote artifacts
    (r'[|}\]]+$', ''),    # trailing bracket/pipe artifacts
    (r'^[|{\[]+', ''),    # leading bracket artifacts
    # Corrupted unit suffixes attached to quantities
    (r'(\d+)\s*m0\b', r'\1mg'),    # 400m0 → 400mg
    (r'(\d+)\s*M9\b', r'\1mg'),    # 420 M9 → 420mg  
    (r'(\d+)\s*m\b(?!l)', r'\1mg'), # 350m → 350mg (but not 350ml)
    (r'(\d+)\s*K[Jj]\b', r'\1kJ'), # 736KJ → 736kJ
    (r'(\d+)\s*Kcal\b', r'\1kcal'),# 200Kcal → 200kcal
]

# ── Nutrient lexicon for Level 2 fuzzy snap ──────────────────────────────────

SNAP_LEXICON = [
    # Macronutrients DE
    "Energie", "Fett", "Fettsäuren", "Kohlenhydrate", "Zucker",
    "Ballaststoffe", "Eiweiß", "Protein", "Salz",
    # Macronutrients EN
    "Energy", "Fat", "Carbohydrate", "Sugars", "Fibre", "Salt",
    "Saturates", "Protein",
    # Minerals DE
    "Magnesium", "Kalzium", "Calcium", "Eisen", "Zink", "Jod",
    "Selen", "Kupfer", "Mangan", "Chrom", "Phosphor", "Kalium",
    "Natrium", "Chlorid", "Fluorid", "Molybdän",
    # Minerals EN
    "Iron", "Zinc", "Iodine", "Selenium", "Copper", "Manganese",
    "Chromium", "Molybdenum", "Phosphorus", "Potassium", "Sodium",
    "Chloride", "Fluoride",
    # Vitamins
    "Vitamin A", "Vitamin B1", "Vitamin B2", "Vitamin B3",
    "Vitamin B5", "Vitamin B6", "Vitamin B7", "Vitamin B9",
    "Vitamin B12", "Vitamin C", "Vitamin D", "Vitamin D3",
    "Vitamin E", "Vitamin K", "Vitamin K1", "Vitamin K2",
    "Thiamin", "Riboflavin", "Niacin", "Pantothensäure",
    "Pantothenic Acid", "Biotin", "Folsäure", "Folic Acid",
    "Cobalamin", "Retinol", "Tocopherol", "Cholecalciferol",
    # Other common nutrients
    "Koffein", "Caffeine", "Kreatin", "Creatine",
    "Inositol", "Cholin", "Choline", "Lutein", "Lycopin",
    "Kaliumcitrat", "Magnesiumoxid", "Magnesiumcitrat",
    "Magnesiumbisglycinat", "Magnesiummalat",
    # Units
    "mg", "g", "kg", "µg", "kJ", "kcal", "ml", "IU", "IE",
]

# Words that must NEVER be snapped to nutrient names
# These are table headers, legal terms, or instruction words
SNAP_BLACKLIST = {
    "facts", "nutrition", "information", "reference", "values",
    "serving", "portion", "per", "daily", "intake", "value",
    "nährwert", "angaben", "bezugswert", "referenz",
    "hinweis", "hinweise", "achtung", "warnung",
    "mindestens", "haltbar", "batch", "lot", "charge",
    "ingredients", "zutaten", "enthält", "contains",
    "manufactured", "hergestellt", "vertrieb",
    "see", "siehe", "vide", "nota",
}

# Minimum length for Level 2 fuzzy snap to apply
MIN_SNAP_LENGTH = 5

# Similarity threshold for snap (0.0–1.0). Higher = more conservative.
SNAP_THRESHOLD = 0.75


# ── Helper: edit distance similarity ─────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio — fast approximate string similarity."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_snap(token: str, lexicon: list,
                   threshold: float = SNAP_THRESHOLD) -> tuple:
    """
    Find best lexicon match for a token using fuzzy similarity.

    Returns:
        (best_match, score) if score >= threshold, else (None, 0.0)
    """
    best_score = 0.0
    best_match = None
    token_lower = token.lower()

    for candidate in lexicon:
        score = similarity(token_lower, candidate.lower())
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0


# ── Corrector ─────────────────────────────────────────────────────────────────

class OCRCorrector:
    """
    Two-level OCR post-correction for supplement label tokens.

    Level 1: Rule-based character/token corrections
    Level 2: Lexicon-guided fuzzy snap for nutrient names
    """

    def __init__(self,
                 snap_threshold: float = SNAP_THRESHOLD,
                 snap_lexicon: list = None,
                 apply_level1: bool = True,
                 apply_level2: bool = True):
        self.snap_threshold = snap_threshold
        self.snap_lexicon = snap_lexicon or SNAP_LEXICON
        self.apply_level1 = apply_level1
        self.apply_level2 = apply_level2

    def level1_correct(self, text: str) -> str:
        """
        Apply rule-based corrections to a token string.

        Steps:
          1. Strip leading/trailing whitespace
          2. Apply TOKEN_CORRECTIONS regex substitutions
          3. Normalize internal whitespace
        """
        if not text:
            return text

        corrected = text.strip()

        for pattern, replacement in TOKEN_CORRECTIONS:
            try:
                corrected = re.sub(pattern, replacement, corrected,
                                   flags=re.IGNORECASE)
            except re.error:
                pass  # Skip malformed patterns

        # Normalize internal whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()

        return corrected

    def level2_snap(self, text: str, original_conf: float) -> tuple:
        """
        Apply lexicon-guided fuzzy snap to a token.

        Only applies if:
          - Token length >= MIN_SNAP_LENGTH
          - Token is not in SNAP_BLACKLIST
          - Token is not already in lexicon exactly
          - Best match score >= snap_threshold

        Returns:
            (corrected_text, was_snapped, snap_score)
        """
        if len(text) < MIN_SNAP_LENGTH:
            return text, False, 0.0

        # Never snap blacklisted words
        if text.lower().strip() in SNAP_BLACKLIST:
            return text, False, 0.0

        # Skip if already an exact match
        if any(text.lower() == lex.lower() for lex in self.snap_lexicon):
            return text, False, 1.0

        best_match, score = find_best_snap(text, self.snap_lexicon,
                                           self.snap_threshold)

        if best_match:
            return best_match, True, score

        return text, False, 0.0

    def correct_token(self, token: dict) -> dict:
        """
        Apply both correction levels to a single token dict.

        Args:
            token: dict with keys: token, x1, y1, x2, y2, conf

        Returns:
            corrected token dict with added keys:
              original_token: original text before correction
              l1_corrected: text after Level 1
              l2_snapped: whether Level 2 snap was applied
              snap_score: Level 2 similarity score
        """
        result = token.copy()
        original = token.get("token", "")
        conf = token.get("conf", 0.0)

        current = original

        # Level 1
        if self.apply_level1:
            current = self.level1_correct(current)

        l1_text = current

        # Level 2
        snapped = False
        snap_score = 0.0
        if self.apply_level2 and len(current) >= MIN_SNAP_LENGTH:
            snapped_text, snapped, snap_score = self.level2_snap(
                current, conf)
            if snapped:
                current = snapped_text

        result["token"] = current
        result["original_token"] = original
        result["l1_corrected"] = l1_text
        result["l2_snapped"] = snapped
        result["snap_score"] = round(snap_score, 4)

        return result

    def correct_all(self, tokens: list) -> list:
        """
        Apply correction to all tokens.

        Args:
            tokens: list of token dicts from ocr_runner.py

        Returns:
            list of corrected token dicts
        """
        return [self.correct_token(t) for t in tokens]

    def correction_report(self, corrected_tokens: list) -> None:
        """Print a report of all corrections made."""
        changed = [
            t for t in corrected_tokens
            if t.get("original_token") != t.get("token")
        ]
        snapped = [t for t in corrected_tokens if t.get("l2_snapped")]

        print(f"\n{'='*65}")
        print(f"OCR CORRECTION REPORT")
        print(f"{'='*65}")
        print(f"Total tokens:     {len(corrected_tokens)}")
        print(f"Tokens changed:   {len(changed)}")
        print(f"  Level 1 (rule): {len(changed) - len(snapped)}")
        print(f"  Level 2 (snap): {len(snapped)}")

        if changed:
            print(f"\nCorrections made:")
            print(f"  {'ORIGINAL':<35} → {'CORRECTED':<35} {'TYPE'}")
            print(f"  {'-'*80}")
            for t in changed:
                orig = t['original_token'][:33]
                corr = t['token'][:33]
                ctype = "SNAP" if t['l2_snapped'] else "RULE"
                score = f"({t['snap_score']:.2f})" if t['l2_snapped'] else ""
                print(f"  {orig:<35} → {corr:<35} {ctype} {score}")
        print(f"{'='*65}\n")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, ".")

    from src.ocr.ocr_runner import run_ocr_on_image
    from src.classification.semantic_classifier import SemanticClassifier
    from src.graph.graph_constructor import GraphConstructor
    from src.matching.association import TupleAssociator

    IMAGE = "data/raw/3.jpeg"
    IMAGE_ID = "3.jpeg"
    THRESHOLD = 0.30

    print(f"Running full corrected pipeline on {IMAGE}...\n")

    # Step 1: OCR
    tokens = run_ocr_on_image(IMAGE)

    # Step 2: Correct
    corrector = OCRCorrector()
    corrected = corrector.correct_all(tokens)
    corrector.correction_report(corrected)

    # Step 3: Classify
    classifier = SemanticClassifier(confidence_threshold=THRESHOLD)
    labeled = classifier.classify_all(corrected)

    # Step 4: Graph
    constructor = GraphConstructor()
    graph = constructor.build(labeled)

    # Step 5: Associate
    associator = TupleAssociator()
    tuples = associator.extract(graph, image_id=IMAGE_ID)
    associator.print_tuples(tuples)

    # Save
    out = "data/ocr_output/3_corrected_tuples.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tuples, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")