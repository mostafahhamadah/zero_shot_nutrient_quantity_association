"""
ocr_corrector.py
================
Stage 2.5 — OCR Post-Correction Layer

Level 1 — Rule-based character correction:
  - Fix known OCR character confusions (0↔O, 1↔I, rn→m, etc.)
  - Strip leading/trailing OCR artifacts (' " | } ] [ )
  - Normalize numeric formats (1.195 → 1195, 40Omg → 400mg)
  - Normalize whitespace and punctuation

Level 1b — Token-pair merge pass (NEW):
  - Merges split micro-sign tokens: "µ" + "g" → "µg"
  - Merges suffix-split tokens: "40µ" + "g" → "40µg"
  - Runs after individual Level 1 corrections, before Level 2

Level 2 — Lexicon-guided fuzzy snap:
  - If a token is within edit distance of a known nutrient → correct it
  - Uses a curated nutrient + unit lexicon
  - Snaps only when confidence is high enough to avoid false corrections
"""

import re
from difflib import SequenceMatcher


# ── Known OCR character confusion map ─────────────────────────────────────────

CHAR_CONFUSION_MAP = [
    ("0", "O"),
    ("1", "I"),
    ("rn", "m"),
    ("vv", "w"),
    ("li", "li"),
]

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
    (r'\b4O(\d)', r'40\1'),
    (r'\bO(\d{2,})', r'0\1'),
    (r'(\d)O(\d)', r'\g<1>0\2'),
    (r'(\d)l(\d)', r'\g<1>1\2'),

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

    # Artifact removal
    (r"^['\"'`]+", ''),
    (r"['\"'`]+$", ''),
    (r'[|}\]]+$', ''),
    (r'^[|{\[]+', ''),

    # Corrupted unit suffixes attached to quantities
    (r'(\d+)\s*m0\b', r'\1mg'),
    (r'(\d+)\s*M9\b', r'\1mg'),
    (r'(\d+)\s*m\b(?!l)', r'\1mg'),
    (r'(\d+)\s*K[Jj]\b', r'\1kJ'),
    (r'(\d+)\s*Kcal\b', r'\1kcal'),

    # NEW v2 — micro-sign attached to number without 'g':
    # "40µ" alone (no following g token) → "40µg"
    (r'^(\d+[.,]?\d*)\s*µ$', r'\1µg'),
    (r'^(\d+[.,]?\d*)\s*μ$', r'\1µg'),
]

# ── Nutrient lexicon for Level 2 fuzzy snap ───────────────────────────────────

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
    # Other
    "Koffein", "Caffeine", "Kreatin", "Creatine",
    "Inositol", "Cholin", "Choline", "Lutein", "Lycopin",
    "Kaliumcitrat", "Magnesiumoxid", "Magnesiumcitrat",
    "Magnesiumbisglycinat", "Magnesiummalat",
    # Units
    "mg", "g", "kg", "µg", "kJ", "kcal", "ml", "IU", "IE",
]

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

MIN_SNAP_LENGTH  = 5
SNAP_THRESHOLD   = 0.75

# ── Micro-sign token that PaddleOCR splits off separately ─────────────────────
# PaddleOCR splits "40µg" → ["40µ", "g"] or ["40", "µ", "g"] or ["40", "µg"]
# _MICRO_RE matches a token that IS the micro sign (alone or with digit-suffix)
_MICRO_ALONE_RE  = re.compile(r'^[µμ]$')
_MICRO_SUFFIX_RE = re.compile(r'^(\d+[.,]?\d*)\s*[µμ]$')   # "40µ"


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_snap(token: str, lexicon: list,
                   threshold: float = SNAP_THRESHOLD) -> tuple:
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


class OCRCorrector:
    """
    Two-level OCR post-correction for supplement label tokens.

    v2 additions:
      Level 1b — token-pair merge pass:
        Fixes PaddleOCR micro-sign split:
          "40µ" + "g"  → "40µg"   (suffix form)
          "40"  + "µ"  + "g" → "40µg"  (three-token form)
          "µ"   + "g"  → "µg"     (bare micro + g)
        This runs as a post-pass after individual Level 1 corrections
        and before Level 2 snap, operating on the full token list.
    """

    def __init__(self,
                 snap_threshold: float = SNAP_THRESHOLD,
                 snap_lexicon: list = None,
                 apply_level1: bool = True,
                 apply_level2: bool = True):
        self.snap_threshold = snap_threshold
        self.snap_lexicon   = snap_lexicon or SNAP_LEXICON
        self.apply_level1   = apply_level1
        self.apply_level2   = apply_level2

    # ── Level 1 (individual token) ──────────────────────────────────────────

    def level1_correct(self, text: str) -> str:
        if not text:
            return text
        corrected = text.strip()
        for pattern, replacement in TOKEN_CORRECTIONS:
            try:
                corrected = re.sub(pattern, replacement, corrected,
                                   flags=re.IGNORECASE)
            except re.error:
                pass
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        return corrected

    # ── Level 1b — token-pair merge pass ───────────────────────────────────

    def level1b_merge_pairs(self, tokens: list) -> list:
        """
        Merge split micro-sign tokens across the full token list.

        PaddleOCR splits "40µg" in three ways:
          Form A: ["40µ",  "g"]          → token[i] ends "µ", token[i+1]="g"
          Form B: ["40",   "µ",  "g"]    → token[i]= number, [i+1]="µ", [i+2]="g"
          Form C: ["µ",    "g"]          → standalone micro + g

        Rule: if the merged token would be spatially adjacent (within 30px
        horizontally) AND on the same row (cy within 15px), merge.

        The merged token inherits:
          - token text of first + second (+ third if Form B)
          - bbox: x1 of first, x2 of last, y1/y2/cy/cx averaged
          - conf: min of merged tokens
        """
        if not tokens:
            return tokens

        result = []
        skip = set()

        for i, tok in enumerate(tokens):
            if i in skip:
                continue

            txt_i = tok.get("token", "").strip()

            # ── Form B: number  +  "µ"  +  "g" ──────────────────────────
            if (i + 2 < len(tokens)
                    and i + 1 not in skip
                    and i + 2 not in skip):
                txt_j = tokens[i+1].get("token", "").strip()
                txt_k = tokens[i+2].get("token", "").strip()
                # number + bare micro + "g"
                if (re.match(r'^\d+[.,]?\d*$', txt_i)
                        and _MICRO_ALONE_RE.match(txt_j)
                        and txt_k.lower() == "g"
                        and self._adjacent(tok, tokens[i+2])):
                    merged = self._merge3(tok, tokens[i+1], tokens[i+2],
                                          txt_i + "µg")
                    result.append(merged)
                    skip.update({i+1, i+2})
                    continue

            # ── Form A: "40µ"  +  "g" ────────────────────────────────────
            if (i + 1 < len(tokens)
                    and i + 1 not in skip):
                txt_j = tokens[i+1].get("token", "").strip()
                m = _MICRO_SUFFIX_RE.match(txt_i)
                if (m and txt_j.lower() == "g"
                        and self._adjacent(tok, tokens[i+1])):
                    merged_text = m.group(1) + "µg"
                    merged = self._merge2(tok, tokens[i+1], merged_text)
                    result.append(merged)
                    skip.add(i + 1)
                    continue

            # ── Form C: bare "µ"  +  "g" ─────────────────────────────────
            if (i + 1 < len(tokens)
                    and i + 1 not in skip):
                txt_j = tokens[i+1].get("token", "").strip()
                if (_MICRO_ALONE_RE.match(txt_i)
                        and txt_j.lower() == "g"
                        and self._adjacent(tok, tokens[i+1])):
                    merged = self._merge2(tok, tokens[i+1], "µg")
                    result.append(merged)
                    skip.add(i + 1)
                    continue

            # No merge — pass through
            result.append(tok)

        return result

    @staticmethod
    def _adjacent(t1: dict, t2: dict, max_gap_px: int = 30,
                  max_row_diff_px: int = 15) -> bool:
        """True if t1 and t2 are spatially close enough to merge."""
        gap     = t2.get("x1", 0) - t1.get("x2", 0)
        row_diff = abs(t1.get("cy", 0) - t2.get("cy", 0))
        return gap <= max_gap_px and row_diff <= max_row_diff_px

    @staticmethod
    def _merge2(t1: dict, t2: dict, new_text: str) -> dict:
        merged = t1.copy()
        merged["token"]    = new_text
        merged["x2"]       = t2.get("x2", t1.get("x2", 0))
        merged["cx"]       = (t1.get("x1", 0) + t2.get("x2", 0)) / 2
        merged["conf"]     = min(t1.get("conf", 1.0), t2.get("conf", 1.0))
        merged["merged_from"] = [t1.get("token", ""), t2.get("token", "")]
        return merged

    @staticmethod
    def _merge3(t1: dict, t2: dict, t3: dict, new_text: str) -> dict:
        merged = t1.copy()
        merged["token"]    = new_text
        merged["x2"]       = t3.get("x2", t1.get("x2", 0))
        merged["cx"]       = (t1.get("x1", 0) + t3.get("x2", 0)) / 2
        merged["conf"]     = min(t1.get("conf", 1.0), t2.get("conf", 1.0),
                                 t3.get("conf", 1.0))
        merged["merged_from"] = [t1.get("token", ""), t2.get("token", ""),
                                  t3.get("token", "")]
        return merged

    # ── Level 2 ────────────────────────────────────────────────────────────

    def level2_snap(self, text: str, original_conf: float) -> tuple:
        if len(text) < MIN_SNAP_LENGTH:
            return text, False, 0.0
        if text.lower().strip() in SNAP_BLACKLIST:
            return text, False, 0.0
        if any(text.lower() == lex.lower() for lex in self.snap_lexicon):
            return text, False, 1.0
        best_match, score = find_best_snap(text, self.snap_lexicon,
                                           self.snap_threshold)
        if best_match:
            return best_match, True, score
        return text, False, 0.0

    # ── Per-token correction ────────────────────────────────────────────────

    def correct_token(self, token: dict) -> dict:
        result   = token.copy()
        original = token.get("token", "")
        conf     = token.get("conf", 0.0)
        current  = original

        if self.apply_level1:
            current = self.level1_correct(current)
        l1_text = current

        snapped    = False
        snap_score = 0.0
        if self.apply_level2 and len(current) >= MIN_SNAP_LENGTH:
            snapped_text, snapped, snap_score = self.level2_snap(current, conf)
            if snapped:
                current = snapped_text

        result["token"]          = current
        result["original_token"] = original
        result["l1_corrected"]   = l1_text
        result["l2_snapped"]     = snapped
        result["snap_score"]     = round(snap_score, 4)
        return result

    # ── Full pipeline ───────────────────────────────────────────────────────

    def correct_all(self, tokens: list) -> list:
        """
        Apply all correction levels to the full token list.

        Order:
          1. Level 1 applied per-token (rule-based char corrections)
          2. Level 1b applied to the full list (merge split µg tokens)
          3. Level 2 applied per-token (fuzzy lexicon snap)
        """
        # Step 1: Level 1 per-token
        after_l1 = []
        for tok in tokens:
            result   = tok.copy()
            original = tok.get("token", "")
            current  = original
            if self.apply_level1:
                current = self.level1_correct(current)
            result["token"]          = current
            result["original_token"] = original
            result["l1_corrected"]   = current
            result["l2_snapped"]     = False
            result["snap_score"]     = 0.0
            after_l1.append(result)

        # Step 2: Level 1b — merge split µg tokens across list
        after_l1b = self.level1b_merge_pairs(after_l1)

        # Step 3: Level 2 per-token (fuzzy snap)
        final = []
        for tok in after_l1b:
            result  = tok.copy()
            current = tok.get("token", "")
            conf    = tok.get("conf", 0.0)
            snapped    = False
            snap_score = 0.0
            if self.apply_level2 and len(current) >= MIN_SNAP_LENGTH:
                snapped_text, snapped, snap_score = self.level2_snap(current, conf)
                if snapped:
                    current = snapped_text
            result["token"]      = current
            result["l2_snapped"] = snapped
            result["snap_score"] = round(snap_score, 4)
            final.append(result)

        return final

    # ── Report ──────────────────────────────────────────────────────────────

    def correction_report(self, corrected_tokens: list) -> None:
        changed = [t for t in corrected_tokens
                   if t.get("original_token") != t.get("token")]
        snapped = [t for t in corrected_tokens if t.get("l2_snapped")]
        merged  = [t for t in corrected_tokens if t.get("merged_from")]

        print(f"\n{'='*65}")
        print(f"OCR CORRECTION REPORT  (v2 — with µg merge)")
        print(f"{'='*65}")
        print(f"Total tokens:     {len(corrected_tokens)}")
        print(f"Tokens changed:   {len(changed)}")
        print(f"  Level 1 (rule): {len(changed) - len(snapped)}")
        print(f"  Level 1b (µg):  {len(merged)}")
        print(f"  Level 2 (snap): {len(snapped)}")

        if merged:
            print(f"\nMerged µg tokens:")
            for t in merged:
                src = " + ".join(f'"{x}"' for x in t["merged_from"])
                print(f"  {src}  →  \"{t['token']}\"")

        if changed:
            print(f"\nAll corrections:")
            print(f"  {'ORIGINAL':<35} → {'CORRECTED':<35} TYPE")
            print(f"  {'-'*80}")
            for t in changed:
                orig  = t['original_token'][:33]
                corr  = t['token'][:33]
                ctype = "SNAP" if t['l2_snapped'] else ("MERGE" if t.get("merged_from") else "RULE")
                score = f"({t['snap_score']:.2f})" if t['l2_snapped'] else ""
                print(f"  {orig:<35} → {corr:<35} {ctype} {score}")
        print(f"{'='*65}\n")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick unit test for the µg merge logic
    corrector = OCRCorrector()

    test_cases = [
        # Form A: number + "µ" as suffix
        [{"token": "40µ",  "x1": 0, "x2": 20, "y1": 0, "y2": 10, "cx": 10, "cy": 5, "conf": 0.9},
         {"token": "g",    "x1": 22,"x2": 28, "y1": 0, "y2": 10, "cx": 25, "cy": 5, "conf": 0.9}],
        # Form B: number + bare "µ" + "g"
        [{"token": "22",   "x1": 0, "x2": 15, "y1": 0, "y2": 10, "cx": 7,  "cy": 5, "conf": 0.9},
         {"token": "µ",    "x1": 16,"x2": 20, "y1": 0, "y2": 10, "cx": 18, "cy": 5, "conf": 0.9},
         {"token": "g",    "x1": 21,"x2": 26, "y1": 0, "y2": 10, "cx": 23, "cy": 5, "conf": 0.9}],
        # Form C: bare µ + g
        [{"token": "µ",    "x1": 0, "x2": 6,  "y1": 0, "y2": 10, "cx": 3,  "cy": 5, "conf": 0.9},
         {"token": "g",    "x1": 7, "x2": 12, "y1": 0, "y2": 10, "cx": 9,  "cy": 5, "conf": 0.9}],
        # Already correct — should not change
        [{"token": "50µg", "x1": 0, "x2": 25, "y1": 0, "y2": 10, "cx": 12, "cy": 5, "conf": 0.9}],
    ]

    print("Testing µg merge logic:")
    for i, tokens in enumerate(test_cases, 1):
        result = corrector.correct_all(tokens)
        out = [t["token"] for t in result if t["token"].strip()]
        print(f"  Case {i}: {[t['token'] for t in tokens]} → {out}")

    print("\nAll cases passed if merged values show 'µg' where expected.")