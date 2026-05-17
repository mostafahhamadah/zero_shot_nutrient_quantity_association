"""
paddleocr_corrector.py
======================
Stage 2 — OCR Correction
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Fix systematic OCR artefacts produced by PaddleOCR on supplement label
images before tokens reach the classifier.  Every rule is narrow and
targeted — tokens that do not match a rule pass through unchanged.

INPUT / OUTPUT SCHEMA
---------------------
Input  : List[Dict]  — token, x1, y1, x2, y2, cx, cy, conf
Output : List[Dict]  — same schema, may have more tokens when splits occur

ACTIVE CORRECTION RULES
-----------------------
C1   Comma / apostrophe decimal separator
     0,8g → 0.8g  |  660'0 → 660.0
     Applied between any two digit characters.

C2   Fused integer quantity + unit
     51g → (51, g)  |  250mg → (250, mg)  |  328kcal → (328, kcal)
     Condition: the ENTIRE token must match <number><unit> exactly.
     Tokens in SPLIT_PROTECT are never split (see below).

C3   Fused decimal quantity + unit  (runs after C1 fixes the decimal point)
     0,8g → [C1] → 0.8g → [C3] → (0.8, g)

C6   Fused ENERGIE label + kJ unit
     ENERGIEkJ/kcal → (ENERGIE, kJ/kcal)
     ENERGIEkJ      → (ENERGIE, kJ)
     ENERGIELWkcal  → (ENERGIE)   — LW is an OCR noise fragment, dropped

C7   Fused context-header tokens
     Je100g  / PRO100g / PR0100g → "je 100g" / "pro 100g"
     PRO10G** / PRO10G           → "pro 10g"
     PER100ML                    → "per 100ml"

     NOTE — trailing text truncation (deliberate):
     The C7 regex matches "pro 100g Pulver" and outputs "pro 100g",
     dropping "Pulver".  This is intentional: on EU supplement labels
     the trailing text after the context header is always a serving-size
     descriptor (noise for classification purposes).  A comment is the
     correct place to document this, not a silent test case label.

C8   Unit character substitutions
     Ug → µg   (uppercase OCR confusion)
     ug → µg   (lowercase variant)

C9   Unit glyph confusion
     m9 → mg   (9 misread as g)
     pg → µg   (p misread as µ)

C10  Border artefact strip
     Strips leading / trailing  |  {  }  [  ]  \\  from token text.

C11  Fuzzy nutrient name snap
     Snaps corrupted nutrient names to their canonical lexicon form using
     difflib.SequenceMatcher (threshold 0.82).  Only fires on tokens with
     length >= 5 that are not pure numbers, known units, or already an
     exact lexicon entry.
     Examples:  "Magneslum"      -> "magnesium"
                "Kohlenhvdrate"  -> "kohlenhydrate"
                "Ballaststoff0"  -> "ballaststoffe"
                "Eiweib"         -> "eiweiss"
                "Pantothensauro" -> "pantothensaure"
                "Gesattigle"     -> "gesattigte"
     NOTE: fires on each split result token, AFTER C6/C2/C3 splits, so
     a fused token like "250Magneslum" is first split into ["250", "Magneslum"]
     and then C11 corrects "Magneslum" only.

C12  Targeted token regex corrections
     Ported from OCRCorrector.TOKEN_CORRECTIONS — fills the gap between
     narrow C8/C9 unit subs and broad C11 fuzzy snap.  Three groups:

     Group A — Numeric digit confusion (O↔0, l/I↔1 inside numbers):
       "4O6"   → "406"   |  "0.4l"  → "0.41"
       Corrupt numbers stay UNKNOWN in the classifier without this fix.

     Group B — Fused unit corruptions not caught by C8/C9:
       "(\d+)m0"   → "mg"   |  "(\d+)M9"  → "mg"
       "(\d+)K[Jj]" → "kJ"  |  "Kca[il]"   → "kcal"

     Group C — Specific nutrient name regex (precise, zero false-positives):
       "Magnes[il]um" → "Magnesium"  |  "Calc[il]um"  → "Calcium"
       "V[il]tam[il]n" → "Vitamin"   |  "Natr[il]um"  → "Natrium"
       Complements C11: regex fires before fuzzy snap on known patterns.

     Runs BEFORE C6/C2/C3 and C11 (inside the non-C7 branch).

WHAT THIS CORRECTOR DOES NOT DO
--------------------------------
- Apply the trailing-digit artefact heuristic (C4/C5) — unacceptable
  false-positive rate on real labels.
- Touch tokens it cannot unambiguously correct.

C15  Nutrient name normalisation to canonical English
     After all corrections and compound merging (C14/C14b), maps
     nutrient tokens to canonical English names matching the normalised
     ground-truth test_set.csv schema.

     Examples:
       "Fett"                            → "Fat"
       "Kohlenhydrate"                   → "Carbohydrate"
       "davon gesättigte Fettsäuren"     → "Saturated Fats"
       "davon Zucker"                    → "Sugars"
       "OF WHICH SATURATES"              → "Saturated Fats"
       "DAVON MEHRWERTIGE ALKOHOLE"      → "Sugar Alcohols"
       "Eiweiß"                          → "Protein"
       "Ballaststoffe"                   → "Fibre"
       "Natrium"                         → "Sodium"
       "Kalium"                          → "Potassium"
       "Pantothensäure"                  → "Pantothenic Acid"
       "Folsäure"                        → "Folic Acid"
       "Vitamin(e) C"                    → "Vitamin C"
       Multilingual: first segment extracted before matching

     Does NOT touch: numbers, units, context headers, or tokens not
     in the canonical mapping.

PROTECTED TOKENS  (never split by C2/C3)
-----------------------------------------
100g, 100 g, 100ml, 100 ml — these are column-header context tokens on
EU supplement labels.  Splitting them would destroy CONTEXT_MAP recognition
in the downstream classifier.

API
---
    from src.utils.paddleocr_corrector import correct_tokens

    corrected = correct_tokens(tokens)
    corrected, log = correct_tokens(tokens, return_log=True)

AUDIT LOG ENTRY FORMAT (one entry per changed input token)
----------------------------------------------------------
    {
        "input_index" : int        — position in the original input list
        "original"    : str        — raw token text before any correction
        "corrected"   : List[str]  — resulting token text(s)
        "rules_fired" : List[str]  — rule IDs in the order they fired
    }
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Union

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Tokens that are NEVER split by C2/C3 regardless of content.
SPLIT_PROTECT: frozenset = frozenset({
    "100g", "100 g",
    "100ml", "100 ml",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPILED PATTERNS  (built once at import time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── C1 ───────────────────────────────────────────────────────────────
_C1_COMMA = re.compile(r'(\d),(\d)')
_C1_APOS  = re.compile(r"(\d)'(\d)")

# ── C2/C3 ─────────────────────────────────────────────────────────────
# Matches the ENTIRE token: optional prefix, digits, optional decimal,
# optional whitespace, then a unit — nothing else.
# Units listed longest-first to prevent 'g' matching before 'mg'.
_C2_FUSED = re.compile(
    r'^([<>~]?\d+\.?\d*)\s*'
    r'('
    r'µg(?:\s*(?:RE|NE|TE|α-TE|a-TE))?'
    r'|mg(?:\s*(?:NE|RE|α-TE|a-TE))?'
    r'|ug'
    r'|kJ|KJ|kj'
    r'|kcal|KCAL'
    r'|ml|ML'
    r'|kg|KG'
    r'|IU|iu|IE|ie'
    r'|KBE|kbe'
    r'|CFU|cfu'
    r'|g|G'
    r')$',
    re.IGNORECASE,
)

# ── C6 ────────────────────────────────────────────────────────────────
_C6_ENERGIE = re.compile(
    r'^(ENERGIE)(kJ|KJ|kj|LW)(/kcal|/KCAL|kcal|KCAL)?$',
    re.IGNORECASE,
)

# ── C13 — Compound energy value splitting (exp31) ────────────────────
# Handles fused energy tokens that PaddleOCR produces as single tokens:
#   "1625 kJ/383kcal"  → ["1625", "kJ", "383", "kcal"]
#   "1383kJ/"          → ["1383", "kJ"]
#   "1696/404"         → ["1696", "404"]   (slash-separated numbers)
#   "438 kJ/103kcal"   → ["438", "kJ", "103", "kcal"]

# Pattern A: full kJ/kcal pair — "1625 kJ/383kcal" or "1625kJ/383 kcal"
_C13_FULL_PAIR = re.compile(
    r'^(\d+[.,]?\d*)\s*(kJ|kcal)\s*/\s*(\d+[.,]?\d*)\s*(kJ|kcal)$',
    re.IGNORECASE,
)

# Pattern B: number+unit with trailing slash — "1383kJ/" or "420 kJ/"
_C13_UNIT_SLASH = re.compile(
    r'^(\d+[.,]?\d*)\s*(kJ|kcal)\s*/?\s*$',
    re.IGNORECASE,
)

# Pattern C: slash-separated bare numbers — "1696/404", "509/121"
# Only matches when BOTH numbers are ≥ 50 (energy values are never tiny)
_C13_SLASH_NUMS = re.compile(
    r'^(\d{2,}[.,]?\d*)\s*/\s*(\d{2,}[.,]?\d*)$',
)

# ── C7 ────────────────────────────────────────────────────────────────
# Je/Pro/PR0 + 100 + optional g + optional trailing text (dropped — see docstring)
_C7_100G  = re.compile(r'^(?:je|pro|pr0)\s*100\s*g?(?:\s+.*)?$', re.IGNORECASE)
# Pro/PR0 + 10G with optional trailing asterisks
_C7_10G   = re.compile(r'^(?:pro|pr0)\s*10\s*g\*{0,2}$', re.IGNORECASE)
# Per + 100ml
_C7_100ML = re.compile(r'^per\s*100\s*ml$', re.IGNORECASE)

# ── C8/C9 ─────────────────────────────────────────────────────────────
# Word boundaries prevent rewriting letters inside legitimate words.
_C8C9_SUBS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r'\bm9\b'),  'mg',  'C9_m9_to_mg'),
    (re.compile(r'pg\b'),    'µg',  'C9_pg_to_µg'),   # no leading \b: works after digit
    (re.compile(r'Ug\b'),    'µg',  'C8_Ug_to_µg'),   # no leading \b: works after digit
    (re.compile(r'\bug\b'),  'µg',  'C8_ug_to_µg'),
]

# ── C10 ───────────────────────────────────────────────────────────────
_C10_STRIP = '|{}[]\\'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERNAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _clone(tok: Dict, text: str) -> Dict:
    """Return a shallow copy of *tok* with the token field replaced."""
    out = dict(tok)
    out['token'] = text.strip()
    return out


def _split_bbox_h(tok: Dict, ratio: float) -> Tuple[Dict, Dict]:
    """
    Split a token's bounding box horizontally.

    *ratio* is the fraction [0..1] assigned to the left part.
    Returns (left_token, right_token) each with updated x1, x2, cx.
    cy is inherited unchanged — vertical centre does not move on a
    horizontal split.  Stage 1 guarantees cx and cy are present.
    """
    x1, x2 = tok['x1'], tok['x2']
    sx      = int(x1 + (x2 - x1) * ratio)
    left    = dict(tok);  left['x2']  = sx;  left['cx']  = (x1 + sx) / 2.0
    right   = dict(tok);  right['x1'] = sx;  right['cx'] = (sx + x2) / 2.0
    return left, right


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PER-RULE FUNCTIONS
# Each function receives the current token dict and the rules-fired list.
# Returns (List[Dict], List[str]) — updated tokens and updated fired list.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_c10(tok: Dict, fired: List[str]) -> Tuple[Dict, List[str]]:
    """C10: strip border artefact characters from token edges."""
    text    = tok['token']
    cleaned = text.strip(_C10_STRIP).strip()
    if cleaned != text:
        fired = fired + ['C10_border_strip']
        tok   = _clone(tok, cleaned)
    return tok, fired


def _apply_c1(tok: Dict, fired: List[str]) -> Tuple[Dict, List[str]]:
    """C1: comma / apostrophe decimal separator → period."""
    text = tok['token']
    out  = _C1_COMMA.sub(r'\1.\2', text)
    out  = _C1_APOS.sub(r'\1.\2', out)
    if out != text:
        fired = fired + ['C1_decimal_sep']
        tok   = _clone(tok, out)
    return tok, fired


def _apply_c8c9(tok: Dict, fired: List[str]) -> Tuple[Dict, List[str]]:
    """C8/C9: unit character substitutions (minimal scope)."""
    text      = tok['token']
    out       = text
    new_fired : List[str] = []
    for pat, rep, rule_id in _C8C9_SUBS:
        replaced = pat.sub(rep, out)
        if replaced != out:
            new_fired.append(rule_id)
            out = replaced
    if out != text:
        fired = fired + new_fired
        tok   = _clone(tok, out)
    return tok, fired


def _apply_c7(tok: Dict, fired: List[str]) -> Tuple[Dict, List[str]]:
    """C7: normalise fused context-header tokens."""
    text = tok['token']

    if _C7_100G.match(text):
        prefix = 'je' if re.match(r'^je', text, re.IGNORECASE) else 'pro'
        return _clone(tok, f'{prefix} 100g'), fired + ['C7_context_norm']

    if _C7_100ML.match(text):
        return _clone(tok, 'per 100ml'), fired + ['C7_context_norm']

    if _C7_10G.match(text):
        return _clone(tok, 'pro 10g'), fired + ['C7_context_norm']

    return tok, fired


def _apply_c6(tok: Dict, fired: List[str]) -> Tuple[List[Dict], List[str]]:
    """C6: split ENERGIEkJ/kcal → (ENERGIE, kJ/kcal)."""
    m = _C6_ENERGIE.match(tok['token'])
    if not m:
        return [tok], fired

    energie_str = m.group(1).upper()
    kj_fragment = m.group(2)
    kcal_part   = m.group(3) or ''
    fired       = fired + ['C6_energie_split']

    if kj_fragment.upper() == 'LW':
        # LW is an OCR noise fragment — drop it, keep ENERGIE only
        return [_clone(tok, energie_str)], fired

    unit_str    = kj_fragment + kcal_part
    ratio       = len(energie_str) / max(len(tok['token']), 1)
    left, right = _split_bbox_h(tok, ratio)
    left['token']  = energie_str
    right['token'] = unit_str
    return [left, right], fired


def _split_bbox_multi(tok: Dict, texts: List[str]) -> List[Dict]:
    """
    Split a token's bounding box into N parts proportional to text lengths.
    Returns N token dicts with updated x1, x2, cx and token text.
    """
    total_chars = max(sum(len(t) for t in texts), 1)
    x1, x2 = tok['x1'], tok['x2']
    width = x2 - x1
    result = []
    cursor = x1
    for t in texts:
        frac = len(t) / total_chars
        part_w = width * frac
        part = dict(tok)
        part['x1'] = int(cursor)
        part['x2'] = int(cursor + part_w)
        part['cx'] = (cursor + cursor + part_w) / 2.0
        part['token'] = t
        result.append(part)
        cursor += part_w
    return result


def _apply_c13(tok: Dict, fired: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    C13 (exp31): split compound energy value tokens.

    PaddleOCR frequently fuses kJ/kcal pairs into single tokens:
      "1625 kJ/383kcal"  → ["1625", "kJ", "383", "kcal"]
      "1383kJ/"          → ["1383", "kJ"]
      "1696/404"         → ["1696", "404"]
    """
    text = tok['token']

    # Pattern A: full pair — "1625 kJ/383kcal"
    m = _C13_FULL_PAIR.match(text)
    if m:
        parts = [m.group(1), m.group(2), m.group(3), m.group(4)]
        return _split_bbox_multi(tok, parts), fired + ['C13_energy_full_pair']

    # Pattern B: number+unit+slash — "1383kJ/" or "420 kJ/"
    m = _C13_UNIT_SLASH.match(text)
    if m:
        parts = [m.group(1), m.group(2)]
        return _split_bbox_multi(tok, parts), fired + ['C13_energy_unit_slash']

    # Pattern C: slash-separated numbers — "1696/404"
    m = _C13_SLASH_NUMS.match(text)
    if m:
        n1, n2 = m.group(1), m.group(2)
        # Safety: only split if at least one number ≥ 100 (energy-scale)
        try:
            if float(n1.replace(',', '.')) >= 100 or float(n2.replace(',', '.')) >= 100:
                parts = [n1, n2]
                return _split_bbox_multi(tok, parts), fired + ['C13_energy_slash_nums']
        except ValueError:
            pass

    return [tok], fired


def _apply_c2c3(tok: Dict, fired: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    C2/C3: split fused quantity+unit token into (quantity, unit).
    Only fires when the ENTIRE token is exactly <number><unit>.
    Tokens in SPLIT_PROTECT pass through unchanged.

    exp37: also handles <number><unit><NRV%> by stripping the trailing
    NRV percentage suffix before matching.  Examples:
      "0,44mg(88%)"    → strip to "0,44mg"  → split ["0,44", "mg"]
      "100µg(200%)*"   → strip to "100µg"   → split ["100", "µg"]
      "2.5µg80%"       → strip to "2.5µg"   → split ["2.5", "µg"]
    """
    text = tok['token']

    if text.strip() in SPLIT_PROTECT:
        return [tok], fired

    # exp37: strip trailing NRV percentage suffix before C2 match
    # Matches: optional paren, digits, %, optional closing paren/asterisk/superscript
    text_stripped = re.sub(
        r'\(?[\d.,]+\s*%\)?[\s*³²¹⁾\)\]]*$', '', text
    ).strip()
    # Only use stripped version if it actually removed something and is non-empty
    use_stripped = (text_stripped != text and len(text_stripped) >= 2)

    m = _C2_FUSED.match(text)
    if not m and use_stripped:
        m = _C2_FUSED.match(text_stripped)
        if m:
            fired = fired + ['C16_nrv_strip']

    if not m:
        return [tok], fired

    qty_str  = m.group(1).strip()
    unit_str = m.group(2).strip()

    try:
        float(qty_str.lstrip('<>~'))
    except ValueError:
        return [tok], fired

    rule_id     = 'C3_fused_dec' if '.' in qty_str else 'C2_fused_int'
    fired       = fired + [rule_id]
    total       = max(len(qty_str) + len(unit_str), 1)
    ratio       = len(qty_str) / total
    left, right = _split_bbox_h(tok, ratio)
    left['token']  = qty_str
    right['token'] = unit_str
    return [left, right], fired



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C11 — FUZZY NUTRIENT NAME SNAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C11_SNAP_THRESHOLD: float = 0.80   # slightly relaxed — C12 regex now handles the precise cases
C11_MIN_LENGTH    : int   = 5      # minimum token length to attempt snap

# Canonical nutrient name lexicon — lowercase, length >= 5.
# Covers German / English / French / Dutch — the EU supplement label languages.
# Focus on names that PaddleOCR plausibly corrupts (length >= 5, non-trivial).
_C11_LEXICON: List[str] = [
    # ── German ─────────────────────────────────────────────────────────
    "magnesium", "calcium", "kalzium", "kalium", "phosphor",
    "chlorid", "fluorid", "natrium", "kupfer", "mangan",
    "selen", "chrom", "jodid",
    "kohlenhydrate", "ballaststoffe", "eiweiss", "fettsauren",
    "fettsaeuren", "gesattigte", "gesaettigte",
    "davon gesattigte", "davon gesaettigte",
    "folsaure", "folsaeure", "pantothensaure", "pantothensaeure",
    "riboflavin", "thiamin", "niacin", "biotin", "inositol",
    "molybdan", "molybdaen", "koffein", "kreatin",
    "brennwert", "energie", "eiweiss", "fettsauren",
    "alpha-liponsaure", "alpha-liponsaeure", "liponsaure",
    # ── English ────────────────────────────────────────────────────────
    "magnesium", "calcium", "potassium", "phosphorus",
    "chromium", "molybdenum", "manganese", "selenium", "chloride", "fluoride",
    "carbohydrate", "carbohydrates",
    "saturated", "saturates", "saturates",
    "dietary fibre", "dietary fiber",
    "riboflavin", "thiamine", "pantothenic", "folate", "cobalamin",
    "protein", "proteins", "fibre", "fiber",
    "sodium", "iodine", "copper",
    "folic acid", "pantothenic acid",
    "creatine", "caffeine", "inositol", "choline",
    # ── French ─────────────────────────────────────────────────────────
    "glucides", "proteines", "lipides", "fibres",
    "graisses", "matières grasses",
    "energie", "valeur energetique",
    # ── Dutch ──────────────────────────────────────────────────────────
    "koolhydraten", "eiwitten", "vetten", "vezels", "zout",
    "verzadigde vetzuren",
    # ── Vitamins (those long enough to benefit from fuzzy snap) ────────
    "tocopherol", "retinol", "cholecalciferol", "colecalciferol",
    "menaquinon", "phylloquinon", "pyridoxin",
    "ascorbinsaure", "ascorbinsaeure",
    "pantothensaure", "pantothensaeure",
    # ── Amino acids ────────────────────────────────────────────────────
    "leucin", "isoleucin", "methionin", "phenylalanin",
    "threonin", "tryptophan", "histidin",
    "l-valine", "l-leucine", "l-isoleucine",
    "l-lysine", "l-histidine", "l-methionine",
    # ── Specialty ──────────────────────────────────────────────────────
    "coenzyme", "coenzym", "carnitin", "lutein", "zeaxanthin",
    "astaxanthin", "resveratrol", "quercetin",
    "lactobacillus", "bifidobacterium",
    "omega", "linolsaure", "linolensaure",
    "polyunsaturated", "monounsaturated",
    # ── Compound supplement names (from Full corrector SNAP_LEXICON) ──────
    "magnesiumoxid", "magnesiumcitrat", "magnesiummalat",
    "magnesiumbisglycinat", "magnesiumglycinat",
    "kaliumcitrat", "calciumcarbonat", "eisenfumarat",
    "zinkgluconat", "zinkcitrat", "chrompicolinat",
    "selenomethionin",
]

# Remove duplicates, keep insertion order
_seen = set(); _C11_LEXICON_DEDUP: List[str] = []
for _e in _C11_LEXICON:
    if _e not in _seen:
        _seen.add(_e); _C11_LEXICON_DEDUP.append(_e)
_C11_LEXICON = _C11_LEXICON_DEDUP

# Set for O(1) "already correct" check
_C11_LEXICON_SET: frozenset = frozenset(_C11_LEXICON)

# Tokens never snapped — short words where fuzzy matching is unreliable,
# or words that are already valid non-nutrient tokens.
_C11_BLACKLIST: frozenset = frozenset({
    "davon", "davon gesattigte", "sowie", "summe",
    "total", "gesamt", "wert", "werte",
    "kbote", "kbotes", "keine", "nicht", "oder",
})

# Known unit strings — skip snap for these
_C11_UNIT_SKIP: frozenset = frozenset({
    "mg", "µg", "mcg", "ug", "μg", "kg", "kj", "kcal", "cal",
    "ml", "dl", "cl", "ie", "iu", "kbe", "cfu", "ne", "re",
    "mg/tag", "µg/tag", "g/tag",
})

# Compiled number pattern — pure numbers are never snapped
_C11_NUMBER_RE = re.compile(r'^[<>~]?\d+[.,]?\d*$')


def _apply_c11(tok: Dict, fired: List[str]) -> Tuple[Dict, List[str]]:
    """
    C11: Fuzzy nutrient name snap.

    Finds the closest entry in _C11_LEXICON using SequenceMatcher.
    Fires only when:
      - token length >= C11_MIN_LENGTH
      - not a pure number
      - not a known unit
      - not already an exact lexicon entry
      - best ratio >= C11_SNAP_THRESHOLD
    """
    text = tok['token']
    norm = text.lower().strip().rstrip('.,;:*"\' ')

    if len(norm) < C11_MIN_LENGTH:
        return tok, fired
    if _C11_NUMBER_RE.match(norm):
        return tok, fired
    if norm in _C11_UNIT_SKIP:
        return tok, fired
    if norm in _C11_BLACKLIST:
        return tok, fired
    if norm in _C11_LEXICON_SET:
        return tok, fired          # already a valid canonical form — no change needed

    best_ratio = 0.0
    best_entry = None

    # ß confusion: PaddleOCR often reads ß as 'b' (e.g. "Eiweib" → "Eiweiss").
    # Build a variant with trailing 'b' replaced by 'ss' for comparison only.
    norm_variants = [norm]
    if norm.endswith('b') and len(norm) >= 5:
        norm_variants.append(norm[:-1] + 'ss')

    for entry in _C11_LEXICON:
        if abs(len(norm) - len(entry)) > 4:
            continue               # length guard — avoids O(n) SM calls on very different lengths
        for n_var in norm_variants:
            ratio = SequenceMatcher(None, n_var, entry).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_entry = entry

    if best_ratio >= C11_SNAP_THRESHOLD and best_entry is not None:
        fired = fired + ['C11_nutrient_snap']
        tok   = _clone(tok, best_entry)

    return tok, fired



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C12 — TARGETED TOKEN REGEX CORRECTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_C12_CORRECTIONS: List[Tuple[re.Pattern, str]] = [

    # ── Group A: Numeric digit confusion (O↔0, l/I↔1) ──────────────────
    # Letter-O at start of a multi-digit number: "O406" → "0406"
    (re.compile(r"\bO(\d{2,})"),          r"0\1"),
    # Letter-O between digits: "4O6" → "406"
    (re.compile(r"(\d)O(\d)"),             r"\g<1>0\2"),
    # Letter-l or I between digits: "4l6" / "4I6" → "416"
    (re.compile(r"(\d)[lI](\d)"),          r"\g<1>1\2"),

    # ── Group B: Fused / corrupted unit tokens ───────────────────────────
    # Corrupted milligram fused with number
    (re.compile(r"(\d+[.,]?\d*)\s*m0\b", re.IGNORECASE), r"\1mg"),
    (re.compile(r"(\d+[.,]?\d*)\s*M9\b", re.IGNORECASE), r"\1mg"),
    # Fused kJ / kcal
    (re.compile(r"(\d+[.,]?\d*)\s*K[Jj]\b"),              r"\1kJ"),
    (re.compile(r"(\d+[.,]?\d*)\s*[Kk]ca[il]\b", re.IGNORECASE), r"\1kcal"),
    # Standalone unit normalisation not covered by C8/C9
    (re.compile(r"^k[Jj]$"),                               "kJ"),
    (re.compile(r"^[Kk]ca[il]$", re.IGNORECASE),          "kcal"),
    (re.compile(r"^[Mm]cg$"),                              "µg"),
    (re.compile(r"^[Kk][Jj]/[Kk]ca[il]$", re.IGNORECASE),"kJ/kcal"),

    # ── Group C: Specific nutrient name regex ────────────────────────────
    # Magnesium variants
    (re.compile(r"^Magnes[il]um$",          re.IGNORECASE), "Magnesium"),
    (re.compile(r"^Magnesiu[mn]$",          re.IGNORECASE), "Magnesium"),
    (re.compile(r"^[HM]agnesium$",          re.IGNORECASE), "Magnesium"),
    # Calcium / Kalzium
    (re.compile(r"^Calc[il]um$",            re.IGNORECASE), "Calcium"),
    (re.compile(r"^Kalz[il]um$",            re.IGNORECASE), "Kalzium"),
    # Vitamin prefix
    (re.compile(r"^V[il]tam[il]n$",         re.IGNORECASE), "Vitamin"),
    (re.compile(r"^V[il]tamin$",            re.IGNORECASE), "Vitamin"),
    # Natrium / Sodium
    (re.compile(r"^Natr[il]um$",            re.IGNORECASE), "Natrium"),
    # Kohlenhydrate
    (re.compile(r"^Koh[il]enhydrate?$",     re.IGNORECASE), "Kohlenhydrate"),
    # Energie
    (re.compile(r"^Energ[il]e[il]?$",       re.IGNORECASE), "Energie"),
    # Ballaststoffe
    (re.compile(r"^Ba[il]{1,2}aststoffe$",  re.IGNORECASE), "Ballaststoffe"),
    # Eiweiß / Eiweiss
    (re.compile(r"^E[il]we[il][sßb]{1,2}$", re.IGNORECASE), "Eiweiss"),
    # Fettsäuren
    (re.compile(r"^Fettsaure[mn]?$",        re.IGNORECASE), "Fettsauren"),
    # Niacin / Biotin
    (re.compile(r"^N[il]ac[il]n$",          re.IGNORECASE), "Niacin"),
    (re.compile(r"^B[il]ot[il]n$",          re.IGNORECASE), "Biotin"),
    # Folsäure / Pantothensäure
    (re.compile(r"^Fo[il]s[aä]ure$",        re.IGNORECASE), "Folsaure"),
    (re.compile(r"^Pantothensaure$",        re.IGNORECASE), "Pantothensaure"),
    # Riboflavin
    (re.compile(r"^R[il]bofla[vi][il]n$",   re.IGNORECASE), "Riboflavin"),
    # Phosphor / Phosphorus
    (re.compile(r"^Ph[o0]sph[o0]r$",        re.IGNORECASE), "Phosphor"),
    # Thiamin
    (re.compile(r"^Th[il]am[il]n$",         re.IGNORECASE), "Thiamin"),
    # Jod / Iodine
    (re.compile(r"^J[o0]d$",               re.IGNORECASE), "Jod"),
]


def _apply_c12(tok: Dict, fired: List[str]) -> Tuple[Dict, List[str]]:
    """
    C12: Targeted token regex corrections.

    Applies patterns in order; all matching patterns fire on the same token
    (unlike C8/C9 which short-circuits on the first match).  The result of
    each pattern feeds the next — this allows chained corrections
    (e.g. "4O6mg" first gets numeric fix → "406mg", then nothing else fires).
    """
    text = tok["token"]
    out  = text
    for pat, rep in _C12_CORRECTIONS:
        out = pat.sub(rep, out)
    if out != text:
        fired = fired + ["C12_regex_fix"]
        tok   = _clone(tok, out)
    return tok, fired

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C17 — HEADER UNIT SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Purpose:
#   Some labels encode the unit inside the nutrient/header token:
#
#       "PROTEIN (g)"       -> "PROTEIN" + "g"
#       "FAT (g)"           -> "FAT" + "g"
#       "SODIUM (mg)"       -> "SODIUM" + "mg"
#       "Energy kJ"         -> "Energy" + "kJ"
#       "Energy kJ (kcal)"  -> "Energy" + "kJ/kcal"
#
# This helps Stage 3 classify the unit as UNIT and helps Stage 5 VLM output
# non-empty units for rows such as protein/fat/sodium tables.
#
# Important:
#   C17 runs AFTER C14/C14b compound merging and BEFORE C15 nutrient
#   canonicalisation. If C15 runs first, "Energy kJ" may become "Energy"
#   and the unit would be lost.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_C17_PAREN_UNIT = re.compile(
    r"^(?P<name>.+?)\s*\(\s*(?P<unit>"
    r"kJ\s*/\s*kcal|kj\s*/\s*kcal|"
    r"kJ|kj|kcal|"
    r"mg\s*(?:NE|RE|α-TE|a-TE)?|"
    r"µg\s*(?:NE|RE|TE|α-TE|a-TE)?|"
    r"ug\s*(?:NE|RE|TE|α-TE|a-TE)?|"
    r"g|kg|ml|ML|IE|IU|KBE|CFU"
    r")\s*\)\s*$",
    re.IGNORECASE,
)

_C17_ENERGY_UNIT = re.compile(
    r"^(?P<name>energy|energie|brennwert)\s+"
    r"(?P<unit>kJ\s*/\s*kcal|kj\s*/\s*kcal|kJ|kj|kcal)"
    r"\s*$",
    re.IGNORECASE,
)

_C17_ENERGY_KJ_KCAL_PAREN = re.compile(
    r"^(?P<name>energy|energie|brennwert)\s+"
    r"(?P<u1>kJ|kj)\s*\(\s*(?P<u2>kcal)\s*\)\s*$",
    re.IGNORECASE,
)

_C17_ALLOWED_HEADER_NAMES = {
    # English
    "energy", "fat", "saturated fat", "saturated fats", "saturates",
    "carbohydrate", "carbohydrates", "sugars", "sugar",
    "protein", "fibre", "fiber", "dietary fibre", "dietary fiber",
    "salt", "sodium",

    # German
    "energie", "brennwert", "fett", "gesättigte fettsäuren",
    "gesattigte fettsauren", "kohlenhydrate", "zucker",
    "eiweiß", "eiweiss", "ballaststoffe", "salz", "natrium",

    # Common multilingual
    "glucides", "proteines", "protéines", "lipides",
    "koolhydraten", "eiwitten", "vetten", "vezels", "zout",
}


def _c17_normalise_unit(unit: str) -> str:
    """Normalise units extracted from header tokens."""
    u = str(unit or "").strip()
    u = re.sub(r"\s+", " ", u)
    u = u.replace("μg", "µg")

    if re.fullmatch(r"ug", u, flags=re.IGNORECASE):
        return "µg"

    if re.fullmatch(r"kj", u, flags=re.IGNORECASE):
        return "kJ"

    if re.fullmatch(r"kcal", u, flags=re.IGNORECASE):
        return "kcal"

    if re.fullmatch(r"kj\s*/\s*kcal", u, flags=re.IGNORECASE):
        return "kJ/kcal"

    if re.fullmatch(r"mg\s*ne", u, flags=re.IGNORECASE):
        return "mg NE"

    if re.fullmatch(r"mg\s*re", u, flags=re.IGNORECASE):
        return "mg RE"

    if re.fullmatch(r"mg\s*(a|α)-?te", u, flags=re.IGNORECASE):
        return "mg α-TE"

    if re.fullmatch(r"µg\s*re", u, flags=re.IGNORECASE):
        return "µg RE"

    return u


def _c17_is_safe_header_name(name: str) -> bool:
    """
    Decide whether a token like 'Protein (g)' is safe to split.

    Conservative:
      - common nutrition row headers are safe
      - vitamin/mineral/supplement nutrient names are safe
      - long ingredient/noise text is not split
    """
    n = str(name or "").strip()
    n_clean = n.lower()
    n_clean = re.sub(r"^[-–—•]\s*", "", n_clean).strip()
    n_clean = re.sub(r"\s+", " ", n_clean)

    if not n_clean:
        return False

    if n_clean in _C17_ALLOWED_HEADER_NAMES:
        return True

    if re.match(
        r"^(vitamin|magnesium|calcium|kalzium|zinc|zink|selen|selenium|"
        r"iron|eisen|copper|kupfer|mangan|manganese|chrom|chromium|"
        r"iodine|jod|kalium|potassium|natrium|sodium|chloride|chlorid|"
        r"phosphor|phosphorus|niacin|biotin|folic acid|folsäure|folsaure|"
        r"pantothenic acid|pantothensäure|pantothensaure|caffeine|koffein|"
        r"creatine|kreatin|inositol|choline|cholin|rutin)\b",
        n_clean,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(r"^l-[a-z]+", n_clean, flags=re.IGNORECASE):
        return True

    return False


def _c17_split_token(tok: Dict, name: str, unit: str) -> List[Dict]:
    """
    Split one token into [name_token, unit_token] with approximate horizontal bboxes.
    """
    name = name.strip()
    unit = _c17_normalise_unit(unit)

    if not name or not unit:
        return [tok]

    total = max(len(name) + len(unit), 1)
    ratio = len(name) / total

    left, right = _split_bbox_h(tok, ratio)
    left["token"] = name
    right["token"] = unit

    return [left, right]


def apply_c17_header_unit_split(tokens: List[Dict]) -> List[Dict]:
    """
    C17 post-pass: split header-embedded units.

    Examples:
      "PROTEIN (g)"      -> ["PROTEIN", "g"]
      "SODIUM (mg)"      -> ["SODIUM", "mg"]
      "Energy kJ"        -> ["Energy", "kJ"]
      "Energy kJ (kcal)" -> ["Energy", "kJ/kcal"]
    """
    if not tokens:
        return tokens

    result: List[Dict] = []

    for tok in tokens:
        text = str(tok.get("token", "") or "").strip()

        if not text:
            continue

        # Energy kJ (kcal)
        m = _C17_ENERGY_KJ_KCAL_PAREN.match(text)
        if m:
            name = m.group("name")
            unit = f"{m.group('u1')}/{m.group('u2')}"
            result.extend(_c17_split_token(tok, name, unit))
            continue

        # Energy kJ / Energy kcal / Energy kJ/kcal
        m = _C17_ENERGY_UNIT.match(text)
        if m:
            name = m.group("name")
            unit = m.group("unit")
            result.extend(_c17_split_token(tok, name, unit))
            continue

        # Protein (g), Fat (g), Sodium (mg), etc.
        m = _C17_PAREN_UNIT.match(text)
        if m:
            name = m.group("name").strip()
            unit = m.group("unit").strip()

            if _c17_is_safe_header_name(name):
                result.extend(_c17_split_token(tok, name, unit))
                continue

        result.append(tok)

    return result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C18 — Energy kJ/kcal pair splitting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Purpose:
#   Split compound energy OCR quantities into atomic quantity+unit tokens.
#
# Examples:
#   ENERGY kJ/kcal        892/213
#   -> Energy, 892 kJ and Energy, 213 kcal
#
#   Energie / energy kJ (kcal)    1541 (363)
#   -> Energy, 1541 kJ and Energy, 363 kcal
#
# Why here:
#   This belongs in the OCR corrector because "892/213" is not one
#   quantity; it is two quantities packed into one OCR token.
#
# Safety:
#   The split is only applied when the token is on an energy row.
#   It does not use numeric magnitude to guess units.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_C18_SLASH_PAIR_RE = re.compile(
    r"^\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*/\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*$"
)

_C18_PAREN_PAIR_RE = re.compile(
    r"^\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*\(\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*\)\s*$"
)

_C18_ENERGY_TEXT_RE = re.compile(
    r"\b("
    r"energy|energie|brennwert|"
    r"kj\s*/\s*kcal|kj\s*\(\s*kcal\s*\)|kcal\s*/\s*kj"
    r")\b",
    flags=re.IGNORECASE,
)

_C18_EXCLUDE_RE = re.compile(
    r"%|nrv|reference|referenz|ri\b|rda\b",
    flags=re.IGNORECASE,
)


def _c18_get_text(tok: dict) -> str:
    return str(
        tok.get("token")
        or tok.get("text")
        or tok.get("raw")
        or tok.get("raw_text")
        or ""
    ).strip()


def _c18_set_text(tok: dict, value: str) -> dict:
    new_tok = dict(tok)

    if "token" in new_tok:
        new_tok["token"] = value
    elif "text" in new_tok:
        new_tok["text"] = value
    else:
        new_tok["token"] = value

    new_tok["norm"] = value
    return new_tok


def _c18_float_text(value: str) -> str:
    """
    Keep numeric string as text but normalize decimal comma.
    """
    s = str(value or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", ".")
    return s


def _c18_num_from_token(tok: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(tok.get(key, default) or default)
    except Exception:
        return default


def _c18_token_cy(tok: dict) -> float:
    if "cy" in tok:
        return _c18_num_from_token(tok, "cy", 0.0)

    y1 = _c18_num_from_token(tok, "y1", 0.0)
    y2 = _c18_num_from_token(tok, "y2", y1)
    return (y1 + y2) / 2.0


def _c18_token_height(tok: dict) -> float:
    y1 = _c18_num_from_token(tok, "y1", 0.0)
    y2 = _c18_num_from_token(tok, "y2", y1)
    return max(1.0, abs(y2 - y1))


def _c18_same_row(tok_a: dict, tok_b: dict) -> bool:
    """
    Row test based on vertical center distance.
    """
    cy_a = _c18_token_cy(tok_a)
    cy_b = _c18_token_cy(tok_b)

    h_a = _c18_token_height(tok_a)
    h_b = _c18_token_height(tok_b)

    tol = max(12.0, 0.75 * max(h_a, h_b))
    return abs(cy_a - cy_b) <= tol


def _c18_row_has_energy_evidence(tok: dict, tokens: list) -> bool:
    """
    Checks whether the candidate pair token is on an Energy/kJ-kcal row.
    """
    row_texts = []

    for other in tokens:
        if _c18_same_row(tok, other):
            txt = _c18_get_text(other)
            if txt:
                row_texts.append(txt)

    row_text = " ".join(row_texts)

    if not row_text:
        return False

    # Avoid splitting NRV percentage pairs or unrelated references.
    if _C18_EXCLUDE_RE.search(row_text):
        return False

    return bool(_C18_ENERGY_TEXT_RE.search(row_text))


def _c18_make_virtual_token(
    base: dict,
    token_text: str,
    x1: float,
    x2: float,
    role: str,
    unit_hint: str = "",
) -> dict:
    """
    Create a virtual token using the original OCR token geometry.
    """
    new_tok = _c18_set_text(base, token_text)

    y1 = _c18_num_from_token(base, "y1", 0.0)
    y2 = _c18_num_from_token(base, "y2", y1)
    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0

    new_tok["x1"] = x1
    new_tok["x2"] = x2
    new_tok["cx"] = cx
    new_tok["y1"] = y1
    new_tok["y2"] = y2
    new_tok["cy"] = cy

    # Keep confidence but mark as virtual/corrected.
    new_tok["source_rule"] = "C18_energy_pair_split"
    new_tok["virtual_token"] = True
    new_tok["energy_pair_role"] = role

    if unit_hint:
        new_tok["unit_hint"] = unit_hint

    return new_tok


def _c18_split_energy_pair_token(tok: dict, tokens: list):
    """
    Returns either:
      None
    or:
      [q1_token, kj_token, q2_token, kcal_token]
    """
    text = _c18_get_text(tok)

    if not text:
        return None

    if _C18_EXCLUDE_RE.search(text):
        return None

    m = _C18_SLASH_PAIR_RE.match(text)
    pair_style = "slash"

    if not m:
        m = _C18_PAREN_PAIR_RE.match(text)
        pair_style = "paren"

    if not m:
        return None

    if not _c18_row_has_energy_evidence(tok, tokens):
        return None

    q1 = _c18_float_text(m.group(1))
    q2 = _c18_float_text(m.group(2))

    x1 = _c18_num_from_token(tok, "x1", 0.0)
    x2 = _c18_num_from_token(tok, "x2", x1)
    w = max(1.0, x2 - x1)

    # Approximate virtual geometry inside the original OCR box.
    # This is enough for downstream row/association logic.
    q1_x1 = x1
    q1_x2 = x1 + 0.38 * w

    kj_x1 = x1 + 0.39 * w
    kj_x2 = x1 + 0.49 * w

    q2_x1 = x1 + 0.51 * w
    q2_x2 = x1 + 0.88 * w

    kcal_x1 = x1 + 0.89 * w
    kcal_x2 = x2

    q1_tok = _c18_make_virtual_token(
        base=tok,
        token_text=q1,
        x1=q1_x1,
        x2=q1_x2,
        role=f"{pair_style}_primary_quantity",
        unit_hint="kJ",
    )

    kj_tok = _c18_make_virtual_token(
        base=tok,
        token_text="kJ",
        x1=kj_x1,
        x2=kj_x2,
        role=f"{pair_style}_primary_unit",
        unit_hint="kJ",
    )

    q2_tok = _c18_make_virtual_token(
        base=tok,
        token_text=q2,
        x1=q2_x1,
        x2=q2_x2,
        role=f"{pair_style}_secondary_quantity",
        unit_hint="kcal",
    )

    kcal_tok = _c18_make_virtual_token(
        base=tok,
        token_text="kcal",
        x1=kcal_x1,
        x2=kcal_x2,
        role=f"{pair_style}_secondary_unit",
        unit_hint="kcal",
    )

    return [q1_tok, kj_tok, q2_tok, kcal_tok]


def apply_c18_energy_pair_split(tokens):
    """
    Split packed Energy kJ/kcal quantity tokens.

    Input token:
        892/213

    Output tokens:
        892, kJ, 213, kcal

    Only applies when the token is on a row with Energy/kJ/kcal evidence.
    """
    if not tokens:
        return tokens

    out = []

    for tok in tokens:
        split_tokens = _c18_split_energy_pair_token(tok, tokens)

        if split_tokens:
            out.extend(split_tokens)
        else:
            out.append(tok)

    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C18b — Energy kJ/kcal pair rewrite, not split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Purpose:
#   Make packed Energy kJ/kcal values explicit for the VLM without
#   creating artificial tokens or fake geometry.
#
# Examples:
#   892/213       -> VLM display: 892 kJ / 213 kcal
#   1541 (363)    -> VLM display: 1541 kJ / 363 kcal
#
# Important:
#   This does NOT replace the original OCR token.
#   It only adds metadata:
#       tok["vlm_display"] = "892 kJ / 213 kcal"
#
# Why:
#   The classifier can still see the original numeric token "892/213".
#   The VLM can see the clearer form "892 kJ / 213 kcal".
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_C18B_SLASH_PAIR_RE = re.compile(
    r"^\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*/\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*$"
)

_C18B_PAREN_PAIR_RE = re.compile(
    r"^\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*\(\s*([<>]?\s*\d+(?:[.,]\d+)?)\s*\)\s*$"
)

_C18B_ENERGY_ROW_RE = re.compile(
    r"energy|energie|brennwert|kj\s*/\s*kcal|kj\s*\(\s*kcal\s*\)",
    flags=re.IGNORECASE,
)

_C18B_EXCLUDE_RE = re.compile(
    r"%|nrv|reference|referenz|ri\b|rda\b",
    flags=re.IGNORECASE,
)


def _c18b_get_text(tok: dict) -> str:
    return str(
        tok.get("token")
        or tok.get("text")
        or tok.get("raw")
        or tok.get("raw_text")
        or ""
    ).strip()


def _c18b_num(tok: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(tok.get(key, default) or default)
    except Exception:
        return default


def _c18b_cy(tok: dict) -> float:
    if "cy" in tok:
        return _c18b_num(tok, "cy", 0.0)

    y1 = _c18b_num(tok, "y1", 0.0)
    y2 = _c18b_num(tok, "y2", y1)
    return (y1 + y2) / 2.0


def _c18b_height(tok: dict) -> float:
    y1 = _c18b_num(tok, "y1", 0.0)
    y2 = _c18b_num(tok, "y2", y1)
    return max(1.0, abs(y2 - y1))


def _c18b_same_row(a: dict, b: dict) -> bool:
    cy_a = _c18b_cy(a)
    cy_b = _c18b_cy(b)

    h_a = _c18b_height(a)
    h_b = _c18b_height(b)

    tol = max(12.0, 0.75 * max(h_a, h_b))
    return abs(cy_a - cy_b) <= tol


def _c18b_row_text(tok: dict, tokens: list) -> str:
    parts = []

    for other in tokens:
        if _c18b_same_row(tok, other):
            txt = _c18b_get_text(other)
            if txt:
                parts.append(txt)

    return " ".join(parts)


def _c18b_norm_number(value: str) -> str:
    s = str(value or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", ".")
    return s


def _c18b_rewrite_token(tok: dict, tokens: list) -> dict:
    text = _c18b_get_text(tok)

    if not text:
        return tok

    if _C18B_EXCLUDE_RE.search(text):
        return tok

    m = _C18B_SLASH_PAIR_RE.match(text)

    if not m:
        m = _C18B_PAREN_PAIR_RE.match(text)

    if not m:
        return tok

    row_text = _c18b_row_text(tok, tokens)

    if not row_text:
        return tok

    if _C18B_EXCLUDE_RE.search(row_text):
        return tok

    if not _C18B_ENERGY_ROW_RE.search(row_text):
        return tok

    q1 = _c18b_norm_number(m.group(1))
    q2 = _c18b_norm_number(m.group(2))

    rewritten = dict(tok)

    # Keep original token unchanged.
    # Add VLM-only display text.
    rewritten["vlm_display"] = f"{q1} kJ / {q2} kcal"

    # Metadata for debugging / optional future use.
    rewritten["source_rule"] = "C18b_energy_pair_rewrite"
    rewritten["energy_pair_explicit"] = True
    rewritten["energy_pair_primary_quantity"] = q1
    rewritten["energy_pair_primary_unit"] = "kJ"
    rewritten["energy_pair_secondary_quantity"] = q2
    rewritten["energy_pair_secondary_unit"] = "kcal"

    return rewritten


def apply_c18b_energy_pair_rewrite(tokens):
    """
    Rewrite packed Energy kJ/kcal cells for VLM display only.

    Does not split tokens.
    Does not change geometry.
    Does not infer from numeric magnitude.
    """
    if not tokens:
        return tokens

    return [_c18b_rewrite_token(tok, tokens) for tok in tokens]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def correct_tokens(
    tokens    : List[Dict],
    return_log: bool = False,
) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
    """
    Apply all active correction rules to a list of PaddleOCR tokens.

    Processing order per token
    ──────────────────────────
      1. C10  border strip
      2. C1   decimal separator
      3. C8/9 unit character substitutions
      4. C7   context-token normalisation
         → if C7 fires, C6/C2/C3 are skipped (token is a context header,
           not a value — splitting it would be wrong)
      5. C6   ENERGIE+kJ split
      5b. C13  compound energy value split (exp31)
      6. C2/3 fused quantity+unit split
      5b. C12  targeted regex corrections (numeric, units, specific nutrients)
      7. C11  fuzzy nutrient name snap (on each result token after splits)

    Parameters
    ----------
    tokens     : List[Dict]  — output of paddleocr_runner.run_ocr_on_image()
    return_log : bool        — if True, return (tokens, audit_log)

    Returns
    -------
    corrected_tokens : List[Dict]
        May be longer than input when C2/C3/C6 splits occur.
    audit_log : List[Dict]   (only when return_log=True)
        One entry per input token where at least one rule fired.
    """
    corrected : List[Dict] = []
    audit_log : List[Dict] = []

    for idx, raw in enumerate(tokens):
        tok           = dict(raw)          # shallow copy — sufficient
        fired         : List[str] = []
        original_text = tok['token']

        # C10: border strip
        tok, fired = _apply_c10(tok, fired)
        if not tok['token']:
            continue                        # empty after strip — discard

        # C1: decimal separator
        tok, fired = _apply_c1(tok, fired)

        # C8/C9: unit character substitutions
        tok, fired = _apply_c8c9(tok, fired)

        # C7: context normalisation
        tok, fired = _apply_c7(tok, fired)

        if 'C7_context_norm' in fired:
            result_toks = [tok]
        else:
            # C12: targeted regex corrections (numeric, units, nutrients)
            tok, fired = _apply_c12(tok, fired)

            # C6: ENERGIE+kJ split
            result_toks, fired = _apply_c6(tok, fired)

            # C13 (exp31): compound energy value split — applied to each part
            c13_result: List[Dict] = []
            for part in result_toks:
                split_parts, fired = _apply_c13(part, fired)
                c13_result.extend(split_parts)
            result_toks = c13_result

            # C2/C3: fused qty+unit split (applied to each part from C6/C13)
            final : List[Dict] = []
            for part in result_toks:
                split_parts, fired = _apply_c2c3(part, fired)
                final.extend(split_parts)
            result_toks = final

            # C11: fuzzy nutrient name snap — operates on each split result
            c11_result: List[Dict] = []
            for part in result_toks:
                part, fired = _apply_c11(part, fired)
                c11_result.append(part)
            result_toks = c11_result

        corrected.extend(result_toks)

        if return_log and fired:
            audit_log.append({
                'input_index': idx,
                'original':    original_text,
                'corrected':   [t['token'] for t in result_toks],
                'rules_fired': fired,
            })

        # ── C14 post-pass: merge split compound nutrients ────────────
    corrected = apply_c14_merge_compounds(corrected)

    # ── C14b: cross-line continuation merge ──────────────────────
    corrected = apply_c14_cross_line_merge(corrected)

    # ── C17: split units embedded in nutrient/header tokens ──────
    corrected = apply_c17_header_unit_split(corrected)

    # ── C18: split Energy kJ/kcal compound quantity cells ────────
    # Example:
    #   892/213    -> 892 kJ + 213 kcal
    #   1541 (363) -> 1541 kJ + 363 kcal
    #corrected = apply_c18_energy_pair_split(corrected)
    #corrected = apply_c18b_energy_pair_rewrite(corrected)

    # ── C15: nutrient name normalisation to canonical English ────
    corrected = apply_c15_normalise_names(corrected)

    if return_log:
        return corrected, audit_log

    return corrected

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C14 — COMPOUND NUTRIENT MERGE (exp32 post-pass)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PaddleOCR splits compound sub-nutrient names across multiple tokens:
#   "davon" + "gesättigte" + "Fettsäuren"  → should be one NUTRIENT
#   "OF" + "WHICH" + "SATURATES"           → should be one NUTRIENT
#   "-" + "davon" + "Zucker"               → should be one NUTRIENT
#
# These split tokens cause the classifier to miss the compound name,
# producing standalone "Fettsäuren" which the evaluator rejects for
# specificity violations against GT "davon gesättigte Fettsäuren".
#
# Strategy: scan the corrected token list for known prefix tokens,
# merge with next 1-2 adjacent same-row tokens into a single compound.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Prefixes that trigger a merge attempt (lowercase)
_C14_PREFIXES = frozenset({
    "davon", "- davon", "-davon", "davon gesättigte", "davon gesattigte",
    "of which", "of", "dont", "dont acides",
    "waarvan", "- waarvan", "-waarvan", "waarvan verzadigde",
    "di cui", "de las cuales", "de los cuales",
    "dos quais", "od tega", "z toho",
    "w tym",        # Polish
    "josta",        # Finnish
    "med andel",    # Danish
    "varav",        # Swedish
})

# Second-word triggers (lowercase) — when first word is "of"
_C14_OF_CONTINUATIONS = frozenset({
    "which", "whom",
})

# Maximum horizontal gap (px) and vertical distance (px) for merge
_C14_MAX_GAP_PX    = 80
_C14_MAX_CY_DIFF   = 20


def _c14_same_row(t1: dict, t2: dict) -> bool:
    """Check if two tokens are on the same visual row and close enough to merge."""
    cy_diff = abs(t1.get("cy", 0) - t2.get("cy", 0))
    gap = t2.get("x1", 0) - t1.get("x2", 0)
    return cy_diff <= _C14_MAX_CY_DIFF and gap <= _C14_MAX_GAP_PX


def _c14_merge_tokens(tokens_to_merge: list) -> dict:
    """Merge a list of adjacent tokens into one compound token."""
    merged = dict(tokens_to_merge[0])
    texts = [t.get("token", "").strip() for t in tokens_to_merge]

    # Strip leading dash/bullet from first token for cleaner merge
    first = texts[0]
    if first in ("-", "–", "•"):
        texts[0] = ""
    elif first.startswith(("- ", "– ")):
        texts[0] = first[2:]

    merged["token"] = " ".join(t for t in texts if t).strip()
    merged["x1"] = tokens_to_merge[0].get("x1", 0)
    merged["x2"] = tokens_to_merge[-1].get("x2", 0)
    merged["cx"] = (merged["x1"] + merged["x2"]) / 2.0
    merged["conf"] = min(t.get("conf", 0.9) for t in tokens_to_merge)
    return merged


def apply_c14_merge_compounds(tokens: list) -> list:
    """
    C14 post-pass: merge split compound nutrient tokens.

    Scans for prefix tokens like "davon", "of which", "dont" etc.
    and merges with the next 1-2 tokens on the same row.

    This runs AFTER all per-token corrections (C1-C13) and BEFORE
    classification, so the classifier sees complete compound names.
    """
    if not tokens:
        return tokens

    result = []
    skip = set()
    n = len(tokens)

    for i in range(n):
        if i in skip:
            continue

        tok = tokens[i]
        text_lower = tok.get("token", "").strip().lower().strip("-–• ")

        # Check if this is a prefix token (exact or startswith)
        is_prefix = text_lower in _C14_PREFIXES or any(
            text_lower.startswith(p) for p in _C14_PREFIXES if len(p) >= 4
        )

        # Special case: bare "of" needs "which" after it
        if text_lower == "of" and i + 1 < n:
            next_lower = tokens[i + 1].get("token", "").strip().lower()
            if next_lower not in _C14_OF_CONTINUATIONS:
                is_prefix = False

        # Special case: bare "-" or "–" needs "davon" or similar after it
        raw_text = tok.get("token", "").strip()
        if raw_text in ("-", "–", "•") and i + 1 < n:
            next_lower = tokens[i + 1].get("token", "").strip().lower()
            if next_lower in ("davon", "dont", "waarvan", "di"):
                is_prefix = True  # merge "-" + "davon" + continuation

        if not is_prefix:
            result.append(tok)
            continue

        # Try to merge with next 1-2 tokens on the same row
        merge_group = [tok]

        # Collect up to 4 more tokens (e.g., "waarvan" + "verzadigde" + "vetzuren" or longer)
        for j in range(i + 1, min(i + 5, n)):
            if j in skip:
                break
            candidate = tokens[j]
            if not _c14_same_row(merge_group[-1], candidate):
                break
            # Don't merge if the candidate looks like a number (quantity)
            cand_text = candidate.get("token", "").strip()
            if re.match(r'^[<>]?\d+[.,]?\d*$', cand_text):
                break
            merge_group.append(candidate)
            skip.add(j)

        if len(merge_group) > 1:
            merged = _c14_merge_tokens(merge_group)
            result.append(merged)
        else:
            result.append(tok)

    return result

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ── C14b: Cross-line continuation merge ──────────────────────────────

_C14_CONTINUATIONS = frozenset({
    # Saturated fat continuations (all languages)
    "fettsauren", "fettsäuren",
    "saturates", "saturated", "saturated fat",
    "fatty acids", "saturated fatty acids",
    "acides gras saturés", "acides gras",
    "verzadigde vetzuren", "vetzuren",
    "acidi grassi saturi", "grassi saturi",
    "nasycené mastné kyseliny", "mastné kyseliny",
    "nasýtené mastné kyseliny",
    "tłuszcze nasycone", "kwasy tłuszczowe nasycone",
    "mättat fett", "mættet fett",
    "ácidos gordos saturados",
    "de las cuales saturadas", "dos quais saturados",
    "od tega nasičene", "nasičene",
    # Sugar continuations
    "zucker", "zuckerarten", "sugars",
    "sucres", "suikers", "cukry", "socker", "sokker", "sukker",
    "zuccheri", "açúcares", "azúcares",
    "sladkorji",
    # Other sub-nutrient continuations
    "mehrwertige alkohole",
    "fructose",
})

_C14_CROSS_LINE_MAX_CY = 60     # increased from 50 for tighter label layouts
_C14_CROSS_LINE_MAX_CX = 140    # increased from 100 for wider column spans


def apply_c14_cross_line_merge(tokens: list) -> list:
    """
    C14b: merge standalone continuation tokens with the nearest
    'davon' prefix on the line above.

    Handles wrapped compound names:
      Line 1: "davon gesättigte"    (cy=300)
      Line 2: "Fettsäuren"          (cy=330)
    """
    if not tokens:
        return tokens

    result = list(tokens)
    skip = set()

    for i, tok in enumerate(result):
        if i in skip:
            continue
        text_lower = tok.get("token", "").strip().lower()

        if text_lower not in _C14_CONTINUATIONS:
            continue

        tok_cy = tok.get("cy", 0)
        tok_cx = tok.get("cx", 0)
        best_j = -1
        best_cy_diff = _C14_CROSS_LINE_MAX_CY + 1

        for j in range(i - 1, max(i - 8, -1), -1):
            if j in skip:
                continue
            cand = result[j]
            cand_text = cand.get("token", "").strip().lower().strip("-–• ")
            cand_cy = cand.get("cy", 0)
            cand_cx = cand.get("cx", 0)

            cy_diff = tok_cy - cand_cy
            cx_diff = abs(tok_cx - cand_cx)

            if cy_diff <= 0 or cy_diff > _C14_CROSS_LINE_MAX_CY:
                continue
            if cx_diff > _C14_CROSS_LINE_MAX_CX:
                continue

            is_davon = any(cand_text.endswith(p) for p in
                          ("davon", "davon gesattigte", "davon gesättigte",
                           "of which", "dont", "waarvan",
                           "waarvan verzadigde",
                           "di cui", "de las cuales", "de los cuales",
                           "dos quais", "od tega",
                           "z toho", "w tym", "josta",
                           "med andel", "varav",
                           "davon mehrwertige"))
            if is_davon and cy_diff < best_cy_diff:
                best_cy_diff = cy_diff
                best_j = j

        if best_j >= 0:
            prefix_tok = result[best_j]
            merged_text = prefix_tok.get("token", "").strip() + " " + tok.get("token", "").strip()
            prefix_tok["token"] = merged_text
            prefix_tok["x2"] = max(prefix_tok.get("x2", 0), tok.get("x2", 0))
            prefix_tok["cx"] = (prefix_tok.get("x1", 0) + prefix_tok["x2"]) / 2.0
            skip.add(i)

    return [t for idx, t in enumerate(result) if idx not in skip]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C15 — NUTRIENT NAME NORMALISATION TO CANONICAL ENGLISH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Runs AFTER C14/C14b merging so compound tokens are already joined.
# Maps nutrient tokens to canonical English names that match the
# normalised ground-truth test_set.csv schema.
#
# Strategy:
#   1. Strip leading dashes/bullets and trailing colons/asterisks
#   2. Try exact-match (case-insensitive) against _C15_EXACT_MAP
#   3. Try first-segment extraction (split on / | separators)
#   4. Try regex pattern matching for remaining edge cases
#   5. If no match, leave the token unchanged
#
# IMPORTANT: only fires on tokens that look like nutrient names
# (length >= 3, not a number, not a known unit, not a context token).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_C15_EXACT_MAP: Dict[str, str] = {
    # ── Energy ────────────────────────────────────────────────────────
    "energie":              "Energy",
    "energy":               "Energy",
    "brennwert":            "Energy",

    # ── Fat ───────────────────────────────────────────────────────────
    "fett":                 "Fat",
    "fat":                  "Fat",
    "vetten":               "Fat",
    "lipides":              "Fat",
    "grassi":               "Fat",
    "grasas":               "Fat",
    "matières grasses":     "Fat",

    # ── Saturated Fats ────────────────────────────────────────────────
    "davon gesättigte fettsäuren":              "Saturated Fats",
    "davon gesattigte fettsauren":              "Saturated Fats",
    "gesättigte fettsäuren":                    "Saturated Fats",
    "gesattigte fettsauren":                    "Saturated Fats",
    "of which saturates":                       "Saturated Fats",
    "of which saturated":                       "Saturated Fats",
    "of which saturated fatty acids":           "Saturated Fats",
    "saturated fat":                            "Saturated Fats",
    "saturated fats":                           "Saturated Fats",
    "saturates":                                "Saturated Fats",
    "waarvan verzadigde vetzuren":              "Saturated Fats",
    "verzadigde vetzuren":                      "Saturated Fats",
    "dont acides gras saturés":                 "Saturated Fats",
    "acides gras saturés":                      "Saturated Fats",
    "dont matières grasses saturées":           "Saturated Fats",
    "matières grasses saturées":                "Saturated Fats",
    "di cui acidi grassi saturi":               "Saturated Fats",
    "acidi grassi saturi":                      "Saturated Fats",
    "de las cuales saturadas":                  "Saturated Fats",
    "dos quais saturados":                      "Saturated Fats",
    "dos quais ácidos gordos saturados":        "Saturated Fats",
    "od tega nasičene":                         "Saturated Fats",
    "w tym tłuszcze nasycone":                  "Saturated Fats",
    "w tym kwasy tłuszczowe nasycone":          "Saturated Fats",
    "z toho nasycené mastné kyseliny":          "Saturated Fats",
    "z toho nasýtené mastné kyseliny":          "Saturated Fats",
    "med andel mættet fett":                    "Saturated Fats",
    "varav mättat fett":                        "Saturated Fats",
    "josta tyydyttyneiden rasvojen osuus":      "Saturated Fats",
    "davon gesättigte fettsäuren/of which saturates":          "Saturated Fats",
    "davon gesättigte fettsäuren/of which saturated fatty acids": "Saturated Fats",

    # ── Carbohydrate ──────────────────────────────────────────────────
    "kohlenhydrate":        "Carbohydrate",
    "carbohydrate":         "Carbohydrate",
    "carbohydrates":        "Carbohydrate",
    "koolhydraten":         "Carbohydrate",
    "glucides":             "Carbohydrate",
    "carboidrati":          "Carbohydrate",
    "hidratos de carbono":  "Carbohydrate",
    "węglowodany":          "Carbohydrate",
    "sacharidy":            "Carbohydrate",
    "kolhydrater":          "Carbohydrate",
    "ogljikovi hidrati":    "Carbohydrate",
    "hiilihydraatit":       "Carbohydrate",
    "karbohydrater":        "Carbohydrate",

    # ── Sugars ────────────────────────────────────────────────────────
    "davon zucker":         "Sugars",
    "davon zuckerarten":    "Sugars",
    "zucker":               "Sugars",
    "zuckerarten":          "Sugars",
    "sugars":               "Sugars",
    "of which sugars":      "Sugars",
    "waarvan suikers":      "Sugars",
    "suikers":              "Sugars",
    "dont sucres":          "Sugars",
    "sucres":               "Sugars",
    "di cui zuccheri":      "Sugars",
    "zuccheri":             "Sugars",
    "de los cuales azúcares": "Sugars",
    "dos quais açúcares":   "Sugars",
    "od tega sladkorji":    "Sugars",
    "w tym cukry":          "Sugars",
    "z toho cukry":         "Sugars",
    "varav socker":         "Sugars",
    "med andel sukker":     "Sugars",
    "josta sokereiden osuus": "Sugars",
    "varav sockerarter":    "Sugars",
    "davon zucker/of which sugars":   "Sugars",
    "davon zuckerarten/of which sugars": "Sugars",

    # ── Sugar Alcohols ────────────────────────────────────────────────
    "davon mehrwertige alkohole":     "Sugar Alcohols",
    "mehrwertige alkohole":           "Sugar Alcohols",

    # ── Fructose ──────────────────────────────────────────────────────
    "davon fructose":       "Fructose",
    "fructose":             "Fructose",

    # ── Protein ───────────────────────────────────────────────────────
    "eiweiß":               "Protein",
    "eiweiss":              "Protein",
    "protein":              "Protein",
    "protéines":            "Protein",
    "proteine":             "Protein",
    "eiwitten":             "Protein",
    "proteiinit":           "Protein",
    "białko":               "Protein",
    "bílkoviny":            "Protein",
    "bielkoviny":           "Protein",
    "proteína":             "Protein",
    "proteínas":            "Protein",
    "beljakovine":          "Protein",

    # ── Fibre ─────────────────────────────────────────────────────────
    "ballaststoffe":        "Fibre",
    "fibre":                "Fibre",
    "fiber":                "Fibre",
    "dietary fibre":        "Fibre",
    "dietary fiber":        "Fibre",
    "fibres":               "Fibre",
    "vezels":               "Fibre",

    # ── Salt ──────────────────────────────────────────────────────────
    "salz":                 "Salt",
    "salt":                 "Salt",
    "sel":                  "Salt",
    "zout":                 "Salt",
    "sól":                  "Salt",
    "sůl":                  "Salt",
    "sale":                 "Salt",
    "sol":                  "Salt",
    "sal":                  "Salt",
    "suola":                "Salt",

    # ── Sodium ────────────────────────────────────────────────────────
    "natrium":              "Sodium",
    "sodium":               "Sodium",

    # ── Calcium ───────────────────────────────────────────────────────
    "calcium":              "Calcium",
    "kalzium":              "Calcium",
    "calcio":               "Calcium",
    "kalcij":               "Calcium",

    # ── Magnesium ─────────────────────────────────────────────────────
    "magnesium":            "Magnesium",
    "magnésium":            "Magnesium",
    "magnesio":             "Magnesium",
    "magnézium":            "Magnesium",
    "magnezij":             "Magnesium",

    # ── Potassium ─────────────────────────────────────────────────────
    "kalium":               "Potassium",
    "potassium":            "Potassium",
    "potassio":             "Potassium",
    "kalij":                "Potassium",

    # ── Chloride ──────────────────────────────────────────────────────
    "chlorid":              "Chloride",
    "chloride":             "Chloride",
    "chlorure":             "Chloride",

    # ── Phosphorus ────────────────────────────────────────────────────
    "phosphor":             "Phosphorus",
    "phosphorus":           "Phosphorus",

    # ── Zinc ──────────────────────────────────────────────────────────
    "zink":                 "Zinc",
    "zinc":                 "Zinc",

    # ── Selenium ──────────────────────────────────────────────────────
    "selen":                "Selenium",
    "selenium":             "Selenium",

    # ── Copper ────────────────────────────────────────────────────────
    "kupfer":               "Copper",
    "copper":               "Copper",

    # ── Manganese ─────────────────────────────────────────────────────
    "mangan":               "Manganese",
    "manganese":            "Manganese",

    # ── Chromium ──────────────────────────────────────────────────────
    "chrom":                "Chromium",
    "chromium":             "Chromium",

    # ── Iodine ────────────────────────────────────────────────────────
    "jod":                  "Iodine",
    "jodid":                "Iodine",
    "iodine":               "Iodine",

    # ── Molybdenum ────────────────────────────────────────────────────
    "molybdän":             "Molybdenum",
    "molybdan":             "Molybdenum",
    "molybdenum":           "Molybdenum",

    # ── Iron ──────────────────────────────────────────────────────────
    "eisen":                "Iron",
    "iron":                 "Iron",

    # ── Vitamins ──────────────────────────────────────────────────────
    "thiamin":              "Vitamin B1",
    "thiamine":             "Vitamin B1",
    "riboflavin":           "Vitamin B2",
    "niacin":               "Niacin",
    "pantothensäure":       "Pantothenic Acid",
    "pantothensaure":       "Pantothenic Acid",
    "pantothenic acid":     "Pantothenic Acid",
    "folsäure":             "Folic Acid",
    "folsaure":             "Folic Acid",
    "folic acid":           "Folic Acid",
    "biotin":               "Biotin",
    "koffein":              "Caffeine",
    "caffeine":             "Caffeine",
    "colecalciferol":       "Colecalciferol",

    # ── Amino acids ───────────────────────────────────────────────────
    "l-valine":             "L-Valine",
    "l-valin":              "L-Valine",
    "l-leucine":            "L-Leucine",
    "l-leucin":             "L-Leucine",
    "l-isoleucine":         "L-Isoleucine",
    "l-isoleucin":          "L-Isoleucine",
    "l-lysine":             "L-Lysine",
    "l-lysin":              "L-Lysine",
    "l-histidine":          "L-Histidine",
    "l-histidin":           "L-Histidine",
    "l-methionine":         "L-Methionine",
    "l-methionin":          "L-Methionine",
    "l-phenylalanine":      "L-Phenylalanine",
    "l-phenylalanin":       "L-Phenylalanine",
    "l-threonine":          "L-Threonine",
    "l-threonin":           "L-Threonine",
    "l-tryptophan":         "L-Tryptophan",

    # ── Specialty (case-normalise only) ───────────────────────────────
    "green tea":            "Green Tea",
    "inositol":             "Inositol",
    "alpha-liponsäure":     "Alpha-Liponsäure",
    "alpha-liponsaure":     "Alpha-Liponsäure",
    "cholin":               "Choline",
    "choline":              "Choline",
    "rutin":                "Rutin",
    "lactobacillus acidophilus": "Lactobacillus Acidophilus",
    "bifidobacterium bifidum":  "Bifidobacterium Bifidum",
    "creatine monohydrate": "Creatine monohydrate",
    "kreatin monohydrat":   "Creatine monohydrate",
    "melissen extrakt":     "Melissen Extrakt",
}

# Tokens that should NOT be normalised (context headers, units, noise)
_C15_SKIP_RE = re.compile(
    r'^('
    r'[<>~]?\d+[.,]?\d*'              # numbers
    r'|(?:je|pro|per)\s+\d+'           # context headers
    r'|(?:mg|µg|g|kg|kj|kcal|ml|iu|ie|kbe|cfu)\b'  # units
    r'|\d+\s*(?:g|ml)\b'              # "100g", "100ml"
    r')$',
    re.IGNORECASE,
)

# Vitamin patterns: "Vitamin B6", "VITAMIN C", "Vitamin(e) B12", "Vitamine E"
_C15_VITAMIN_RE = re.compile(
    r'^(?:vitamin(?:e|\(e\))?)\s*(a|b1|b2|b3|b5|b6|b12|c|d|d3|e|k)\b',
    re.IGNORECASE,
)

_C15_VITAMIN_MAP = {
    "a": "Vitamin A", "b1": "Vitamin B1", "b2": "Vitamin B2",
    "b3": "Niacin", "b5": "Pantothenic Acid", "b6": "Vitamin B6",
    "b12": "Vitamin B12", "c": "Vitamin C", "d": "Vitamin D",
    "d3": "Vitamin D3", "e": "Vitamin E", "k": "Vitamin K",
}

# Short standalone vitamin/mineral letters: "C" → "Vitamin C", "B3" → "Niacin"
_C15_SHORT_MAP: Dict[str, str] = {
    "c": "Vitamin C", "b1": "Vitamin B1", "b2": "Vitamin B2",
    "b3": "Niacin", "b5": "Pantothenic Acid", "b6": "Vitamin B6",
    "b12": "Vitamin B12",
}


def _c15_clean(text: str) -> str:
    """Strip formatting artefacts for C15 matching."""
    s = text.strip()
    s = re.sub(r'^[-–—•]\s*', '', s)       # leading dash/bullet
    s = s.rstrip(':*')                       # trailing colon/asterisks
    s = re.sub(r'\s*\.{2,}\s*', ' / ', s)   # dots → slash
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _c15_first_segment(name: str) -> str:
    """Extract first segment before / or | separator."""
    for sep in [' | ', ' / ', '/ ', '/']:
        if sep in name:
            return name.split(sep)[0].strip()
    return name


def apply_c15_normalise_names(tokens: list) -> list:
    """
    C15 post-pass: normalise nutrient names to canonical English.

    Runs after C14/C14b merging.  Matches each token against
    _C15_EXACT_MAP (case-insensitive).  For multilingual tokens
    (containing / or |), extracts the first segment and matches that.

    Tokens that look like numbers, units, or context headers are skipped.
    """
    result = []
    for tok in tokens:
        text = tok.get("token", "")
        cleaned = _c15_clean(text)

        # Skip numbers, units, context headers
        if _C15_SKIP_RE.match(cleaned):
            result.append(tok)
            continue

        # Skip very short tokens (single chars that aren't vitamin letters)
        if len(cleaned) <= 1 and cleaned.lower() not in _C15_SHORT_MAP:
            result.append(tok)
            continue

        canonical = None

        # 1) Exact match on full cleaned text
        cl = cleaned.lower()
        if cl in _C15_EXACT_MAP:
            canonical = _C15_EXACT_MAP[cl]

        # 2) Try first segment (multilingual names)
        if canonical is None and ('/' in cleaned or '|' in cleaned):
            first = _c15_first_segment(cleaned).lower()
            # Clean the first segment too
            first = re.sub(r'^[-–—•]\s*', '', first).strip()
            if first in _C15_EXACT_MAP:
                canonical = _C15_EXACT_MAP[first]

        # 3) Vitamin pattern: "Vitamin B6", "Vitamin(e) C", etc.
        if canonical is None:
            vm = _C15_VITAMIN_RE.match(cleaned)
            if vm:
                vit_letter = vm.group(1).lower()
                canonical = _C15_VITAMIN_MAP.get(vit_letter)

        # 4) Short standalone: "C", "B3", "B12"
        if canonical is None and cl in _C15_SHORT_MAP:
            canonical = _C15_SHORT_MAP[cl]

        # 5) Koffein/Caffeine with parenthetical (e.g., "Koffein (aus Guarana...)")
        if canonical is None and re.match(r'(?i)^koffein\b', cleaned):
            canonical = "Caffeine"

        # 6) "Vitamin A (aus/from Provitamin A)" and similar
        if canonical is None and re.match(r'(?i)^vitamin\s*a\b', cleaned):
            canonical = "Vitamin A"

        # 7) Energy with unit suffix: "Energie/energy kJ", "ENERGIE kcal"
        if canonical is None and re.match(r'(?i)^(?:energie|energy|brennwert)\b', cleaned):
            canonical = "Energy"

        # 8) Niacin with qualifier: "Niacin (NE)", "Niacin(e)"
        if canonical is None and re.match(r'(?i)^niacin', cleaned):
            canonical = "Niacin"

        # 9) Vitamin E with qualifier: "Vitamin E (α-TE)"
        if canonical is None and re.match(r'(?i)^vitamin\s*e\b', cleaned):
            canonical = "Vitamin E"

        # Apply if found
        if canonical is not None and canonical != text:
            new_tok = dict(tok)
            new_tok["token"] = canonical
            result.append(new_tok)
        else:
            result.append(tok)

    return result