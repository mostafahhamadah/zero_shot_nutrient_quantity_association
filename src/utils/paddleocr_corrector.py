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
- Fix nutrient name misspellings (EiweiB, Hagnesium, …).
- Apply the trailing-digit artefact heuristic (C4/C5) — unacceptable
  false-positive rate on real labels.
- Touch tokens it cannot unambiguously correct.

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
    r'|ng'
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
    r'^(\d+[.,]?\d*)\s*(kJ|kcal|kca|KJ|KCAL|KCA)\s*/\s*(\d+[.,]?\d*)\s*(kJ|kcal|kca|KJ|KCAL|KCA)$',
    re.IGNORECASE,
)

# Pattern B: number+unit with trailing slash — "1383kJ/" or "420 kJ/"
_C13_UNIT_SLASH = re.compile(
    r'^(\d+[.,]?\d*)\s*(kJ|kcal|kca|KJ|KCAL|KCA)\s*/?\s*$',
    re.IGNORECASE,
)

# Pattern C: slash-separated bare numbers — "1696/404", "509/121"
# Only matches when BOTH numbers are ≥ 50 (energy values are never tiny)
_C13_SLASH_NUMS = re.compile(
    r'^(\d{2,}[.,]?\d*)\s*/\s*(\d{2,}[.,]?\d*)$',
)

# Pattern D (NEW): bare number/number+unit — "952/224kca", "942/255kJ/KCAL"
# First number has no unit, second has a unit
_C13_BARE_SLASH_UNIT = re.compile(
    r'^(\d+[.,]?\d*)\s*/\s*(\d+[.,]?\d*)\s*(kJ|kcal|kca|KJ|KCAL|KCA)(?:\s*/\s*(kJ|kcal|kca|KJ|KCAL|KCA))?$',
    re.IGNORECASE,
)

# Pattern E (NEW): missing separator — "223 kJ52kcal" or "438kJ103kcal"
_C13_NOSEP = re.compile(
    r'^(\d+[.,]?\d*)\s*(kJ|KJ|kj)\s*(\d+[.,]?\d*)\s*(kcal|KCAL|kca|KCA)$',
    re.IGNORECASE,
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

# ── C17 (NEW) — Scientific notation + unit ────────────────────────────
# Handles: "3.6x10KBE" → ["3.6 x 10^8", "KBE"]
#          "1x10^9CFU" → ["1x10^9", "CFU"]
_C17_SCIUNIT = re.compile(
    r'^(\d+[.,]?\d*\s*[xX×]\s*10[\^⁰¹²³⁴⁵⁶⁷⁸⁹\d]*)\s*'
    r'(KBE|kbe|CFU|cfu|IU|iu|IE|ie)$',
    re.IGNORECASE,
)


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


def _norm_energy_unit(u: str) -> str:
    """Normalize OCR-corrupted energy units: kca→kcal, KJ→kJ."""
    low = u.lower()
    if low in ('kca',):
        return 'kcal'
    if low == 'kj':
        return 'kJ'
    if low == 'kcal':
        return 'kcal'
    return u


def _apply_c13(tok: Dict, fired: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    C13 (exp31): split compound energy value tokens.

    PaddleOCR frequently fuses kJ/kcal pairs into single tokens:
      "1625 kJ/383kcal"  → ["1625", "kJ", "383", "kcal"]
      "1383kJ/"          → ["1383", "kJ"]
      "1696/404"         → ["1696", "404"]
      "952/224kca"       → ["952", "224", "kcal"]   (NEW: OCR kca→kcal)
      "223 kJ52kcal"     → ["223", "kJ", "52", "kcal"]  (NEW: missing separator)
    """
    text = tok['token']

    # Pattern A: full pair — "1625 kJ/383kcal"
    m = _C13_FULL_PAIR.match(text)
    if m:
        parts = [m.group(1), _norm_energy_unit(m.group(2)),
                 m.group(3), _norm_energy_unit(m.group(4))]
        return _split_bbox_multi(tok, parts), fired + ['C13_energy_full_pair']

    # Pattern E (NEW): missing separator — "223 kJ52kcal"
    m = _C13_NOSEP.match(text)
    if m:
        parts = [m.group(1), _norm_energy_unit(m.group(2)),
                 m.group(3), _norm_energy_unit(m.group(4))]
        return _split_bbox_multi(tok, parts), fired + ['C13_energy_nosep']

    # Pattern D (NEW): bare number/number+unit — "952/224kca"
    m = _C13_BARE_SLASH_UNIT.match(text)
    if m:
        parts = [m.group(1), m.group(2)]
        unit = _norm_energy_unit(m.group(3))
        parts.append(unit)
        if m.group(4):
            parts.append(_norm_energy_unit(m.group(4)))
        return _split_bbox_multi(tok, parts), fired + ['C13_energy_bare_slash']

    # Pattern B: number+unit+slash — "1383kJ/" or "420 kJ/"
    m = _C13_UNIT_SLASH.match(text)
    if m:
        parts = [m.group(1), _norm_energy_unit(m.group(2))]
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


def _apply_c17(tok: Dict, fired: List[str]) -> Tuple[List[Dict], List[str]]:
    """C17: split scientific notation + unit (e.g., 3.6x10KBE → [3.6x10, KBE])."""
    m = _C17_SCIUNIT.match(tok['token'])
    if not m:
        return [tok], fired
    qty_str = m.group(1).strip()
    unit_str = m.group(2).strip()
    ratio = len(qty_str) / max(len(tok['token']), 1)
    left, right = _split_bbox_h(tok, ratio)
    left['token'] = qty_str
    right['token'] = unit_str
    return [left, right], fired + ['C17_sci_unit_split']


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
    # Matches: optional paren or slash, digits, %, optional closing paren/asterisk/superscript
    # Now also handles: "100.8mg/126%" and "0.44mg(88%)" and "2.5µg80%"
    text_stripped = re.sub(
        r'[/\(]?\s*[\d.,]+\s*%\)?[\s*³²¹⁾\)\]]*$', '', text
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

            # C17 (NEW): scientific notation + unit split
            c17_result: List[Dict] = []
            for part in result_toks:
                split_parts, fired = _apply_c17(part, fired)
                c17_result.extend(split_parts)
            result_toks = c17_result

            # C2/C3: fused qty+unit split (applied to each part from C6/C13/C17)
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

    # ── C14b: cross-line continuation merge (exp40) ──────────────
    corrected = apply_c14_cross_line_merge(corrected)

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
    "davon", "- davon", "-davon",
    "of which", "of", "dont",
    "waarvan", "di cui", "de las cuales",
    "w tym",   # Polish
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

        # Collect up to 2 more tokens (e.g., "davon" + "gesättigte" + "Fettsäuren")
        for j in range(i + 1, min(i + 3, n)):
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
    "fettsauren", "fettsäuren", "saturates", "saturated",
    "fatty acids", "saturated fatty acids",
    "acides gras saturés", "verzadigde vetzuren",
    "acidi grassi saturi",
})

_C14_CROSS_LINE_MAX_CY = 50
_C14_CROSS_LINE_MAX_CX = 100


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
                           "of which", "dont", "waarvan", "di cui"))
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