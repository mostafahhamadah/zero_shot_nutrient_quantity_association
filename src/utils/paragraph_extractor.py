"""
paragraph_extractor.py — Extract tuples from paragraph-style fused OCR text.

Some supplement labels use paragraph layout instead of tables. PaddleOCR reads
entire lines as single tokens like:
  "Salz/Salt15.0g/0.63g/1.26 g.Mineralstoffe/Minerals:Natrium/Sodium 5962 mg/200 mg"

This module extracts (nutrient, quantity, unit) triples using regex patterns.
Activated as a fallback when the normal pipeline produces very few tuples
relative to the quantity tokens detected.

Two extraction families live here:

  1. TABLE-PROSE  (`_extract_table_prose`, original behaviour)
     High-density fused nutrition tables where a *nutrient name precedes* its
     quantity/unit (e.g. image 15). Pattern: NUTRIENT -> QTY -> UNIT.

  2. PHARMA-PROSE (`_extract_pharma`, added for image 67 / VIGANTOL)
     Sparse pharmaceutical labels with a single active ingredient stated in
     running prose, where the *quantity/unit precedes* the nutrient and two
     dose-forms are linked by "entsprechend":
       "Jede Tablette enthaelt 25 ug Colecalciferol, entsprechend 1000 I.E. Vitamin D3."
       "Tagesdosis bis zu 1000 I.E. (entsprechend 0,025 mg)."
     Pattern: QTY -> UNIT -> (optional NUTRIENT). Context comes from prose
     verbs ("Jede Tablette" -> per_serving, "Tagesdosis" -> per_daily_dose).
"""

import re
from typing import List, Dict, Optional

# ======================================================================
# FAMILY 1 — TABLE-PROSE (original)
# ======================================================================

# Known nutrient names for paragraph extraction (multilingual)
_PARAGRAPH_NUTRIENTS = [
    # German / English pairs
    r"Energie/Energy|Energy/Energie|Energie|Energy",
    r"Brennwert",
    r"Fett/Fat|Fat/Fett|Fett|Fat",
    r"davon\s*ges[aä]ttigte\s*Fetts[aä]uren|of\s*which\s*saturates",
    r"Kohlenhydrate/Carbohydrate|Carbohydrate|Kohlenhydrate",
    r"davon\s*Zucker/of\s*which\s*sugars|davon\s*Zucker|of\s*which\s*sugars|Zucker|sugars",
    r"Ballaststoffe|Fibre|Fiber",
    r"Eiwei[sß]/Protein|Protein/Eiwei[sß]|Eiwei[sß]|Protein",
    r"Salz/Salt|Salt/Salz|Salz|Salt",
    r"Natrium/Sodium|Sodium/Natrium|Natrium|Sodium",
    r"Kalium/Potassium|Potassium/Kalium|Kalium|Potassium",
    r"Chlorid(?:e)?",
    r"Magnesium",
    r"Calcium",
    r"Koffein/Caf[ef]in|Koffein|Caffeine|Cafein",
    r"Vitamin\s*[A-Z]\d*",
    r"Zink/Zinc|Zink|Zinc",
    r"Eisen/Iron|Eisen|Iron",
    r"Jod/Iodine|Jod|Iodine",
    r"Selen/Selenium|Selen|Selenium",
]

# Build combined pattern: nutrient name followed by quantity+unit
_NUTRIENT_PATTERN = '|'.join(f'(?:{p})' for p in _PARAGRAPH_NUTRIENTS)

# Match: NutrientName + one or more qty+unit groups separated by / or spaces
_EXTRACT_RE = re.compile(
    rf'({_NUTRIENT_PATTERN})'           # Group 1: nutrient name
    r'[\s:./]*'                          # separator noise
    r'(\d+[.,]?\d*)\s*'                 # Group 2: first quantity
    r'(mg|g|kg|µg|mcg|kJ|kcal|ml|IE|IU)',  # Group 3: first unit
    re.IGNORECASE,
)

# Match additional qty+unit pairs after the first one (for multi-column values)
_ADDITIONAL_QTY_RE = re.compile(
    r'[/\s]*(\d+[.,]?\d*)\s*(mg|g|kg|µg|mcg|kJ|kcal|ml|IE|IU)',
    re.IGNORECASE,
)


# ======================================================================
# FAMILY 2 — PHARMA-PROSE (single active ingredient, qty-before-nutrient)
# ======================================================================

# Active-ingredient names found in single-substance pharma labels.
# Order matters: longer / more specific alternatives first.
_PHARMA_NUTRIENT_RE = re.compile(
    r'(Colecalciferol|Cholecalciferol'
    r'|Cyanocobalamin|Methylcobalamin'
    r'|Vitamin\s*[A-EK]\s*[0-9₀-₉]?'      # Vitamin D3 / D / B12 / K2 ...
    r'|Vitamin\s*B\s*1?[0-9]?)',
    re.IGNORECASE,
)

# A dose unit (mass or activity). 'I.E.' = German IU.  Tolerate OCR variants
# (ug for µg). Order longer-before-shorter so 'g' never steals 'mg'/'µg'.
_PHARMA_UNIT = r'(?:µg|ug|mcg|mg|kg|g|I\.?\s?E\.?|IU)'

# A single dose token: quantity + unit.
_PHARMA_DOSE_RE = re.compile(r'(\d+[.,]?\d*)\s*(' + _PHARMA_UNIT + r')', re.IGNORECASE)

# Context cues (per-line / per-segment).
_CTX_DAILY_RE = re.compile(
    r'tagesdosis|tagesration|tages\s*dosis|t[äa]gliche?r?\s+(?:verzehr|dosis|menge)'
    r'|daily\s*dose|per\s*day|pro\s*tag',
    re.IGNORECASE,
)
_CTX_SERVING_RE = re.compile(
    r'(?:tablette|kapsel|dragee|kaudragee|portion)\s+enth[aä]lt'
    r'|(?:pro|je|jede[rs]?|per)\s+(?:tablette|kapsel|dragee|portion)'
    r'|jede\s+tablette|jede\s+kapsel',
    re.IGNORECASE,
)


def _norm_pharma_unit(u: str) -> str:
    """Normalise a pharma unit to its canonical ground-truth spelling."""
    s = u.strip().lower().replace(' ', '')
    if s in ('i.e.', 'i.e', 'ie', 'ie.', 'iu'):
        return 'I.E.'
    if s in ('µg', 'ug', 'mcg'):
        return 'µg'
    if s == 'mg':
        return 'mg'
    if s == 'kg':
        return 'kg'
    if s == 'g':
        return 'g'
    return u.strip()


def _unit_class(canon_unit: str) -> str:
    """Group units so a bare dose can inherit a name learnt from any sibling."""
    return 'activity' if canon_unit == 'I.E.' else 'mass'


def _canon_pharma_nutrient(name: str, is_d3_label: bool = False) -> str:
    """Canonicalise an active-ingredient name to its ground-truth form."""
    n = ' '.join(name.split())
    low = n.lower().replace('₃', '3').replace('₂', '2').replace('₁', '1')
    if 'colecalciferol' in low or 'cholecalciferol' in low:
        return 'Colecalciferol'
    if low.startswith('vitamin'):
        m = re.search(r'vitamin\s*([a-ek])\s*([0-9]+)?', low)
        if m:
            letter = m.group(1).upper()
            num = m.group(2) or ''
            # colecalciferol == vitamin D3: if the label names colecalciferol,
            # treat a bare "Vitamin D" as D3 even when OCR dropped the subscript.
            if letter == 'D' and not num and is_d3_label:
                num = '3'
            return f'Vitamin {letter}{num}'.rstrip()
    return n


def _segment_token(text: str):
    """
    Split one OCR line into (subtext, context) parts.

    Robust to two OCR shapes:
      * one line per sentence  -> each line gets its own context
      * everything fused        -> split at the daily-dose cue so the
                                    per-serving clause and per-daily clause
                                    are separated.
    """
    dm = _CTX_DAILY_RE.search(text)
    if dm:
        before, after = text[:dm.start()], text[dm.start():]
        segs = []
        if _CTX_SERVING_RE.search(before) and re.search(r'\d', before):
            segs.append((before, 'per_serving'))
        segs.append((after, 'per_daily_dose'))
        return segs
    if _CTX_SERVING_RE.search(text):
        return [(text, 'per_serving')]
    return [(text, None)]


def _extract_pharma(tokens: List[Dict], image_id: str) -> List[Dict]:
    """
    Extract tuples from single-active-ingredient pharmaceutical prose.

    Strategy:
      Pass A: learn a unit-class -> nutrient-name map from explicit
              "QTY UNIT NAME" clauses (e.g. "25 ug Colecalciferol").
      Pass B: per line/segment, resolve a context, extract every dose, and
              assign each dose the adjacent name if present else the learnt
              name for its unit class.
    """
    full_text = ' '.join(t.get('token', '') for t in tokens)
    low_all = full_text.lower()

    # Only engage on genuinely pharma-prose text.
    if not (re.search(r'colecalciferol|cholecalciferol', low_all)
            or _CTX_DAILY_RE.search(full_text)
            or re.search(r'(?:tablette|kapsel|dragee)\s+enth[aä]lt', low_all)):
        return []

    is_d3_label = bool(re.search(r'colecalciferol|cholecalciferol', low_all))

    # ---- Pass A: learn unit-class -> name ----
    class_to_name: Dict[str, str] = {}
    for t in tokens:
        txt = t.get('token', '')
        for m in _PHARMA_DOSE_RE.finditer(txt):
            unit = _norm_pharma_unit(m.group(2))
            tail = txt[m.end():m.end() + 30]
            nm = _PHARMA_NUTRIENT_RE.search(tail)
            if nm and nm.start() <= 18:
                cls = _unit_class(unit)
                class_to_name.setdefault(cls, _canon_pharma_nutrient(nm.group(1), is_d3_label))

    # ---- Pass B: emit doses with resolved context + nutrient ----
    tuples: List[Dict] = []
    seen = set()
    for t in tokens:
        txt = t.get('token', '')
        for seg_text, ctx in _segment_token(txt):
            if ctx is None:
                continue  # no context cue -> skip to avoid spurious tuples
            for m in _PHARMA_DOSE_RE.finditer(seg_text):
                qty = m.group(1).replace(',', '.')
                unit = _norm_pharma_unit(m.group(2))
                tail = seg_text[m.end():m.end() + 30]
                nm = _PHARMA_NUTRIENT_RE.search(tail)
                if nm and nm.start() <= 18:
                    nutrient = _canon_pharma_nutrient(nm.group(1), is_d3_label)
                else:
                    nutrient = class_to_name.get(_unit_class(unit))
                if not nutrient:
                    continue
                key = (nutrient, qty, unit, ctx)
                if key in seen:
                    continue
                seen.add(key)
                tuples.append({
                    'image_id': image_id,
                    'nutrient': nutrient,
                    'quantity': qty,
                    'unit': unit,
                    'context': ctx,
                })
    return tuples


# ======================================================================
# TABLE-PROSE extraction (original logic, refactored into a helper)
# ======================================================================

def _extract_table_prose(tokens: List[Dict], image_id: str,
                         contexts: List[str] = None) -> List[Dict]:
    """Original nutrient-before-quantity fused-table extraction."""
    full_text = ' '.join(t.get('token', '') for t in tokens)

    if not contexts:
        contexts = _detect_contexts(full_text)

    tuples = []
    seen = set()

    for m in _EXTRACT_RE.finditer(full_text):
        nutrient = m.group(1).strip()
        first_qty = m.group(2).replace(',', '.')
        first_unit = m.group(3).lower()

        qty_unit_pairs = [(first_qty, first_unit)]

        remaining = full_text[m.end():]
        search_window = remaining[:80]
        for am in _ADDITIONAL_QTY_RE.finditer(search_window):
            qty = am.group(1).replace(',', '.')
            unit = am.group(2).lower()
            if re.search(_NUTRIENT_PATTERN, search_window[:am.start()], re.IGNORECASE):
                break
            qty_unit_pairs.append((qty, unit))
            if len(qty_unit_pairs) >= len(contexts) or len(qty_unit_pairs) >= 3:
                break

        for i, (qty, unit) in enumerate(qty_unit_pairs):
            ctx = contexts[i] if i < len(contexts) else contexts[0] if contexts else None
            key = (nutrient, qty, unit)
            if key in seen:
                continue
            seen.add(key)
            tuples.append({
                'image_id': image_id,
                'nutrient': nutrient,
                'quantity': qty,
                'unit': unit,
                'context': ctx,
            })

    return tuples


# ======================================================================
# PUBLIC ENTRY POINT
# ======================================================================

def extract_from_paragraph(tokens: List[Dict], image_id: str,
                           contexts: List[str] = None) -> List[Dict]:
    """
    Extract tuples from paragraph-style fused OCR tokens.

    Runs both the table-prose extractor (nutrient->qty) and the pharma-prose
    extractor (qty->nutrient) and returns the deduplicated union. The pharma
    extractor self-gates on pharma markers, so it returns [] for ordinary
    fused tables.

    Args:
        tokens: classified OCR tokens (from Stage 3)
        image_id: image filename
        contexts: list of context strings in column order
                  (e.g., ["per_100g", "per_serving", "per_daily_dose"])

    Returns:
        List of tuple dicts: image_id, nutrient, quantity, unit, context
    """
    tuples = _extract_table_prose(tokens, image_id, contexts)

    seen = {(t['nutrient'], t['quantity'], t['unit'], t.get('context')) for t in tuples}
    for t in _extract_pharma(tokens, image_id):
        key = (t['nutrient'], t['quantity'], t['unit'], t.get('context'))
        if key not in seen:
            seen.add(key)
            tuples.append(t)

    return tuples


def _detect_contexts(text: str) -> List[str]:
    """Detect context headers from paragraph text."""
    contexts = []
    text_lower = text.lower()

    # Look for "per/pro 100g" pattern
    if re.search(r'(per|pro|je)\s*(/per\s*)?100\s*(g|ml)', text_lower):
        contexts.append('per_100g')
    elif '100 g' in text_lower or '100g' in text_lower:
        contexts.append('per_100g')

    # Look for non-100g amounts in header: "42 g" "21.42 g" "60ml" etc.
    header_amts = re.findall(r'(?:per|pro|je|/)\s*(\d+[.,]?\d*)\s*(g|ml)', text_lower)
    for amt, unit in header_amts:
        if amt not in ('100',):
            if 'per_serving' not in contexts:
                contexts.append('per_serving')
            if len(contexts) >= 2 and 'per_daily_dose' not in contexts:
                contexts.append('per_daily_dose')
                break

    # Look for daily dose
    if re.search(r'(tagesdosis|daily\s*dose|tagesration)', text_lower):
        if 'per_daily_dose' not in contexts:
            contexts.append('per_daily_dose')

    if not contexts:
        contexts = ['per_100g']

    return contexts


# ======================================================================
# ACTIVATION GATE
# ======================================================================

# Pharma-prose markers (must be specific to avoid firing on tables that merely
# mention "Tablette"/"Kapsel" in a serving-size descriptor).
_PHARMA_GATE_RE = re.compile(
    r'colecalciferol|cholecalciferol'
    r'|(?:tablette|kapsel|dragee)\s+enth[aä]lt'
    r'|tagesdosis|tagesration',
    re.IGNORECASE,
)
_PHARMA_DOSE_GATE_RE = re.compile(
    r'\d+[.,]?\d*\s*(?:µg|ug|mcg|mg|kg|g|I\.?\s?E\.?|IU)', re.IGNORECASE)


def should_use_paragraph_mode(tokens: List[Dict], normal_tuple_count: int) -> bool:
    """
    Determine if paragraph mode should be activated.

    Two independent triggers:

      A) High-density fused TABLE (original):
           - normal pipeline produced < 8 tuples
           - >= 2 long tokens (>60 chars)
           - >= 8 quantity patterns in the fused text

      B) Sparse PHARMA prose (single active ingredient, e.g. VIGANTOL):
           - normal pipeline produced < 4 tuples
           - specific pharma marker present (colecalciferol / "Tablette
             enthaelt" / "Tagesdosis")
           - >= 2 dose patterns (counting I.E., which branch A misses)
    """
    full_text = ' '.join(t.get('token', '') for t in tokens)

    # --- Trigger A: high-density fused table ---
    if normal_tuple_count < 8:
        long_tokens = [t for t in tokens if len(t.get('token', '')) > 60]
        qty_count = len(re.findall(
            r'\d+[.,]?\d*\s*(?:mg|g|kJ|kcal|ml|µg)', full_text, re.IGNORECASE))
        if len(long_tokens) >= 2 and qty_count >= 8:
            return True

    # --- Trigger B: sparse pharma prose ---
    if normal_tuple_count < 4:
        if (_PHARMA_GATE_RE.search(full_text)
                and len(_PHARMA_DOSE_GATE_RE.findall(full_text)) >= 2):
            return True

    return False


if __name__ == '__main__':
    # ---- Test 1: original fused-table case (image 15) ----
    test_text = [
        {'token': 'Nahrwertangaben/Nutrition Information pro/per 100 g/42 g/21.42 g'},
        {'token': 'Salz/Salt15.0g/0.63g/1.26 g.Mineralstoffe/Minerals:Natrium/Sodium 5962 mg/200 mg/500mg.Kalium'},
        {'token': 'sium 3571mg/150mg7,5%*/300mg15%*)Chloride4762mg/200mg25%/400 mg 50%,Calcium 1428'},
        {'token': 'mg/60mg7,5%*/120mg15%*)Magnesium670 mg/28,1 mg 7,5%/56,3 mg15%Koffein/Cafen 1786'},
    ]
    print("=== TABLE-PROSE test (image 15) ===")
    print(f"Should activate: {should_use_paragraph_mode(test_text, 3)}")
    for t in extract_from_paragraph(test_text, '15.png'):
        print(f"  {t['nutrient']:<30} {t['quantity']:>8} {t['unit']:<5} {t['context']}")

    # ---- Test 2: pharma prose, line-token shape (image 67 / VIGANTOL) ----
    vigantol_lines = [
        {'token': 'VIGANTOL'},
        {'token': '1000 I.E. Vitamin D3 Tabletten'},
        {'token': 'Für Säuglinge, Kinder und Erwachsene. Zum Einnehmen.'},
        {'token': 'Jede Tablette enthält 25 µg Colecalciferol, entsprechend 1000 I.E. Vitamin D3.'},
        {'token': 'Enthält Sucrose (Zucker). Tagesdosis bis zu 1000 I.E. (entsprechend 0,025 mg).'},
        {'token': 'Nicht über 25 °C lagern. Im Umkarton aufbewahren,'},
        {'token': 'Apothekenpflichtig. Zul.-Nr. 6154298.01.00'},
    ]
    print("\n=== PHARMA-PROSE test (image 67, line tokens) ===")
    print(f"Should activate: {should_use_paragraph_mode(vigantol_lines, 0)}")
    got = extract_from_paragraph(vigantol_lines, '67.jpeg')
    for t in got:
        print(f"  {t['nutrient']:<16} {t['quantity']:>7} {t['unit']:<6} {t['context']}")

    expected = {
        ('Colecalciferol', '25', 'µg', 'per_serving'),
        ('Vitamin D3', '1000', 'I.E.', 'per_serving'),
        ('Colecalciferol', '0.025', 'mg', 'per_daily_dose'),
        ('Vitamin D3', '1000', 'I.E.', 'per_daily_dose'),
    }
    got_set = {(t['nutrient'], t['quantity'], t['unit'], t['context']) for t in got}
    print(f"  -> {len(got_set & expected)}/4 GT tuples matched; spurious: {got_set - expected}")

    # ---- Test 3: pharma prose, FULLY-FUSED single token (worst-case OCR) ----
    vigantol_fused = [{'token':
        'Jede Tablette enthält 25 µg Colecalciferol, entsprechend 1000 I.E. '
        'Vitamin D3. Enthält Sucrose (Zucker). Tagesdosis bis zu 1000 I.E. '
        '(entsprechend 0,025 mg). Nicht über 25 °C lagern.'}]
    print("\n=== PHARMA-PROSE test (image 67, fully fused token) ===")
    print(f"Should activate: {should_use_paragraph_mode(vigantol_fused, 0)}")
    got2 = extract_from_paragraph(vigantol_fused, '67.jpeg')
    for t in got2:
        print(f"  {t['nutrient']:<16} {t['quantity']:>7} {t['unit']:<6} {t['context']}")
    got2_set = {(t['nutrient'], t['quantity'], t['unit'], t['context']) for t in got2}
    print(f"  -> {len(got2_set & expected)}/4 GT tuples matched; spurious: {got2_set - expected}")