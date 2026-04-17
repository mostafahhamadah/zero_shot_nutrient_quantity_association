"""
paragraph_extractor.py — Extract tuples from paragraph-style fused OCR text.

Some supplement labels use paragraph layout instead of tables. PaddleOCR reads
entire lines as single tokens like:
  "Salz/Salt15.0g/0.63g/1.26 g.Mineralstoffe/Minerals:Natrium/Sodium 5962 mg/200 mg"

This module extracts (nutrient, quantity, unit) triples using regex patterns.
Activated as a fallback when the normal pipeline produces very few tuples
relative to the quantity tokens detected.
"""

import re
from typing import List, Dict, Optional

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


def extract_from_paragraph(tokens: List[Dict], image_id: str,
                           contexts: List[str] = None) -> List[Dict]:
    """
    Extract tuples from paragraph-style fused OCR tokens.

    Args:
        tokens: classified OCR tokens (from Stage 3)
        image_id: image filename
        contexts: list of context strings in column order
                  (e.g., ["per_100g", "per_serving", "per_daily_dose"])

    Returns:
        List of tuple dicts: image_id, nutrient, quantity, unit, context
    """
    # Concatenate all token text
    full_text = ' '.join(t.get('token', '') for t in tokens)

    # Try to detect contexts from the text
    if not contexts:
        contexts = _detect_contexts(full_text)

    tuples = []
    seen = set()

    # Find all nutrient mentions
    for m in _EXTRACT_RE.finditer(full_text):
        nutrient = m.group(1).strip()
        first_qty = m.group(2).replace(',', '.')
        first_unit = m.group(3).lower()

        # Collect all qty+unit pairs following this nutrient
        qty_unit_pairs = [(first_qty, first_unit)]

        # Search for additional values after the first match
        remaining = full_text[m.end():]
        # Only look at the next ~50 chars for additional values
        search_window = remaining[:80]
        for am in _ADDITIONAL_QTY_RE.finditer(search_window):
            qty = am.group(1).replace(',', '.')
            unit = am.group(2).lower()
            # Stop if we hit another nutrient name
            if re.search(_NUTRIENT_PATTERN, search_window[:am.start()], re.IGNORECASE):
                break
            qty_unit_pairs.append((qty, unit))
            if len(qty_unit_pairs) >= len(contexts) or len(qty_unit_pairs) >= 3:
                break

        # Create tuples — one per context column
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


def should_use_paragraph_mode(tokens: List[Dict], normal_tuple_count: int) -> bool:
    """
    Determine if paragraph mode should be activated.

    Triggers when:
      - Normal pipeline produces very few tuples (< 8)
      - AND there are long tokens (>60 chars)
      - AND many quantity patterns exist in the fused text
    """
    if normal_tuple_count >= 8:
        return False

    long_tokens = [t for t in tokens if len(t.get('token', '')) > 60]
    if len(long_tokens) < 2:
        return False

    # Count quantity patterns in the fused text
    full_text = ' '.join(t.get('token', '') for t in tokens)
    qty_count = len(re.findall(r'\d+[.,]?\d*\s*(?:mg|g|kJ|kcal|ml|µg)', full_text, re.IGNORECASE))

    return qty_count >= 8


if __name__ == '__main__':
    # Quick test
    test_text = [
        {'token': 'Nahrwertangaben/Nutrition Information pro/per 100 g/42 g/21.42 g'},
        {'token': 'Salz/Salt15.0g/0.63g/1.26 g.Mineralstoffe/Minerals:Natrium/Sodium 5962 mg/200 mg/500mg.Kalium'},
        {'token': 'sium 3571mg/150mg7,5%*/300mg15%*)Chloride4762mg/200mg25%/400 mg 50%,Calcium 1428'},
        {'token': 'mg/60mg7,5%*/120mg15%*)Magnesium670 mg/28,1 mg 7,5%/56,3 mg15%Koffein/Cafen 1786'},
    ]

    print("Paragraph extraction test:")
    print(f"Should activate: {should_use_paragraph_mode(test_text, 3)}")
    tuples = extract_from_paragraph(test_text, '15.png')
    for t in tuples:
        print(f"  {t['nutrient']:<30} {t['quantity']:>8} {t['unit']:<5} {t['context']}")