"""
sentence_extractor.py — Sentence-Level Nutrient Extraction Fallback
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Extract nutrient tuples from running text (non-table images) using
regex patterns. This is a FALLBACK called when the graph-based pipeline
produces 0 tuples — e.g., image 67.jpeg which has paragraph text like:

  "Jede Tablette enthält 25 µg Colecalciferol, entsprechend 1000 I.E. Vitamin D₃"

Patterns handled:
  - "enthält X µg/mg/g Y" (German dosage statements)
  - "X I.E./IU Y" (International Units)  
  - "X µg/mg Y" (direct quantity-unit-nutrient)
  - "entsprechend X,XXX mg" (equivalence statements)
  - "Tagesdosis X I.E." (daily dose statements)
"""

from __future__ import annotations

import re
from typing import Dict, List

# Known nutrient names for sentence extraction
_NUTRIENT_NAMES = {
    "vitamin a", "vitamin b1", "vitamin b2", "vitamin b3", "vitamin b5",
    "vitamin b6", "vitamin b12", "vitamin c", "vitamin d", "vitamin d3",
    "vitamin e", "vitamin k", "vitamin k2",
    "colecalciferol", "cholecalciferol", "thiamin", "riboflavin",
    "niacin", "folsaure", "folsäure", "folic acid", "biotin",
    "pantothensaure", "pantothensäure", "pantothenic acid",
    "magnesium", "calcium", "kalzium", "kalium", "potassium",
    "zink", "zinc", "eisen", "iron", "selen", "selenium",
    "jod", "iodine", "kupfer", "copper", "mangan", "manganese",
    "chrom", "chromium", "molybdan", "molybdenum",
    "natrium", "sodium", "phosphor", "phosphorus",
    "omega-3", "omega 3", "epa", "dha",
    "kreatin", "creatine", "inositol", "choline", "cholin",
    "coenzym q10", "coenzyme q10", "q10",
    "l-carnitin", "l-carnitine", "l-arginin", "l-arginine",
    "glucosamin", "glucosamine", "msm",
}

_UNIT_PATTERN = r'(?:µg|ug|mcg|mg|g|kg|ml|I\.?E\.?|IU|IE|kcal|kJ)'


def extract_from_sentences(tokens: List[Dict], image_id: str = "unknown") -> List[Dict]:
    """
    Extract nutrient tuples from OCR tokens by joining them into text
    and scanning for quantity-unit-nutrient patterns.
    
    Args:
        tokens: OCR tokens with 'token' field
        image_id: image identifier for output tuples
        
    Returns:
        List of dicts with keys: image_id, nutrient, quantity, unit, context
    """
    # Join all tokens into a single text string
    text = " ".join(t.get("token", "") for t in tokens if t.get("token", "").strip())
    
    if not text.strip():
        return []
    
    tuples = []
    seen = set()
    
    # Pattern 1: "enthält X unit Nutrient" (German)
    # e.g. "enthält 25 µg Colecalciferol"
    for m in re.finditer(
        r'enth[aä]lt\s+(\d+[.,]?\d*)\s*(' + _UNIT_PATTERN + r')\s+([A-Za-zäöüßéè][\w\-]+)',
        text, re.IGNORECASE
    ):
        qty, unit, nutrient = m.group(1), m.group(2), m.group(3)
        _add_tuple(tuples, seen, image_id, nutrient, qty, unit)
    
    # Pattern 2: "X I.E. Nutrient" or "X IU Nutrient"
    # e.g. "1000 I.E. Vitamin D₃"
    for m in re.finditer(
        r'(\d+[.,]?\d*)\s*(I\.?E\.?|IU|IE)\s+([A-Za-zäöüßéè][\w\s\-]+?)(?:[.,\s]|$)',
        text, re.IGNORECASE
    ):
        qty, unit, nutrient = m.group(1), m.group(2), m.group(3).strip()
        _add_tuple(tuples, seen, image_id, nutrient, qty, _norm_unit(unit))
    
    # Pattern 3: "entsprechend X unit" (equivalence)
    # e.g. "entsprechend 0,025 mg"
    for m in re.finditer(
        r'entsprechend\s+(\d+[.,]?\d*)\s*(' + _UNIT_PATTERN + r')',
        text, re.IGNORECASE
    ):
        qty, unit = m.group(1), m.group(2)
        # Look for nutrient name nearby (within 50 chars before)
        start = max(0, m.start() - 80)
        context_text = text[start:m.start()].lower()
        for name in _NUTRIENT_NAMES:
            if name in context_text:
                _add_tuple(tuples, seen, image_id, name.title(), qty, unit)
                break
    
    # Pattern 4: "Tagesdosis X unit" or "Tagesdosis bis zu X unit"
    for m in re.finditer(
        r'[Tt]agesdosis\s+(?:bis\s+zu\s+)?(\d+[.,]?\d*)\s*(' + _UNIT_PATTERN + r')',
        text, re.IGNORECASE
    ):
        qty, unit = m.group(1), m.group(2)
        # Look for nutrient name within 80 chars
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        context_text = text[start:end].lower()
        for name in _NUTRIENT_NAMES:
            if name in context_text:
                _add_tuple(tuples, seen, image_id, name.title(), qty, _norm_unit(unit), "per_daily_dose")
                break
    
    # Pattern 5: Direct "Nutrient X unit" for known nutrients
    # e.g. "Vitamin D3 25 µg"
    for name in _NUTRIENT_NAMES:
        pattern = re.escape(name) + r'\s+(\d+[.,]?\d*)\s*(' + _UNIT_PATTERN + r')'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            qty, unit = m.group(1), m.group(2)
            _add_tuple(tuples, seen, image_id, name.title(), qty, unit)
    
    return tuples


def _add_tuple(tuples, seen, image_id, nutrient, qty, unit, context="per_daily_dose"):
    """Add a tuple if not already seen (dedup by nutrient+qty)."""
    qty = qty.replace(",", ".")
    unit = _norm_unit(unit)
    key = (nutrient.lower(), qty, unit)
    if key not in seen:
        seen.add(key)
        tuples.append({
            "image_id": image_id,
            "nutrient": nutrient,
            "quantity": qty,
            "unit": unit,
            "context": context,
        })


def _norm_unit(unit: str) -> str:
    """Normalize unit strings."""
    u = unit.strip().lower().replace(".", "")
    if u in ("ie", "iu"):
        return "IE"
    if u in ("µg", "ug", "mcg"):
        return "µg"
    return unit.strip()