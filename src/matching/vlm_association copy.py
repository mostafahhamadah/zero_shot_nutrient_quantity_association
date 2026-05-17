"""
vlm_association.py
==================
Stage 5-VLM — Vision-Language Model Tuple Association
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

Supports three backends:

  1. LM Studio / any OpenAI-compatible local server (DEFAULT)
     backend = "openai_compat"
     endpoint = http://127.0.0.1:1234/v1/chat/completions

  2. Hugging Face Inference Providers OpenAI-compatible router
     backend = "hf"
     endpoint = https://router.huggingface.co/v1/chat/completions

  3. Ollama local API (legacy)
     backend = "ollama"
     endpoint = http://localhost:11434/api/chat

Active Layer 1 fixes (vs vlm_exp01):
  - F1.1 NRV tokens dropped from prompt (output schema is 4 fields).
  - F1.2 Token table metadata stripped to essentials by default.
  - F1.3 Images resized to max_side=896px before base64 encoding.
  - F1.4 (Ollama only) num_ctx=8192 sent in options.

Layer 2 truncation fixes:
  - F2.1 max_tokens raised 768 -> 4096.

Layer 3 robustness:
  - F3.1 Optional retry-on-fail with progressive image shrink
        (896 -> 768 -> 640 -> 512). Enable via retry_max > 0.

Layer 4 safe additions in this version:
  - F4.1 Partial JSON salvage from truncated VLM responses.
  - F4.2 Conservative post-processing of VLM tuples:
        * quantity cleanup
        * unit cleanup
        * nutrient cleanup
        * context canonicalization
  - F4.3 No invalid-context dropping.
  - F4.4 No default chunking.

Output contract:
  List[Dict] with keys:
      image_id, nutrient, quantity, unit, context
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import mimetypes
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEFAULT CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CONFIG = {
    # ── Backend selection ────────────────────────────────────────────
    "backend": "openai_compat",

    # ── OpenAI-compatible endpoint (LM Studio by default) ────────────
    "openai_base_url": "http://127.0.0.1:1234/v1/chat/completions",
    "openai_model": "google/gemma-3-4b",
    "openai_auth_env": None,

    # ── HuggingFace Inference Providers (fallback / cloud option) ────
    "hf_base_url": "https://router.huggingface.co/v1/chat/completions",
    "hf_model": "zai-org/GLM-4.5V:fastest",
    "hf_token_env": "HF_TOKEN",

    # ── Ollama (legacy) ──────────────────────────────────────────────
    "ollama_base_url": "http://localhost:11434",
    "model": "google/gemma-3-4b",
    "ollama_num_ctx": 8192,

    # ── Generation ───────────────────────────────────────────────────
    "temperature": 0.1,
    "max_tokens": 4096,
    "num_predict": None,
    "timeout_s": 600,
    "retry_max": 0,
    "image_shrink_on_retry": True,

    # ── Image preprocessing (F1.3) ───────────────────────────────────
    "image_max_side": 896,
    "image_jpeg_quality": 90,

    # ── Token-table content (F1.1, F1.2) ─────────────────────────────
    "allowed_labels": {"NUTRIENT", "QUANTITY", "UNIT", "CONTEXT"},
    "include_noise": False,
    "include_geometry": True,
    "include_structure": False,

    # ── Safe output post-processing ──────────────────────────────────
    "postprocess_output": True,
    "canonicalize_context": True,

    # Important:
    # Do NOT drop tuples only because the context is raw/unknown.
    # This preserves recall and lets the evaluator decide.
    "drop_invalid_contexts": False,

    # Important:
    # This weak baseline does not use chunking.
    # Chunking changed model behavior and hurt global metrics.
    "use_chunking": False,

    # ── Debugging ────────────────────────────────────────────────────
    "save_prompts": False,
    "prompt_debug_dir": "outputs/debug_vlm_prompts",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM_PROMPT = """\
You are a nutrition-label tuple association expert.

You will receive:
  1. An image of a nutritional supplement label.
  2. A structured table of OCR tokens that have already been classified
     as NUTRIENT, QUANTITY, UNIT, or CONTEXT, with coordinates and
     optional structural metadata.

YOUR TASK:
  Use the provided token table as the ONLY text source.
  Associate existing NUTRIENT tokens with existing QUANTITY, UNIT, and CONTEXT
  tokens to produce structured nutrition tuples.

IMPORTANT:
  The image is NOT a source for reading new text.
  Use the image only to verify the visual layout and spatial relationships
  between tokens.

LAYOUT PRINCIPLE:
  The label may be a regular table, a multi-column table, or a free-form layout.
  Do NOT assume one fixed structure.
  Use row alignment, column alignment, proximity, grouping, headers, indentation,
  and visual layout as evidence, not as absolute rules.

RULES:
  - Use ONLY tokens that appear in the token table.
  - Do NOT invent nutrient names, quantities, units, or contexts.
  - Do NOT read new text from the image.
  - Match each NUTRIENT with the most plausible QUANTITY and UNIT using:
      * token coordinates,
      * visual proximity,
      * row or line alignment,
      * column or block membership,
      * context/header information,
      * structural metadata when available.
  - Same-row alignment is strong evidence, especially in tables, but it is not
    mandatory if the visual layout clearly uses another structure.
  - Column alignment is strong evidence in multi-column tables, but it is not
    mandatory for free-form layouts.
  - UNIT should be selected from an existing UNIT token that visually belongs
    to the chosen QUANTITY.
  - CONTEXT should come from an existing CONTEXT token or from the context
    metadata in the token table.
  - Context may be above a column, near a dosage group, in a section header,
    or encoded in the token metadata.
  - If a label has multiple dosage/context groups, produce one tuple per
    nutrient per valid quantity/context group.
  - If a nutrient has no plausible matching QUANTITY token, skip it.
  - If a quantity has no plausible matching NUTRIENT token, skip it.
  - Ignore obvious headers or fragments even if classified as NUTRIENT, such as:
    "nährstoffe", "nahrstoffe", "nahr", "nährwerte", "nutrition facts",
    "supplement facts", "ingredients", "zutaten".
  - Preserve nutrient names exactly as they appear in the token table.
  - Quantity must be numeric only, e.g. "400", "0.8", "1.5".
  - Unit must be the measurement unit only, e.g. "mg", "µg", "g", "kJ", "kcal".
  - Context must be one of:
      per_100g, per_100ml, per_serving, per_daily_dose
    or the exact context string from the token table.

OUTPUT FORMAT:
  Return ONLY a valid JSON array.
  No markdown fences.
  No explanation.
  No comments.

Each object must have exactly these keys:
  {"nutrient": "...", "quantity": "...", "unit": "...", "context": "..."}

Example:
[
  {"nutrient": "Magnesium", "quantity": "400", "unit": "mg", "context": "per_daily_dose"},
  {"nutrient": "Vitamin C", "quantity": "80", "unit": "mg", "context": "per_daily_dose"}
]
"""



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPT / TOKEN-TABLE BUILDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_token_table(enriched_tokens: List[Dict], config: Dict) -> str:
    """
    Format enriched tokens into a structured text table for the VLM prompt.

    Columns:
      Always:        idx, token, label
      If geometry:   x1, y1, x2, y2
      If structure:  row, col, col_role, context, stream
    """
    include_noise = config.get("include_noise", False)
    include_geo = config.get("include_geometry", True)
    include_str = config.get("include_structure", False)

    allowed_labels = config.get(
        "allowed_labels",
        {"NUTRIENT", "QUANTITY", "UNIT", "CONTEXT"},
    )

    active = []
    for i, tok in enumerate(enriched_tokens):
        label = tok.get("label", "UNKNOWN")

        if not include_noise and label == "NOISE":
            continue
        if allowed_labels and label not in allowed_labels:
            continue

        active.append((i, tok))

    if not active:
        return "(No relevant classified tokens found.)"

    cols = ["idx", "token", "label"]

    if include_geo:
        cols += ["x1", "y1", "x2", "y2"]
    if include_str:
        cols += ["row", "col", "col_role", "context", "stream"]

    lines = [" | ".join(cols)]
    lines.append("-" * len(lines[0]))

    for i, tok in active:
        token_text = str(tok.get("token", "") or "").strip()
        norm = str(tok.get("norm", "") or "").strip()
        label = tok.get("label", "UNKNOWN")

        # Preserve weak baseline behavior:
        # use norm if available, otherwise raw token text.
        display = norm if norm else token_text

        row = [str(i), display, label]

        if include_geo:
            row += [
                str(int(tok.get("x1", 0) or 0)),
                str(int(tok.get("y1", 0) or 0)),
                str(int(tok.get("x2", 0) or 0)),
                str(int(tok.get("y2", 0) or 0)),
            ]

        if include_str:
            row += [
                str(tok.get("row_id", -1)),
                str(tok.get("column_id", -1)),
                str(tok.get("column_role", "?") or "?"),
                str(tok.get("column_context_id", "") or ""),
                str(tok.get("dosage_stream_id", -1)),
            ]

        lines.append(" | ".join(row))

    return "\n".join(lines)


def _build_user_prompt(token_table: str, image_id: str) -> str:
    return (
        f"Image: {image_id}\n\n"
        f"TOKEN TABLE — USE THIS AS THE ONLY TEXT SOURCE:\n"
        f"```\n{token_table}\n```\n\n"
        f"Task:\n"
        f"Associate existing NUTRIENT tokens with existing QUANTITY, UNIT, "
        f"and CONTEXT tokens.\n\n"
        f"The label may be a table, multi-column layout, or free-form layout. "
        f"Do not force one layout pattern. Use the image only to verify spatial "
        f"relationships between the provided tokens.\n\n"
        f"Return ONLY a valid JSON array with objects of this exact form:\n"
        f"[{{\"nutrient\":\"...\",\"quantity\":\"...\",\"unit\":\"...\",\"context\":\"...\"}}]"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMAGE PREPROCESSING (F1.3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _prepare_image_bytes(
    image_path: str,
    max_side: int = 896,
    jpeg_quality: int = 90,
) -> bytes:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return buf.getvalue()


def _encode_image_base64(
    image_path: str,
    max_side: int = 896,
    jpeg_quality: int = 90,
) -> str:
    raw = _prepare_image_bytes(image_path, max_side=max_side, jpeg_quality=jpeg_quality)
    return base64.b64encode(raw).decode("utf-8")


def _encode_image_data_uri(
    image_path: str,
    max_side: int = 896,
    jpeg_quality: int = 90,
) -> str:
    b64 = _encode_image_base64(image_path, max_side=max_side, jpeg_quality=jpeg_quality)
    return f"data:image/jpeg;base64,{b64}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESPONSE PARSER + PARTIAL JSON SALVAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _clean_vlm_text(raw_text: str) -> str:
    if raw_text is None:
        return ""

    text = str(raw_text).strip()

    # Remove thinking blocks if the model emits them.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def _validate_tuple_item(item: Dict) -> Optional[Dict]:
    if not isinstance(item, dict):
        return None

    nutrient = str(item.get("nutrient", "") or "").strip()
    quantity = str(item.get("quantity", "") or "").strip()
    unit = str(item.get("unit", "") or "").strip()
    context = str(item.get("context", "") or "").strip()

    if not nutrient or not quantity:
        return None

    return {
        "nutrient": nutrient,
        "quantity": quantity,
        "unit": unit,
        "context": context,
    }


def _try_parse_full_json_array(text: str) -> Optional[List[Dict]]:
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end + 1]

    # Remove trailing commas before } or ].
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    valid = []
    for item in data:
        clean = _validate_tuple_item(item)
        if clean is not None:
            valid.append(clean)

    return valid


def _salvage_json_objects(text: str) -> List[Dict]:
    """
    Recover complete JSON objects from a truncated JSON array.

    Example:
      [
        {"nutrient":"A","quantity":"1","unit":"mg","context":"per_100g"},
        {"nutrient":"B","quantity":"2","unit":"mg","context":"per_100g"},
        {"nutrient":"C","quantity":"3"

    This returns A and B instead of returning zero tuples.
    """
    if not text:
        return []

    start = text.find("[")
    if start != -1:
        text = text[start + 1:]

    objects: List[Dict] = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True

        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1

        elif ch == "}":
            if depth > 0:
                depth -= 1

                if depth == 0 and obj_start is not None:
                    candidate = text[obj_start:i + 1]

                    try:
                        item = json.loads(candidate)
                        clean = _validate_tuple_item(item)
                        if clean is not None:
                            objects.append(clean)
                    except Exception:
                        pass

                    obj_start = None

    return objects


def _parse_vlm_response(raw_text: str) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
    """
    Parse VLM response.

    Returns:
        tuples_or_none, metadata

    metadata["parse_mode"]:
        "full"      -> complete valid JSON array
        "salvaged"  -> recovered complete objects from truncated response
        "fail"      -> no usable JSON
    """
    text = _clean_vlm_text(raw_text)

    if not text:
        return None, {
            "parse_mode": "fail",
            "salvaged_objects": 0,
        }

    full = _try_parse_full_json_array(text)

    if full is not None:
        return full, {
            "parse_mode": "full",
            "salvaged_objects": 0,
        }

    salvaged = _salvage_json_objects(text)

    if salvaged:
        logger.warning(
            f"VLM JSON was incomplete; salvaged {len(salvaged)} complete object(s)."
        )
        return salvaged, {
            "parse_mode": "salvaged",
            "salvaged_objects": len(salvaged),
        }

    logger.warning(f"No JSON array found in VLM response: {text[:300]}")
    return None, {
        "parse_mode": "fail",
        "salvaged_objects": 0,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAFE VLM OUTPUT POST-PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CANONICAL_CONTEXTS = {
    "per_100g",
    "per_100ml",
    "per_serving",
    "per_daily_dose",
}


def _clean_unit(unit: str) -> str:
    u = str(unit or "").strip()

    if not u:
        return ""

    u = u.replace("μg", "µg")
    u = re.sub(r"\s+", " ", u).strip()

    if re.fullmatch(r"ug", u, flags=re.IGNORECASE):
        return "µg"

    if re.fullmatch(r"mcg", u, flags=re.IGNORECASE):
        return "µg"

    if re.fullmatch(r"kj", u, flags=re.IGNORECASE):
        return "kJ"

    if re.fullmatch(r"kcal", u, flags=re.IGNORECASE):
        return "kcal"

    if re.fullmatch(r"0\s*g", u, flags=re.IGNORECASE):
        return "g"

    if re.fullmatch(r"mg\s*ne", u, flags=re.IGNORECASE):
        return "mg NE"

    if re.fullmatch(r"mg\s*re", u, flags=re.IGNORECASE):
        return "mg RE"

    if re.fullmatch(r"mg\s*(a|α)-?te", u, flags=re.IGNORECASE):
        return "mg α-TE"

    if re.fullmatch(r"µg\s*re", u, flags=re.IGNORECASE):
        return "µg RE"

    if re.fullmatch(r"kj\s*/\s*kcal", u, flags=re.IGNORECASE):
        return "kJ/kcal"

    return u


def _clean_quantity(quantity: str, unit: str = "") -> str:
    q = str(quantity or "").strip()

    if not q:
        return ""

    q = q.replace(",", ".")
    q = re.sub(r"\s+", " ", q).strip()

    # "70g" -> "70"
    q = re.sub(
        r"^([<>~]?\d+(?:\.\d+)?)\s*"
        r"(mg|µg|ug|mcg|g|kg|ml|kJ|kj|kcal|IE|IU|KBE|CFU)$",
        r"\1",
        q,
        flags=re.IGNORECASE,
    )

    # "1195 (281)" -> "1195"
    # Useful when VLM returns kJ with kcal in parentheses.
    m = re.match(r"^([<>~]?\d+(?:\.\d+)?)\s*\(", q)
    if m:
        return m.group(1)

    # "892/213" with unit kJ -> "892"
    if "/" in q and str(unit).lower() == "kj":
        return q.split("/")[0].strip()

    # "892/213" with unit kcal -> "213"
    if "/" in q and str(unit).lower() == "kcal":
        parts = q.split("/")
        if len(parts) > 1:
            return parts[1].strip()

    return q


def _clean_nutrient_and_infer_unit(
    nutrient: str,
    unit: str,
) -> Tuple[str, str]:
    n = str(nutrient or "").strip()
    u = str(unit or "").strip()

    if not n:
        return n, _clean_unit(u)

    # Infer unit from header-like nutrient text:
    # "protein (g)" -> nutrient="protein", unit="g"
    m = re.search(
        r"\((mg|µg|ug|mcg|g|kg|ml|kJ|kj|kcal|IE|IU|KBE|CFU)\)",
        n,
        flags=re.IGNORECASE,
    )

    if m and not u:
        u = _clean_unit(m.group(1))
        n = re.sub(
            r"\((mg|µg|ug|mcg|g|kg|ml|kJ|kj|kcal|IE|IU|KBE|CFU)\)",
            "",
            n,
            flags=re.IGNORECASE,
        ).strip()

    # Remove leading quantity + unit:
    # "800 mg Phosphor/Phosphorus" -> "Phosphor/Phosphorus"
    n = re.sub(
        r"^[<>~]?\d+(?:[.,]\d+)?\s*"
        r"(mg|µg|ug|mcg|g|kg|ml|kJ|kj|kcal|IE|IU|KBE|CFU)\s+",
        "",
        n,
        flags=re.IGNORECASE,
    ).strip()

    # Clean energy variants.
    if re.match(r"(?i)^(energy|energie|brennwert)\b", n):
        n = "Energy"

    return n, _clean_unit(u)


def _is_ambiguous_10g_context(ctx: str) -> bool:
    """
    pro_10_g / per_10g is ambiguous in your data.

    In image 120.jpeg, pro_10_g corresponds to the daily dose context.
    In other labels, similar Xg contexts can mean serving.

    We handle this with image-level majority.
    """
    s = str(ctx or "").lower().strip()
    s = s.replace("-", "_")
    return bool(re.search(r"\b(pro|per)[_\s-]?10[_\s-]?g\b", s))



def _canonicalize_context(
    ctx: str,
    majority: Optional[str] = None,
) -> str:
    """
    Conservative context canonicalization.

    Important:
      - Map known raw context variants to thesis canonical contexts.
      - Keep unknown raw strings.
      - Do not drop tuples.
    """
    raw = str(ctx or "").strip()

    if not raw:
        return ""

    s = raw.lower().strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", " ", s)

    if s in _CANONICAL_CONTEXTS:
        return s

    # Daily dose variants
    if re.search(
        r"tagesdosis|daily\s*dose|daily\s*intake|daily\s*value|"
        r"recommended\s*daily|reference\s*intake|\bnrv\b|\brda\b",
        s,
        flags=re.IGNORECASE,
    ):
        return "per_daily_dose"

    # Ambiguous "pro 10g" / "per 10g".
    # If the image already mostly looks like per_daily_dose, map it there.
    if _is_ambiguous_10g_context(s):
        if majority == "per_daily_dose":
            return "per_daily_dose"
        return "per_serving"

    # Per 100 g / ml
    if re.search(r"100\s*g|per_100g|pro_100g|pour_100g|je_100g", s):
        return "per_100g"

    if re.search(r"100\s*ml|per_100ml|pro_100ml|je_100ml", s):
        return "per_100ml"

    # Explicit serving/portion aliases
    if re.search(
        r"portion|port\.?|serving|serve|serv\.|stick|bar|drink|shot|scoop|"
        r"cuill|löffel|loeffel|spoon|capsule|capsules|kapsel|"
        r"tablette|tabletten|tablet|piece|stück|stuck|"
        r"pro_portion|per_portion|pro portion|per portion|"
        r"wasser|water",
        s,
        flags=re.IGNORECASE,
    ):
        return "per_serving"

    # Contexts produced by weak VLM:
    # per_25g, per_60g, per_14g, per_3_tabletten, per_30go, por 30go, etc.
    if re.search(
        r"\bper[_\s]?\d+\s*g\b|"
        r"\bpro[_\s]?\d+\s*g\b|"
        r"\bpor[_\s]?\d+\s*g[o0]?\b|"
        r"\bper_\d+g\b|"
        r"\bpro_\d+g\b|"
        r"\bper_\d+_tabletten\b|"
        r"\bpro_\d+_tabletten\b|"
        r"\bper_\d+_capsules?\b",
        s,
        flags=re.IGNORECASE,
    ):
        return "per_serving"

    return raw


def _infer_image_context_majority(tuples: List[Dict]) -> Optional[str]:
    """
    Determine majority context for one image after first-pass canonicalization.

    Used only to resolve ambiguous contexts such as pro_10_g.
    """
    values = []

    for t in tuples:
        ctx = str(t.get("context", "") or "").strip()

        # Do not let ambiguous 10g contexts decide the majority.
        if _is_ambiguous_10g_context(ctx):
            continue

        c = _canonicalize_context(ctx, majority=None)

        if c in _CANONICAL_CONTEXTS:
            values.append(c)

    if not values:
        return None

    return Counter(values).most_common(1)[0][0]


def _dedupe_tuples(tuples: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    out = []
    dropped = 0

    for t in tuples:
        key = (
            str(t.get("nutrient", "")).strip().lower(),
            str(t.get("quantity", "")).strip().lower(),
            str(t.get("unit", "")).strip().lower(),
            str(t.get("context", "")).strip().lower(),
        )

        if key in seen:
            dropped += 1
            continue

        seen.add(key)
        out.append(t)

    return out, dropped
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NUTRIENT OUTPUT CANONICALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _clean_for_nutrient_match(name: str) -> str:
    """
    Light text cleanup used only for matching nutrient names to canonical forms.
    Does not return the final display name.
    """
    s = str(name or "").strip()
    s = s.replace("μ", "µ")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;|/\\-_*")
    return s


def _canonicalize_nutrient_name(name: str) -> str:
    """
    C15-like output normalization for VLM-produced tuple nutrient names.

    Why this exists:
      C15 normalizes OCR tokens before the VLM, but the VLM creates new tuple
      strings. Those strings must also be canonicalized before evaluation.

    Conservative:
      - only maps common nutrition/supplement variants
      - does not use GT
      - does not change quantity/unit/context association
    """
    raw = _clean_for_nutrient_match(name)

    if not raw:
        return ""

    s = raw.lower()
    s_ascii = (
        s.replace("ä", "ae")
         .replace("ö", "oe")
         .replace("ü", "ue")
         .replace("ß", "ss")
         .replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("á", "a")
         .replace("à", "a")
         .replace("ç", "c")
    )

    # Remove common parenthetical qualifiers that should not define the nutrient.
    # Example:
    #   "Niacin (NE)" -> "Niacin"
    #   "Vitamin A (aus/from Provitamin A)" -> "Vitamin A"
    no_paren = re.sub(r"\([^)]*\)", "", raw).strip()
    no_paren_l = no_paren.lower()
    no_paren_ascii = (
        no_paren_l.replace("ä", "ae")
                  .replace("ö", "oe")
                  .replace("ü", "ue")
                  .replace("ß", "ss")
                  .replace("é", "e")
                  .replace("è", "e")
                  .replace("ê", "e")
                  .replace("á", "a")
                  .replace("à", "a")
                  .replace("ç", "c")
    )

    # ── Energy ─────────────────────────────────────────────────────
    if re.search(r"\b(energy|energie|brennwert)\b", s_ascii):
        return "Energy"

    # ── Important: saturated fats before generic fat ───────────────
    if re.search(
        r"saturated\s*fats?|saturates|"
        r"gesaettigte|gesattigte|"
        r"acides?\s+gras\s+satures?|"
        r"davon\s+gesaettigte|davon\s+gesattigte",
        s_ascii,
        flags=re.IGNORECASE,
    ):
        return "Saturated Fats"
    if re.search(r"\bfructose\b|fruktose", s_ascii):
        return "Fructose"
    # ── Sugars before carbohydrate ─────────────────────────────────
    if re.search(
        r"\b(sugars?|zucker|sucre|sucres|azucar|azucares)\b|"
        r"of\s+which\s+sugars?|davon\s+zucker",
        s_ascii,
        flags=re.IGNORECASE,
    ):
        return "Sugars"

    # ── Carbohydrate ───────────────────────────────────────────────
    if re.search(
        r"carbohydrates?|carbohydrate|"
        r"kohlenhydrate|kohlhydrate|kublephydrate|"
        r"hydrate|glucides|koolhydraten",
        s_ascii,
        flags=re.IGNORECASE,
    ):
        return "Carbohydrate"

    # ── Fat ────────────────────────────────────────────────────────
    if re.search(
        r"\b(fat|fett|fetten|fatten|rasva|lipides?|mati[eè]res?\s+grasses?|vetten)\b",
        s_ascii,
        flags=re.IGNORECASE,
    ):
        return "Fat"

    # ── Protein ────────────────────────────────────────────────────
    if re.search(
        r"\b(protein|proteins|proteine|proteines|eiweiss|eiweiß|eiwiitten|"
        r"białko|bialko|eiwitten)\b",
        s_ascii,
        flags=re.IGNORECASE,
    ):
        return "Protein"

    # ── Fibre ──────────────────────────────────────────────────────
    if re.search(
        r"\b(fibre|fiber|dietary\s+fibre|dietary\s+fiber|ballaststoffe|vezels)\b",
        s_ascii,
        flags=re.IGNORECASE,
    ):
        return "Fibre"

    # ── Salt / Sodium ──────────────────────────────────────────────
    if re.search(r"\b(salt|salz|sel|zout|s[oó]l)\b", s_ascii):
        return "Salt"

    if re.search(r"\b(sodium|natrium)\b", s_ascii):
        return "Sodium"

    # ── Common minerals ────────────────────────────────────────────
    if re.search(r"\b(potassium|kalium)\b", s_ascii):
        return "Potassium"

    if re.search(r"\b(calcium|kalzium)\b", s_ascii):
        return "Calcium"

    if re.search(r"\b(magnesium)\b", s_ascii):
        return "Magnesium"

    if re.search(r"\b(zinc|zink)\b", s_ascii):
        return "Zinc"

    if re.search(r"\b(chloride|chlorid)\b", s_ascii):
        return "Chloride"

    if re.search(r"\b(phosphorus|phosphor)\b", s_ascii):
        return "Phosphorus"

    if re.search(r"\b(selenium|selen)\b", s_ascii):
        return "Selenium"

    if re.search(r"\b(chromium|chrom)\b", s_ascii):
        return "Chromium"

    if re.search(r"\b(manganese|mangan)\b", s_ascii):
        return "Manganese"

    if re.search(r"\b(copper|kupfer)\b", s_ascii):
        return "Copper"

    if re.search(r"\b(iron|eisen)\b", s_ascii):
        return "Iron"

    if re.search(r"\b(iodine|jod)\b", s_ascii):
        return "Iodine"

    # ── Vitamins ───────────────────────────────────────────────────
    # Handle short forms like "B6", "B12"
    m = re.fullmatch(r"b\s*(1|2|3|5|6|7|9|12)", no_paren_ascii.strip())
    if m:
        return f"Vitamin B{m.group(1)}"

    m = re.search(r"\bvitamin\s*([abcdek])\s*(\d+)?\b", no_paren_ascii)
    if m:
        letter = m.group(1).upper()
        suffix = m.group(2) or ""
        return f"Vitamin {letter}{suffix}"

    if re.search(r"\bvitamin\s*b\s*1\b|\bthiamin\b|\bthiamine\b", s_ascii):
        return "Vitamin B1"

    if re.search(r"\bvitamin\s*b\s*2\b|\briboflavin\b", s_ascii):
        return "Vitamin B2"

    if re.search(r"\bvitamin\s*b\s*6\b|\bpyridoxin\b|\bpyridoxine\b", s_ascii):
        return "Vitamin B6"

    if re.search(r"\bvitamin\s*b\s*12\b|\bcobalamin\b", s_ascii):
        return "Vitamin B12"

    if re.search(r"\bvitamin\s*c\b|ascorbic", s_ascii):
        return "Vitamin C"

    if re.search(r"\bvitamin\s*d3\b", s_ascii):
        return "Vitamin D3"

    if re.search(r"\bvitamin\s*d\b", s_ascii):
        return "Vitamin D"

    if re.search(r"\bvitamin\s*e\b", s_ascii):
        return "Vitamin E"

    if re.search(r"\bvitamin\s*a\b", s_ascii):
        return "Vitamin A"

    if re.search(r"\bvitamin\s*k\b", s_ascii):
        return "Vitamin K"

    # ── B-vitamin related nutrients ────────────────────────────────
    if re.search(r"\bniacin|niacin\(e\)|niacine\b", s_ascii):
        return "Niacin"

    if re.search(r"folic\s+acid|folsaeure|folsaure|folic\s*acid|folsäure", s_ascii):
        return "Folic Acid"

    if re.search(r"pantothenic\s+acid|pantothensaeure|pantothensaure|pantothensäure", s_ascii):
        return "Pantothenic Acid"

    if re.search(r"\bbiotin\b", s_ascii):
        return "Biotin"

    if re.search(r"\bcholine\b|\bcholin\b", s_ascii):
        return "Choline"

    if re.search(r"\binositol\b", s_ascii):
        return "Inositol"

    # ── Other common supplement compounds ──────────────────────────
    if re.search(r"alpha[-\s]?lipon|alpha[-\s]?lipoic", s_ascii):
        return "Alpha-Liponsäure"

    if re.search(r"\brutin\b", s_ascii):
        return "Rutin"

    if re.search(r"\bcaffeine\b|\bkoffein\b", s_ascii):
        return "Caffeine"

    if re.search(r"\bcreatine\s+monohydrate\b|\bkreatin\s+monohydrat\b", s_ascii):
        return "Creatine monohydrate"

    if re.search(r"\bcreatine\b|\bkreatin\b", s_ascii):
        return "Creatine"

    # ── Amino acids ────────────────────────────────────────────────
    amino_map = {
        "l-valine": "L-Valine",
        "valine": "L-Valine",
        "l-leucine": "L-Leucine",
        "leucine": "L-Leucine",
        "l-isoleucine": "L-Isoleucine",
        "isoleucine": "L-Isoleucine",
        "l-methionine": "L-Methionine",
        "methionine": "L-Methionine",
        "l-phenylalanine": "L-Phenylalanine",
        "phenylalanine": "L-Phenylalanine",
        "l-tryptophan": "L-Tryptophan",
        "tryptophan": "L-Tryptophan",
    }

    for key, canonical in amino_map.items():
        if key in s_ascii:
            return canonical

    # ── Probiotics ─────────────────────────────────────────────────
    if re.search(r"lactobacillus\s+acidophilus", s_ascii):
        return "Lactobacillus Acidophilus"

    if re.search(r"bifidobacterium\s+bifidum", s_ascii):
        return "Bifidobacterium Bifidum"

    # If parenthetical cleanup produced a cleaner vitamin/mineral name,
    # keep that cleaned version.
    if no_paren and no_paren != raw:
        return no_paren

    return raw
def _postprocess_vlm_tuples(tuples: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Safe post-processing for VLM outputs.

    v3_7 behavior:
      - No energy-pair expansion.
      - No C18/C18b VLM display logic.
      - No chunking.
      - No invalid-context dropping.

    v3_7+ addition:
      - Fructose nutrient canonicalization is handled inside
        _canonicalize_nutrient_name().
    """

    if not tuples:
        return tuples, {
            "postprocessed": 0,
            "context_mapped": 0,
            "deduped": 0,
            "context_majority": None,
        }

    context_mapped = 0

    # First pass: clean nutrient/unit/quantity.
    for t in tuples:
        nutrient, unit = _clean_nutrient_and_infer_unit(
            t.get("nutrient", ""),
            t.get("unit", ""),
        )

        # C15-like normalization after VLM output creation.
        nutrient = _canonicalize_nutrient_name(nutrient)

        unit = _clean_unit(unit)
        quantity = _clean_quantity(t.get("quantity", ""), unit)

        t["nutrient"] = nutrient
        t["unit"] = unit
        t["quantity"] = quantity

    # Infer image-level majority context after initial cleaning.
    majority = _infer_image_context_majority(tuples)

    # Second pass: context canonicalization with majority hint.
    for t in tuples:
        original = str(t.get("context", "") or "").strip()
        canonical = _canonicalize_context(original, majority=majority)

        if canonical != original:
            context_mapped += 1

        t["context"] = canonical

    tuples, deduped = _dedupe_tuples(tuples)

    return tuples, {
        "postprocessed": len(tuples),
        "context_mapped": context_mapped,
        "deduped": deduped,
        "context_majority": majority,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VLMAssociator:
    """
    Stage 5-VLM — VLM-based tuple association.

    Backends:
      - openai_compat (LM Studio, vLLM, etc.) — default
      - hf            (Hugging Face Inference Providers)
      - ollama        (legacy)
    """

    _SHRINK_SCHEDULE = [896, 768, 640, 512]

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.diagnostics: Dict[str, Any] = {}
        self._last_raw_response: str = ""

    def extract(
        self,
        enriched_tokens: List[Dict],
        image_path: str,
        image_id: str = "unknown",
    ) -> List[Dict]:
        t0 = time.time()

        token_table = _build_token_table(enriched_tokens, self.config)
        user_prompt = _build_user_prompt(token_table, image_id)

        filtered_labels = self.config.get(
            "allowed_labels",
            {"NUTRIENT", "QUANTITY", "UNIT", "CONTEXT"},
        )

        n_active = sum(
            1 for t in enriched_tokens
            if t.get("label") in filtered_labels
        )
        n_nutrients = sum(
            1 for t in enriched_tokens
            if t.get("label") == "NUTRIENT"
        )
        n_quantities = sum(
            1 for t in enriched_tokens
            if t.get("label") == "QUANTITY"
        )

        logger.info(
            f"[VLM] {image_id}: backend={self.config['backend']} | "
            f"{n_active} prompt tokens "
            f"({n_nutrients} NUT, {n_quantities} QTY) | "
            f"img_max={self.config.get('image_max_side')}px"
        )

        if self.config.get("save_prompts", False):
            self._save_debug_prompt(image_id, user_prompt)

        tuples: Optional[List[Dict]] = None
        attempts = 0
        max_attempts = int(self.config.get("retry_max", 0)) + 1

        default_max_side = int(self.config.get("image_max_side", 896))
        jpeg_q = int(self.config.get("image_jpeg_quality", 90))
        shrink = bool(self.config.get("image_shrink_on_retry", True))

        schedule = [default_max_side]
        if shrink:
            for s in self._SHRINK_SCHEDULE:
                if s < default_max_side and s not in schedule:
                    schedule.append(s)

        last_max_side = default_max_side
        last_error_kind: Optional[str] = None
        parse_mode = "not_called"
        salvaged_objects = 0
        postprocess_meta: Dict[str, Any] = {}

        while tuples is None and attempts < max_attempts:
            attempts += 1

            if shrink and attempts <= len(schedule):
                current_max_side = schedule[attempts - 1]
            else:
                current_max_side = default_max_side

            last_max_side = current_max_side

            try:
                image_b64 = _encode_image_base64(
                    image_path,
                    max_side=current_max_side,
                    jpeg_quality=jpeg_q,
                )
                image_data_uri = (
                    f"data:image/jpeg;base64,{image_b64}"
                )
            except Exception as e:
                logger.error(f"[VLM] {image_id}: image encoding failed: {e}")
                last_error_kind = "image_encode"
                break

            backend = self.config["backend"]

            if backend == "ollama":
                raw_response = self._call_ollama(image_b64, user_prompt)
            elif backend in ("openai_compat", "lmstudio"):
                raw_response = self._call_openai_compat(
                    image_data_uri, user_prompt
                )
            elif backend == "hf":
                raw_response = self._call_huggingface(
                    image_data_uri, user_prompt
                )
            else:
                raise ValueError(
                    f"Unsupported VLM backend: {backend!r}. "
                    f"Use 'openai_compat', 'hf', or 'ollama'."
                )

            self._last_raw_response = raw_response or ""

            if raw_response:
                parsed, parse_meta = _parse_vlm_response(raw_response)
                parse_mode = parse_meta.get("parse_mode", "fail")
                salvaged_objects = int(parse_meta.get("salvaged_objects", 0))

                if parsed is None:
                    last_error_kind = "parse_fail"
                    logger.warning(
                        f"[VLM] {image_id}: parse failed "
                        f"(attempt {attempts}/{max_attempts}, "
                        f"img={current_max_side}px)"
                    )
                else:
                    tuples = parsed
                    last_error_kind = None
            else:
                last_error_kind = "no_response"

        if tuples is None:
            logger.error(
                f"[VLM] {image_id}: all {max_attempts} attempts failed "
                f"(last_error={last_error_kind})"
            )
            tuples = []

        # Safe post-processing after successful parsing/salvage.
        if self.config.get("postprocess_output", True) and tuples:
            tuples, postprocess_meta = _postprocess_vlm_tuples(tuples)

        # Important:
        # Do NOT drop invalid contexts.
        # The default is False and this block is intentionally absent.

        for t in tuples:
            t["image_id"] = image_id

        elapsed = time.time() - t0

        self.diagnostics = {
            "mode": "vlm",
            "backend": self.config["backend"],
            "model": self._active_model_name(),
            "active_tokens": n_active,
            "nutrients": n_nutrients,
            "quantities": n_quantities,
            "tuples": len(tuples),
            "attempts": attempts,
            "image_max_side": last_max_side,
            "last_error": last_error_kind,
            "elapsed_s": round(elapsed, 2),
            "prompt_char_len": len(user_prompt),
            "max_tokens": self.config.get("max_tokens"),
            "parse_mode": parse_mode,
            "salvaged_objects": salvaged_objects,
            "postprocess": postprocess_meta,
        }

        return tuples

    def _call_openai_compat(
        self,
        image_data_uri: str,
        user_prompt: str,
        base_url_override: Optional[str] = None,
        model_override: Optional[str] = None,
        auth_env_override: Optional[str] = None,
    ) -> Optional[str]:
        url = base_url_override or self.config.get("openai_base_url")
        model = model_override or self.config.get("openai_model")

        headers = {"Content-Type": "application/json"}

        auth_env = (
            auth_env_override
            if auth_env_override is not None
            else self.config.get("openai_auth_env")
        )
        if auth_env:
            token = os.environ.get(auth_env)
            if not token:
                logger.error(
                    f"[VLM] Missing auth token. "
                    f"Set environment variable {auth_env}."
                )
                return None
            headers["Authorization"] = f"Bearer {token}"

        payload = {
            "model": model,
            "stream": False,
            "temperature": float(self.config.get("temperature", 0.1)),
            "max_tokens": int(self.config.get("max_tokens", 4096)),
            "messages": [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                    ],
                },
            ],
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config["timeout_s"],
            )

            if resp.status_code >= 400:
                logger.error(
                    f"[VLM] OpenAI-compat error {resp.status_code}: "
                    f"{resp.text[:1000]}"
                )
                return None

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.error(
                    f"[VLM] OpenAI-compat returned no choices: {data}"
                )
                return None

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "") or ""))
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)

            return str(content)

        except requests.exceptions.Timeout:
            logger.error(
                f"[VLM] OpenAI-compat timeout after "
                f"{self.config['timeout_s']}s"
            )
            return None

        except requests.exceptions.ConnectionError:
            logger.error(f"[VLM] Cannot connect to {url}")
            return None

        except Exception as e:
            logger.error(f"[VLM] OpenAI-compat error: {e}")
            return None

    def _call_huggingface(
        self,
        image_data_uri: str,
        user_prompt: str,
    ) -> Optional[str]:
        return self._call_openai_compat(
            image_data_uri,
            user_prompt,
            base_url_override=self.config.get("hf_base_url"),
            model_override=self.config.get("hf_model"),
            auth_env_override=self.config.get("hf_token_env"),
        )

    def _call_ollama(
        self,
        image_b64: str,
        user_prompt: str,
    ) -> Optional[str]:
        url = f"{self.config['ollama_base_url']}/api/chat"

        num_predict = self.config.get("num_predict")
        if num_predict is None:
            num_predict = self.config.get("max_tokens", 4096)

        num_ctx = int(self.config.get("ollama_num_ctx", 8192))

        payload = {
            "model": self.config["model"],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": float(self.config.get("temperature", 0.1)),
                "num_predict": int(num_predict),
                "num_ctx": num_ctx,
            },
            "messages": [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_b64],
                },
            ],
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config["timeout_s"],
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

        except requests.exceptions.Timeout:
            logger.error(
                f"[VLM] Ollama timeout after {self.config['timeout_s']}s"
            )
            return None

        except requests.exceptions.ConnectionError:
            logger.error("[VLM] Cannot connect to Ollama — is it running?")
            return None

        except Exception as e:
            logger.error(f"[VLM] Ollama error: {e}")
            return None

    def _active_model_name(self) -> str:
        backend = self.config.get("backend")
        if backend in ("openai_compat", "lmstudio"):
            return str(self.config.get("openai_model", ""))
        if backend == "hf":
            return str(self.config.get("hf_model", ""))
        return str(self.config.get("model", ""))

    def _save_debug_prompt(self, image_id: str, user_prompt: str) -> None:
        debug_dir = Path(
            self.config.get("prompt_debug_dir", "outputs/debug_vlm_prompts")
        )
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_image_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", image_id)
        out_path = debug_dir / f"{safe_image_id}.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("===== SYSTEM PROMPT =====\n\n")
            f.write(_SYSTEM_PROMPT)
            f.write("\n\n===== USER PROMPT =====\n\n")
            f.write(user_prompt)

    def print_tuples(self, tuples: List[Dict]) -> None:
        print(f"\n{'=' * 80}")
        print(f"VLM-EXTRACTED TUPLES  ({len(tuples)} total)")
        print(f"{'=' * 80}")
        print(f"{'NUTRIENT':<30} {'QTY':<10} {'UNIT':<8} {'CONTEXT':<20}")
        print("-" * 80)

        for t in tuples:
            print(
                f"{str(t.get('nutrient', ''))[:29]:<30} "
                f"{str(t.get('quantity', ''))[:9]:<10} "
                f"{str(t.get('unit', ''))[:7]:<8} "
                f"{str(t.get('context', ''))[:19]:<20}"
            )

        print(f"{'=' * 80}\n")

    def save_csv(self, tuples: List[Dict], output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["image_id", "nutrient", "quantity", "unit", "context"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(tuples)

        print(f"VLM tuples saved: {output_path}  ({len(tuples)} rows)")

    def print_diagnostics(self) -> None:
        d = self.diagnostics

        print(f"\n{'=' * 50}")
        print("VLM ASSOCIATION DIAGNOSTICS")
        print(f"{'=' * 50}")

        for k, v in d.items():
            print(f"  {k:<20} : {v}")

        print(f"{'=' * 50}\n")