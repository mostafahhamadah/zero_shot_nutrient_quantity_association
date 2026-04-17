"""
text_serializer.py
==================
Stage 3A — OCR Token Serializer for GLiNER

PURPOSE
-------
Convert corrected, already-split OCR tokens (output of Stage 2 / PaddleOCR
corrector, which includes split_fused_token()) into a serialized text string
suitable for GLiNER span extraction, while preserving exact character offsets
so GLiNER span predictions can be mapped back to original token indices and
bounding boxes for Stage 4 graph construction.

DESIGN DECISIONS (locked 2025-04-06)
--------------------------------------
split_fused_token()   : Runs BEFORE this module (inside PaddleOCR Stage 2
                        corrector). This serializer always receives already-split
                        clean tokens — no fused tokens like "400mg" will appear.

Line grouping         : Approach 3 — local neighbourhood threshold per token.
                        For each token at position i in cy-sorted order, the
                        grouping threshold = max(8px, 0.5 * median height of the
                        K nearest cy-neighbours). This correctly handles mixed-font
                        supplement labels where header rows, data rows, and
                        sub-nutrient rows have different font sizes on the same image.

Multi-token spans     : Tag-all strategy. Every token whose [start_char, end_char)
                        overlaps a GLiNER span receives that span's label. Stage 4
                        graph_constructor.py requires zero changes.

GLiNER conf threshold : 0.3 — set in gliner_classifier.py, not here.
GLiNER fallback label : UNKNOWN — tokens not covered by any span are labelled
                        UNKNOWN and passed to the rule-based fallback chain.
                        Set in gliner_classifier.py.

PIPELINE POSITION
-----------------
Stage 2 (PaddleOCR corrector)
    → [this module] text_serializer.py   →  {text, token_spans, lines}
    → gliner_classifier.py               →  per-token label list
    → graph_constructor.py  (Stage 4, unchanged)

EXPECTED INPUT TOKEN SCHEMA
---------------------------
Each token dict must contain:
    token       : str    — corrected token text (already split, no fused values)
    x1, y1      : int    — top-left bounding box corner
    x2, y2      : int    — bottom-right bounding box corner
    cx, cy      : float  — centroid coordinates
    conf        : float  — OCR confidence score

RETURNS
-------
{
    "text"        : str,
    "token_spans" : List[Dict],
    "lines"       : List[Dict],
}

token_spans — one entry per serialized token:
{
    "token_index" : int,    original index into the input token list
    "token_text"  : str,    token text as serialized (stripped)
    "start_char"  : int,    inclusive character offset in "text"
    "end_char"    : int,    exclusive character offset in "text"
    "line_id"     : int,    which visual line this token belongs to
}

lines — one entry per visual line:
{
    "line_id"       : int,
    "token_indices" : List[int],   original indices of tokens in this line
    "y_center"      : float,       mean cy of tokens in this line
}

USAGE EXAMPLE
-------------
    from text_serializer import serialize_tokens_for_gliner

    corrected_tokens = ocr_corrector.run(image_path)   # Stage 2 output
    serialized = serialize_tokens_for_gliner(corrected_tokens)

    gliner_input_text  = serialized["text"]
    token_spans        = serialized["token_spans"]   # passed to GLiNER classifier
    lines              = serialized["lines"]          # passed to diagnostics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Neighbourhood half-width for local threshold estimation.
#: Token i uses neighbours at positions [i-K, i+K] in cy-sorted order.
_DEFAULT_K: int = 3

#: Absolute floor for the line-grouping threshold (pixels).
#: Prevents collapse on images with extremely small fonts.
_THRESH_FLOOR_PX: float = 8.0

#: Fraction of median neighbour height used as the grouping threshold.
#: Tokens whose cy differs by less than (SCALE * median_height) from a line's
#: y_center are considered on the same visual line.
_THRESH_HEIGHT_SCALE: float = 0.5


# ---------------------------------------------------------------------------
# Private coordinate helpers
# ---------------------------------------------------------------------------

def _text(tok: Dict[str, Any]) -> str:
    """Return stripped token text, or empty string if key missing."""
    return str(tok.get("token", "")).strip()


def _height(tok: Dict[str, Any]) -> float:
    """
    Return bounding-box height (y2 - y1) for a token.
    Returns 0.0 if coordinates are missing, non-numeric, or negative.
    """
    try:
        h = float(tok["y2"]) - float(tok["y1"])
        return max(0.0, h)
    except (KeyError, TypeError, ValueError):
        return 0.0


def _cy(tok: Dict[str, Any]) -> float:
    """Return vertical centroid. Returns 0.0 on missing/invalid data."""
    try:
        return float(tok["cy"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _cx(tok: Dict[str, Any]) -> float:
    """Return horizontal centroid. Returns 0.0 on missing/invalid data."""
    try:
        return float(tok["cx"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _median(values: List[float]) -> float:
    """Return median of a list. Returns 0.0 for empty input."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# Approach 3 — Local neighbourhood threshold
# ---------------------------------------------------------------------------

def _compute_local_thresholds(
    sorted_valid: List[Tuple[int, Dict[str, Any]]],
    k: int = _DEFAULT_K,
) -> List[float]:
    """
    Compute a per-token local line-grouping threshold (Approach 3).

    For each token at position i in the cy-sorted list, collect the bounding-box
    heights of the K nearest neighbours on each side (clamped to list bounds).
    Threshold = max(_THRESH_FLOOR_PX, _THRESH_HEIGHT_SCALE * median_height).

    This makes the threshold adaptive to the local font size at each vertical
    position in the image, correctly separating dense sub-nutrient rows from
    large header rows on the same label.

    Parameters
    ----------
    sorted_valid : list of (original_token_index, token_dict) sorted by cy asc.
    k            : neighbourhood half-width. Default 3 uses up to 6 neighbours.

    Returns
    -------
    List of float thresholds, one per entry in sorted_valid (same order).
    """
    n = len(sorted_valid)
    thresholds: List[float] = []

    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)  # exclusive upper bound
        neighbourhood = sorted_valid[lo:hi]

        heights = [h for _, tok in neighbourhood if (h := _height(tok)) > 0]
        median_h = _median(heights)

        if median_h > 0:
            thresh = max(_THRESH_FLOOR_PX, _THRESH_HEIGHT_SCALE * median_h)
        else:
            thresh = _THRESH_FLOOR_PX

        thresholds.append(thresh)

    return thresholds


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------

def group_tokens_into_lines(
    tokens: List[Dict[str, Any]],
    k: int = _DEFAULT_K,
) -> List[Dict[str, Any]]:
    """
    Group OCR tokens into visual lines using local neighbourhood thresholds.

    Algorithm
    ---------
    1. FILTER  — keep tokens with non-empty text and valid cx/cy fields.
    2. SORT    — sort filtered tokens by (cy, cx) for top-to-bottom,
                 left-to-right processing.
    3. THRESH  — compute per-token local threshold via _compute_local_thresholds().
    4. SWEEP   — greedy single pass: assign each token to the nearest open line
                 whose y_center is within the token's local threshold.
                 If no line qualifies, open a new line.
                 Running y_center = mean cy of all tokens assigned so far.
    5. FINALIZE — re-sort lines by y_center (top to bottom); re-sort tokens
                  within each line by cx (left to right reading order).
                  Re-index line_id values 0, 1, 2, ...

    Why greedy sweep works here
    ---------------------------
    Supplement labels are structured documents. Tokens on the same visual row
    share very similar cy values; the inter-row gap is consistently larger than
    the intra-row variation. The greedy sweep exploits this by always assigning
    to the closest existing line rather than building a global clustering — this
    is both faster (O(n * L) where L = number of lines ≈ 10-30) and more
    predictable for debugging.

    Parameters
    ----------
    tokens : raw token list from Stage 2 corrector (split_fused_token() already
             applied upstream).
    k      : neighbourhood half-width for local threshold. Default 3.

    Returns
    -------
    List of line dicts:
    {
        "line_id"  : int,
        "tokens"   : List[Tuple[int, Dict]],   (original_index, token_dict)
        "y_center" : float,
    }
    The returned list is sorted by y_center ascending (top of label first).
    """
    # ---- Step 1: filter ----
    valid: List[Tuple[int, Dict[str, Any]]] = []
    for idx, tok in enumerate(tokens):
        if not _text(tok):
            continue
        if "cx" not in tok or "cy" not in tok:
            continue
        valid.append((idx, tok))

    if not valid:
        return []

    # ---- Step 2: sort by cy, then cx ----
    valid.sort(key=lambda item: (_cy(item[1]), _cx(item[1])))

    # ---- Step 3: per-token local thresholds ----
    thresholds = _compute_local_thresholds(valid, k=k)

    # ---- Step 4: greedy sweep ----
    lines: List[Dict[str, Any]] = []

    for i, (orig_idx, tok) in enumerate(valid):
        cy_tok = _cy(tok)
        tok_thresh = thresholds[i]

        best_line: Optional[Dict[str, Any]] = None
        best_dist = float("inf")

        for line in lines:
            dist = abs(cy_tok - line["y_center"])
            if dist <= tok_thresh and dist < best_dist:
                best_line = line
                best_dist = dist

        if best_line is None:
            # Open a new line
            lines.append({
                "line_id": len(lines),          # temporary id, re-indexed below
                "tokens": [(orig_idx, tok)],
                "y_center": cy_tok,
            })
        else:
            best_line["tokens"].append((orig_idx, tok))
            # Update running mean y_center
            cy_vals = [_cy(t) for _, t in best_line["tokens"]]
            best_line["y_center"] = sum(cy_vals) / len(cy_vals)

    # ---- Step 5: finalize ----
    lines.sort(key=lambda ln: ln["y_center"])
    for new_id, line in enumerate(lines):
        line["line_id"] = new_id
        line["tokens"].sort(key=lambda item: _cx(item[1]))

    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serialize_tokens_for_gliner(
    tokens: List[Dict[str, Any]],
    k: int = _DEFAULT_K,
) -> Dict[str, Any]:
    """
    Serialize corrected OCR tokens into a GLiNER-ready text string.

    Tokens on the same visual line are joined by a single space character.
    Lines are separated by newline characters (\\n). No trailing newline.

    Character offsets use half-open intervals [start_char, end_char) so that
    the mapping condition for tag-all span assignment in gliner_classifier.py is:

        span.start <= token.start_char  and  token.end_char <= span.end

    This guarantees every token fully contained within a GLiNER span receives
    that span's label (tag-all strategy).

    Parameters
    ----------
    tokens : token list from Stage 2 corrector. split_fused_token() has already
             run — no fused tokens expected.
    k      : neighbourhood half-width passed to group_tokens_into_lines().

    Returns
    -------
    dict with keys:
        "text"        — str: the full serialized string for GLiNER input.
        "token_spans" — list of span dicts (see module docstring).
        "lines"       — list of line dicts (for diagnostics / graph construction).

    Raises
    ------
    Does not raise. Returns empty text and empty lists if all tokens are filtered.
    """
    lines = group_tokens_into_lines(tokens, k=k)

    parts: List[str] = []
    token_spans: List[Dict[str, Any]] = []
    line_summaries: List[Dict[str, Any]] = []

    cursor: int = 0

    for line_pos, line in enumerate(lines):
        line_token_indices: List[int] = []
        first_in_line = True

        for orig_idx, tok in line["tokens"]:
            token_text = _text(tok)
            if not token_text:
                # Guard: filtered during group_tokens_into_lines but double-check
                continue

            # Space separator between tokens on the same line
            if not first_in_line:
                parts.append(" ")
                cursor += 1
            first_in_line = False

            start_char = cursor
            parts.append(token_text)
            cursor += len(token_text)
            end_char = cursor          # exclusive

            token_spans.append({
                "token_index": orig_idx,
                "token_text":  token_text,
                "start_char":  start_char,
                "end_char":    end_char,    # exclusive: [start_char, end_char)
                "line_id":     line["line_id"],
            })

            line_token_indices.append(orig_idx)

        line_summaries.append({
            "line_id":       line["line_id"],
            "token_indices": line_token_indices,
            "y_center":      line["y_center"],
        })

        # Newline separator between lines — NOT after the last line
        if line_pos < len(lines) - 1:
            parts.append("\n")
            cursor += 1

    return {
        "text":        "".join(parts),
        "token_spans": token_spans,
        "lines":       line_summaries,
    }