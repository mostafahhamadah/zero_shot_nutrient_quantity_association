"""
geometry_engine.py
==================
Direction-Aware Geometry Engine
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Treat each OCR token as a directional vector object.  This module provides
the geometric primitives needed to handle skewed, inclined, and curved
supplement label images where raw (x, y) thresholds break.

CORE CONCEPTS
-------------
  Token as vector:
    Each token has a DIRECTION (along its reading line) and a NORMAL
    (perpendicular to direction).  Two tokens are on the same logical
    row when they share direction AND have small perpendicular distance.

  Direction vector (d):
    Unit vector along the token's reading direction.
    For a token with bbox corners, direction = (x2-x1, y2-y1) of the
    top edge, normalized.  For axis-aligned tokens, d ≈ (1, 0).

  Normal vector (n):
    Perpendicular to direction, pointing downward (into the table).
    n = rotate(d, +90°) = (-d_y, d_x).

  Parallel displacement:
    Projection of center-to-center vector onto direction.
    Large parallel displacement = tokens far apart along the row.

  Perpendicular displacement:
    Projection of center-to-center vector onto normal.
    Small perpendicular displacement = tokens on the same row.

WHY LOCAL GEOMETRY
------------------
A globally estimated skew angle is insufficient because:
  1. Different columns may have different local angles on curved labels.
  2. A single rotation correction cannot handle per-column variation.
  3. Direction-aware logic handles flat, skewed, AND curved in one framework.

REFERENCE
---------
Design inspired by SPADE (Hwang et al., 2021) relation-based reasoning
and PICK (Yu et al., 2020) spatial distance weighting, adapted for
zero-shot conditions.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VECTOR UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a 2D vector.  Returns (1, 0) for zero-length input."""
    mag = np.linalg.norm(v)
    if mag < 1e-8:
        return np.array([1.0, 0.0])
    return v / mag


def _angle_deg(v: np.ndarray) -> float:
    """Angle of a 2D vector in degrees, range [-180, 180]."""
    return math.degrees(math.atan2(v[1], v[0]))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two unit vectors, in degrees [0, 180]."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _perpendicular(d: np.ndarray) -> np.ndarray:
    """
    Normal vector: rotate direction 90° clockwise.
    If direction is (dx, dy), normal is (-dy, dx).
    Points 'downward' into the table for standard reading direction.
    """
    return np.array([-d[1], d[0]])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOKEN GEOMETRY ESTIMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def estimate_token_direction(token: dict) -> np.ndarray:
    """
    Estimate the reading direction of a token from its bounding box.

    Uses the top edge vector (top-left → top-right) of the bbox.
    For axis-aligned bboxes this yields (1, 0).
    For rotated/quadrilateral bboxes from PaddleOCR, uses the actual
    polygon corners if available, otherwise falls back to AABB.

    Args:
        token: dict with at minimum {x1, y1, x2, y2}.
               May optionally contain 'quad' = [[x,y], [x,y], [x,y], [x,y]]
               for the original 4-corner polygon (TL, TR, BR, BL).

    Returns:
        Unit direction vector as np.ndarray of shape (2,).
    """
    if "quad" in token and len(token["quad"]) == 4:
        tl, tr = np.array(token["quad"][0]), np.array(token["quad"][1])
        return _normalize(tr - tl)

    # Fallback: AABB top edge → always horizontal
    dx = token["x2"] - token["x1"]
    dy = 0.0  # AABB has no rotation information
    return _normalize(np.array([dx, dy]))


def estimate_local_angle(token: dict) -> float:
    """Return the local angle (degrees) of the token's reading direction."""
    d = estimate_token_direction(token)
    return _angle_deg(d)


def compute_token_geometry(token: dict) -> dict:
    """
    Compute all geometry fields for a single token.

    Returns dict with:
        center:    (cx, cy) as floats
        width:     bbox width
        height:    bbox height
        direction: unit direction vector [dx, dy]
        normal:    unit normal vector [nx, ny]
        angle_deg: local angle in degrees
    """
    cx = (token["x1"] + token["x2"]) / 2.0
    cy = (token["y1"] + token["y2"]) / 2.0
    w  = token["x2"] - token["x1"]
    h  = token["y2"] - token["y1"]

    d = estimate_token_direction(token)
    n = _perpendicular(d)
    angle = _angle_deg(d)

    return {
        "center":    (cx, cy),
        "width":     w,
        "height":    h,
        "direction": d.tolist(),
        "normal":    n.tolist(),
        "angle_deg": angle,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAIRWISE GEOMETRY PREDICATES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def displacement_components(a: dict, b: dict) -> Tuple[float, float]:
    """
    Decompose the center-to-center vector from token a to token b
    into parallel (along-row) and perpendicular (across-row) components,
    using token a's direction as the reference frame.

    Returns:
        (parallel, perpendicular) — signed floats in pixels.
        parallel > 0 means b is ahead of a in reading direction.
        perpendicular > 0 means b is below a (in normal direction).
    """
    d_a = np.array(a.get("direction", [1.0, 0.0]))
    n_a = np.array(a.get("normal", [0.0, 1.0]))

    ca = np.array(a.get("center", ((a["x1"]+a["x2"])/2, (a["y1"]+a["y2"])/2)))
    cb = np.array(b.get("center", ((b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2)))

    delta = cb - ca
    parallel = float(np.dot(delta, d_a))
    perpendicular = float(np.dot(delta, n_a))
    return parallel, perpendicular


def direction_compatible(a: dict, b: dict, max_angle_deg: float = 15.0) -> bool:
    """
    True if two tokens have compatible reading directions.

    Handles anti-parallel directions (180° flip) by comparing
    the minimum of angle and |180° - angle|.
    """
    d_a = np.array(a.get("direction", [1.0, 0.0]))
    d_b = np.array(b.get("direction", [1.0, 0.0]))
    angle = _angle_between(d_a, d_b)
    # Anti-parallel is also compatible (text read left or right)
    effective = min(angle, abs(180.0 - angle))
    return effective <= max_angle_deg


def row_compatible(a: dict, b: dict,
                   max_perp_px: float = 25.0,
                   max_angle_deg: float = 15.0) -> Tuple[bool, float]:
    """
    Direction-aware row compatibility check.

    Two tokens are row-compatible when:
      1. Their directions are compatible (within max_angle_deg).
      2. Their perpendicular displacement is within max_perp_px.

    Returns:
        (is_compatible, confidence)
        confidence ∈ [0, 1], higher = more likely same row.
    """
    if not direction_compatible(a, b, max_angle_deg):
        return False, 0.0

    _, perp = displacement_components(a, b)
    abs_perp = abs(perp)

    if abs_perp > max_perp_px:
        return False, 0.0

    confidence = 1.0 - (abs_perp / max_perp_px)
    return True, round(confidence, 4)


def column_compatible(a: dict, b: dict,
                      max_parallel_px: float = 30.0,
                      max_angle_deg: float = 15.0) -> Tuple[bool, float]:
    """
    Direction-aware column compatibility check.

    Two tokens are column-compatible when:
      1. Their directions are compatible.
      2. Their PARALLEL displacement (along row) is small — meaning
         they are vertically stacked, not side by side.

    Returns:
        (is_compatible, confidence)
    """
    if not direction_compatible(a, b, max_angle_deg):
        return False, 0.0

    para, _ = displacement_components(a, b)
    abs_para = abs(para)

    if abs_para > max_parallel_px:
        return False, 0.0

    confidence = 1.0 - (abs_para / max_parallel_px)
    return True, round(confidence, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW INDUCTION (Section D) — Gap-based grouping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def induce_rows(tokens: List[dict],
                max_perp_px: float = 25.0,
                max_angle_deg: float = 15.0) -> List[List[int]]:
    """
    Cluster tokens into logical rows using gap-based grouping.

    Algorithm:
      1. Compute perpendicular position for each token (cy for flat images,
         projection onto normal for skewed images).
      2. Sort tokens by perpendicular position.
      3. Walk sorted list: when the gap between consecutive tokens exceeds
         max_perp_px, start a new row.
      4. Sort tokens within each row by parallel position (reading order).

    Why gap-based instead of union-find:
      Union-find with single-linkage creates CHAINS: if A↔B and B↔C are
      compatible, A and C merge even if |A-C| > threshold.  On dense labels
      with 30–40px row spacing and 25px threshold, this chains the entire
      table into one row (ROWS:1 bug).

      Gap-based grouping is immune to chaining because it only compares
      consecutive tokens in sorted order.  Two tokens 40px apart will
      always be in different rows regardless of intermediate tokens.

    Handles:
      - Flat rows: perp ≈ cy, behaves like y-threshold.
      - Skewed rows: perp is projected onto the normal direction.

    Returns:
        List of rows, each row is a list of token indices (0-based),
        sorted in reading order within each row.
    """
    if not tokens:
        return []

    n = len(tokens)

    # Compute perpendicular position for each token
    # For flat images (direction=(1,0)), this equals cy
    perp_positions = []
    for i, tok in enumerate(tokens):
        center = tok.get("center",
                         ((tok["x1"]+tok["x2"])/2, (tok["y1"]+tok["y2"])/2))
        normal = np.array(tok.get("normal", [0.0, 1.0]))
        # Perpendicular position = dot(center, normal)
        perp = float(np.dot(np.array(center), normal))
        perp_positions.append((perp, i))

    # Sort by perpendicular position
    perp_positions.sort(key=lambda x: x[0])

    # Group by gap
    rows = []
    current_row = [perp_positions[0][1]]
    current_perp = perp_positions[0][0]

    for k in range(1, n):
        perp_val, idx = perp_positions[k]
        if perp_val - current_perp > max_perp_px:
            rows.append(current_row)
            current_row = [idx]
        else:
            current_row.append(idx)
        current_perp = perp_val

    if current_row:
        rows.append(current_row)

    # Sort within each row by parallel position (reading order, left→right)
    for row in rows:
        if len(row) <= 1:
            continue
        ref = tokens[row[0]]
        def _sort_key(idx, _ref=ref):
            para, _ = displacement_components(_ref, tokens[idx])
            return para
        row.sort(key=_sort_key)

    # Sort rows top-to-bottom by mean perpendicular position
    rows.sort(key=lambda idxs: np.mean([
        tokens[i].get("center", ((tokens[i]["x1"]+tokens[i]["x2"])/2,
                                  (tokens[i]["y1"]+tokens[i]["y2"])/2))[1]
        for i in idxs
    ]))

    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN INDUCTION (Section E) — Gap-based grouping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def induce_columns(tokens: List[dict],
                   max_parallel_px: float = 60.0,
                   max_angle_deg: float = 15.0) -> List[List[int]]:
    """
    Cluster tokens into logical columns using gap-based grouping.

    Algorithm:
      1. Compute parallel position for each token (cx for flat images).
      2. Sort by parallel position.
      3. Walk sorted list: when gap > max_parallel_px, start new column.
      4. Sort within each column top-to-bottom.

    Same anti-chaining rationale as induce_rows.

    Returns:
        List of columns, each column is a list of token indices,
        sorted top-to-bottom within each column.
    """
    if not tokens:
        return []

    n = len(tokens)

    # Compute parallel position for each token
    # For flat images (direction=(1,0)), this equals cx
    para_positions = []
    for i, tok in enumerate(tokens):
        center = tok.get("center",
                         ((tok["x1"]+tok["x2"])/2, (tok["y1"]+tok["y2"])/2))
        direction = np.array(tok.get("direction", [1.0, 0.0]))
        para = float(np.dot(np.array(center), direction))
        para_positions.append((para, i))

    # Sort by parallel position
    para_positions.sort(key=lambda x: x[0])

    # Group by gap
    columns = []
    current_col = [para_positions[0][1]]
    current_para = para_positions[0][0]

    for k in range(1, n):
        para_val, idx = para_positions[k]
        if para_val - current_para > max_parallel_px:
            columns.append(current_col)
            current_col = [idx]
        else:
            current_col.append(idx)
        current_para = para_val

    if current_col:
        columns.append(current_col)

    # Sort within each column top-to-bottom by perpendicular position (cy)
    for col in columns:
        if len(col) <= 1:
            continue
        col.sort(key=lambda idx: tokens[idx].get("center",
            ((tokens[idx]["x1"]+tokens[idx]["x2"])/2,
             (tokens[idx]["y1"]+tokens[idx]["y2"])/2))[1])

    # Sort columns left-to-right by mean cx
    columns.sort(key=lambda idxs: np.mean([
        tokens[i].get("center", ((tokens[i]["x1"]+tokens[i]["x2"])/2,
                                  (tokens[i]["y1"]+tokens[i]["y2"])/2))[0]
        for i in idxs
    ]))

    return columns