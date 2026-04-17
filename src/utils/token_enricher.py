"""
token_enricher.py
=================
Stage 3.5 — Token Enrichment
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Enrich classified OCR tokens with geometry, structural, and matching fields.
This is the NEW stage inserted between Stage 3 (classifier) and Stage 4
(graph construction).  The enricher:

  0. Deskews tilted labels by detecting and correcting cy drift (exp36).
  1. Computes direction-aware geometry per token (center, direction, normal, angle).
  2. Induces logical rows and columns using direction-aware clustering.
  3. Classifies column roles (NUTRIENT, DOSAGE, UNIT, NRV).
  4. Detects headers and assigns context scope per dosage column.
  5. Assigns dosage stream IDs so each dosage column is an independent
     context-bearing stream.
  6. Computes rank within rows and columns for fallback matching.

INPUT
-----
List[Dict] — classified tokens from Stage 3:
    token, x1, y1, x2, y2, cx, cy, conf, label, norm
    Optionally: quad (4-corner polygon from PaddleOCR)

OUTPUT
------
List[Dict] — enriched tokens with ALL original fields PLUS:
    # Geometry
    center, width, height, direction, normal, angle_deg

    # Structure
    row_id, column_id, rank_in_row, data_rank_in_column,
    column_role, is_header, column_context_id, dosage_stream_id

    # Matching
    row_confidence, column_confidence

DESIGN DECISIONS
----------------
1. One table per image — no block detection needed.
2. Column role is inferred from majority semantic label of column members.
3. Dosage columns get independent context headers.
4. Rank is computed EXCLUDING headers (data_rank_in_column).
5. When geometry fails (insufficient tokens, near-flat ambiguity),
   rank serves as the structural fallback for association.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from src.utils.geometry_engine import (
    compute_token_geometry,
    induce_rows,
    induce_columns,
    displacement_components,
    row_compatible,
    column_compatible,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEFAULT CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CONFIG = {
    # Row induction
    "row_max_perp_px":       25.0,
    "row_max_angle_deg":     15.0,

    # Column induction
    "col_max_parallel_px":   60.0,
    "col_max_angle_deg":     15.0,

    # Column role classification
    "role_min_tokens":       2,      # min tokens in column to assign role

    # Header detection
    "header_context_labels": {"CONTEXT"},
    "header_max_row_id":     2,      # headers must be in first N rows

    # Excluded labels (NOISE tokens excluded from enrichment)
    "excluded_labels":       {"NOISE"},

    # Deskew (exp36)
    "deskew_enabled":        False,
    "deskew_min_slope":      0.02,   # ~1.1°, below this = straight (raised from 0.008)
    "deskew_max_slope":      0.15,   # ~8.5°, above this = not a table
    "deskew_min_cx_sep":     100,    # min horizontal gap for pair
    "deskew_max_cy_diff":    30,     # max cy gap for same-row pair
    "deskew_min_pairs":      8,      # min pairs for robust median (raised from 5)
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DESKEW — TILT CORRECTION (exp36, Step 0)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# On tilted or curved labels, tokens on the same visual row have
# linearly drifting cy values.  This causes downstream row/column
# induction to produce incorrect groupings.
#
# Detection: for every pair of tokens with sufficient horizontal
# separation (>100px) and small cy gap (<30px), compute the local
# slope Δcy/Δcx.  The median of all pairwise slopes gives the
# dominant tilt angle.
#
# Correction: subtract slope × (cx - cx_center) from all cy/y1/y2.
# This "levels" the table so that tokens on the same visual row
# share the same cy, enabling correct row induction.
#
# Safety: only fires when |slope| > 0.008 (~0.5°) and at least 5
# pairs agree.  Straight labels are untouched.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _deskew_tokens(tokens: List[dict], config: dict) -> List[dict]:
    """
    Detect and correct tilt in token cy values using pairwise slope
    estimation.  Modifies tokens in place and returns them.

    This runs BEFORE geometry computation and row/column induction
    so that all downstream spatial analysis works on leveled coordinates.
    """
    if not config.get("deskew_enabled", True):
        return tokens

    min_slope  = config.get("deskew_min_slope", 0.008)
    max_slope  = config.get("deskew_max_slope", 0.15)
    min_cx_sep = config.get("deskew_min_cx_sep", 100)
    max_cy_diff = config.get("deskew_max_cy_diff", 30)
    min_pairs  = config.get("deskew_min_pairs", 5)

    n = len(tokens)
    if n < 6:
        return tokens

    # Step 1: collect pairwise slopes from same-row token pairs
    slopes = []
    for i in range(n):
        cx_i = tokens[i].get("cx", 0)
        cy_i = tokens[i].get("cy", 0)
        for j in range(i + 1, n):
            cx_j = tokens[j].get("cx", 0)
            cy_j = tokens[j].get("cy", 0)

            cx_diff = abs(cx_j - cx_i)
            cy_diff = abs(cy_j - cy_i)

            if cx_diff >= min_cx_sep and cy_diff <= max_cy_diff:
                slope = (cy_j - cy_i) / (cx_j - cx_i)
                if abs(slope) < max_slope:
                    slopes.append(slope)

    if len(slopes) < min_pairs:
        return tokens

    # Step 2: median slope (robust to outlier cross-row pairs)
    slopes.sort()
    median_slope = slopes[len(slopes) // 2]

    if abs(median_slope) < min_slope:
        return tokens  # label is straight

    # Step 3: apply correction
    all_cx = [t.get("cx", 0) for t in tokens]
    cx_center = sum(all_cx) / len(all_cx)

    for tok in tokens:
        cx = tok.get("cx", 0)
        correction = median_slope * (cx - cx_center)
        tok["cy"] = tok.get("cy", 0) - correction
        tok["y1"] = tok.get("y1", 0) - correction
        tok["y2"] = tok.get("y2", 0) - correction

    logger.info(f"Deskew applied: slope={median_slope:.4f} "
                f"({len(slopes)} pairs), cx_center={cx_center:.0f}")

    return tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN ROLE CLASSIFICATION (Section G)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _classify_column_role(tokens: List[dict], indices: List[int]) -> str:
    """
    Infer the role of a column from the majority semantic label of
    its non-header members.

    Roles:
        NUTRIENT — majority NUTRIENT tokens (leftmost name column)
        DOSAGE   — majority QUANTITY tokens (value columns)
        UNIT     — majority UNIT tokens
        NRV      — QUANTITY tokens where most have '%' or NRV context
        MIXED    — no clear majority
    """
    labels = [tokens[i].get("label", "UNKNOWN") for i in indices
              if not tokens[i].get("is_header", False)]

    if not labels:
        return "MIXED"

    counts = Counter(labels)
    total = len(labels)
    top_label, top_count = counts.most_common(1)[0]

    # Majority threshold: >40% of column tokens share a label
    if top_count / total < 0.4:
        return "MIXED"

    role_map = {
        "NUTRIENT": "NUTRIENT",
        "QUANTITY": "DOSAGE",
        "UNIT":     "UNIT",
        "CONTEXT":  "DOSAGE",   # context-only columns are dosage headers
    }
    return role_map.get(top_label, "MIXED")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER DETECTION (Section F)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_headers(tokens: List[dict], rows: List[List[int]],
                    columns: List[List[int]], config: dict) -> None:
    """
    Mark header tokens and assign column_context_id.

    Header detection logic:
      1. CONTEXT-labeled tokens in the first N rows are headers.
      2. Each header's column_context_id = its norm (per_100g, per_serving, etc.).
      3. The header's context propagates to ALL tokens in the same column
         (column_context_id field).

    Handles multi-dosage-column tables:
      column 2 header = "per 100g"  → column_context_id = "per_100g"
      column 3 header = "per serving" → column_context_id = "per_serving"
      Both map to the same nutrient column independently.
    """
    max_header_row = config.get("header_max_row_id", 2)
    context_labels = config.get("header_context_labels", {"CONTEXT"})

    # Build column lookup: token_index → column_id
    col_lookup = {}
    for col_id, col_indices in enumerate(columns):
        for idx in col_indices:
            col_lookup[idx] = col_id

    # Build row lookup: token_index → row_id
    row_lookup = {}
    for row_id, row_indices in enumerate(rows):
        for idx in row_indices:
            row_lookup[idx] = row_id

    # Phase 1: detect headers
    # Map: column_id → context string
    column_contexts: Dict[int, str] = {}

    for idx, tok in enumerate(tokens):
        tok_row = row_lookup.get(idx, 999)
        tok_col = col_lookup.get(idx, -1)
        tok_label = tok.get("label", "UNKNOWN")

        if tok_label in context_labels and tok_row <= max_header_row:
            tok["is_header"] = True
            context_str = tok.get("norm") or tok.get("token", "").lower()
            tok["column_context_id"] = context_str

            if tok_col >= 0:
                column_contexts[tok_col] = context_str
            logger.debug(f"Header detected: '{tok.get('token')}' → "
                         f"context='{context_str}' col={tok_col}")

    # Phase 2: propagate context to all tokens in the column
    for col_id, context_str in column_contexts.items():
        for idx in columns[col_id]:
            tokens[idx]["column_context_id"] = context_str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOSAGE STREAM ASSIGNMENT (Section K)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _assign_dosage_streams(tokens: List[dict], columns: List[List[int]]) -> int:
    """
    Assign dosage_stream_id to each token in a DOSAGE-role column.

    Each DOSAGE column is an INDEPENDENT stream.  This is critical for
    multi-column tables where column 2 and column 3 have different angles,
    different contexts, and must be mapped to the nutrient column
    independently.

    Returns:
        Number of dosage streams detected.
    """
    stream_id = 0
    for col_indices in columns:
        if not col_indices:
            continue
        # Check if this column is DOSAGE
        role = tokens[col_indices[0]].get("column_role", "MIXED")
        if role != "DOSAGE":
            continue

        for idx in col_indices:
            tokens[idx]["dosage_stream_id"] = stream_id
        stream_id += 1

    return stream_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RANK COMPUTATION (Section H)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_ranks(tokens: List[dict], rows: List[List[int]],
                   columns: List[List[int]]) -> None:
    """
    Compute positional ranks for structure-based matching.

    rank_in_row:
        0-indexed position within the row (reading order).

    data_rank_in_column:
        0-indexed position within the column, EXCLUDING headers.

    qty_rank_in_column:
        0-indexed position counting ONLY QUANTITY tokens in the column.
        This is the PRIMARY rank for association: nutrient at
        data_rank N should match the quantity at qty_rank N.

        Why label-specific rank matters:
        In a dosage column, QUANTITY and UNIT tokens interleave:
          QUANTITY "1500"  data_rank=0  qty_rank=0
          UNIT     "kJ"   data_rank=1  qty_rank=-1
          QUANTITY "20"   data_rank=2  qty_rank=1
          UNIT     "g"    data_rank=3  qty_rank=-1
        NUTRIENT column has ranks 0,1,2 (one per nutrient).
        data_rank alignment fails (0→0 ok, 1→2, 2→4).
        qty_rank alignment succeeds (0→0, 1→1, 2→2).
    """
    # rank_in_row
    for row_indices in rows:
        for rank, idx in enumerate(row_indices):
            tokens[idx]["rank_in_row"] = rank

    # data_rank_in_column (headers excluded)
    for col_indices in columns:
        data_rank = 0
        for idx in col_indices:
            if tokens[idx].get("is_header", False):
                tokens[idx]["data_rank_in_column"] = -1
                continue
            tokens[idx]["data_rank_in_column"] = data_rank
            data_rank += 1

    # qty_rank_in_column (QUANTITY tokens only, for dosage columns)
    # nutrient_rank_in_column (NUTRIENT tokens only, for nutrient column)
    for col_indices in columns:
        qty_rank = 0
        nut_rank = 0
        for idx in col_indices:
            tokens[idx]["qty_rank_in_column"] = -1
            tokens[idx]["nutrient_rank_in_column"] = -1

            if tokens[idx].get("is_header", False):
                continue

            label = tokens[idx].get("label", "")
            if label == "QUANTITY":
                tokens[idx]["qty_rank_in_column"] = qty_rank
                qty_rank += 1
            elif label == "NUTRIENT":
                tokens[idx]["nutrient_rank_in_column"] = nut_rank
                nut_rank += 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENRICHMENT PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TokenEnricher:
    """
    Stage 3.5 — Token Enrichment.

    Takes classified tokens and adds geometry, structure, column roles,
    headers, dosage streams, and ranks.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.diagnostics: dict = {}

    def enrich(self, classified_tokens: List[dict]) -> List[dict]:
        """
        Enrich a list of classified tokens.

        Args:
            classified_tokens: output of Stage 3 (SemanticClassifier).
                Required fields: token, x1, y1, x2, y2, label
                Optional: cx, cy, conf, norm, quad

        Returns:
            List of enriched tokens with all structural fields populated.
            NOISE tokens are excluded from structural enrichment but
            preserved in output with is_enriched=False.
        """
        # ── Step 0: Deskew tilted labels (exp36) ──────────────────────
        # Must run BEFORE geometry/row/column induction so that all
        # downstream spatial analysis works on leveled coordinates.
        classified_tokens = _deskew_tokens(classified_tokens, self.config)

        excluded = self.config["excluded_labels"]

        # Separate active tokens from noise
        active_indices = []
        for i, tok in enumerate(classified_tokens):
            if tok.get("label") not in excluded:
                active_indices.append(i)

        # ── Step 1: Geometry ──────────────────────────────────────────
        active_tokens = []
        for i in active_indices:
            tok = {**classified_tokens[i]}  # shallow copy
            geo = compute_token_geometry(tok)
            tok.update(geo)
            tok["is_enriched"] = True
            tok["is_header"] = False
            tok["column_context_id"] = None
            tok["dosage_stream_id"] = -1
            tok["row_id"] = -1
            tok["column_id"] = -1
            tok["rank_in_row"] = -1
            tok["data_rank_in_column"] = -1
            tok["qty_rank_in_column"] = -1
            tok["nutrient_rank_in_column"] = -1
            tok["column_role"] = "UNKNOWN"
            tok["row_confidence"] = 0.0
            tok["column_confidence"] = 0.0
            active_tokens.append(tok)

        n_active = len(active_tokens)
        logger.info(f"Enriching {n_active} tokens "
                     f"({len(classified_tokens) - n_active} NOISE excluded)")

        if n_active == 0:
            self.diagnostics = {"active_tokens": 0}
            return self._rebuild_output(classified_tokens, active_tokens,
                                         active_indices)

        # ── Auto-scale thresholds by image resolution ─────────────────
        # Default thresholds are tuned for ~500px image height.
        # High-res images (1600px+) have proportionally larger pixel
        # distances between rows/columns — thresholds must scale too.
        max_y = max((t.get("y2", 0) for t in classified_tokens), default=500)
        scale = max(max_y / 500.0, 1.0)   # cap at 2.5x   # never shrink below defaults
        row_perp = self.config["row_max_perp_px"] * scale
        col_para = self.config["col_max_parallel_px"] * scale
        # Store scaled values for _compute_confidence
        self._scaled_row_perp = row_perp
        self._scaled_col_para = col_para
        logger.info(f"Resolution scale: {scale:.2f}x (max_y={max_y}) "
                     f"→ row_perp={row_perp:.0f}px, col_para={col_para:.0f}px")

        # ── Step 2: Row induction ─────────────────────────────────────
        rows = induce_rows(
            active_tokens,
            max_perp_px=row_perp,
            max_angle_deg=self.config["row_max_angle_deg"],
        )
        for row_id, row_indices in enumerate(rows):
            for idx in row_indices:
                active_tokens[idx]["row_id"] = row_id

        # ── Step 3: Column induction ──────────────────────────────────
        columns = induce_columns(
            active_tokens,
            max_parallel_px=col_para,
            max_angle_deg=self.config["col_max_angle_deg"],
        )
        for col_id, col_indices in enumerate(columns):
            for idx in col_indices:
                active_tokens[idx]["column_id"] = col_id

        # ── Step 4: Column role classification ────────────────────────
        for col_id, col_indices in enumerate(columns):
            if len(col_indices) < self.config["role_min_tokens"]:
                role = "SINGLETON"
            else:
                role = _classify_column_role(active_tokens, col_indices)
            for idx in col_indices:
                active_tokens[idx]["column_role"] = role

        # ── Step 5: Header detection + context scope ──────────────────
        _detect_headers(active_tokens, rows, columns, self.config)

        # ── Step 6: Dosage stream assignment ──────────────────────────
        n_streams = _assign_dosage_streams(active_tokens, columns)

        # ── Step 7: Rank computation ──────────────────────────────────
        _compute_ranks(active_tokens, rows, columns)

        # ── Step 8: Compute confidence scores ─────────────────────────
        self._compute_confidence(active_tokens, rows, columns)

        # ── Diagnostics ───────────────────────────────────────────────
        self.diagnostics = self._build_diagnostics(
            active_tokens, rows, columns, n_streams)

        return self._rebuild_output(classified_tokens, active_tokens,
                                     active_indices)

    # ── Internal helpers ──────────────────────────────────────────────

    def _compute_confidence(self, tokens: List[dict],
                            rows: List[List[int]],
                            columns: List[List[int]]) -> None:
        """
        Compute row_confidence and column_confidence for each token.

        row_confidence: mean row compatibility score with row neighbors.
        column_confidence: mean column compatibility score with column neighbors.
        """
        for row_indices in rows:
            for i, idx_a in enumerate(row_indices):
                scores = []
                for j, idx_b in enumerate(row_indices):
                    if i == j:
                        continue
                    _, conf = row_compatible(
                        tokens[idx_a], tokens[idx_b],
                        self._scaled_row_perp,
                        self.config["row_max_angle_deg"])
                    scores.append(conf)
                tokens[idx_a]["row_confidence"] = (
                    round(sum(scores) / len(scores), 4) if scores else 1.0)

        for col_indices in columns:
            for i, idx_a in enumerate(col_indices):
                scores = []
                for j, idx_b in enumerate(col_indices):
                    if i == j:
                        continue
                    _, conf = column_compatible(
                        tokens[idx_a], tokens[idx_b],
                        self._scaled_col_para,
                        self.config["col_max_angle_deg"])
                    scores.append(conf)
                tokens[idx_a]["column_confidence"] = (
                    round(sum(scores) / len(scores), 4) if scores else 1.0)

    def _rebuild_output(self, original: List[dict],
                        enriched: List[dict],
                        active_indices: List[int]) -> List[dict]:
        """
        Rebuild the full token list, interleaving enriched tokens
        at their original positions and marking NOISE as not enriched.
        """
        output = []
        enriched_map = {active_indices[i]: enriched[i]
                        for i in range(len(enriched))}

        for i, tok in enumerate(original):
            if i in enriched_map:
                output.append(enriched_map[i])
            else:
                out_tok = {**tok}
                out_tok["is_enriched"] = False
                output.append(out_tok)

        return output

    def _build_diagnostics(self, tokens: List[dict],
                           rows: List[List[int]],
                           columns: List[List[int]],
                           n_streams: int) -> dict:
        """Build diagnostics dict for pipeline inspection."""
        col_roles = {}
        col_angles = {}
        col_contexts = {}

        for col_id, col_indices in enumerate(columns):
            if col_indices:
                role = tokens[col_indices[0]].get("column_role", "?")
                col_roles[col_id] = role
                angles = [tokens[i].get("angle_deg", 0) for i in col_indices]
                col_angles[col_id] = round(sum(angles) / len(angles), 2)
                ctx = tokens[col_indices[0]].get("column_context_id")
                if ctx:
                    col_contexts[col_id] = ctx

        # Rank consistency: check if nutrient col and dosage cols have
        # same number of data-ranked tokens
        nutrient_cols = [cid for cid, r in col_roles.items() if r == "NUTRIENT"]
        dosage_cols = [cid for cid, r in col_roles.items() if r == "DOSAGE"]
        rank_counts = {}
        for col_id, col_indices in enumerate(columns):
            data_count = sum(1 for i in col_indices
                             if tokens[i].get("data_rank_in_column", -1) >= 0)
            rank_counts[col_id] = data_count

        rank_consistent = True
        if nutrient_cols and dosage_cols:
            nut_count = rank_counts.get(nutrient_cols[0], 0)
            for dc in dosage_cols:
                if rank_counts.get(dc, 0) != nut_count:
                    rank_consistent = False
                    break

        diag = {
            "active_tokens":       len(tokens),
            "num_rows":            len(rows),
            "num_columns":         len(columns),
            "column_roles":        col_roles,
            "column_angles":       col_angles,
            "column_contexts":     col_contexts,
            "dosage_streams":      n_streams,
            "rank_counts":         rank_counts,
            "rank_consistent":     rank_consistent,
            "headers_detected":    sum(1 for t in tokens if t.get("is_header")),
        }
        logger.info(f"Enrichment diagnostics: {diag}")
        return diag

    # ── Debug ─────────────────────────────────────────────────────────

    def print_diagnostics(self) -> None:
        d = self.diagnostics
        W = 60
        print(f"\n{'='*W}")
        print("TOKEN ENRICHER DIAGNOSTICS")
        print(f"{'='*W}")
        print(f"  Active tokens     : {d.get('active_tokens', 0)}")
        print(f"  Logical rows      : {d.get('num_rows', 0)}")
        print(f"  Logical columns   : {d.get('num_columns', 0)}")
        print(f"  Dosage streams    : {d.get('dosage_streams', 0)}")
        print(f"  Headers detected  : {d.get('headers_detected', 0)}")
        print(f"  Rank consistent   : {d.get('rank_consistent', '?')}")

        roles = d.get("column_roles", {})
        angles = d.get("column_angles", {})
        contexts = d.get("column_contexts", {})
        ranks = d.get("rank_counts", {})

        if roles:
            print(f"\n  {'COL':<5} {'ROLE':<12} {'ANGLE':<10} "
                  f"{'RANKS':<8} {'CONTEXT'}")
            print(f"  {'-'*50}")
            for col_id in sorted(roles.keys()):
                print(f"  {col_id:<5} {roles[col_id]:<12} "
                      f"{angles.get(col_id, '?'):<10} "
                      f"{ranks.get(col_id, '?'):<8} "
                      f"{contexts.get(col_id, '—')}")
        print(f"{'='*W}\n")