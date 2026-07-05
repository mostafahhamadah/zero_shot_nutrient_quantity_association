"""
graph_constructor_v2.py
=======================
Stage 4 — Geometry-Aware Semantic Graph Construction
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Build a typed semantic graph over ENRICHED tokens (output of Stage 3.5).
This replaces the original graph_constructor.py which used raw pixel
thresholds for SAME_ROW and SAME_COL.

EDGE CONSTRUCTION MODES (exp39)
-------------------------------
ROW_COMPAT and COL_COMPAT can each be built several ways, selected by config:

  row_edge_mode:
    "cy"                    ROW_COMPAT = |cy_a - cy_b| <= same_row_threshold.
                            Pairwise, geometric. Cannot chain across a gap.
    "rank"                  ROW_COMPAT = equal data_rank_in_column in DIFFERENT
                            columns. Breaks on interleaved units — kept for ablation.
    "role_rank"             ROW_COMPAT = equal LABEL-SPECIFIC rank:
                            NUTRIENT k <-> QUANTITY k <-> UNIT k, same OR
                            different column. Pure-structural rows.
    "role_rank_cy"          HYBRID: link if role-ranks match (structural) OR the
                            tokens are on the same line (|cy_a - cy_b| <= thresh).
                            cy recovers quantities lost to rank slips (value sits
                            on the nutrient's row but column/rank shifted), while
                            role-rank keeps unit linkage and skew tolerance.
    "overlap"     (default) INTERSECTION: ROW_COMPAT if the two tokens' VERTICAL
                            extents [y1, y2] overlap by >= row_overlap_min of the
                            shorter box. Swept y-coverage instead of center
                            distance, so a short token (e.g. a unit) inside a
                            taller nutrient's vertical span still links.

  col_edge_mode:
    "cx"                    COL_COMPAT = |cx_a - cx_b| <= same_col_threshold.
                            Pairwise, geometric.
    "column_id"             COL_COMPAT = shared enricher column_id. Structural.
    "overlap"     (default) INTERSECTION: COL_COMPAT if the two tokens' HORIZONTAL
                            extents [x1, x2] overlap by >= col_overlap_min of the
                            shorter box. Swept x-coverage instead of center distance.

CONTEXT_SCOPE uses V1-style vertical distance (context above data). Its LATERAL
gate is selected by context_scope_mode:
    "cy"                    no lateral gate — context governs every node below it
                            within ctx_max_y (previous V1 behaviour).
    "overlap"     (default) INTERSECTION: context governs a node only if their
                            HORIZONTAL extents [x1, x2] overlap by
                            >= context_overlap_min, i.e. the context's horizontal
                            line covers the token's column (replaces nearest-cx).
HEADER_SCOPE uses structural column_id (context header -> same column).

NOTE (default = "overlap" trio): the defaults below set row_edge_mode,
col_edge_mode and context_scope_mode all to "overlap", with the tuned thresholds
(row 0.30 / col 0.20 / context 0.10) from the coordinate-descent overlap sweep
(cy/cx 40.7% -> overlap 42.3% 4F-F1). To recover the previous geometric-centroid
behaviour set row_edge_mode="cy", col_edge_mode="cx", context_scope_mode="cy".
The three flags are independent and stay composable for ablation. Unit linkage is
unchanged: units are ordinary nodes riding ROW_COMPAT/COL_COMPAT, and the
downstream unit picker (association_v2) is untouched.

NODE SCHEMA
-----------
Nodes carry ALL enriched fields. Graph consumers access structural
fields (row_id, column_id, column_role, dosage_stream_id, rank, etc.)
directly from nodes -- no recomputation needed.

EDGE TYPES
----------
  ROW_COMPAT       Tokens on the same logical row.
                   Built per row_edge_mode ("cy" | "rank" | "role_rank" |
                   "role_rank_cy" | "overlap").
                   Weight = cy-proximity score (geometric link), 1.0 (rank link),
                   or vertical-overlap fraction ("overlap"). Bidirectional.

  COL_COMPAT       Tokens in the same logical column.
                   Built per col_edge_mode ("cx" geometric | "column_id"
                   structural | "overlap").
                   Weight = cx-proximity score ("cx"), 1.0 ("column_id"), or
                   horizontal-overlap fraction ("overlap"). Bidirectional.

  DIRECTIONAL_ADJ  Tokens physically close but NOT in same row or column.
                   Weight = inverse of bbox gap distance. Bidirectional.

  CONTEXT_SCOPE    CONTEXT token -> data nodes below it (vertical scope).
                   Lateral gate per context_scope_mode ("cy" none | "overlap").
                   Directed: context -> governed token.

  HEADER_SCOPE     CONTEXT header -> tokens in the same column (structural).
                   Directed: header -> governed token.
"""

from __future__ import annotations

import json
import math
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.geometry_engine import (
    direction_compatible,
    displacement_components,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    # Row/column edge thresholds (used by the geometric "cy"/"cx" modes,
    # and by the cy half of "role_rank_cy")
    "same_row_threshold":      25,
    "same_col_threshold":      20,

    # ── Edge construction modes (exp39) ───────────────────────────────
    #   row_edge_mode: "cy"           -> |cy_a - cy_b| <= thresh
    #                  "rank"         -> equal data_rank across columns
    #                  "role_rank"    -> equal label-specific rank (NUT/QTY/UNIT)
    #                  "role_rank_cy" -> role_rank OR cy  (hybrid; fixes qty slips)
    #                  "overlap"      -> vertical bbox-extent overlap >= row_overlap_min   (default)
    #                  "overlap_mnn"  -> overlap candidates, then competitive mutual-nearest pruning
    #   col_edge_mode: "cx"           -> |cx_a - cx_b| <= thresh
    #                  "column_id"    -> shared enricher column_id
    #                  "overlap"      -> horizontal bbox-extent overlap >= col_overlap_min (default)
    "row_edge_mode":           "overlap",
    "col_edge_mode":           "overlap",

    # ── Overlap-mode thresholds (used only by the "overlap" modes) ────
    # Minimum swept coverage fraction (overlap_length / shorter_segment)
    # required to create an edge. Range (0, 1]. Tune per-axis if the
    # overlap mode under-/over-connects (raise for precision, lower for recall).
    # Defaults are the coordinate-descent sweep result (cy/cx 40.7% -> 42.3% 4F-F1).
    "row_overlap_min":         0.30,   # ROW: vertical   [y1,y2] coverage  (decisive axis)
    "col_overlap_min":         0.20,   # COL: horizontal [x1,x2] coverage  (inert in sweep)
    "context_overlap_min":     0.10,   # CONTEXT lateral [x1,x2] coverage  (inert in sweep)

    # Overlap denominator — how the swept coverage fraction is normalised:
    #   "min" (default) overlap/shorter  (most permissive; the adopted setting)
    #   "max"           overlap/longer   (most strict; penalises size mismatch)
    #   "iou"           overlap/union    (1-D IoU; intermediate, symmetric)
    #   "mean"          overlap/mean-len
    # Changing this rescales the thresholds above, so re-tune per denominator.
    "overlap_denom":           "min",

    # Competitive row pruning — only used by row_edge_mode = "overlap_mnn".
    # After building the overlap row candidates, keep an edge a-b only if its
    # overlap weight is within mnn_tol of EACH endpoint's best row edge to a node
    # of the other's label. Smaller = stricter (closer to one-to-one). Per-label
    # competition keeps legitimate same-row multiples (e.g. per_100g + per_serving)
    # while dropping leakage to an adjacent row. Pair with a LOOSE row_overlap_min
    # so there are adjacent-row candidates for MNN to compete away.
    "mnn_tol":                  0.10,

    # Context scope lateral gate: "overlap" (default) | "cy" (no lateral gate)
    "context_scope_mode":      "overlap",

    # Directional adjacency
    "adj_gap_max_px":          60,
    "adj_max_angle_deg":       15.0,

    # Context scope (V1-style vertical distance)
    "context_scope_max_y":     600,

    # Excluded labels
    "excluded_labels":         {"NOISE"},
}


def _bbox_gap(a: dict, b: dict) -> float:
    """Euclidean gap between two bounding boxes (0 if overlapping)."""
    h_gap = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
    v_gap = max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
    return math.sqrt(h_gap ** 2 + v_gap ** 2)


def _interval_overlap(a_min: float, a_max: float,
                      b_min: float, b_max: float) -> float:
    """Length of the 1-D overlap between [a_min, a_max] and [b_min, b_max].
    0.0 if the intervals are disjoint."""
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _overlap_ratio(a_min: float, a_max: float,
                   b_min: float, b_max: float,
                   denom: str = "min") -> float:
    """
    Swept coverage fraction of two 1-D segments — the 'intersection' criterion
    used by row_edge_mode / col_edge_mode = 'overlap' and the 'overlap' context
    lateral gate.

        ratio = overlap_length / D

    where the denominator D selects how strict the criterion is (config key
    `overlap_denom`):

        "min"  (default) D = min(len_a, len_b)  -> overlap / shorter segment.
                         MOST PERMISSIVE: a small box fully inside a larger one
                         scores ~1.0 (rescues a short unit/quantity box onto a
                         taller nutrient row). Equivalent to
                         max(overlap/len_a, overlap/len_b).
        "max"            D = max(len_a, len_b)  -> overlap / longer segment.
                         MOST STRICT: a size-mismatched pair scores low even when
                         the short box is fully covered (penalises tiny fragments).
        "iou"            D = len_a + len_b - overlap  -> 1-D intersection-over-union.
                         INTERMEDIATE: symmetric, penalises both size mismatch and
                         partial overlap.
        "mean"           D = (len_a + len_b) / 2.

    All variants lie in (0, 1] (overlap <= min(len) <= every denominator).
    Returns 0.0 when the segments are disjoint or the denominator is degenerate.
    """
    overlap = _interval_overlap(a_min, a_max, b_min, b_max)
    if overlap <= 0.0:
        return 0.0
    len_a = a_max - a_min
    len_b = b_max - b_min
    if denom == "max":
        d = max(len_a, len_b)
    elif denom == "iou":
        d = len_a + len_b - overlap          # union
    elif denom == "mean":
        d = 0.5 * (len_a + len_b)
    else:                                     # "min" (default, most permissive)
        d = min(len_a, len_b)
    if d <= 0.0:
        return 0.0
    return overlap / d


def _role_rank_value(node: dict) -> int:
    """
    Label-specific ordinal rank used by row_edge_mode='role_rank'/'role_rank_cy'.

    Returns the rank that counts only tokens of this node's own role:
        NUTRIENT -> nutrient_rank_in_column
        QUANTITY -> qty_rank_in_column
        UNIT     -> unit_rank_in_column
        (anything else, incl. CONTEXT/NRV) -> -1  (not row-linked by rank)

    Two tokens with equal, non-negative role-rank occupy the same ordinal
    slot within their respective roles and are treated as one logical row.
    """
    label = node.get("label", "")
    if label == "NUTRIENT":
        return node.get("nutrient_rank_in_column", -1)
    if label == "QUANTITY":
        return node.get("qty_rank_in_column", -1)
    if label == "UNIT":
        return node.get("unit_rank_in_column", -1)
    return -1


def _prune_row_mnn(edges: List[dict], nodes: List[dict], tol: float) -> List[dict]:
    """
    Competitive ('mutual-nearest') pruning of ROW_COMPAT edges — the post-pass
    for row_edge_mode='overlap_mnn'. Keeps a row edge a-b only if its overlap
    weight is within `tol` of BOTH endpoints' best row edge to a node of the
    other endpoint's label:

        keep(a, b)  iff  w(a, b) >= wmax[a][label_b] - tol
                    and  w(a, b) >= wmax[b][label_a] - tol

    Using the per-DESTINATION-LABEL maximum (not a single global nearest) is the
    crucial choice: a nutrient carrying both a per_100g and a per_serving quantity
    overlaps both almost equally, so both clear wmax[nutrient][QUANTITY]-tol and
    survive, while a weaker link to a quantity that actually sits on an adjacent
    row (lower overlap) is dropped. Symmetric in (a, b) at equal weight, so the
    two directed copies of an undirected pair share the keep/drop decision.
    Singletons (a node with one row edge of a given label) are always kept.
    Non-ROW_COMPAT edges pass through untouched.
    """
    node_label = {n["id"]: n.get("label", "") for n in nodes}
    row_e   = [e for e in edges if e.get("type") == "ROW_COMPAT"]
    other_e = [e for e in edges if e.get("type") != "ROW_COMPAT"]
    if not row_e:
        return edges

    # wmax[src][dst_label] = best overlap weight from src to any node of dst_label
    wmax: Dict[int, Dict[str, float]] = {}
    for e in row_e:
        a, lab_b, w = e["src"], node_label.get(e["dst"], ""), e.get("weight", 0.0)
        d = wmax.setdefault(a, {})
        if w > d.get(lab_b, -1.0):
            d[lab_b] = w

    kept = []
    for e in row_e:
        a, b, w = e["src"], e["dst"], e.get("weight", 0.0)
        la, lb = node_label.get(a, ""), node_label.get(b, "")
        a_best = wmax.get(a, {}).get(lb, w)   # defaults to w -> singleton kept
        b_best = wmax.get(b, {}).get(la, w)
        if w >= a_best - tol and w >= b_best - tol:
            kept.append(e)
    return other_e + kept


class GraphConstructorV2:
    """
    Build a typed semantic graph from enriched tokens.

    Nodes = enriched tokens (all fields preserved).
    Edges = ROW_COMPAT, COL_COMPAT, DIRECTIONAL_ADJ, CONTEXT_SCOPE, HEADER_SCOPE.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def build(self, enriched_tokens: List[dict]) -> dict:
        """
        Build graph from Stage 3.5 enriched tokens.

        EDGE BUILDING STRATEGY:
          ROW_COMPAT / COL_COMPAT are built according to row_edge_mode /
          col_edge_mode (see module docstring). DEFAULT is "overlap" on all three
          axes (row/col/context).

          Geometric modes ("cy" / "cx") use V1-style DIRECT PAIRWISE comparison
          (|cy_a - cy_b| <= threshold). Pairwise comparison cannot chain --
          two tokens 40px apart in cy never get a ROW_COMPAT edge regardless of
          intermediate tokens.

          Structural row modes:
            "rank"         equal data_rank across DIFFERENT columns. Fails on
                           interleaved units (data_rank misaligns).
            "role_rank"    equal LABEL-SPECIFIC rank (nutrient/qty/unit), same OR
                           different column. Restores nutrient->quantity and
                           quantity->unit (incl. inline units). Pure-structural,
                           but a rank slip (dropped/fused token, mis-split column)
                           pairs a nutrient with the wrong-rank quantity.
            "role_rank_cy" HYBRID: role_rank OR cy. The cy half reconnects a
                           quantity that sits on the nutrient's visual row even
                           when its rank slipped — which is the dominant cause of
                           quantity errors under pure role_rank — while the
                           role_rank half preserves unit linkage and skew
                           tolerance. NOT pure-structural (reintroduces cy on the
                           row axis); use "role_rank" for the pure-structural number.

          Overlap (intersection) modes — DEFAULT:
            row "overlap"  ROW_COMPAT if the VERTICAL extents [y1, y2] of the two
                           boxes overlap by >= row_overlap_min of the shorter box.
                           Pairwise and geometric like "cy", but tolerant to
                           box-height differences (small token inside a tall row).
                           Weight = the overlap fraction.
            col "overlap"  COL_COMPAT if the HORIZONTAL extents [x1, x2] overlap by
                           >= col_overlap_min of the shorter box. Weight = fraction.

          Structural modes ride on the enricher's induce_rows / induce_columns
          output, so column-induction errors propagate into the graph.

          CONTEXT_SCOPE uses V1-style vertical distance (context above data). Its
          lateral gate is set by context_scope_mode: "overlap" (default) requires
          the context and the token to share >= context_overlap_min horizontal-
          extent overlap (the context's horizontal line must cover the token's
          column); "cy" applies no lateral gate (previous behaviour).
          HEADER_SCOPE uses structural column_id (context header -> same column).
          Both are emitted -- association can use whichever is available.

        All geometric thresholds are auto-scaled by image resolution. Overlap-mode
        thresholds are fractions and are NOT scaled (they are resolution-invariant).
        """
        excluded = self.config["excluded_labels"]

        # Filter to enriched, non-noise tokens
        nodes = []
        for i, tok in enumerate(enriched_tokens):
            if tok.get("label") in excluded:
                continue
            if not tok.get("is_enriched", False):
                continue
            node = {**tok, "id": i}
            nodes.append(node)

        node_ids = {n["id"] for n in nodes}
        n = len(nodes)

        # ── Auto-scale thresholds by image resolution ─────────────────
        max_y = max((nd.get("y2", 0) for nd in nodes), default=500)
        # REVERT both token_enricher.py and graph_constructor_v2.py back to:
        scale = max(max_y / 500.0, 1.0)
        row_thresh = self.config["same_row_threshold"] * scale
        col_thresh = self.config["same_col_threshold"] * scale
        adj_gap    = self.config["adj_gap_max_px"]     * scale
        ctx_max_y  = self.config["context_scope_max_y"] * scale

        row_mode = self.config.get("row_edge_mode", "overlap")
        col_mode = self.config.get("col_edge_mode", "overlap")
        ctx_mode = self.config.get("context_scope_mode", "overlap")

        # Overlap-mode fractions (resolution-invariant, not scaled)
        row_overlap_min     = self.config.get("row_overlap_min", 0.30)
        col_overlap_min     = self.config.get("col_overlap_min", 0.20)
        context_overlap_min = self.config.get("context_overlap_min", 0.10)
        overlap_denom       = self.config.get("overlap_denom", "min")
        mnn_tol             = self.config.get("mnn_tol", 0.10)

        edges = []

        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                a_id, b_id = a["id"], b["id"]

                # Centres hoisted here: CONTEXT_SCOPE needs cy regardless of
                # which row/col mode is active.
                a_cy = a.get("cy", (a["y1"] + a["y2"]) / 2.0)
                b_cy = b.get("cy", (b["y1"] + b["y2"]) / 2.0)
                a_cx = a.get("cx", (a["x1"] + a["x2"]) / 2.0)
                b_cx = b.get("cx", (b["x1"] + b["x2"]) / 2.0)

                has_row = False
                has_col = False

                # ── ROW_COMPAT ────────────────────────────────────────
                if row_mode == "role_rank_cy":
                    # HYBRID: link if role-ranks match (structural) OR the two
                    # tokens are on the same line (geometric cy).
                    #   - rank half  -> nutrient<->quantity<->unit, inline units,
                    #                   skew tolerance (same as role_rank).
                    #   - cy half    -> reconnects a quantity that sits on the
                    #                   nutrient's visual row even when rank/column
                    #                   slipped (the main quantity failure under
                    #                   pure role_rank). Structural edge wins the
                    #                   weight (1.0); cy edge keeps proximity weight.
                    ra = _role_rank_value(a)
                    rb = _role_rank_value(b)
                    rank_hit = (ra >= 0 and ra == rb)
                    cy_hit   = abs(a_cy - b_cy) <= row_thresh
                    if rank_hit or cy_hit:
                        w = 1.0 if rank_hit else round(1.0 - abs(a_cy - b_cy) / row_thresh, 4)
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "ROW_COMPAT", "weight": w})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "ROW_COMPAT", "weight": w})
                        has_row = True
                elif row_mode == "role_rank":
                    # Structural, role-pure: connect tokens at the SAME
                    # ordinal position WITHIN THEIR ROLE.
                    #   NUTRIENT k  <->  QUANTITY k  <->  UNIT k
                    # Same-column pairs ARE allowed, so a quantity and its
                    # inline unit (one dosage column, qty_rank k / unit_rank k)
                    # still get linked — the case plain "rank" loses.
                    ra = _role_rank_value(a)
                    rb = _role_rank_value(b)
                    if ra >= 0 and ra == rb:
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "ROW_COMPAT", "weight": 1.0})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "ROW_COMPAT", "weight": 1.0})
                        has_row = True
                elif row_mode == "rank":
                    # Structural: tokens at the SAME data_rank in DIFFERENT
                    # columns belong to the same logical row.
                    a_dr  = a.get("data_rank_in_column", -1)
                    b_dr  = b.get("data_rank_in_column", -1)
                    a_col = a.get("column_id", -1)
                    b_col = b.get("column_id", -1)
                    if (a_dr >= 0 and a_dr == b_dr
                            and a_col >= 0 and b_col >= 0 and a_col != b_col):
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "ROW_COMPAT", "weight": 1.0})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "ROW_COMPAT", "weight": 1.0})
                        has_row = True
                elif row_mode == "cy":
                    # V1-style direct cy comparison (geometric centroid).
                    if abs(a_cy - b_cy) <= row_thresh:
                        w = round(1.0 - abs(a_cy - b_cy) / row_thresh, 4)
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "ROW_COMPAT", "weight": w})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "ROW_COMPAT", "weight": w})
                        has_row = True
                else:  # "overlap" (default) and "overlap_mnn" candidate stage
                    # Same row if the two boxes' VERTICAL extents ([y1, y2])
                    # overlap by >= row_overlap_min of the SHORTER box. Swept
                    # y-coverage instead of center distance, so a short token
                    # (e.g. a unit) inside a taller nutrient's vertical span
                    # still links. Weight = the overlap fraction. For
                    # "overlap_mnn" these are CANDIDATES; the competitive
                    # _prune_row_mnn() post-pass thins them after the loop.
                    ratio = _overlap_ratio(a["y1"], a["y2"], b["y1"], b["y2"],
                                           denom=overlap_denom)
                    if ratio >= row_overlap_min:
                        w = round(ratio, 4)
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "ROW_COMPAT", "weight": w})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "ROW_COMPAT", "weight": w})
                        has_row = True

                # ── COL_COMPAT ────────────────────────────────────────
                if col_mode == "column_id":
                    # Structural: tokens sharing the enricher's column_id.
                    a_col = a.get("column_id", -1)
                    b_col = b.get("column_id", -1)
                    if a_col >= 0 and a_col == b_col:
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "COL_COMPAT", "weight": 1.0})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "COL_COMPAT", "weight": 1.0})
                        has_col = True
                elif col_mode == "cx":
                    # V1-style direct cx comparison (geometric centroid).
                    if abs(a_cx - b_cx) <= col_thresh:
                        w = round(1.0 - abs(a_cx - b_cx) / col_thresh, 4)
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "COL_COMPAT", "weight": w})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "COL_COMPAT", "weight": w})
                        has_col = True
                else:  # "overlap" — horizontal bbox-extent intersection (default)
                    # Same column if the two boxes' HORIZONTAL extents ([x1, x2])
                    # overlap by >= col_overlap_min of the SHORTER box. Swept
                    # x-coverage instead of center distance. Weight = fraction.
                    ratio = _overlap_ratio(a["x1"], a["x2"], b["x1"], b["x2"],
                                           denom=overlap_denom)
                    if ratio >= col_overlap_min:
                        w = round(ratio, 4)
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "COL_COMPAT", "weight": w})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "COL_COMPAT", "weight": w})
                        has_col = True

                # ── DIRECTIONAL_ADJ (only when no row/col edge) ───────
                if not has_row and not has_col:
                    gap = _bbox_gap(a, b)
                    if gap <= adj_gap:
                        w = round(1.0 - gap / adj_gap, 4)
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "DIRECTIONAL_ADJ", "weight": w})
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "DIRECTIONAL_ADJ", "weight": w})

                # ── CONTEXT_SCOPE (context → data nodes below) ────────
                # Connects CONTEXT nodes to ALL data nodes below them
                # (NUTRIENT, QUANTITY, UNIT) — not just NUTRIENT.
                # This enables per-quantity context resolution in the
                # association stage.
                #
                # Vertical relation (context above data, within ctx_max_y) is
                # ALWAYS required. context_scope_mode controls the LATERAL gate:
                #   "overlap" (default) the context governs a node only if their
                #             HORIZONTAL extents ([x1, x2]) overlap by
                #             >= context_overlap_min (overlap is symmetric, so the
                #             same ratio is used in both directions). This uses the
                #             context's horizontal line to scope the column instead
                #             of relying on nearest-cx.
                #   "cy"                no lateral gate — every node below the
                #             context within ctx_max_y is governed (V1 behaviour).
                if a.get("label") == "CONTEXT" and b.get("label") not in ("CONTEXT", "NOISE"):
                    if a_cy <= b_cy + row_thresh and (b_cy - a_cy) <= ctx_max_y:
                        lateral_ok = (
                            _overlap_ratio(a["x1"], a["x2"], b["x1"], b["x2"],
                                           denom=overlap_denom) >= context_overlap_min
                            if ctx_mode == "overlap" else True
                        )
                        if lateral_ok:
                            edges.append({"src": a_id, "dst": b_id,
                                          "type": "CONTEXT_SCOPE", "weight": 1.0})
                if b.get("label") == "CONTEXT" and a.get("label") not in ("CONTEXT", "NOISE"):
                    if b_cy <= a_cy + row_thresh and (a_cy - b_cy) <= ctx_max_y:
                        lateral_ok = (
                            _overlap_ratio(a["x1"], a["x2"], b["x1"], b["x2"],
                                           denom=overlap_denom) >= context_overlap_min
                            if ctx_mode == "overlap" else True
                        )
                        if lateral_ok:
                            edges.append({"src": b_id, "dst": a_id,
                                          "type": "CONTEXT_SCOPE", "weight": 1.0})

                # ── HEADER_SCOPE (structural: same column_id) ─────────
                if a.get("is_header") and not b.get("is_header"):
                    if (a.get("column_id", -1) >= 0 and
                        a["column_id"] == b.get("column_id", -2)):
                        edges.append({"src": a_id, "dst": b_id,
                                      "type": "HEADER_SCOPE", "weight": 1.0})
                if b.get("is_header") and not a.get("is_header"):
                    if (b.get("column_id", -1) >= 0 and
                        b["column_id"] == a.get("column_id", -2)):
                        edges.append({"src": b_id, "dst": a_id,
                                      "type": "HEADER_SCOPE", "weight": 1.0})

        # Post-pass: competitive mutual-nearest row pruning (overlap_mnn only).
        # Runs on the fully-built overlap candidate set so it composes with the
        # col/context/adjacency edges already emitted above.
        if row_mode == "overlap_mnn":
            n_before = sum(1 for e in edges if e.get("type") == "ROW_COMPAT")
            edges = _prune_row_mnn(edges, nodes, mnn_tol)
            n_after = sum(1 for e in edges if e.get("type") == "ROW_COMPAT")
            logger.info(f"  overlap_mnn: pruned ROW_COMPAT {n_before} -> {n_after} "
                        f"(mnn_tol={mnn_tol})")

        graph = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes":     nodes,
            "edges":     edges,
            "_scale":    round(scale, 2),
        }

        _mnn_note = f", mnn_tol={mnn_tol}" if row_mode == "overlap_mnn" else ""
        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges "
                     f"(scale={scale:.2f}x, row_mode={row_mode}, col_mode={col_mode}, "
                     f"ctx_mode={ctx_mode}, overlap_denom={overlap_denom}{_mnn_note}, "
                     f"row_thresh={row_thresh:.0f}px)")
        return graph

    # ── Traversal helpers ─────────────────────────────────────────────

    def get_neighbors(self, graph: dict, node_id: int,
                      edge_types: List[str] = None) -> List[dict]:
        node_map = {n["id"]: n for n in graph["nodes"]}
        return [
            {"node": node_map[e["dst"]], "edge_type": e["type"],
             "weight": e["weight"]}
            for e in graph["edges"]
            if e["src"] == node_id
            and (edge_types is None or e["type"] in edge_types)
            and e["dst"] in node_map
        ]

    def get_row_neighbors(self, graph: dict, node_id: int) -> List[dict]:
        return [x["node"] for x in
                self.get_neighbors(graph, node_id, ["ROW_COMPAT"])]

    def get_col_neighbors(self, graph: dict, node_id: int) -> List[dict]:
        return [x["node"] for x in
                self.get_neighbors(graph, node_id, ["COL_COMPAT"])]

    # ── Debug ─────────────────────────────────────────────────────────

    def print_graph(self, graph: dict) -> None:
        print(f"\n{'='*60}")
        print("GEOMETRY-AWARE SEMANTIC GRAPH SUMMARY (V2)")
        print(f"{'='*60}")
        print(f"Nodes: {graph['num_nodes']}  |  Edges: {graph['num_edges']}")

        edge_counts  = Counter(e["type"] for e in graph["edges"])
        label_counts = Counter(n["label"] for n in graph["nodes"])

        print("\nEdge types:")
        for etype in ["ROW_COMPAT", "COL_COMPAT", "DIRECTIONAL_ADJ",
                       "CONTEXT_SCOPE", "HEADER_SCOPE"]:
            print(f"  {etype:<20} {edge_counts.get(etype, 0):>5}")

        print("\nNode labels:")
        for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "UNKNOWN"]:
            c = label_counts.get(label, 0)
            if c:
                print(f"  {label:<12} {c:>4}")

        # Column role distribution
        role_counts = Counter(n.get("column_role", "?") for n in graph["nodes"])
        print("\nColumn roles (node counts):")
        for role, cnt in role_counts.most_common():
            print(f"  {role:<12} {cnt:>4}")

        # Dosage streams
        streams = set(n.get("dosage_stream_id", -1) for n in graph["nodes"])
        streams.discard(-1)
        print(f"\nDosage streams: {len(streams)}")
        for sid in sorted(streams):
            stream_nodes = [n for n in graph["nodes"]
                           if n.get("dosage_stream_id") == sid]
            ctx = stream_nodes[0].get("column_context_id", "?") if stream_nodes else "?"
            print(f"  Stream {sid}: {len(stream_nodes)} tokens, context={ctx}")

        print(f"{'='*60}\n")

    def save(self, graph: dict, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays to lists for JSON serialization
        import copy
        serializable = copy.deepcopy(graph)
        for node in serializable["nodes"]:
            for key in ["direction", "normal", "center"]:
                val = node.get(key)
                if val is not None and hasattr(val, 'tolist'):
                    node[key] = val.tolist()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Graph saved: {output_path} "
              f"(nodes={graph['num_nodes']}, edges={graph['num_edges']})")