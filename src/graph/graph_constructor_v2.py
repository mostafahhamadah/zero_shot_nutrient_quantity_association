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

REDESIGN vs ORIGINAL
---------------------
  Original:  SAME_ROW = |cy_a - cy_b| < 25px           (flat assumption)
  New:       ROW_COMPAT = direction-aware perpendicular  (handles skew/curve)

  Original:  SAME_COL = |cx_a - cx_b| < 20px            (flat assumption)
  New:       COL_COMPAT = direction-aware parallel        (handles skew/curve)

  Original:  CONTEXT_SCOPE = vertical distance only
  New:       HEADER_SCOPE = column_context_id propagation (structural, not spatial)

  New:       RANK_COMPAT = data_rank_in_column match     (structural fallback)

NODE SCHEMA
-----------
Nodes carry ALL enriched fields.  Graph consumers access structural
fields (row_id, column_id, column_role, dosage_stream_id, rank, etc.)
directly from nodes — no recomputation needed.

EDGE TYPES
----------
  ROW_COMPAT       Tokens on the same logical row (row_id match).
                   Weight = row_confidence score from enricher.
                   Bidirectional.

  COL_COMPAT       Tokens in the same logical column (column_id match).
                   Weight = column_confidence score from enricher.
                   Bidirectional.

  DIRECTIONAL_ADJ  Tokens physically close AND direction-compatible,
                   but NOT in same row or column.
                   Weight = inverse of bbox gap distance.
                   Bidirectional.

  HEADER_SCOPE     CONTEXT header → tokens in the same column.
                   Structural (not spatial) — uses column_context_id.
                   Directed: header → governed token.
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
    # Row/column edge thresholds (V1-style pairwise comparison)
    "same_row_threshold":      25,
    "same_col_threshold":      20,

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


class GraphConstructorV2:
    """
    Build a typed semantic graph from enriched tokens.

    Nodes = enriched tokens (all fields preserved).
    Edges = ROW_COMPAT, COL_COMPAT, DIRECTIONAL_ADJ, HEADER_SCOPE.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def build(self, enriched_tokens: List[dict]) -> dict:
        """
        Build graph from Stage 3.5 enriched tokens.

        EDGE BUILDING STRATEGY:
          ROW_COMPAT and COL_COMPAT use V1-style DIRECT PAIRWISE comparison
          (|cy_a - cy_b| < threshold) — NOT enricher row_id.

          Why: The enricher's gap-based row grouping can fail on dense labels
          (merging multiple rows into one group), but pairwise cy comparison
          CANNOT chain — two tokens 40px apart in cy will never get a
          ROW_COMPAT edge regardless of intermediate tokens.

          The enricher's row_id, column_id, column_role, dosage_stream_id
          remain on the nodes for the association stage to use for rank-based
          matching and stream detection.

          CONTEXT_SCOPE uses V1-style vertical distance (context above nutrient).
          HEADER_SCOPE uses structural column_id (context header → same column).
          Both are emitted — association can use whichever is available.

        All thresholds are auto-scaled by image resolution.
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

        edges = []

        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                a_id, b_id = a["id"], b["id"]
                has_row = False
                has_col = False

                # ── ROW_COMPAT (V1-style: direct cy comparison) ───────
                a_cy = a.get("cy", (a["y1"] + a["y2"]) / 2.0)
                b_cy = b.get("cy", (b["y1"] + b["y2"]) / 2.0)
                if abs(a_cy - b_cy) <= row_thresh:
                    w = round(1.0 - abs(a_cy - b_cy) / row_thresh, 4)
                    edges.append({"src": a_id, "dst": b_id,
                                  "type": "ROW_COMPAT", "weight": w})
                    edges.append({"src": b_id, "dst": a_id,
                                  "type": "ROW_COMPAT", "weight": w})
                    has_row = True

                # ── COL_COMPAT (V1-style: direct cx comparison) ───────
                a_cx = a.get("cx", (a["x1"] + a["x2"]) / 2.0)
                b_cx = b.get("cx", (b["x1"] + b["x2"]) / 2.0)
                if abs(a_cx - b_cx) <= col_thresh:
                    w = round(1.0 - abs(a_cx - b_cx) / col_thresh, 4)
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

                # ── CONTEXT_SCOPE (vertical: context → data nodes below) ─
                # Connects CONTEXT nodes to ALL data nodes below them
                # (NUTRIENT, QUANTITY, UNIT) — not just NUTRIENT.
                # This enables per-quantity context resolution in the
                # association stage.
                if a.get("label") == "CONTEXT" and b.get("label") not in ("CONTEXT", "NOISE"):
                    if a_cy <= b_cy + row_thresh:
                        if (b_cy - a_cy) <= ctx_max_y:
                            edges.append({"src": a_id, "dst": b_id,
                                          "type": "CONTEXT_SCOPE", "weight": 1.0})
                if b.get("label") == "CONTEXT" and a.get("label") not in ("CONTEXT", "NOISE"):
                    if b_cy <= a_cy + row_thresh:
                        if (a_cy - b_cy) <= ctx_max_y:
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

        graph = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes":     nodes,
            "edges":     edges,
            "_scale":    round(scale, 2),
        }

        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges "
                     f"(scale={scale:.2f}x, row_thresh={row_thresh:.0f}px)")
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
                       "HEADER_SCOPE"]:
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