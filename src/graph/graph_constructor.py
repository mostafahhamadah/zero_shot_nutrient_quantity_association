"""
graph_constructor.py
====================
Stage 4 — Semantic Graph Construction
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Build a typed spatial-semantic graph over the labeled OCR tokens
produced by Stage 3 (classifier).  The graph encodes layout
relationships between tokens so Stage 5 (association) can traverse
edges to reconstruct nutritional tuples without any training data.

INPUT / OUTPUT SCHEMA
---------------------
Input  : List[Dict]  — labeled tokens (token, x1, y1, x2, y2, cx, cy, conf,
                        label, norm)
Output : Dict        — {nodes, edges, num_nodes, num_edges}

EDGE TYPES
----------
  SAME_ROW      Tokens on the same visual row.
                Condition: |cy_a - cy_b| <= same_row_threshold (25 px)
                Bidirectional.

  SAME_COL      Tokens in the same vertical column.
                Condition: |cx_a - cx_b| <= same_col_threshold (20 px)
                Bidirectional.

  ADJACENT      Tokens in close spatial proximity that are NOT already
                connected by SAME_ROW or SAME_COL.
                Condition: bbox gap <= adjacency_gap_max (60 px)
                Bidirectional.  Weight = 1.0 - (gap / max_gap).

                DESIGN NOTE — mutual exclusivity:
                ADJACENT is only built when neither SAME_ROW nor SAME_COL
                was built for the same pair.  This means ADJACENT only ever
                fires for tokens that are close but on different rows AND
                different columns (diagonal proximity).  Stage 5 uses
                ADJACENT as a unit-fallback search, which works because units
                that missed SAME_ROW (due to y-jitter) are still physically
                close.  This is intentional, not a bug.

  CONTEXT_SCOPE Context header governs NUTRIENT nodes below it.
                Conditions:
                  1. context cy <= nutrient cy  (header is above)
                  2. vertical distance <= context_scope_max_y (600 px)
                Directed: context → nutrient only.

                DESIGN NOTE — NUTRIENT targets only:
                CONTEXT_SCOPE is restricted to NUTRIENT nodes.  Stage 5
                (association) finds quantities and units for a matched
                nutrient via SAME_ROW traversal — no direct context →
                quantity/unit edges are needed, and adding them would create
                O(context × (qty+unit)) redundant edges.

CONFIDENCE FILTERING
--------------------
NOISE-labeled tokens are excluded from the graph.  No additional
confidence threshold is applied here — that is Stage 3's responsibility.
"""

import json
import math
from pathlib import Path


DEFAULT_CONFIG = {
    "same_row_threshold":          25,
    "same_col_threshold":          20,
    "adjacency_gap_max":           60,
    "context_scope_max_y":        600,   # covers full table height
    "context_scope_require_above": True,
    "excluded_labels":            {"NOISE"},
}


def make_node(token: dict, node_id: int) -> dict:
    """Create a graph node from a labeled token dict."""
    return {
        "id":    node_id,
        "token": token["token"],
        "norm":  token.get("norm", token["token"].lower().strip()),
        "label": token.get("label", "UNKNOWN"),
        "x1":    token["x1"],
        "y1":    token["y1"],
        "x2":    token["x2"],
        "y2":    token["y2"],
        "cx":    (token["x1"] + token["x2"]) / 2.0,
        "cy":    (token["y1"] + token["y2"]) / 2.0,
        "conf":  token.get("conf", 0.0),
    }


def make_edge(src_id: int, dst_id: int, edge_type: str,
              weight: float = 1.0) -> dict:
    """Create a directed graph edge."""
    return {
        "src":    src_id,
        "dst":    dst_id,
        "type":   edge_type,
        "weight": round(weight, 4),
    }


class GraphConstructor:
    """
    Builds a typed semantic spatial graph from classified OCR tokens.
    NOISE tokens are excluded.  All other labels become nodes.
    """

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG

    # ── Spatial predicates ────────────────────────────────────────────

    def _bbox_gap(self, a: dict, b: dict) -> float:
        """Euclidean gap between two bounding boxes (0 if overlapping)."""
        h_gap = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
        v_gap = max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
        return math.sqrt(h_gap ** 2 + v_gap ** 2)

    def _same_row(self, a: dict, b: dict) -> bool:
        return abs(a["cy"] - b["cy"]) <= self.config["same_row_threshold"]

    def _same_col(self, a: dict, b: dict) -> bool:
        return abs(a["cx"] - b["cx"]) <= self.config["same_col_threshold"]

    def _is_adjacent(self, a: dict, b: dict) -> bool:
        return self._bbox_gap(a, b) <= self.config["adjacency_gap_max"]

    def _is_context_scope(self, context_node: dict,
                          nutrient_node: dict) -> bool:
        """
        True if a CONTEXT node scopes over a NUTRIENT node.

        Only vertical conditions are checked.  Context headers appear in
        the value column (right side of table) while nutrient names appear
        in the name column (left side) — horizontal proximity is irrelevant.
        A context header governs all nutrient rows below it in the table.

        Conditions:
          1. context cy <= nutrient cy  (header is above or at same level)
          2. vertical distance <= context_scope_max_y
        """
        row_tol = self.config["same_row_threshold"]

        if self.config["context_scope_require_above"]:
            if context_node["cy"] > nutrient_node["cy"] + row_tol:
                return False

        v_dist = nutrient_node["cy"] - context_node["cy"]
        return v_dist <= self.config["context_scope_max_y"]

    # ── Graph construction ────────────────────────────────────────────

    def build(self, labeled_tokens: list) -> dict:
        """
        Build the semantic graph from a list of labeled token dicts.

        NOISE tokens are excluded.  All other tokens become nodes.
        No confidence filtering is applied — that is Stage 3's concern.

        Thresholds are auto-scaled by image resolution.  The default
        values (same_row=25, same_col=20, adj=60, ctx=600) are tuned
        for ~500px image height.  For higher-resolution images the
        pixel distances between tokens scale proportionally, so
        thresholds must scale too.
        """
        excluded = self.config["excluded_labels"]

        # ── Auto-scale thresholds by image resolution ─────────────────
        # Estimate image size from token bounding boxes (no need to
        # pass image dimensions through the pipeline).
        max_y = max((t.get("y2", 0) for t in labeled_tokens), default=500)
        scale = max(max_y / 500.0, 1.0)   # never shrink below defaults

        scaled_config = {**self.config}
        scaled_config["same_row_threshold"]  = self.config["same_row_threshold"] * scale
        scaled_config["same_col_threshold"]  = self.config["same_col_threshold"] * scale
        scaled_config["adjacency_gap_max"]   = self.config["adjacency_gap_max"]  * scale
        scaled_config["context_scope_max_y"] = self.config["context_scope_max_y"] * scale

        # Use scaled_config for all spatial predicates in this call
        orig_config  = self.config
        self.config  = scaled_config

        nodes = [
            make_node(token, idx)
            for idx, token in enumerate(labeled_tokens)
            if token.get("label") not in excluded
        ]

        edges = []
        n     = len(nodes)

        for i in range(n):
            for j in range(i + 1, n):
                a = nodes[i]
                b = nodes[j]
                added_types: set = set()

                # SAME_ROW
                if self._same_row(a, b):
                    edges.append(make_edge(a["id"], b["id"], "SAME_ROW"))
                    edges.append(make_edge(b["id"], a["id"], "SAME_ROW"))
                    added_types.add("SAME_ROW")

                # SAME_COL
                if self._same_col(a, b):
                    edges.append(make_edge(a["id"], b["id"], "SAME_COL"))
                    edges.append(make_edge(b["id"], a["id"], "SAME_COL"))
                    added_types.add("SAME_COL")

                # ADJACENT — only when not already same-row or same-col
                # (see module docstring — this mutual exclusivity is deliberate)
                if not added_types and self._is_adjacent(a, b):
                    gap    = self._bbox_gap(a, b)
                    weight = 1.0 - (gap / self.config["adjacency_gap_max"])
                    edges.append(make_edge(a["id"], b["id"], "ADJACENT", weight))
                    edges.append(make_edge(b["id"], a["id"], "ADJACENT", weight))

                # CONTEXT_SCOPE — context header → NUTRIENT nodes below it
                # (see module docstring — NUTRIENT targets only, directed)
                if a["label"] == "CONTEXT" and b["label"] == "NUTRIENT":
                    if self._is_context_scope(a, b):
                        edges.append(make_edge(a["id"], b["id"], "CONTEXT_SCOPE"))

                if b["label"] == "CONTEXT" and a["label"] == "NUTRIENT":
                    if self._is_context_scope(b, a):
                        edges.append(make_edge(b["id"], a["id"], "CONTEXT_SCOPE"))

        # Restore original config
        self.config = orig_config

        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes":     nodes,
            "edges":     edges,
            "_scale":    round(scale, 2),
        }

    # ── Graph traversal helpers ───────────────────────────────────────

    def get_neighbors(self, graph: dict, node_id: int,
                      edge_types: list = None) -> list:
        """Return all outgoing neighbors of *node_id* by edge type."""
        node_map = {n["id"]: n for n in graph["nodes"]}
        return [
            {
                "node":      node_map[e["dst"]],
                "edge_type": e["type"],
                "weight":    e["weight"],
            }
            for e in graph["edges"]
            if e["src"] == node_id
            and (edge_types is None or e["type"] in edge_types)
            and e["dst"] in node_map
        ]

    def get_row_group(self, graph: dict, node_id: int) -> list:
        """Return all SAME_ROW neighbors of *node_id*."""
        return self.get_neighbors(graph, node_id, edge_types=["SAME_ROW"])

    # ── Debug utilities ───────────────────────────────────────────────

    def print_graph(self, graph: dict) -> None:
        """Print a summary of nodes, edge type counts and CONTEXT_SCOPE edges."""
        from collections import Counter
        print(f"\n{'='*60}")
        print("SEMANTIC GRAPH SUMMARY")
        print(f"{'='*60}")
        print(f"Nodes: {graph['num_nodes']}  |  Edges: {graph['num_edges']}")

        edge_counts  = Counter(e["type"]  for e in graph["edges"])
        label_counts = Counter(n["label"] for n in graph["nodes"])

        print("\nEdge types:")
        for etype in ["SAME_ROW", "SAME_COL", "ADJACENT", "CONTEXT_SCOPE"]:
            print(f"  {etype:<16} {edge_counts.get(etype, 0):>4}")

        print("\nNode labels:")
        for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "UNKNOWN"]:
            c = label_counts.get(label, 0)
            if c:
                print(f"  {label:<12} {c:>4}")

        node_map  = {n["id"]: n for n in graph["nodes"]}
        ctx_edges = [e for e in graph["edges"] if e["type"] == "CONTEXT_SCOPE"]
        print(f"\nCONTEXT_SCOPE edges ({len(ctx_edges)}):")
        if ctx_edges:
            for e in ctx_edges[:30]:
                s = node_map.get(e["src"], {})
                d = node_map.get(e["dst"], {})
                print(f"  {s.get('token','?')[:25]:<27} → "
                      f"{d.get('token','?')[:25]:<27} ({d.get('label','?')})")
        else:
            print("  (none — check CONTEXT token classification in Stage 3)")
        print(f"{'='*60}\n")

    def save(self, graph: dict, output_path: str) -> None:
        """Save the graph to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"Graph saved: {output_path} "
              f"(nodes={graph['num_nodes']}, edges={graph['num_edges']})")