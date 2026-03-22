"""
graph_constructor.py
====================
Stage 4 of the zero-shot nutrient extraction pipeline.

FIX: CONTEXT_SCOPE horizontal overlap check removed.

Root cause of the original bug:
  Context tokens ('Je 100g', 'per 100g', '100g') appear as column
  headers above the VALUE column (right side of table, x ≈ 400-600).
  Nutrient names appear in the NAME column (left side, x ≈ 0-200).
  The old check `nut_x1 <= ctx_cx <= nut_x2` always failed because
  the context header's x-position is never within the nutrient's x-range.

Fix:
  A context header governs ALL nutrients in the table below it.
  Only vertical conditions are checked for CONTEXT_SCOPE:
    1. Context cy <= nutrient cy  (context is above)
    2. |cy_diff| <= context_scope_max_y  (within range)
"""

import json
import math
from pathlib import Path


DEFAULT_CONFIG = {
    "same_row_threshold":       25,
    "same_col_threshold":       20,
    "adjacency_gap_max":        60,
    "context_scope_max_y":      600,   # increased from 400 to cover full table height
    "context_scope_require_above": True,
    "min_confidence":           0.30,
    "excluded_labels":          {"NOISE"},
}


def make_node(token: dict, node_id: int) -> dict:
    cx = (token["x1"] + token["x2"]) / 2
    cy = (token["y1"] + token["y2"]) / 2
    return {
        "id": node_id,
        "token": token["token"],
        "norm": token.get("norm", token["token"].lower().strip()),
        "label": token.get("label", "UNKNOWN"),
        "x1": token["x1"], "y1": token["y1"],
        "x2": token["x2"], "y2": token["y2"],
        "cx": cx, "cy": cy,
        "conf": token.get("conf", 0.0),
    }


def make_edge(src_id: int, dst_id: int, edge_type: str, weight: float = 1.0) -> dict:
    return {
        "src": src_id,
        "dst": dst_id,
        "type": edge_type,
        "weight": round(weight, 4),
    }


class GraphConstructor:
    """
    Builds a typed semantic spatial graph from classified OCR tokens.

    Edge types:
      SAME_ROW      — horizontal alignment
      SAME_COL      — vertical alignment
      ADJACENT      — spatial proximity
      CONTEXT_SCOPE — semantic scoping (context header → nutrients below it)
    """

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG

    def _bbox_gap(self, a: dict, b: dict) -> float:
        h_gap = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
        v_gap = max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
        return math.sqrt(h_gap ** 2 + v_gap ** 2)

    def _same_row(self, a: dict, b: dict) -> bool:
        return abs(a["cy"] - b["cy"]) <= self.config["same_row_threshold"]

    def _same_col(self, a: dict, b: dict) -> bool:
        return abs(a["cx"] - b["cx"]) <= self.config["same_col_threshold"]

    def _is_adjacent(self, a: dict, b: dict) -> bool:
        return self._bbox_gap(a, b) <= self.config["adjacency_gap_max"]

    def _is_context_scope(self, context_node: dict, nutrient_node: dict) -> bool:
        """
        True if a CONTEXT token scopes over a NUTRIENT token.

        FIXED: No horizontal overlap check.
        Context headers appear in the value column (right side), nutrients
        appear in the name column (left side). Horizontal proximity is
        irrelevant — context governs all rows below it in the table.

        Conditions:
          1. context cy <= nutrient cy  (context is above nutrient)
          2. vertical distance <= context_scope_max_y
        """
        row_tol = self.config["same_row_threshold"]

        # Context must be above or at same level as nutrient
        if self.config["context_scope_require_above"]:
            if context_node["cy"] > nutrient_node["cy"] + row_tol:
                return False

        # Vertical distance check
        v_dist = nutrient_node["cy"] - context_node["cy"]
        if v_dist > self.config["context_scope_max_y"]:
            return False

        return True

    def build(self, labeled_tokens: list) -> dict:
        excluded = self.config["excluded_labels"]
        min_conf = self.config["min_confidence"]

        nodes = []
        for idx, token in enumerate(labeled_tokens):
            if token.get("label") in excluded:
                continue
            if token.get("conf", 0.0) < min_conf:
                continue
            nodes.append(make_node(token, node_id=idx))

        edges = []
        n = len(nodes)

        for i in range(n):
            for j in range(i + 1, n):
                a = nodes[i]
                b = nodes[j]
                added_types = set()

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

                # ADJACENT (only if not already same row/col)
                if "SAME_ROW" not in added_types and "SAME_COL" not in added_types:
                    if self._is_adjacent(a, b):
                        gap = self._bbox_gap(a, b)
                        weight = 1.0 - (gap / self.config["adjacency_gap_max"])
                        edges.append(make_edge(a["id"], b["id"], "ADJACENT", weight))
                        edges.append(make_edge(b["id"], a["id"], "ADJACENT", weight))

                # CONTEXT_SCOPE — context → {NUTRIENT, QUANTITY, UNIT}
                targets = {"NUTRIENT", "QUANTITY", "UNIT"}
                if a["label"] == "CONTEXT" and b["label"] in targets:
                    if self._is_context_scope(a, b):
                        edges.append(make_edge(a["id"], b["id"], "CONTEXT_SCOPE"))

                if b["label"] == "CONTEXT" and a["label"] in targets:
                    if self._is_context_scope(b, a):
                        edges.append(make_edge(b["id"], a["id"], "CONTEXT_SCOPE"))

        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def get_neighbors(self, graph: dict, node_id: int,
                      edge_types: list = None) -> list:
        node_map = {n["id"]: n for n in graph["nodes"]}
        return [
            {"node": node_map[e["dst"]], "edge_type": e["type"], "weight": e["weight"]}
            for e in graph["edges"]
            if e["src"] == node_id
            and (edge_types is None or e["type"] in edge_types)
            and e["dst"] in node_map
        ]

    def get_row_group(self, graph: dict, node_id: int) -> list:
        return self.get_neighbors(graph, node_id, edge_types=["SAME_ROW"])

    def print_graph(self, graph: dict) -> None:
        print(f"\n{'='*60}")
        print("SEMANTIC GRAPH SUMMARY")
        print(f"{'='*60}")
        print(f"Nodes: {graph['num_nodes']}  |  Edges: {graph['num_edges']}")

        from collections import Counter
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

        print("\nCONTEXT_SCOPE edges:")
        node_map  = {n["id"]: n for n in graph["nodes"]}
        ctx_edges = [e for e in graph["edges"] if e["type"] == "CONTEXT_SCOPE"]
        if ctx_edges:
            for e in ctx_edges[:30]:
                s = node_map.get(e["src"], {})
                d = node_map.get(e["dst"], {})
                print(f"  {s.get('token','?')[:25]:<27} → "
                      f"{d.get('token','?')[:25]:<27} ({d.get('label','?')})")
        else:
            print("  (none — check CONTEXT token classification)")
        print(f"{'='*60}\n")

    def save(self, graph: dict, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"Graph saved: {output_path} "
              f"(nodes={graph['num_nodes']}, edges={graph['num_edges']})")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.ocr.ocr_runner import run_ocr_on_image
    from src.utils.ocr_corrector import OCRCorrector
    from src.classification.semantic_classifier import SemanticClassifier

    IMAGE = "data/raw/102.jpeg"
    print(f"Testing CONTEXT_SCOPE fix on {IMAGE}\n")

    tokens    = run_ocr_on_image(IMAGE)
    corrected = OCRCorrector().correct_all(tokens)
    labeled   = SemanticClassifier(0.30, split_fused_tokens=True).classify_all(corrected)

    constructor = GraphConstructor()
    graph = constructor.build(labeled)
    constructor.print_graph(graph)

    ctx_edges = [e for e in graph["edges"] if e["type"] == "CONTEXT_SCOPE"]
    nutrient_ids = {n["id"] for n in graph["nodes"] if n["label"] == "NUTRIENT"}
    nutrients_with_ctx = {e["dst"] for e in ctx_edges if e["dst"] in nutrient_ids}
    print(f"CONTEXT_SCOPE edges : {len(ctx_edges)}")
    print(f"Nutrients with ctx  : {len(nutrients_with_ctx)} / {len(nutrient_ids)}")