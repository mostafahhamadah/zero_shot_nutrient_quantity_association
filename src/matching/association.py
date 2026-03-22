"""
association.py
==============
Stage 5 of the zero-shot nutrient extraction pipeline.

Traverses the semantic graph to extract structured nutritional tuples:
    <nutrient, quantity, unit, context, nrv_percent, serving_size>

v3 fix — Unit search now performed on QUANTITY node's neighbors first,
not on the nutrient node's neighbors. This correctly handles split tokens
where "400mg" becomes "400" (QUANTITY) + "mg" (UNIT) sitting adjacent
to each other, not adjacent to the nutrient name.

Search order for unit per quantity:
  1. Inline unit already in quantity token text ("0.43g" → "g")
  2. SAME_ROW neighbors of the QUANTITY node
  3. ADJACENT neighbors of the QUANTITY node
  4. SAME_ROW neighbors of the NUTRIENT node (broad fallback)
"""

import re
import csv
import json
from pathlib import Path


# ── Quantity parsing ──────────────────────────────────────────────────────────

def parse_quantity(token_text: str) -> tuple:
    text = token_text.strip("*'\"()[]. ")
    m = re.match(r'^(\d+[.,]?\d*)\s*(mg|g|kg|µg|mcg|ug|kj|kcal|ml|iu|ie)$',
                 text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "."), m.group(2).lower()
    m = re.match(r'^(\d+[.,]?\d*)[\(\[]\d+[\)\]%]?$', text)
    if m:
        return m.group(1).replace(",", "."), None
    m = re.match(r'^(\d+[.,]?\d*)$', text)
    if m:
        return m.group(1).replace(",", "."), None
    return text, None


def parse_nrv(token_text: str) -> str:
    text = token_text.strip("*'\"() ")
    m = re.search(r'(\d+)\s*%', text)
    if m:
        return m.group(1)
    m = re.search(r'[\(\[](\d+)[\)\]%]', token_text)
    if m:
        return m.group(1)
    return None


def normalize_nutrient(token_text: str) -> str:
    text = token_text.strip("'\"*.,[]()/ ")
    return re.sub(r'\s+', ' ', text)


# ── Associator ────────────────────────────────────────────────────────────────

class TupleAssociator:

    def __init__(self, config: dict = None):
        self.config = config or {
            "max_quantities_per_nutrient": 3,
            "prefer_high_confidence": True,
            "min_quantity_conf": 0.30,
        }

    # ── Graph helpers ─────────────────────────────────────────────────────────

    def _build_node_map(self, graph: dict) -> dict:
        return {n["id"]: n for n in graph["nodes"]}

    def _get_neighbors(self, graph: dict, node_id: int,
                       node_map: dict, edge_types: list) -> list:
        """Return neighbors via any of the given edge types (both directions)."""
        neighbors = []
        for e in graph["edges"]:
            if e["type"] not in edge_types:
                continue
            if e["src"] == node_id and e["dst"] in node_map:
                neighbors.append(node_map[e["dst"]])
            elif e["dst"] == node_id and e["src"] in node_map:
                neighbors.append(node_map[e["src"]])
        return neighbors

    def _get_same_row_neighbors(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["SAME_ROW"])

    def _get_adjacent_neighbors(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["ADJACENT"])

    def _get_same_col_neighbors(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["SAME_COL"])

    def _get_context_ancestors(self, graph, node_id, node_map):
        return [
            node_map[e["src"]]
            for e in graph["edges"]
            if e["dst"] == node_id and e["type"] == "CONTEXT_SCOPE"
            and e["src"] in node_map
            and node_map[e["src"]]["label"] == "CONTEXT"
        ]

    def _get_serving_nodes(self, graph):
        return [n for n in graph["nodes"] if n["label"] == "SERVING"]

    # ── Unit search (v3 fix) ──────────────────────────────────────────────────

    def _find_unit_for_quantity(self, qty_node: dict,
                                 nutrient_row_nodes: list,
                                 graph: dict,
                                 node_map: dict) -> str:
        """
        Find UNIT for a given QUANTITY node.

        v3 fix: searches qty node's own neighbors FIRST before
        falling back to nutrient row. Correctly links split tokens.
        """
        # 1. Inline unit in token text
        _, inline_unit = parse_quantity(qty_node["token"])
        if inline_unit:
            return inline_unit

        # 2. SAME_ROW neighbors of the QUANTITY node
        for node in self._get_same_row_neighbors(
                graph, qty_node["id"], node_map):
            if node["label"] == "UNIT":
                return node["norm"].strip(".,*()[]| ")

        # 3. ADJACENT neighbors of the QUANTITY node
        for node in self._get_adjacent_neighbors(
                graph, qty_node["id"], node_map):
            if node["label"] == "UNIT":
                return node["norm"].strip(".,*()[]| ")

        # 4. Fallback: SAME_ROW of the NUTRIENT node
        for node in nutrient_row_nodes:
            if node["label"] == "UNIT":
                return node["norm"].strip(".,*()[]| ")

        return None

    # ── NRV search ────────────────────────────────────────────────────────────

    def _find_nrv_for_quantity(self, qty_node: dict,
                                nutrient_row_nodes: list,
                                graph: dict,
                                node_map: dict) -> str:
        embedded = parse_nrv(qty_node["token"])
        if embedded:
            return embedded

        for node in self._get_same_row_neighbors(
                graph, qty_node["id"], node_map):
            if node["label"] == "NRV":
                nrv = parse_nrv(node["token"])
                if nrv:
                    return nrv

        for node in nutrient_row_nodes:
            if node["label"] == "NRV":
                nrv = parse_nrv(node["token"])
                if nrv:
                    return nrv

        return None

    # ── Column fallback ───────────────────────────────────────────────────────

    def _find_quantity_by_column(self, graph, nutrient_node, node_map):
        col_neighbors   = self._get_same_col_neighbors(
            graph, nutrient_node["id"], node_map)
        nutrient_height = max(nutrient_node["y2"] - nutrient_node["y1"], 20)
        max_v_distance  = nutrient_height * 2.5

        qty_nodes = [
            n for n in col_neighbors
            if n["label"] == "QUANTITY"
            and n["conf"] >= self.config["min_quantity_conf"]
            and abs(n["cy"] - nutrient_node["cy"]) <= max_v_distance
        ]
        qty_nodes.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
        return qty_nodes[:self.config["max_quantities_per_nutrient"]]

    # ── Main extraction ───────────────────────────────────────────────────────

    def extract(self, graph: dict, image_id: str = "unknown") -> list:
        node_map      = self._build_node_map(graph)
        serving_nodes = self._get_serving_nodes(graph)
        serving_size  = serving_nodes[0]["token"] if serving_nodes else None

        tuples     = []
        seen_pairs = set()

        for node in graph["nodes"]:
            if node["label"] != "NUTRIENT":
                continue

            nutrient_text = normalize_nutrient(node["token"])
            row_neighbors = self._get_same_row_neighbors(
                graph, node["id"], node_map)

            qty_nodes = [
                n for n in row_neighbors
                if n["label"] == "QUANTITY"
                and n["conf"] >= self.config["min_quantity_conf"]
            ]

            ctx_ancestors = self._get_context_ancestors(
                graph, node["id"], node_map)
            context_str = None
            if ctx_ancestors:
                ctx_ancestors.sort(
                    key=lambda n: (n["x2"]-n["x1"]) * (n["y2"]-n["y1"]))
                context_str = normalize_nutrient(ctx_ancestors[0]["token"])

            if self.config["prefer_high_confidence"]:
                qty_nodes.sort(key=lambda n: n["conf"], reverse=True)
            qty_nodes = qty_nodes[:self.config["max_quantities_per_nutrient"]]

            if not qty_nodes:
                qty_nodes = self._find_quantity_by_column(
                    graph, node, node_map)

            if not qty_nodes:
                pair_key = (nutrient_text, None)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    tuples.append({
                        "image_id":     image_id,
                        "nutrient":     nutrient_text,
                        "quantity":     None,
                        "unit":         None,
                        "context":      context_str,
                        "nrv_percent":  None,
                        "serving_size": serving_size,
                    })
            else:
                for qty_node in qty_nodes:
                    qty_value, inline_unit = parse_quantity(qty_node["token"])
                    unit = inline_unit or self._find_unit_for_quantity(
                        qty_node, row_neighbors, graph, node_map)
                    nrv  = self._find_nrv_for_quantity(
                        qty_node, row_neighbors, graph, node_map)

                    pair_key = (nutrient_text, qty_value)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    tuples.append({
                        "image_id":     image_id,
                        "nutrient":     nutrient_text,
                        "quantity":     qty_value,
                        "unit":         unit,
                        "context":      context_str,
                        "nrv_percent":  nrv,
                        "serving_size": serving_size,
                    })

        return tuples

    # ── Output helpers ────────────────────────────────────────────────────────

    def print_tuples(self, tuples: list) -> None:
        print(f"\n{'='*70}")
        print(f"EXTRACTED TUPLES  ({len(tuples)} total)")
        print(f"{'='*70}")
        print(f"{'NUTRIENT':<35} {'QTY':<10} {'UNIT':<6} "
              f"{'CONTEXT':<20} {'NRV':<6}")
        print("-" * 80)
        for t in tuples:
            print(
                f"{str(t['nutrient'])[:34]:<35} "
                f"{str(t['quantity'] or '')[:9]:<10} "
                f"{str(t['unit'] or '')[:5]:<6} "
                f"{str(t['context'] or '')[:19]:<20} "
                f"{str(t['nrv_percent'] or '')[:5]:<6}"
            )
        print(f"{'='*70}\n")

    def save_csv(self, tuples: list, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_id", "nutrient", "quantity", "unit",
                      "context", "nrv_percent", "serving_size"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tuples)
        print(f"Tuples saved: {output_path}  ({len(tuples)} rows)")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.ocr.ocr_runner import run_ocr_on_image
    from src.utils.ocr_corrector import OCRCorrector
    from src.classification.semantic_classifier import SemanticClassifier
    from src.graph.graph_constructor import GraphConstructor

    IMAGE    = "data/raw/1.jpeg"
    IMAGE_ID = "1.jpeg"

    tokens    = run_ocr_on_image(IMAGE)
    corrected = OCRCorrector().correct_all(tokens)
    labeled   = SemanticClassifier(confidence_threshold=0.30,
                                   split_fused_tokens=True).classify_all(corrected)
    graph     = GraphConstructor().build(labeled)

    associator = TupleAssociator()
    tuples     = associator.extract(graph, image_id=IMAGE_ID)
    associator.print_tuples(tuples)

    units_in_tuples = sum(1 for t in tuples if t["unit"])
    print(f"Tuples with units: {units_in_tuples} / {len(tuples)}")