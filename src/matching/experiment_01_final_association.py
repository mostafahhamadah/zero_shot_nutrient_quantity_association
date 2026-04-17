"""
association.py
==============
Stage 5 — Graph Traversal Tuple Associator
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Traverse the semantic graph produced by Stage 4 to extract structured
nutritional tuples.  For each NUTRIENT node the associator finds its
QUANTITY, UNIT and CONTEXT by following typed edges.

INPUT / OUTPUT SCHEMA
---------------------
Input  : Dict        — graph {nodes, edges, num_nodes, num_edges}
Output : List[Dict]  — one dict per tuple:
             image_id, nutrient, quantity, unit, context

EXTRACTION STRATEGY
-------------------
For each NUTRIENT node:

  1. QUANTITY search — SAME_ROW neighbors of the nutrient node.
     If multiple quantities found (multi-column table), sort left→right
     by x-coordinate: leftmost = per_100g column, rightmost = per_serving.

  2. QUANTITY fallback — if SAME_ROW yields nothing, search SAME_COL
     within a tight (2.5× height) then wide (8× height) vertical window.

  3. CONTEXT per quantity — each QUANTITY node has its own CONTEXT_SCOPE
     ancestors (the column header above it).  Reading context from the
     quantity node correctly distinguishes per_100g from per_serving in
     dual-column tables.  Falls back to the nutrient-level context if the
     quantity has no direct context ancestor.

  4. UNIT search — four-level fallback:
       a. Inline unit already embedded in the quantity token text
       b. SAME_ROW neighbors of the quantity node
       c. ADJACENT neighbors of the quantity node
       d. SAME_ROW neighbors of the nutrient node
"""

import re
import csv
from pathlib import Path


# ── Quantity parsing ──────────────────────────────────────────────────────────

def parse_quantity(token_text: str) -> tuple:
    """
    Parse a QUANTITY token into (value_string, inline_unit_or_None).

    '400'       → ('400',  None)
    '0.43g'     → ('0.43', 'g')
    '1195(28]'  → ('1195', None)
    """
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


def normalize_nutrient(token_text: str) -> str:
    text = token_text.strip("'\"*.,[]()/ ")
    return re.sub(r'\s+', ' ', text)


# ── Associator ────────────────────────────────────────────────────────────────

class TupleAssociator:
    """
    Extracts 4-field nutritional tuples from the semantic graph.
    Output fields: image_id, nutrient, quantity, unit, context.
    """

    def __init__(self, config: dict = None):
        self.config = config or {
            "max_quantities_per_nutrient": 3,
        }

    # ── Graph helpers ─────────────────────────────────────────────────────────

    def _build_node_map(self, graph: dict) -> dict:
        return {n["id"]: n for n in graph["nodes"]}

    def _get_neighbors(self, graph: dict, node_id: int,
                       node_map: dict, edge_types: list) -> list:
        """
        Return all neighbors reachable from node_id via outgoing edges
        of the given types.

        Stage 4 adds bidirectional edges for SAME_ROW, SAME_COL and
        ADJACENT, so following outgoing edges only is correct and sufficient.
        The original bidirectional traversal (also following incoming edges)
        caused duplicate results for every SAME_ROW/SAME_COL pair because
        both the A→B and B→A edges were traversed independently.
        """
        return [
            node_map[e["dst"]]
            for e in graph["edges"]
            if e["type"] in edge_types
            and e["src"] == node_id
            and e["dst"] in node_map
        ]

    def _get_same_row(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["SAME_ROW"])

    def _get_adjacent(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["ADJACENT"])

    def _get_same_col(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["SAME_COL"])

    def _get_context_ancestors(self, graph, node_id, node_map):
        """
        Return CONTEXT nodes that scope over node_id via incoming
        CONTEXT_SCOPE edges.  Uses directed traversal (incoming only)
        because CONTEXT_SCOPE is a directed edge: context → nutrient.
        """
        return [
            node_map[e["src"]]
            for e in graph["edges"]
            if e["dst"] == node_id
            and e["type"] == "CONTEXT_SCOPE"
            and e["src"] in node_map
            and node_map[e["src"]]["label"] == "CONTEXT"
        ]

    def _get_context_for_qty(self, qty_node: dict, graph: dict,
                              node_map: dict) -> str:
        """
        Get the canonical context for a specific QUANTITY node.

        In multi-column tables each column has its own context header
        (e.g. 'per 100g' on the left, 'per serving' on the right).
        Reading context from the quantity's own CONTEXT_SCOPE ancestors
        correctly assigns per_100g to the left value and per_serving to
        the right value.  Falls back to None if no ancestor found.
        """
        ctx_ancestors = self._get_context_ancestors(graph, qty_node["id"],
                                                     node_map)
        if not ctx_ancestors:
            return None
        # Most specific context = smallest bounding box area
        ctx_ancestors.sort(
            key=lambda n: (n["x2"] - n["x1"]) * (n["y2"] - n["y1"]))
        return ctx_ancestors[0].get("norm") or ctx_ancestors[0]["token"]

    # ── Unit search ───────────────────────────────────────────────────────────

    def _find_unit(self, qty_node: dict, nutrient_row: list,
                   graph: dict, node_map: dict) -> str:
        """
        Find the UNIT for a quantity node.

        Search order (first match wins):
          1. Inline unit already embedded in the quantity token text
          2. SAME_ROW neighbors of the quantity node
          3. ADJACENT neighbors of the quantity node
          4. SAME_ROW neighbors of the nutrient node (fallback)
        """
        _, inline = parse_quantity(qty_node["token"])
        if inline:
            return inline

        for n in self._get_same_row(graph, qty_node["id"], node_map):
            if n["label"] == "UNIT":
                return (n["norm"] or "").strip(".,*()[]| ")

        for n in self._get_adjacent(graph, qty_node["id"], node_map):
            if n["label"] == "UNIT":
                return (n["norm"] or "").strip(".,*()[]| ")

        for n in nutrient_row:
            if n["label"] == "UNIT":
                return (n["norm"] or "").strip(".,*()[]| ")

        return None

    # ── Column fallback ───────────────────────────────────────────────────────

    def _column_fallback(self, graph, nutrient_node, node_map):
        """
        Find QUANTITY nodes in the same column as the nutrient.

        Two-pass search:

        Pass 1 — tight (2.5× token height):
          Standard multi-column table layout where the quantity column
          aligns closely with the nutrient row.

        Pass 2 — wide (8× token height):
          Single-column vitamin/supplement tables where the quantity sits
          far to the right and SAME_ROW was not built due to y-jitter.
          Only runs if Pass 1 returns nothing.
        """
        col        = self._get_same_col(graph, nutrient_node["id"], node_map)
        nut_height = max(nutrient_node["y2"] - nutrient_node["y1"], 20)

        tight = [
            n for n in col
            if n["label"] == "QUANTITY"
            and abs(n["cy"] - nutrient_node["cy"]) <= nut_height * 2.5
        ]
        if tight:
            tight.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
            return tight[:self.config["max_quantities_per_nutrient"]]

        wide = [
            n for n in col
            if n["label"] == "QUANTITY"
            and abs(n["cy"] - nutrient_node["cy"]) <= nut_height * 8.0
        ]
        wide.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
        return wide[:self.config["max_quantities_per_nutrient"]]

    # ── Main extraction ───────────────────────────────────────────────────────

    def extract(self, graph: dict, image_id: str = "unknown") -> list:
        """
        Extract 4-field tuples from the semantic graph.

        Returns List[Dict] with keys: image_id, nutrient, quantity, unit, context.
        """
        node_map   = self._build_node_map(graph)
        tuples     = []
        seen_pairs = set()

        for node in graph["nodes"]:
            if node["label"] != "NUTRIENT":
                continue

            nutrient_text = normalize_nutrient(node["token"])
            row_neighbors = self._get_same_row(graph, node["id"], node_map)

            # Quantities on the same row
            qty_nodes = [
                n for n in row_neighbors
                if n["label"] == "QUANTITY"
            ]

            # Nutrient-level context (fallback when quantity has no direct context)
            ctx_ancestors = self._get_context_ancestors(graph, node["id"], node_map)
            context_str   = None
            if ctx_ancestors:
                ctx_ancestors.sort(
                    key=lambda n: (n["x2"] - n["x1"]) * (n["y2"] - n["y1"]))
                context_str = ctx_ancestors[0].get("norm") or ctx_ancestors[0]["token"]

            # Sort quantities left→right: leftmost = per_100g, rightmost = per_serving
            qty_nodes.sort(key=lambda n: n["cx"])
            qty_nodes = qty_nodes[:self.config["max_quantities_per_nutrient"]]

            # Column fallback when no SAME_ROW quantities found
            if not qty_nodes:
                qty_nodes = self._column_fallback(graph, node, node_map)

            # No quantity found — emit bare nutrient tuple
            if not qty_nodes:
                pair_key = (nutrient_text, None)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    tuples.append({
                        "image_id": image_id,
                        "nutrient": nutrient_text,
                        "quantity": None,
                        "unit":     None,
                        "context":  context_str,
                    })
                continue

            for qty_node in qty_nodes:
                qty_value, inline_unit = parse_quantity(qty_node["token"])
                unit = inline_unit or self._find_unit(
                    qty_node, row_neighbors, graph, node_map)

                # Per-quantity context — correct for multi-column tables
                qty_context = (
                    self._get_context_for_qty(qty_node, graph, node_map)
                    or context_str
                )

                pair_key = (nutrient_text, qty_value)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                tuples.append({
                    "image_id": image_id,
                    "nutrient": nutrient_text,
                    "quantity": qty_value,
                    "unit":     unit,
                    "context":  qty_context,
                })

        return tuples

    # ── Output helpers ────────────────────────────────────────────────────────

    def print_tuples(self, tuples: list) -> None:
        print(f"\n{'='*65}")
        print(f"EXTRACTED TUPLES  ({len(tuples)} total)")
        print(f"{'='*65}")
        print(f"{'NUTRIENT':<35} {'QTY':<10} {'UNIT':<8} CONTEXT")
        print("-" * 75)
        for t in tuples:
            print(
                f"{str(t['nutrient'])[:34]:<35} "
                f"{str(t['quantity'] or '')[:9]:<10} "
                f"{str(t['unit'] or '')[:7]:<8} "
                f"{str(t['context'] or '')[:20]}"
            )
        print(f"{'='*65}\n")

    def save_csv(self, tuples: list, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_id", "nutrient", "quantity", "unit", "context"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tuples)
        print(f"Tuples saved: {output_path}  ({len(tuples)} rows)")