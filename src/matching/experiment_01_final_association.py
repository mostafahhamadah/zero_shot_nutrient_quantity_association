"""
association.py  v6
==================
Stage 5 — Graph Traversal Tuple Associator

CHANGE IN v6 (Classifier v4 compatibility):
  Context fields now read from node["norm"] instead of node["token"].
  SemanticClassifier v4 stores canonical context in the norm field
  (per_100g, per_serving, per_daily_dose) at classification time.
  Association must read norm to get the canonical value.

CHANGE IN v5 (Fix 3 — Column Selector):
  Multi-column table quantity disambiguation.

  Problem:
    EU supplement labels present two quantities per nutrient row —
    per_100g (left column) and per_serving (right column).
    The v4 algorithm sorted quantities by confidence (descending),
    which arbitrarily assigned left/right values, causing systematic
    per_100g/per_serving value swaps.

  Fix:
    1. Sort SAME_ROW quantities by x-coordinate (ascending).
       Left-to-right ordering matches EU label column conventions:
       leftmost quantity = per_100g, rightmost = per_serving.

    2. Get context per quantity node via its own CONTEXT_SCOPE edges.
       Each column header ('per 100g', 'per serving') scopes over
       the quantities below it. Using per-quantity context correctly
       distinguishes which value belongs to which column context.

CHANGE IN v4:
  Output schema: nutrient, quantity, unit, context.
  NRV and serving_size removed.
"""

import re
import csv
from pathlib import Path


# ── Quantity parsing ──────────────────────────────────────────────────────────

def parse_quantity(token_text: str) -> tuple:
    """
    Parse a QUANTITY token into (value_string, inline_unit_or_None).

    '400'      → ('400',  None)
    '0.43g'    → ('0.43', 'g')
    '1195(28]' → ('1195', None)
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

    Output per tuple: image_id, nutrient, quantity, unit, context.
    NRV% and serving_size are no longer extracted (v4).
    """

    def __init__(self, config: dict = None):
        self.config = config or {
            "max_quantities_per_nutrient": 3,
            "prefer_high_confidence":      True,
            "min_quantity_conf":           0.30,
        }

    # ── Graph helpers ─────────────────────────────────────────────────────────

    def _build_node_map(self, graph: dict) -> dict:
        return {n["id"]: n for n in graph["nodes"]}

    def _get_neighbors(self, graph: dict, node_id: int,
                       node_map: dict, edge_types: list) -> list:
        """Return all neighbors via given edge types (both directions)."""
        neighbors = []
        for e in graph["edges"]:
            if e["type"] not in edge_types:
                continue
            if e["src"] == node_id and e["dst"] in node_map:
                neighbors.append(node_map[e["dst"]])
            elif e["dst"] == node_id and e["src"] in node_map:
                neighbors.append(node_map[e["src"]])
        return neighbors

    def _get_same_row(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["SAME_ROW"])

    def _get_adjacent(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["ADJACENT"])

    def _get_same_col(self, graph, node_id, node_map):
        return self._get_neighbors(graph, node_id, node_map, ["SAME_COL"])

    def _get_context_ancestors(self, graph, node_id, node_map):
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
        Get context for a specific QUANTITY node via its own CONTEXT_SCOPE edges.

        FIX 3: Each quantity in a multi-column table has its own context
        header above it. By looking at the quantity node's own CONTEXT_SCOPE
        ancestors (not the nutrient's), we correctly distinguish:
          - leftmost quantity  → scoped under 'per 100g' header
          - rightmost quantity → scoped under 'per serving' header

        Falls back to None if no direct context found for this quantity.
        """
        ctx_ancestors = self._get_context_ancestors(graph, qty_node["id"], node_map)
        if not ctx_ancestors:
            return None
        # Pick the most specific context (smallest bounding box area)
        ctx_ancestors.sort(key=lambda n: (n["x2"]-n["x1"]) * (n["y2"]-n["y1"]))
        # v4 classifier: CONTEXT norm field already holds canonical form (per_100g etc.)
        return ctx_ancestors[0].get("norm") or ctx_ancestors[0]["token"]

    # ── Unit search ───────────────────────────────────────────────────────────

    def _find_unit(self, qty_node: dict, nutrient_row: list,
                   graph: dict, node_map: dict) -> str:
        """
        Find UNIT for a quantity node.

        Search order:
          1. Inline unit already in qty token text
          2. SAME_ROW of the quantity node
          3. ADJACENT to the quantity node
          4. SAME_ROW of the nutrient node (fallback)
        """
        _, inline = parse_quantity(qty_node["token"])
        if inline:
            return inline

        for n in self._get_same_row(graph, qty_node["id"], node_map):
            if n["label"] == "UNIT":
                return n["norm"].strip(".,*()[]| ")

        for n in self._get_adjacent(graph, qty_node["id"], node_map):
            if n["label"] == "UNIT":
                return n["norm"].strip(".,*()[]| ")

        for n in nutrient_row:
            if n["label"] == "UNIT":
                return n["norm"].strip(".,*()[]| ")

        return None

    # ── Column fallback ───────────────────────────────────────────────────────

    def _column_fallback(self, graph, nutrient_node, node_map):
        """
        Find QUANTITY in same column as nutrient.

        FIX 6: Two-pass column search.

        Pass 1 — tight (2.5× height): standard multi-column table search.
          Handles labels where quantity header is just above the first
          nutrient row.

        Pass 2 — wide (8× height): for single-column vitamin/supplement
          tables where the quantity column is far to the right and not
          SAME_COL aligned with the nutrient name column. In these layouts
          the quantity sits on the same visual row but the SAME_ROW edge
          was not built (y-jitter). The wider column search catches the
          nearest quantity in the same x-band.

        Pass 2 only runs if Pass 1 returns nothing.
        """
        col        = self._get_same_col(graph, nutrient_node["id"], node_map)
        nut_height = max(nutrient_node["y2"] - nutrient_node["y1"], 20)

        # Pass 1: tight vertical window
        tight = [
            n for n in col
            if n["label"] == "QUANTITY"
            and n["conf"] >= self.config["min_quantity_conf"]
            and abs(n["cy"] - nutrient_node["cy"]) <= nut_height * 2.5
        ]
        if tight:
            tight.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
            return tight[:self.config["max_quantities_per_nutrient"]]

        # Pass 2: wide vertical window — vitamin table fallback
        wide = [
            n for n in col
            if n["label"] == "QUANTITY"
            and n["conf"] >= self.config["min_quantity_conf"]
            and abs(n["cy"] - nutrient_node["cy"]) <= nut_height * 8.0
        ]
        wide.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
        return wide[:self.config["max_quantities_per_nutrient"]]

    # ── Main extraction ───────────────────────────────────────────────────────

    def extract(self, graph: dict, image_id: str = "unknown") -> list:
        """
        Extract 4-field tuples from the semantic graph.

        Returns list of dicts with keys:
          image_id, nutrient, quantity, unit, context
        """
        node_map   = self._build_node_map(graph)
        tuples     = []
        seen_pairs = set()

        for node in graph["nodes"]:
            if node["label"] != "NUTRIENT":
                continue

            nutrient_text = normalize_nutrient(node["token"])
            row_neighbors = self._get_same_row(graph, node["id"], node_map)

            # Quantities on same row
            qty_nodes = [
                n for n in row_neighbors
                if n["label"] == "QUANTITY"
                and n["conf"] >= self.config["min_quantity_conf"]
            ]

            # Context
            ctx_ancestors = self._get_context_ancestors(
                graph, node["id"], node_map)
            context_str = None
            if ctx_ancestors:
                ctx_ancestors.sort(
                    key=lambda n: (n["x2"]-n["x1"]) * (n["y2"]-n["y1"]))
                # v4 classifier: CONTEXT norm field already holds canonical form
                context_str = ctx_ancestors[0].get("norm") or ctx_ancestors[0]["token"]

            # FIX 3: Sort quantities left-to-right by x-coordinate.
            qty_nodes.sort(key=lambda n: n["cx"])
            qty_nodes = qty_nodes[:self.config["max_quantities_per_nutrient"]]

            # Column fallback
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
            else:
                for qty_node in qty_nodes:
                    qty_value, inline_unit = parse_quantity(qty_node["token"])
                    unit = inline_unit or self._find_unit(
                        qty_node, row_neighbors, graph, node_map)

                    # FIX 3: Try to get context specific to this quantity node.
                    # In multi-column tables each column has its own context header.
                    # If the quantity has its own CONTEXT_SCOPE ancestor, use it.
                    # Otherwise fall back to the nutrient-level context.
                    qty_context = self._get_context_for_qty(
                        qty_node, graph, node_map) or context_str

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
        print(f"EXTRACTED TUPLES  ({len(tuples)} total)  [4-field schema]")
        print(f"{'='*65}")
        print(f"{'NUTRIENT':<35} {'QTY':<10} {'UNIT':<8} {'CONTEXT'}")
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ocr.ocr_runner import run_ocr_on_image
    from src.utils.ocr_corrector import OCRCorrector
    from src.classification.semantic_classifier import SemanticClassifier
    from src.graph.graph_constructor import GraphConstructor

    IMAGE = "data/raw/1.jpeg"
    tokens    = run_ocr_on_image(IMAGE)
    corrected = OCRCorrector().correct_all(tokens)
    labeled   = SemanticClassifier(0.30, split_fused_tokens=True).classify_all(corrected)
    graph     = GraphConstructor().build(labeled)
    tuples    = TupleAssociator().extract(graph, "1.jpeg")
    TupleAssociator().print_tuples(tuples)