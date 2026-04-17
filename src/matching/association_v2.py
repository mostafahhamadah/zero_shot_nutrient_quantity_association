"""
association_v2.py
=================
Stage 5 — Geometry-Aware Tuple Association
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Extract structured nutritional tuples by traversing the enriched
semantic graph.  This replaces the original association.py which
relied on flat SAME_ROW traversal.

STRATEGY (Sections J–M of the master design)
---------------------------------------------
For each NUTRIENT node (anchor):

  Step 1 — Identify dosage streams
    Each DOSAGE column is an independent stream with its own context.
    The nutrient column maps to EACH dosage stream independently.

  Step 2 — Per-stream matching
    For each dosage stream, find the best QUANTITY match for this
    nutrient using a combined score:
      - geometry score:  row compatibility (perpendicular distance)
      - rank score:      data_rank_in_column alignment
      - role score:      column role compatibility bonus

  Step 3 — Unit resolution
    4-level fallback: inline → row neighbor → adjacent → nutrient row

  Step 4 — Context resolution
    Context = norm of the closest CONTEXT ancestor by cx-proximity.
    IMPORTANT: norm is preferred over column_context_id because the
    enricher may overwrite column_context_id when two context headers
    share the same induced column (e.g. "100 g" and "75 g (300 ml)"
    both in Col 2).  norm is set by the classifier and never overwritten.

  Step 5 — Collision resolution (Section L)
    One-to-one constraint: each quantity can only be claimed by one
    nutrient.  When multiple nutrients compete for the same quantity,
    the highest-scoring match wins.

SCORING (Section M)
-------------------
  total_score = w_geo * geometry_score
              + w_rank * rank_score
              + w_role * role_score

  geometry_score:
    1.0 - (perp_distance / max_perp) if row_compatible, else 0.0

  rank_score:
    1.0 if data_rank matches exactly
    0.5 if rank differs by 1
    0.0 otherwise

  role_score:
    0.2 bonus if quantity is in a DOSAGE-role column
    0.0 otherwise

  Default weights: w_geo=0.5, w_rank=0.4, w_role=0.1

  When to trust geometry vs rank:
    - High row_confidence → trust geometry
    - Low row_confidence (skewed/curved) → trust rank
    Adaptive: weight_geo is scaled by mean row_confidence.

EXP31 — ENERGY-AWARE QUANTITY FILTERING
----------------------------------------
  Energy nutrients (Energie/Energy/Brennwert) frequently get matched to
  wrong-row quantities (e.g. Fett's "0.8 g" instead of "1625 kJ").
  This happens because ROW_COMPAT edges include adjacent-row quantities
  and the cy-proximity filter picks the closest non-energy value.

  Fix: after collecting qty_nodes for an energy nutrient, check if any
  have energy-compatible units (kJ/kcal). If so, discard non-energy
  quantities. If none have energy units, fall back to a broader graph
  search for energy-unit quantities near this nutrient's cy.
"""

from __future__ import annotations

import re
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.geometry_engine import row_compatible, displacement_components

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEFAULT CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CONFIG = {
    "w_geo":                    0.5,
    "w_rank":                   0.4,
    "w_role":                   0.1,
    "max_perp_px":              25.0,
    "max_angle_deg":            15.0,
    "max_quantities_per_nutrient": 3,
    "adaptive_weighting":       True,   # scale geo weight by row_confidence
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENERGY-AWARE CONSTANTS (exp31)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ENERGY_NAME_RE = re.compile(
    r'energie|energy|brennwert|énergie|energia|energy\s*value|'
    r'valor\s*energ|valeur\s*energ|valore\s*energ|wartość\s*energ|'
    r'energetick|energijska',
    re.IGNORECASE,
)

_ENERGY_UNITS = frozenset({'kj', 'kcal', 'cal', 'kj/kcal', 'kj/kca', 'kcal/kj'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUANTITY PARSING (unchanged from v1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_quantity(token_text: str) -> Tuple[str, Optional[str]]:
    text = token_text.strip("*'\"()[]. ")
    # "123mg", "0.5g" — number fused with unit
    m = re.match(r'^(\d+[.,]?\d*)\s*(mg|g|kg|µg|mcg|ug|kj|kcal|ml|iu|ie)$',
                 text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "."), m.group(2).lower()
    # "1391 kJ (330kcal)", "278 kJ (66kcal)" — compound energy with unit
    m = re.match(r'^(\d+[.,]?\d*)\s*(kj|kcal)\s*[\(\[]',
                 text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "."), m.group(2).lower()
    # "1578 (37", "925(218)" — number with parenthetical (strip suffix)
    m = re.match(r'^(\d+[.,]?\d*)\s*[\(\[]', text)
    if m:
        return m.group(1).replace(",", "."), None
    # "123" — plain number
    m = re.match(r'^(\d+[.,]?\d*)$', text)
    if m:
        return m.group(1).replace(",", "."), None
    return text, None


def normalize_nutrient(token_text: str) -> str:
    text = token_text.strip("'\"*.,[]()/ ")
    return re.sub(r'\s+', ' ', text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIT REFINEMENT (exp35: right-of-qty directional preference)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In nutrition tables, the unit ALWAYS appears to the right of its
# quantity: "40 µg", "383 kcal", "0.8 g".
#
# The default CY-sort fails when ROW edges (|cy| < 25px) span two
# dense rows, pulling in UNIT tokens from an adjacent row that happen
# to have closer CY.  Example:
#
#   BIOTIN     40   µg   80%    (cy ≈ 500)
#   CALCIUM   121   mg   15%    (cy ≈ 518)
#
# CY-sort for qty "40" might pick "mg" from row below (cy_dist=18)
# over "µg" from its own row (cy_dist=0) if OCR cy values jitter.
#
# Fix: among UNIT candidates, prefer the CLOSEST one to the RIGHT
# of the quantity that is on the same TIGHT row (within 0.8× token
# height).  Falls back to CY-sort when no right-side candidate exists
# (e.g., column-header layouts where unit is above the quantity).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _refine_unit_pick(unit_cands: list, qty_node: dict) -> str:
    """Pick best unit: closest right-side same-row, else CY-sorted top."""
    if not unit_cands:
        return None

    qty_cx = qty_node.get("cx", 0)
    qty_cy = qty_node.get("cy", 0)
    qty_h  = max(qty_node.get("y2", 0) - qty_node.get("y1", 0), 15)

    # Phase 1: find UNIT candidates to the RIGHT of qty on same tight row
    right_same_row = []
    for c in unit_cands:
        c_cx = c.get("cx", 0)
        c_cy = c.get("cy", 0)
        # Must be to the right (reading order: number then unit)
        # AND on the same tight row (within 0.8× token height)
        if c_cx > qty_cx and abs(c_cy - qty_cy) <= qty_h * 0.8:
            right_same_row.append(c)

    if right_same_row:
        # Pick the closest right-side candidate (smallest horizontal gap)
        right_same_row.sort(key=lambda n: n.get("cx", 0) - qty_cx)
        return (right_same_row[0].get("norm") or "").strip(".,*()[]| ")

    # Phase 2: fallback to CY-sorted top pick (original behavior)
    # Handles column-header layouts where unit is above the quantity
    return (unit_cands[0].get("norm") or "").strip(".,*()[]| ")

def _geometry_score(nutrient: dict, quantity: dict,
                    max_perp: float, max_angle: float) -> float:
    compat, conf = row_compatible(nutrient, quantity, max_perp, max_angle)
    return conf if compat else 0.0


def _rank_score(nutrient: dict, quantity: dict) -> float:
    nr = nutrient.get("nutrient_rank_in_column", -1)
    qr = quantity.get("qty_rank_in_column", -1)
    if nr < 0:
        nr = nutrient.get("data_rank_in_column", -1)
    if qr < 0:
        qr = quantity.get("data_rank_in_column", -1)
    if nr < 0 or qr < 0:
        return 0.0
    diff = abs(nr - qr)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    elif diff == 2:
        return 0.2
    return 0.0


def _role_score(quantity: dict) -> float:
    return 0.2 if quantity.get("column_role") == "DOSAGE" else 0.0


def compute_match_score(nutrient: dict, quantity: dict,
                        config: dict) -> float:
    nut_row = nutrient.get("row_id", -1)
    qty_row = quantity.get("row_id", -2)
    if nut_row >= 0 and nut_row == qty_row:
        rank = _rank_score(nutrient, quantity)
        role = _role_score(quantity)
        return round(0.9 + 0.05 * rank + 0.05 * role, 4)

    geo  = _geometry_score(nutrient, quantity,
                           config["max_perp_px"], config["max_angle_deg"])
    rank = _rank_score(nutrient, quantity)
    role = _role_score(quantity)

    w_geo  = config["w_geo"]
    w_rank = config["w_rank"]
    w_role = config["w_role"]

    if config.get("adaptive_weighting"):
        mean_conf = (nutrient.get("row_confidence", 1.0) +
                     quantity.get("row_confidence", 1.0)) / 2.0
        w_geo_adj  = w_geo * mean_conf
        w_rank_adj = w_rank + (w_geo * (1.0 - mean_conf))
    else:
        w_geo_adj  = w_geo
        w_rank_adj = w_rank

    total = w_geo_adj * geo + w_rank_adj * rank + w_role * role
    return round(total, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ASSOCIATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TupleAssociatorV2:

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.diagnostics: dict = {}

    # ── Energy-aware helpers (exp31) ──────────────────────────────────

    @staticmethod
    def _is_energy_nutrient(nutrient_text: str) -> bool:
        """True if this nutrient is an energy/Energie/Brennwert type."""
        return bool(_ENERGY_NAME_RE.search(nutrient_text))

    @staticmethod
    def _qty_has_energy_unit(qty_node: dict,
                             row_edges: dict, adj_edges: dict) -> bool:
        """
        Check whether a QUANTITY node carries an energy unit.
        Checks: (1) inline unit from parse_quantity, (2) ROW neighbor
        UNIT tokens, (3) ADJ neighbor UNIT tokens.
        """
        _, inline = parse_quantity(qty_node.get("token", ""))
        if inline and inline.lower() in _ENERGY_UNITS:
            return True
        # Check row/adj neighbors for energy UNIT tokens
        for nb in row_edges.get(qty_node.get("id"), []):
            if nb.get("label") == "UNIT":
                u = (nb.get("norm") or "").strip(".,*()[]| ").lower()
                if u in _ENERGY_UNITS:
                    return True
        for nb in adj_edges.get(qty_node.get("id"), []):
            if nb.get("label") == "UNIT":
                u = (nb.get("norm") or "").strip(".,*()[]| ").lower()
                if u in _ENERGY_UNITS:
                    return True
        return False

    def _energy_filter_qtys(self, nutrient_text: str, qty_nodes: list,
                            row_edges: dict, adj_edges: dict,
                            all_quantities: list, nut_cy: float,
                            max_cy_dist: float) -> list:
        """
        exp31: For energy nutrients, prefer quantities with energy units.

        Strategy:
          1. If any qty_nodes already have energy units → keep only those.
          2. If none do → search ALL graph QUANTITY nodes within max_cy_dist
             of the nutrient's cy for energy-unit quantities.
          3. If still none → return original qty_nodes unchanged.

        This prevents Energie from grabbing Fett's "0.8 g" when "1625 kJ"
        is available nearby.
        """
        if not self._is_energy_nutrient(nutrient_text):
            return qty_nodes

        # Step 1: filter existing qty_nodes
        energy_qtys = [q for q in qty_nodes
                       if self._qty_has_energy_unit(q, row_edges, adj_edges)]

        if energy_qtys:
            return energy_qtys

        # Step 2: broader search across ALL quantities in the graph
        broad_candidates = []
        for q in all_quantities:
            cy_dist = abs(q.get("cy", 0) - nut_cy)
            if cy_dist <= max_cy_dist and self._qty_has_energy_unit(q, row_edges, adj_edges):
                broad_candidates.append((cy_dist, q))

        if broad_candidates:
            broad_candidates.sort(key=lambda x: x[0])
            max_per = self.config["max_quantities_per_nutrient"]
            # Energy nutrients get 2x budget (kJ + kcal per context)
            result = [q for _, q in broad_candidates[:max_per * 2]]
            result.sort(key=lambda n: n.get("cx", 0))  # left→right
            return result

        # Step 3: nothing found — return originals unchanged
        return qty_nodes

    # ── Main entry ────────────────────────────────────────────────────

    def extract(self, graph: dict, image_id: str = "unknown") -> List[dict]:
        node_map = {n["id"]: n for n in graph["nodes"]}

        max_y = max((n.get("y2", 0) for n in graph["nodes"]), default=500)
        scale = max(max_y / 500.0, 1.0)
        scaled_config = {**self.config}
        scaled_config["max_perp_px"] = self.config["max_perp_px"] * scale

        nutrients = [n for n in graph["nodes"] if n.get("label") == "NUTRIENT"]
        quantities = [n for n in graph["nodes"] if n.get("label") == "QUANTITY"]

        stream_qtys: Dict[int, List[dict]] = defaultdict(list)
        unstreamed_qtys: List[dict] = []
        for q in quantities:
            sid = q.get("dosage_stream_id", -1)
            if sid >= 0:
                stream_qtys[sid].append(q)
            else:
                unstreamed_qtys.append(q)

        has_streams = len(stream_qtys) > 0

        if has_streams:
            tuples = self._extract_v2_streams(
                graph, node_map, nutrients, quantities,
                stream_qtys, unstreamed_qtys, scaled_config, image_id)
            mode = "v2_streams"
        else:
            tuples = self._extract_v1_rows(
                graph, node_map, nutrients, quantities, image_id)
            mode = "v1_rows"

        self.diagnostics = {
            "nutrients":       len(nutrients),
            "quantities":      len(quantities),
            "dosage_streams":  len(stream_qtys),
            "mode":            mode,
            "tuples":          len(tuples),
        }

        return tuples

    # ══════════════════════════════════════════════════════════════════
    # V2 PATH — dosage stream matching (for multi-column tables)
    # ══════════════════════════════════════════════════════════════════

    def _extract_v2_streams(self, graph, node_map, nutrients, quantities,
                            stream_qtys, unstreamed_qtys, scaled_config,
                            image_id) -> List[dict]:
        max_per_nut = self.config["max_quantities_per_nutrient"]

        row_edges = defaultdict(list)
        col_edges = defaultdict(list)
        adj_edges = defaultdict(list)
        for e in graph["edges"]:
            dst = node_map.get(e["dst"])
            if not dst:
                continue
            if e["type"] == "ROW_COMPAT":
                row_edges[e["src"]].append(dst)
            elif e["type"] == "COL_COMPAT":
                col_edges[e["src"]].append(dst)
            elif e["type"] == "DIRECTIONAL_ADJ":
                adj_edges[e["src"]].append(dst)

        tuples = []
        seen_pairs = set()

        for nut in nutrients:
            nutrient_text = normalize_nutrient(nut.get("token", ""))
            nut_id = nut["id"]
            nut_cy = nut.get("cy", 0)

            row_neighbors = row_edges.get(nut_id, [])
            qty_nodes = [n for n in row_neighbors if n.get("label") == "QUANTITY"]

            # CY-PROXIMITY FILTER: when ROW_COMPAT is loose (high-res images),
            # a nutrient may connect to quantities from adjacent rows.
            # Keep only the closest by vertical distance, then sort left→right.
            qty_nodes.sort(key=lambda n: abs(n.get("cy", 0) - nut_cy))
            qty_nodes = qty_nodes[:max_per_nut]
            # Sort left→right for multi-column assignment (per_100g | per_serving)
            qty_nodes.sort(key=lambda n: n.get("cx", 0))

            if not qty_nodes:
                col_neighbors = col_edges.get(nut_id, [])
                nut_height = max(nut["y2"] - nut["y1"], 20)
                tight = [n for n in col_neighbors
                         if n.get("label") == "QUANTITY"
                         and abs(n.get("cy", 0) - nut.get("cy", 0)) <= nut_height * 2.5]
                if tight:
                    tight.sort(key=lambda n: abs(n.get("cy", 0) - nut.get("cy", 0)))
                    qty_nodes = tight[:max_per_nut]
                else:
                    wide = [n for n in col_neighbors
                            if n.get("label") == "QUANTITY"
                            and abs(n.get("cy", 0) - nut.get("cy", 0)) <= nut_height * 8.0]
                    wide.sort(key=lambda n: abs(n.get("cy", 0) - nut.get("cy", 0)))
                    qty_nodes = wide[:max_per_nut]

            if not qty_nodes:
                nut_rank = nut.get("nutrient_rank_in_column", -1)
                if nut_rank >= 0:
                    for sid, qty_list in stream_qtys.items():
                        for q in qty_list:
                            if q.get("qty_rank_in_column", -1) == nut_rank:
                                qty_nodes.append(q)
                    qty_nodes = qty_nodes[:max_per_nut]

            # ── exp31: energy-aware quantity filtering ──────────────
            # For Energie/Energy/Brennwert, prefer quantities with
            # energy units (kJ/kcal) over non-energy quantities (g/mg).
            nut_height = max(nut.get("y2", 0) - nut.get("y1", 0), 20)
            qty_nodes = self._energy_filter_qtys(
                nutrient_text, qty_nodes,
                row_edges, adj_edges,
                quantities, nut_cy,
                max_cy_dist=nut_height * 4.0,
            )
            # ────────────────────────────────────────────────────────

            if not qty_nodes:
                context = nut.get("column_context_id")
                if not context:
                    context = self._context_from_graph(nut, graph, node_map)
                pair_key = (nutrient_text, None)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    tuples.append({
                        "image_id": image_id,
                        "nutrient": nutrient_text,
                        "quantity": None,
                        "unit":     None,
                        "context":  context,
                        "_score":   0.0,
                        "_stream":  -1,
                    })
                continue

            for qty_node in qty_nodes:
                qty_value, inline_unit = parse_quantity(qty_node.get("token", ""))

                # Unit: inline → closest-cy row of qty → closest-cy adj of qty → closest-cy row of nutrient
                unit = inline_unit
                if not unit:
                    unit_cands = [n for n in row_edges.get(qty_node["id"], [])
                                  if n.get("label") == "UNIT"]
                    if unit_cands:
                        unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
                        unit = _refine_unit_pick(unit_cands, qty_node)
                if not unit:
                    unit_cands = [n for n in adj_edges.get(qty_node["id"], [])
                                  if n.get("label") == "UNIT"]
                    if unit_cands:
                        unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
                        unit = _refine_unit_pick(unit_cands, qty_node)
                if not unit:
                    unit_cands = [n for n in row_neighbors if n.get("label") == "UNIT"]
                    if unit_cands:
                        unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
                        unit = _refine_unit_pick(unit_cands, qty_node)
                if not unit:
                    unit = self._extract_embedded_unit(nut.get("token", ""))

                context = self._resolve_context_for_qty(
                    qty_node, nut, graph, node_map)

                pair_key = (nutrient_text, qty_value)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                tuples.append({
                    "image_id": image_id,
                    "nutrient": nutrient_text,
                    "quantity": qty_value,
                    "unit":     unit,
                    "context":  context,
                    "_score":   1.0,
                    "_stream":  qty_node.get("dosage_stream_id", -1),
                })

        return tuples

    # ══════════════════════════════════════════════════════════════════
    # V1 PATH — SAME_ROW traversal (proven baseline for flat tables)
    # ══════════════════════════════════════════════════════════════════

    def _extract_v1_rows(self, graph, node_map, nutrients,
                         quantities, image_id) -> List[dict]:
        edge_types_present = set(e["type"] for e in graph.get("edges", []))
        ROW_EDGE = "ROW_COMPAT" if "ROW_COMPAT" in edge_types_present else "SAME_ROW"
        COL_EDGE = "COL_COMPAT" if "COL_COMPAT" in edge_types_present else "SAME_COL"
        ADJ_EDGE = "DIRECTIONAL_ADJ" if "DIRECTIONAL_ADJ" in edge_types_present else "ADJACENT"
        CTX_EDGES = [et for et in ["CONTEXT_SCOPE", "HEADER_SCOPE"] if et in edge_types_present]
        if not CTX_EDGES:
            CTX_EDGES = ["CONTEXT_SCOPE"]

        def _get_neighbors(node_id, edge_type):
            return [node_map[e["dst"]] for e in graph["edges"]
                    if e["src"] == node_id and e["type"] == edge_type
                    and e["dst"] in node_map]

        def _get_context_ancestors(node_id):
            ancestors = []
            for ctx_type in CTX_EDGES:
                ancestors.extend([
                    node_map[e["src"]] for e in graph["edges"]
                    if e["dst"] == node_id and e["type"] == ctx_type
                    and e["src"] in node_map
                    and node_map[e["src"]].get("label") == "CONTEXT"
                ])
            return ancestors

        def _get_context_for_qty(qty_node):
            ancestors = _get_context_ancestors(qty_node["id"])
            if not ancestors:
                return qty_node.get("column_context_id")
            qty_cx = qty_node.get("cx", (qty_node.get("x1",0)+qty_node.get("x2",0))/2.0)
            ancestors.sort(key=lambda n: abs(
                n.get("cx", (n.get("x1",0)+n.get("x2",0))/2.0) - qty_cx))
            # FIX: prefer norm (classifier-set, never overwritten) over
            # column_context_id (enricher-set, may be wrong when two
            # context headers share one induced column)
            return (ancestors[0].get("norm")
                    or ancestors[0].get("column_context_id")
                    or ancestors[0]["token"])

        def _find_unit_v1(qty_node, nutrient_row):
            _, inline = parse_quantity(qty_node["token"])
            if inline:
                return inline
            qty_cy = qty_node.get("cy", 0)
            unit_cands = [n for n in _get_neighbors(qty_node["id"], ROW_EDGE)
                          if n.get("label") == "UNIT"]
            if unit_cands:
                unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
                return _refine_unit_pick(unit_cands, qty_node)
            unit_cands = [n for n in _get_neighbors(qty_node["id"], ADJ_EDGE)
                          if n.get("label") == "UNIT"]
            if unit_cands:
                unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
                return _refine_unit_pick(unit_cands, qty_node)
            unit_cands = [n for n in nutrient_row if n.get("label") == "UNIT"]
            if unit_cands:
                unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
                return _refine_unit_pick(unit_cands, qty_node)
            return None

        def _column_fallback(nutrient_node):
            col = _get_neighbors(nutrient_node["id"], COL_EDGE)
            nut_height = max(nutrient_node["y2"] - nutrient_node["y1"], 20)
            tight = [n for n in col if n.get("label") == "QUANTITY"
                     and abs(n["cy"] - nutrient_node["cy"]) <= nut_height * 2.5]
            if tight:
                tight.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
                return tight[:self.config["max_quantities_per_nutrient"]]
            wide = [n for n in col if n.get("label") == "QUANTITY"
                    and abs(n["cy"] - nutrient_node["cy"]) <= nut_height * 8.0]
            wide.sort(key=lambda n: abs(n["cy"] - nutrient_node["cy"]))
            return wide[:self.config["max_quantities_per_nutrient"]]

        # Build edge dicts for energy filter helper
        row_edges_v1 = defaultdict(list)
        adj_edges_v1 = defaultdict(list)
        for e in graph["edges"]:
            dst = node_map.get(e["dst"])
            if not dst:
                continue
            if e["type"] == ROW_EDGE:
                row_edges_v1[e["src"]].append(dst)
            elif e["type"] == ADJ_EDGE:
                adj_edges_v1[e["src"]].append(dst)

        tuples = []
        seen_pairs = set()
        max_per_nut = self.config["max_quantities_per_nutrient"]

        for node in graph["nodes"]:
            if node.get("label") != "NUTRIENT":
                continue

            nutrient_text = normalize_nutrient(node["token"])
            row_neighbors = _get_neighbors(node["id"], ROW_EDGE)

            qty_nodes = [n for n in row_neighbors if n.get("label") == "QUANTITY"]

            ctx_ancestors = _get_context_ancestors(node["id"])
            context_str = None
            if ctx_ancestors:
                nut_cx = node.get("cx", (node.get("x1",0)+node.get("x2",0))/2.0)
                ctx_ancestors.sort(key=lambda n: abs(
                    n.get("cx", (n.get("x1",0)+n.get("x2",0))/2.0) - nut_cx))
                # FIX: prefer norm over column_context_id
                context_str = (ctx_ancestors[0].get("norm")
                               or ctx_ancestors[0].get("column_context_id")
                               or ctx_ancestors[0]["token"])
            if not context_str:
                context_str = node.get("column_context_id")

            # CY-PROXIMITY FILTER: keep closest quantities by vertical distance
            node_cy = node.get("cy", 0)
            qty_nodes.sort(key=lambda n: abs(n.get("cy", 0) - node_cy))
            qty_nodes = qty_nodes[:max_per_nut]
            # Sort left→right for multi-column assignment
            qty_nodes.sort(key=lambda n: n.get("cx", 0))

            if not qty_nodes:
                qty_nodes = _column_fallback(node)

            # ── exp31: energy-aware quantity filtering ──────────────
            nut_height = max(node.get("y2", 0) - node.get("y1", 0), 20)
            qty_nodes = self._energy_filter_qtys(
                nutrient_text, qty_nodes,
                row_edges_v1, adj_edges_v1,
                quantities, node_cy,
                max_cy_dist=nut_height * 4.0,
            )
            # ────────────────────────────────────────────────────────

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
                        "_score":   0.0,
                        "_stream":  -1,
                    })
                continue

            for qty_node in qty_nodes:
                qty_value, inline_unit = parse_quantity(qty_node["token"])
                unit = inline_unit or _find_unit_v1(qty_node, row_neighbors)
                if not unit:
                    unit = self._extract_embedded_unit(node.get("token", ""))

                qty_context = _get_context_for_qty(qty_node) or context_str

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
                    "_score":   1.0,
                    "_stream":  -1,
                })

        return tuples

    # ── Unit search (4-level fallback) ────────────────────────────────

    def _find_unit(self, qty_node: dict, nut_node: dict,
                   graph: dict, node_map: dict) -> Optional[str]:
        for e in graph["edges"]:
            if e["src"] == qty_node["id"] and e["type"] == "ROW_COMPAT":
                nb = node_map.get(e["dst"])
                if nb and nb.get("label") == "UNIT":
                    return (nb.get("norm") or "").strip(".,*()[]| ")

        for e in graph["edges"]:
            if e["src"] == qty_node["id"] and e["type"] == "DIRECTIONAL_ADJ":
                nb = node_map.get(e["dst"])
                if nb and nb.get("label") == "UNIT":
                    return (nb.get("norm") or "").strip(".,*()[]| ")

        for e in graph["edges"]:
            if e["src"] == nut_node["id"] and e["type"] == "ROW_COMPAT":
                nb = node_map.get(e["dst"])
                if nb and nb.get("label") == "UNIT":
                    return (nb.get("norm") or "").strip(".,*()[]| ")

        return self._extract_embedded_unit(nut_node.get("token", ""))

    # ── Embedded unit extraction ──────────────────────────────────

    _EMBEDDED_UNIT_RE = re.compile(
        r'[\(\s/]'
        r'(kj/kcal|kj/kca|kcal/kj|kj|kcal|cal'
        r'|mg\s*(?:ne|re|α-te|a-te)?|µg\s*(?:re|ne|te)?'
        r'|mcg|ug|μg|kg|ml|dl|cl|ie|iu|kbe|cfu'
        r'|g)\s*[\)\s,.:;]*$',
        re.IGNORECASE,
    )

    def _extract_embedded_unit(self, nutrient_text: str) -> Optional[str]:
        m = self._EMBEDDED_UNIT_RE.search(nutrient_text)
        if m:
            return m.group(1).strip().lower()
        return None

    def _context_from_graph(self, node: dict, graph: dict,
                            node_map: dict) -> Optional[str]:
        """
        Resolve context via CONTEXT_SCOPE / HEADER_SCOPE incoming edges.

        When multiple CONTEXT ancestors exist (dual-column tables), picks
        the one closest in cx to the target node — this selects the correct
        column's context header (per_100g vs per_serving).

        FIX: reads norm (classifier-set) before column_context_id (enricher-set)
        to avoid the shared-column overwrite bug where the enricher propagates
        the last header's context to ALL tokens in the column.
        """
        node_cx = node.get("cx", (node.get("x1", 0) + node.get("x2", 0)) / 2.0)
        candidates = []

        for e in graph["edges"]:
            if e["dst"] == node["id"] and e["type"] in ("HEADER_SCOPE", "CONTEXT_SCOPE"):
                src = node_map.get(e["src"])
                if src and src.get("label") == "CONTEXT":
                    src_cx = src.get("cx", (src.get("x1", 0) + src.get("x2", 0)) / 2.0)
                    h_dist = abs(src_cx - node_cx)
                    # FIX: prefer norm over column_context_id
                    text = (src.get("norm")
                            or src.get("column_context_id")
                            or src.get("token"))
                    if text:
                        candidates.append((h_dist, text))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _resolve_context_for_qty(self, qty_node: dict, nut_node: dict,
                                  graph: dict, node_map: dict) -> Optional[str]:
        """
        Full context resolution chain for a (nutrient, quantity) pair.

        Priority:
          1. CONTEXT_SCOPE/HEADER_SCOPE ancestors of QUANTITY — closest cx wins.
          2. column_context_id from enricher (on the quantity node).
          3. CONTEXT_SCOPE/HEADER_SCOPE ancestors of NUTRIENT — closest cx wins.
          4. column_context_id from enricher (on the nutrient node).
        """
        ctx = self._context_from_graph(qty_node, graph, node_map)
        if ctx:
            return ctx
        ctx = qty_node.get("column_context_id")
        if ctx:
            return ctx
        ctx = self._context_from_graph(nut_node, graph, node_map)
        if ctx:
            return ctx
        ctx = nut_node.get("column_context_id")
        if ctx:
            return ctx
        return None

    # ── Output ────────────────────────────────────────────────────────

    def print_tuples(self, tuples: List[dict]) -> None:
        print(f"\n{'='*80}")
        print(f"EXTRACTED TUPLES V2  ({len(tuples)} total)")
        print(f"{'='*80}")
        print(f"{'NUTRIENT':<30} {'QTY':<10} {'UNIT':<8} {'CONTEXT':<18} "
              f"{'SCORE':<8} {'STREAM'}")
        print("-" * 80)
        for t in tuples:
            print(
                f"{str(t['nutrient'])[:29]:<30} "
                f"{str(t['quantity'] or '')[:9]:<10} "
                f"{str(t['unit'] or '')[:7]:<8} "
                f"{str(t['context'] or '')[:17]:<18} "
                f"{t.get('_score', 0):<8.3f} "
                f"{t.get('_stream', -1)}"
            )
        print(f"{'='*80}\n")

    def save_csv(self, tuples: List[dict], output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_id", "nutrient", "quantity", "unit", "context"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tuples)
        print(f"Tuples saved: {output_path}  ({len(tuples)} rows)")

    def print_diagnostics(self) -> None:
        d = self.diagnostics
        print(f"\n{'='*50}")
        print("ASSOCIATION V2 DIAGNOSTICS")
        print(f"{'='*50}")
        for k, v in d.items():
            print(f"  {k:<20} : {v}")
        print(f"{'='*50}\n")