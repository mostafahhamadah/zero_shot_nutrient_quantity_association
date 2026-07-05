"""
association_rank_first.py
=========================
Stage 5 — Rank-First Tuple Association
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Extract structured nutritional tuples from the enriched semantic graph using
RANK ALIGNMENT as the primary matching method.

Main difference from association_v2.py:
    V2 current behavior: ROW_COMPAT first → COL fallback → rank fallback.
    This version:       RANK first      → ROW fallback → COL fallback → geometry fallback.

EXPECTED INPUT
--------------
Graph produced by graph_constructor_v2.py or a compatible graph.
Nodes are expected to contain enriched fields where available:
    label, token, norm, x1,y1,x2,y2,cx,cy,
    row_id, column_id, column_role,
    dosage_stream_id,
    nutrient_rank_in_column, qty_rank_in_column, data_rank_in_column,
    column_context_id.

OUTPUT
------
List[Dict] with fields:
    image_id, nutrient, quantity, unit, context

Diagnostic fields are included internally in tuples:
    _score, _stream, _method, _rank_score
They are ignored when saving CSV.
"""

from __future__ import annotations

import re
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

try:
    from src.utils.geometry_engine import row_compatible
except Exception:  # keeps the file importable during isolated testing
    def row_compatible(a: dict, b: dict, max_perp: float, max_angle: float):
        """Fallback row compatibility: direct cy distance only."""
        a_cy = a.get("cy", (a.get("y1", 0) + a.get("y2", 0)) / 2.0)
        b_cy = b.get("cy", (b.get("y1", 0) + b.get("y2", 0)) / 2.0)
        dist = abs(a_cy - b_cy)
        if dist <= max_perp:
            return True, max(0.0, 1.0 - dist / max_perp)
        return False, 0.0

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    # Candidate limits
    "max_quantities_per_nutrient": 3,
    "energy_max_multiplier": 2,

    # Rank-first behavior
    "allow_near_rank_fallback": True,
    "near_rank_max_diff": 1,
    "rank_first_exact_only_per_stream": True,

    # Rank-primary acceptance filter
    # Rank is tried first, but it is accepted only when it is confirmed by
    # independent structural evidence. The rank candidate must be:
    #   1) an exact rank match,
    #   2) from a DOSAGE column, and
    #   3) either ROW_COMPAT/SAME_ROW-confirmed OR stream+context-confirmed.
    # This prevents rank from blindly accepting structurally aligned but
    # semantically invalid numbers such as NRV%, serving-size, or noisy values.
    "rank_primary_filter_enabled": True,
    "rank_primary_min_rank_score": 1.0,
    "rank_primary_require_dosage_role": True,
    "rank_primary_allow_missing_role": False,
    "rank_primary_require_confirmation": True,
    "rank_primary_allow_row_confirmation": True,
    "rank_primary_allow_stream_context_confirmation": True,

    # Geometry fallback/scoring
    "max_perp_px": 25.0,
    "max_angle_deg": 15.0,
    "min_global_score": 0.35,
    "w_geo": 0.35,
    "w_rank": 0.55,
    "w_role": 0.10,
    "adaptive_weighting": True,

    # Column fallback windows
    "col_tight_height_mult": 2.5,
    "col_wide_height_mult": 8.0,
}


_ENERGY_NAME_RE = re.compile(
    r'energie|energy|brennwert|énergie|energia|energy\s*value|'
    r'valor\s*energ|valeur\s*energ|valore\s*energ|wartość\s*energ|'
    r'energetick|energijska',
    re.IGNORECASE,
)

_ENERGY_UNITS = frozenset({'kj', 'kcal', 'cal', 'kj/kcal', 'kj/kca', 'kcal/kj'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parsing / normalisation helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_quantity(token_text: str) -> Tuple[str, Optional[str]]:
    """Parse a QUANTITY token into (quantity_value, inline_unit_or_None)."""
    text = str(token_text or "").strip("*'\"()[]. ")

    # "123mg", "0.5 g", "1391 kJ"
    m = re.match(r'^(\d+[.,]?\d*)\s*(mg|g|kg|µg|mcg|ug|μg|kj|kcal|cal|ml|iu|ie)$',
                 text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "."), m.group(2).lower()

    # "1391 kJ (330kcal)", "278 kJ (66kcal)"
    m = re.match(r'^(\d+[.,]?\d*)\s*(kj|kcal|cal)\s*[\(\[]',
                 text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "."), m.group(2).lower()

    # "1578 (37", "925(218)" — keep first number
    m = re.match(r'^(\d+[.,]?\d*)\s*[\(\[]', text)
    if m:
        return m.group(1).replace(",", "."), None

    # plain number
    m = re.match(r'^(\d+[.,]?\d*)$', text)
    if m:
        return m.group(1).replace(",", "."), None

    return text, None


def normalize_nutrient(token_text: str) -> str:
    text = str(token_text or "").strip("'\"*.,[]()/ ")
    return re.sub(r'\s+', ' ', text)


_EMBEDDED_UNIT_RE = re.compile(
    r'[\(\s/]'
    r'(kj/kcal|kj/kca|kcal/kj|kj|kcal|cal'
    r'|mg\s*(?:ne|re|α-te|a-te)?|µg\s*(?:re|ne|te)?'
    r'|mcg|ug|μg|kg|ml|dl|cl|ie|iu|kbe|cfu'
    r'|g)\s*[\)\s,.:;]*$',
    re.IGNORECASE,
)


def extract_embedded_unit(text: str) -> Optional[str]:
    m = _EMBEDDED_UNIT_RE.search(str(text or ""))
    if m:
        return m.group(1).strip().lower()
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ranking / scoring helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_nutrient_rank(n: dict) -> int:
    for key in ("nutrient_rank_in_column", "data_rank_in_column", "rank"):
        val = n.get(key, -1)
        if isinstance(val, (int, float)) and int(val) >= 0:
            return int(val)
    return -1


def _get_quantity_rank(q: dict) -> int:
    for key in ("qty_rank_in_column", "data_rank_in_column", "rank"):
        val = q.get(key, -1)
        if isinstance(val, (int, float)) and int(val) >= 0:
            return int(val)
    return -1


def _rank_score(nutrient: dict, quantity: dict) -> float:
    nr = _get_nutrient_rank(nutrient)
    qr = _get_quantity_rank(quantity)
    if nr < 0 or qr < 0:
        return 0.0
    diff = abs(nr - qr)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.2
    return 0.0


def _role_score(quantity: dict) -> float:
    return 0.2 if quantity.get("column_role") == "DOSAGE" else 0.0


def _geometry_score(nutrient: dict, quantity: dict,
                    max_perp: float, max_angle: float) -> float:
    compat, conf = row_compatible(nutrient, quantity, max_perp, max_angle)
    return float(conf) if compat else 0.0


def _cy_dist(a: dict, b: dict) -> float:
    return abs(a.get("cy", 0.0) - b.get("cy", 0.0))


def _cx_dist(a: dict, b: dict) -> float:
    return abs(a.get("cx", 0.0) - b.get("cx", 0.0))


def compute_match_score_rank_first(nutrient: dict, quantity: dict,
                                   config: dict) -> float:
    """
    Rank-dominant score used only for tie-breaking and final fallback.

    Important: exact rank alignment is intentionally weighted higher than
    geometry. Geometry should refine rank matches, not replace them.
    """
    rank = _rank_score(nutrient, quantity)
    geo = _geometry_score(nutrient, quantity,
                          config["max_perp_px"], config["max_angle_deg"])
    role = _role_score(quantity)

    w_rank = config["w_rank"]
    w_geo = config["w_geo"]
    w_role = config["w_role"]

    if config.get("adaptive_weighting"):
        mean_conf = (nutrient.get("row_confidence", 1.0) +
                     quantity.get("row_confidence", 1.0)) / 2.0
        # If row confidence is low, shift even more trust toward rank.
        w_geo_adj = w_geo * mean_conf
        w_rank_adj = w_rank + (w_geo * (1.0 - mean_conf))
    else:
        w_geo_adj = w_geo
        w_rank_adj = w_rank

    return round(w_rank_adj * rank + w_geo_adj * geo + w_role * role, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unit helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _refine_unit_pick(unit_cands: List[dict], qty_node: dict) -> Optional[str]:
    """Prefer closest UNIT to the right of the quantity on the same tight row."""
    if not unit_cands:
        return None

    qty_cx = qty_node.get("cx", 0)
    qty_cy = qty_node.get("cy", 0)
    qty_h = max(qty_node.get("y2", 0) - qty_node.get("y1", 0), 15)

    right_same_row = []
    for c in unit_cands:
        if (c.get("cx", 0) > qty_cx and
                abs(c.get("cy", 0) - qty_cy) <= qty_h * 0.8):
            right_same_row.append(c)

    if right_same_row:
        right_same_row.sort(key=lambda n: n.get("cx", 0) - qty_cx)
        return (right_same_row[0].get("norm") or "").strip(".,*()[]| ") or None

    # fallback: candidate list should already be sorted by vertical closeness
    return (unit_cands[0].get("norm") or "").strip(".,*()[]| ") or None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main associator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TupleAssociatorRankFirst:
    """
    Rank-first tuple associator.

    Matching priority for each nutrient:
      1. Rank-primary candidate search.
      2. Accept rank only if: exact rank match + DOSAGE column + independent confirmation.
         Confirmation means either ROW_COMPAT/SAME_ROW exists, or the quantity has
         a valid dosage stream and a resolvable context.
      3. If rank is missing or rejected, use ROW_COMPAT/SAME_ROW fallback.
      4. If row fallback fails, use COL_COMPAT/SAME_COL fallback.
      5. If column fallback fails, use global score fallback.

    Unit and context resolution reuse the stronger V2 logic.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.diagnostics: dict = {}

    # ── Graph edge helpers ──────────────────────────────────────────

    def _edge_names(self, graph: dict) -> dict:
        types = {e.get("type") for e in graph.get("edges", [])}
        return {
            "ROW": "ROW_COMPAT" if "ROW_COMPAT" in types else "SAME_ROW",
            "COL": "COL_COMPAT" if "COL_COMPAT" in types else "SAME_COL",
            "ADJ": "DIRECTIONAL_ADJ" if "DIRECTIONAL_ADJ" in types else "ADJACENT",
            "CTX": [t for t in ("HEADER_SCOPE", "CONTEXT_SCOPE") if t in types] or ["CONTEXT_SCOPE"],
        }

    def _build_edge_maps(self, graph: dict, node_map: dict) -> dict:
        names = self._edge_names(graph)
        row_edges = defaultdict(list)
        col_edges = defaultdict(list)
        adj_edges = defaultdict(list)
        ctx_in_edges = defaultdict(list)

        for e in graph.get("edges", []):
            src, dst, et = e.get("src"), e.get("dst"), e.get("type")
            dst_node = node_map.get(dst)
            src_node = node_map.get(src)
            if not src_node or not dst_node:
                continue
            if et == names["ROW"]:
                row_edges[src].append(dst_node)
            elif et == names["COL"]:
                col_edges[src].append(dst_node)
            elif et == names["ADJ"]:
                adj_edges[src].append(dst_node)
            elif et in names["CTX"] and src_node.get("label") == "CONTEXT":
                ctx_in_edges[dst].append(src_node)

        return {
            "names": names,
            "row": row_edges,
            "col": col_edges,
            "adj": adj_edges,
            "ctx_in": ctx_in_edges,
        }

    # ── Energy helpers ──────────────────────────────────────────────

    @staticmethod
    def _is_energy_nutrient(nutrient_text: str) -> bool:
        return bool(_ENERGY_NAME_RE.search(nutrient_text or ""))

    @staticmethod
    def _qty_has_energy_unit(qty_node: dict,
                             row_edges: dict, adj_edges: dict) -> bool:
        _, inline = parse_quantity(qty_node.get("token", ""))
        if inline and inline.lower() in _ENERGY_UNITS:
            return True

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

    def _energy_filter_qtys(self, nutrient_text: str, qty_nodes: List[dict],
                            row_edges: dict, adj_edges: dict,
                            all_quantities: List[dict], nut_cy: float,
                            max_cy_dist: float) -> List[dict]:
        """For energy nutrients, prefer kJ/kcal/cal quantities."""
        if not self._is_energy_nutrient(nutrient_text):
            return qty_nodes

        energy_qtys = [q for q in qty_nodes
                       if self._qty_has_energy_unit(q, row_edges, adj_edges)]
        if energy_qtys:
            return energy_qtys

        broad = []
        for q in all_quantities:
            if abs(q.get("cy", 0) - nut_cy) <= max_cy_dist:
                if self._qty_has_energy_unit(q, row_edges, adj_edges):
                    broad.append((_cy_dist({"cy": nut_cy}, q), q))

        if broad:
            broad.sort(key=lambda x: x[0])
            limit = self.config["max_quantities_per_nutrient"] * self.config["energy_max_multiplier"]
            out = [q for _, q in broad[:limit]]
            out.sort(key=lambda n: n.get("cx", 0))
            return out

        return qty_nodes

    # ── Candidate search: rank first ─────────────────────────────────

    def _sort_candidates(self, nut: dict, candidates: Iterable[dict],
                         config: dict) -> List[dict]:
        """Sort by rank-first score, then vertical closeness, then left-to-right."""
        unique = {}
        for q in candidates:
            unique[q["id"]] = q
        items = list(unique.values())
        items.sort(
            key=lambda q: (
                -_rank_score(nut, q),
                -compute_match_score_rank_first(nut, q, config),
                _cy_dist(nut, q),
                q.get("cx", 0),
            )
        )
        return items

    def _rank_first_candidates(self, nut: dict,
                               quantities: List[dict],
                               stream_qtys: Dict[int, List[dict]],
                               config: dict) -> List[dict]:
        """
        Primary matching method.

        If dosage streams exist, pick rank-aligned quantity candidates per stream.
        If streams do not exist, pick rank-aligned candidates globally.
        """
        nut_rank = _get_nutrient_rank(nut)
        if nut_rank < 0:
            return []

        max_per = self.config["max_quantities_per_nutrient"]
        selected: List[dict] = []

        # Stream-aware rank matching: one candidate per dosage stream.
        if stream_qtys:
            for sid in sorted(stream_qtys.keys()):
                qlist = stream_qtys[sid]

                exact = [q for q in qlist if _get_quantity_rank(q) == nut_rank]
                if exact:
                    selected.extend(self._sort_candidates(nut, exact, config)[:1])
                    continue

                if self.config["allow_near_rank_fallback"]:
                    near = [q for q in qlist
                            if _get_quantity_rank(q) >= 0 and
                            abs(_get_quantity_rank(q) - nut_rank) <= self.config["near_rank_max_diff"]]
                    if near:
                        selected.extend(self._sort_candidates(nut, near, config)[:1])

            # In normal nutrient rows, keep max_per; energy rows are filtered later.
            return self._sort_candidates(nut, selected, config)[:max_per]

        # No streams available: global rank-first matching.
        exact = [q for q in quantities if _get_quantity_rank(q) == nut_rank]
        if exact:
            return self._sort_candidates(nut, exact, config)[:max_per]

        if self.config["allow_near_rank_fallback"]:
            near = [q for q in quantities
                    if _get_quantity_rank(q) >= 0 and
                    abs(_get_quantity_rank(q) - nut_rank) <= self.config["near_rank_max_diff"]]
            if near:
                return self._sort_candidates(nut, near, config)[:max_per]

        return []

    def _has_row_confirmation(self, nut: dict, qty: dict, row_edges: dict) -> bool:
        """True when the candidate quantity is explicitly connected to the nutrient by ROW/SAME_ROW."""
        nut_id = nut.get("id")
        qty_id = qty.get("id")
        if nut_id is None or qty_id is None:
            return False

        for nb in row_edges.get(nut_id, []):
            if nb.get("id") == qty_id:
                return True

        # Backup for graphs that preserve row_id but where an explicit row edge is absent.
        # This is deliberately conservative: both IDs must exist and match.
        nut_row = nut.get("row_id", -1)
        qty_row = qty.get("row_id", -2)
        if isinstance(nut_row, (int, float)) and isinstance(qty_row, (int, float)):
            return int(nut_row) >= 0 and int(nut_row) == int(qty_row)

        return False

    def _has_stream_context_confirmation(self, nut: dict, qty: dict, ctx_in_edges: dict) -> bool:
        """True when quantity belongs to a detected dosage stream and has resolvable context."""
        sid = qty.get("dosage_stream_id", -1)
        has_stream = isinstance(sid, (int, float)) and int(sid) >= 0
        if not has_stream:
            return False

        context = self._resolve_context_for_qty(qty, nut, ctx_in_edges)
        return bool(str(context or "").strip())

    def _rank_primary_acceptance_reason(self, nut: dict, qty: dict,
                                        config: dict, row_edges: dict,
                                        ctx_in_edges: dict) -> Tuple[bool, str]:
        """
        Decide whether a rank-selected candidate is strong enough to accept
        as a primary rank-first match.

        Confirmation-based rule:
            accept rank only if:
              1. nutrient and quantity ranks match exactly,
              2. the quantity belongs to a DOSAGE column, and
              3. the match is independently confirmed by either:
                    a) ROW_COMPAT/SAME_ROW evidence, or
                    b) a valid dosage stream plus a resolvable context.

        If any part fails, the candidate is rejected and the engine tries
        row fallback, column fallback, then global score fallback.
        """
        if not config.get("rank_primary_filter_enabled", True):
            return True, "accepted_filter_disabled"

        rank = _rank_score(nut, qty)
        if rank < config.get("rank_primary_min_rank_score", 1.0):
            return False, "rank_not_exact"

        if config.get("rank_primary_require_dosage_role", True):
            role = str(qty.get("column_role") or "").upper().strip()
            if role != "DOSAGE":
                if not (config.get("rank_primary_allow_missing_role", False) and not role):
                    return False, "not_dosage_column"

        if not config.get("rank_primary_require_confirmation", True):
            return True, "accepted_no_confirmation_required"

        row_confirmed = (
            config.get("rank_primary_allow_row_confirmation", True)
            and self._has_row_confirmation(nut, qty, row_edges)
        )
        if row_confirmed:
            return True, "accepted_row_confirmed"

        stream_context_confirmed = (
            config.get("rank_primary_allow_stream_context_confirmation", True)
            and self._has_stream_context_confirmation(nut, qty, ctx_in_edges)
        )
        if stream_context_confirmed:
            return True, "accepted_stream_context_confirmed"

        return False, "no_row_or_stream_context_confirmation"

    def _filter_rank_primary_candidates(self, nut: dict, candidates: List[dict],
                                        config: dict, row_edges: dict,
                                        ctx_in_edges: dict) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
        """Keep only rank candidates that pass exact-rank + dosage-role + confirmation checks."""
        accepted = []
        reject_counts = defaultdict(int)
        accept_counts = defaultdict(int)
        for q in candidates:
            ok, reason = self._rank_primary_acceptance_reason(
                nut, q, config, row_edges, ctx_in_edges
            )
            if ok:
                accepted.append(q)
                accept_counts[reason] += 1
            else:
                reject_counts[reason] += 1
        return self._sort_candidates(nut, accepted, config), dict(reject_counts), dict(accept_counts)

    # ── Fallbacks after rank ────────────────────────────────────────

    def _row_fallback(self, nut: dict, row_edges: dict, config: dict) -> List[dict]:
        max_per = self.config["max_quantities_per_nutrient"]
        row_neighbors = row_edges.get(nut["id"], [])
        qtys = [n for n in row_neighbors if n.get("label") == "QUANTITY"]
        qtys.sort(key=lambda q: (_cy_dist(nut, q), q.get("cx", 0)))
        return qtys[:max_per]

    def _col_fallback(self, nut: dict, col_edges: dict) -> List[dict]:
        max_per = self.config["max_quantities_per_nutrient"]
        col_neighbors = col_edges.get(nut["id"], [])
        nut_height = max(nut.get("y2", 0) - nut.get("y1", 0), 20)

        tight = [n for n in col_neighbors
                 if n.get("label") == "QUANTITY"
                 and abs(n.get("cy", 0) - nut.get("cy", 0)) <= nut_height * self.config["col_tight_height_mult"]]
        if tight:
            tight.sort(key=lambda q: (_cy_dist(nut, q), q.get("cx", 0)))
            return tight[:max_per]

        wide = [n for n in col_neighbors
                if n.get("label") == "QUANTITY"
                and abs(n.get("cy", 0) - nut.get("cy", 0)) <= nut_height * self.config["col_wide_height_mult"]]
        wide.sort(key=lambda q: (_cy_dist(nut, q), q.get("cx", 0)))
        return wide[:max_per]

    def _global_score_fallback(self, nut: dict, quantities: List[dict],
                               config: dict) -> List[dict]:
        max_per = self.config["max_quantities_per_nutrient"]
        scored = []
        for q in quantities:
            score = compute_match_score_rank_first(nut, q, config)
            if score >= self.config["min_global_score"]:
                scored.append((score, q))
        scored.sort(key=lambda x: (-x[0], _cy_dist(nut, x[1]), x[1].get("cx", 0)))
        return [q for _, q in scored[:max_per]]

    # ── Unit/context resolution ─────────────────────────────────────

    def _find_unit(self, qty_node: dict, nut_node: dict,
                   row_edges: dict, adj_edges: dict) -> Optional[str]:
        qty_value, inline = parse_quantity(qty_node.get("token", ""))
        if inline:
            return inline

        unit_cands = [n for n in row_edges.get(qty_node["id"], [])
                      if n.get("label") == "UNIT"]
        if unit_cands:
            unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
            unit = _refine_unit_pick(unit_cands, qty_node)
            if unit:
                return unit

        unit_cands = [n for n in adj_edges.get(qty_node["id"], [])
                      if n.get("label") == "UNIT"]
        if unit_cands:
            unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
            unit = _refine_unit_pick(unit_cands, qty_node)
            if unit:
                return unit

        unit_cands = [n for n in row_edges.get(nut_node["id"], [])
                      if n.get("label") == "UNIT"]
        if unit_cands:
            unit_cands.sort(key=lambda n: abs(n.get("cy", 0) - qty_node.get("cy", 0)))
            unit = _refine_unit_pick(unit_cands, qty_node)
            if unit:
                return unit

        return extract_embedded_unit(nut_node.get("token", ""))

    def _context_from_graph(self, node: dict, ctx_in_edges: dict) -> Optional[str]:
        ancestors = ctx_in_edges.get(node.get("id"), [])
        if not ancestors:
            return None

        node_cx = node.get("cx", (node.get("x1", 0) + node.get("x2", 0)) / 2.0)
        candidates = []
        for ctx in ancestors:
            ctx_cx = ctx.get("cx", (ctx.get("x1", 0) + ctx.get("x2", 0)) / 2.0)
            text = ctx.get("norm") or ctx.get("column_context_id") or ctx.get("token")
            if text:
                candidates.append((abs(ctx_cx - node_cx), text))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _resolve_context_for_qty(self, qty_node: dict, nut_node: dict,
                                 ctx_in_edges: dict) -> Optional[str]:
        return (
            self._context_from_graph(qty_node, ctx_in_edges)
            or qty_node.get("column_context_id")
            or self._context_from_graph(nut_node, ctx_in_edges)
            or nut_node.get("column_context_id")
        )

    # ── Main extraction ─────────────────────────────────────────────

    def extract(self, graph: dict, image_id: str = "unknown") -> List[dict]:
        node_map = {n["id"]: n for n in graph.get("nodes", [])}
        edge_maps = self._build_edge_maps(graph, node_map)

        max_y = max((n.get("y2", 0) for n in graph.get("nodes", [])), default=500)
        scale = max(max_y / 500.0, 1.0)
        scaled_config = {**self.config, "max_perp_px": self.config["max_perp_px"] * scale}

        nutrients = [n for n in graph.get("nodes", []) if n.get("label") == "NUTRIENT"]
        quantities = [n for n in graph.get("nodes", []) if n.get("label") == "QUANTITY"]

        stream_qtys: Dict[int, List[dict]] = defaultdict(list)
        unstreamed_qtys: List[dict] = []
        for q in quantities:
            sid = q.get("dosage_stream_id", -1)
            if isinstance(sid, (int, float)) and int(sid) >= 0:
                stream_qtys[int(sid)].append(q)
            else:
                unstreamed_qtys.append(q)

        tuples: List[dict] = []
        seen_pairs = set()
        claimed_qty_ids = set()
        method_counts = defaultdict(int)

        for nut in nutrients:
            nutrient_text = normalize_nutrient(nut.get("token", ""))
            nut_cy = nut.get("cy", 0)
            nut_height = max(nut.get("y2", 0) - nut.get("y1", 0), 20)

            method = "rank_first"
            raw_rank_nodes = self._rank_first_candidates(
                nut, quantities, stream_qtys, scaled_config
            )
            qty_nodes, rank_rejects, rank_accepts = self._filter_rank_primary_candidates(
                nut, raw_rank_nodes, scaled_config, edge_maps["row"], edge_maps["ctx_in"]
            )
            for reason, count in rank_rejects.items():
                method_counts[f"rank_rejected_{reason}"] += count
            for reason, count in rank_accepts.items():
                method_counts[f"rank_{reason}"] += count

            # Confirmation-based rank-first behavior:
            # Rank proposes the first candidate, but rank is accepted only when
            # it is exact, from a DOSAGE column, and confirmed by either ROW
            # evidence or dosage-stream+context evidence. Otherwise, let the
            # row/column/global fallbacks try to recover the tuple.
            if raw_rank_nodes and not qty_nodes:
                method_counts["rank_candidates_rejected_then_fallback"] += 1

            if not qty_nodes:
                method = "row_fallback"
                qty_nodes = self._row_fallback(nut, edge_maps["row"], scaled_config)

            if not qty_nodes:
                method = "col_fallback"
                qty_nodes = self._col_fallback(nut, edge_maps["col"])

            if not qty_nodes:
                method = "global_score_fallback"
                qty_nodes = self._global_score_fallback(nut, quantities, scaled_config)

            # Energy correction after candidate selection.
            qty_nodes = self._energy_filter_qtys(
                nutrient_text,
                qty_nodes,
                edge_maps["row"],
                edge_maps["adj"],
                quantities,
                nut_cy,
                max_cy_dist=nut_height * 4.0,
            )

            # Keep deterministic order: rank-first score, then left-to-right.
            qty_nodes = self._sort_candidates(nut, qty_nodes, scaled_config)

            # Avoid quantity collisions except when the same nutrient legitimately
            # has multiple quantities from different dosage streams.
            filtered_qtys = []
            for q in qty_nodes:
                if q.get("id") in claimed_qty_ids:
                    continue
                filtered_qtys.append(q)
            qty_nodes = filtered_qtys

            if not qty_nodes:
                context = (
                    self._context_from_graph(nut, edge_maps["ctx_in"])
                    or nut.get("column_context_id")
                )
                pair_key = (nutrient_text, None, context)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    method_counts["no_quantity"] += 1
                    tuples.append({
                        "image_id": image_id,
                        "nutrient": nutrient_text,
                        "quantity": None,
                        "unit": None,
                        "context": context,
                        "_score": 0.0,
                        "_stream": -1,
                        "_method": "no_quantity",
                        "_rank_score": 0.0,
                    })
                continue

            max_out = self.config["max_quantities_per_nutrient"]
            if self._is_energy_nutrient(nutrient_text):
                max_out *= self.config["energy_max_multiplier"]
            qty_nodes = qty_nodes[:max_out]

            for qty_node in qty_nodes:
                qty_value, inline_unit = parse_quantity(qty_node.get("token", ""))
                unit = inline_unit or self._find_unit(qty_node, nut, edge_maps["row"], edge_maps["adj"])
                context = self._resolve_context_for_qty(qty_node, nut, edge_maps["ctx_in"])

                # Include context in the key, otherwise two same-valued dosage streams
                # can accidentally suppress each other.
                pair_key = (nutrient_text, qty_value, unit, context)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                claimed_qty_ids.add(qty_node.get("id"))

                score = compute_match_score_rank_first(nut, qty_node, scaled_config)
                method_counts[method] += 1
                tuples.append({
                    "image_id": image_id,
                    "nutrient": nutrient_text,
                    "quantity": qty_value,
                    "unit": unit,
                    "context": context,
                    "_score": score,
                    "_stream": qty_node.get("dosage_stream_id", -1),
                    "_method": method,
                    "_rank_score": _rank_score(nut, qty_node),
                })

        self.diagnostics = {
            "mode": "rank_first",
            "nutrients": len(nutrients),
            "quantities": len(quantities),
            "dosage_streams": len(stream_qtys),
            "tuples": len(tuples),
            "method_counts": dict(method_counts),
            "scale": round(scale, 2),
        }
        return tuples

    # ── Output helpers ──────────────────────────────────────────────

    def print_tuples(self, tuples: List[dict]) -> None:
        print(f"\n{'='*100}")
        print(f"EXTRACTED TUPLES — RANK FIRST ({len(tuples)} total)")
        print(f"{'='*100}")
        print(f"{'NUTRIENT':<30} {'QTY':<10} {'UNIT':<8} {'CONTEXT':<18} "
              f"{'SCORE':<7} {'RANK':<5} {'STREAM':<7} METHOD")
        print("-" * 100)
        for t in tuples:
            print(
                f"{str(t.get('nutrient'))[:29]:<30} "
                f"{str(t.get('quantity') or '')[:9]:<10} "
                f"{str(t.get('unit') or '')[:7]:<8} "
                f"{str(t.get('context') or '')[:17]:<18} "
                f"{float(t.get('_score', 0.0)):<7.3f} "
                f"{float(t.get('_rank_score', 0.0)):<5.1f} "
                f"{str(t.get('_stream', -1)):<7} "
                f"{t.get('_method', '')}"
            )
        print(f"{'='*100}\n")

    def print_diagnostics(self) -> None:
        print(f"\n{'='*60}")
        print("RANK-FIRST ASSOCIATION DIAGNOSTICS")
        print(f"{'='*60}")
        for k, v in self.diagnostics.items():
            print(f"  {k:<20}: {v}")
        print(f"{'='*60}\n")

    def save_csv(self, tuples: List[dict], output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_id", "nutrient", "quantity", "unit", "context"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tuples)
        print(f"Tuples saved: {output_path} ({len(tuples)} rows)")
