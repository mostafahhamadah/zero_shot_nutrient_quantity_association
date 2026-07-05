"""
merged_graph_associator.py
==========================
Sequential dual-mode extraction with DEFAULT-PREFERRED union.
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

WHAT IT DOES
------------
For ONE image's enriched tokens, this component runs the full
graph-build + association twice and merges the results:

    1. DEFAULT  (row_edge_mode="cy")        -> build graph -> associate -> default tuples
    2. STRUCTURAL (row_edge_mode="role_rank") -> build graph -> associate -> role_rank tuples
    3. MERGE: keep EVERY default tuple, then append ONLY the role_rank
              tuples whose (image_id, nutrient, quantity, unit, context)
              is not already present.

So the default (cy) result is never altered — role_rank can only ADD
tuples the default run missed. It cannot replace or remove a default tuple.

WHY A SEPARATE COMPONENT (not edge fusion)
------------------------------------------
This is LATE fusion at the tuple level: each mode gets its own clean graph
and its own association pass, then outputs are unioned. That is different
from row_edge_mode="role_rank_cy", which fuses the two row signals inside a
single graph before one association pass. Here the two passes never interact;
only their final tuples are merged.

DROP-IN USE (replaces Stage 4 + Stage 5 in the runner)
------------------------------------------------------
    from src.matching.merged_graph_associator import MergedModeAssociator

    merger = MergedModeAssociator()                 # once, before the image loop
    ...
    tuples     = merger.extract(enriched, image_id=image_key)
    graph      = merger.default_graph               # so existing edge-count logging still works
    assoc_diag = merger.associator.diagnostics      # diagnostics of the default pass

The runner's Stage 5b paragraph fallback can stay exactly as-is — it just
wraps the merged tuple list.

NOTE ON PRECISION
-----------------
Because role_rank's unique tuples are added unconditionally, any role_rank
tuple that disagrees with the default (e.g. a different quantity for the same
nutrient) is added as a NEW tuple and will be a false positive if wrong. The
union therefore tends to raise recall and lower precision — by design, since
you asked to add all unique role_rank tuples. The `source` field tags each
tuple ("default" / "role_rank") so you can measure exactly what role_rank
contributed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.graph.graph_constructor_v2 import GraphConstructorV2
from src.matching.association_v2    import TupleAssociatorV2

logger = logging.getLogger(__name__)


# Fields that define tuple identity for the dedup
FIELDS = ("image_id", "nutrient", "quantity", "unit", "context")


def _norm(s) -> str:
    """Case/whitespace-insensitive normalisation for the dedup key only.
    The original surface form is preserved in the merged tuples."""
    return " ".join(str(s if s is not None else "").strip().casefold().split())


def _key(t: dict) -> tuple:
    return tuple(_norm(t.get(k, "")) for k in FIELDS)


def merge_keep_default(default_tuples: List[dict],
                       rolerank_tuples: List[dict]) -> tuple:
    """
    Keep ALL default tuples; append only the role_rank tuples whose 5-field
    key is not already present among the default tuples.

    Returns (merged, n_added, n_consensus):
        merged       — default tuples (tagged source='default') followed by
                       role_rank's unique tuples (tagged source='role_rank')
        n_added      — how many role_rank tuples were uniquely added
        n_consensus  — how many role_rank tuples were dropped because the
                       default run already had them (both runs agreed)
    """
    merged: List[dict] = []
    seen = set()

    for t in default_tuples:
        tt = dict(t)
        tt["source"] = "default"
        merged.append(tt)
        seen.add(_key(tt))

    default_keys = set(seen)
    n_added = 0
    n_consensus = 0
    for t in rolerank_tuples:
        k = _key(t)
        if k in seen:
            if k in default_keys:
                n_consensus += 1
            continue
        tt = dict(t)
        tt["source"] = "role_rank"
        merged.append(tt)
        seen.add(k)
        n_added += 1

    return merged, n_added, n_consensus


class MergedModeAssociator:
    """
    Spans Stage 4 (graph) + Stage 5 (association) for BOTH the default (cy)
    and structural (role_rank) row modes, then merges tuples default-first.

    Parameters
    ----------
    base_config : dict, optional
        Extra GraphConstructorV2 config shared by both modes (thresholds, etc.).
        row_edge_mode is overridden per pass; do not set it here.
    associator_config : dict, optional
        Config forwarded to TupleAssociatorV2.
    col_edge_mode : str
        Column edge mode used by BOTH passes (default "cx", matching baseline).
        Ignored if base_config already sets "col_edge_mode".
    """

    DEFAULT_ROW_MODE = "cy"
    STRUCT_ROW_MODE  = "role_rank"

    def __init__(self,
                 base_config: dict = None,
                 associator_config: dict = None,
                 col_edge_mode: str = "cx"):
        base = dict(base_config or {})
        col  = base.get("col_edge_mode", col_edge_mode)

        self.gc_default = GraphConstructorV2(
            {**base, "row_edge_mode": self.DEFAULT_ROW_MODE, "col_edge_mode": col})
        self.gc_struct = GraphConstructorV2(
            {**base, "row_edge_mode": self.STRUCT_ROW_MODE, "col_edge_mode": col})

        # One associator is reused for both passes (its .diagnostics is
        # overwritten per call; we snapshot it after each extract).
        self.associator = TupleAssociatorV2(associator_config)

        self.diagnostics: dict = {}
        self.default_graph: Optional[dict] = None
        self.struct_graph:  Optional[dict] = None

    def extract(self, enriched_tokens: List[dict], image_id: str = "unknown") -> List[dict]:
        """
        Run default (cy) then structural (role_rank), merge default-first,
        and return the merged tuple list.
        """
        # ── Pass 1: DEFAULT (cy) ──────────────────────────────────────
        g_def = self.gc_default.build(enriched_tokens)
        t_def = self.associator.extract(g_def, image_id=image_id)
        diag_def = dict(self.associator.diagnostics or {})

        # ── Pass 2: STRUCTURAL (role_rank) ────────────────────────────
        g_str = self.gc_struct.build(enriched_tokens)
        t_str = self.associator.extract(g_str, image_id=image_id)
        diag_str = dict(self.associator.diagnostics or {})

        # ── Merge: keep default, add only role_rank uniques ───────────
        merged, n_added, n_consensus = merge_keep_default(t_def, t_str)

        self.default_graph = g_def
        self.struct_graph  = g_str
        self.diagnostics = {
            "default_tuples":      len(t_def),
            "rolerank_tuples":     len(t_str),
            "added_from_rolerank": n_added,
            "consensus":           n_consensus,
            "merged_tuples":       len(merged),
            "default_matches":     diag_def.get("matches", 0),
            "rolerank_matches":    diag_str.get("matches", 0),
        }
        logger.info(
            f"[merge] {image_id}: default={len(t_def)} role_rank={len(t_str)} "
            f"-> +{n_added} unique ({n_consensus} consensus) = {len(merged)}")
        return merged

    # convenience: edge counts of the default graph for runner logging
    @property
    def default_edge_counts(self) -> dict:
        from collections import Counter
        if not self.default_graph:
            return {}
        return dict(Counter(e["type"] for e in self.default_graph.get("edges", [])))