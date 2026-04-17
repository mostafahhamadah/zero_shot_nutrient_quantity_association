"""
debug_energy_association.py
===========================
Diagnostic script for exp31 energy-aware quantity filtering.

Runs the full pipeline on specified images (default: energy-failing ones)
and dumps the graph neighborhood of every Energie/Energy/Brennwert node,
showing exactly which QUANTITY and UNIT nodes are reachable and why the
energy filter can't find kJ/kcal units.

Usage:
    python debug_energy_association.py
    python debug_energy_association.py --images 101.png 109.jpeg 116.jpeg
    python debug_energy_association.py --all
"""

from __future__ import annotations
import argparse
import json
import re
import logging
from pathlib import Path
from collections import defaultdict

from src.ocr.paddleocr_runner import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier
from src.utils.token_enricher import TokenEnricher
from src.graph.graph_constructor_v2 import GraphConstructorV2
from src.matching.association_v2 import TupleAssociatorV2, parse_quantity

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("debug_energy")

# ── Config ────────────────────────────────────────────────────────────

IMAGE_DIR = Path("data/raw")
ANNOTATIONS_DIR = Path("data/annotations")
OUTPUT_DIR = Path("outputs/_debug_energy")

# Images where Energie gets wrong qty (from exp30 analysis)
DEFAULT_IMAGES = [
    "101.png", "102.png", "103.jpeg", "104.jpeg", "105.png",
    "109.jpeg", "115.png", "116.jpeg",
]

_ENERGY_NAME_RE = re.compile(
    r'energie|energy|brennwert|énergie|energia|energy\s*value|'
    r'valor\s*energ|valeur\s*energ|valore\s*energ',
    re.IGNORECASE,
)
_ENERGY_UNITS = frozenset({'kj', 'kcal', 'cal', 'kj/kcal', 'kj/kca', 'kcal/kj'})


def is_energy_nutrient(text: str) -> bool:
    return bool(_ENERGY_NAME_RE.search(text))


def run_pipeline(image_path: Path) -> dict:
    """Run stages 1-4 and return the enriched graph."""
    tokens = run_ocr_on_image(str(image_path))
    tokens = correct_tokens(tokens)
    classifier = SemanticClassifier()
    tokens = classifier.classify_all(tokens)
    enricher = TokenEnricher()
    tokens = enricher.enrich(tokens)
    graph_builder = GraphConstructorV2()
    graph = graph_builder.build(tokens)
    return graph, tokens


def debug_image(image_id: str):
    """Run pipeline and dump energy node neighborhoods."""
    image_path = IMAGE_DIR / image_id
    if not image_path.exists():
        print(f"\n  ✗ Image not found: {image_path}")
        return

    print(f"\n{'='*80}")
    print(f"  DEBUG: {image_id}")
    print(f"{'='*80}")

    graph, tokens = run_pipeline(image_path)
    node_map = {n["id"]: n for n in graph["nodes"]}

    # Build edge indices
    row_edges = defaultdict(list)  # src → [dst_nodes]
    col_edges = defaultdict(list)
    adj_edges = defaultdict(list)
    ctx_edges = defaultdict(list)  # dst → [src_nodes] (incoming)

    for e in graph["edges"]:
        dst = node_map.get(e["dst"])
        src = node_map.get(e["src"])
        if not dst or not src:
            continue
        if e["type"] in ("ROW_COMPAT", "SAME_ROW"):
            row_edges[e["src"]].append(dst)
        elif e["type"] in ("COL_COMPAT", "SAME_COL"):
            col_edges[e["src"]].append(dst)
        elif e["type"] in ("DIRECTIONAL_ADJ", "ADJACENT"):
            adj_edges[e["src"]].append(dst)
        if e["type"] in ("CONTEXT_SCOPE", "HEADER_SCOPE"):
            ctx_edges[e["dst"]].append(src)

    # Find all NUTRIENT nodes
    nutrients = [n for n in graph["nodes"] if n.get("label") == "NUTRIENT"]
    quantities = [n for n in graph["nodes"] if n.get("label") == "QUANTITY"]
    units = [n for n in graph["nodes"] if n.get("label") == "UNIT"]

    # Summary
    print(f"\n  Nodes: {len(graph['nodes'])}  Edges: {len(graph['edges'])}")
    print(f"  NUTRIENT: {len(nutrients)}  QUANTITY: {len(quantities)}  UNIT: {len(units)}")

    # ── Dump ALL energy-unit tokens in the graph ──────────────────
    print(f"\n  ── All UNIT tokens with energy units ──")
    energy_unit_nodes = [u for u in units
                         if (u.get("norm") or "").strip(".,*()[]| ").lower() in _ENERGY_UNITS]
    if not energy_unit_nodes:
        print(f"  ✗ NO energy UNIT tokens found in graph!")
        print(f"  All UNIT norms: {[u.get('norm') for u in units]}")
    else:
        for u in energy_unit_nodes:
            print(f"    UNIT id={u['id']} token='{u.get('token','')}' "
                  f"norm='{u.get('norm','')}' cy={u.get('cy',0):.0f} cx={u.get('cx',0):.0f}")

    # ── Dump ALL quantity tokens that look like energy values ─────
    print(f"\n  ── Quantities with energy inline units ──")
    energy_inline_qtys = []
    for q in quantities:
        _, inline = parse_quantity(q.get("token", ""))
        if inline and inline.lower() in _ENERGY_UNITS:
            energy_inline_qtys.append((q, inline))
            print(f"    QTY id={q['id']} token='{q.get('token','')}' "
                  f"inline_unit={inline} cy={q.get('cy',0):.0f} cx={q.get('cx',0):.0f}")
    if not energy_inline_qtys:
        print(f"  ✗ No quantities with inline energy units")

    # ── For each energy NUTRIENT, dump its neighborhood ───────────
    energy_nutrients = [n for n in nutrients if is_energy_nutrient(n.get("token", ""))]
    if not energy_nutrients:
        print(f"\n  ✗ No energy nutrients found! All nutrients:")
        for n in nutrients:
            print(f"    '{n.get('token','')}'  norm='{n.get('norm','')}'")
        return

    for nut in energy_nutrients:
        nut_cy = nut.get("cy", 0)
        nut_cx = nut.get("cx", 0)
        print(f"\n  ── NUTRIENT: '{nut.get('token','')}' ──")
        print(f"     id={nut['id']}  cy={nut_cy:.0f}  cx={nut_cx:.0f}  "
              f"bbox=[{nut.get('x1',0):.0f},{nut.get('y1',0):.0f},"
              f"{nut.get('x2',0):.0f},{nut.get('y2',0):.0f}]")

        # ROW neighbors
        row_nbs = row_edges.get(nut["id"], [])
        print(f"\n     ROW_COMPAT neighbors ({len(row_nbs)}):")
        for nb in sorted(row_nbs, key=lambda n: n.get("cx", 0)):
            cy_dist = abs(nb.get("cy", 0) - nut_cy)
            print(f"       {nb.get('label','?'):10s} '{nb.get('token','')[:30]}'  "
                  f"norm='{(nb.get('norm','') or '')[:20]}'  "
                  f"cy={nb.get('cy',0):.0f} (Δcy={cy_dist:.0f})  cx={nb.get('cx',0):.0f}")

        # ROW QUANTITY neighbors specifically
        row_qtys = [n for n in row_nbs if n.get("label") == "QUANTITY"]
        print(f"\n     ROW QUANTITY nodes ({len(row_qtys)}):")
        for q in sorted(row_qtys, key=lambda n: n.get("cx", 0)):
            _, inline = parse_quantity(q.get("token", ""))
            # Check this qty's own ROW/ADJ for UNIT tokens
            q_row_units = [n for n in row_edges.get(q["id"], []) if n.get("label") == "UNIT"]
            q_adj_units = [n for n in adj_edges.get(q["id"], []) if n.get("label") == "UNIT"]
            unit_info = []
            for u in q_row_units:
                unit_info.append(f"ROW:{u.get('norm','')}")
            for u in q_adj_units:
                unit_info.append(f"ADJ:{u.get('norm','')}")

            has_energy = (inline and inline.lower() in _ENERGY_UNITS) or \
                         any((u.get("norm","").strip(".,*()[]| ").lower() in _ENERGY_UNITS)
                             for u in q_row_units + q_adj_units)

            marker = "⚡" if has_energy else "  "
            print(f"       {marker} token='{q.get('token','')}' → qty_value, inline_unit={inline}  "
                  f"cy={q.get('cy',0):.0f} (Δcy={abs(q.get('cy',0)-nut_cy):.0f})  "
                  f"units_reachable={unit_info or 'NONE'}")

        # COL neighbors (quantities only)
        col_nbs = col_edges.get(nut["id"], [])
        col_qtys = [n for n in col_nbs if n.get("label") == "QUANTITY"]
        if col_qtys:
            print(f"\n     COL QUANTITY nodes ({len(col_qtys)}):")
            for q in sorted(col_qtys, key=lambda n: abs(n.get("cy",0) - nut_cy))[:5]:
                _, inline = parse_quantity(q.get("token", ""))
                print(f"       token='{q.get('token','')}' inline={inline}  "
                      f"cy={q.get('cy',0):.0f} (Δcy={abs(q.get('cy',0)-nut_cy):.0f})")

        # Context ancestors
        ctx_anc = ctx_edges.get(nut["id"], [])
        ctx_anc = [c for c in ctx_anc if c.get("label") == "CONTEXT"]
        if ctx_anc:
            print(f"\n     CONTEXT ancestors ({len(ctx_anc)}):")
            for c in ctx_anc:
                print(f"       norm='{c.get('norm','')}' token='{c.get('token','')[:30]}'  "
                      f"cx={c.get('cx',0):.0f}")

        # ── Nearby quantities (within 4x height, any connection) ──
        nut_height = max(nut.get("y2", 0) - nut.get("y1", 0), 20)
        nearby = [(abs(q.get("cy",0) - nut_cy), q) for q in quantities
                  if abs(q.get("cy",0) - nut_cy) <= nut_height * 4.0]
        nearby.sort(key=lambda x: x[0])
        print(f"\n     ALL quantities within 4x height ({len(nearby)}):")
        for dist, q in nearby[:10]:
            _, inline = parse_quantity(q.get("token", ""))
            connected = q["id"] in {n["id"] for n in row_qtys}
            q_row_units = [n for n in row_edges.get(q["id"], []) if n.get("label") == "UNIT"]
            q_adj_units = [n for n in adj_edges.get(q["id"], []) if n.get("label") == "UNIT"]
            all_units = [(u.get("norm","")) for u in q_row_units + q_adj_units]

            has_energy = (inline and inline.lower() in _ENERGY_UNITS) or \
                         any((u.get("norm","").strip(".,*()[]| ").lower() in _ENERGY_UNITS)
                             for u in q_row_units + q_adj_units)

            marker = "⚡" if has_energy else "  "
            conn = "CONNECTED" if connected else "not connected"
            print(f"       {marker} '{q.get('token','')}'  Δcy={dist:.0f}  inline={inline}  "
                  f"units={all_units or 'NONE'}  {conn}")

    # ── Save graph JSON for inspection ────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{image_id.replace('.', '_')}_graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"\n  Graph saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug energy quantity association")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Image filenames to debug")
    parser.add_argument("--all", action="store_true",
                        help="Run on all images in data/raw/")
    args = parser.parse_args()

    if args.all:
        images = sorted(p.name for p in IMAGE_DIR.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    elif args.images:
        images = args.images
    else:
        images = DEFAULT_IMAGES

    print(f"\nDebugging {len(images)} images: {images}\n")

    for img in images:
        try:
            debug_image(img)
        except Exception as e:
            print(f"\n  ✗ ERROR on {img}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"  Done. Graph JSONs saved to {OUTPUT_DIR}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()