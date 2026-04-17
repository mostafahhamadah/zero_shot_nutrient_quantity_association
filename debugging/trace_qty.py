"""Trace wrong-qty associations on specific images to find root cause."""
import sys, os, csv, json
from collections import defaultdict
sys.path.insert(0, ".")

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "data/raw/34.png"
IMAGE_ID = os.path.basename(IMAGE).replace(".png", ".jpeg").replace(".jpg", ".jpeg")
img_stem = os.path.splitext(os.path.basename(IMAGE))[0]

print(f"Image: {IMAGE}  (id={IMAGE_ID})")
print("=" * 80)

# Run pipeline
from src.ocr.paddleocr_runner import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier
from src.utils.token_enricher import TokenEnricher
try:
    from src.graph.graph_constructor_v2 import GraphConstructorV2 as GraphConstructor
except ImportError:
    from src.graph.graph_constructor import GraphConstructor
from src.matching.association_v2 import TupleAssociatorV2

tokens = run_ocr_on_image(IMAGE)
corrected = correct_tokens(tokens)
classified = SemanticClassifier(confidence_threshold=0.30).classify_all(corrected)
enriched = TokenEnricher().enrich(classified)
graph = GraphConstructor().build(enriched)
node_map = {n["id"]: n for n in graph["nodes"]}

# ── 1. Show all QUANTITY nodes with their column assignment ──────
print("\n" + "=" * 80)
print("ALL QUANTITY NODES (sorted by cy, then cx)")
print("=" * 80)
qtys = sorted([n for n in graph["nodes"] if n.get("label") == "QUANTITY"],
              key=lambda n: (n.get("cy", 0), n.get("cx", 0)))
print(f"  {'TOKEN':<20} {'CX':>6} {'CY':>6} {'COL':>4} {'ROW':>4} {'COL_CTX':<18} {'STREAM':>6}")
print(f"  {'-'*75}")
for q in qtys:
    print(f"  {str(q.get('token',''))[:19]:<20} "
          f"{q.get('cx',0):6.0f} {q.get('cy',0):6.0f} "
          f"{q.get('column_id',-1):4d} {q.get('row_id',-1):4d} "
          f"{str(q.get('column_context_id','') or '—')[:17]:<18} "
          f"{q.get('dosage_stream_id',-1):6d}")

# ── 2. Show ROW_COMPAT edges from each NUTRIENT ─────────────────
print("\n" + "=" * 80)
print("NUTRIENT → QUANTITY edges (ROW_COMPAT)")
print("=" * 80)
nutrients = sorted([n for n in graph["nodes"] if n.get("label") == "NUTRIENT"],
                   key=lambda n: n.get("cy", 0))

edge_types = set(e["type"] for e in graph["edges"])
ROW_EDGE = "ROW_COMPAT" if "ROW_COMPAT" in edge_types else "SAME_ROW"

for nut in nutrients[:20]:  # limit to first 20
    row_qtys = []
    for e in graph["edges"]:
        if e["src"] == nut["id"] and e["type"] == ROW_EDGE:
            dst = node_map.get(e["dst"])
            if dst and dst.get("label") == "QUANTITY":
                row_qtys.append(dst)
    row_qtys.sort(key=lambda n: n.get("cx", 0))
    
    qty_str = "  |  ".join(
        f"{q.get('token','')[:12]} cx={q.get('cx',0):.0f} col={q.get('column_id',-1)}"
        for q in row_qtys
    ) if row_qtys else "(none)"
    
    print(f"  {nut.get('token','')[:35]:<37} cy={nut.get('cy',0):.0f} → {len(row_qtys)} qty: {qty_str}")

# ── 3. Show extracted tuples vs gold ─────────────────────────────
print("\n" + "=" * 80)
print("EXTRACTED TUPLES vs GOLD")
print("=" * 80)

assoc = TupleAssociatorV2()
tuples = assoc.extract(graph, image_id=IMAGE_ID)

# Load gold
gt_path = f"data/annotations/{img_stem}.json"
gt_rows = []
if os.path.exists(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_rows = gt_data if isinstance(gt_data, list) else gt_data.get("nutrients", gt_data.get("tuples", []))

print(f"\nPredicted: {len(tuples)}  |  Gold: {len(gt_rows)}")
print(f"\n  {'NUTRIENT':<30} {'P_QTY':<10} {'P_CTX':<16} {'G_QTY':<10} {'G_CTX':<16} {'MATCH'}")
print(f"  {'-'*95}")

# Simple matching for display
used_pred = set()
for gt in gt_rows:
    gt_nut = gt.get("nutrient", "")
    gt_qty = str(gt.get("quantity", ""))
    gt_ctx = gt.get("context", "")
    
    best_pred = None
    best_idx = -1
    for i, t in enumerate(tuples):
        if i in used_pred:
            continue
        pred_nut = str(t.get("nutrient", ""))
        if gt_nut.lower()[:15] in pred_nut.lower() or pred_nut.lower()[:15] in gt_nut.lower():
            best_pred = t
            best_idx = i
            break
    
    if best_pred:
        used_pred.add(best_idx)
        pred_qty = str(best_pred.get("quantity", ""))
        pred_ctx = str(best_pred.get("context", ""))
        qty_ok = "✓" if gt_qty.replace(",",".") == pred_qty.replace(",",".") else "✗"
        print(f"  {gt_nut[:29]:<30} {pred_qty[:9]:<10} {pred_ctx[:15]:<16} {gt_qty[:9]:<10} {gt_ctx[:15]:<16} {qty_ok}")
    else:
        print(f"  {gt_nut[:29]:<30} {'—':<10} {'—':<16} {gt_qty[:9]:<10} {gt_ctx[:15]:<16} MISS")