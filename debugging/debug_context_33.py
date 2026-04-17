"""
debug_context_33.py — Diagnose context per_100g↔per_serving flips on 33.png/33.jpeg

Run from project root:
    python debug_context_33.py

Prints:
  1. All CONTEXT tokens: token, norm, column_context_id, cx, column_id, row_id
  2. All HEADER_SCOPE / CONTEXT_SCOPE edges
  3. For each NUTRIENT+QUANTITY pair: which context was resolved, via which path
"""

import sys, os, json, glob
sys.path.insert(0, ".")

# ── Find the image ──────────────────────────────────────────────────
if len(sys.argv) > 1:
    IMAGE = sys.argv[1]
else:
    IMAGE_CANDIDATES = [
        "data/raw/33.jpeg", "data/raw/33.png", "data/raw/33.jpg",
        "data/images/33.jpeg", "data/images/33.png",
    ]
    IMAGE = None
    for p in IMAGE_CANDIDATES:
        if os.path.exists(p):
            IMAGE = p
            break
if not IMAGE or not os.path.exists(IMAGE):
    print(f"ERROR: Cannot find image. Usage: python debug_context_33.py <path>")
    sys.exit(1)

IMAGE_ID = os.path.basename(IMAGE).replace(".png", ".jpeg")
print(f"Image: {IMAGE}  (image_id={IMAGE_ID})")
print("=" * 80)

# ── Run pipeline stages 1-4 ────────────────────────────────────────
# Adjust imports to match your project structure
try:
    from src.ocr.paddleocr_runner import run_ocr_on_image
except ImportError:
    from paddleocr_runner import run_ocr_on_image

try:
    from src.utils.paddleocr_corrector import OCRCorrector
except ImportError:
    try:
        from paddleocr_corrector import OCRCorrector
    except ImportError:
        from src.utils.ocr_corrector import OCRCorrector

try:
    from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier
except ImportError:
    from experiment_01_final_semantic_classifier import SemanticClassifier

try:
    from src.utils.token_enricher import TokenEnricher
except ImportError:
    from token_enricher import TokenEnricher

try:
    from src.graph.graph_constructor_v2 import GraphConstructorV2 as GraphConstructor
except ImportError:
    try:
        from graph_constructor_v2 import GraphConstructorV2 as GraphConstructor
    except ImportError:
        from src.graph.graph_constructor import GraphConstructor

try:
    from src.matching.association_v2 import TupleAssociatorV2
except ImportError:
    from association_v2 import TupleAssociatorV2

print("Stage 1: OCR...")
tokens = run_ocr_on_image(IMAGE)
print(f"  → {len(tokens)} tokens")

print("Stage 2: Correction...")
corrector = OCRCorrector()
corrected = corrector.correct_all(tokens) if hasattr(corrector, 'correct_all') else corrector.correct_tokens(tokens)
print(f"  → {len(corrected)} tokens")

print("Stage 3: Classification...")
classifier = SemanticClassifier(confidence_threshold=0.30)
classified = classifier.classify_all(corrected)
print(f"  → {len(classified)} tokens")

print("Stage 3.5: Enrichment...")

# Quick dump: all tokens sorted by cx to see column structure
print("\n" + "=" * 80)
print("DIAGNOSTIC 0: ALL CLASSIFIED TOKENS (sorted by cx)")
print("=" * 80)
print(f"  {'TOKEN':<35} {'LABEL':<12} {'NORM':<18} {'CX':>6} {'CY':>6}")
print(f"  {'-'*80}")
for t in sorted(classified, key=lambda x: x.get("cx", 0)):
    print(f"  {str(t.get('token',''))[:34]:<35} "
          f"{str(t.get('label','')):<12} "
          f"{str(t.get('norm',''))[:17]:<18} "
          f"{t.get('cx',0):6.0f} "
          f"{t.get('cy',0):6.0f}")
print()

enricher = TokenEnricher()
enriched = enricher.enrich(classified)
enricher.print_diagnostics()

print("Stage 4: Graph construction...")
graph_builder = GraphConstructor()
graph = graph_builder.build(enriched)
print(f"  → {graph['num_nodes']} nodes, {graph['num_edges']} edges")

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: All CONTEXT tokens
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 1: CONTEXT TOKENS")
print("=" * 80)
ctx_nodes = [n for n in graph["nodes"] if n.get("label") == "CONTEXT"]
print(f"Found {len(ctx_nodes)} CONTEXT nodes:\n")
print(f"  {'TOKEN':<30} {'NORM':<18} {'COL_CTX_ID':<18} {'CX':>6} {'COL':>4} {'ROW':>4} {'HEADER':>7}")
print(f"  {'-'*105}")
for n in sorted(ctx_nodes, key=lambda x: x.get("cx", 0)):
    print(f"  {str(n.get('token',''))[:29]:<30} "
          f"{str(n.get('norm',''))[:17]:<18} "
          f"{str(n.get('column_context_id',''))[:17]:<18} "
          f"{n.get('cx',0):6.0f} "
          f"{n.get('column_id',-1):4d} "
          f"{n.get('row_id',-1):4d} "
          f"{str(n.get('is_header','')):>7}")

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: CONTEXT_SCOPE / HEADER_SCOPE edges
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 2: CONTEXT/HEADER SCOPE EDGES")
print("=" * 80)
node_map = {n["id"]: n for n in graph["nodes"]}
scope_edges = [e for e in graph["edges"]
               if e["type"] in ("CONTEXT_SCOPE", "HEADER_SCOPE")]
print(f"Found {len(scope_edges)} scope edges:\n")

# Group by source (context node)
from collections import defaultdict
by_src = defaultdict(list)
for e in scope_edges:
    by_src[e["src"]].append(e)

for src_id, edges in sorted(by_src.items()):
    src = node_map.get(src_id, {})
    print(f"  Context: '{src.get('token','')}' (cx={src.get('cx',0):.0f}, "
          f"col_ctx={src.get('column_context_id','')}, norm={src.get('norm','')}) "
          f"→ {len(edges)} targets")
    # Show first 5 targets
    for e in edges[:5]:
        dst = node_map.get(e["dst"], {})
        print(f"    → {dst.get('label','?'):<10} '{dst.get('token','')[:25]}' "
              f"(cx={dst.get('cx',0):.0f}, col_ctx={dst.get('column_context_id','')})")
    if len(edges) > 5:
        print(f"    ... and {len(edges)-5} more")

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Column structure
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 3: COLUMN STRUCTURE (cx ranges per column)")
print("=" * 80)
cols = defaultdict(list)
for n in graph["nodes"]:
    cid = n.get("column_id", -1)
    if cid >= 0:
        cols[cid].append(n)

for cid in sorted(cols.keys()):
    members = cols[cid]
    cxs = [m.get("cx", 0) for m in members]
    labels = [m.get("label", "?") for m in members]
    from collections import Counter
    lc = Counter(labels)
    ctx = members[0].get("column_context_id") or "—"
    role = members[0].get("column_role") or "?"
    print(f"  Col {cid}: cx=[{min(cxs):.0f}–{max(cxs):.0f}]  "
          f"role={role:<10} ctx={ctx:<18} "
          f"labels={dict(lc)}")

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Run association and trace context resolution
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 4: EXTRACTED TUPLES (context trace)")
print("=" * 80)

assoc = TupleAssociatorV2()
tuples = assoc.extract(graph, image_id=IMAGE_ID)
assoc.print_diagnostics()

print(f"\n  {'NUTRIENT':<35} {'QTY':<8} {'UNIT':<6} {'CONTEXT':<18} {'STREAM'}")
print(f"  {'-'*85}")
for t in tuples:
    print(f"  {str(t['nutrient'])[:34]:<35} "
          f"{str(t.get('quantity',''))[:7]:<8} "
          f"{str(t.get('unit',''))[:5]:<6} "
          f"{str(t.get('context',''))[:17]:<18} "
          f"{t.get('_stream', -1)}")

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: Compare to gold (JSON per-image annotations)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 5: CONTEXT MISMATCH vs GOLD")
print("=" * 80)

# Find JSON annotation for this image
img_stem = os.path.splitext(os.path.basename(IMAGE))[0]  # "33"
GT_CANDIDATES = [
    f"data/annotations/{img_stem}.json",
    f"data/annotations/{IMAGE_ID.replace('.jpeg','.json').replace('.png','.json')}",
]
gt_path = None
for p in GT_CANDIDATES:
    if os.path.exists(p):
        gt_path = p
        break

if gt_path:
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    # Support both list-of-dicts and {"tuples": [...]} formats
    gt_rows = gt_data if isinstance(gt_data, list) else gt_data.get("nutrients", gt_data.get("tuples", gt_data.get("annotations", [])))

    print(f"Gold file: {gt_path}  ({len(gt_rows)} tuples)")
    print(f"\nContext mismatches:")
    print(f"  {'GT_NUTRIENT':<30} {'GT_QTY':<8} {'GT_CTX':<18} {'PRED_CTX':<18} {'MATCH'}")
    print(f"  {'-'*95}")

    mismatches = 0
    for gt in gt_rows:
        gt_nut = str(gt.get("nutrient", ""))
        gt_qty = str(gt.get("quantity", ""))
        gt_ctx = str(gt.get("context", ""))

        # Find matching predicted tuple
        pred_ctx = "NOT_FOUND"
        for t in tuples:
            pred_nut = str(t.get("nutrient", ""))
            pred_qty = str(t.get("quantity", ""))
            if (gt_qty and pred_qty and
                gt_qty.replace(",",".") == pred_qty.replace(",",".") and
                (gt_nut.lower() in pred_nut.lower() or pred_nut.lower() in gt_nut.lower())):
                pred_ctx = str(t.get("context", ""))
                break

        match = "✓" if gt_ctx == pred_ctx else "✗"
        if gt_ctx != pred_ctx:
            mismatches += 1
            print(f"  {gt_nut[:29]:<30} {gt_qty:<8} {gt_ctx:<18} {pred_ctx:<18} {match}")

    print(f"\nTotal context mismatches: {mismatches} / {len(gt_rows)}")
    print("Done.")
else:
    print(f"Gold JSON not found for image '{img_stem}'.")
    print(f"Searched: {GT_CANDIDATES}")
    print("Listing data/annotations/:")
    ann_dir = "data/annotations"
    if os.path.isdir(ann_dir):
        files = sorted(os.listdir(ann_dir))[:20]
        for f in files:
            print(f"  {f}")
        if len(os.listdir(ann_dir)) > 20:
            print(f"  ... and {len(os.listdir(ann_dir))-20} more")
    else:
        print(f"  Directory not found: {ann_dir}")