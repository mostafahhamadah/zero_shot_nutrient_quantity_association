"""
Analyze how well nutrient_rank ↔ qty_rank alignment works as a matching signal.

For each image, run the pipeline up to enrichment, then check:
  1. How many nutrients have a rank-matched quantity in each dosage column?
  2. When ROW_COMPAT gives multiple quantities, does rank pick the correct one?
  3. What's the rank consistency (same count of nutrients and quantities per column)?

This tells us if rank-based disambiguation is reliable enough to use.
"""
import sys, os, json
from collections import defaultdict, Counter
sys.path.insert(0, ".")

from src.ocr.paddleocr_runner import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier
from src.utils.token_enricher import TokenEnricher
try:
    from src.graph.graph_constructor_v2 import GraphConstructorV2 as GraphConstructor
except ImportError:
    from src.graph.graph_constructor import GraphConstructor
from src.matching.association_v2 import TupleAssociatorV2, parse_quantity

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data/raw"
ANN_DIR = "data/annotations"

images = sorted([f for f in os.listdir(DATA_DIR)
                 if f.lower().endswith(('.png', '.jpeg', '.jpg'))])

print(f"Analyzing {len(images)} images...\n")

total_stats = {
    "rank_consistent": 0,
    "rank_inconsistent": 0,
    "nutrients_with_rank": 0,
    "nutrients_rank_matched": 0,
    "nutrients_rank_unmatched": 0,
    "row_compat_multi_qty": 0,
    "row_compat_multi_rank_correct": 0,
    "row_compat_multi_rank_wrong": 0,
}

problem_images = []

for img_file in images:
    img_path = os.path.join(DATA_DIR, img_file)
    img_stem = os.path.splitext(img_file)[0]
    img_id = img_file.replace(".png", ".jpeg").replace(".jpg", ".jpeg")

    try:
        tokens = run_ocr_on_image(img_path)
        corrected = correct_tokens(tokens)
        classified = SemanticClassifier(confidence_threshold=0.30).classify_all(corrected)
        enriched = TokenEnricher().enrich(classified)
        graph = GraphConstructor().build(enriched)
        node_map = {n["id"]: n for n in graph["nodes"]}
    except Exception as e:
        print(f"  SKIP {img_file}: {e}")
        continue

    # Load gold
    gt_path = os.path.join(ANN_DIR, f"{img_stem}.json")
    if not os.path.exists(gt_path):
        continue
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_rows = gt_data if isinstance(gt_data, list) else gt_data.get("nutrients", [])

    nutrients = [n for n in graph["nodes"] if n.get("label") == "NUTRIENT"]
    quantities = [n for n in graph["nodes"] if n.get("label") == "QUANTITY"]

    # Check rank consistency per column
    enricher_diag = {}
    for n in graph["nodes"]:
        col = n.get("column_id", -1)
        role = n.get("column_role", "")
        if col >= 0:
            if col not in enricher_diag:
                enricher_diag[col] = {"role": role, "nut_ranks": 0, "qty_ranks": 0}
            if n.get("nutrient_rank_in_column", -1) >= 0:
                enricher_diag[col]["nut_ranks"] += 1
            if n.get("qty_rank_in_column", -1) >= 0:
                enricher_diag[col]["qty_ranks"] += 1

    # Find dosage columns
    nut_cols = [c for c, d in enricher_diag.items() if d["role"] == "NUTRIENT"]
    dos_cols = [c for c, d in enricher_diag.items() if d["role"] == "DOSAGE"]

    if nut_cols and dos_cols:
        nut_count = enricher_diag[nut_cols[0]]["nut_ranks"]
        consistent = True
        for dc in dos_cols:
            if enricher_diag[dc]["qty_ranks"] != nut_count:
                consistent = False
                break
        if consistent:
            total_stats["rank_consistent"] += 1
        else:
            total_stats["rank_inconsistent"] += 1

    # Check rank matching quality
    edge_types = set(e["type"] for e in graph["edges"])
    ROW_EDGE = "ROW_COMPAT" if "ROW_COMPAT" in edge_types else "SAME_ROW"

    img_multi = 0
    img_rank_ok = 0
    img_rank_bad = 0

    for nut in nutrients:
        nut_rank = nut.get("nutrient_rank_in_column", -1)
        if nut_rank < 0:
            continue
        total_stats["nutrients_with_rank"] += 1

        # Get ROW_COMPAT quantities
        row_qtys = [node_map[e["dst"]] for e in graph["edges"]
                    if e["src"] == nut["id"] and e["type"] == ROW_EDGE
                    and e["dst"] in node_map
                    and node_map[e["dst"]].get("label") == "QUANTITY"]

        if len(row_qtys) > 2:
            total_stats["row_compat_multi_qty"] += 1
            img_multi += 1

            # Check if any qty has matching rank
            rank_matched = [q for q in row_qtys
                            if q.get("qty_rank_in_column", -1) == nut_rank]

            # Check against gold: does rank-matched qty have the correct value?
            nut_text = nut.get("token", "").lower()
            gt_match = None
            for gt in gt_rows:
                gt_nut = gt.get("nutrient", "").lower()
                if nut_text[:10] in gt_nut or gt_nut[:10] in nut_text:
                    gt_match = gt
                    break

            if rank_matched and gt_match:
                rm_value, _ = parse_quantity(rank_matched[0].get("token", ""))
                gt_qty = str(gt_match.get("quantity", "")).replace(",", ".")
                if rm_value == gt_qty:
                    total_stats["row_compat_multi_rank_correct"] += 1
                    img_rank_ok += 1
                else:
                    total_stats["row_compat_multi_rank_wrong"] += 1
                    img_rank_bad += 1

        # General rank match check
        all_qtys_in_dosage = [q for q in quantities
                              if q.get("qty_rank_in_column", -1) == nut_rank
                              and q.get("column_role") == "DOSAGE"]
        if all_qtys_in_dosage:
            total_stats["nutrients_rank_matched"] += 1
        else:
            total_stats["nutrients_rank_unmatched"] += 1

    if img_multi > 0:
        problem_images.append((img_file, img_multi, img_rank_ok, img_rank_bad))

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RANK ALIGNMENT ANALYSIS")
print("=" * 70)

print(f"\nRank consistency (nut count == qty count per dosage col):")
print(f"  Consistent:   {total_stats['rank_consistent']}")
print(f"  Inconsistent: {total_stats['rank_inconsistent']}")

print(f"\nNutrients with rank assigned: {total_stats['nutrients_with_rank']}")
print(f"  Rank-matched qty exists:   {total_stats['nutrients_rank_matched']}")
print(f"  No rank match:             {total_stats['nutrients_rank_unmatched']}")

print(f"\nROW_COMPAT with >2 quantities (ambiguous — rank needed):")
print(f"  Total occurrences:  {total_stats['row_compat_multi_qty']}")
print(f"  Rank picks correct: {total_stats['row_compat_multi_rank_correct']}")
print(f"  Rank picks wrong:   {total_stats['row_compat_multi_rank_wrong']}")
if total_stats['row_compat_multi_rank_correct'] + total_stats['row_compat_multi_rank_wrong'] > 0:
    acc = total_stats['row_compat_multi_rank_correct'] / (
        total_stats['row_compat_multi_rank_correct'] + total_stats['row_compat_multi_rank_wrong'])
    print(f"  Rank accuracy:      {acc:.1%}")

print(f"\nImages with ambiguous ROW_COMPAT ({len(problem_images)}):")
print(f"  {'IMAGE':<18} {'MULTI':>6} {'RANK_OK':>8} {'RANK_BAD':>9}")
for img, multi, ok, bad in sorted(problem_images, key=lambda x: -x[1]):
    print(f"  {img:<18} {multi:>6} {ok:>8} {bad:>9}")