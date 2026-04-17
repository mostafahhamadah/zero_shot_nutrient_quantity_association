"""
run_experiment_paddle.py
======================
Experiment 2v1 — PaddleOCR Engine Swap
Zero-Shot Nutrient Extraction | Moustafa Hamada | THD + USB

GT SOURCE
---------
  Ground truth is loaded from data/annotations/*.json (one file per image).
  These JSON files are the single source of truth — edited via app.py.
  GT_CSV is no longer used.

SCIENTIFIC QUESTION
-------------------
How much of the Experiment 1 baseline failure is OCR-dependent?
Controlled variable: Stage 1 (OCR engine) + Stage 2 (corrector) only.
Stages 3-5 are byte-for-byte identical to Experiment 1.

CHANGES FROM run_experiment.py (v4 / Experiment 1)
---------------------------------------------------
  Stage 1 : src/ocr/paddleocr_runner.py       (was ocr_runner — EasyOCR)
  Stage 2 : src/utils/paddleocr_corrector.py  (was ocr_corrector)
  Stage 6 : LLMTupleEvaluator                 (was TupleEvaluator)
  Stages 3-5 : UNCHANGED

USAGE
-----
  python run_experiment_paddle.py
  python run_experiment_paddle.py --no-llm
  python run_experiment_paddle.py --baseline outputs/experiment_09_lexicon_expansion/evaluation_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from src.ocr.paddleocr_runner import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier
from src.graph.graph_constructor import GraphConstructor
from src.matching.experiment_01_final_association import TupleAssociator
from src.evaluation.llm_evaluator import LLMTupleEvaluator

# ── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "experiment_02v1_paddleocr"
ANNOTATIONS_DIR  = Path("data/annotations")          # ← JSON source of truth
IMAGE_DIR        = Path("data/raw")
BASELINE_JSON    = Path("outputs/experiment_09_lexicon_expansion/evaluation_results.json")
CONF_THRESHOLD   = 0.30
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("run_experiment_paddle")


# ── GT loader — JSON is the single source of truth ───────────────────────────

def load_gt_from_json() -> list:
    """
    Load ground-truth tuples from per-image JSON files in data/annotations/.
    Returns a flat list of dicts: {image_id, nutrient, quantity, unit, context}.
    Compatible with LLMTupleEvaluator(gt_rows=...).
    """
    rows = []
    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    if not json_files:
        logger.warning("No JSON annotation files found in %s", ANNOTATIONS_DIR)
        return rows
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping %s: %s", jf.name, e)
            continue
        raw_id   = Path(data.get("image_id", jf.stem + ".jpeg")).name
        image_id = Path(raw_id).stem + Path(raw_id).suffix.lower()
        for n in data.get("nutrients", []):
            nutrient = str(n.get("nutrient", "")).strip()
            quantity = str(n.get("quantity", "")).strip()
            unit     = str(n.get("unit",     "")).strip()
            context  = str(n.get("context",  "")).strip()
            if not any([nutrient, quantity, unit, context]):
                continue
            rows.append({
                "image_id": image_id,
                "nutrient": nutrient,
                "quantity": quantity,
                "unit":     unit,
                "context":  context,
            })
    logger.info("GT loaded from JSON: %d tuples across %d files",
                len(rows), len(json_files))
    return rows


# ── Stage runners ─────────────────────────────────────────────────────────────

def _run_stage1(image_path: Path) -> list:
    """Stage 1: PaddleOCR. Returns all tokens without confidence filtering."""
    return run_ocr_on_image(str(image_path))


def _run_stage2(tokens: list) -> tuple[list, dict]:
    """Stage 2: PaddleOCR corrector."""
    corrected, audit_log = correct_tokens(tokens, return_log=True)
    rules_fired: Counter = Counter()
    for entry in audit_log:
        for rule in entry["rules_fired"]:
            rules_fired[rule] += 1
    diag = {
        "splits":         len(corrected) - len(tokens),
        "rules_fired":    dict(rules_fired),
        "changed_tokens": len(audit_log),
    }
    return corrected, diag


def _run_stage3(tokens: list, classifier: SemanticClassifier) -> tuple[list, dict]:
    """Stage 3: Semantic classifier."""
    labeled      = classifier.classify_all(tokens)
    label_counts = Counter(t["label"] for t in labeled)
    clf_below    = sum(1 for t in labeled
                       if t["label"] == "NOISE" and t.get("conf", 1.0) < CONF_THRESHOLD)
    diag = {
        "total_tokens":    len(labeled),
        "below_threshold": clf_below,
        "NUTRIENT":        label_counts.get("NUTRIENT", 0),
        "QUANTITY":        label_counts.get("QUANTITY", 0),
        "UNIT":            label_counts.get("UNIT",     0),
        "CONTEXT":         label_counts.get("CONTEXT",  0),
        "NOISE":           label_counts.get("NOISE",    0),
        "UNKNOWN":         label_counts.get("UNKNOWN",  0),
    }
    return labeled, diag


def _run_stage4(labeled: list, graph_builder: GraphConstructor) -> tuple[dict, dict]:
    """Stage 4: Graph construction."""
    graph       = graph_builder.build(labeled)
    edge_counts = Counter(e["type"] for e in graph["edges"])
    diag = {
        "num_nodes":     graph["num_nodes"],
        "num_edges":     graph["num_edges"],
        "SAME_ROW":      edge_counts.get("SAME_ROW",      0),
        "SAME_COL":      edge_counts.get("SAME_COL",      0),
        "ADJACENT":      edge_counts.get("ADJACENT",      0),
        "CONTEXT_SCOPE": edge_counts.get("CONTEXT_SCOPE", 0),
    }
    return graph, diag


def _run_stage5(graph: dict, image_id: str,
                associator: TupleAssociator) -> tuple[list, dict]:
    """Stage 5: Tuple association."""
    tuples = associator.extract(graph, image_id=image_id)
    diag   = {
        "tuples_extracted": len(tuples),
        "assoc_no_ctx":     sum(1 for t in tuples if not t.get("context")),
        "assoc_no_qty":     sum(1 for t in tuples if not t.get("quantity")),
    }
    return tuples, diag


# ── Diagnostics helpers ───────────────────────────────────────────────────────

def _init_aggregate_diag() -> dict:
    return {
        "ocr_total_tokens":    0,
        "corrector_splits":    0,
        "corrector_changed":   0,
        "corrector_rules":     Counter(),
        "clf_total_tokens":    0,
        "clf_below_threshold": 0,
        "clf_NUTRIENT":        0,
        "clf_QUANTITY":        0,
        "clf_UNIT":            0,
        "clf_CONTEXT":         0,
        "clf_NOISE":           0,
        "clf_UNKNOWN":         0,
        "graph_nodes":         0,
        "graph_edges":         0,
        "edge_SAME_ROW":       0,
        "edge_SAME_COL":       0,
        "edge_ADJACENT":       0,
        "edge_CONTEXT_SCOPE":  0,
        "assoc_tuples":        0,
        "assoc_no_ctx":        0,
        "assoc_no_qty":        0,
        "zero_ocr_images":     [],
        "near_noise_images":   [],
    }


def _accumulate_diag(agg: dict, img_id: str,
                     d1: dict, d2: dict, d3: dict, d4: dict, d5: dict) -> None:
    agg["ocr_total_tokens"]    += d1.get("total_tokens", 0)
    agg["corrector_splits"]    += d2["splits"]
    agg["corrector_changed"]   += d2["changed_tokens"]
    agg["corrector_rules"].update(d2["rules_fired"])
    agg["clf_total_tokens"]    += d3["total_tokens"]
    agg["clf_below_threshold"] += d3["below_threshold"]
    for lbl in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE", "UNKNOWN"]:
        agg[f"clf_{lbl}"] += d3[lbl]
    agg["graph_nodes"] += d4["num_nodes"]
    agg["graph_edges"] += d4["num_edges"]
    for et in ["SAME_ROW", "SAME_COL", "ADJACENT", "CONTEXT_SCOPE"]:
        agg[f"edge_{et}"] += d4[et]
    agg["assoc_tuples"]  += d5["tuples_extracted"]
    agg["assoc_no_ctx"]  += d5["assoc_no_ctx"]
    agg["assoc_no_qty"]  += d5["assoc_no_qty"]
    if d1.get("total_tokens", 0) == 0:
        agg["zero_ocr_images"].append(img_id)
    clf_total = d3["total_tokens"]
    if clf_total > 0 and d3["NOISE"] / clf_total >= 0.90:
        agg["near_noise_images"].append(img_id)


def _print_diagnostics(agg: dict, n_images: int) -> None:
    W = 65
    clf_t = agg["clf_total_tokens"]
    pct   = lambda n: f"({n/clf_t*100:.1f}%)" if clf_t else ""
    print(f"\n{'='*W}\n  PIPELINE DIAGNOSTICS\n{'='*W}")
    print(f"  Images processed   : {n_images}")
    print(f"\n  ── Stage 1 (PaddleOCR) ──────────────────────────────")
    print(f"  OCR tokens total   : {agg['ocr_total_tokens']}")
    if agg["zero_ocr_images"]:
        print(f"  Zero-OCR images    : {agg['zero_ocr_images']}")
    print(f"\n  ── Stage 2 (Corrector) ──────────────────────────────")
    print(f"  Tokens changed     : {agg['corrector_changed']}")
    print(f"  Net token splits   : {agg['corrector_splits']}")
    if agg["corrector_rules"]:
        for rule, cnt in sorted(agg["corrector_rules"].items(), key=lambda x: -x[1]):
            print(f"    {rule:<30} {cnt:>5}")
    print(f"\n  ── Stage 3 (Classifier) ─────────────────────────────")
    print(f"  Total tokens       : {clf_t}")
    print(f"  Below threshold    : {agg['clf_below_threshold']}  {pct(agg['clf_below_threshold'])}")
    for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE", "UNKNOWN"]:
        n = agg[f"clf_{label}"]
        print(f"  {label:<10}         : {n:>5}  {pct(n)}")
    if agg["near_noise_images"]:
        print(f"  Near-100% NOISE    : {agg['near_noise_images']}")
    print(f"\n  ── Stage 4 (Graph) ──────────────────────────────────")
    print(f"  Nodes              : {agg['graph_nodes']}")
    print(f"  Edges              : {agg['graph_edges']}")
    for etype in ["SAME_ROW", "SAME_COL", "ADJACENT", "CONTEXT_SCOPE"]:
        print(f"  {etype:<16}     : {agg[f'edge_{etype}']:>5}")
    print(f"\n  ── Stage 5 (Association) ────────────────────────────")
    print(f"  Tuples extracted   : {agg['assoc_tuples']}")
    print(f"  No context         : {agg['assoc_no_ctx']}")
    print(f"  No quantity        : {agg['assoc_no_qty']}")
    print(f"{'='*W}\n")


def _save_diagnostics_csv(agg: dict, out_dir: Path) -> None:
    row = {
        "ocr_total_tokens":    agg["ocr_total_tokens"],
        "ocr_zero_images":     len(agg["zero_ocr_images"]),
        "corrector_splits":    agg["corrector_splits"],
        "corrector_changed":   agg["corrector_changed"],
        "clf_total_tokens":    agg["clf_total_tokens"],
        "clf_below_threshold": agg["clf_below_threshold"],
        "clf_NUTRIENT":        agg["clf_NUTRIENT"],
        "clf_QUANTITY":        agg["clf_QUANTITY"],
        "clf_UNIT":            agg["clf_UNIT"],
        "clf_CONTEXT":         agg["clf_CONTEXT"],
        "clf_NOISE":           agg["clf_NOISE"],
        "clf_UNKNOWN":         agg["clf_UNKNOWN"],
        "graph_nodes":         agg["graph_nodes"],
        "graph_edges":         agg["graph_edges"],
        "edge_SAME_ROW":       agg["edge_SAME_ROW"],
        "edge_SAME_COL":       agg["edge_SAME_COL"],
        "edge_ADJACENT":       agg["edge_ADJACENT"],
        "edge_CONTEXT_SCOPE":  agg["edge_CONTEXT_SCOPE"],
        "assoc_tuples":        agg["assoc_tuples"],
        "assoc_no_ctx":        agg["assoc_no_ctx"],
        "assoc_no_qty":        agg["assoc_no_qty"],
    }
    out_path = out_dir / "pipeline_diagnostics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info("Diagnostics saved: %s", out_path)


def _print_summary(current: dict, baseline_path: Path) -> None:
    METRIC_ROWS = [
        ("nutrient_f1",          "Nutrient F1"),
        ("quantity_acc",         "* Quantity Acc"),
        ("unit_acc",             "* Unit Acc"),
        ("context_acc",          "* Context Acc"),
        ("full3f_f1",            "* Full Tuple F1 (3-field)"),
        ("full_tuple_f1",        "* Full Tuple F1 (4-field)"),
        ("context_cost_pp",      "  Context cost 3f->4f (pp)"),
        ("full_tuple_precision", "  Full Tuple Prec (4f)"),
        ("full_tuple_recall",    "  Full Tuple Rec  (4f)"),
    ]
    baseline = None
    if baseline_path and baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
    W = 65
    if baseline:
        title  = f"SUMMARY vs. {baseline.get('experiment', baseline_path.stem)}"
        header = f"  {'METRIC':<26}  {'BASELINE':>10}  {'CURRENT':>10}  {'DELTA':>10}"
    else:
        title  = "EXPERIMENT SUMMARY"
        header = f"  {'METRIC':<26}  {'VALUE':>10}"
    print(f"\n{'='*W}\n  {title}\n{'='*W}\n{header}\n  {'-'*W}")
    for key, label in METRIC_ROWS:
        c_val = current.get(key, 0.0)
        if baseline:
            b_val = baseline.get(key, 0.0)
            print(f"  {label:<26}  {b_val:>10.4f}  {c_val:>10.4f}  {c_val-b_val:>+.4f}")
        else:
            print(f"  {label:<26}  {c_val:>10.4f}")
    print(f"{'='*W}\n")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_experiment(
    use_llm:         bool = True,
    baseline_path:   Path = BASELINE_JSON,
    experiment_name: str  = EXPERIMENT_NAME,
    model:           str  = "qwen2.5:7b",
) -> dict:
    out_dir    = Path("outputs") / experiment_name
    tuples_csv = out_dir / f"{experiment_name}_tuples.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in IMAGE_DIR.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        logger.error("No images found in %s", IMAGE_DIR)
        sys.exit(1)
    logger.info("Found %d images in %s", len(images), IMAGE_DIR)

    classifier    = SemanticClassifier(confidence_threshold=CONF_THRESHOLD)
    graph_builder = GraphConstructor()
    associator    = TupleAssociator()

    all_tuples: list = []
    agg_diag         = _init_aggregate_diag()
    t_start          = time.time()

    for img_path in images:
        image_id = img_path.stem + img_path.suffix.lower()
        logger.info("Processing: %s", image_id)
        try:
            raw_tokens            = _run_stage1(img_path)
            d1                    = {"total_tokens": len(raw_tokens)}
            corrected_tokens, d2  = _run_stage2(raw_tokens)
            labeled_tokens, d3    = _run_stage3(corrected_tokens, classifier)
            graph, d4             = _run_stage4(labeled_tokens, graph_builder)
            tuples, d5            = _run_stage5(graph, image_id, associator)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", image_id, exc, exc_info=True)
            d1 = {"total_tokens": 0}
            d2 = {"splits": 0, "changed_tokens": 0, "rules_fired": {}}
            d3 = {"total_tokens": 0, "below_threshold": 0,
                  "NUTRIENT": 0, "QUANTITY": 0, "UNIT": 0,
                  "CONTEXT": 0, "NOISE": 0, "UNKNOWN": 0}
            d4 = {"num_nodes": 0, "num_edges": 0, "SAME_ROW": 0, "SAME_COL": 0,
                  "ADJACENT": 0, "CONTEXT_SCOPE": 0}
            d5 = {"tuples_extracted": 0, "assoc_no_ctx": 0, "assoc_no_qty": 0}
            tuples = []

        all_tuples.extend(tuples)
        _accumulate_diag(agg_diag, image_id, d1, d2, d3, d4, d5)

    elapsed = round(time.time() - t_start, 1)
    logger.info("Pipeline complete: %d images, %d tuples, %.1fs",
                len(images), len(all_tuples), elapsed)

    fieldnames = ["image_id", "nutrient", "quantity", "unit", "context"]
    with open(tuples_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_tuples)
    logger.info("Tuples saved: %s  (%d rows)", tuples_csv, len(all_tuples))

    _print_diagnostics(agg_diag, len(images))
    _save_diagnostics_csv(agg_diag, out_dir)

    # ── Stage 6: GT loaded from JSON — single source of truth ─────────────────
    gt_rows   = load_gt_from_json()
    evaluator = LLMTupleEvaluator(gt_rows=gt_rows, use_llm=use_llm, model=model)
    metrics   = evaluator.run(
        predictions = all_tuples,
        experiment  = experiment_name,
        out_dir     = out_dir,
        notes       = (
            "PaddleOCR 3.0 / PP-OCRv5 engine swap. "
            "GT source: data/annotations/*.json (single source of truth). "
            "Stages 3-5 identical to experiment_09_lexicon_expansion. "
            f"LLM evaluator: {'enabled' if use_llm else 'disabled (fast run)'}."
        ),
        diagnostics = {k: v for k, v in agg_diag.items()
                       if not isinstance(v, (Counter, list))},
    )
    _print_summary(metrics, baseline_path)
    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR experiment — Experiment 2v1")
    parser.add_argument("--experiment", type=str,  default=EXPERIMENT_NAME)
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--model",      type=str,  default="qwen2.5:7b")
    parser.add_argument("--baseline",   type=Path, default=BASELINE_JSON)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        experiment_name = args.experiment,
        use_llm         = not args.no_llm,
        model           = args.model,
        baseline_path   = args.baseline,
    )