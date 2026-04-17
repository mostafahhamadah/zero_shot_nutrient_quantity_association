"""
run_experiment_gemma.py
=======================
Experiment 5 — Gemma Semantic Classifier
Zero-Shot Nutrient Extraction | Moustafa Hamada | THD + USB

GT SOURCE
---------
  Ground truth is loaded from data/annotations/*.json (one file per image).
  These JSON files are the single source of truth — edited via app.py.
  GT_CSV is no longer used.

SCIENTIFIC QUESTION
-------------------
Does Google's Gemma model produce better zero-shot semantic classification
of supplement label tokens compared to Qwen (Experiment 4) and the
rule-based classifier (Experiment 2)?

Controlled variable: Stage 3 (classifier) only.
Stages 1, 2, 4, 5 are byte-for-byte identical to Experiment 2v1.

CHANGES FROM run_experiment_qwen.py (Experiment 4)
---------------------------------------------------
  Stage 3 : src/classification/gemma_classifier.py   (was qwen_classifier)
            Model     : gemma3:4b via Ollama local inference
            Strategy  : Option B — full serialized text per image (57 calls)
            Schema    : Schema A — entity text + label only, fuzzy remapping
            Threshold : 0.65 fuzzy match (SequenceMatcher)
            Labels    : NUTRIENT, QUANTITY, UNIT, CONTEXT, NOISE (5-label)
            Fallback  : UNKNOWN → unmatched tokens
  Stages 1, 2, 4, 5 : UNCHANGED from Experiment 2v1

USAGE
-----
  python run_experiment_gemma.py
  python run_experiment_gemma.py --no-llm
  python run_experiment_gemma.py --gemma-model gemma3:12b
  python run_experiment_gemma.py --baseline outputs/experiment_04_qwen_classifier/evaluation_results.json
  python run_experiment_gemma.py --experiment experiment_05_gemma3_4b_classifier
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

from src.ocr.paddleocr_runner                     import run_ocr_on_image
from src.utils.paddleocr_corrector                import correct_tokens
from src.classification.gemma_classifier          import GemmaClassifier
from src.graph.graph_constructor                  import GraphConstructor
from src.matching.experiment_01_final_association import TupleAssociator
from src.evaluation.llm_evaluator                 import LLMTupleEvaluator

# ── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "experiment_05_gemma_classifier"
ANNOTATIONS_DIR  = Path("data/annotations")
IMAGE_DIR        = Path("data/raw")
BASELINE_JSON    = Path("outputs/experiment_04_qwen_classifier/evaluation_results.json")
GEMMA_MODEL      = "gemma4:e4b"    # Stage 3 classification — light model
EVAL_MODEL       = "qwen2.5:7b"   # Stage 6 evaluation — stronger model for judgment
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("run_experiment_gemma")


# ── GT loader — JSON is the single source of truth ───────────────────────────

def load_gt_from_json() -> list:
    """
    Load ground-truth tuples from per-image JSON files in data/annotations/.
    Returns a flat list of dicts: {image_id, nutrient, quantity, unit, context,
    serving_size}.
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
        image_id = Path(data.get("image_id", jf.stem + ".jpeg")).name
        for n in data.get("nutrients", []):
            nutrient     = str(n.get("nutrient",     "")).strip()
            quantity     = str(n.get("quantity",     "")).strip()
            unit         = str(n.get("unit",         "")).strip()
            context      = str(n.get("context",      "")).strip()
            serving_size = str(n.get("serving_size", "") or "").strip()
            if not any([nutrient, quantity, unit, context]):
                continue
            rows.append({
                "image_id":    image_id,
                "nutrient":    nutrient,
                "quantity":    quantity,
                "unit":        unit,
                "context":     context,
                "serving_size": serving_size,
            })
    logger.info("GT loaded from JSON: %d tuples across %d files",
                len(rows), len(json_files))
    return rows


# ── Stage runners ─────────────────────────────────────────────────────────────

def _run_stage1(image_path: Path) -> list:
    """Stage 1: PaddleOCR — unchanged from Experiment 2v1."""
    return run_ocr_on_image(str(image_path))


def _run_stage2(tokens: list) -> tuple[list, dict]:
    """Stage 2: PaddleOCR corrector — unchanged from Experiment 2v1."""
    corrected, audit_log = correct_tokens(tokens, return_log=True)
    rules_fired: Counter = Counter()
    for entry in audit_log:
        for rule in entry["rules_fired"]:
            rules_fired[rule] += 1
    return corrected, {
        "splits":         len(corrected) - len(tokens),
        "rules_fired":    dict(rules_fired),
        "changed_tokens": len(audit_log),
    }


def _run_stage3_gemma(tokens: list, classifier: GemmaClassifier) -> tuple[list, dict]:
    """Stage 3: Gemma semantic classifier."""
    labelled     = classifier.classify(tokens)
    label_counts = Counter(t["label"] for t in labelled)

    context_tokens        = [t for t in labelled if t["label"] == "CONTEXT"]
    context_norm_resolved = sum(1 for t in context_tokens if t.get("norm") is not None)
    context_norm_missing  = len(context_tokens) - context_norm_resolved

    matched_scores   = [t["gemma_score"] for t in labelled if t.get("gemma_score") is not None]
    matched_entities = {t.get("gemma_entity") for t in labelled if t.get("gemma_entity") is not None}

    return labelled, {
        "total_tokens":          len(labelled),
        "NUTRIENT":              label_counts.get("NUTRIENT", 0),
        "QUANTITY":              label_counts.get("QUANTITY", 0),
        "UNIT":                  label_counts.get("UNIT",     0),
        "CONTEXT":               label_counts.get("CONTEXT",  0),
        "UNKNOWN":               label_counts.get("UNKNOWN",  0),
        "context_norm_resolved": context_norm_resolved,
        "context_norm_missing":  context_norm_missing,
        "gemma_matched":         len(matched_entities),
        "gemma_avg_score":       (round(sum(matched_scores) / len(matched_scores), 4)
                                  if matched_scores else 0.0),
    }


def _run_stage4(labeled: list, graph_builder: GraphConstructor) -> tuple[dict, dict]:
    """Stage 4: Graph construction — unchanged from Experiment 2v1."""
    graph       = graph_builder.build(labeled)
    edge_counts = Counter(e["type"] for e in graph["edges"])
    return graph, {
        "num_nodes":     graph["num_nodes"],
        "num_edges":     graph["num_edges"],
        "SAME_ROW":      edge_counts.get("SAME_ROW",      0),
        "SAME_COL":      edge_counts.get("SAME_COL",      0),
        "ADJACENT":      edge_counts.get("ADJACENT",      0),
        "CONTEXT_SCOPE": edge_counts.get("CONTEXT_SCOPE", 0),
    }


def _run_stage5(graph: dict, image_id: str,
                associator: TupleAssociator) -> tuple[list, dict]:
    """Stage 5: Tuple association — unchanged from Experiment 2v1."""
    tuples = associator.extract(graph, image_id=image_id)
    return tuples, {
        "tuples_extracted": len(tuples),
        "assoc_no_ctx":     sum(1 for t in tuples if not t.get("context")),
        "assoc_no_qty":     sum(1 for t in tuples if not t.get("quantity")),
    }


# ── Diagnostics helpers ───────────────────────────────────────────────────────

def _init_aggregate_diag() -> dict:
    return {
        "ocr_total_tokens":      0,
        "corrector_splits":      0,
        "corrector_changed":     0,
        "corrector_rules":       Counter(),
        "clf_total_tokens":      0,
        "clf_NUTRIENT":          0,
        "clf_QUANTITY":          0,
        "clf_UNIT":              0,
        "clf_CONTEXT":           0,
        "clf_UNKNOWN":           0,
        "context_norm_resolved": 0,
        "context_norm_missing":  0,
        "gemma_calls":           0,
        "gemma_matched":         0,
        "gemma_score_sum":       0.0,
        "gemma_score_n":         0,
        "graph_nodes":           0,
        "graph_edges":           0,
        "edge_SAME_ROW":         0,
        "edge_SAME_COL":         0,
        "edge_ADJACENT":         0,
        "edge_CONTEXT_SCOPE":    0,
        "assoc_tuples":          0,
        "assoc_no_ctx":          0,
        "assoc_no_qty":          0,
        "zero_ocr_images":       [],
        "near_noise_images":     [],
        "zero_nutrient_images":  [],
        "gemma_failed_images":   [],
    }


def _accumulate_diag(agg: dict, img_id: str,
                     d1: dict, d2: dict, d3: dict, d4: dict, d5: dict,
                     gemma_ok: bool) -> None:
    agg["ocr_total_tokens"]      += d1.get("total_tokens", 0)
    agg["corrector_splits"]      += d2["splits"]
    agg["corrector_changed"]     += d2["changed_tokens"]
    agg["corrector_rules"].update(d2["rules_fired"])
    agg["clf_total_tokens"]      += d3["total_tokens"]
    for lbl in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "UNKNOWN"]:
        agg[f"clf_{lbl}"] += d3[lbl]
    agg["context_norm_resolved"] += d3["context_norm_resolved"]
    agg["context_norm_missing"]  += d3["context_norm_missing"]
    agg["gemma_calls"]           += 1
    agg["gemma_matched"]         += d3["gemma_matched"]
    if d3["gemma_avg_score"] > 0:
        agg["gemma_score_sum"] += d3["gemma_avg_score"] * d3["gemma_matched"]
        agg["gemma_score_n"]   += d3["gemma_matched"]
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
    if clf_total > 0 and d3["UNKNOWN"] / clf_total >= 0.90:
        agg["near_noise_images"].append(img_id)
    if d3["NUTRIENT"] == 0:
        agg["zero_nutrient_images"].append(img_id)
    if not gemma_ok:
        agg["gemma_failed_images"].append(img_id)


def _print_diagnostics(agg: dict, n_images: int) -> None:
    W     = 65
    clf_t = agg["clf_total_tokens"]
    pct   = lambda n: f"({n / clf_t * 100:.1f}%)" if clf_t else ""
    g_avg = (round(agg["gemma_score_sum"] / agg["gemma_score_n"], 4)
             if agg["gemma_score_n"] > 0 else 0.0)
    ctx_t = agg["clf_CONTEXT"]

    print(f"\n{'='*W}")
    print("  PIPELINE DIAGNOSTICS — Experiment 5 (Gemma)")
    print(f"{'='*W}")
    print(f"  Images processed        : {n_images}")
    print()
    print(f"  ── Stage 1 (PaddleOCR) ──────────────────────────────")
    print(f"  OCR tokens total        : {agg['ocr_total_tokens']}")
    if agg["zero_ocr_images"]:
        print(f"  Zero-OCR images         : {agg['zero_ocr_images']}")
    print()
    print(f"  ── Stage 2 (Corrector) ──────────────────────────────")
    print(f"  Tokens changed          : {agg['corrector_changed']}")
    print(f"  Net token splits        : {agg['corrector_splits']}")
    if agg["corrector_rules"]:
        for rule, cnt in sorted(agg["corrector_rules"].items(), key=lambda x: -x[1]):
            print(f"    {rule:<30} {cnt:>5}")
    print()
    print(f"  ── Stage 3 (Gemma) ──────────────────────────────────")
    print(f"  Total tokens            : {clf_t}")
    print(f"  Ollama calls made       : {agg['gemma_calls']}")
    print(f"  Entities matched        : {agg['gemma_matched']}")
    print(f"  Mean match score        : {g_avg:.4f}")
    if agg["gemma_failed_images"]:
        print(f"  Failed Gemma calls      : {agg['gemma_failed_images']}")
    print()
    for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "UNKNOWN"]:
        n = agg[f"clf_{label}"]
        print(f"  {label:<10}              : {n:>5}  {pct(n)}")
    print()
    print(f"  ── CONTEXT quality ──────────────────────────────────")
    print(f"  CONTEXT tokens total    : {ctx_t}")
    print(f"  Norm resolved           : {agg['context_norm_resolved']}"
          + (f"  ({agg['context_norm_resolved']/ctx_t*100:.1f}%)" if ctx_t else ""))
    print(f"  Norm MISSING            : {agg['context_norm_missing']}"
          + (f"  ({agg['context_norm_missing']/ctx_t*100:.1f}%)" if ctx_t else ""))
    if agg["near_noise_images"]:
        print(f"  >=90% UNKNOWN images    : {agg['near_noise_images']}")
    if agg["zero_nutrient_images"]:
        print(f"  Zero-NUTRIENT images    : {agg['zero_nutrient_images']}")
    print()
    print(f"  ── Stage 4 (Graph) ──────────────────────────────────")
    print(f"  Nodes                   : {agg['graph_nodes']}")
    print(f"  Edges                   : {agg['graph_edges']}")
    for etype in ["SAME_ROW", "SAME_COL", "ADJACENT", "CONTEXT_SCOPE"]:
        print(f"  {etype:<16}          : {agg[f'edge_{etype}']:>5}")
    print()
    print(f"  ── Stage 5 (Association) ────────────────────────────")
    print(f"  Tuples extracted        : {agg['assoc_tuples']}")
    print(f"  No context              : {agg['assoc_no_ctx']}")
    print(f"  No quantity             : {agg['assoc_no_qty']}")
    print(f"{'='*W}\n")


def _save_diagnostics_csv(agg: dict, out_dir: Path) -> None:
    g_avg = (round(agg["gemma_score_sum"] / agg["gemma_score_n"], 4)
             if agg["gemma_score_n"] > 0 else 0.0)
    row = {
        "ocr_total_tokens":      agg["ocr_total_tokens"],
        "ocr_zero_images":       len(agg["zero_ocr_images"]),
        "corrector_splits":      agg["corrector_splits"],
        "corrector_changed":     agg["corrector_changed"],
        "clf_total_tokens":      agg["clf_total_tokens"],
        "clf_NUTRIENT":          agg["clf_NUTRIENT"],
        "clf_QUANTITY":          agg["clf_QUANTITY"],
        "clf_UNIT":              agg["clf_UNIT"],
        "clf_CONTEXT":           agg["clf_CONTEXT"],
        "clf_UNKNOWN":           agg["clf_UNKNOWN"],
        "context_norm_resolved": agg["context_norm_resolved"],
        "context_norm_missing":  agg["context_norm_missing"],
        "gemma_calls":           agg["gemma_calls"],
        "gemma_matched":         agg["gemma_matched"],
        "gemma_avg_score":       g_avg,
        "gemma_failed_images":   len(agg["gemma_failed_images"]),
        "graph_nodes":           agg["graph_nodes"],
        "graph_edges":           agg["graph_edges"],
        "edge_SAME_ROW":         agg["edge_SAME_ROW"],
        "edge_SAME_COL":         agg["edge_SAME_COL"],
        "edge_ADJACENT":         agg["edge_ADJACENT"],
        "edge_CONTEXT_SCOPE":    agg["edge_CONTEXT_SCOPE"],
        "assoc_tuples":          agg["assoc_tuples"],
        "assoc_no_ctx":          agg["assoc_no_ctx"],
        "assoc_no_qty":          agg["assoc_no_qty"],
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
        title  = "EXPERIMENT 5 SUMMARY (Gemma)"
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
    eval_model:      str  = EVAL_MODEL,
    gemma_model:     str  = GEMMA_MODEL,
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

    logger.info("Initialising Gemma classifier: model=%s", gemma_model)
    classifier    = GemmaClassifier(model_id=gemma_model)
    graph_builder = GraphConstructor()
    associator    = TupleAssociator()

    logger.info("Warming up Gemma (loading model into memory)...")
    import requests as _req
    try:
        _req.post("http://localhost:11434/api/generate",
                  json={"model": gemma_model, "prompt": "say ready", "stream": False},
                  timeout=300)
        logger.info("Gemma warmup complete.")
    except Exception:
        logger.warning("Gemma warmup failed — model may be slow on first image.")

    all_tuples: list = []
    agg_diag         = _init_aggregate_diag()
    t_start          = time.time()

    for img_path in images:
        image_id = img_path.name
        logger.info("Processing: %s", image_id)
        gemma_ok = True
        try:
            raw_tokens            = _run_stage1(img_path)
            d1                    = {"total_tokens": len(raw_tokens)}
            corrected_tokens, d2  = _run_stage2(raw_tokens)
            labeled_tokens, d3    = _run_stage3_gemma(corrected_tokens, classifier)
            if d3["NUTRIENT"] == 0 and d3["QUANTITY"] == 0 and d3["UNIT"] == 0:
                gemma_ok = False
                logger.warning("Gemma returned no entities for %s — all UNKNOWN.", image_id)
            graph, d4  = _run_stage4(labeled_tokens, graph_builder)
            tuples, d5 = _run_stage5(graph, image_id, associator)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", image_id, exc, exc_info=True)
            gemma_ok = False
            d1 = {"total_tokens": 0}
            d2 = {"splits": 0, "changed_tokens": 0, "rules_fired": {}}
            d3 = {"total_tokens": 0, "NUTRIENT": 0, "QUANTITY": 0, "UNIT": 0,
                  "CONTEXT": 0, "UNKNOWN": 0, "context_norm_resolved": 0,
                  "context_norm_missing": 0, "gemma_matched": 0, "gemma_avg_score": 0.0}
            d4 = {"num_nodes": 0, "num_edges": 0, "SAME_ROW": 0, "SAME_COL": 0,
                  "ADJACENT": 0, "CONTEXT_SCOPE": 0}
            d5 = {"tuples_extracted": 0, "assoc_no_ctx": 0, "assoc_no_qty": 0}
            tuples = []

        all_tuples.extend(tuples)
        _accumulate_diag(agg_diag, image_id, d1, d2, d3, d4, d5, gemma_ok)

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
    evaluator = LLMTupleEvaluator(gt_rows=gt_rows, use_llm=use_llm, model=eval_model)
    metrics   = evaluator.run(
        predictions = all_tuples,
        experiment  = experiment_name,
        out_dir     = out_dir,
        notes       = (
            f"Gemma {gemma_model} Stage 3 classifier. "
            f"GT source: data/annotations/*.json (single source of truth). "
            f"Strategy: Option B (full serialized text per image). "
            f"Schema: A (entity text + label, fuzzy remapping threshold=0.65). "
            f"Labels: NUTRIENT/QUANTITY/UNIT/CONTEXT/NOISE (5-label). "
            f"Fallback: UNKNOWN. "
            f"Stages 1,2,4,5 identical to experiment_02v1_paddleocr. "
            f"LLM evaluator: {'enabled' if use_llm else 'disabled'}."
        ),
        diagnostics = {k: v for k, v in agg_diag.items()
                       if not isinstance(v, (Counter, list))},
    )
    _print_summary(metrics, baseline_path)
    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma classifier experiment — Experiment 5")
    parser.add_argument("--experiment",   type=str,  default=EXPERIMENT_NAME)
    parser.add_argument("--no-llm",       action="store_true")
    parser.add_argument("--eval-model",   type=str,  default=EVAL_MODEL)
    parser.add_argument("--gemma-model",  type=str,  default=GEMMA_MODEL)
    parser.add_argument("--baseline",     type=Path, default=BASELINE_JSON)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        experiment_name = args.experiment,
        use_llm         = not args.no_llm,
        eval_model      = args.eval_model,
        gemma_model     = args.gemma_model,
        baseline_path   = args.baseline,
    )