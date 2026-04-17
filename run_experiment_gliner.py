"""
run_experiment_gliner.py
========================
Experiment 3 — GLiNER Semantic Classifier Swap
Zero-Shot Nutrient Extraction | Moustafa Hamada | THD + USB

SCIENTIFIC QUESTION
-------------------
What does replacing the rule-based semantic classifier with a zero-shot
span-extraction model add to nutrient recognition quality?
Controlled variable: Stage 3 (classifier) only.
Stages 1, 2, 4, 5 are byte-for-byte identical to Experiment 2v1.

CHANGES FROM run_experiment_paddle.py (Experiment 2v1)
------------------------------------------------------
  Stage 3 : src/classification/gliner_classifier.py   (was experiment_01_final_semantic_classifier)
            Model  : Ihor/gliner-biomed-bi-base-v1.0
            Labels : Set B descriptive strings (4 entity types)
            Threshold: 0.3 (permissive, maximise recall)
            Fallback : UNKNOWN → passed to rule-based fallback chain
  Stages 1, 2, 4, 5 : UNCHANGED from Experiment 2v1

DIAGNOSTICS ADDITIONS vs. Experiment 2v1
-----------------------------------------
  gliner_spans_accepted   : total GLiNER spans accepted across all images (≥0.3 conf)
  gliner_avg_score        : mean confidence score of all accepted spans
  context_norm_resolved   : CONTEXT tokens where CONTEXT_MAP returned a canonical form
  context_norm_missing    : CONTEXT tokens where CONTEXT_MAP returned None
                            (German context header not in map — known failure mode)
  label_UNKNOWN           : tokens not covered by any GLiNER span (fallback)

USAGE
-----
  # Full run (LLM evaluator active — requires Ollama + phi4-mini)
  python run_experiment_gliner.py

  # Fast run without LLM
  python run_experiment_gliner.py --no-llm

  # Evaluate against Experiment 2v1 baseline for delta comparison
  python run_experiment_gliner.py --baseline outputs/experiment_02v1_paddleocr/evaluation_results.json

  # Override model ID (e.g. to test uni-encoder ablation)
  python run_experiment_gliner.py --gliner-model Ihor/gliner-biomed-large-v1.0
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

# ── Pipeline stage imports ────────────────────────────────────────────────────
from src.ocr.paddleocr_runner           import run_ocr_on_image                # Stage 1
from src.utils.paddleocr_corrector      import correct_tokens                  # Stage 2  (split_fused_token runs here)
from src.classification.gliner_classifier import GLiNERClassifier              # Stage 3  ← CHANGED
from src.graph.graph_constructor        import GraphConstructor                # Stage 4
from src.matching.experiment_01_final_association import TupleAssociator       # Stage 5
from src.evaluation.llm_evaluator       import LLMTupleEvaluator               # Stage 6

# ── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_NAME   = "experiment_03_gliner_biomed_bi_base"
GT_CSV            = "data/annotations/gold_annotations_4field.csv"
IMAGE_DIR         = Path("data/raw")

#: Default baseline for delta comparison — set to Exp 2v1 so the summary
#: table shows exactly what GLiNER adds on top of PaddleOCR.
BASELINE_JSON     = Path("outputs/experiment_02v1_paddleocr/evaluation_results.json")

#: GLiNER confidence threshold — must match CONFIDENCE_THRESHOLD in gliner_classifier.py
GLINER_THRESHOLD  = 0.30

IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png"}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(levelname)s | %(message)s",
)
logger = logging.getLogger("run_experiment_gliner")


# ── Stage runners ─────────────────────────────────────────────────────────────

def _run_stage1(image_path: Path) -> list:
    """
    Stage 1: PaddleOCR — unchanged from Experiment 2v1.
    Returns all tokens without confidence filtering.
    Schema: token, x1, y1, x2, y2, cx, cy, conf
    """
    return run_ocr_on_image(str(image_path))


def _run_stage2(tokens: list) -> tuple[list, dict]:
    """
    Stage 2: PaddleOCR corrector — unchanged from Experiment 2v1.
    split_fused_token() runs here — GLiNER classifier receives already-split tokens.

    Returns (corrected_tokens, corrector_diag).
    """
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


def _run_stage3_gliner(tokens: list, classifier: GLiNERClassifier) -> tuple[list, dict]:
    """
    Stage 3: GLiNER semantic classifier.

    Replaces the rule-based SemanticClassifier from Experiments 1 and 2.
    Calls GLiNERClassifier.classify() which:
      - Serialises tokens via text_serializer (Approach 3 local threshold)
      - Runs Ihor/gliner-biomed-bi-base-v1.0 with Set B descriptive labels
      - Applies tag-all span-to-token assignment at threshold=0.3
      - Normalises CONTEXT tokens via CONTEXT_MAP
      - Assigns UNKNOWN to unmatched tokens

    Returns (labelled_tokens, clf_diag).

    clf_diag keys
    -------------
    total_tokens          : int   total tokens entering Stage 3
    NUTRIENT              : int   tokens labelled NUTRIENT by GLiNER
    QUANTITY              : int   tokens labelled QUANTITY by GLiNER
    UNIT                  : int   tokens labelled UNIT by GLiNER
    CONTEXT               : int   tokens labelled CONTEXT by GLiNER
    UNKNOWN               : int   tokens not covered by any GLiNER span
    context_norm_resolved : int   CONTEXT tokens where CONTEXT_MAP returned a canonical form
    context_norm_missing  : int   CONTEXT tokens where CONTEXT_MAP returned None
                                  (German context header — known failure mode for Exp 3)
    gliner_spans_accepted : int   total accepted GLiNER spans for this image
    gliner_avg_score      : float mean confidence of accepted spans (0.0 if none)
    """
    labelled = classifier.classify(tokens)

    label_counts = Counter(t["label"] for t in labelled)

    # CONTEXT quality: how many CONTEXT tokens resolved to a canonical form
    context_tokens        = [t for t in labelled if t["label"] == "CONTEXT"]
    context_norm_resolved = sum(1 for t in context_tokens if t.get("norm") is not None)
    context_norm_missing  = len(context_tokens) - context_norm_resolved

    # GLiNER span statistics: collect from tokens that have a score
    accepted_scores = [
        t["gliner_score"]
        for t in labelled
        if t.get("gliner_score") is not None
    ]
    # Deduplicate by span text + score to count spans not tokens
    # (multi-token spans inflate the count — use unique (span_text, score) pairs)
    unique_spans = {
        (t.get("gliner_span_text"), t.get("gliner_score"))
        for t in labelled
        if t.get("gliner_score") is not None
    }

    diag = {
        "total_tokens":          len(labelled),
        "NUTRIENT":              label_counts.get("NUTRIENT", 0),
        "QUANTITY":              label_counts.get("QUANTITY", 0),
        "UNIT":                  label_counts.get("UNIT",     0),
        "CONTEXT":               label_counts.get("CONTEXT",  0),
        "UNKNOWN":               label_counts.get("UNKNOWN",  0),
        "context_norm_resolved": context_norm_resolved,
        "context_norm_missing":  context_norm_missing,
        "gliner_spans_accepted": len(unique_spans),
        "gliner_avg_score":      (
            round(sum(accepted_scores) / len(accepted_scores), 4)
            if accepted_scores else 0.0
        ),
    }
    return labelled, diag


def _run_stage4(labeled: list, graph_builder: GraphConstructor) -> tuple[dict, dict]:
    """
    Stage 4: Graph construction — unchanged from Experiment 2v1.
    Returns (graph, graph_diag) with per-edge-type counts.
    """
    graph = graph_builder.build(labeled)

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
    """
    Stage 5: Tuple association — unchanged from Experiment 2v1.
    Returns (tuples, assoc_diag).
    """
    tuples  = associator.extract(graph, image_id=image_id)
    no_ctx  = sum(1 for t in tuples if not t.get("context"))
    no_qty  = sum(1 for t in tuples if not t.get("quantity"))

    diag = {
        "tuples_extracted": len(tuples),
        "assoc_no_ctx":     no_ctx,
        "assoc_no_qty":     no_qty,
    }
    return tuples, diag


# ── Diagnostics helpers ───────────────────────────────────────────────────────

def _init_aggregate_diag() -> dict:
    return {
        # Stage 1 — OCR
        "ocr_total_tokens":       0,
        # Stage 2 — Corrector
        "corrector_splits":       0,
        "corrector_changed":      0,
        "corrector_rules":        Counter(),
        # Stage 3 — GLiNER classifier
        "clf_total_tokens":       0,
        "clf_NUTRIENT":           0,
        "clf_QUANTITY":           0,
        "clf_UNIT":               0,
        "clf_CONTEXT":            0,
        "clf_UNKNOWN":            0,
        "context_norm_resolved":  0,
        "context_norm_missing":   0,
        "gliner_spans_accepted":  0,
        "gliner_score_sum":       0.0,   # used to compute global avg score
        "gliner_score_n":         0,     # denominator for avg score
        # Stage 4 — Graph
        "graph_nodes":            0,
        "graph_edges":            0,
        "edge_SAME_ROW":          0,
        "edge_SAME_COL":          0,
        "edge_ADJACENT":          0,
        "edge_CONTEXT_SCOPE":     0,
        # Stage 5 — Association
        "assoc_tuples":           0,
        "assoc_no_ctx":           0,
        "assoc_no_qty":           0,
        # Special image flags
        "zero_ocr_images":        [],
        "near_noise_images":      [],    # ≥ 90% UNKNOWN (equivalent to near-noise in Exp 1)
        "zero_nutrient_images":   [],    # GLiNER found 0 NUTRIENT tokens
    }


def _accumulate_diag(agg: dict, img_id: str,
                     d1: dict, d2: dict, d3: dict, d4: dict, d5: dict) -> None:
    agg["ocr_total_tokens"]      += d1.get("total_tokens", 0)
    agg["corrector_splits"]      += d2["splits"]
    agg["corrector_changed"]     += d2["changed_tokens"]
    agg["corrector_rules"].update(d2["rules_fired"])
    agg["clf_total_tokens"]      += d3["total_tokens"]
    agg["clf_NUTRIENT"]          += d3["NUTRIENT"]
    agg["clf_QUANTITY"]          += d3["QUANTITY"]
    agg["clf_UNIT"]              += d3["UNIT"]
    agg["clf_CONTEXT"]           += d3["CONTEXT"]
    agg["clf_UNKNOWN"]           += d3["UNKNOWN"]
    agg["context_norm_resolved"] += d3["context_norm_resolved"]
    agg["context_norm_missing"]  += d3["context_norm_missing"]
    agg["gliner_spans_accepted"] += d3["gliner_spans_accepted"]
    if d3["gliner_avg_score"] > 0:
        agg["gliner_score_sum"]  += d3["gliner_avg_score"] * d3["gliner_spans_accepted"]
        agg["gliner_score_n"]    += d3["gliner_spans_accepted"]
    agg["graph_nodes"]           += d4["num_nodes"]
    agg["graph_edges"]           += d4["num_edges"]
    agg["edge_SAME_ROW"]         += d4["SAME_ROW"]
    agg["edge_SAME_COL"]         += d4["SAME_COL"]
    agg["edge_ADJACENT"]         += d4["ADJACENT"]
    agg["edge_CONTEXT_SCOPE"]    += d4["CONTEXT_SCOPE"]
    agg["assoc_tuples"]          += d5["tuples_extracted"]
    agg["assoc_no_ctx"]          += d5["assoc_no_ctx"]
    agg["assoc_no_qty"]          += d5["assoc_no_qty"]

    if d1.get("total_tokens", 0) == 0:
        agg["zero_ocr_images"].append(img_id)

    clf_total = d3["total_tokens"]
    if clf_total > 0:
        unknown_ratio = d3["UNKNOWN"] / clf_total
        if unknown_ratio >= 0.90:
            agg["near_noise_images"].append(img_id)

    if d3["NUTRIENT"] == 0:
        agg["zero_nutrient_images"].append(img_id)


def _print_diagnostics(agg: dict, n_images: int) -> None:
    W = 65
    clf_t = agg["clf_total_tokens"]
    pct   = lambda n: f"({n / clf_t * 100:.1f}%)" if clf_t else ""
    g_avg = (
        round(agg["gliner_score_sum"] / agg["gliner_score_n"], 4)
        if agg["gliner_score_n"] > 0 else 0.0
    )

    print(f"\n{'='*W}")
    print("  PIPELINE DIAGNOSTICS — Experiment 3 (GLiNER)")
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
        for rule, cnt in sorted(agg["corrector_rules"].items(),
                                key=lambda x: -x[1]):
            print(f"    {rule:<30} {cnt:>5}")
    print()
    print(f"  ── Stage 3 (GLiNER — Ihor/gliner-biomed-bi-base) ───")
    print(f"  Total tokens            : {clf_t}")
    print(f"  Spans accepted (≥{GLINER_THRESHOLD}) : {agg['gliner_spans_accepted']}")
    print(f"  Mean span confidence    : {g_avg:.4f}")
    print()
    for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "UNKNOWN"]:
        n = agg[f"clf_{label}"]
        print(f"  {label:<10}              : {n:>5}  {pct(n)}")
    print()
    print(f"  ── CONTEXT quality ──────────────────────────────────")
    ctx_total = agg["clf_CONTEXT"]
    print(f"  CONTEXT tokens total    : {ctx_total}")
    print(f"  Norm resolved           : {agg['context_norm_resolved']}"
          + (f"  ({agg['context_norm_resolved']/ctx_total*100:.1f}%)" if ctx_total else ""))
    print(f"  Norm MISSING (German?)  : {agg['context_norm_missing']}"
          + (f"  ({agg['context_norm_missing']/ctx_total*100:.1f}%)" if ctx_total else ""))
    if agg["near_noise_images"]:
        print(f"  ≥90%% UNKNOWN images     : {agg['near_noise_images']}")
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
    """Save flat diagnostics to pipeline_diagnostics.csv."""
    g_avg = (
        round(agg["gliner_score_sum"] / agg["gliner_score_n"], 4)
        if agg["gliner_score_n"] > 0 else 0.0
    )
    row = {
        "ocr_total_tokens":       agg["ocr_total_tokens"],
        "ocr_zero_images":        len(agg["zero_ocr_images"]),
        "corrector_splits":       agg["corrector_splits"],
        "corrector_changed":      agg["corrector_changed"],
        "clf_total_tokens":       agg["clf_total_tokens"],
        "clf_NUTRIENT":           agg["clf_NUTRIENT"],
        "clf_QUANTITY":           agg["clf_QUANTITY"],
        "clf_UNIT":               agg["clf_UNIT"],
        "clf_CONTEXT":            agg["clf_CONTEXT"],
        "clf_UNKNOWN":            agg["clf_UNKNOWN"],
        "context_norm_resolved":  agg["context_norm_resolved"],
        "context_norm_missing":   agg["context_norm_missing"],
        "gliner_spans_accepted":  agg["gliner_spans_accepted"],
        "gliner_avg_score":       g_avg,
        "graph_nodes":            agg["graph_nodes"],
        "graph_edges":            agg["graph_edges"],
        "edge_SAME_ROW":          agg["edge_SAME_ROW"],
        "edge_SAME_COL":          agg["edge_SAME_COL"],
        "edge_ADJACENT":          agg["edge_ADJACENT"],
        "edge_CONTEXT_SCOPE":     agg["edge_CONTEXT_SCOPE"],
        "assoc_tuples":           agg["assoc_tuples"],
        "assoc_no_ctx":           agg["assoc_no_ctx"],
        "assoc_no_qty":           agg["assoc_no_qty"],
    }
    out_path = out_dir / "pipeline_diagnostics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"Diagnostics saved: {out_path}")


# ── Baseline delta comparison ──────────────────────────────────────────────────

def _print_summary(current: dict, baseline_path: Path) -> None:
    """
    Print metric comparison table.
    When baseline JSON exists: shows BASELINE | CURRENT | DELTA columns.
    When baseline is missing: shows CURRENT column only.
    """
    METRIC_ROWS = [
        ("nutrient_f1",          "Nutrient F1"),
        ("quantity_acc",         "* Quantity Acc"),
        ("unit_acc",             "* Unit Acc"),
        ("context_acc",          "* Context Acc"),
        ("full3f_f1",            "* Full Tuple F1 (3-field)"),
        ("full_tuple_f1",        "* Full Tuple F1 (4-field)"),
        ("context_cost_pp",      "  Context cost 3f→4f (pp)"),
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
        title  = "EXPERIMENT 3 SUMMARY"
        header = f"  {'METRIC':<26}  {'VALUE':>10}"

    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(header)
    print(f"  {'-'*W}")

    for key, label in METRIC_ROWS:
        c_val = current.get(key, 0.0)
        if baseline:
            b_val = baseline.get(key, 0.0)
            delta = c_val - b_val
            print(f"  {label:<26}  {b_val:>10.4f}  {c_val:>10.4f}  {delta:>+.4f}")
        else:
            print(f"  {label:<26}  {c_val:>10.4f}")

    print(f"{'='*W}\n")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_experiment(
    use_llm:         bool  = True,
    baseline_path:   Path  = BASELINE_JSON,
    experiment_name: str   = EXPERIMENT_NAME,
    model:           str   = "qwen2.5:7b",
    gliner_model_id: str   = "Ihor/gliner-biomed-bi-large-v1.0",
) -> dict:
    """
    Run the GLiNER Experiment 3 across all images in IMAGE_DIR.

    Parameters
    ----------
    use_llm         : bool  — pass False to skip Ollama calls (fast evaluation)
    baseline_path   : Path  — path to Exp 2v1 evaluation_results.json for delta
    experiment_name : str   — sets output folder and CSV prefix
    model           : str   — Ollama model name for LLM evaluation
    gliner_model_id : str   — HF model ID for GLiNER (override for ablation)

    Returns
    -------
    dict  — evaluation metrics from LLMTupleEvaluator.run()
    """
    out_dir    = Path("outputs") / experiment_name
    tuples_csv = out_dir / f"{experiment_name}_tuples.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        logger.error(f"No images found in {IMAGE_DIR}  (expected .jpg/.jpeg/.png)")
        sys.exit(1)
    logger.info(f"Found {len(images)} images in {IMAGE_DIR}")

    # ── Instantiate stage objects ─────────────────────────────────────────────
    # GLiNERClassifier loads the model and caches label embeddings ONCE here.
    # All 43 image runs reuse the same instance — no reload per image.
    logger.info(f"Loading GLiNER classifier: {gliner_model_id}")
    classifier    = GLiNERClassifier(model_id=gliner_model_id, threshold=GLINER_THRESHOLD)
    graph_builder = GraphConstructor()
    associator    = TupleAssociator()

    all_tuples: list = []
    agg_diag         = _init_aggregate_diag()
    t_start          = time.time()

    # ── Per-image pipeline ────────────────────────────────────────────────────
    for img_path in images:
        image_id = img_path.name
        logger.info(f"Processing: {image_id}")

        try:
            # Stage 1 — PaddleOCR
            raw_tokens = _run_stage1(img_path)
            d1 = {"total_tokens": len(raw_tokens)}

            # Stage 2 — Corrector (split_fused_token runs here)
            corrected_tokens, d2 = _run_stage2(raw_tokens)

            # Stage 3 — GLiNER classifier ← CHANGED
            labeled_tokens, d3 = _run_stage3_gliner(corrected_tokens, classifier)

            # Stage 4 — Graph construction
            graph, d4 = _run_stage4(labeled_tokens, graph_builder)

            # Stage 5 — Tuple association
            tuples, d5 = _run_stage5(graph, image_id, associator)

        except Exception as exc:
            logger.error(f"Pipeline failed for {image_id}: {exc}", exc_info=True)
            d1 = {"total_tokens": 0}
            d2 = {"splits": 0, "changed_tokens": 0, "rules_fired": {}}
            d3 = {
                "total_tokens": 0, "NUTRIENT": 0, "QUANTITY": 0,
                "UNIT": 0, "CONTEXT": 0, "UNKNOWN": 0,
                "context_norm_resolved": 0, "context_norm_missing": 0,
                "gliner_spans_accepted": 0, "gliner_avg_score": 0.0,
            }
            d4 = {
                "num_nodes": 0, "num_edges": 0,
                "SAME_ROW": 0, "SAME_COL": 0, "ADJACENT": 0, "CONTEXT_SCOPE": 0,
            }
            d5 = {"tuples_extracted": 0, "assoc_no_ctx": 0, "assoc_no_qty": 0}
            tuples = []

        all_tuples.extend(tuples)
        _accumulate_diag(agg_diag, image_id, d1, d2, d3, d4, d5)

    elapsed = round(time.time() - t_start, 1)
    logger.info(
        f"Pipeline complete: {len(images)} images, "
        f"{len(all_tuples)} tuples, {elapsed}s"
    )

    # ── Save tuples CSV ───────────────────────────────────────────────────────
    fieldnames = ["image_id", "nutrient", "quantity", "unit", "context"]
    with open(tuples_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_tuples)
    logger.info(f"Tuples saved: {tuples_csv}  ({len(all_tuples)} rows)")

    # ── Print + save diagnostics ──────────────────────────────────────────────
    _print_diagnostics(agg_diag, len(images))
    _save_diagnostics_csv(agg_diag, out_dir)

    # ── Stage 6 — LLM Evaluator ───────────────────────────────────────────────
    evaluator = LLMTupleEvaluator(
        gt_csv  = GT_CSV,
        use_llm = use_llm,
        model   = model,
    )
    metrics = evaluator.run(
        predictions = all_tuples,
        experiment  = experiment_name,
        out_dir     = out_dir,
        notes       = (
            f"GLiNER Stage 3 swap. Model: {gliner_model_id}. "
            f"Entity labels: Set B descriptive. Threshold: {GLINER_THRESHOLD}. "
            f"Stages 1,2,4,5 identical to experiment_02v1_paddleocr. "
            f"LLM evaluator: {'enabled' if use_llm else 'disabled (fast run)'}."
        ),
        diagnostics = {
            k: v for k, v in agg_diag.items()
            if not isinstance(v, (Counter, list))
        },
    )

    # ── Delta vs. baseline ────────────────────────────────────────────────────
    _print_summary(metrics, baseline_path)

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLiNER classifier experiment — Experiment 3",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT_NAME,
        help=f"Experiment name — sets output folder (default: {EXPERIMENT_NAME})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable Ollama LLM calls — rule-based evaluation only (faster)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model for LLM evaluation (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_JSON,
        help="Path to baseline evaluation_results.json for delta comparison "
             "(default: Experiment 2v1)",
    )
    parser.add_argument(
        "--gliner-model",
        type=str,
        default="Ihor/gliner-biomed-bi-large-v1.0",
        help=(
            "HF model ID for GLiNER. Override for ablation experiments.\n"
            "  Bi-base (default): Ihor/gliner-biomed-bi-base-v1.0\n"
            "  Uni-large ablation: Ihor/gliner-biomed-large-v1.0"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        experiment_name = args.experiment,
        use_llm         = not args.no_llm,
        model           = args.model,
        baseline_path   = args.baseline,
        gliner_model_id = args.gliner_model,
    )