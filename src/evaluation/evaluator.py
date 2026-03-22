"""
evaluator.py  v3
================
Stage 6 — Tuple Evaluator

CHANGE IN v3:
  1. Full tuple definition corrected:
     fok = qok AND uok AND cok  (nutrient + quantity + unit + context all correct)
     Previously fok = qok AND uok only — context was ignored.

  2. Full Tuple Precision, Recall, F1 added:
     - Full Tuple Precision = full_correct / total_predicted
       (of all tuples the pipeline produced, what fraction are fully correct)
     - Full Tuple Recall    = full_correct / total_gt
       (of all GT tuples, what fraction did the pipeline get fully correct)
     - Full Tuple F1        = harmonic mean of precision and recall

  3. Deep pipeline diagnostics:
     Accepts optional per-image diagnostics dict from run_experiment.py.
     Saves pipeline_diagnostics.csv with per-stage token and edge counts.
     Enables post-experiment investigation of exactly where tuples are lost.

CHANGE IN v2:
  All evaluation logic moved from run_experiment.py into this module.
  norm_qty / norm_unit consolidated here.
  norm_context removed — handled by SemanticClassifier v4.
"""

import csv
import json
from pathlib import Path
from datetime import datetime


# ── Normalisation helpers ─────────────────────────────────────────────────────

def norm_qty(v) -> str:
    s = str(v or '').strip().replace(',', '.')
    s = s.lstrip('<>~')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f'{f:.10g}'
    except ValueError:
        return s.lower()


def norm_unit(v) -> str:
    s = str(v or '').strip().lower()
    s = s.replace('\u03bcg', '\u00b5g')
    s = s.replace('mcg',     '\u00b5g')
    s = s.replace('ug',      '\u00b5g')
    return s


def norm_str(v) -> str:
    return str(v or '').strip().lower()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TupleEvaluator:
    """
    Evaluates predicted tuples against ground truth annotations.

    v3: Full tuple definition corrected (qty + unit + context all required).
        Full Tuple Precision, Recall, F1 added.
        Pipeline diagnostics CSV saved when diagnostics dict provided.

    Usage:
        evaluator = TupleEvaluator(gt_csv='data/annotations/gold_annotations.csv')
        metrics = evaluator.run(
            predictions  = all_tuples,
            experiment   = 'experiment_07',
            out_dir      = Path('results/experiment_07'),
            notes        = 'description',
            diagnostics  = per_image_diagnostics,   # optional
        )
    """

    def __init__(self, gt_csv: str = 'data/annotations/gold_annotations.csv'):
        self.gt_csv = gt_csv

    def _load_gt(self) -> list:
        rows = []
        with open(self.gt_csv, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append({
                    'image_id': row.get('image_id', '').strip(),
                    'nutrient': row.get('nutrient', '').strip(),
                    'quantity': row.get('quantity', '').strip(),
                    'unit':     row.get('unit',     '').strip(),
                    'context':  row.get('context',  '').strip(),
                })
        return rows

    def run(self, predictions: list, experiment: str,
            out_dir: Path, notes: str = '',
            diagnostics: dict = None) -> dict:
        """
        Run full evaluation.

        Args:
            predictions : list of tuple dicts from TupleAssociator
            experiment  : experiment name string
            out_dir     : Path to results directory (must exist)
            notes       : optional notes string
            diagnostics : optional dict keyed by image_id with per-stage counts
                          produced by run_experiment.py pipeline loop.
                          Schema per image_id:
                          {
                            'ocr_total':        int,  # raw tokens from OCR
                            'ocr_low_conf':     int,  # tokens below conf threshold
                            'clf_nutrient':     int,  # tokens classified NUTRIENT
                            'clf_quantity':     int,
                            'clf_unit':         int,
                            'clf_context':      int,
                            'clf_noise':        int,
                            'clf_unknown':      int,
                            'graph_same_row':   int,  # edges per type
                            'graph_same_col':   int,
                            'graph_adjacent':   int,
                            'graph_ctx_scope':  int,
                            'assoc_tuples':     int,  # tuples produced
                            'assoc_no_qty':     int,  # nutrients with no qty
                            'assoc_no_unit':    int,  # nutrients with no unit
                            'assoc_no_ctx':     int,  # nutrients with no context
                          }

        Returns:
            dict with all metric keys
        """
        out_dir = Path(out_dir)

        # ── Load GT ───────────────────────────────────────────────────────────
        gt_rows = self._load_gt()
        gt_by_img   = {}
        pred_by_img = {}
        for row in gt_rows:
            gt_by_img.setdefault(row['image_id'], []).append(row)
        for t in predictions:
            pred_by_img.setdefault(t['image_id'], []).append(t)

        all_images = sorted(
            set(list(gt_by_img.keys()) + list(pred_by_img.keys()))
        )

        # ── Match loop ────────────────────────────────────────────────────────
        total_gt   = len(gt_rows)
        total_pred = len(predictions)
        tp = fp = fn = 0
        full_correct = 0          # tuples where ALL 4 fields match
        per_image    = []
        pair_rows    = []
        analysis_rows = []

        for img in all_images:
            gt_list   = gt_by_img.get(img, [])
            pred_list = pred_by_img.get(img, [])
            matched = qty_m = unit_m = ctx_m = full_m = 0
            used    = set()

            for gt in gt_list:
                gt_n = norm_str(gt['nutrient'])
                best = None
                for i, pred in enumerate(pred_list):
                    if i in used:
                        continue
                    pn = norm_str(pred.get('nutrient', ''))
                    if gt_n == pn or gt_n in pn or pn in gt_n:
                        best = (i, pred)
                        break

                if best is None:
                    fn += 1
                    analysis_rows.append({
                        'tag':           'missing',
                        'image_id':      img,
                        'gt_nutrient':   gt['nutrient'],
                        'gt_quantity':   gt['quantity'],
                        'gt_unit':       gt['unit'],
                        'gt_context':    gt['context'],
                        'pred_nutrient': '',
                        'pred_quantity': '',
                        'pred_unit':     '',
                        'pred_context':  '',
                        'reason':        'nutrient not found in predictions',
                    })
                    continue

                idx, pred = best
                used.add(idx)
                matched += 1
                tp      += 1

                gq = norm_qty(gt['quantity'])
                gu = norm_unit(gt['unit'])
                gc = norm_str(gt['context'])

                pq = norm_qty(pred.get('quantity'))
                pu = norm_unit(pred.get('unit'))
                pc = norm_str(pred.get('context'))

                qok = bool(gq) and gq == pq
                uok = bool(gu) and gu == pu
                cok = bool(gc) and gc == pc

                # v3: full tuple requires ALL FOUR fields correct
                # nutrient is already matched above (tp condition)
                fok = qok and uok and cok

                if qok: qty_m  += 1
                if uok: unit_m += 1
                if cok: ctx_m  += 1
                if fok:
                    full_m       += 1
                    full_correct += 1
                else:
                    reasons = []
                    if not qok:
                        reasons.append(
                            f'qty: got {pred.get("quantity","")} '
                            f'expected {gt["quantity"]}'
                        )
                    if not uok:
                        reasons.append(
                            f'unit: got {pred.get("unit","")} '
                            f'expected {gt["unit"]}'
                        )
                    if not cok:
                        reasons.append(
                            f'context: got {pred.get("context","")} '
                            f'expected {gt["context"]}'
                        )
                    analysis_rows.append({
                        'tag':           'incorrect',
                        'image_id':      img,
                        'gt_nutrient':   gt['nutrient'],
                        'gt_quantity':   gt['quantity'],
                        'gt_unit':       gt['unit'],
                        'gt_context':    gt['context'],
                        'pred_nutrient': pred.get('nutrient', ''),
                        'pred_quantity': pred.get('quantity', ''),
                        'pred_unit':     pred.get('unit', ''),
                        'pred_context':  pred.get('context', ''),
                        'reason':        ' | '.join(reasons),
                    })

                pair_rows.append({
                    'image_id':      img,
                    'gt_nutrient':   gt['nutrient'],
                    'pred_nutrient': pred.get('nutrient', ''),
                    'gt_qty':        gt['quantity'],
                    'pred_qty':      pred.get('quantity', ''),
                    'gt_unit':       gt['unit'],
                    'pred_unit':     pred.get('unit', ''),
                    'gt_context':    gt['context'],
                    'pred_context':  pred.get('context', ''),
                    'qty_match':     qok,
                    'unit_match':    uok,
                    'ctx_match':     cok,
                    'full_match':    fok,
                })

            fp += len(pred_list) - len(used)
            nm = matched
            per_image.append({
                'image_id': img,
                'gt':       len(gt_list),
                'pred':     len(pred_list),
                'match':    nm,
                'qty_pct':  round(qty_m  / nm * 100) if nm else 0,
                'unit_pct': round(unit_m / nm * 100) if nm else 0,
                'ctx_pct':  round(ctx_m  / nm * 100) if nm else 0,
                'full_pct': round(full_m / nm * 100) if nm else 0,
            })

        # ── Compute metrics ───────────────────────────────────────────────────
        # Nutrient-level
        nutr_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        nutr_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        nutr_f1        = (2 * nutr_precision * nutr_recall /
                          (nutr_precision + nutr_recall)
                          if (nutr_precision + nutr_recall) > 0 else 0)
        np_ = len(pair_rows)

        # Field-level match accuracies (among nutrient-matched pairs)
        qty_acc  = sum(1 for r in pair_rows if r['qty_match'])  / max(np_, 1)
        unit_acc = sum(1 for r in pair_rows if r['unit_match']) / max(np_, 1)
        ctx_acc  = sum(1 for r in pair_rows if r['ctx_match'])  / max(np_, 1)

        # Full tuple metrics
        # Precision: of all predicted tuples, how many are fully correct
        # Recall:    of all GT tuples, how many did we get fully correct
        # F1:        harmonic mean
        full_precision = full_correct / total_pred if total_pred > 0 else 0
        full_recall    = full_correct / total_gt   if total_gt   > 0 else 0
        full_f1        = (2 * full_precision * full_recall /
                          (full_precision + full_recall)
                          if (full_precision + full_recall) > 0 else 0)

        # ── Print results ─────────────────────────────────────────────────────
        W = 65
        print(f"{'='*W}")
        print(f"  EXPERIMENT RESULTS : {experiment}")
        print(f"{'='*W}")
        print(f"  Images evaluated  : {len(all_images)}")
        print(f"  GT tuples         : {total_gt}")
        print(f"  Predicted tuples  : {total_pred}")
        print(f"  Matched pairs     : {np_}")
        print(f"{'─'*W}")
        print(f"  Nutrient Precision : {nutr_precision:.3f}")
        print(f"  Nutrient Recall    : {nutr_recall:.3f}")
        print(f"  Nutrient F1        : {nutr_f1:.3f}")
        print(f"{'─'*W}")
        print(f"  Quantity Match Acc : {qty_acc*100:.1f}%")
        print(f"  Unit Match Acc     : {unit_acc*100:.1f}%")
        print(f"  Context Match Acc  : {ctx_acc*100:.1f}%")
        print(f"{'─'*W}")
        print(f"  Full Tuple (nutrient + qty + unit + ctx)")
        print(f"  ★ Full Tuple Precision : {full_precision*100:.1f}%")
        print(f"  ★ Full Tuple Recall    : {full_recall*100:.1f}%")
        print(f"  ★ Full Tuple F1        : {full_f1*100:.1f}%")
        print(f"{'='*W}")
        print()
        print(f"  {'IMAGE':<17} {'GT':>4}  {'PRED':>5}  {'MATCH':>6}  "
              f"{'QTY%':>5}  {'UNIT%':>6}  {'CTX%':>5}  {'FULL%':>6}")
        print(f"  {'─'*W}")
        for row in per_image:
            print(f"  {row['image_id']:<17} {row['gt']:>4}  {row['pred']:>5}  "
                  f"{row['match']:>6}  {row['qty_pct']:>4}%  "
                  f"{row['unit_pct']:>5}%  {row['ctx_pct']:>4}%  "
                  f"{row['full_pct']:>5}%")

        # ── Build metrics dict ────────────────────────────────────────────────
        metrics = {
            'experiment':           experiment,
            'timestamp':            datetime.now().isoformat(),
            'notes':                notes,
            'schema':               'nutrient|quantity|unit|context',
            'images_evaluated':     len(all_images),
            'gt_tuples':            total_gt,
            'predicted_tuples':     total_pred,
            'matched_pairs':        np_,
            'nutrient_precision':   round(nutr_precision, 4),
            'nutrient_recall':      round(nutr_recall,    4),
            'nutrient_f1':          round(nutr_f1,        4),
            'quantity_acc':         round(qty_acc,        4),
            'unit_acc':             round(unit_acc,       4),
            'context_acc':          round(ctx_acc,        4),
            'full_tuple_correct':   full_correct,
            'full_tuple_precision': round(full_precision, 4),
            'full_tuple_recall':    round(full_recall,    4),
            'full_tuple_f1':        round(full_f1,        4),
        }

        # ── Save output files ─────────────────────────────────────────────────
        metrics_json = out_dir / 'evaluation_results.json'
        summary_csv  = out_dir / 'evaluation_summary.csv'
        per_img_csv  = out_dir / 'per_image_results.csv'
        pairs_csv    = out_dir / 'pair_details.csv'
        analysis_csv = out_dir / f'{experiment}_analysis.csv'

        with open(metrics_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            w.writeheader()
            w.writerow(metrics)

        with open(per_img_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'image_id', 'gt', 'pred', 'match',
                'qty_pct', 'unit_pct', 'ctx_pct', 'full_pct'
            ])
            w.writeheader()
            w.writerows(per_image)

        with open(pairs_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'image_id', 'gt_nutrient', 'pred_nutrient',
                'gt_qty', 'pred_qty', 'gt_unit', 'pred_unit',
                'gt_context', 'pred_context',
                'qty_match', 'unit_match', 'ctx_match', 'full_match'
            ])
            w.writeheader()
            w.writerows(pair_rows)

        analysis_fields = [
            'tag', 'image_id',
            'gt_nutrient',   'gt_quantity',   'gt_unit',   'gt_context',
            'pred_nutrient', 'pred_quantity', 'pred_unit', 'pred_context',
            'reason',
        ]
        with open(analysis_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=analysis_fields)
            w.writeheader()
            w.writerows(analysis_rows)

        # ── Save pipeline diagnostics CSV (if provided) ───────────────────────
        if diagnostics:
            diag_csv = out_dir / 'pipeline_diagnostics.csv'
            diag_fields = [
                'image_id',
                # OCR stage
                'ocr_total', 'ocr_low_conf', 'ocr_passed',
                # Classifier stage
                'clf_nutrient', 'clf_quantity', 'clf_unit',
                'clf_context',  'clf_noise',    'clf_unknown',
                'clf_noise_pct',
                # Graph stage
                'graph_same_row', 'graph_same_col',
                'graph_adjacent', 'graph_ctx_scope',
                'graph_total_edges',
                # Association stage
                'assoc_tuples', 'assoc_no_qty',
                'assoc_no_unit', 'assoc_no_ctx',
                # GT reference
                'gt_tuples',
                # Loss funnel
                'lost_at_ocr', 'lost_at_clf', 'lost_at_graph', 'lost_at_assoc',
            ]
            diag_rows = []
            for img in all_images:
                d   = diagnostics.get(img, {})
                gt  = len(gt_by_img.get(img, []))
                diag_rows.append({
                    'image_id':          img,
                    # OCR
                    'ocr_total':         d.get('ocr_total',    0),
                    'ocr_low_conf':      d.get('ocr_low_conf', 0),
                    'ocr_passed':        d.get('ocr_total',    0) - d.get('ocr_low_conf', 0),
                    # Classifier
                    'clf_nutrient':      d.get('clf_nutrient', 0),
                    'clf_quantity':      d.get('clf_quantity', 0),
                    'clf_unit':          d.get('clf_unit',     0),
                    'clf_context':       d.get('clf_context',  0),
                    'clf_noise':         d.get('clf_noise',    0),
                    'clf_unknown':       d.get('clf_unknown',  0),
                    'clf_noise_pct':     round(
                        d.get('clf_noise', 0) /
                        max(d.get('ocr_total', 1), 1) * 100, 1),
                    # Graph
                    'graph_same_row':    d.get('graph_same_row',  0),
                    'graph_same_col':    d.get('graph_same_col',  0),
                    'graph_adjacent':    d.get('graph_adjacent',  0),
                    'graph_ctx_scope':   d.get('graph_ctx_scope', 0),
                    'graph_total_edges': (d.get('graph_same_row', 0) +
                                          d.get('graph_same_col', 0) +
                                          d.get('graph_adjacent', 0) +
                                          d.get('graph_ctx_scope', 0)),
                    # Association
                    'assoc_tuples':      d.get('assoc_tuples',  0),
                    'assoc_no_qty':      d.get('assoc_no_qty',  0),
                    'assoc_no_unit':     d.get('assoc_no_unit', 0),
                    'assoc_no_ctx':      d.get('assoc_no_ctx',  0),
                    # GT reference
                    'gt_tuples':         gt,
                    # Loss funnel — how many GT tuples are unaccounted at each stage
                    # If clf_nutrient < gt → lost at classification
                    # If graph_ctx_scope == 0 and gt has context → lost at graph
                    # If assoc_no_qty > 0 → lost at association
                    'lost_at_ocr':   max(0, gt - d.get('clf_nutrient', 0)),
                    'lost_at_clf':   d.get('clf_unknown', 0),
                    'lost_at_graph': max(0, gt - d.get('graph_ctx_scope', 1)
                                         if d.get('clf_context', 0) > 0 else 0),
                    'lost_at_assoc': d.get('assoc_no_qty', 0),
                })

            with open(diag_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=diag_fields)
                w.writeheader()
                w.writerows(diag_rows)

            # Print diagnostics summary
            print(f"\n{'='*W}")
            print(f"  PIPELINE DIAGNOSTICS SUMMARY")
            print(f"{'='*W}")
            total_ocr    = sum(d.get('ocr_total',    0) for d in diagnostics.values())
            total_lowc   = sum(d.get('ocr_low_conf', 0) for d in diagnostics.values())
            total_nutr   = sum(d.get('clf_nutrient', 0) for d in diagnostics.values())
            total_qty    = sum(d.get('clf_quantity', 0) for d in diagnostics.values())
            total_unit   = sum(d.get('clf_unit',     0) for d in diagnostics.values())
            total_ctx    = sum(d.get('clf_context',  0) for d in diagnostics.values())
            total_noise  = sum(d.get('clf_noise',    0) for d in diagnostics.values())
            total_cscope = sum(d.get('graph_ctx_scope', 0) for d in diagnostics.values())
            total_nqty   = sum(d.get('assoc_no_qty',  0) for d in diagnostics.values())
            total_nunit  = sum(d.get('assoc_no_unit', 0) for d in diagnostics.values())
            total_nctx   = sum(d.get('assoc_no_ctx',  0) for d in diagnostics.values())

            print(f"  [OCR]")
            print(f"    Total tokens extracted : {total_ocr}")
            print(f"    Below conf threshold   : {total_lowc}"
                  f"  ({total_lowc/max(total_ocr,1)*100:.1f}% of tokens)")
            print(f"  [CLASSIFIER]")
            print(f"    NUTRIENT  : {total_nutr}")
            print(f"    QUANTITY  : {total_qty}")
            print(f"    UNIT      : {total_unit}")
            print(f"    CONTEXT   : {total_ctx}")
            print(f"    NOISE     : {total_noise}"
                  f"  ({total_noise/max(total_ocr,1)*100:.1f}% of tokens)")
            print(f"  [GRAPH]")
            print(f"    CONTEXT_SCOPE edges    : {total_cscope}")
            print(f"  [ASSOCIATION]")
            print(f"    Nutrients with no qty  : {total_nqty}")
            print(f"    Nutrients with no unit : {total_nunit}")
            print(f"    Nutrients with no ctx  : {total_nctx}")
            print(f"{'='*W}")
            print(f"    {diag_csv}")

        # ── Print analysis summary ────────────────────────────────────────────
        n_missing   = sum(1 for r in analysis_rows if r['tag'] == 'missing')
        n_incorrect = sum(1 for r in analysis_rows if r['tag'] == 'incorrect')
        print(f"\n{'='*W}")
        print(f"  ANALYSIS SUMMARY")
        print(f"{'='*W}")
        print(f"  Missing tuples   : {n_missing}"
              f"  (nutrient not found in predictions)")
        print(f"  Incorrect tuples : {n_incorrect}"
              f"  (nutrient matched but fields wrong)")
        print(f"  Total flagged    : {len(analysis_rows)}")
        print(f"{'='*W}")

        print(f"\n  Saved:")
        for p in [metrics_json, summary_csv, per_img_csv,
                  pairs_csv, analysis_csv]:
            print(f"    {p}")

        return metrics