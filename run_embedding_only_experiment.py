"""
run_embedding_only_experiment.py
================================
Runs the V2 pipeline end-to-end using the embedding classifier in
`embedding_only` mode (no rule lexicon for nutrients). Threshold and
margin default to the values picked by the token-level grid search
on the relabelled GT (t=0.59, m=0.04).

PIPELINE (identical to run_graph_v2_experiment.py)
--------------------------------------------------
  Stage 1   — PaddleOCR
  Stage 2   — PaddleOCR Corrector
  Stage 3   — EmbeddingSemanticClassifier  (mode=embedding_only)
  Stage 3.5 — Token Enricher
  Stage 4   — Graph V2
  Stage 5   — Association V2 (+ paragraph fallback)
  Stage 6   — LLM-Assisted Tuple Evaluator

USAGE
-----
    python run_embedding_only_experiment.py --experiment exp40_embonly_tuned
    python run_embedding_only_experiment.py --experiment exp40_t60 --threshold 0.60
    python run_embedding_only_experiment.py --experiment exp40_quick --no-llm
    python run_embedding_only_experiment.py --experiment exp40 --compare graph_v2_exp38
"""

import sys, csv, json, argparse, os
import importlib.util
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

from src.ocr.paddleocr_runner          import run_ocr_on_image
from src.utils.paddleocr_corrector     import correct_tokens
from src.utils.token_enricher          import TokenEnricher
from src.utils.paragraph_extractor     import (extract_from_paragraph,
                                               should_use_paragraph_mode)
from src.utils.sentence_extractor      import extract_from_sentences
from src.graph.graph_constructor_v2    import GraphConstructorV2
from src.matching.association_v2       import TupleAssociatorV2
from src.evaluation.llm_evaluator      import LLMTupleEvaluator


# ── Dynamic loader (mirrors run_graph_v2_experiment.py pattern) ───────────────

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = os.path.dirname(__file__)
_emb_module = _load_module(
    'emb_classifier',
    os.path.join(_base, 'src', 'classification',
                 'embedding_semantic_classifier.py'),
)
EmbeddingSemanticClassifier = _emb_module.EmbeddingSemanticClassifier


# ── GT loader ─────────────────────────────────────────────────────────────────

def load_gt_from_json(annotations_dir: str) -> list:
    rows = []
    for jf in sorted(Path(annotations_dir).glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[load_gt] Failed to load {jf.name}: {e}")
            continue
        raw_id   = Path(data.get("image_id", jf.stem + ".jpeg")).name
        image_id = Path(raw_id).stem + Path(raw_id).suffix.lower()
        for n in data.get("nutrients", []):
            context      = str(n.get("context",      "")).strip()
            serving_size = str(n.get("serving_size", "") or "").strip()
            context_full = (f"{context} ({serving_size})"
                            if serving_size else context)
            rows.append({
                "image_id": image_id,
                "nutrient": str(n.get("nutrient", "")).strip(),
                "quantity": str(n.get("quantity", "")).strip(),
                "unit":     str(n.get("unit",     "")).strip(),
                "context":  context_full,
            })
    return rows


# ── Config ────────────────────────────────────────────────────────────────────

GT_ANNOTATIONS = 'data/annotations'
RAW_DIR        = 'data/raw'
CONF_THRESH    = 0.30
FIELDNAMES     = ['image_id', 'nutrient', 'quantity', 'unit', 'context']

# Tuned via token-level grid search on relabelled GT (462 pos / 334 neg)
TUNED_THRESHOLD = 0.59
TUNED_MARGIN    = 0.04


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
parser.add_argument('--images',    default=None,
                    help="Comma-separated image stems (e.g. '1,101,118')")
parser.add_argument('--compare',   default=None,
                    help="Prior experiment folder name to diff against")
parser.add_argument('--notes',     default='')
parser.add_argument('--threshold', type=float, default=TUNED_THRESHOLD,
                    help=f"NUTRIENT cosine threshold (default {TUNED_THRESHOLD})")
parser.add_argument('--margin',    type=float, default=TUNED_MARGIN,
                    help=f"NUTRIENT vs second-best margin (default {TUNED_MARGIN})")
parser.add_argument('--no-llm',    action='store_true',
                    help='Skip LLM evaluation — fast rule-based pass only')
args = parser.parse_args()

EXPERIMENT = args.experiment
USE_LLM    = not args.no_llm
THRESHOLD  = args.threshold
MARGIN     = args.margin

OUT_DIR    = Path(f'outputs/{EXPERIMENT}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TUPLES_CSV = OUT_DIR / 'tuples.csv'
LOG_FILE   = OUT_DIR / 'run_log.txt'


# ── Image discovery ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".tif"}

if args.images:
    wanted = {s.strip() for s in args.images.split(',') if s.strip()}
    IMAGE_FILES = [f for f in Path(RAW_DIR).iterdir()
                   if f.stem in wanted and f.suffix.lower() in IMAGE_EXTENSIONS]
else:
    IMAGE_FILES = [f for f in Path(RAW_DIR).iterdir()
                   if f.suffix.lower() in IMAGE_EXTENSIONS]
IMAGE_FILES.sort(key=lambda f: (int(f.stem) if f.stem.isdigit() else 1e9, f.name))


# ── Banner ────────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  EXPERIMENT : {EXPERIMENT}")
print(f"  Pipeline   : V2 (PaddleOCR + Embedding Classifier + Graph V2)")
print(f"  Classifier : embedding_only  (BGE-M3, t={THRESHOLD}, m={MARGIN})")
print(f"  Schema     : nutrient | quantity | unit | context")
print(f"  Evaluator  : {'LLM (qwen2.5:3b)' if USE_LLM else 'fast rule-based'}")
print(f"  Images     : {len(IMAGE_FILES)} from {RAW_DIR}/")
print(f"  Output     : outputs/{EXPERIMENT}/")
if args.notes:
    print(f"  Notes      : {args.notes}")
print(f"{'='*65}\n")


# ── Pipeline instances ────────────────────────────────────────────────────────

classifier  = EmbeddingSemanticClassifier(
    mode                 = "embedding_only",
    nutrient_threshold   = THRESHOLD,
    margin               = MARGIN,
    confidence_threshold = CONF_THRESH,
)
enricher    = TokenEnricher()
constructor = GraphConstructorV2()
associator  = TupleAssociatorV2()


# ── Per-image loop ────────────────────────────────────────────────────────────

all_tuples  = []
diagnostics = {}
ok = err = 0

for img_path in IMAGE_FILES:
    image_key = img_path.stem + img_path.suffix.lower()
    try:
        # 1. OCR
        tokens       = run_ocr_on_image(str(img_path))
        ocr_total    = len(tokens)
        ocr_low_conf = sum(1 for t in tokens if t.get('conf', 1.0) < CONF_THRESH)

        # 2. Corrector
        corrected, _ = correct_tokens(tokens, return_log=True)

        # 3. Embedding classifier
        labeled    = classifier.classify_all(corrected)
        clf_counts = Counter(t['label'] for t in labeled)
        clf_emb    = sum(1 for t in labeled
                         if t.get('classification_method') == 'embedding')

        # 3.5 Enricher
        enriched = enricher.enrich(labeled)
        enr_diag = enricher.diagnostics

        # 4. Graph V2
        graph       = constructor.build(enriched)
        edge_counts = Counter(e['type'] for e in graph.get('edges', []))

        # 5. Association V2
        tuples     = associator.extract(graph, image_id=image_key)
        assoc_diag = associator.diagnostics

        # 5b. Paragraph-mode fallback
        if should_use_paragraph_mode(labeled, len(tuples)):
            para = extract_from_paragraph(labeled, image_key)
            if len(para) > len(tuples):
                print(f"  [paragraph] {image_key}: {len(tuples)} -> {len(para)}")
                tuples = para

        all_tuples.extend(tuples)

        diagnostics[image_key] = {
            'ocr_total':           ocr_total,
            'ocr_low_conf':        ocr_low_conf,
            'clf_nutrient':        clf_counts.get('NUTRIENT', 0),
            'clf_quantity':        clf_counts.get('QUANTITY', 0),
            'clf_unit':            clf_counts.get('UNIT',     0),
            'clf_context':         clf_counts.get('CONTEXT',  0),
            'clf_noise':           clf_counts.get('NOISE',    0),
            'clf_unknown':         clf_counts.get('UNKNOWN',  0),
            'clf_via_embedding':   clf_emb,
            'enr_rows':            enr_diag.get('num_rows',         0),
            'enr_columns':         enr_diag.get('num_columns',      0),
            'enr_dosage_streams':  enr_diag.get('dosage_streams',   0),
            'enr_headers':         enr_diag.get('headers_detected', 0),
            'enr_rank_consistent': enr_diag.get('rank_consistent',  False),
            'graph_row_compat':    edge_counts.get('ROW_COMPAT',       0),
            'graph_col_compat':    edge_counts.get('COL_COMPAT',       0),
            'graph_dir_adj':       edge_counts.get('DIRECTIONAL_ADJ',  0),
            'graph_header_scope':  edge_counts.get('HEADER_SCOPE',     0),
            'assoc_tuples':        len(tuples),
            'assoc_matches':       assoc_diag.get('matches', 0),
            'assoc_no_qty':        sum(1 for t in tuples if not t.get('quantity')),
            'assoc_no_unit':       sum(1 for t in tuples if not t.get('unit')),
            'assoc_no_ctx':        sum(1 for t in tuples if not t.get('context')),
        }

        print(f"OK    {image_key}  -> {len(tuples):>3} tuples  "
              f"| NUTR:{clf_counts.get('NUTRIENT',0):>3}  "
              f"EMB:{clf_emb:>3}  "
              f"ROWS:{enr_diag.get('num_rows',0):>3}  "
              f"STREAMS:{enr_diag.get('dosage_streams',0)}")
        ok += 1

    except Exception as e:
        import traceback
        print(f"ERR   {image_key}  -> {e}")
        traceback.print_exc()
        err += 1


# ── Save tuples ───────────────────────────────────────────────────────────────

with open(TUPLES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    writer.writerows([{k: t.get(k, '') for k in FIELDNAMES} for t in all_tuples])

print(f"\n{'='*65}")
print(f"  Processed: {ok}  Errors: {err}  Total tuples: {len(all_tuples)}")
print(f"{'='*65}\n")


# ── Evaluation ────────────────────────────────────────────────────────────────

print("Running evaluation...\n")
gt_rows = load_gt_from_json(GT_ANNOTATIONS)
print(f"GT loaded: {len(gt_rows)} tuples\n")

evaluator = LLMTupleEvaluator(
    gt_rows = gt_rows,
    use_llm = USE_LLM,
    model   = "qwen2.5:3b",
)
metrics = evaluator.run(
    predictions = all_tuples,
    experiment  = EXPERIMENT,
    out_dir     = OUT_DIR,
    notes       = args.notes,
    diagnostics = diagnostics,
)


# ── Run log ───────────────────────────────────────────────────────────────────

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Experiment   : {EXPERIMENT}\n")
    f.write(f"Pipeline     : V2 + EmbeddingSemanticClassifier (embedding_only)\n")
    f.write(f"Threshold    : {THRESHOLD}\n")
    f.write(f"Margin       : {MARGIN}\n")
    f.write(f"Evaluator    : {'LLM (qwen2.5:3b)' if USE_LLM else 'fast rule-based'}\n")
    f.write(f"Timestamp    : {metrics['timestamp']}\n")
    f.write(f"Notes        : {args.notes}\n\n")
    f.write(f"GT tuples            : {metrics['gt_tuples']}\n")
    f.write(f"Predicted tuples     : {metrics['predicted_tuples']}\n")
    f.write(f"Matched pairs        : {metrics['matched_pairs']}\n")
    f.write(f"Full 4F correct      : {metrics.get('full4f_correct', 0)}\n")
    f.write(f"Full Tuple Precision : {metrics['full_tuple_precision']*100:.1f}%\n")
    f.write(f"Full Tuple Recall    : {metrics['full_tuple_recall']*100:.1f}%\n")
    f.write(f"Full Tuple F1        : {metrics['full_tuple_f1']*100:.1f}%\n")
    f.write(f"Nutrient F1          : {metrics['nutrient_f1']:.3f}\n")
    f.write(f"Unit Acc             : {metrics['unit_acc']*100:.1f}%\n")
    f.write(f"Context Acc          : {metrics['context_acc']*100:.1f}%\n")


# ── Headline (the number you actually care about) ─────────────────────────────

print(f"\n{'='*65}")
print(f"  HEADLINE — true hits (full 4-field matches)")
print(f"{'='*65}")
print(f"  GT tuples           : {metrics['gt_tuples']}")
print(f"  Predicted tuples    : {metrics['predicted_tuples']}")
print(f"  TRUE HITS (4F corr) : {metrics.get('full4f_correct', '?')}")
print(f"  Full Tuple F1       : {metrics['full_tuple_f1']*100:.2f}%")
print(f"  Nutrient F1         : {metrics['nutrient_f1']:.3f}")
print(f"{'='*65}\n")
print(f"Saved: {TUPLES_CSV}")
print(f"Saved: {LOG_FILE}")
print(f"Saved: outputs/{EXPERIMENT}/evaluation_summary.csv")


# ── Optional comparison ───────────────────────────────────────────────────────

if args.compare:
    cmp_path = Path(f'outputs/{args.compare}/evaluation_results.json')
    if cmp_path.exists():
        prev = json.loads(cmp_path.read_text(encoding='utf-8'))
        print(f"\n{'='*65}")
        print(f"  COMPARISON : {args.compare}  ->  {EXPERIMENT}")
        print(f"{'='*65}")
        for label, key in [
            ('TRUE HITS (4F correct)', 'full4f_correct'),
            ('Full Tuple Precision',   'full_tuple_precision'),
            ('Full Tuple Recall',      'full_tuple_recall'),
            ('Full Tuple F1',          'full_tuple_f1'),
            ('Nutrient F1',            'nutrient_f1'),
            ('Unit Match Acc',         'unit_acc'),
            ('Quantity Match Acc',     'quantity_acc'),
            ('Context Match Acc',      'context_acc'),
            ('Matched Pairs',          'matched_pairs'),
            ('Predicted Tuples',       'predicted_tuples'),
        ]:
            old = prev.get(key, 0)
            new = metrics.get(key, 0)
            try:
                d    = float(new) - float(old)
                sign = '+' if d >= 0 else ''
                arr  = '↑' if d > 0 else ('↓' if d < 0 else '=')
                print(f"  {label:<26}  {old:.3f}  ->  {new:.3f}  "
                      f"({sign}{d:.3f}) {arr}")
            except Exception:
                print(f"  {label:<26}  {old}  ->  {new}")
        print(f"{'='*65}\n")
    else:
        print(f"\n  [compare] Not found: {cmp_path}")