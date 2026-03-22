"""
run_pipeline_v3.py
==================
Full pipeline runner — v3 (token splitter + OCR corrector fixes)

Runs the complete pipeline on all annotated images:
  Stage 1   — OCR (EasyOCR)
  Stage 2   — OCR Corrector (rule-based + lexicon snap)
  Stage 3   — Semantic Classifier (with token splitter for fused qty+unit)
  Stage 4   — Graph Constructor
  Stage 5   — Tuple Associator

Saves predictions to: results/tables/all_tuples_v3.csv

Run:
    python run_pipeline_v3.py

Then evaluate:
    python src/evaluation/evaluator.py --predictions results/tables/all_tuples_v3.csv
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, '.')

from src.ocr.ocr_runner import run_ocr_on_image
from src.utils.ocr_corrector import OCRCorrector
from src.classification.semantic_classifier import SemanticClassifier
from src.graph.graph_constructor import GraphConstructor
from src.matching.association import TupleAssociator

# ── Config ────────────────────────────────────────────────────────────────────

GT_IMAGES = [
    '1', '2', '6', '8', '9', '12', '15', '16', '17', '20',
    '30', '33', '34', '35', '42', '45', '51', '56', '59', '61', '63'
]

OUTPUT_CSV   = 'results/tables/all_tuples_v3.csv'
CONF_THRESH  = 0.30
FIELDNAMES   = ['image_id', 'nutrient', 'quantity', 'unit',
                'context', 'nrv_percent', 'serving_size']

# ── Pipeline components ───────────────────────────────────────────────────────

corrector   = OCRCorrector()
classifier  = SemanticClassifier(confidence_threshold=CONF_THRESH,
                                 split_fused_tokens=True)
constructor = GraphConstructor()
associator  = TupleAssociator()

# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  PIPELINE v3 — Token Splitter + OCR Corrector")
print("=" * 60)

all_tuples  = []
ok_count    = 0
skip_count  = 0
err_count   = 0

for img_id in GT_IMAGES:
    img_path = f'data/raw/{img_id}.jpeg'

    if not Path(img_path).exists():
        print(f'SKIP  {img_id}.jpeg  (not found)')
        skip_count += 1
        continue

    try:
        tokens    = run_ocr_on_image(img_path)
        corrected = corrector.correct_all(tokens)
        labeled   = classifier.classify_all(corrected)
        graph     = constructor.build(labeled)
        tuples    = associator.extract(graph, image_id=f'{img_id}.jpeg')
        all_tuples.extend(tuples)

        unit_count = sum(1 for t in labeled if t['label'] == 'UNIT')
        print(f'OK    {img_id}.jpeg  -> {len(tuples):>3} tuples  '
              f'| units in graph: {unit_count}')
        ok_count += 1

    except Exception as e:
        print(f'ERR   {img_id}.jpeg  -> {e}')
        err_count += 1

# ── Save ──────────────────────────────────────────────────────────────────────

Path('results/tables').mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(all_tuples)

print()
print("=" * 60)
print(f"  Images processed: {ok_count}  |  Skipped: {skip_count}  |  Errors: {err_count}")
print(f"  Total tuples:     {len(all_tuples)}")
print(f"  Output:           {OUTPUT_CSV}")
print("=" * 60)
print()
print("Next step:")
print("  python src/evaluation/evaluator.py "
      "--predictions results/tables/all_tuples_v3.csv")