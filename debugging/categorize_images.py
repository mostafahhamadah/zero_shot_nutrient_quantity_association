import sys
from collections import Counter
sys.path.insert(0, '.')

from src.ocr.ocr_runner import run_ocr_on_image
from src.utils.ocr_corrector import OCRCorrector
from src.classification.semantic_classifier import SemanticClassifier
from pathlib import Path

corrector  = OCRCorrector()
classifier = SemanticClassifier(confidence_threshold=0.30)

IMAGES = ['1','2','6','8','9','12','15','16','17','20',
          '30','33','34','35','42','45','51','56','59','61','63']

print(f'{"IMAGE":<10} {"TOTAL":>6} {"NUTR":>6} {"QTY":>5} {"UNIT":>5} '
      f'{"NOISE%":>7} {"UNKN%":>7}  VERDICT')
print('-' * 70)

for img_id in IMAGES:
    img_path = f'data/raw/{img_id}.jpeg'
    if not Path(img_path).exists():
        print(f'{img_id+".jpeg":<10}  NOT FOUND')
        continue

    try:
        tokens    = run_ocr_on_image(img_path)
        corrected = corrector.correct_all(tokens)
        labeled   = classifier.classify_all(corrected)

        counts = Counter(t['label'] for t in labeled)
        total  = len(labeled)
        noise_pct = counts.get('NOISE', 0) / total * 100
        unkn_pct  = counts.get('UNKNOWN', 0) / total * 100
        n_qty  = counts.get('QUANTITY', 0)
        n_unit = counts.get('UNIT', 0)
        n_nutr = counts.get('NUTRIENT', 0)

        # Verdict
        if n_qty >= 3 and n_unit >= 1 and noise_pct < 50:
            verdict = "✅ GOOD"
        elif n_qty >= 1 and noise_pct < 70:
            verdict = "⚠️  PARTIAL"
        else:
            verdict = "❌ POOR"

        print(f'{img_id+".jpeg":<10} {total:>6} {n_nutr:>6} {n_qty:>5} '
              f'{n_unit:>5} {noise_pct:>6.0f}% {unkn_pct:>6.0f}%  {verdict}')

    except Exception as e:
        print(f'{img_id+".jpeg":<10}  ERROR: {e}')