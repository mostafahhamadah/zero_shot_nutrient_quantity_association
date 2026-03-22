import sys
from collections import Counter
sys.path.insert(0, '.')

from src.ocr.ocr_runner import run_ocr_on_image
from src.utils.ocr_corrector import OCRCorrector
from src.classification.semantic_classifier import SemanticClassifier

corrector  = OCRCorrector()
classifier = SemanticClassifier(confidence_threshold=0.30)

tokens    = run_ocr_on_image('data/raw/45.jpeg')
corrected = corrector.correct_all(tokens)
labeled   = classifier.classify_all(corrected)

print(f'Total raw tokens: {len(tokens)}')
print(f'After classifier: {len(labeled)}')
print()

counts = Counter(t['label'] for t in labeled)
print('Label distribution:', dict(counts))
print()

print('ALL tokens with labels:')
print(f'  {"LABEL":<12} {"CONF":<6}  TOKEN')
print('  ' + '-'*70)
for t in sorted(labeled, key=lambda x: x['y1']):
    print(f'  {t["label"]:<12} {t["conf"]:.2f}   {t["token"]}')