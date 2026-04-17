# save as diag_context.py in your project root
import sys, os
sys.path.insert(0, '.')
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

clf_mod = _load('clf', 'src/classification/experiment_01_final_semantic_classifier.py')

from src.ocr.ocr_runner import run_ocr_on_image
from src.utils.ocr_corrector import OCRCorrector

clf = clf_mod.SemanticClassifier(0.30)
cor = OCRCorrector()

TARGET_IMAGES = ['103', '104', '116', '118', '119', '120']

for img_id in TARGET_IMAGES:
    path = f'data/raw/{img_id}.jpeg'
    print(f'\n{"="*70}')
    print(f'  {img_id}.jpeg')
    print(f'{"="*70}')
    tokens    = run_ocr_on_image(path)
    corrected = cor.correct_all(tokens)
    labeled   = clf.classify_all(corrected)

    # Sort top-to-bottom, left-to-right
    labeled.sort(key=lambda t: (t['y1'], t['x1']))

    print(f"  {'TOKEN':<40} {'LABEL':<12} {'NORM':<30} {'CONF':<6} Y")
    print(f"  {'-'*100}")
    for t in labeled:
        print(f"  {t['token']:<40} {t['label']:<12} {t['norm']:<30} "
              f"{t['conf']:.2f}  y={t['y1']}")