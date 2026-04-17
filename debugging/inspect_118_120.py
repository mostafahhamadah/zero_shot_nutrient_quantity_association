"""Inspect 118/119 qty failures and 120 context failures."""
import sys, os, csv
sys.path.insert(0, '.')

from src.ocr.paddleocr_runner import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
import importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_clf = _load_module('clf', os.path.join('src', 'classification',
    'experiment_01_final_semantic_classifier.py'))
classifier = _clf.SemanticClassifier(confidence_threshold=0.30)

# 118.jpeg - show tokens with positions to understand row assignment
for img_name in ['118.jpeg', '120.jpeg']:
    img_path = os.path.join('data/raw', img_name)
    if not os.path.exists(img_path):
        continue

    tokens = run_ocr_on_image(img_path)
    corrected, _ = correct_tokens(tokens, return_log=True)
    labeled = classifier.classify_all(corrected)

    # Only show NUTRIENT, QUANTITY, UNIT, CONTEXT tokens
    relevant = [t for t in labeled if t.get('label') in ('NUTRIENT','QUANTITY','UNIT','CONTEXT')]

    print(f"\n{'='*100}")
    print(f"  {img_name} - {len(relevant)} relevant tokens (sorted by cy)")
    print(f"{'='*100}")
    print(f"  {'LABEL':<10} {'TOKEN':<35} {'CX':>6} {'CY':>6} {'X1':>5}-{'X2':<5} {'Y1':>5}-{'Y2':<5}")
    print(f"  {'-'*10} {'-'*35} {'-'*6} {'-'*6} {'-'*11} {'-'*11}")

    relevant.sort(key=lambda t: (t.get('cy',0), t.get('cx',0)))
    for t in relevant:
        label = t.get('label','')
        tok = t.get('token','')[:34]
        cx = t.get('cx',0)
        cy = t.get('cy',0)
        print(f"  {label:<10} {tok:<35} {cx:6.0f} {cy:6.0f} {t.get('x1',0):5}-{t.get('x2',0):<5} {t.get('y1',0):5}-{t.get('y2',0):<5}")