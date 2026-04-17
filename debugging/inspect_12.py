"""Inspect 12.png tokens to find why nutrients aren't detected."""
import sys, os
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

tokens = run_ocr_on_image('data/raw/12.png')
corrected, _ = correct_tokens(tokens, return_log=True)
labeled = classifier.classify_all(corrected)

print(f"12.png — {len(labeled)} tokens\n")
print(f"  {'LABEL':<10} {'TOKEN':<60} {'CX':>5} {'CY':>5}")
print(f"  {'-'*10} {'-'*60} {'-'*5} {'-'*5}")
for t in sorted(labeled, key=lambda x: (x.get('cy',0), x.get('cx',0))):
    label = t.get('label','')
    tok = t.get('token','')[:59]
    # Mark potential nutrients classified wrong
    marker = '  '
    if label in ('UNKNOWN','NOISE') and any(kw in t.get('token','').lower() for kw in
        ['vet', 'fett', 'fat', 'grasse', 'koolhydr', 'glucid', 'carb', 'kohlen',
         'eiwit', 'protein', 'zout', 'salt', 'salz', 'sel', 'suiker', 'sugar',
         'zucker', 'sucre', 'verzadig', 'saturate', 'fibre', 'faser']):
        marker = '! '
    print(f"{marker}{label:<10} {tok:<60} {t.get('cx',0):5.0f} {t.get('cy',0):5.0f}")