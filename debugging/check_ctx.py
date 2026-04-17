"""Inspect OCR tokens and classifier labels for context-failing images."""
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
SemanticClassifier = _clf.SemanticClassifier

classifier = SemanticClassifier(confidence_threshold=0.30)

IMAGES = ['6.png', '17.png', '64.png', '120.jpeg']
RAW_DIR = 'data/raw'

for img_name in IMAGES:
    img_path = os.path.join(RAW_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"\n{'='*70}\n  {img_name} — FILE NOT FOUND\n{'='*70}")
        continue

    tokens = run_ocr_on_image(img_path)
    corrected, _ = correct_tokens(tokens, return_log=True)
    labeled = classifier.classify_all(corrected)

    print(f"\n{'='*70}")
    print(f"  {img_name} — {len(labeled)} tokens")
    print(f"{'='*70}")
    print(f"  {'TOKEN':<40} {'LABEL':<10} {'NORM':<25} {'CONF':.4}")
    print(f"  {'-'*40} {'-'*10} {'-'*25} {'-'*6}")

    for t in labeled:
        tok = t.get('token', '')[:39]
        label = t.get('label', '')
        norm = t.get('norm', '')[:24]
        conf = t.get('conf', 0)
        # Highlight CONTEXT and UNKNOWN tokens
        marker = '  '
        if label == 'CONTEXT':
            marker = '✓ '
        elif label == 'UNKNOWN':
            marker = '? '
        elif label == 'NOISE' and any(kw in tok.lower() for kw in
            ['per', 'pro', 'je', 'portion', 'serving', 'dose', 'daily',
             'kapsel', 'tablette', 'stick', 'piece', 'bar', 'ampulle',
             'drink', 'pulver', 'riegel', 'tagesdosis']):
            marker = '! '  # potential context classified as NOISE
        print(f"{marker}{tok:<40} {label:<10} {norm:<25} {conf:.3f}")