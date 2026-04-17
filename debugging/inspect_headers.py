"""Show ALL tokens near the top of failing images to find missing context headers."""
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

IMAGES = ['59.png', '33.png', '90.png', '91.png', '2.png', '104.jpeg']

for img_name in IMAGES:
    img_path = os.path.join('data/raw', img_name)
    if not os.path.exists(img_path):
        continue

    tokens = run_ocr_on_image(img_path)
    corrected, _ = correct_tokens(tokens, return_log=True)
    labeled = classifier.classify_all(corrected)

    # Show tokens in the top 25% of the image (header region)
    max_y = max(t.get('y2', 0) for t in labeled) if labeled else 500
    header_cutoff = max_y * 0.25

    header_tokens = [t for t in labeled if t.get('cy', 0) < header_cutoff]
    header_tokens.sort(key=lambda t: (t.get('cy', 0), t.get('cx', 0)))

    print(f"\n{'='*100}")
    print(f"  {img_name} - header region tokens (cy < {header_cutoff:.0f})")
    print(f"{'='*100}")
    for t in header_tokens:
        marker = '  '
        if t.get('label') == 'CONTEXT':
            marker = 'C '
        elif t.get('label') in ('UNKNOWN', 'NOISE') and any(kw in t.get('token','').lower()
            for kw in ['serv', 'portion', 'port', 'dose', 'drink', 'riegel',
                       'kapsel', 'stick', 'tube', 'esslof', 'scoop', 'piece',
                       'pro ', 'per ', 'je ', 'pour ']):
            marker = '! '
        print(f"  {marker}{t.get('label',''):<10} {t.get('token','')[:50]:<52} "
              f"cx={t.get('cx',0):6.0f} cy={t.get('cy',0):6.0f}")