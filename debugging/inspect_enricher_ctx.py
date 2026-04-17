"""Show column_context_id assignments from enricher for context-failing images."""
import sys, os, json
sys.path.insert(0, '.')

from src.ocr.paddleocr_runner import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
from src.utils.token_enricher import TokenEnricher
import importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_clf = _load_module('clf', os.path.join('src', 'classification',
    'experiment_01_final_semantic_classifier.py'))
classifier = _clf.SemanticClassifier(confidence_threshold=0.30)
enricher = TokenEnricher()

# Top context-failure images from inspect_ctx_gap.py
IMAGES = ['59.png', '33.png', '90.png', '91.png']

for img_name in IMAGES:
    img_path = os.path.join('data/raw', img_name)
    if not os.path.exists(img_path):
        continue

    tokens = run_ocr_on_image(img_path)
    corrected, _ = correct_tokens(tokens, return_log=True)
    labeled = classifier.classify_all(corrected)
    enriched = enricher.enrich(labeled)

    # Show CONTEXT tokens and their column_context_id
    ctx_tokens = [t for t in enriched if t.get('label') == 'CONTEXT']
    qty_tokens = [t for t in enriched if t.get('label') == 'QUANTITY']

    print(f"\n{'='*90}")
    print(f"  {img_name}")
    print(f"{'='*90}")

    print(f"\n  CONTEXT tokens:")
    for t in ctx_tokens:
        print(f"    token={t['token'][:30]:<32} cx={t.get('cx',0):6.0f} cy={t.get('cy',0):6.0f} "
              f"col_ctx={t.get('column_context_id',''):<20} col_id={t.get('column_id',-1)}")

    print(f"\n  QUANTITY tokens with column_context_id (first 12):")
    for t in qty_tokens[:12]:
        print(f"    token={t['token'][:10]:<12} cx={t.get('cx',0):6.0f} cy={t.get('cy',0):6.0f} "
              f"col_ctx={str(t.get('column_context_id','')):<20} col_id={t.get('column_id',-1)} "
              f"stream={t.get('dosage_stream_id',-1)}")