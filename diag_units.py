import sys
sys.path.insert(0, '.')

from src.ocr.ocr_runner import run_ocr_on_image
from src.utils.ocr_corrector import OCRCorrector
from src.classification.semantic_classifier import SemanticClassifier
from src.graph.graph_constructor import GraphConstructor

corrector   = OCRCorrector()
classifier  = SemanticClassifier(confidence_threshold=0.30)
constructor = GraphConstructor()

tokens    = run_ocr_on_image('data/raw/45.jpeg')
corrected = corrector.correct_all(tokens)
labeled   = classifier.classify_all(corrected)
graph     = constructor.build(labeled)

print('=== UNIT nodes ===')
for n in graph['nodes']:
    if n['label'] == 'UNIT':
        print(f"  id={n['id']} token={n['token']} cy={n['cy']} x1={n['x1']}")

print()
print('=== QUANTITY nodes ===')
for n in graph['nodes']:
    if n['label'] == 'QUANTITY':
        print(f"  id={n['id']} token={n['token']} cy={n['cy']} x1={n['x1']}")

print()
print('=== NUTRIENT nodes ===')
for n in graph['nodes']:
    if n['label'] == 'NUTRIENT':
        print(f"  id={n['id']} token={n['token'][:40]} cy={n['cy']} x1={n['x1']}")

print()
print('=== SAME_ROW edges involving UNIT nodes ===')
unit_ids = {n['id'] for n in graph['nodes'] if n['label'] == 'UNIT'}
node_map = {n['id']: n for n in graph['nodes']}
for e in graph['edges']:
    if e['type'] == 'SAME_ROW' and (e['src'] in unit_ids or e['dst'] in unit_ids):
        src = node_map.get(e['src'], {})
        dst = node_map.get(e['dst'], {})
        print(f"  {src.get('token','?')[:20]:<22} ({src.get('label','?')}) "
              f"-> {dst.get('token','?')[:20]:<22} ({dst.get('label','?')})")