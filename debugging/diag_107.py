import sys
sys.path.insert(0, '.')
from src.ocr.ocr_runner import run_ocr_on_image

tokens = run_ocr_on_image('data/raw/107.jpeg')
for t in sorted(tokens, key=lambda x: x['y1']):
    print(f"{t['conf']:.3f}  {t['token']}")