"""
test_path.py — run this from the project root:
    python test_path.py
"""
import os
from pathlib import Path

print("=" * 60)
print(f"__file__     : {os.path.abspath(__file__)}")
print(f"PROJECT_ROOT : {Path(os.path.dirname(os.path.abspath(__file__)))}")
print(f"DATA_RAW     : {Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'raw'}")
print(f"CWD          : {os.getcwd()}")

data_raw = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "raw"
print(f"Exists       : {data_raw.exists()}")

if data_raw.exists():
    images = sorted(p.name for p in data_raw.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    print(f"Images found : {len(images)}")
    for img in images[:5]:
        print(f"  {img}")
else:
    print("FOLDER DOES NOT EXIST")
print("=" * 60)