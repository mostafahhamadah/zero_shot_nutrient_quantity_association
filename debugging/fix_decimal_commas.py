import re, shutil
from pathlib import Path

CSV = Path("data/annotations/gold_annotations.csv")
BACKUP = Path("data/annotations/gold_annotations_backup.csv")

# Safety backup
shutil.copy(CSV, BACKUP)
print(f"Backup saved to {BACKUP}")

text = CSV.read_text(encoding="utf-8")

# Replace European decimal commas inside numeric/threshold fields only.
# Pattern: a digit (or >) followed by comma followed by digit — inside a CSV field.
# We replace  \d,\d  with  \d.\d  but ONLY when NOT at end of field
# (i.e. the comma is between two digits, not a field separator)
fixed = re.sub(r'(?<=[\d<>])(\s*),(\s*)(?=\d)', r'.\2', text)

CSV.write_text(fixed, encoding="utf-8")
print("Fixed. Lines changed:")
orig_lines  = text.splitlines()
fixed_lines = fixed.splitlines()
changed = [(i+1, o, f) for i,(o,f) in enumerate(zip(orig_lines, fixed_lines)) if o != f]
for lineno, orig, fix in changed:
    print(f"  Line {lineno}: {orig.strip()[:80]}")
    print(f"         -> {fix.strip()[:80]}")
print(f"\nTotal lines changed: {len(changed)}")
