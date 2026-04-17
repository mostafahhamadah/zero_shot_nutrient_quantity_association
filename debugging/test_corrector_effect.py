"""
test_corrector_effect.py
========================
Analyses the effect of paddleocr_corrector on your real OCR data.

Reads  : results/ocr_inspection/raw_ocr_tokens.csv
         (produced by inspect_paddleocr.py — no pipeline needed)
Runs   : correct_tokens() on every image's token list
Outputs:
  - Console report  : rule hit counts, per-image stats, before/after examples
  - corrector_effect_report.csv : per-image summary
  - corrector_effect_examples.txt : before → after examples for each rule

Usage:
    python test_corrector_effect.py
    python test_corrector_effect.py --csv results/ocr_inspection/raw_ocr_tokens.csv
    python test_corrector_effect.py --show-examples 20
"""

import sys, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, '.')

from src.utils.paddleocr_corrector import correct_tokens

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--csv',
    default='results/ocr_inspection/raw_ocr_tokens.csv',
    help='Path to raw_ocr_tokens.csv from inspect_paddleocr.py')
parser.add_argument('--show-examples', type=int, default=10,
    help='Max before→after examples to print per rule (default 10)')
parser.add_argument('--out-dir',
    default='results/ocr_inspection',
    help='Where to save the CSV and TXT report files')
args = parser.parse_args()

CSV_PATH   = Path(args.csv)
OUT_DIR    = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EFFECT_CSV = OUT_DIR / 'corrector_effect_report.csv'
EXAMPLES_TXT = OUT_DIR / 'corrector_effect_examples.txt'

# ── Load raw_ocr_tokens.csv ───────────────────────────────────────────────────

if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} not found.")
    print("Run inspect_paddleocr.py first to generate raw_ocr_tokens.csv")
    sys.exit(1)

# Group tokens by image_id preserving original order
images: dict = {}          # image_id → list of raw token dicts
with open(CSV_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img = row['image_id']
        if img not in images:
            images[img] = []
        # Reconstruct the token dict expected by correct_tokens()
        images[img].append({
            'token': row['token'],
            'conf':  float(row.get('conf', 0.9)),
            'x1':    float(row.get('x1', 0)),
            'y1':    float(row.get('y1', 0)),
            'x2':    float(row.get('x2', 100)),
            'y2':    float(row.get('y2', 20)),
            'cx':    float(row.get('cx', 50)),
            'cy':    float(row.get('cy', 10)),
        })

print(f"\n{'='*65}")
print(f"  INPUT  : {CSV_PATH}")
print(f"  Images : {len(images)}")
print(f"  Tokens : {sum(len(v) for v in images.values())}")
print(f"{'='*65}\n")

# ── Run corrector on every image ──────────────────────────────────────────────

global_rule_counter  = Counter()           # rule_id → total hits
global_rule_examples = defaultdict(list)   # rule_id → list of (original, corrected)
per_image_stats      = []                  # one row per image for CSV

total_in  = 0
total_out = 0
total_changed = 0
total_splits  = 0

for img_id, tokens in images.items():
    corrected, log = correct_tokens(tokens, return_log=True)

    n_in      = len(tokens)
    n_out     = len(corrected)
    n_changed = len(log)
    n_splits  = sum(1 for e in log if len(e['corrected']) > 1)

    total_in      += n_in
    total_out     += n_out
    total_changed += n_changed
    total_splits  += n_splits

    # Count rules fired in this image
    img_rules = Counter()
    for entry in log:
        for rule in entry['rules_fired']:
            img_rules[rule] += 1
            global_rule_counter[rule] += 1

        # Collect examples (capped per rule)
        for rule in entry['rules_fired']:
            ex_list = global_rule_examples[rule]
            if len(ex_list) < args.show_examples:
                ex_list.append({
                    'image_id': img_id,
                    'original': entry['original'],
                    'corrected': entry['corrected'],
                })

    per_image_stats.append({
        'image_id':   img_id,
        'tokens_in':  n_in,
        'tokens_out': n_out,
        'changed':    n_changed,
        'splits':     n_splits,
        'pct_changed': f'{n_changed/max(n_in,1)*100:.1f}',
        **{rule: img_rules.get(rule, 0) for rule in [
            'C1_decimal_sep',
            'C2_fused_int', 'C3_fused_dec',
            'C6_energie_split',
            'C7_context_norm',
            'C8_Ug_to_µg', 'C8_ug_to_µg',
            'C9_m9_to_mg', 'C9_pg_to_µg',
            'C10_border_strip',
        ]}
    })

# ── Console report ────────────────────────────────────────────────────────────

print(f"{'─'*65}")
print(f"  OVERALL CORRECTION STATS")
print(f"{'─'*65}")
print(f"  Input tokens          : {total_in}")
print(f"  Output tokens         : {total_out}  (+{total_out - total_in} from splits)")
print(f"  Tokens changed        : {total_changed}  ({total_changed/max(total_in,1)*100:.1f}% of input)")
print(f"  Tokens split into 2   : {total_splits}")
print()

print(f"{'─'*65}")
print(f"  RULE HIT COUNTS  (across all {len(images)} images)")
print(f"{'─'*65}")

RULE_DESCRIPTIONS = {
    'C1_decimal_sep':   'C1   Comma/apostrophe → decimal period',
    'C2_fused_int':     'C2   Fused integer qty+unit split',
    'C3_fused_dec':     'C3   Fused decimal qty+unit split',
    'C6_energie_split': 'C6   ENERGIE+kJ label split',
    'C7_context_norm':  'C7   Context token normalised',
    'C8_Ug_to_µg':      'C8   Ug → µg',
    'C8_ug_to_µg':      'C8   ug → µg',
    'C9_m9_to_mg':      'C9   m9 → mg',
    'C9_pg_to_µg':      'C9   pg → µg',
    'C10_border_strip': 'C10  Border chars stripped',
}

for rule_id, desc in RULE_DESCRIPTIONS.items():
    count = global_rule_counter.get(rule_id, 0)
    bar   = '█' * min(count, 50)
    print(f"  {desc:<40}  {count:>5}  {bar}")

print()

# ── Per-image table ───────────────────────────────────────────────────────────

print(f"{'─'*65}")
print(f"  PER-IMAGE SUMMARY  (images with most corrections first)")
print(f"{'─'*65}")
print(f"  {'Image':<14}  {'In':>5}  {'Out':>5}  {'Chng':>5}  {'Spl':>4}  {'%':>5}  Top rule")

sorted_stats = sorted(per_image_stats, key=lambda r: int(r['changed']), reverse=True)
for row in sorted_stats:
    # Find the top rule for this image
    img_rules_for_row = {k: v for k, v in row.items()
                         if k in RULE_DESCRIPTIONS and isinstance(v, int) and v > 0}
    top_rule = max(img_rules_for_row, key=img_rules_for_row.get) if img_rules_for_row else '—'
    top_desc = RULE_DESCRIPTIONS.get(top_rule, top_rule).split()[1] if top_rule != '—' else '—'
    print(f"  {row['image_id']:<14}  {row['tokens_in']:>5}  {row['tokens_out']:>5}  "
          f"{row['changed']:>5}  {row['splits']:>4}  {row['pct_changed']:>5}%  {top_desc}")

print()

# ── Before → after examples per rule ─────────────────────────────────────────

print(f"{'─'*65}")
print(f"  BEFORE → AFTER EXAMPLES  (up to {args.show_examples} per rule)")
print(f"{'─'*65}")

for rule_id, desc in RULE_DESCRIPTIONS.items():
    examples = global_rule_examples.get(rule_id, [])
    if not examples:
        continue
    print(f"\n  {desc}  [{global_rule_counter.get(rule_id, 0)} hits total]")
    for ex in examples[:args.show_examples]:
        orig = ex['original']
        corr = ex['corrected']
        img  = ex['image_id']
        if len(corr) == 1:
            print(f"    {img:<12}  {orig!r:>25}  →  {corr[0]!r}")
        else:
            print(f"    {img:<12}  {orig!r:>25}  →  {corr[0]!r}  +  {corr[1]!r}  [SPLIT]")

print()

# ── Save per-image CSV ────────────────────────────────────────────────────────

fieldnames = ['image_id', 'tokens_in', 'tokens_out', 'changed', 'splits', 'pct_changed',
              'C1_decimal_sep', 'C2_fused_int', 'C3_fused_dec', 'C6_energie_split',
              'C7_context_norm', 'C8_Ug_to_µg', 'C8_ug_to_µg',
              'C9_m9_to_mg', 'C9_pg_to_µg', 'C10_border_strip']

with open(EFFECT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(per_image_stats)

# ── Save examples TXT ─────────────────────────────────────────────────────────

with open(EXAMPLES_TXT, 'w', encoding='utf-8') as f:
    f.write('CORRECTOR BEFORE→AFTER EXAMPLES\n')
    f.write('='*65 + '\n\n')
    for rule_id, desc in RULE_DESCRIPTIONS.items():
        examples = global_rule_examples.get(rule_id, [])
        if not examples:
            continue
        f.write(f'{desc}\n')
        f.write('-'*50 + '\n')
        for ex in examples:
            orig = ex['original']
            corr = ex['corrected']
            img  = ex['image_id']
            if len(corr) == 1:
                f.write(f"  {img:<12}  {orig!r:>25}  →  {corr[0]!r}\n")
            else:
                f.write(f"  {img:<12}  {orig!r:>25}  →  {corr[0]!r}  +  {corr[1]!r}  [SPLIT]\n")
        f.write('\n')

print(f"{'─'*65}")
print(f"  Saved: {EFFECT_CSV}")
print(f"  Saved: {EXAMPLES_TXT}")
print(f"{'─'*65}\n")

# ── Quick sanity check: tokens that should NOT have been changed ───────────────

print(f"{'─'*65}")
print(f"  SANITY CHECK — verify protected tokens were NOT split")
print(f"{'─'*65}")

protected = {'100g', '100 g', '100ml', '100 ml'}
violations = []
for img_id, tokens in images.items():
    corrected, log = correct_tokens(tokens, return_log=True)
    for entry in log:
        if entry['original'] in protected:
            violations.append((img_id, entry['original'], entry['corrected']))

if violations:
    print(f"  ✗  {len(violations)} protected token(s) were incorrectly changed:")
    for img, orig, corr in violations:
        print(f"     {img}  {orig!r} → {corr}")
else:
    print(f"  ✓  All protected tokens (100g / 100 g / 100ml / 100 ml) passed through unchanged.")

print()

# ── Nutrients that should be untouched ────────────────────────────────────────

print(f"{'─'*65}")
print(f"  SANITY CHECK — common nutrient names should be untouched")
print(f"{'─'*65}")

KNOWN_NUTRIENTS = [
    'Magnesium', 'Calcium', 'Energie', 'Energie', 'Fett', 'Kohlenhydrate',
    'Ballaststoffe', 'Eiweiss', 'Protein', 'Natrium', 'Kalium', 'Salz',
    'Vitamin B1', 'Vitamin B2', 'Vitamin B6', 'Vitamin B12', 'Vitamin C',
    'Vitamin D', 'Vitamin E', 'Niacin', 'Folsaure', 'Biotin',
]

nutrient_violations = []
for img_id, tokens in images.items():
    for tok in tokens:
        if tok['token'] in KNOWN_NUTRIENTS:
            result, log = correct_tokens([tok], return_log=True)
            if log:  # something changed
                nutrient_violations.append(
                    (img_id, tok['token'], result[0]['token'], log[0]['rules_fired']))

if nutrient_violations:
    print(f"  ✗  {len(nutrient_violations)} nutrient token(s) were changed (investigate):")
    for img, orig, corr, rules in nutrient_violations[:10]:
        print(f"     {img}  {orig!r} → {corr!r}  rules={rules}")
else:
    print(f"  ✓  All checked nutrient names passed through unchanged.")

print(f"\n{'='*65}\n")