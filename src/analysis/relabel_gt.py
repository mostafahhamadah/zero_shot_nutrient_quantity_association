"""
relabel_gt.py
=============
Recompute gt_label in an existing embedding_scores_*.csv WITHOUT rerunning
BGE-M3. The original labelling logic compared the whole OCR token against
GT nutrient words, which fails on multilingual slash-joined tokens like
"Magnesium/magnesium/Magnesio/Magnesium/Magnesio/Magnezij" — the substring
length-ratio gate rejected them as non-nutrients.

This relabeller splits the OCR token on the same delimiters used by the
GT-builder, filters stopwords, then matches sub-word against GT. The
scores in the CSV are untouched.

USAGE
-----
    python src/analysis/relabel_gt.py \
        --scores-csv outputs/threshold_sweep/embedding_scores_embedding_only.csv \
        --annotations data/annotations

Then re-run the sweep on the fixed CSV. The script writes
    embedding_scores_embedding_only.fixed.csv
alongside the original.
"""

from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd


STOPWORDS: Set[str] = {
    "davon", "der", "die", "das", "von", "und", "oder",
    "gesamt", "ges",
    "of", "which", "and", "or", "the",
    "de", "du", "la", "le", "les", "del", "delle", "con",
    "total", "free",
}
MIN_WORD_LEN = 3


def _fold(s: str) -> str:
    return (s.lower()
             .replace("ß", "ss")
             .replace("ä", "a").replace("ö", "o").replace("ü", "u")
             .replace("é", "e").replace("è", "e")
             .replace("á", "a").replace("à", "a")
             .replace("í", "i").replace("ú", "u").replace("ý", "y"))


_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9äöüßÄÖÜéèáàíúýÉÈ]+")


def _split_words(phrase: str) -> List[str]:
    return [_fold(w) for w in _TOKEN_SPLIT_RE.split(phrase) if w.strip()]


def build_gt(annotations_dir: str) -> Dict[str, Set[str]]:
    per_image: Dict[str, Set[str]] = {}
    for jf in sorted(Path(annotations_dir).glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[gt] WARN {jf.name}: {e}")
            continue
        raw_id = Path(data.get("image_id", jf.stem + ".jpeg")).name
        image_id = Path(raw_id).stem + Path(raw_id).suffix.lower()
        words: Set[str] = set()
        for n in data.get("nutrients", []):
            for w in _split_words(str(n.get("nutrient", ""))):
                if w in STOPWORDS or len(w) < 2:
                    continue
                words.add(w)
        per_image[image_id] = words
    return per_image


def is_nutrient_token(token_text: str, gt_words: Set[str]) -> int:
    """
    Corrected: split OCR token into sub-words, then match per sub-word.
    """
    if not token_text or not gt_words:
        return 0
    sub_words = [w for w in _split_words(token_text)
                 if w not in STOPWORDS and len(w) >= 2]
    if not sub_words:
        return 0

    # 1) Direct sub-word match
    for w in sub_words:
        if w in gt_words:
            return 1

    # 2) Per-sub-word substring (OCR fragments like "magnesi" → "magnesium")
    for w in sub_words:
        if len(w) < MIN_WORD_LEN:
            continue
        for gw in gt_words:
            if len(gw) < MIN_WORD_LEN:
                continue
            if w in gw or gw in w:
                shorter, longer = sorted([len(w), len(gw)])
                if shorter / longer >= 0.6:
                    return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv",  required=True)
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--out",         default=None,
                    help="Output path. Default: <input>.fixed.csv")
    args = ap.parse_args()

    csv_in = Path(args.scores_csv)
    csv_out = Path(args.out) if args.out else csv_in.with_suffix(".fixed.csv")

    df = pd.read_csv(csv_in)
    print(f"[relabel] loaded {len(df)} rows from {csv_in}")

    gt = build_gt(args.annotations)
    print(f"[relabel] GT nutrient words for {len(gt)} images")

    old_label = df["gt_label"].astype(int).values.copy()
    new_label = df.apply(
        lambda r: is_nutrient_token(str(r["token"]), gt.get(str(r["image_id"]), set())),
        axis=1,
    ).astype(int).values

    df["gt_label_old"] = old_label
    df["gt_label"]     = new_label

    flipped_to_pos = int(((old_label == 0) & (new_label == 1)).sum())
    flipped_to_neg = int(((old_label == 1) & (new_label == 0)).sum())
    n_pos_old, n_pos_new = int(old_label.sum()), int(new_label.sum())

    print(f"[relabel] OLD: positives={n_pos_old}  negatives={len(df) - n_pos_old}")
    print(f"[relabel] NEW: positives={n_pos_new}  negatives={len(df) - n_pos_new}")
    print(f"[relabel] flipped 0→1: {flipped_to_pos}  (mislabelled negatives recovered)")
    print(f"[relabel] flipped 1→0: {flipped_to_neg}")

    df.to_csv(csv_out, index=False)
    print(f"[relabel] saved → {csv_out}")

    # Sample of newly-recovered positives so user can spot-check
    if flipped_to_pos:
        sample = df[(df.gt_label_old == 0) & (df.gt_label == 1)].head(20)
        print("\n[relabel] sample of 0→1 flips (first 20):")
        for _, row in sample.iterrows():
            print(f"  {str(row['image_id']):<14}  {str(row['token'])[:80]}")


if __name__ == "__main__":
    main()