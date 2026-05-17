"""
collect_embedding_scores.py
===========================
Phase 1 of embedding-classifier threshold/margin tuning.

WHAT IT DOES
------------
For every token in the dataset that reaches the embedding pathway
(i.e. tokens that aren't deterministically labelled by the rule
classifier), dump:

    image_id, token, context, nutrient_score, second_best_score,
    quantity_score, unit_score, context_score, noise_score,
    rule_label, gt_label

GT_LABEL is a token-level binary label derived from the per-image
JSON annotations:
    1  → token is a meaningful word inside any GT nutrient phrase
    0  → token is not part of any GT nutrient phrase

This CSV is the input for sweep_threshold_margin.py, which produces
the plots and picks (threshold, margin) by empirical F1.

USAGE
-----
    python src/analysis/collect_embedding_scores.py \
        --mode embedding_only \
        --out outputs/threshold_sweep

    python src/analysis/collect_embedding_scores.py \
        --mode hybrid

NOTES
-----
- Slow stage: BGE-M3 encoding over ~thousands of tokens. Run once.
- Re-uses the existing EmbeddingSemanticClassifier.classify_all so
  the scoring pipeline is bit-identical to production. Threshold and
  margin passed here are irrelevant to score collection — we only
  read the raw scores dict from each result.
"""

from __future__ import annotations

import sys, csv, json, argparse, re
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, '.')

from src.ocr.paddleocr_runner               import run_ocr_on_image
from src.utils.paddleocr_corrector          import correct_tokens
from src.classification.embedding_semantic_classifier import EmbeddingSemanticClassifier


# ── GT token-level label construction ─────────────────────────────────────────

# Words that appear inside multi-word nutrient phrases but are NOT themselves
# nutrients. Excluded from positives so the classifier isn't penalised for
# correctly rejecting them.
STOPWORDS: Set[str] = {
    # German function/structural words
    "davon", "der", "die", "das", "von", "und", "oder",
    "gesamt", "ges",                    # qualifiers, not nutrient names
    # English
    "of", "which", "and", "or", "the",
    # French / Italian / Spanish / NL function words
    "de", "du", "la", "le", "les", "del", "delle", "con",
    # Common qualifying tokens
    "total", "free",
}

MIN_WORD_LEN = 3   # 1- and 2-char fragments (e.g. "B", "B6") handled separately


def _fold(s: str) -> str:
    """Lowercase + fold German/accented chars. Mirrors llm_evaluator._fold_german."""
    return (s.lower()
             .replace("ß", "ss")
             .replace("ä", "a").replace("ö", "o").replace("ü", "u")
             .replace("é", "e").replace("è", "e")
             .replace("á", "a").replace("à", "a")
             .replace("í", "i").replace("ú", "u").replace("ý", "y"))


_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9äöüßÄÖÜéèáàíúýÉÈ]+")


def _split_words(phrase: str) -> List[str]:
    """Split a nutrient phrase into normalised word tokens."""
    return [_fold(w) for w in _TOKEN_SPLIT_RE.split(phrase) if w.strip()]


def build_gt_nutrient_words(annotations_dir: str) -> Dict[str, Set[str]]:
    """
    Return {image_id: {set of nutrient-bearing words}}.

    A token is a positive if its folded form is in this set, OR it
    matches a longer nutrient word as a substring (e.g. OCR fragment
    "magnesi" → "magnesium").
    """
    per_image: Dict[str, Set[str]] = {}
    for jf in sorted(Path(annotations_dir).glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[gt] WARN failed to load {jf.name}: {e}")
            continue
        raw_id   = Path(data.get("image_id", jf.stem + ".jpeg")).name
        image_id = Path(raw_id).stem + Path(raw_id).suffix.lower()

        words: Set[str] = set()
        for n in data.get("nutrients", []):
            for w in _split_words(str(n.get("nutrient", ""))):
                if w in STOPWORDS:
                    continue
                # Keep short codes like "b12", "b6", "k2" — they're real
                # nutrient identifiers. But drop length-1 ("b").
                if len(w) < 2:
                    continue
                words.add(w)
        per_image[image_id] = words
    return per_image


def is_nutrient_token(token_text: str, gt_words: Set[str]) -> int:
    """1 if this token is part of any GT nutrient phrase, else 0."""
    if not token_text or not gt_words:
        return 0
    t = _fold(token_text.strip())
    if len(t) < 2:
        return 0
    if t in gt_words:
        return 1
    # Substring matching (handles OCR fragments and German compounds)
    if len(t) >= MIN_WORD_LEN:
        for w in gt_words:
            if len(w) >= MIN_WORD_LEN and (t in w or w in t):
                # Length sanity: shorter must be at least 60% of longer
                shorter, longer = sorted([len(t), len(w)])
                if shorter / longer >= 0.6:
                    return 1
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        default="embedding_only",
                    choices=["embedding_only", "hybrid"],
                    help="Embedding classifier mode")
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--raw",         default="data/raw")
    ap.add_argument("--out",         default="outputs/threshold_sweep")
    ap.add_argument("--images",      default=None,
                    help="Comma-separated stems to limit to (e.g. '1,12,59')")
    # Threshold/margin here only affect the label field in the CSV
    # ('would_be_nutrient'). They do NOT affect raw scores.
    ap.add_argument("--threshold",   type=float, default=0.40)
    ap.add_argument("--margin",      type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"embedding_scores_{args.mode}.csv"

    # ── Discover images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".tif"}
    raw  = Path(args.raw)
    if args.images:
        wanted = {s.strip() for s in args.images.split(",") if s.strip()}
        files = [f for f in raw.iterdir()
                 if f.suffix.lower() in exts and f.stem in wanted]
    else:
        files = [f for f in raw.iterdir() if f.suffix.lower() in exts]
    files.sort(key=lambda f: (int(f.stem) if f.stem.isdigit() else 1e9, f.name))

    print(f"[collect] mode={args.mode}  images={len(files)}  out={csv_path}")

    # ── GT
    gt_words_per_image = build_gt_nutrient_words(args.annotations)
    total_gt_words = sum(len(v) for v in gt_words_per_image.values())
    print(f"[collect] GT: {len(gt_words_per_image)} images, "
          f"{total_gt_words} nutrient-word entries")

    # ── Classifier (loads BGE-M3 once)
    clf = EmbeddingSemanticClassifier(
        mode=args.mode,
        nutrient_threshold=args.threshold,
        margin=args.margin,
    )

    # ── Iterate
    rows = []
    skipped = 0
    for img_path in files:
        image_key = img_path.stem + img_path.suffix.lower()
        gt_words  = gt_words_per_image.get(image_key, set())
        if not gt_words:
            skipped += 1
        try:
            tokens         = run_ocr_on_image(str(img_path))
            corrected, _   = correct_tokens(tokens, return_log=True)
            results        = clf.classify_all(corrected)
        except Exception as e:
            print(f"[collect] ERR {image_key}: {e}")
            continue

        n_emb = 0
        for r in results:
            if r.get("classification_method") != "embedding":
                continue
            scores = r.get("embedding_scores", {}) or {}
            if not scores:
                continue
            nutrient_score = float(scores.get("NUTRIENT", 0.0))
            other_scores   = {k: v for k, v in scores.items() if k != "NUTRIENT"}
            second_best    = float(max(other_scores.values()) if other_scores else 0.0)
            second_best_cat = max(other_scores, key=other_scores.get) \
                              if other_scores else ""

            token_text = r.get("token", "") or r.get("norm", "")
            gt_label   = is_nutrient_token(token_text, gt_words)

            rows.append({
                "image_id":          image_key,
                "token":             token_text,
                "context":           r.get("embedding_context", "") or "",
                "rule_label":        r.get("rule_label", "") or
                                     r.get("label_before_emb", ""),
                "nutrient_score":    round(nutrient_score, 4),
                "second_best_score": round(second_best, 4),
                "second_best_cat":   second_best_cat,
                "quantity_score":    round(float(scores.get("QUANTITY", 0.0)), 4),
                "unit_score":        round(float(scores.get("UNIT",     0.0)), 4),
                "context_score":     round(float(scores.get("CONTEXT",  0.0)), 4),
                "noise_score":       round(float(scores.get("NOISE",    0.0)), 4),
                "margin_actual":     round(nutrient_score - second_best, 4),
                "gt_label":          gt_label,
                "current_pred":      int(r.get("label", "") == "NUTRIENT"),
            })
            n_emb += 1

        print(f"  {image_key:<14s} embedded={n_emb:>3d}  "
              f"gt_words={len(gt_words):>3d}")

    # ── Save
    if not rows:
        print("[collect] No embedding-pathway tokens found. Nothing to save.")
        return

    keys = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(r["gt_label"] for r in rows)
    n_neg = len(rows) - n_pos
    print(f"\n[collect] DONE  {len(rows)} rows  "
          f"({n_pos} positives, {n_neg} negatives, "
          f"{skipped} images had no GT nutrient set)")
    print(f"[collect] Saved → {csv_path}")


if __name__ == "__main__":
    main()