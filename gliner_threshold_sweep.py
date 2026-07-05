"""
gliner_threshold_sweep.py
=========================
Threshold calibration for the GLiNER-biomed nutrient node.
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

WHAT THIS DOES (analogue of Methodology §4.3.4 for the GLiNER node)
-------------------------------------------------------------------
The GLiNER nutrient node (gliner_classifier.py) labels a nutrient
candidate NUTRIENT when a GLiNER span covering it scores at or above an
acceptance threshold.  This script *chooses* that threshold from data,
exactly as the embedding node's cosine threshold was chosen: it builds a
token-level dataset of nutrient decisions, sweeps the acceptance threshold,
and reports the precision / recall / F1 trade-off and the F1-optimal
operating point.

Two phases:

  PHASE A  (slow; needs PaddleOCR + GLiNER)
      For every image it runs the SAME upstream pipeline the experiment
      runner uses — OCR (paddleocr_runner.run_ocr_on_image) then the
      corrector (paddleocr_corrector_v2.correct_tokens, which applies C15)
      — and hands the
      corrected tokens to GLiNERSemanticClassifier.score_candidates(),
      which runs the shared cascade and returns the raw GLiNER score for
      every nutrient candidate.  The raw per-candidate scores are cached to
      gliner_candidate_scores.csv.  GLiNER scores the candidate's ACTUAL
      surface form — whatever the corrector produced — so the calibration
      reflects what the deployed classifier scores.

  PHASE B  (instant; needs only the cache + the gold CSV)
      Re-derives each candidate's ground-truth nutrient label from the
      cache, sweeps the acceptance threshold over [--tmin, --tmax] in
      --tstep increments, writes gliner_threshold_sweep.csv, plots
      threshold_sweep_gliner.png, and prints the F1-optimal threshold.

Because labelling lives in Phase B, editing the ground-truth rule
(is_true_nutrient) only requires re-running — NO --rebuild, NO OCR/GLiNER.
Pass --rebuild only to regenerate the scores themselves.

GROUND TRUTH  (the DE/EN bridge)
--------------------------------
The gold names in test_set_normalized.csv are canonical English
(Energy, Fat, Protein, Sugars, ...).  The v2 corrector applies C15
(apply_c15_normalise_names) at the end of correct_tokens, so candidates
normally reach this stage already in canonical English and match the gold
directly.  As a safety net — and so that caches built before C15 was
applied (German surface forms: Energie, Fett, Eiweiss, ...) can still be
re-labelled without --rebuild — is_true_nutrient ALSO runs each candidate
through C15 before matching.  This affects the LABEL ONLY; the cached
GLiNER score is always taken on the surface form the classifier scored.

Because this sweep uses the same cascade, the same images and the same
gold file as the embedding-node sweep, it should produce the SAME
candidate count and a comparable true/non split (§4.3.4: ~796 candidates,
462 true, 334 non-nutrient).  The split is printed; is_true_nutrient is the
single knob to reconcile it.

USAGE
-----
    python gliner_threshold_sweep.py                 # build cache, then sweep
    python gliner_threshold_sweep.py --tstep 0.005   # re-sweep + re-label cache (no OCR/GLiNER)
    python gliner_threshold_sweep.py --rebuild       # regenerate scores from scratch
    python gliner_threshold_sweep.py --images 1,101,118 --rebuild

Run from the project root (so that `src...` imports resolve).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, ".")  # match the experiment runner's import root

# ── Defaults (mirror run_graph_v2_experiment.py) ───────────────────────────────
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_CSV = "test_set_normalized.csv"
DEFAULT_OUT_DIR = "outputs/gliner_threshold_sweep"
DEFAULT_CONF_THRESH = 0.30
DEFAULT_CLASSIFIER_PATH = "src/classification/gliner_classifier.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

#: Shortest token allowed to drive a component match — guards against a
#: candidate matching on a 1–2 char fragment.  Vitamin codes (b1, c) are
#: never used as match keys: they are resolved by C15 into full canonical
#: names ("Vitamin C"), so specific vitamins compare by their full form.
_MIN_PIECE_LEN = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C15 canonicalisation  (reuse the pipeline's own map; fallback if absent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_c15() -> Optional[Callable[[list], list]]:
    """Locate apply_c15_normalise_names in the project's corrector module."""
    for modname in (
        "src.utils.paddleocr_corrector_v2",
        "src.utils.paddleOCR_corrector_v2",
        "src.utils.paddleocr_corrector",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(mod, "apply_c15_normalise_names", None)
        if callable(fn):
            return fn
    return None


_C15_FN: Optional[Callable[[list], list]] = _load_c15()

#: Compact multilingual → canonical-English fallback for the macronutrients
#: and common minerals, used only when C15 cannot be imported.  Keys are in
#: folded form (see _fold).  C15 (when present) is authoritative and richer.
_FALLBACK_CANON: Dict[str, str] = {
    # energy
    "energie": "Energy", "brennwert": "Energy",
    "valeur energetique": "Energy", "valore energetico": "Energy",
    "valor energetico": "Energy",
    # fat
    "fett": "Fat", "fette": "Fat", "graisses": "Fat", "lipides": "Fat",
    "matieres grasses": "Fat", "grassi": "Fat", "grasas": "Fat", "vetten": "Fat",
    # saturated fats
    "gesattigte fettsauren": "Saturated Fats",
    "davon gesattigte fettsauren": "Saturated Fats",
    "of which saturates": "Saturated Fats",
    "acides gras satures": "Saturated Fats",
    "grassi saturi": "Saturated Fats", "grasas saturadas": "Saturated Fats",
    "verzadigde vetzuren": "Saturated Fats",
    # carbohydrate
    "kohlenhydrate": "Carbohydrate", "glucides": "Carbohydrate",
    "carboidrati": "Carbohydrate", "hidratos de carbono": "Carbohydrate",
    "koolhydraten": "Carbohydrate",
    # sugars
    "zucker": "Sugars", "davon zucker": "Sugars", "zuckerarten": "Sugars",
    "sucres": "Sugars", "zuccheri": "Sugars", "azucares": "Sugars",
    "suikers": "Sugars",
    # fibre
    "ballaststoffe": "Fibre", "fibres": "Fibre", "fibra": "Fibre",
    "vezels": "Fibre",
    # protein
    "eiweiss": "Protein", "proteine": "Protein", "proteines": "Protein",
    "proteinas": "Protein", "eiwitten": "Protein",
    # salt / sodium
    "salz": "Salt", "sel": "Salt", "sale": "Salt", "sal": "Salt", "zout": "Salt",
    "natrium": "Sodium", "sodio": "Sodium",
    # minerals (DE → EN)
    "eisen": "Iron", "zink": "Zinc", "jod": "Iodine", "jodid": "Iodine",
    "selen": "Selenium", "kupfer": "Copper", "mangan": "Manganese",
    "kalzium": "Calcium", "kalium": "Potassium", "phosphor": "Phosphorus",
    "chlorid": "Chloride", "fluorid": "Fluoride", "chrom": "Chromium",
    "molybdan": "Molybdenum",
    # acids
    "folsaure": "Folic Acid", "pantothensaure": "Pantothenic Acid",
}


def _canonicalise_nutrient(text: str) -> str:
    """Map a nutrient surface form to its canonical English name.

    Prefers the pipeline's C15 (authoritative); falls back to the compact
    multilingual table above.  Returns the input unchanged when nothing
    matches (so already-English and unknown names pass through)."""
    if not text:
        return text
    # 1) C15 (authoritative): wrap the string as a one-token list.
    if _C15_FN is not None:
        try:
            out = _C15_FN([{"token": text}])
            cand = out[0].get("token", text) if out else text
            if cand and cand.lower() != text.lower():
                return cand
        except Exception:
            pass
    # 2) fallback exact (folded)
    f = _fold(text)
    if f in _FALLBACK_CANON:
        return _FALLBACK_CANON[f]
    # 3) fallback on the first slash/pipe segment
    seg = re.split(r"[\/|]", text)[0].strip()
    if seg and seg != text:
        fs = _fold(seg)
        if fs in _FALLBACK_CANON:
            return _FALLBACK_CANON[fs]
    return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Folding + ground-truth matching  (the editable ground-truth rule)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fold(s: str) -> str:
    """Lowercase, German fold (ß→ss, ä→a, ö→o, ü→u), strip accents, turn
    separators (/ | , ( ) - . :) into spaces, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = s.replace("ß", "ss").replace("ä", "a").replace("ö", "o").replace("ü", "u")
    s = "".join(c for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c))
    s = re.sub(r"[\/|,()\[\].\-–•:*]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_true_nutrient(candidate_text: str, gold_names: Set[str]) -> bool:
    """Decide whether a nutrient candidate corresponds to a gold nutrient.

    --- THIS IS THE GROUND-TRUTH RULE FOR THE SWEEP ---
    Both the candidate and each gold name are first canonicalised to English
    (C15), then folded.  A match holds when any of:
      • the candidate equals the full gold name (exact);
      • the candidate is a single significant token equal to a token of the
        gold name (a split name, e.g. "Magnesium" vs "Magnesium (gesamt)",
        "Saturated" vs "Saturated Fats");
      • the gold name is a single significant token contained in the
        candidate's tokens.
    Significant = length ≥ _MIN_PIECE_LEN.  Specific vitamins compare only by
    their full canonical form, so "Vitamin C" never matches "Vitamin B1".
    Adjust this function to reconcile the split with the embedding sweep.
    """
    cf = _fold(_canonicalise_nutrient(candidate_text))
    if not cf:
        return False
    c_tokens = set(cf.split())
    c_single = cf if " " not in cf else None

    for g in gold_names:
        gf = _fold(_canonicalise_nutrient(g))
        if not gf:
            continue
        if cf == gf:                                          # exact
            return True
        g_tokens = set(gf.split())
        if (c_single and len(c_single) >= _MIN_PIECE_LEN
                and c_single in g_tokens):                    # cand ⊂ gold
            return True
        if (" " not in gf and len(gf) >= _MIN_PIECE_LEN
                and gf in c_tokens):                          # gold ⊂ cand
            return True
    return False


def norm_image_key(name: str) -> str:
    """Join key used on BOTH sides: stem + lowercased suffix.
    Matches run_graph_v2_experiment.py (handles '1.PNG' vs '1.png')."""
    p = Path(name)
    return p.stem + p.suffix.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gold loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_gold_names(csv_path: str) -> Dict[str, Set[str]]:
    """image_key -> set of gold nutrient surface names."""
    gold: Dict[str, Set[str]] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "image_id" not in reader.fieldnames or "nutrient" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must have 'image_id' and 'nutrient' columns; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            key = norm_image_key(str(row.get("image_id", "")).strip())
            name = str(row.get("nutrient", "")).strip()
            if not key or not name:
                continue
            gold.setdefault(key, set()).add(name)
    return gold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase A — build the candidate dataset  (OCR + corrector + GLiNER scoring)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _discover_images(raw_dir: str, images_filter: Optional[str]) -> List[Path]:
    raw = Path(raw_dir)
    if not raw.is_dir():
        raise FileNotFoundError(f"RAW_DIR not found: {raw_dir}")
    if images_filter:
        wanted = {s.strip() for s in images_filter.split(",") if s.strip()}
        files = [f for f in raw.iterdir()
                 if f.suffix.lower() in IMAGE_EXTENSIONS and f.stem in wanted]
    else:
        files = [f for f in raw.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.name))
    return files


def _load_classifier_class(classifier_path: str):
    """Import GLiNERSemanticClassifier (package import preferred)."""
    try:
        from src.classification.gliner_classifier import GLiNERSemanticClassifier
        return GLiNERSemanticClassifier
    except Exception as e_pkg:
        p = Path(classifier_path)
        if not p.exists():
            raise ImportError(
                "Could not import GLiNERSemanticClassifier. Place gliner_classifier.py "
                f"at src/classification/ or pass --classifier-path. (package import said: {e_pkg})"
            )
        spec = importlib.util.spec_from_file_location("gliner_classifier", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.GLiNERSemanticClassifier


def _import_correct_tokens():
    """Import correct_tokens from the v2 corrector (C15-aware).

    No silent fallback to a non-v2 corrector: that would skip C15 and
    reintroduce untranslated candidate names, which is exactly the bug
    this calibration must avoid."""
    last_err = None
    for modname in ("src.utils.paddleocr_corrector_v2",
                    "src.utils.paddleOCR_corrector_v2"):
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, "correct_tokens", None)
            if callable(fn):
                return fn
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import correct_tokens from src.utils.paddleocr_corrector_v2 "
        f"(C15-aware corrector). Last error: {last_err}"
    )


def build_candidate_dataset(
    raw_dir: str,
    mode: str,
    model_id: Optional[str],
    conf_thresh: float,
    images_filter: Optional[str],
    classifier_path: str,
    device: Optional[str],
    gold: Dict[str, Set[str]],
) -> List[Dict]:
    """Run OCR + corrector + GLiNER scoring; cache raw per-candidate scores."""
    from src.ocr.paddleocr_runner import run_ocr_on_image
    correct_tokens = _import_correct_tokens()  # v2 corrector (C15-aware)

    GLiNERSemanticClassifier = _load_classifier_class(classifier_path)

    images = _discover_images(raw_dir, images_filter)
    if not images:
        raise RuntimeError(f"No images discovered in {raw_dir}")

    clf_kwargs = dict(mode=mode, threshold=0.0, confidence_threshold=conf_thresh,
                      device=device)
    if model_id:
        clf_kwargs["model_id"] = model_id
    clf = GLiNERSemanticClassifier(**clf_kwargs)

    print(f"\n{'='*65}")
    print("  GLiNER THRESHOLD SWEEP — Phase A (building candidate dataset)")
    print(f"  Images       : {len(images)} from {raw_dir}/")
    print(f"  Gold images  : {len(gold)} annotated")
    print(f"  C15 source   : {'pipeline apply_c15_normalise_names' if _C15_FN else 'built-in fallback map'}")
    print(f"  Classifier   : mode={mode}  conf_thresh={conf_thresh}")
    print(f"{'='*65}\n")

    records: List[Dict] = []
    missing_gold: List[str] = []

    for i, img in enumerate(images, 1):
        key = norm_image_key(img.name)
        gold_names = gold.get(key, set())
        if not gold_names:
            missing_gold.append(key)

        try:
            tokens = run_ocr_on_image(str(img))
            corrected, _ = correct_tokens(tokens, return_log=True)
            cands = clf.score_candidates(corrected)
        except Exception as e:
            print(f"  [{i}/{len(images)}] {key:<14} ERROR: {e}")
            continue

        n_true = 0
        for c in cands:
            tok = c.get("token", "")
            truth = is_true_nutrient(tok, gold_names)  # live readout only
            n_true += int(truth)
            records.append({
                "image_id": key,
                "token": tok,
                "canon": _canonicalise_nutrient(tok),
                "norm": c.get("norm", ""),
                "rule_label": c.get("rule_label", ""),
                "gliner_score": float(c.get("gliner_score", 0.0) or 0.0),
            })
        print(f"  [{i}/{len(images)}] {key:<14} candidates={len(cands):>3}  true={n_true:>3}")

    if missing_gold:
        print(f"\n  WARN  {len(missing_gold)} image(s) had no gold rows: "
              f"{', '.join(missing_gold[:12])}" + (" ..." if len(missing_gold) > 12 else ""))

    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cache I/O   (stores raw scores only; labels are re-derived in Phase B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CACHE_FIELDS = ["image_id", "token", "canon", "norm", "rule_label", "gliner_score"]


def write_cache(records: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CACHE_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in _CACHE_FIELDS})


def read_cache(path: Path) -> List[Dict]:
    """Tolerant of older caches (missing 'canon'); needs image_id, token, score."""
    out: List[Dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({
                "image_id": r.get("image_id", ""),
                "token": r.get("token", ""),
                "canon": r.get("canon", ""),
                "norm": r.get("norm", ""),
                "rule_label": r.get("rule_label", ""),
                "gliner_score": float(r.get("gliner_score", 0.0) or 0.0),
            })
    return out


def label_records(records: List[Dict], gold: Dict[str, Set[str]]) -> None:
    """Phase-B labelling: (re)derive is_true_nutrient for every record.
    Cheap and re-runnable — this is why editing the rule needs no --rebuild."""
    for r in records:
        gold_names = gold.get(r["image_id"], set())
        r["is_true_nutrient"] = int(is_true_nutrient(r["token"], gold_names))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase B — sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _frange(lo: float, hi: float, step: float) -> List[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


def sweep_thresholds(records: List[Dict], tmin: float, tmax: float,
                     tstep: float) -> List[Dict]:
    """precision / recall / F1 at each threshold over the cached candidates."""
    scores = [r["gliner_score"] for r in records]
    truths = [r["is_true_nutrient"] for r in records]
    n_true = sum(truths)

    rows: List[Dict] = []
    for t in _frange(tmin, tmax, tstep):
        tp = fp = fn = 0
        for s, y in zip(scores, truths):
            pred = s >= t
            if pred and y:
                tp += 1
            elif pred and not y:
                fp += 1
            elif (not pred) and y:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rows.append({"threshold": t, "precision": prec, "recall": rec,
                     "f1": f1, "tp": tp, "fp": fp, "fn": fn, "pred_pos": tp + fp})
    return rows


def pick_operating_point(rows: List[Dict]) -> Dict:
    """F1-optimal threshold; ties broken toward the lower (more permissive)
    threshold, consistent with the node favouring recall."""
    return sorted(rows, key=lambda r: (-r["f1"], r["threshold"]))[0]


def write_sweep_csv(rows: List[Dict], path: Path) -> None:
    fields = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "pred_pos"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_sweep(rows: List[Dict], best: Dict, png_path: Path,
               n_true: int, n_total: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [r["threshold"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(ts, prec, label="Precision", linewidth=2)
    ax.plot(ts, rec, label="Recall", linewidth=2)
    ax.plot(ts, f1, label="F1", linewidth=2.4)
    ax.axvline(best["threshold"], color="0.35", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"  τ* = {best['threshold']:.2f}\n  F1 = {best['f1']:.3f}\n"
        f"  P = {best['precision']:.3f}  R = {best['recall']:.3f}",
        xy=(best["threshold"], best["f1"]),
        xytext=(6, -38), textcoords="offset points",
        fontsize=9, va="top",
    )
    ax.set_xlabel("GLiNER span-acceptance threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"GLiNER nutrient-node threshold calibration\n"
        f"{n_total} candidates ({n_true} nutrients, {n_total - n_true} non-nutrients)"
    )
    ax.set_xlim(min(ts), max(ts))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the GLiNER nutrient-node threshold.")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--images", default=None, help="comma-separated stems, e.g. 1,101,118")
    ap.add_argument("--mode", default="gliner_only", choices=["gliner_only", "hybrid"])
    ap.add_argument("--model-id", default=None, help="override the GLiNER model id")
    ap.add_argument("--conf-thresh", type=float, default=DEFAULT_CONF_THRESH)
    ap.add_argument("--classifier-path", default=DEFAULT_CLASSIFIER_PATH)
    ap.add_argument("--device", default=None, help="cpu | cuda | (auto)")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--tstep", type=float, default=0.01)
    ap.add_argument("--rebuild", action="store_true", help="regenerate scores (re-run OCR/GLiNER)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "gliner_candidate_scores.csv"
    sweep_csv = out_dir / "gliner_threshold_sweep.csv"
    png_path = out_dir / "threshold_sweep_gliner.png"

    gold = load_gold_names(args.csv)  # needed by both build (readout) and labelling

    # Phase A (or load cached scores)
    if args.rebuild or not cache_path.exists():
        records = build_candidate_dataset(
            raw_dir=args.raw_dir, mode=args.mode, model_id=args.model_id,
            conf_thresh=args.conf_thresh, images_filter=args.images,
            classifier_path=args.classifier_path, device=args.device, gold=gold,
        )
        write_cache(records, cache_path)
        print(f"\n  Cached {len(records)} candidate scores → {cache_path}")
    else:
        records = read_cache(cache_path)
        print(f"  Loaded {len(records)} cached candidate scores from {cache_path}")
        print(f"  (labels re-derived from cache; pass --rebuild only to re-run OCR/GLiNER)")

    if not records:
        print("  No candidates — nothing to sweep.")
        return

    # Phase B: authoritative labelling (cheap, re-runnable) + sweep
    label_records(records, gold)
    n_total = len(records)
    n_true = sum(r["is_true_nutrient"] for r in records)
    n_non = n_total - n_true

    rows = sweep_thresholds(records, args.tmin, args.tmax, args.tstep)
    write_sweep_csv(rows, sweep_csv)
    best = pick_operating_point(rows)
    plot_sweep(rows, best, png_path, n_true=n_true, n_total=n_total)

    print(f"\n{'='*65}")
    print("  GLiNER THRESHOLD SWEEP — result")
    print(f"{'='*65}")
    print(f"  C15 canonicaliser : {'pipeline' if _C15_FN else 'fallback map'}")
    print(f"  Candidates        : {n_total}   (nutrients {n_true} / non-nutrients {n_non})")
    print(f"  Compare to §4.3.4 : embedding sweep had ~796 (462 / 334)")
    print(f"  Grid              : [{args.tmin}, {args.tmax}] step {args.tstep}")
    print(f"  --- F1-optimal operating point ---")
    print(f"  threshold (τ*)    : {best['threshold']:.3f}")
    print(f"  F1                : {best['f1']:.4f}")
    print(f"  precision         : {best['precision']:.4f}")
    print(f"  recall            : {best['recall']:.4f}")
    print(f"  TP/FP/FN          : {best['tp']} / {best['fp']} / {best['fn']}")
    print(f"\n  Figure            : {png_path}")
    print(f"  Sweep table       : {sweep_csv}")
    print(f"  Candidate cache   : {cache_path}")
    print(f"\n  → Set NUTRIENT_THRESHOLD = {best['threshold']:.2f} in gliner_classifier.py")
    print(f"  → Update §4.3.6 to report this calibrated value (it currently says fixed).")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()