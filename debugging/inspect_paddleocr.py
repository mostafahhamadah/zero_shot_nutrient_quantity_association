"""
inspect_paddleocr.py  —  PaddleOCR Raw Output Inspector
Zero-Shot Nutrient Extraction | Moustafa Hamada | THD + USB

PURPOSE
-------
Runs PaddleOCR on ALL images in data/raw/ and writes the full raw token
output to a plain text report file for corrector analysis.

USAGE
-----
# Full inspection of all images (default):
    python inspect_paddleocr.py

# Change confidence threshold for LOW_CONF flagging:
    python inspect_paddleocr.py --conf 0.50

# Sort tokens by confidence ascending (worst first):
    python inspect_paddleocr.py --sort conf

# Only show tokens matching a regex pattern:
    python inspect_paddleocr.py --grep "^\d+[a-zA-Zµ]"

OUTPUT
------
results/ocr_inspection/
    raw_ocr_report.txt   — full human-readable report for every image
    raw_ocr_tokens.csv   — one row per token, all images (for Excel filtering)
    raw_ocr_summary.csv  — one row per image summary
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known OCR error patterns — flagged in the report for corrector analysis
# ---------------------------------------------------------------------------

SUSPICIOUS_PATTERNS = [
    # Trailing digit artefact: 1.4 -> 1409, 86.1 -> 86.19, 2.5 -> 2509
    (r"^\d+09$",                    "trailing-digit artefact (09 suffix)"),
    (r"^\d+\.\d+9$",                "trailing-digit artefact (decimal +9)"),
    (r"^\d+19$",                    "trailing-digit artefact (19 suffix)"),
    # Apostrophe or comma as decimal: 660'0, 52,1
    (r"\d[',]\d",                   "apostrophe/comma as decimal separator"),
    # Fused kJ/kcal label: 1391k330, ENERGIEkJ
    (r"\d+k\d+",                    "fused kJ/kcal value"),
    (r"[A-Za-z]+kJ",                "fused label+kJ unit"),
    # Fused quantity+unit: 400mg, 52.1g
    (r"^\d+[\.,]\d*[a-zA-Zµ]+$",   "fused quantity+unit (decimal)"),
    (r"^\d+[a-zA-Zµ]+$",           "fused quantity+unit (integer)"),
    # Fused context tokens: Je1009, pro106, PORTION**
    (r"(?i)Je\d+",                  "fused context (Je+number -> je 100g?)"),
    (r"(?i)pro\d+",                 "fused context (pro+number -> pro 100g?)"),
    (r"\*{2}",                      "PORTION** context token"),
    # Table/border artefact characters
    (r"[|{}\[\]\\]",                "border/table artefact character"),
    # Single non-alphanumeric noise
    (r"^[^a-zA-Z0-9µ.,]$",         "single non-alphanumeric noise character"),
    # Unit confusion
    (r"^Ug$|^ug$",                  "unit confusion: Ug/ug -> µg"),
    # Multilingual slash-variant too long (4+ parts)
    (r"(?:[A-Za-zÀ-ÿ]+/){3,}",     "multilingual slash-variant (4+ parts)"),
    # Fused tokens (camelCase boundary inside word)
    (r"[a-z][A-Z][a-z]",           "possible fused tokens (camelCase boundary)"),
    # Letter embedded inside number: 2A5, 1B4
    (r"^\d+[A-Za-z]\d+$",          "letter embedded in number (OCR confusion)"),
    # Unclosed parenthesis in quantity: 1578 (37
    (r"\d+\s*\(\s*\d+$",           "unclosed parenthesis in quantity"),
]

COMPILED_PATTERNS = [
    (re.compile(pat), desc) for pat, desc in SUSPICIOUS_PATTERNS
]


def flag_suspicious(token: str):
    return [desc for pat, desc in COMPILED_PATTERNS if pat.search(token)]


# ---------------------------------------------------------------------------
# PaddleOCR runner — raw output, no corrector applied
# ---------------------------------------------------------------------------

def run_raw_ocr(image_path: Path) -> list:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("ERROR: paddleocr not installed.  pip install paddleocr")
        sys.exit(1)

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=True)
    result = ocr.ocr(str(image_path), cls=True)
    tokens = []

    if not result or result[0] is None:
        return tokens

    for line in result[0]:
        if line is None:
            continue
        bbox_pts, (text, conf) = line
        xs = [pt[0] for pt in bbox_pts]
        ys = [pt[1] for pt in bbox_pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        tokens.append({
            "token": text,
            "conf":  round(float(conf), 4),
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": round((x1 + x2) / 2, 1),
            "cy": round((y1 + y2) / 2, 1),
        })
    return tokens


def get_image_size(image_path: Path):
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# Per-image report builder
# ---------------------------------------------------------------------------

def build_image_block(
    image_path    : Path,
    conf_threshold: float,
    sort_by       : str,
    grep_pattern  : str | None,
) -> tuple[str, list, dict]:
    """
    Run OCR on one image.
    Returns (report_block_str, enriched_token_list, summary_dict).
    """
    tokens  = run_raw_ocr(image_path)
    w, h    = get_image_size(image_path)
    grep_re = re.compile(grep_pattern, re.IGNORECASE) if grep_pattern else None

    enriched = []
    for i, tok in enumerate(tokens):
        flags = flag_suspicious(tok["token"])
        enriched.append({
            **tok,
            "idx":          i,
            "flags":        flags,
            "below_thresh": tok["conf"] < conf_threshold,
        })

    if grep_re:
        enriched = [t for t in enriched if grep_re.search(t["token"])]

    sort_fn = {
        "cy":    lambda t: t["cy"],
        "conf":  lambda t: t["conf"],
        "idx":   lambda t: t["idx"],
        "token": lambda t: t["token"].lower(),
    }.get(sort_by, lambda t: t["cy"])
    enriched.sort(key=sort_fn)

    low_conf   = [t for t in enriched if t["below_thresh"]]
    suspicious = [t for t in enriched if t["flags"]]

    W   = 110
    SEP = "-" * W
    L   = []

    L.append("=" * W)
    L.append(
        f"  IMAGE : {image_path.name:<28} ({w}x{h}px)"
        f"  |  TOTAL TOKENS: {len(tokens)}"
        f"  |  LOW_CONF (< {conf_threshold}): {len(low_conf)}"
        f"  |  SUSPICIOUS: {len(suspicious)}"
    )
    L.append(SEP)
    L.append(
        f"  {'IDX':>4}  {'TOKEN':<38}  {'CONF':>6}"
        f"  {'X1':>5}  {'Y1':>5}  {'X2':>5}  {'Y2':>5}"
        f"  {'CX':>7}  {'CY':>7}"
    )
    L.append("  " + SEP)

    for tok in enriched:
        marker   = " *" if tok["below_thresh"] else "  "
        flag_str = "   <-- " + " | ".join(tok["flags"]) if tok["flags"] else ""
        L.append(
            f"  {tok['idx']:>4}{marker} {tok['token']:<38}  {tok['conf']:>6.4f}"
            f"  {tok['x1']:>5}  {tok['y1']:>5}  {tok['x2']:>5}  {tok['y2']:>5}"
            f"  {tok['cx']:>7.1f}  {tok['cy']:>7.1f}{flag_str}"
        )

    if low_conf:
        L.append("")
        L.append(f"  LOW CONFIDENCE (conf < {conf_threshold}):")
        for tok in sorted(low_conf, key=lambda t: t["conf"]):
            L.append(
                f"    [{tok['idx']:>3}]  conf={tok['conf']:.4f}"
                f"  token='{tok['token']}'"
            )

    if suspicious:
        L.append("")
        L.append("  SUSPICIOUS TOKENS:")
        for tok in suspicious:
            for flag in tok["flags"]:
                L.append(
                    f"    [{tok['idx']:>3}]  conf={tok['conf']:.4f}"
                    f"  token='{tok['token']}'   --> {flag}"
                )

    L.append("")

    summary = {
        "image_id":   image_path.name,
        "width":      w,
        "height":     h,
        "n_tokens":   len(tokens),
        "low_conf":   len(low_conf),
        "suspicious": len(suspicious),
        "low_pct":    round(len(low_conf) / len(tokens) * 100) if tokens else 0,
    }

    return "\n".join(L), enriched, summary


# ---------------------------------------------------------------------------
# Main — all images
# ---------------------------------------------------------------------------

def run_all(
    raw_dir       : Path,
    out_dir       : Path,
    conf_threshold: float,
    sort_by       : str,
    grep_pattern  : str | None,
) -> None:

    images = sorted(
        list(raw_dir.glob("*.jpeg")) +
        list(raw_dir.glob("*.jpg"))  +
        list(raw_dir.glob("*.png"))
    )

    if not images:
        print(f"ERROR: No images found in {raw_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path  = out_dir / "raw_ocr_report.txt"
    tokens_path  = out_dir / "raw_ocr_tokens.csv"
    summary_path = out_dir / "raw_ocr_summary.csv"

    all_token_rows = []
    all_summaries  = []

    print(f"\nPaddleOCR Raw Inspection")
    print(f"  Images     : {len(images)}")
    print(f"  Source     : {raw_dir}")
    print(f"  Report     : {report_path}")
    print(f"  Conf flag  : < {conf_threshold}")
    print(f"  Sort       : {sort_by}")
    if grep_pattern:
        print(f"  Filter     : {grep_pattern}")
    print()

    with open(report_path, "w", encoding="utf-8") as txt:

        # File header
        txt.write("=" * 110 + "\n")
        txt.write("  PADDLEOCR RAW OUTPUT INSPECTION REPORT\n")
        txt.write(f"  Images processed : {len(images)}\n")
        txt.write(f"  Image directory  : {raw_dir}\n")
        txt.write(f"  Conf threshold   : {conf_threshold}  (* = below threshold)\n")
        txt.write(f"  Sort order       : {sort_by}\n")
        if grep_pattern:
            txt.write(f"  Token filter     : {grep_pattern}\n")
        txt.write("  <-- = suspicious pattern flag\n")
        txt.write("=" * 110 + "\n\n")

        for i, img_path in enumerate(images, 1):
            print(f"  [{i:>3}/{len(images)}]  {img_path.name:<25}", end="  ", flush=True)

            block, enriched, summary = build_image_block(
                img_path, conf_threshold, sort_by, grep_pattern
            )

            txt.write(block + "\n")
            txt.flush()

            print(
                f"{summary['n_tokens']:>4} tokens"
                f"  |  low_conf: {summary['low_conf']}"
                f"  |  suspicious: {summary['suspicious']}"
            )

            for tok in enriched:
                all_token_rows.append({
                    "image_id":     img_path.name,
                    "idx":          tok["idx"],
                    "token":        tok["token"],
                    "conf":         tok["conf"],
                    "below_thresh": tok["below_thresh"],
                    "x1":           tok["x1"],
                    "y1":           tok["y1"],
                    "x2":           tok["x2"],
                    "y2":           tok["y2"],
                    "cx":           tok["cx"],
                    "cy":           tok["cy"],
                    "flags":        " | ".join(tok["flags"]) if tok["flags"] else "",
                })
            all_summaries.append(summary)

        # Global summary at bottom of report
        W = 110
        txt.write("=" * W + "\n")
        txt.write("  GLOBAL SUMMARY\n")
        txt.write("-" * W + "\n")
        txt.write(
            f"  {'IMAGE':<28}  {'TOKENS':>7}  {'SIZE':>11}"
            f"  {'LOW_CONF':>9}  {'LOW%':>5}  {'SUSPICIOUS':>11}\n"
        )
        txt.write("  " + "-" * W + "\n")
        for s in all_summaries:
            txt.write(
                f"  {s['image_id']:<28}  {s['n_tokens']:>7}"
                f"  {s['width']:>5}x{s['height']:<5}"
                f"  {s['low_conf']:>9}  {s['low_pct']:>4}%"
                f"  {s['suspicious']:>11}\n"
            )
        total_tok  = sum(s["n_tokens"]   for s in all_summaries)
        total_low  = sum(s["low_conf"]   for s in all_summaries)
        total_susp = sum(s["suspicious"] for s in all_summaries)
        txt.write("  " + "-" * W + "\n")
        txt.write(
            f"  {'TOTAL':<28}  {total_tok:>7}"
            f"  {'':>11}"
            f"  {total_low:>9}"
            f"  {round(total_low/total_tok*100) if total_tok else 0:>4}%"
            f"  {total_susp:>11}\n"
        )
        txt.write("=" * W + "\n")

    # Write CSVs
    if all_token_rows:
        with open(tokens_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_token_rows[0].keys()))
            w.writeheader()
            w.writerows(all_token_rows)

    if all_summaries:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_summaries)

    print(f"\nDone.")
    print(f"  {report_path}")
    print(f"  {tokens_path}")
    print(f"  {summary_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR raw output inspector — all images, saves to TXT + CSV."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Directory containing label images (default: data/raw)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/ocr_inspection"),
        help="Output directory (default: results/ocr_inspection)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.30,
        help="Confidence threshold for LOW_CONF flag (default: 0.30)"
    )
    parser.add_argument(
        "--sort", choices=["cy", "conf", "idx", "token"], default="cy",
        help="Token sort order within each image (default: cy = top-to-bottom)"
    )
    parser.add_argument(
        "--grep", type=str, default=None,
        help="Only include tokens matching this regex"
    )
    args = parser.parse_args()

    run_all(
        raw_dir       = args.raw_dir,
        out_dir       = args.out,
        conf_threshold= args.conf,
        sort_by       = args.sort,
        grep_pattern  = args.grep,
    )


if __name__ == "__main__":
    main()