"""
inspect_serializer.py
=====================
Standalone inspection tool for Stage 3A — text_serializer output.

For each image in IMAGE_DIR, runs:
  Stage 1 : PaddleOCR
  Stage 2 : PaddleOCR corrector
  Stage 3A: text_serializer.serialize_tokens_for_gliner()

Saves a human-readable report per image showing:
  - Serialized text exactly as Qwen/GLiNER receives it
  - Visual line groupings (which tokens landed on which line)
  - Token spans with character offsets

USAGE
-----
  # Inspect all images, save to outputs/serializer_inspection/
  python inspect_serializer.py

  # Inspect a single image
  python inspect_serializer.py --image 1.jpeg

  # Show output in terminal instead of saving
  python inspect_serializer.py --image 1.jpeg --print
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ocr.paddleocr_runner      import run_ocr_on_image
from src.utils.paddleocr_corrector import correct_tokens
from src.utils.text_serializer     import serialize_tokens_for_gliner

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("inspect_serializer")

IMAGE_DIR    = Path("data/raw")
OUT_DIR      = Path("outputs/serializer_inspection")
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}


def _format_report(image_id: str, serialized: dict, tokens: list) -> str:
    """
    Build a human-readable inspection report for one image.

    Sections:
      1. SERIALIZED TEXT  — exact string sent to Qwen/GLiNER
      2. LINE GROUPINGS   — which tokens landed on each visual line
      3. TOKEN SPANS      — char offsets per token (for debugging remapping)
    """
    text        = serialized["text"]
    token_spans = serialized["token_spans"]
    lines       = serialized["lines"]

    # Build token index → original token dict lookup
    tok_by_idx  = {i: t for i, t in enumerate(tokens)}

    sep = "─" * 65
    out = []

    out.append(f"{'='*65}")
    out.append(f"  IMAGE: {image_id}")
    out.append(f"  OCR tokens (after corrector): {len(tokens)}")
    out.append(f"  Serialized tokens:            {len(token_spans)}")
    out.append(f"  Visual lines detected:        {len(lines)}")
    out.append(f"{'='*65}")

    # ── Section 1: Serialized text ─────────────────────────────────────────
    out.append("")
    out.append("  ── SERIALIZED TEXT (exact Qwen/GLiNER input) ───────────")
    out.append("")
    for i, line_text in enumerate(text.split("\n")):
        out.append(f"  Line {i+1:02d} │ {line_text}")
    out.append("")

    # ── Section 2: Line groupings ──────────────────────────────────────────
    out.append(f"  ── LINE GROUPINGS ───────────────────────────────────────")
    out.append("")

    # Build token_index → span lookup
    span_by_idx = {ts["token_index"]: ts for ts in token_spans}

    for line in lines:
        line_tokens = line["token_indices"]
        token_texts = []
        for idx in line_tokens:
            tok = tok_by_idx.get(idx, {})
            txt = str(tok.get("token", "?"))
            token_texts.append(txt)

        y_center = line["y_center"]
        out.append(
            f"  Line {line['line_id']+1:02d}  y={y_center:6.1f}px  "
            f"[{len(line_tokens)} tokens]"
        )
        out.append(f"    → {' | '.join(token_texts)}")

    out.append("")

    # ── Section 3: Token spans (char offsets) ──────────────────────────────
    out.append(f"  ── TOKEN SPANS (char offsets for remapping) ─────────────")
    out.append("")
    out.append(f"  {'IDX':>4}  {'LINE':>4}  {'START':>6}  {'END':>6}  {'TEXT'}")
    out.append(f"  {sep}")

    for ts in token_spans:
        out.append(
            f"  {ts['token_index']:>4}  "
            f"{ts['line_id']:>4}  "
            f"{ts['start_char']:>6}  "
            f"{ts['end_char']:>6}  "
            f"{ts['token_text']}"
        )

    out.append("")
    out.append(f"{'='*65}")
    out.append("")

    return "\n".join(out)


def inspect_image(image_path: Path, print_only: bool = False) -> str:
    """
    Run the full Stage 1→2→3A pipeline for one image and return the report string.
    """
    image_id = image_path.name

    # Stage 1 — PaddleOCR
    raw_tokens = run_ocr_on_image(str(image_path))
    if not raw_tokens:
        return f"{'='*65}\n  IMAGE: {image_id}\n  ⚠ Zero OCR tokens — skipped.\n{'='*65}\n"

    # Stage 2 — Corrector
    corrected, _ = correct_tokens(raw_tokens, return_log=True)

    # Stage 3A — Serializer
    serialized = serialize_tokens_for_gliner(corrected)

    return _format_report(image_id, serialized, corrected)


def run_all(image_filter: str | None, print_only: bool) -> None:
    """
    Inspect all (or one filtered) image(s).
    """
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        logger.error("No images found in %s", IMAGE_DIR)
        sys.exit(1)

    if image_filter:
        images = [p for p in images if p.name == image_filter]
        if not images:
            logger.error("Image '%s' not found in %s", image_filter, IMAGE_DIR)
            sys.exit(1)

    logger.info("Inspecting %d image(s)...", len(images))

    if not print_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_reports = []

    for img_path in images:
        logger.info("  %s", img_path.name)
        try:
            report = inspect_image(img_path, print_only=print_only)
        except Exception as exc:
            report = (
                f"{'='*65}\n"
                f"  IMAGE: {img_path.name}\n"
                f"  ⚠ ERROR: {exc}\n"
                f"{'='*65}\n"
            )
            logger.error("Failed on %s: %s", img_path.name, exc)

        if print_only:
            print(report)
        else:
            all_reports.append(report)

            # Save individual file per image
            out_file = OUT_DIR / f"{img_path.stem}_serializer.txt"
            out_file.write_text(report, encoding="utf-8")

    if not print_only:
        # Save one combined report with all images
        combined_path = OUT_DIR / "_all_images_serializer.txt"
        combined_path.write_text("\n".join(all_reports), encoding="utf-8")
        logger.info("Individual reports: %s/", OUT_DIR)
        logger.info("Combined report:    %s", combined_path)
        logger.info("Done. %d image(s) inspected.", len(images))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect text_serializer output for supplement label images.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Inspect a single image by filename (e.g. --image 1.jpeg). "
             "Omit to inspect all images.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print report to terminal instead of saving to file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(
        image_filter = args.image,
        print_only   = args.print,
    )