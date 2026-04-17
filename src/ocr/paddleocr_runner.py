"""
paddleocr_runner.py
===================
Stage 1 — OCR
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Run PaddleOCR 3.0 (PP-OCRv5) on a supplement label image and return
every recognised token with its bounding box and confidence score.

No confidence filtering is applied here.  All tokens are returned so
that Stage 3 (classifier) can decide what is noise — keeping the
threshold decision in one place and making every token visible in the
pipeline UI.

OUTPUT SCHEMA
-------------
List[Dict], one entry per token:
    token : str   — recognised text
    x1    : int   — left   edge (pixels)
    y1    : int   — top    edge (pixels)
    x2    : int   — right  edge (pixels)
    y2    : int   — bottom edge (pixels)
    cx    : float — horizontal centre  ((x1+x2)/2)
    cy    : float — vertical   centre  ((y1+y2)/2)
    conf  : float — recognition confidence [0.0–1.0]

DESIGN DECISIONS
----------------
1. Multi-pass (EN + Latin):
   Supplement labels contain mixed-language text.  One pass per language
   model recovers tokens missed by a single model.  Results are merged
   via IoU deduplication (highest-confidence token wins on overlap).
   Set MULTI_PASS = False to use single-pass EN only.

2. GPU enabled by default (USE_GPU = True):
   PaddlePaddle auto-selects the first available CUDA device.
   Set USE_GPU = False to force CPU.

3. Polygon to AABB:
   PaddleOCR returns 4-corner polygons.  Converted to axis-aligned
   (x1, y1, x2, y2) via min/max — matches EasyOCR output format.

4. Model cache:
   PaddleOCR instances are cached in _MODEL_CACHE to avoid reloading
   on each call during batch processing.

REFERENCE
---------
Cui et al. (2025). PaddleOCR 3.0 Technical Report.
doi:10.48550/arXiv.2507.05595
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

MULTI_PASS      : bool      = True
PASS_LANGUAGES  : List[str] = ["en", "latin"]   # latin covers DE/FR/NL/IT/ES
IOU_DEDUP_THRESH: float     = 0.50
USE_GPU         : bool      = False
USE_ANGLE_CLS   : bool      = True
SHOW_PADDLE_LOG : bool      = False

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("paddleocr_runner")

# ── Model cache ───────────────────────────────────────────────────────────────

_MODEL_CACHE: Dict[str, object] = {}


def _get_model(lang: str):
    """Return a cached PaddleOCR instance for *lang*."""
    if lang not in _MODEL_CACHE:
        try:
            from paddleocr import PaddleOCR   # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed.\n"
                "  pip uninstall paddlepaddle paddlepaddle-gpu -y\n"
                "  pip install paddlepaddle-gpu==3.0.0 "
                "-i https://www.paddlepaddle.org.cn/packages/stable/cu123/\n"
                "  pip install paddleocr\n"
            ) from exc

        logger.info(f"[PaddleOCR] Loading model — lang='{lang}' gpu={USE_GPU}")
        _MODEL_CACHE[lang] = PaddleOCR(
            use_angle_cls = USE_ANGLE_CLS,
            lang          = lang,
            use_gpu       = USE_GPU,
            show_log      = SHOW_PADDLE_LOG,
        )
        logger.info(f"[PaddleOCR] Model ready — lang='{lang}'")

    return _MODEL_CACHE[lang]


# ── Bounding box utilities ────────────────────────────────────────────────────

def _polygon_to_aabb(polygon: List[List[float]]) -> Tuple[int, int, int, int]:
    """
    Convert a PaddleOCR 4-corner polygon to an axis-aligned bounding box.
    Input : [[x,y], [x,y], [x,y], [x,y]]  (clockwise)
    Output: (x1, y1, x2, y2)  with x1<x2, y1<y2
    """
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (
        int(round(min(xs))), int(round(min(ys))),
        int(round(max(xs))), int(round(max(ys))),
    )


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Intersection-over-Union for two axis-aligned bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Single-pass OCR ───────────────────────────────────────────────────────────

def _run_single_pass(image: np.ndarray, lang: str) -> List[Dict]:
    model  = _get_model(lang)
    result = model.ocr(image, cls=USE_ANGLE_CLS)
     
    if result is None:
        return []

    page = result[0] if (result and isinstance(result[0], list)) else result
    if page is None:
        return []

    tokens: List[Dict] = []
    for line in page:
        if line is None:
            continue
        try:
            polygon, (text, conf) = line
        except (TypeError, ValueError):
            continue

        text = str(text).strip()
        if not text:
            continue

        x1, y1, x2, y2 = _polygon_to_aabb(polygon)
        if x2 <= x1 or y2 <= y1:
            continue

        tokens.append({
            "token": text,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "conf": round(float(conf), 4),
        })

    return tokens

# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(token_lists: List[List[Dict]]) -> List[Dict]:
    """
    Merge token lists from multiple language passes.
    Greedy: sort by confidence descending; keep a token only if its bbox
    does not overlap any already-accepted token above IOU_DEDUP_THRESH.
    Result is sorted into reading order (top to bottom, left to right).
    """
    flat = [t for tlist in token_lists for t in tlist]
    flat.sort(key=lambda t: t["conf"], reverse=True)

    accepted      : List[Dict]                   = []
    accepted_boxes: List[Tuple[int,int,int,int]] = []

    for tok in flat:
        box = (tok["x1"], tok["y1"], tok["x2"], tok["y2"])
        if any(_iou(box, ab) > IOU_DEDUP_THRESH for ab in accepted_boxes):
            continue
        accepted.append(tok)
        accepted_boxes.append(box)

    accepted.sort(key=lambda t: (t["cy"], t["cx"]))
    return accepted


# ── Public API ────────────────────────────────────────────────────────────────

def run_ocr_on_image(
    image_path: str,
    languages : Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run PaddleOCR 3.0 on a supplement label image.

    The `languages` parameter is accepted for API compatibility with
    ocr_runner.py call sites but is ignored — PASS_LANGUAGES is used.

    Parameters
    ----------
    image_path : str        — path to JPEG / PNG image
    languages  : List[str]  — ignored (API compatibility only)

    Returns
    -------
    List[Dict]  — one dict per token, ALL tokens (no threshold):
        token, x1, y1, x2, y2, cx, cy, conf

    Raises
    ------
    FileNotFoundError — image_path does not exist
    ImportError       — PaddleOCR / PaddlePaddle not installed
    RuntimeError      — OpenCV cannot decode the image
    """
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"[PaddleOCR] Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"[PaddleOCR] OpenCV could not decode: {image_path}")

    image_name = Path(image_path).name
    h, w       = image.shape[:2]
    logger.info(f"[PaddleOCR] '{image_name}' ({w}x{h}px)")

    if MULTI_PASS:
        raw_lists = [_run_single_pass(image, lang) for lang in PASS_LANGUAGES]
        tokens    = _deduplicate(raw_lists)
    else:
        tokens = _run_single_pass(image, PASS_LANGUAGES[0])
        tokens.sort(key=lambda t: (t["cy"], t["cx"]))

    logger.info(f"[PaddleOCR] '{image_name}' — {len(tokens)} tokens returned")
    return tokens