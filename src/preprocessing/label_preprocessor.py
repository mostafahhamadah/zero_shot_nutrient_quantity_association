"""
label_preprocessor.py
=====================
Preprocessing module for supplement label images.

Handles two image types:
  - Flat label photos / scans  → enhance contrast + denoise only
  - Curved bottle photos       → detect label region → perspective dewarp → enhance

Pipeline stages:
  1. Load image
  2. Classify image type (flat vs curved) via edge curvature heuristic
  3. If curved → detect label quad → perspective warp to flat rectangle
  4. Enhance contrast (CLAHE)
  5. Optional denoise
  6. Save processed image to data/processed/

Usage:
    # Single image
    python label_preprocessor.py --image data/raw/1.jpeg

    # Full folder
    python label_preprocessor.py --folder data/raw --output data/processed
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json


# ── Core helpers ──────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to LAB L-channel for contrast enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    """Light denoise — preserves text sharpness."""
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


# ── Curvature classifier ──────────────────────────────────────────────────────

def is_curved_label(image: np.ndarray, debug: bool = False) -> bool:
    """
    Heuristic: detect if label is on a curved bottle surface.

    Strategy:
      - Convert to grayscale + edge detect
      - Find long contours
      - Fit a line to the longest horizontal edge
      - If the residual from a straight line is high → curved
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # Sort by arc length, take top 5
    long_contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)[:5]

    curvature_scores = []
    for cnt in long_contours:
        if len(cnt) < 10:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        # Fit polynomial degree 2 to x->y
        if pts[:, 0].max() - pts[:, 0].min() < 50:
            continue
        coeffs = np.polyfit(pts[:, 0], pts[:, 1], 2)
        # Quadratic coefficient magnitude = curvature
        curvature_scores.append(abs(coeffs[0]))

    if not curvature_scores:
        return False

    mean_curvature = np.mean(curvature_scores)
    is_curved = mean_curvature > 0.0005  # tuned threshold

    if debug:
        print(f"  Curvature score: {mean_curvature:.6f} → {'CURVED' if is_curved else 'FLAT'}")

    return is_curved


# ── Label region detection ────────────────────────────────────────────────────

def detect_label_quad(image: np.ndarray) -> np.ndarray | None:
    """
    Detect the four corners of the label region in a bottle photo.

    Returns:
        4x2 numpy array of corner points (ordered TL, TR, BR, BL),
        or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect broken label edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image.shape[:2]
    image_area = h * w

    # Find largest rectangular-ish contour
    best_quad = None
    best_area = 0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.05:  # must cover at least 5% of image
            continue
        if area > image_area * 0.98:  # exclude full-image contour
            continue

        # Approximate polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            if area > best_area:
                best_area = area
                best_quad = approx.reshape(4, 2)

    if best_quad is None:
        # Fallback: use bounding rect of largest contour
        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        best_quad = np.array([[x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]], dtype=np.float32)

    return best_quad.astype(np.float32)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # TL
    rect[2] = pts[np.argmax(s)]    # BR
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect


def perspective_dewarp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to dewarp the label to a flat rectangle.
    """
    rect = order_points(quad)
    tl, tr, br, bl = rect

    # Compute output dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


# ── Cylindrical unwarping (for strongly curved labels) ───────────────────────

def cylindrical_unwarp(image: np.ndarray, focal_length: float = None) -> np.ndarray:
    """
    Cylindrical projection to unwarp curved bottle labels.

    Args:
        image: Input image (after perspective dewarp if needed).
        focal_length: Camera focal length in pixels. If None, estimated from image width.

    Returns:
        Unwarped image.
    """
    h, w = image.shape[:2]
    if focal_length is None:
        focal_length = w * 1.2  # reasonable estimate for phone camera

    # Build cylindrical map
    K = np.array([[focal_length, 0, w / 2],
                  [0, focal_length, h / 2],
                  [0, 0, 1]], dtype=np.float32)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            theta = (x - w / 2) / focal_length
            h_cyl = (y - h / 2) / focal_length

            X = np.sin(theta)
            Y = h_cyl
            Z = np.cos(theta)

            x_src = focal_length * X / Z + w / 2
            y_src = focal_length * Y / Z + h / 2

            map_x[y, x] = x_src
            map_y[y, x] = y_src

    unwarped = cv2.remap(image, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return unwarped


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_label(image_path: str,
                     output_path: str = None,
                     apply_dewarp: bool = True,
                     apply_cylindrical: bool = False,
                     apply_enhance: bool = True,
                     apply_denoise: bool = False,
                     debug: bool = False) -> tuple:
    """
    Full preprocessing pipeline for a single label image.

    Args:
        image_path: Input image path.
        output_path: Where to save processed image. If None, saves next to input.
        apply_dewarp: Whether to apply perspective dewarping.
        apply_cylindrical: Whether to also apply cylindrical unwarping (heavy).
        apply_enhance: Apply CLAHE contrast enhancement.
        apply_denoise: Apply noise reduction (slower).
        debug: Print diagnostic info.

    Returns:
        (processed_image, metadata_dict)
    """
    image_path = Path(image_path).resolve()
    image = load_image(image_path)
    h_orig, w_orig = image.shape[:2]

    metadata = {
        "image": image_path.name,
        "original_size": [w_orig, h_orig],
        "curved_detected": False,
        "dewarp_applied": False,
        "cylindrical_applied": False,
        "enhance_applied": False,
        "denoise_applied": False,
    }

    processed = image.copy()

    # Stage 1: Detect curvature
    curved = is_curved_label(processed, debug=debug)
    metadata["curved_detected"] = curved

    if debug:
        print(f"  Image: {image_path.name} | Curved: {curved}")

    # Stage 2: Perspective dewarp (for curved or angled photos)
    if apply_dewarp and curved:
        quad = detect_label_quad(processed)
        if quad is not None:
            try:
                processed = perspective_dewarp(processed, quad)
                metadata["dewarp_applied"] = True
                if debug:
                    print(f"  Perspective dewarp applied. New size: {processed.shape[1]}x{processed.shape[0]}")
            except Exception as e:
                if debug:
                    print(f"  Dewarp failed: {e}. Skipping.")

    # Stage 3: Cylindrical unwarping (optional, expensive)
    if apply_cylindrical and curved:
        try:
            processed = cylindrical_unwarp(processed)
            metadata["cylindrical_applied"] = True
            if debug:
                print("  Cylindrical unwarp applied.")
        except Exception as e:
            if debug:
                print(f"  Cylindrical unwarp failed: {e}")

    # Stage 4: Contrast enhancement
    if apply_enhance:
        processed = enhance_contrast(processed)
        metadata["enhance_applied"] = True

    # Stage 5: Denoise (optional)
    if apply_denoise:
        processed = denoise(processed)
        metadata["denoise_applied"] = True

    # Save output
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_processed.jpg"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), processed)

    metadata["output_path"] = str(output_path)
    metadata["output_size"] = [processed.shape[1], processed.shape[0]]

    return processed, metadata


def preprocess_folder(input_folder: str,
                      output_folder: str,
                      save_metadata: bool = True,
                      debug: bool = False) -> list:
    """
    Preprocess all images in a folder.

    Args:
        input_folder: Folder with raw label images.
        output_folder: Folder to save processed images.
        save_metadata: Save per-image metadata as JSON.
        debug: Print diagnostic info.

    Returns:
        List of metadata dicts for all processed images.
    """
    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([f for f in input_folder.iterdir()
                     if f.suffix.lower() in supported])

    if not images:
        print(f"No images found in {input_folder}")
        return []

    print(f"Found {len(images)} images. Starting preprocessing...\n")
    all_metadata = []

    for idx, img_path in enumerate(images, 1):
        out_path = output_folder / f"{img_path.stem}_processed.jpg"
        print(f"[{idx}/{len(images)}] {img_path.name}", end=" → ")

        try:
            _, meta = preprocess_label(
                image_path=str(img_path),
                output_path=str(out_path),
                apply_dewarp=True,
                apply_cylindrical=False,
                apply_enhance=True,
                apply_denoise=False,
                debug=debug
            )
            all_metadata.append(meta)
            tag = "CURVED+DEWARP" if meta["dewarp_applied"] else ("CURVED" if meta["curved_detected"] else "FLAT")
            print(f"{tag} | {meta['output_size'][0]}x{meta['output_size'][1]}")
        except Exception as e:
            print(f"ERROR: {e}")
            all_metadata.append({"image": img_path.name, "error": str(e)})

    # Save metadata summary
    if save_metadata:
        meta_path = output_folder / "preprocessing_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        print(f"\nMetadata saved: {meta_path}")

    curved_count = sum(1 for m in all_metadata if m.get("curved_detected"))
    flat_count = len(all_metadata) - curved_count
    print(f"\nSummary: {len(images)} images | Curved: {curved_count} | Flat: {flat_count}")

    return all_metadata


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supplement label preprocessor")
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--folder", type=str, help="Folder of images to process")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output folder (default: data/processed)")
    parser.add_argument("--debug", action="store_true", help="Print diagnostic info")
    parser.add_argument("--cylindrical", action="store_true",
                        help="Also apply cylindrical unwarping (slower)")
    args = parser.parse_args()

    if args.image:
        out = Path(args.output) / (Path(args.image).stem + "_processed.jpg")
        _, meta = preprocess_label(
            image_path=args.image,
            output_path=str(out),
            apply_cylindrical=args.cylindrical,
            debug=True
        )
        print(f"\nResult: {meta}")

    elif args.folder:
        preprocess_folder(
            input_folder=args.folder,
            output_folder=args.output,
            debug=args.debug
        )
    else:
        print("Provide --image or --folder. Use --help for options.")