import json
import cv2
import easyocr
from pathlib import Path


def run_ocr_on_image(image_path: str, languages: list = ["en", "de"]) -> list:
    """
    Run EasyOCR on a single image and return structured token list.

    Args:
        image_path: Path to the label image.
        languages: ["en", "de"] covers German + English supplement labels.

    Returns:
        List of dicts: {token, x1, y1, x2, y2, conf}
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    results = reader.readtext(str(image_path))

    tokens = []
    for (bbox, text, conf) in results:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]

        token_dict = {
            "token": text.strip(),
            "x1": int(min(x_coords)),
            "y1": int(min(y_coords)),
            "x2": int(max(x_coords)),
            "y2": int(max(y_coords)),
            "conf": round(float(conf), 4)
        }
        tokens.append(token_dict)

    return tokens


def run_ocr_on_folder(image_folder: str, output_folder: str) -> None:
    """
    Run OCR on all images in a folder and save individual JSON files.

    Args:
        image_folder: Folder containing label images.
        output_folder: Folder where JSON token files will be saved.
    """
    image_folder = Path(image_folder).resolve()
    output_folder = Path(output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        f for f in image_folder.iterdir()
        if f.suffix.lower() in supported_extensions
    ]

    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(image_files)} images. Starting OCR...\n")

    # Initialize reader once for all images to save time
    reader = easyocr.Reader(["en", "de"], gpu=False, verbose=False)

    for idx, image_path in enumerate(sorted(image_files), 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

        try:
            results = reader.readtext(str(image_path))

            tokens = []
            for (bbox, text, conf) in results:
                x_coords = [pt[0] for pt in bbox]
                y_coords = [pt[1] for pt in bbox]
                tokens.append({
                    "token": text.strip(),
                    "x1": int(min(x_coords)),
                    "y1": int(min(y_coords)),
                    "x2": int(max(x_coords)),
                    "y2": int(max(y_coords)),
                    "conf": round(float(conf), 4)
                })

            output_path = output_folder / f"{image_path.stem}_ocr.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(tokens, f, ensure_ascii=False, indent=2)

            print(f"  -> {len(tokens)} tokens -> {output_path.name}")

        except Exception as e:
            print(f"  ERROR on {image_path.name}: {e}")

    print(f"\nDone. Results saved to: {output_folder}")


def visualize_ocr_result(image_path: str, tokens: list,
                          output_path: str = None) -> None:
    """
    Draw bounding boxes and token text on the image for visual verification.

    Args:
        image_path: Path to original image.
        tokens: List of token dicts from run_ocr_on_image().
        output_path: Where to save the annotated image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for token in tokens:
        x1, y1, x2, y2 = token["x1"], token["y1"], token["x2"], token["y2"]
        text = token["token"]
        conf = token["conf"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = f"{text} ({conf:.2f})"
        cv2.putText(image, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")
    else:
        cv2.imshow("OCR Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Entry point: run on full folder ──────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_FOLDER  = r"C:\Users\Mostafa\Desktop\zero_shot_nutrient_association\data\raw"
    OUTPUT_FOLDER = r"C:\Users\Mostafa\Desktop\zero_shot_nutrient_association\data\ocr_output"

    run_ocr_on_folder(IMAGE_FOLDER, OUTPUT_FOLDER)