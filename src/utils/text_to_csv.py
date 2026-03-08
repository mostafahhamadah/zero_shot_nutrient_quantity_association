from pathlib import Path
import csv


EXPECTED_HEADER = [
    "image_id",
    "nutrient",
    "quantity",
    "unit",
    "context",
    "nrv_percent",
    "serving_size",
]


def normalize_line(line: str) -> str:
    return line.strip()


def find_header_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        cols = [c.strip() for c in line.split(",")]
        if cols == EXPECTED_HEADER:
            return i
    raise ValueError("Expected header not found in the text file.")


def parse_annotation_text_flexible(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        raw_lines = [normalize_line(line) for line in f if normalize_line(line)]

    header_index = find_header_index(raw_lines)
    lines = raw_lines[header_index:]

    header = [c.strip() for c in lines[0].split(",")]
    num_columns = len(header)

    rows = []
    for i, line in enumerate(lines[1:], start=header_index + 2):
        parts = [part.strip() for part in line.split(",")]

        if len(parts) != num_columns:
            print(f"Skipping malformed line {i}: {line}")
            continue

        rows.append(parts)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"CSV created successfully: {output_file}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    input_txt = BASE_DIR / "data" / "annotations" / "annotations_raw.txt"
    output_csv = BASE_DIR / "data" / "annotations" / "gold_annotations.csv"

    parse_annotation_text_flexible(input_txt, output_csv)