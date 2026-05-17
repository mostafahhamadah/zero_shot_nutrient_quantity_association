"""
run_vlm_experiment.py
======================
VLM-Assisted Pipeline — Experiment Runner
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PIPELINE
--------
  Stage 1   — OCR              (PaddleOCR)
  Stage 2   — OCR Corrector V2
  Stage 3   — EmbeddingSemanticClassifier  (embedding_only)
  Stage 3.5 — Token Enricher
  Stage 5   — VLM Association   ← REPLACES Graph + rule-based association
  Stage 6   — LLM-Assisted Evaluation

Note: Stage 4 (Graph Construction) is SKIPPED — the VLM uses the image
directly to verify spatial relationships, making graph edges redundant.

Default backend is now LM Studio via OpenAI-compatible API (port 1234).
Use --vlm-backend ollama or --vlm-backend hf to switch backends.

Usage:
    # LM Studio (default)
    python run_vlm_experiment.py --experiment vlm_lmstudio_exp01

    # Ollama (legacy)
    python run_vlm_experiment.py --experiment vlm_ollama_exp01 --vlm-backend ollama

    # Hugging Face cloud
    python run_vlm_experiment.py --experiment vlm_hf_exp01 --vlm-backend hf

    # Subset
    python run_vlm_experiment.py --experiment smoke --images 1,4,33

    # Skip LLM evaluator (fast rule-based only)
    python run_vlm_experiment.py --experiment vlm_lmstudio_exp01 --no-llm
"""

import sys
import csv
import json
import argparse
import time
import importlib.util
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps

sys.path.insert(0, ".")


# ── Dynamic module loader ─────────────────────────────────────────────────────

def _load_module(name, filepath):
    filepath = str(filepath)
    spec = importlib.util.spec_from_file_location(name, filepath)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {filepath}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_existing_file(candidates, label):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n" +
        "\n".join(f"  - {Path(p)}" for p in candidates)
    )


_base = Path(__file__).resolve().parent


# ── Stage imports ─────────────────────────────────────────────────────────────

from src.ocr.paddleocr_runner      import run_ocr_on_image
from src.utils.token_enricher      import TokenEnricher
from src.evaluation.llm_evaluator  import LLMTupleEvaluator
from src.matching.vlm_association  import VLMAssociator


# ── Stage 2 Corrector V2 import ───────────────────────────────────────────────

try:
    from src.utils.paddleOCR_corrector_v2 import correct_tokens
    CORRECTOR_SOURCE = "src.utils.paddleOCR_corrector_v2"
except Exception:
    _corr_path = _first_existing_file(
        [
            _base / "src" / "utils" / "paddleOCR_corrector_v2.py",
            _base / "paddleOCR_corrector_v2.py",
        ],
        label="paddleOCR_corrector_v2.py",
    )

    _corr_module = _load_module("paddleOCR_corrector_v2", _corr_path)
    correct_tokens = _corr_module.correct_tokens
    CORRECTOR_SOURCE = str(_corr_path)


def run_corrector_v2(tokens):
    """
    Robust wrapper for paddleOCR_corrector_v2.correct_tokens.

    Supports both possible signatures:
        correct_tokens(tokens, return_log=True)
    and:
        correct_tokens(tokens)

    Returns:
        corrected_tokens, correction_log
    """
    try:
        result = correct_tokens(tokens, return_log=True)
    except TypeError:
        result = correct_tokens(tokens)

    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]

    return result, []


# ── Stage 3 Embedding classifier import ───────────────────────────────────────

_emb_module = _load_module(
    "emb_classifier",
    _base / "src" / "classification" / "embedding_semantic_classifier.py",
)

EmbeddingSemanticClassifier = _emb_module.EmbeddingSemanticClassifier


# ── CSV annotation loader ─────────────────────────────────────────────────────

def _clean(value):
    return str(value if value is not None else "").strip()


def _row_get(row, *names):
    """
    Case-insensitive safe getter for CSV columns.
    """
    lowered = {str(k).strip().lower(): v for k, v in row.items()}

    for name in names:
        if name in row and _clean(row.get(name)):
            return _clean(row.get(name))

        key = name.lower()
        if key in lowered and _clean(lowered[key]):
            return _clean(lowered[key])

    return ""


def _normalize_image_id(raw_image_id, image_lookup=None):
    """
    Normalizes image ids so that:
        1.png               -> 1.png
        data/raw/1.png      -> 1.png
        1                   -> 1.png if 1.png exists in RAW_DIR
        101.PNG             -> 101.png
    """
    image_lookup = image_lookup or {}

    raw = _clean(raw_image_id)
    if not raw:
        return ""

    name = Path(raw).name
    p = Path(name)

    stem = p.stem
    suffix = p.suffix.lower()

    if stem in image_lookup:
        return image_lookup[stem]

    if suffix:
        return stem + suffix

    return stem


def load_gt_from_csv(csv_path, image_lookup=None, allowed_image_ids=None):
    """
    Load ground-truth tuples from test_set_normalized.csv.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    allowed_image_ids = set(allowed_image_ids or [])
    rows = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"GT CSV has no header row: {csv_path}")

        for i, row in enumerate(reader, start=2):
            raw_image_id = _row_get(row, "image_id", "filename", "file", "image", "img")
            image_id = _normalize_image_id(raw_image_id, image_lookup=image_lookup)

            if not image_id:
                print(f"[load_gt_csv] WARN row {i}: missing image_id, skipped")
                continue

            if allowed_image_ids and image_id not in allowed_image_ids:
                continue

            nutrient = _row_get(row, "nutrient", "nutrient_name", "name")
            quantity = _row_get(row, "quantity", "amount", "value")
            unit     = _row_get(row, "unit")
            context  = _row_get(row, "context", "context_full", "normalized_context")

            serving_size = _row_get(row, "serving_size")
            if serving_size and context and serving_size not in context:
                context = f"{context} ({serving_size})"

            rows.append({
                "image_id": image_id,
                "nutrient": nutrient,
                "quantity": quantity,
                "unit": unit,
                "context": context,
            })

    return rows


# ── VLM image preparation ─────────────────────────────────────────────────────

def prepare_vlm_image_copy(img_path, out_dir, max_side=896):
    """
    Creates a resized RGB JPG copy for the VLM.

    The associator itself will also resize internally to its
    configured image_max_side. When both values match (default
    896 on each side), the associator's internal resize is a
    no-op pass-through. The disk copy is kept for debugging.

    OCR still uses the original image.
    """
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        original_w, original_h = im.size

        if max(original_w, original_h) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        resized_w, resized_h = im.size

        out_path = out_dir / f"{img_path.stem}_vlm_{resized_w}x{resized_h}.jpg"
        im.save(out_path, format="JPEG", quality=95, optimize=True)

    return out_path, {
        "original_width": original_w,
        "original_height": original_h,
        "vlm_width": resized_w,
        "vlm_height": resized_h,
        "vlm_image_path": str(out_path),
    }


# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR     = "data/raw"
CONF_THRESH = 0.30

FIELDNAMES = ["image_id", "nutrient", "quantity", "unit", "context"]

DEFAULT_GT_CSV = "test_set_normalized.csv"

TUNED_THRESHOLD = 0.59
TUNED_MARGIN    = 0.04


# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="VLM-Assisted Experiment Runner")

parser.add_argument("--experiment", required=True)

parser.add_argument(
    "--images",
    default=None,
    help="Comma-separated image stems, e.g. 1,6,101"
)

parser.add_argument(
    "--compare",
    default=None,
    help="Previous experiment name to compare against"
)

parser.add_argument("--notes", default="")

parser.add_argument(
    "--threshold",
    type=float,
    default=TUNED_THRESHOLD,
    help=f"NUTRIENT cosine threshold. Default: {TUNED_THRESHOLD}"
)

parser.add_argument(
    "--margin",
    type=float,
    default=TUNED_MARGIN,
    help=f"NUTRIENT vs second-best margin. Default: {TUNED_MARGIN}"
)

parser.add_argument(
    "--no-llm",
    action="store_true",
    help="Skip LLM evaluation — use fast rule-based pass only"
)

# ── Backend selection ───────────────────────────────────────────────
parser.add_argument(
    "--vlm-backend",
    choices=["openai_compat", "lmstudio", "ollama", "hf"],
    default="openai_compat",
    help="VLM backend: openai_compat (LM Studio, default), ollama, or hf."
)

# ── OpenAI-compatible endpoint (LM Studio by default) ───────────────
parser.add_argument(
    "--openai-base-url",
    default="http://127.0.0.1:1234/v1/chat/completions",
    help="OpenAI-compatible chat completions endpoint. Default: LM Studio."
)
parser.add_argument(
    "--openai-model",
    default="gemma-3b",
    help="Model identifier for the OpenAI-compatible endpoint. Default: google/gemma 3b."
)
parser.add_argument(
    "--openai-auth-env",
    default=None,
    help="Environment variable for bearer token (None = no auth, fine for LM Studio)."
)

# ── Ollama legacy ──────────────────────────────────────────────────
parser.add_argument(
    "--model",
    default="qwen2.5vl:7b-q4_K_M",
    help="Ollama VLM model name. Default: qwen2.5vl:7b-q4_K_M"
)
parser.add_argument(
    "--ollama-num-ctx",
    type=int,
    default=8192,
    help="Ollama num_ctx (KV cache cap). Default: 8192"
)

# ── HuggingFace ─────────────────────────────────────────────────────
parser.add_argument(
    "--hf-model",
    default="zai-org/GLM-4.5V:fastest",
    help="Hugging Face VLM model. Default: zai-org/GLM-4.5V:fastest"
)
parser.add_argument(
    "--hf-token-env",
    default="HF_TOKEN",
    help="Environment variable containing your Hugging Face token. Default: HF_TOKEN"
)

# ── Shared generation ───────────────────────────────────────────────
parser.add_argument(
    "--vlm-max-tokens",
    type=int,
    default=4096,
    help="Maximum generated tokens from VLM. Default: 4096"
)

parser.add_argument(
    "--vlm-retry-max",
    type=int,
    default=0,
    help="Number of VLM retries after parse failure or timeout. Default: 0"
)

parser.add_argument(
    "--save-vlm-prompts",
    action="store_true",
    help="Save the exact VLM system/user prompts for debugging"
)

parser.add_argument(
    "--eval-model",
    default="qwen2.5:3b",
    help="Ollama text model for LLM-assisted evaluation. Default: qwen2.5:3b"
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.1,
    help="VLM temperature. Default: 0.1"
)

parser.add_argument(
    "--timeout",
    type=int,
    default=600,
    help="Per-image VLM timeout in seconds. Default: 600"
)

parser.add_argument(
    "--gt-csv",
    default=DEFAULT_GT_CSV,
    help="Ground-truth CSV file. Default: test_set_normalized.csv"
)

parser.add_argument(
    "--vlm-max-side",
    type=int,
    default=896,
    help="Maximum width/height for the image sent to the VLM. Default: 896"
)

# ── Prompt-content toggles (Layer 1 fixes) ──────────────────────────
parser.add_argument(
    "--vlm-include-nrv",
    action="store_true",
    help="Include NRV tokens in the VLM token table (off by default — reduces prompt size)."
)
parser.add_argument(
    "--vlm-include-structure",
    action="store_true",
    help="Include enricher structural metadata (row/col/stream) in token table (off by default)."
)
parser.add_argument(
    "--vlm-include-noise",
    action="store_true",
    help="Include NOISE tokens in token table (off by default)."
)

args = parser.parse_args()


EXPERIMENT = args.experiment
USE_LLM    = not args.no_llm
THRESHOLD  = args.threshold
MARGIN     = args.margin

OUT_DIR = Path(f"outputs/{EXPERIMENT}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TUPLES_CSV = OUT_DIR / "tuples.csv"
LOG_FILE   = OUT_DIR / "run_log.txt"
TIME_CSV   = OUT_DIR / "processing_times.csv"

VLM_IMAGE_DIR = OUT_DIR / "vlm_resized_images"
VLM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Build allowed labels (Layer 1 F1.1) ─────────────────────────────

ALLOWED_LABELS = {"NUTRIENT", "QUANTITY", "UNIT", "CONTEXT"}
if args.vlm_include_nrv:
    ALLOWED_LABELS.add("NRV")


# ── Resolve GT CSV path ───────────────────────────────────────────────────────

GT_CSV = _first_existing_file(
    [
        args.gt_csv,
        _base / args.gt_csv,
        _base / "test_set_normalized.csv",
        _base / "data" / "test_set_normalized.csv",
        _base / "data" / "annotations" / "test_set_normalized.csv",
    ],
    label="test_set_normalized.csv",
)


# ── Discover images ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

raw_dir_path = Path(RAW_DIR)

if args.images:
    IMAGE_FILES = []

    for stem in (i.strip() for i in args.images.split(",")):
        found = [
            f for f in raw_dir_path.iterdir()
            if f.stem == stem and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if found:
            IMAGE_FILES.append(found[0])
        else:
            print(f"WARN  No image found for stem: {stem}")

    IMAGE_FILES.sort(
        key=lambda f: (
            int(f.stem) if f.stem.isdigit() else float("inf"),
            f.name
        )
    )

else:
    IMAGE_FILES = sorted(
        [
            f for f in raw_dir_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda f: (
            int(f.stem) if f.stem.isdigit() else float("inf"),
            f.name
        )
    )


IMAGE_LOOKUP = {
    f.stem: f.stem + f.suffix.lower()
    for f in IMAGE_FILES
}

PROCESSED_IMAGE_IDS = {
    f.stem + f.suffix.lower()
    for f in IMAGE_FILES
}


# ── Identify active VLM model for display ─────────────────────────────────────

if args.vlm_backend in ("openai_compat", "lmstudio"):
    ACTIVE_VLM_LABEL = f"{args.openai_model} @ {args.openai_base_url}"
elif args.vlm_backend == "hf":
    ACTIVE_VLM_LABEL = f"{args.hf_model} (HuggingFace Inference)"
else:
    ACTIVE_VLM_LABEL = f"{args.model} (Ollama)"


# ── Header ────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 75}")
print(f"  EXPERIMENT : {EXPERIMENT}")
print(f"  Pipeline   : VLM-Assisted (PaddleOCR + Corrector V2 + Embedding Clf + Enricher + VLM)")
print(f"  Schema     : nutrient | quantity | unit | context")
print(f"  Corrector  : {CORRECTOR_SOURCE}")
print(f"  Classifier : embedding_only  (BGE-M3, t={THRESHOLD}, m={MARGIN})")
print(f"  VLM Backend: {args.vlm_backend}")
print(f"  VLM Model  : {ACTIVE_VLM_LABEL}")
print(f"  VLM Image  : resized JPG, max side {args.vlm_max_side}px")
print(f"  VLM Output : max_tokens={args.vlm_max_tokens}, retry_max={args.vlm_retry_max}")
print(f"  Token Table: labels={sorted(ALLOWED_LABELS)} | structure={args.vlm_include_structure}")
print(f"  Evaluator  : {f'LLM-Assisted ({args.eval_model})' if USE_LLM else 'Fast rule-based only'}")
print(f"  Images     : {len(IMAGE_FILES)} discovered in {RAW_DIR}/")
print(f"  GT CSV     : {GT_CSV}")
print(f"  Output     : outputs/{EXPERIMENT}/")
if args.notes:
    print(f"  Notes      : {args.notes}")
print(f"{'=' * 75}\n")


# ── Pipeline instances ────────────────────────────────────────────────────────

classifier = EmbeddingSemanticClassifier(
    mode="embedding_only",
    nutrient_threshold=THRESHOLD,
    margin=MARGIN,
    confidence_threshold=CONF_THRESH,
)

enricher = TokenEnricher()

vlm_config = {
    "backend": args.vlm_backend,

    # OpenAI-compatible (LM Studio default)
    "openai_base_url": args.openai_base_url,
    "openai_model": args.openai_model,
    "openai_auth_env": args.openai_auth_env,

    # Ollama
    "model": args.model,
    "ollama_num_ctx": args.ollama_num_ctx,

    # Hugging Face
    "hf_model": args.hf_model,
    "hf_token_env": args.hf_token_env,

    # Shared generation
    "temperature": args.temperature,
    "timeout_s": args.timeout,
    "max_tokens": args.vlm_max_tokens,
    "num_predict": args.vlm_max_tokens,
    "retry_max": args.vlm_retry_max,
    "image_shrink_on_retry": True,

    # Image
    "image_max_side": args.vlm_max_side,
    "image_jpeg_quality": 90,

    # Prompt control
    "include_noise": args.vlm_include_noise,
    "include_geometry": True,
    "include_structure": args.vlm_include_structure,
    "allowed_labels": ALLOWED_LABELS,

    # Debugging
    "save_prompts": args.save_vlm_prompts,
    "prompt_debug_dir": str(OUT_DIR / "debug_vlm_prompts"),
}

vlm_associator = VLMAssociator(config=vlm_config)


# ── Tracking ──────────────────────────────────────────────────────────────────

all_tuples = []
diagnostics = {}
processing_time_rows = []

ok = 0
skip = 0
err = 0

total_vlm_time = 0.0
total_processing_time = 0.0


# ── Main loop ─────────────────────────────────────────────────────────────────

for img_path in IMAGE_FILES:
    image_key = img_path.stem + img_path.suffix.lower()
    image_start_time = time.perf_counter()

    try:
        # Stage 1 — OCR
        tokens = run_ocr_on_image(str(img_path))

        ocr_total = len(tokens)
        ocr_low_conf = sum(
            1 for t in tokens
            if t.get("conf", 1.0) < CONF_THRESH
        )

        # Stage 2 — PaddleOCR Corrector V2
        corrected, correction_log = run_corrector_v2(tokens)

        # Stage 3 — Embedding classifier
        labeled = classifier.classify_all(corrected)
        clf_counts = Counter(t["label"] for t in labeled)

        clf_emb = sum(
            1 for t in labeled
            if t.get("classification_method") == "embedding"
        )

        # Stage 3.5 — Token Enricher
        enriched = enricher.enrich(labeled)
        enr_diag = enricher.diagnostics

        # Prepare resized JPG for VLM debug + as input to the associator.
        # The associator will also resize internally; at matching max_side
        # values this is a pass-through.
        vlm_image_path, vlm_image_info = prepare_vlm_image_copy(
            img_path=img_path,
            out_dir=VLM_IMAGE_DIR,
            max_side=args.vlm_max_side,
        )

        # Stage 5-VLM — VLM Association
        tuples = vlm_associator.extract(
            enriched_tokens=enriched,
            image_path=str(vlm_image_path),
            image_id=image_key,
        )

        tuples = tuples or []

        vlm_diag = vlm_associator.diagnostics
        vlm_elapsed_s = float(vlm_diag.get("elapsed_s", 0) or 0)

        image_processing_time = time.perf_counter() - image_start_time

        total_vlm_time += vlm_elapsed_s
        total_processing_time += image_processing_time

        all_tuples.extend(tuples)

        diagnostics[image_key] = {
            "ocr_total":           ocr_total,
            "ocr_low_conf":        ocr_low_conf,

            "corrector_v2_edits":  len(correction_log),

            "clf_nutrient":        clf_counts.get("NUTRIENT", 0),
            "clf_quantity":        clf_counts.get("QUANTITY", 0),
            "clf_unit":            clf_counts.get("UNIT",     0),
            "clf_context":         clf_counts.get("CONTEXT",  0),
            "clf_noise":           clf_counts.get("NOISE",    0),
            "clf_unknown":         clf_counts.get("UNKNOWN",  0),
            "clf_via_embedding":   clf_emb,
            "clf_threshold":       THRESHOLD,
            "clf_margin":          MARGIN,

            "enr_rows":            enr_diag.get("num_rows",         0),
            "enr_columns":         enr_diag.get("num_columns",      0),
            "enr_dosage_streams":  enr_diag.get("dosage_streams",   0),
            "enr_headers":         enr_diag.get("headers_detected", 0),
            "enr_rank_consistent": enr_diag.get("rank_consistent",  False),

            "vlm_backend":         args.vlm_backend,
            "vlm_model":           vlm_diag.get("model",     ""),
            "vlm_tuples":          vlm_diag.get("tuples",    len(tuples)),
            "vlm_attempts":        vlm_diag.get("attempts",  0),
            "vlm_elapsed_s":       vlm_elapsed_s,
            "vlm_last_error":      vlm_diag.get("last_error", None),
            "vlm_image_max_side":  vlm_diag.get("image_max_side", args.vlm_max_side),
            "vlm_prompt_chars":    vlm_diag.get("prompt_char_len", 0),

            "vlm_original_width":  vlm_image_info["original_width"],
            "vlm_original_height": vlm_image_info["original_height"],
            "vlm_width":           vlm_image_info["vlm_width"],
            "vlm_height":          vlm_image_info["vlm_height"],
            "vlm_image_path":      vlm_image_info["vlm_image_path"],

            "total_processing_time_s": image_processing_time,
        }

        processing_time_rows.append({
            "image_id": image_key,
            "status": "ok",
            "ocr_tokens": ocr_total,
            "corrector_edits": len(correction_log),
            "predicted_tuples": len(tuples),
            "clf_nutrient": clf_counts.get("NUTRIENT", 0),
            "clf_quantity": clf_counts.get("QUANTITY", 0),
            "clf_unit": clf_counts.get("UNIT", 0),
            "clf_context": clf_counts.get("CONTEXT", 0),
            "clf_via_embedding": clf_emb,
            "vlm_backend": args.vlm_backend,
            "vlm_original_width": vlm_image_info["original_width"],
            "vlm_original_height": vlm_image_info["original_height"],
            "vlm_width": vlm_image_info["vlm_width"],
            "vlm_height": vlm_image_info["vlm_height"],
            "vlm_image_path": vlm_image_info["vlm_image_path"],
            "vlm_time_s": round(vlm_elapsed_s, 3),
            "vlm_attempts": vlm_diag.get("attempts", 0),
            "vlm_last_error": vlm_diag.get("last_error", "") or "",
            "total_processing_time_s": round(image_processing_time, 3),
            "error": "",
        })

        print(
            f"OK    {image_key:<16} → {len(tuples):>3} tuples  "
            f"| NUTR:{clf_counts.get('NUTRIENT', 0):>3}  "
            f"EMB:{clf_emb:>3}  "
            f"QTY:{clf_counts.get('QUANTITY', 0):>3}  "
            f"CORR:{len(correction_log):>3}  "
            f"VLM_IMG:{vlm_image_info['vlm_width']}x{vlm_image_info['vlm_height']}  "
            f"VLM:{vlm_elapsed_s:.1f}s  "
            f"TOTAL:{image_processing_time:.1f}s  "
            f"attempts:{vlm_diag.get('attempts', 1)}"
        )

        ok += 1

    except Exception as e:
        import traceback

        image_processing_time = time.perf_counter() - image_start_time
        total_processing_time += image_processing_time

        processing_time_rows.append({
            "image_id": image_key,
            "status": "error",
            "ocr_tokens": "",
            "corrector_edits": "",
            "predicted_tuples": "",
            "clf_nutrient": "",
            "clf_quantity": "",
            "clf_unit": "",
            "clf_context": "",
            "clf_via_embedding": "",
            "vlm_backend": args.vlm_backend,
            "vlm_original_width": "",
            "vlm_original_height": "",
            "vlm_width": "",
            "vlm_height": "",
            "vlm_image_path": "",
            "vlm_time_s": "",
            "vlm_attempts": "",
            "vlm_last_error": "",
            "total_processing_time_s": round(image_processing_time, 3),
            "error": str(e),
        })

        print(f"ERR   {image_key}  → {e}  | TOTAL:{image_processing_time:.1f}s")
        traceback.print_exc()
        err += 1


# ── Save prediction tuples CSV ────────────────────────────────────────────────

with open(TUPLES_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows([
        {k: t.get(k, "") for k in FIELDNAMES}
        for t in all_tuples
    ])


# ── Save per-image processing times CSV ───────────────────────────────────────

with open(TIME_CSV, "w", newline="", encoding="utf-8") as f:
    time_fieldnames = [
        "image_id",
        "status",
        "ocr_tokens",
        "corrector_edits",
        "predicted_tuples",
        "clf_nutrient",
        "clf_quantity",
        "clf_unit",
        "clf_context",
        "clf_via_embedding",
        "vlm_backend",
        "vlm_original_width",
        "vlm_original_height",
        "vlm_width",
        "vlm_height",
        "vlm_image_path",
        "vlm_time_s",
        "vlm_attempts",
        "vlm_last_error",
        "total_processing_time_s",
        "error",
    ]

    writer = csv.DictWriter(f, fieldnames=time_fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(processing_time_rows)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print(f"  Processed: {ok}  |  Skipped: {skip}  |  Errors: {err}")
print(f"  Total tuples: {len(all_tuples)}")
print(
    f"  Total VLM time: {total_vlm_time:.1f}s  "
    f"({total_vlm_time / max(ok, 1):.1f}s/successful image avg)"
)
print(
    f"  Total processing time: {total_processing_time:.1f}s  "
    f"({total_processing_time / max(ok + err, 1):.1f}s/image avg)"
)
print(f"{'=' * 65}\n")


# ── Evaluation ────────────────────────────────────────────────────────────────

print("Running evaluation...\n")

gt_rows = load_gt_from_csv(
    GT_CSV,
    image_lookup=IMAGE_LOOKUP,
    allowed_image_ids=PROCESSED_IMAGE_IDS,
)

print(f"GT loaded: {len(gt_rows)} tuples from {GT_CSV}\n")

evaluator = LLMTupleEvaluator(
    gt_rows=gt_rows,
    use_llm=USE_LLM,
    model=args.eval_model,
)

metrics = evaluator.run(
    predictions=all_tuples,
    experiment=EXPERIMENT,
    out_dir=OUT_DIR,
    notes=args.notes,
    diagnostics=diagnostics,
)


# ── Save run log ──────────────────────────────────────────────────────────────

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"Experiment  : {EXPERIMENT}\n")
    f.write(f"Pipeline    : VLM-Assisted + Corrector V2 + EmbeddingSemanticClassifier (embedding_only)\n")
    f.write(f"Corrector   : {CORRECTOR_SOURCE}\n")
    f.write(f"Classifier  : embedding_only\n")
    f.write(f"Threshold   : {THRESHOLD}\n")
    f.write(f"Margin      : {MARGIN}\n")
    f.write(f"GT CSV      : {GT_CSV}\n")
    f.write(f"Evaluator   : {f'LLM-Assisted ({args.eval_model})' if USE_LLM else 'Fast rule-based'}\n")
    f.write("Schema      : nutrient|quantity|unit|context\n")
    f.write(f"VLM Backend : {args.vlm_backend}\n")
    f.write(f"VLM Model   : {ACTIVE_VLM_LABEL}\n")
    f.write(f"VLM Image   : resized JPG, max side {args.vlm_max_side}px\n")
    f.write(f"VLM MaxTok  : {args.vlm_max_tokens}\n")
    f.write(f"VLM Retries : {args.vlm_retry_max}\n")
    f.write(f"Token Labels: {sorted(ALLOWED_LABELS)}\n")
    f.write(f"Structure   : {args.vlm_include_structure}\n")
    f.write(f"Timestamp   : {metrics['timestamp']}\n")
    f.write(f"Notes       : {args.notes}\n\n")

    f.write(f"GT tuples            : {metrics.get('gt_tuples', 0)}\n")
    f.write(f"Predicted tuples     : {metrics.get('predicted_tuples', 0)}\n")
    f.write(f"Matched pairs        : {metrics.get('matched_pairs', 0)}\n")
    f.write(f"Full 4F correct      : {metrics.get('full4f_correct', 0)}\n")

    f.write(f"Full Tuple Precision : {metrics['full_tuple_precision'] * 100:.1f}%\n")
    f.write(f"Full Tuple Recall    : {metrics['full_tuple_recall'] * 100:.1f}%\n")
    f.write(f"Full Tuple F1        : {metrics['full_tuple_f1'] * 100:.1f}%\n")
    f.write(f"Nutrient F1          : {metrics['nutrient_f1']:.3f}\n")
    f.write(f"Unit Acc             : {metrics['unit_acc'] * 100:.1f}%\n")
    f.write(f"Context Acc          : {metrics['context_acc'] * 100:.1f}%\n")

    if "quantity_acc" in metrics:
        f.write(f"Quantity Acc         : {metrics['quantity_acc'] * 100:.1f}%\n")

    f.write(f"\nVLM total time        : {total_vlm_time:.1f}s\n")
    f.write(f"VLM avg/success image : {total_vlm_time / max(ok, 1):.1f}s\n")
    f.write(f"Total processing time : {total_processing_time:.1f}s\n")
    f.write(f"Avg processing/image  : {total_processing_time / max(ok + err, 1):.1f}s\n")
    f.write(f"Processing time CSV   : {TIME_CSV}\n")

    if USE_LLM:
        f.write(f"\nLLM calls    : {metrics.get('llm_calls', 0)}\n")
        f.write(f"Fast hits    : {metrics.get('fast_hits', 0)}\n")
        f.write(f"Eval time    : {metrics.get('eval_time_s', 0)}s\n")


# ── Headline ──────────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print(f"  HEADLINE — true hits")
print(f"{'=' * 65}")
print(f"  GT tuples           : {metrics.get('gt_tuples', 0)}")
print(f"  Predicted tuples    : {metrics.get('predicted_tuples', 0)}")
print(f"  TRUE HITS (4F corr) : {metrics.get('full4f_correct', '?')}")
print(f"  Full Tuple F1       : {metrics['full_tuple_f1'] * 100:.2f}%")
print(f"  Nutrient F1         : {metrics['nutrient_f1']:.3f}")
print(f"{'=' * 65}\n")

print(f"Saved: {TUPLES_CSV}")
print(f"Saved: {TIME_CSV}")
print(f"Saved: {LOG_FILE}")
print(f"Saved: outputs/{EXPERIMENT}/evaluation_summary.csv")


# ── Comparison ────────────────────────────────────────────────────────────────

if args.compare:
    cmp_path = Path(f"outputs/{args.compare}/evaluation_results.json")

    if cmp_path.exists():
        with open(cmp_path, encoding="utf-8") as f:
            prev = json.load(f)

        print(f"\n{'=' * 65}")
        print(f"  COMPARISON : {args.compare}  →  {EXPERIMENT}")
        print(f"{'=' * 65}")

        for label, key, up_good in [
            ("TRUE HITS (4F correct)", "full4f_correct",       True),
            ("Full Tuple Precision",   "full_tuple_precision", True),
            ("Full Tuple Recall",      "full_tuple_recall",    True),
            ("Full Tuple F1",          "full_tuple_f1",        True),
            ("Nutrient F1",            "nutrient_f1",          True),
            ("Unit Match Acc",         "unit_acc",             True),
            ("Quantity Match Acc",     "quantity_acc",         True),
            ("Context Match Acc",      "context_acc",          True),
            ("Matched Pairs",          "matched_pairs",        True),
            ("Predicted Tuples",       "predicted_tuples",     False),
        ]:
            old = prev.get(key, 0)
            new = metrics.get(key, 0)

            try:
                d = float(new) - float(old)
                sign = "+" if d >= 0 else ""
                arr = (
                    "↑" if (d > 0) == up_good and d != 0
                    else ("↓" if d != 0 else "=")
                )

                print(
                    f"  {label:<26}  {old:.3f}  →  {new:.3f}"
                    f"  ({sign}{d:.3f}) {arr}"
                )

            except Exception:
                print(f"  {label:<26}  {old}  →  {new}")

        print(f"{'=' * 65}\n")

    else:
        print(f"\n  [compare] Not found: {cmp_path}")