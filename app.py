"""
app.py  —  Pipeline Inspection UI  (v4 — Embedding + GLiNER + Dual Evaluation)
Zero-Shot Nutrient Extraction | Moustafa Hamada | M.Sc. AI & Data Science | THD + USB

Run from project root:
    streamlit run app.py

Pages
─────
  🏠  Run Pipeline           — configure + run, see GT vs predicted
  📝  Annotation Editor      — edit GT JSON annotations inline and save
  🔍  Stage 1 · OCR          — raw OCR tokens + bbox overlay
  ✂️  Stage 2 · Corrector    — before/after token diff
  📄  Stage 3A · Serializer  — text_serializer output per image
  🏷️  Stage 3 · Classifier   — labelled tokens + colour overlay (Rule/Embedding/GLiNER/Qwen)
  🧬  Stage 3.5 · Enricher   — geometry-aware rows/columns/streams
  🕸️  Stage 4 · Graph        — nodes, edge types, relation explanations
  🔗  Stage 5 · Association  — extracted tuples + diff vs GT
  🧪  Evaluation             — run BOTH fast + LLM evaluators on current pipeline/prev exp
  📊  Previous Experiments   — load any experiment output, filter + diff table
  📓  Notes                  — all saved per-image/per-stage comments

v4 changes:
  • Embedding classifier entries added (BGE-M3 Hybrid + Embedding-Only)
  • Image format support extended (.webp .tif added)
  • New "Evaluation" page with dual fast/LLM pass + comparison
  • "LLM Evaluator" page renamed "Previous Experiments"
  • Stage 4 Graph page: edge-type explanation cards
  • Embedding classifier badge (purple)
"""


from __future__ import annotations

# ══ CRITICAL: Pre-load torch DLLs on main thread ══
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

try:
    import torch
    import torch.nn.functional
    _ = torch.zeros(1)
except Exception as _e:
    print(f"[torch preload error] {_e}")

try:
    import sentence_transformers
except Exception as _e:
    print(f"[sentence_transformers preload] {_e}")



import importlib.util
import json as _json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# ════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT   = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
GT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations"
NOTES_FILE     = PROJECT_ROOT / "data" / "pipeline_notes.tsv"

CONF_THRESH = 0.30
GT_COLS     = ["nutrient", "quantity", "unit", "context"]

# Supported image extensions — extended for v4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

METRIC_LABELS = {
    "nutrient_f1":   "Nutrient F1",
    "quantity_acc":  "Qty Acc",
    "unit_acc":      "Unit Acc",
    "context_acc":   "Context Acc",
    "full3f_f1":     "Full F1 (3f)",
    "full_tuple_f1": "Full F1 (4f)",
}
EXP1_BASELINE = {
    "nutrient_f1":  0.465, "quantity_acc": 0.263,
    "unit_acc":     0.629, "context_acc":  0.293, "full_tuple_f1": 0.034,
}

C_CORRECT   = "#d4f5d4"
C_WRONG     = "#ffd6d6"
C_FUZZY     = "#fff9c4"
C_GT_ONLY   = "#f0f0f0"
C_PRED_ONLY = "#d6eaff"

LABEL_COLOURS = {
    "NUTRIENT": (74,  144, 217),
    "QUANTITY": (92,  184,  92),
    "UNIT":     (240, 173,  78),
    "CONTEXT":  (155,  89, 182),
    "NOISE":    (170, 170, 170),
    "UNKNOWN":  (231,  76,  60),
}

STAGE_NAMES = [
    "Stage 1 · OCR",
    "Stage 2 · Corrector",
    "Stage 3A · Serializer",
    "Stage 3 · Classifier",
    "Stage 3.5 · Token Enricher",
    "Stage 4 · Graph",
    "Stage 5 · Association",
    "Evaluation",
]

# ════════════════════════════════════════════════════════════════════════════
# EDGE TYPE DOCUMENTATION (Stage 4 Graph page)
# ════════════════════════════════════════════════════════════════════════════

EDGE_TYPE_DOCS: Dict[str, Dict[str, str]] = {
    # V1 edges
    "SAME_ROW": {
        "title":   "SAME_ROW — tokens on the same visual row",
        "what":    "Connects any two tokens whose vertical centres (cy) are close "
                   "(default: within 25px). Bidirectional.",
        "why":     "Nutrient labels are tabular — a nutrient name, its quantity, "
                   "and its unit typically share a single row. SAME_ROW is the "
                   "primary mechanism for linking 'Magnesium' → '400' → 'mg'.",
        "example": "'Magnesium' ↔ '400' ↔ 'mg'   (three tokens on one row)",
    },
    "SAME_COL": {
        "title":   "SAME_COL — tokens in the same visual column",
        "what":    "Connects any two tokens whose horizontal centres (cx) are close "
                   "(default: within 20px). Bidirectional.",
        "why":     "Multi-column labels (per 100g | per serving) stack quantities "
                   "vertically under their headers. SAME_COL enables fallback "
                   "matching when SAME_ROW fails due to OCR misalignment.",
        "example": "'per 100g' ↕ '400' ↕ '200'   (header + two data rows)",
    },
    "ADJACENT": {
        "title":   "ADJACENT — physically close, not row/col-aligned",
        "what":    "Connects tokens within ~60px bbox gap that are NOT already linked "
                   "by SAME_ROW or SAME_COL. Bidirectional.",
        "why":     "Catches the rare case where a unit sits diagonally adjacent "
                   "to its quantity due to layout quirks (stacked sub-labels, "
                   "tight spacing, OCR line splits).",
        "example": "'400' (row 5) → 'mg' (row 6, just below-right)",
    },
    "CONTEXT_SCOPE": {
        "title":   "CONTEXT_SCOPE — context header governs data below",
        "what":    "DIRECTED edge from a CONTEXT token (e.g. 'per 100g') to any "
                   "data token below it within 600px vertical range.",
        "why":     "Resolves which contextual frame each nutrient belongs to. "
                   "In a label with both 'per 100g' and 'per serving' headers, "
                   "this edge determines which quantity applies to which context.",
        "example": "'per 100g' → 'Magnesium' → '400' → 'mg'   (all under header)",
    },
    # V2 edges
    "ROW_COMPAT": {
        "title":   "ROW_COMPAT — geometry-aware row compatibility (V2)",
        "what":    "Like SAME_ROW but weighted by row_confidence from the Token "
                   "Enricher. Captures logical row membership even under skew.",
        "why":     "V2 upgrade — the enricher clusters tokens into logical rows "
                   "using direction-aware geometry, handling curved/skewed labels "
                   "that flat cy-comparison misses.",
        "example": "'Magnesium' ↔ '400' ↔ 'mg'   (row_id=3, weight=0.92)",
    },
    "COL_COMPAT": {
        "title":   "COL_COMPAT — geometry-aware column compatibility (V2)",
        "what":    "Like SAME_COL but using logical column_id from the enricher, "
                   "not pixel-position. Weighted by column_confidence.",
        "why":     "V2 upgrade — handles multi-column tables where columns are "
                   "not perfectly vertical (e.g., scanned at an angle).",
        "example": "'per 100g' ↕ '1200' ↕ '45'   (column_id=2, weight=0.88)",
    },
    "DIRECTIONAL_ADJ": {
        "title":   "DIRECTIONAL_ADJ — adjacency with direction awareness (V2)",
        "what":    "Like ADJACENT but respects the primary text direction detected "
                   "by the enricher. Filters out connections that cross row or "
                   "column boundaries non-sensically.",
        "why":     "V2 upgrade — cleaner fallback edges than V1 ADJACENT, "
                   "especially on labels with rotated sub-sections.",
        "example": "'400' → 'mg' (immediately right, within 60px gap)",
    },
    "HEADER_SCOPE": {
        "title":   "HEADER_SCOPE — structural header propagation (V2)",
        "what":    "DIRECTED edge from a CONTEXT header to every token sharing its "
                   "column_id (from the enricher). Structural, not spatial.",
        "why":     "V2 upgrade — replaces V1's 600px vertical rule with column-"
                   "based scoping. More accurate on dense multi-column tables.",
        "example": "'per serving' (col=2) → 'Magnesium' → '200' → 'mg' (all col=2)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# STAGE REGISTRY
# ════════════════════════════════════════════════════════════════════════════

STAGE_REGISTRY: Dict[str, Dict[str, Dict]] = {
    "ocr": {
        "EasyOCR": {
            "label":   "EasyOCR  (Exp 1 — baseline)",
            "rel_path":"src/ocr/ocr_runner.py",
            "symbol":  "run_ocr_on_image", "kind": "fn", "ready": True,
            "note":    "CRAFT + CRNN | lang=[de,en] | conf ≥ 0.30",
        },
        "PaddleOCR": {
            "label":   "PaddleOCR  (Exp 2v1 — PP-OCRv5)",
            "rel_path":"src/ocr/paddleocr_runner.py",
            "symbol":  "run_ocr_on_image", "kind": "fn", "ready": True,
            "note":    "VLM-distilled | 1-EditDist SOTA 0.804 | multi-pass DE+EN+FR",
        },
    },
    "corrector": {
        "Full": {
            "label":   "Full  (Levels 1 + 2 + 3)",
            "rel_path":"src/utils/ocr_corrector.py",
            "symbol":  "OCRCorrector", "method": "correct_all", "kind": "cls", "ready": True,
            "note":    "Char subs + border strip + fuzzy lexicon snap | for EasyOCR",
        },
        "PaddleCorrectorV2": {
            "label":   "PaddleOCR Corrector v2  (C1-C12)",
            "rel_path":"src/utils/paddleocr_corrector.py",
            "symbol":  "correct_tokens", "kind": "paddle-corrector", "ready": True,
            "note":    "C1 decimal · C2/C3 fused split · C6 ENERGIE · C7 context · C8/C9 glyphs · C10 border · C11 fuzzy snap · C12 targeted regex",
        },
        "Partial": {
            "label":   "Partial  (Levels 1 + 2 only)", "rel_path": "", "symbol": "",
            "kind":    "builtin-partial", "ready": True,
            "note":    "Char subs + border strip, no fuzzy snap",
        },
        "Bypass": {
            "label":   "Bypass  (no correction)", "rel_path": "", "symbol": "",
            "kind":    "builtin-bypass", "ready": True, "note": "Pass tokens unchanged",
        },
    },
    "classifier": {
        "Rule-based": {
            "label":   "Rule-based  (Exp 1 — SemanticClassifier v5.4)",
            "rel_path":"src/classification/experiment_01_final_semantic_classifier.py",
            "symbol":  "SemanticClassifier", "kind": "cls", "ready": True,
            "call_method": "classify_all",
            "init_kwargs": {"confidence_threshold": CONF_THRESH},
            "note":    "Priority chain: NOISE→NRV%→UNIT→CONTEXT→QUANTITY→NUTRIENT",
            "uses_serializer": False,
        },
        # ── NEW v4: Embedding classifier ────────────────────────────
        "Embedding-Hybrid": {
            "label":   "Embedding Hybrid  (BGE-M3 + rules)",
            "rel_path":"src/classification/embedding_semantic_classifier.py",
            "symbol":  "EmbeddingSemanticClassifier", "kind": "cls", "ready": True,
            "call_method": "classify_all",
            "init_kwargs": {"mode": "hybrid", "confidence_threshold": CONF_THRESH},
            "note":    "Rules own QTY/UNIT/CTX/NOISE · lexicon first for NUTRIENT · BGE-M3 rescues UNKNOWN tokens",
            "uses_serializer": False,
        },
        "Embedding-Only": {
            "label":   "Embedding Only  (BGE-M3 pure zero-shot)",
            "rel_path":"src/classification/embedding_semantic_classifier.py",
            "symbol":  "EmbeddingSemanticClassifier", "kind": "cls", "ready": True,
            "call_method": "classify_all",
            "init_kwargs": {"mode": "embedding_only", "confidence_threshold": CONF_THRESH},
            "note":    "No NUTRIENT_LEXICON · all NUTRIENT detection via BGE-M3 embeddings · pure zero-shot",
            "uses_serializer": False,
        },
        # ── GLiNER ──────────────────────────────────────────────────
        "GLiNER-BiBase": {
            "label":   "GLiNER Bi-Base  (Exp 3 — Ihor/gliner-biomed-bi-base-v1.0)",
            "rel_path":"src/classification/gliner_classifier.py",
            "symbol":  "GLiNERClassifier", "kind": "cls", "ready": True,
            "call_method": "classify",
            "init_kwargs": {"model_id": "Ihor/gliner-biomed-bi-base-v1.0"},
            "note":    "DeBERTa-v3-base + BGE-small | bi-encoder | Set B labels | threshold 0.3",
            "uses_serializer": True,
        },
        "GLiNER-BiLarge": {
            "label":   "GLiNER Bi-Large  (Exp 3 — Ihor/gliner-biomed-bi-large-v1.0)",
            "rel_path":"src/classification/gliner_classifier.py",
            "symbol":  "GLiNERClassifier", "kind": "cls", "ready": True,
            "call_method": "classify",
            "init_kwargs": {"model_id": "Ihor/gliner-biomed-bi-large-v1.0"},
            "note":    "DeBERTa-v3-large + BGE-base | best zero-shot F1 (58.31%) | Set B labels",
            "uses_serializer": True,
        },
        "GLiNER-UniLarge": {
            "label":   "GLiNER Uni-Large  (Exp 3 ablation — Ihor/gliner-biomed-large-v1.0)",
            "rel_path":"src/classification/gliner_classifier.py",
            "symbol":  "GLiNERClassifier", "kind": "cls", "ready": True,
            "call_method": "classify",
            "init_kwargs": {"model_id": "Ihor/gliner-biomed-large-v1.0"},
            "note":    "DeBERTa-v3-large uni-encoder | best overall zero-shot F1 (59.77%) | Set B labels",
            "uses_serializer": True,
        },
        # ── Qwen ────────────────────────────────────────────────────
        "Qwen-7b": {
            "label":   "Qwen 2.5:7b  (Exp 4 — Ollama local)",
            "rel_path":"src/classification/qwen_classifier.py",
            "symbol":  "QwenClassifier", "kind": "cls", "ready": True,
            "call_method": "classify",
            "init_kwargs": {"model_id": "qwen2.5:7b"},
            "note":    "Multilingual LLM | full serialized text per image | 1 Ollama call | fuzzy remapping 0.65",
            "uses_serializer": True,
        },
        "Qwen-3b": {
            "label":   "Qwen 2.5:3b  (Exp 4 fast — Ollama local)",
            "rel_path":"src/classification/qwen_classifier.py",
            "symbol":  "QwenClassifier", "kind": "cls", "ready": True,
            "call_method": "classify",
            "init_kwargs": {"model_id": "qwen2.5:3b"},
            "note":    "Smaller/faster Qwen variant — use when 7b is too slow on CPU",
            "uses_serializer": True,
        },
    },
    "enricher": {
        "Bypass": {
            "label":   "Bypass  (no enrichment — Exp 1 path)",
            "rel_path":"", "symbol": "", "kind": "builtin-bypass", "ready": True,
            "note":    "Skip token enrichment — use original flat pipeline",
        },
        "Geometry-Aware": {
            "label":   "Geometry-Aware Token Enricher  (V2)",
            "rel_path":"src/utils/token_enricher.py",
            "symbol":  "TokenEnricher", "method": "enrich", "kind": "cls",
            "init_kwargs": {}, "ready": True,
            "note":    "Direction-aware rows/columns · column roles · header scope · dosage streams · rank",
        },
    },
    "graph": {
        "Current": {
            "label":   "Current  (Exp 1 — typed edges)",
            "rel_path":"src/graph/graph_constructor.py",
            "symbol":  "GraphConstructor", "method": "build", "kind": "cls",
            "init_kwargs": {}, "ready": True,
            "note":    "SAME_ROW / SAME_COL / ADJACENT / CONTEXT_SCOPE (600px)",
            "version": "v1",
        },
        "Geometry-Aware V2": {
            "label":   "Geometry-Aware V2  (direction-aware edges)",
            "rel_path":"src/graph/graph_constructor_v2.py",
            "symbol":  "GraphConstructorV2", "method": "build", "kind": "cls",
            "init_kwargs": {}, "ready": True,
            "note":    "ROW_COMPAT / COL_COMPAT / DIRECTIONAL_ADJ / HEADER_SCOPE — requires Token Enricher",
            "version": "v2",
        },
    },
    "association": {
        "Current": {
            "label":   "Current  (Exp 1 — TupleAssociator v6)",
            "rel_path":"src/matching/experiment_01_final_association.py",
            "symbol":  "TupleAssociator", "method": "extract", "kind": "cls",
            "init_kwargs": {}, "ready": True,
            "note":    "x-sort quantities + 4-level unit search + column fallback",
            "version": "v1",
        },
        "Geometry-Aware V2": {
            "label":   "Geometry-Aware V2  (dosage streams + rank + scoring)",
            "rel_path":"src/matching/association_v2.py",
            "symbol":  "TupleAssociatorV2", "method": "extract", "kind": "cls",
            "init_kwargs": {}, "ready": True,
            "note":    "Per-stream matching · geometry+rank scoring · collision resolution — requires Token Enricher",
            "version": "v2",
        },
    },
}

# Classifiers that require text_serializer preprocessing
SERIALIZER_CLASSIFIERS = {k for k,v in STAGE_REGISTRY["classifier"].items()
                           if v.get("uses_serializer", False)}

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Nutrient Inspector",
    page_icon="🧪", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.stDataFrame td, .stDataFrame th {
    font-size: 12px !important;
    padding: 4px 8px !important;
}
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid #dee2e6;
}
.note { font-size: 11px; color: #6c757d; margin-top: -6px; margin-bottom: 10px; }
.stage-banner {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    color: #e2e8f0;
    padding: 16px 24px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 4px solid #4a90d9;
}
.stage-banner h2 { margin: 0; font-size: 1.3rem; }
.stage-banner p  { margin: 4px 0 0; font-size: 0.85rem; color: #94a3b8; }
.io-box {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    border-left: 3px solid #4a90d9;
    font-size: 13px;
}
.io-box.output { border-left-color: #5cb85c; }
/* Serializer line card */
.ser-line {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 4px;
    border-left: 3px solid #89b4fa;
    white-space: pre-wrap;
    word-break: break-all;
}
.ser-line .ln { color: #585b70; margin-right: 10px; user-select: none; }
/* Classifier badges */
.clf-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 8px;
}
.clf-badge.rule   { background:#e8f5e9; color:#2e7d32; }
.clf-badge.gliner { background:#e3f2fd; color:#1565c0; }
.clf-badge.qwen   { background:#fce4ec; color:#880e4f; }
.clf-badge.emb    { background:#f3e5f5; color:#6a1b9a; }
/* Edge type explanation cards */
.edge-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #4a90d9;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.edge-card h4 { margin: 0 0 6px; font-size: 13px; color: #1a365d; }
.edge-card p  { margin: 4px 0; font-size: 12px; line-height: 1.4; color: #333; }
.edge-card .example {
    font-family: 'Courier New', monospace;
    background: #f5f5f5;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    margin-top: 6px;
    color: #444;
}
.edge-card.v1 { border-left-color: #4a90d9; }
.edge-card.v2 { border-left-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MODULE LOADER
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_symbol(rel_path: str, symbol: str) -> Any:
    abs_path = str(PROJECT_ROOT / rel_path)
    spec     = importlib.util.spec_from_file_location(symbol, abs_path)
    module   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)

@st.cache_resource
def _load_serializer():
    return _load_symbol("src/utils/text_serializer.py", "serialize_tokens_for_gliner")

@st.cache_resource
def _load_paragraph_extractor():
    """Returns (extract_fn, should_use_fn) for paragraph-mode fallback."""
    try:
        extract_fn = _load_symbol("src/utils/paragraph_extractor.py", "extract_from_paragraph")
        should_fn  = _load_symbol("src/utils/paragraph_extractor.py", "should_use_paragraph_mode")
        return extract_fn, should_fn
    except Exception as e:
        print(f"[paragraph_extractor] unavailable: {e}")
        return None, None

@st.cache_resource
def _load_sentence_extractor():
    """Returns extract_fn for sentence-level fallback (0-tuple rescue)."""
    try:
        return _load_symbol("src/utils/sentence_extractor.py", "extract_from_sentences")
    except Exception as e:
        print(f"[sentence_extractor] unavailable: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# CORRECTOR BUILTINS
# ════════════════════════════════════════════════════════════════════════════

_CHAR_SUBS = [
    ("rn","m"),("Ug","µg"),("ug","µg"),
    ("Hagnesium","Magnesium"),("Hagneslum","Magnesium"),
    ("Calclum","Calcium"),("Selenlum","Selenium"),
]
_BORDER = set("|{}[]\\")

def _partial_correct(tokens: List[Dict]) -> List[Dict]:
    out = []
    for tok in tokens:
        t = dict(tok); s = t["token"]
        while s and s[0]  in _BORDER: s = s[1:]
        while s and s[-1] in _BORDER: s = s[:-1]
        for old, new in _CHAR_SUBS: s = s.replace(old, new)
        t["token"] = s.strip()
        if t["token"]: out.append(t)
    return out

_MU_ALONE  = re.compile(r'^[µμ]$')
_MU_SUFFIX = re.compile(r'^(\d+[.,]?\d*)\s*[µμ]$')

def _mu_close(t1, t2, max_gap=30, max_row=15):
    return (t2.get("x1",0) - t1.get("x2",0) <= max_gap
            and abs(t1.get("cy",0) - t2.get("cy",0)) <= max_row)

def _mu_merged(t1, t2, text):
    m = dict(t1); m["token"] = text
    m["x2"] = t2.get("x2", t1.get("x2",0))
    m["cx"]  = (t1.get("x1",0) + t2.get("x2",0)) / 2
    m["conf"] = min(t1.get("conf",1.0), t2.get("conf",1.0))
    return m

def _merge_mu_splits(tokens):
    merged, skip = [], set()
    for i, tok in enumerate(tokens):
        if i in skip: continue
        s = tok["token"].strip()
        if i+2 < len(tokens) and (i+1) not in skip and (i+2) not in skip:
            s1, s2 = tokens[i+1]["token"].strip(), tokens[i+2]["token"].strip()
            if re.match(r'^\d+[.,]?\d*$',s) and _MU_ALONE.match(s1) and s2.lower()=="g" and _mu_close(tok,tokens[i+2]):
                merged.append(_mu_merged(tok, tokens[i+2], s+"µg")); skip.update({i+1,i+2}); continue
        if i+1 < len(tokens) and (i+1) not in skip:
            s1 = tokens[i+1]["token"].strip(); m = _MU_SUFFIX.match(s)
            if m and s1.lower()=="g" and _mu_close(tok,tokens[i+1]):
                merged.append(_mu_merged(tok, tokens[i+1], m.group(1)+"µg")); skip.add(i+1); continue
        if i+1 < len(tokens) and (i+1) not in skip:
            s1 = tokens[i+1]["token"].strip()
            if _MU_ALONE.match(s) and s1.lower()=="g" and _mu_close(tok,tokens[i+1]):
                merged.append(_mu_merged(tok, tokens[i+1], "µg")); skip.add(i+1); continue
        merged.append(tok)
    return merged

# ════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_images() -> List[str]:
    """Return all image files in data/raw/ matching supported extensions."""
    if not DATA_RAW.exists(): return []
    return sorted(f.name for f in DATA_RAW.iterdir()
                  if f.suffix.lower() in IMAGE_EXTENSIONS)

def load_gt_df() -> pd.DataFrame:
    rows = []
    if not GT_ANNOTATIONS.exists():
        return pd.DataFrame(columns=["image_id"]+GT_COLS)

    # Build a stem → actual filename map from files on disk
    # This ensures annotations match the real image regardless of extension
    stem_to_filename: Dict[str, str] = {}
    if DATA_RAW.exists():
        for f in DATA_RAW.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                stem_to_filename[f.stem] = f.name

    for jf in sorted(GT_ANNOTATIONS.glob("*.json")):
        try:
            data = _json.loads(jf.read_text(encoding="utf-8"))
            # Match annotation to actual image on disk by stem (regardless of format)
            stem = jf.stem
            if stem in stem_to_filename:
                iid = stem_to_filename[stem]  # e.g. "1.png" not "1.jpeg"
            else:
                iid = Path(data.get("image_id", stem + ".jpeg")).name
            for n in data.get("nutrients", []):
                ctx = str(n.get("context","")).strip()
                ssz = str(n.get("serving_size","") or "").strip()
                rows.append({
                    "image_id": iid,
                    "nutrient": str(n.get("nutrient","")).strip(),
                    "quantity": str(n.get("quantity","")).strip(),
                    "unit":     str(n.get("unit","")).strip(),
                    "context":  f"{ctx} ({ssz})" if ssz else ctx,
                })
        except Exception as e:
            st.warning(f"[load_gt] {jf.name}: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["image_id"]+GT_COLS)
def load_annotation_json(image_name: str) -> Optional[Dict]:
    stem = Path(image_name).stem
    jf   = GT_ANNOTATIONS / f"{stem}.json"
    if not jf.exists(): return None
    try:
        return _json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_annotation_json(image_name: str, data: Dict) -> bool:
    stem = Path(image_name).stem
    jf   = GT_ANNOTATIONS / f"{stem}.json"
    try:
        GT_ANNOTATIONS.mkdir(parents=True, exist_ok=True)
        jf.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Save failed: {e}"); return False

# ════════════════════════════════════════════════════════════════════════════
# NOTES SYSTEM
# ════════════════════════════════════════════════════════════════════════════

def load_notes() -> pd.DataFrame:
    cols = ["timestamp","image_id","stage","note"]
    if not NOTES_FILE.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(NOTES_FILE, sep="\t", names=cols, encoding="utf-8")
        return df
    except Exception:
        return pd.DataFrame(columns=cols)

def save_note(image_id: str, stage: str, note: str):
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{ts}\t{image_id}\t{stage}\t{note.strip()}\n"
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def get_note(image_id: str, stage: str) -> str:
    df = load_notes()
    if df.empty: return ""
    mask = (df["image_id"] == image_id) & (df["stage"] == stage)
    hits = df[mask]
    return str(hits.iloc[-1]["note"]) if not hits.empty else ""

# ════════════════════════════════════════════════════════════════════════════
# IMAGE OVERLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _open_img(image_name: str) -> Optional[Image.Image]:
    p = DATA_RAW / image_name
    if p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception as e:
            st.warning(f"Failed to open {image_name}: {e}")
            return None
    return None

def overlay_bboxes(image_name: str, tokens: List[Dict],
                   colour_fn=None, label_key="label") -> Optional[Image.Image]:
    from PIL import ImageDraw
    img = _open_img(image_name)
    if img is None: return None
    draw = ImageDraw.Draw(img, "RGBA")
    for tok in tokens:
        colour = (colour_fn(tok) if colour_fn
                  else LABEL_COLOURS.get(tok.get(label_key,"UNKNOWN"),(200,200,200)))
        x1,y1,x2,y2 = tok["x1"],tok["y1"],tok["x2"],tok["y2"]
        draw.rectangle([x1,y1,x2,y2], fill=(*colour,40), outline=(*colour,220), width=2)
        lbl = tok.get(label_key,"")
        if lbl: draw.text((x1+2,y1+1), str(lbl)[:3], fill=(*colour,255))
    return img

def overlay_graph(image_name: str, G: Dict) -> Optional[Image.Image]:
    from PIL import ImageDraw
    img = _open_img(image_name)
    if img is None: return None
    draw = ImageDraw.Draw(img, "RGBA")
    node_map = {n["id"]: n for n in G.get("nodes",[])}
    EDGE_COLOURS = {
        "SAME_ROW":      (100,149,237,160),
        "SAME_COL":      (255,165,0,160),
        "ADJACENT":      (60,179,113,160),
        "CONTEXT_SCOPE": (220,80,80,200),
    }
    for edge in G.get("edges",[]):
        src = node_map.get(edge["src"]); dst = node_map.get(edge["dst"])
        if not src or not dst: continue
        sc = ((src["x1"]+src["x2"])//2, (src["y1"]+src["y2"])//2)
        dc = ((dst["x1"]+dst["x2"])//2, (dst["y1"]+dst["y2"])//2)
        col = EDGE_COLOURS.get(edge["type"], (180,180,180,120))
        draw.line([sc, dc], fill=col, width=2 if edge["type"]!="CONTEXT_SCOPE" else 3)
    for node in G.get("nodes",[]):
        col = LABEL_COLOURS.get(node.get("label","UNKNOWN"),(200,200,200))
        x1,y1,x2,y2 = node["x1"],node["y1"],node["x2"],node["y2"]
        draw.rectangle([x1,y1,x2,y2], fill=(*col,30), outline=(*col,200), width=1)
    return img

# ════════════════════════════════════════════════════════════════════════════
# SERIALIZER HELPER — standalone run for inspection page
# ════════════════════════════════════════════════════════════════════════════

def run_serializer_standalone(image_name: str, ocr_key: str, corr_key: str) -> Optional[Dict]:
    img_path = str(DATA_RAW / image_name)
    try:
        cfg = STAGE_REGISTRY["ocr"][ocr_key]
        fn  = _load_symbol(cfg["rel_path"], cfg["symbol"])
        tokens = fn(img_path)

        kind = STAGE_REGISTRY["corrector"][corr_key]["kind"]
        if kind == "builtin-bypass":
            pass
        elif kind == "builtin-partial":
            tokens = _partial_correct(tokens)
        elif kind == "paddle-corrector":
            cfg2 = STAGE_REGISTRY["corrector"][corr_key]
            fn2  = _load_symbol(cfg2["rel_path"], cfg2["symbol"])
            tokens, _ = fn2(tokens, return_log=True)
            tokens = _merge_mu_splits(tokens)
        else:
            cfg2 = STAGE_REGISTRY["corrector"][corr_key]
            Cls  = _load_symbol(cfg2["rel_path"], cfg2["symbol"])
            tokens = Cls().correct_all(tokens)

        serialize_fn = _load_serializer()
        return serialize_fn(tokens), tokens

    except Exception as e:
        st.error(f"Serializer error: {e}")
        return None, []

# ════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_name, ocr, corr, clf, enr, graph, assoc):
    img_path = str(DATA_RAW / image_name)
    diag: Dict = {"stages": {"ocr":ocr,"corrector":corr,"classifier":clf,"enricher":enr,"graph":graph,"association":assoc}}

    # Stage 1: OCR
    try:
        cfg = STAGE_REGISTRY["ocr"][ocr]
        fn  = _load_symbol(cfg["rel_path"], cfg["symbol"])
        t0  = time.time()
        tokens = fn(img_path)
        diag["ocr_total"]        = len(tokens)
        diag["ocr_below_thresh"] = sum(1 for t in tokens if t.get("conf",1) < CONF_THRESH)
        diag["ocr_time_s"]       = round(time.time()-t0,2)
        tokens_raw = list(tokens)
    except Exception as e:
        return [], {**diag,"error":f"Stage 1 OCR: {e}"}, {}

    # Stage 2: Corrector
    corr_log_paddle: List[Dict] = []
    try:
        kind = STAGE_REGISTRY["corrector"][corr]["kind"]
        if   kind == "builtin-bypass":  diag["corrector"] = "BYPASS"
        elif kind == "builtin-partial":
            before = len(tokens); tokens = _partial_correct(tokens)
            diag["corrector"] = f"PARTIAL | dropped {before-len(tokens)}"
        elif kind == "paddle-corrector":
            cfg = STAGE_REGISTRY["corrector"][corr]
            fn  = _load_symbol(cfg["rel_path"], cfg["symbol"])
            tokens, corr_log_paddle = fn(tokens, return_log=True)
            tokens = _merge_mu_splits(tokens)
            nc     = len(corr_log_paddle)
            ns     = sum(1 for e in corr_log_paddle if len(e.get("corrected",[]))>1)
            n_snaps= sum(1 for e in corr_log_paddle if "C11_nutrient_snap" in e.get("rules_fired",[]))
            n_regex= sum(1 for e in corr_log_paddle if "C12_regex_fix"     in e.get("rules_fired",[]))
            diag["corrector"] = f"PADDLE | {nc} changes · {ns} splits · {n_snaps} snaps · {n_regex} regex · {len(tokens)} tokens"
        else:
            cfg = STAGE_REGISTRY["corrector"][corr]
            Cls = _load_symbol(cfg["rel_path"], cfg["symbol"])
            before = len(tokens); tokens = Cls().correct_all(tokens)
            diag["corrector"] = f"FULL | dropped {before-len(tokens)}"
        tokens_corrected = list(tokens)
    except Exception as e:
        diag["corrector_warning"] = str(e); tokens_corrected = list(tokens)

    # Stage 3A: Serializer (only for GLiNER / Qwen classifiers)
    serialized_output: Dict = {}
    clf_cfg = STAGE_REGISTRY["classifier"][clf]
    uses_serializer = clf_cfg.get("uses_serializer", False)

    if uses_serializer:
        try:
            serialize_fn = _load_serializer()
            serialized_output = serialize_fn(tokens_corrected)
            diag["serializer_tokens"] = len(serialized_output.get("token_spans", []))
            diag["serializer_lines"]  = len(serialized_output.get("lines", []))
        except Exception as e:
            diag["serializer_warning"] = str(e)

    # Stage 3: Classifier
    try:
        Cls        = _load_symbol(clf_cfg["rel_path"], clf_cfg["symbol"])
        call_method = clf_cfg.get("call_method", "classify_all")
        init_kwargs = clf_cfg.get("init_kwargs", {})
        t0          = time.time()
        clf_instance = Cls(**init_kwargs)
        classified   = getattr(clf_instance, call_method)(tokens_corrected)
        diag["classifier_time_s"] = round(time.time()-t0,2)
        counts: Dict[str,int] = {}
        for t in classified:
            lbl = t.get("label","UNKNOWN")
            counts[lbl] = counts.get(lbl,0)+1
        diag["classifier_counts"] = counts
        # classifier type detection (rule / embedding / gliner / qwen)
        if "GLiNER" in clf:
            diag["classifier_type"] = "gliner"
        elif "Qwen" in clf:
            diag["classifier_type"] = "qwen"
        elif "Embedding" in clf:
            diag["classifier_type"] = "embedding"
        else:
            diag["classifier_type"] = "rule"
    except Exception as e:
        return [], {**diag,"error":f"Stage 3 Classifier: {e}"}, {}

    # Stage 3.5: Token Enricher
    enriched = classified
    enricher_diag = {}
    try:
        enr_cfg = STAGE_REGISTRY["enricher"][enr]
        if enr_cfg["kind"] != "builtin-bypass":
            Cls = _load_symbol(enr_cfg["rel_path"], enr_cfg["symbol"])
            t0  = time.time()
            enricher_instance = Cls(**enr_cfg.get("init_kwargs", {}))
            enriched = enricher_instance.enrich(classified)
            enricher_diag = enricher_instance.diagnostics
            diag["enricher_time_s"]       = round(time.time()-t0, 2)
            diag["enricher_rows"]         = enricher_diag.get("num_rows", 0)
            diag["enricher_columns"]      = enricher_diag.get("num_columns", 0)
            diag["enricher_streams"]      = enricher_diag.get("dosage_streams", 0)
            diag["enricher_headers"]      = enricher_diag.get("headers_detected", 0)
            diag["enricher_rank_consist"] = enricher_diag.get("rank_consistent", "?")
        else:
            diag["enricher"] = "BYPASS"
    except Exception as e:
        diag["enricher_warning"] = str(e)

    # Stage 4: Graph
    try:
        cfg = STAGE_REGISTRY["graph"][graph]
        Cls = _load_symbol(cfg["rel_path"], cfg["symbol"])
        t0  = time.time()
        G   = Cls(**cfg.get("init_kwargs",{})).build(enriched)
        diag["graph_nodes"]  = G["num_nodes"]; diag["graph_edges"] = G["num_edges"]
        diag["graph_time_s"] = round(time.time()-t0,2)
    except Exception as e:
        return [], {**diag,"error":f"Stage 4 Graph: {e}"}, {}

    # Stage 5: Association
    try:
        cfg    = STAGE_REGISTRY["association"][assoc]
        Cls    = _load_symbol(cfg["rel_path"], cfg["symbol"])
        t0     = time.time()
        tuples = Cls(**cfg.get("init_kwargs",{})).extract(G, image_id=image_name)
        diag["tuples_extracted"]  = len(tuples)
        diag["assoc_time_s"]      = round(time.time()-t0,2)
        diag["assoc_tuples_orig"] = len(tuples)  # pre-fallback count
    except Exception as e:
        return [], {**diag,"error":f"Stage 5 Association: {e}"}, {}

    # Stage 5b: Paragraph-mode fallback (for fused OCR text like 15.png)
    try:
        para_extract, should_use_paragraph = _load_paragraph_extractor()
        if para_extract and should_use_paragraph:
            if should_use_paragraph(classified, len(tuples)):
                para_tuples = para_extract(classified, image_name)
                if len(para_tuples) > len(tuples):
                    diag["paragraph_mode_triggered"] = True
                    diag["paragraph_tuples_before"]  = len(tuples)
                    diag["paragraph_tuples_after"]   = len(para_tuples)
                    tuples = para_tuples
                    diag["tuples_extracted"] = len(tuples)
                else:
                    diag["paragraph_mode_triggered"] = False
                    diag["paragraph_mode_reason"]    = f"extractor produced {len(para_tuples)} ≤ {len(tuples)} existing"
            else:
                diag["paragraph_mode_triggered"] = False
        else:
            diag["paragraph_mode_available"] = False
    except Exception as e:
        diag["paragraph_mode_error"] = str(e)

    # Stage 5c: Sentence-mode fallback (last resort — 0-tuple rescue for paragraph-style labels)
    try:
        if len(tuples) == 0:
            sent_extract = _load_sentence_extractor()
            if sent_extract:
                sent_tuples = sent_extract(classified, image_name)
                if len(sent_tuples) > 0:
                    diag["sentence_mode_triggered"] = True
                    diag["sentence_tuples"]         = len(sent_tuples)
                    tuples = sent_tuples
                    diag["tuples_extracted"] = len(tuples)
    except Exception as e:
        diag["sentence_mode_error"] = str(e)

    _lc = locals()
    inter = {
        "stage1_tokens":       _lc.get("tokens_raw",        []),
        "stage2_tokens":       _lc.get("tokens_corrected",  []),
        "stage2_paddle_log":   corr_log_paddle,
        "stage3a_serialized":  serialized_output,
        "stage3a_used":        uses_serializer,
        "stage3_classified":   _lc.get("classified",        []),
        "stage35_enriched":    _lc.get("enriched",          []),
        "stage35_diag":        enricher_diag,
        "stage35_used":        enr_cfg["kind"] != "builtin-bypass" if 'enr_cfg' in _lc else False,
        "stage4_graph":        _lc.get("G",                 {}),
        "stage5_tuples":       list(tuples),
    }
    return tuples, diag, inter

# ════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def run_evaluation(gt_df, pred_tuples, image_id, use_llm=False):
    image_gt = (
        gt_df[gt_df["image_id"]==image_id][GT_COLS]
        .assign(image_id=image_id).to_dict(orient="records")
    )
    preds = [{**{k:t.get(k,"") for k in GT_COLS},"image_id":image_id} for t in pred_tuples]
    if not image_gt and not preds:
        return {k:0.0 for k in METRIC_LABELS}, pd.DataFrame()
    try:
        LLMEval = _load_symbol("src/evaluation/llm_evaluator.py","LLMTupleEvaluator")
        ev      = LLMEval(gt_rows=image_gt, use_llm=use_llm, model="qwen2.5:7b")
        with tempfile.TemporaryDirectory() as tmp:
            tp  = Path(tmp)
            m   = ev.run(predictions=preds, experiment="ui_eval", out_dir=tp, notes="", diagnostics={})
            detail_csv = tp / "llm_evaluation_details.csv"
            pairs_csv  = tp / "pair_details.csv"
            pair_file  = detail_csv if detail_csv.exists() else pairs_csv
            pdf = pd.read_csv(pair_file) if pair_file.exists() else pd.DataFrame()
        return m, pdf
    except Exception as e:
        st.warning(f"Evaluator error: {e}"); return None, None

def run_full_evaluation(gt_rows, pred_tuples, use_llm=False, model="qwen2.5:7b"):
    """
    Run the full evaluator (all images) — returns metrics dict and pair DataFrame.
    Used by the new Evaluation page for full-dataset fast/LLM comparison.
    """
    try:
        LLMEval = _load_symbol("src/evaluation/llm_evaluator.py","LLMTupleEvaluator")
        ev      = LLMEval(gt_rows=gt_rows, use_llm=use_llm, model=model)
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            m  = ev.run(predictions=pred_tuples, experiment="ui_full_eval",
                        out_dir=tp, notes="", diagnostics={})
            detail_csv = tp / "llm_evaluation_details.csv"
            pairs_csv  = tp / "pair_details.csv"
            pair_file  = detail_csv if detail_csv.exists() else pairs_csv
            pdf = pd.read_csv(pair_file) if pair_file.exists() else pd.DataFrame()
        return m, pdf
    except Exception as e:
        st.error(f"Full evaluation error: {e}")
        return None, None

def build_diff_df(pair_df: pd.DataFrame):
    if pair_df is None or pair_df.empty: return pd.DataFrame(), pd.DataFrame()
    col    = pair_df.columns.tolist()
    qty_gt = "gt_qty"       if "gt_qty"      in col else "gt_quantity"
    qty_pd = "pred_qty"     if "pred_qty"     in col else "pred_quantity"
    nm_col = "nutrient_match"   if "nutrient_match"   in col else None
    qm_col = "quantity_match"   if "quantity_match"   in col else "qty_match"
    cm_col = "context_match"    if "context_match"    in col else "ctx_match"
    f4_col = ("full_match_4f"   if "full_match_4f"    in col
               else "full_match" if "full_match"       in col else None)
    f3_col = "full_match_3f"    if "full_match_3f"    in col else None
    em_col = "eval_method"      if "eval_method"      in col else None

    def _b(v):
        if isinstance(v,bool): return v
        if isinstance(v,float): return bool(v) if v==v else False
        return str(v).lower() in ("true","1","yes")

    rows, styles = [], []
    for _, r in pair_df.iterrows():
        gn = str(r.get("gt_nutrient","") or "");  pn = str(r.get("pred_nutrient","") or "")
        gq = str(r.get(qty_gt,"") or "");          pq = str(r.get(qty_pd,"") or "")
        gu = str(r.get("gt_unit","") or "");       pu = str(r.get("pred_unit","") or "")
        gc = str(r.get("gt_context","") or "");    pc = str(r.get("pred_context","") or "")
        qo  = _b(r.get(qm_col,False)); uo  = _b(r.get("unit_match",False))
        co  = _b(r.get(cm_col,False))
        fo  = _b(r.get(f4_col,False)) if f4_col else (qo and uo and co and bool(gn and pn))
        f3o = _b(r.get(f3_col,False)) if f3_col else (qo and uo and bool(gn and pn))
        no  = _b(r.get(nm_col,False)) if nm_col else bool(gn and pn)
        mth = str(r.get(em_col,"")) if em_col else ""
        gt_only = bool(gn and not pn); pd_only = bool(pn and not gn)
        rb = C_GT_ONLY if gt_only else (C_PRED_ONLY if pd_only else "")
        def bg(ok): return rb if rb else (C_CORRECT if ok else C_WRONG)
        nbg = (C_GT_ONLY if gt_only else C_PRED_ONLY if pd_only else C_CORRECT if no else C_FUZZY)
        icon = ("— GT only"    if gt_only
                else "🔵 Pred only" if pd_only
                else "✅ Full 4f"   if fo
                else "🟨 Full 3f"   if f3o
                else "🟡 Partial"   if no
                else "❌ Mismatch")
        if mth: icon += f" [{mth}]"
        rows.append({"GT Nutrient":gn,"GT Qty":gq,"GT Unit":gu,"GT Context":gc,
                     "Pred Nutrient":pn,"Pred Qty":pq,"Pred Unit":pu,"Pred Context":pc,"Match":icon})
        styles.append({"GT Nutrient":f"background-color:{nbg}","GT Qty":f"background-color:{bg(qo)}",
                       "GT Unit":f"background-color:{bg(uo)}","GT Context":f"background-color:{bg(co)}",
                       "Pred Nutrient":f"background-color:{nbg}","Pred Qty":f"background-color:{bg(qo)}",
                       "Pred Unit":f"background-color:{bg(uo)}","Pred Context":f"background-color:{bg(co)}",
                       "Match":f"background-color:{rb or '#ffffff'}"})
    return pd.DataFrame(rows), pd.DataFrame(styles)

# ════════════════════════════════════════════════════════════════════════════
# SHARED WIDGETS
# ════════════════════════════════════════════════════════════════════════════

def _keys(stage): return list(STAGE_REGISTRY[stage].keys())
def _ready(stage,key): return STAGE_REGISTRY[stage][key].get("ready",False)
def _fmt(stage,key):
    cfg = STAGE_REGISTRY[stage][key]
    return cfg["label"] if cfg.get("ready") else f"🚧  {cfg['label']}"

def _clf_badge(clf_key: str) -> str:
    if "GLiNER" in clf_key:
        return "<span class='clf-badge gliner'>GLiNER</span>"
    if "Qwen" in clf_key:
        return "<span class='clf-badge qwen'>Qwen LLM</span>"
    if "Embedding" in clf_key:
        return "<span class='clf-badge emb'>Embedding (BGE-M3)</span>"
    return "<span class='clf-badge rule'>Rule-based</span>"

def note_widget(image_id: str, stage_name: str):
    st.markdown("---")
    st.markdown("**📝 Stage Note**")
    existing = get_note(image_id, stage_name)
    new_note = st.text_area(
        f"Add a note for `{image_id}` · `{stage_name}`",
        value=existing, height=80, key=f"note_{image_id}_{stage_name}",
        placeholder="Observations, errors seen, ideas for fixes…",
    )
    if st.button("💾 Save Note", key=f"save_note_{image_id}_{stage_name}"):
        if new_note.strip():
            save_note(image_id, stage_name, new_note)
            st.success("Note saved.")
        else:
            st.warning("Nothing to save.")

def metrics_row(metrics: Dict):
    cols = st.columns(6)
    for col, (k, lbl) in zip(cols, METRIC_LABELS.items()):
        val  = metrics.get(k, 0.0)
        base = EXP1_BASELINE.get(k, 0.0)
        col.metric(lbl, f"{val:.1%}", delta=f"{val-base:+.1%}")

def stage_banner(title: str, input_desc: str, output_desc: str):
    st.markdown(f"""
    <div class="stage-banner">
        <h2>{title}</h2>
        <p>
            <strong>INPUT:</strong> {input_desc}&nbsp;&nbsp;&nbsp;
            <strong>OUTPUT:</strong> {output_desc}
        </p>
    </div>""", unsafe_allow_html=True)

def pipeline_required():
    st.info("⚠️ Run the pipeline first on the **🏠 Run Pipeline** page.")

def render_edge_explanation(edge_types: List[str], is_v2: bool):
    """Render explanation cards for the given edge types."""
    st.markdown("### 📖 Edge Type Explanations")
    st.caption("What each relation means and why it exists in the graph.")
    for et in edge_types:
        if et not in EDGE_TYPE_DOCS: continue
        doc = EDGE_TYPE_DOCS[et]
        version_class = "v2" if et in ("ROW_COMPAT","COL_COMPAT","DIRECTIONAL_ADJ","HEADER_SCOPE") else "v1"
        st.markdown(f"""
        <div class="edge-card {version_class}">
            <h4>{doc['title']}</h4>
            <p><strong>What:</strong> {doc['what']}</p>
            <p><strong>Why:</strong> {doc['why']}</p>
            <div class="example">Example: {doc['example']}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧪 Nutrient Inspector")
    st.caption("Zero-Shot Nutrient Extraction")
    st.caption(f"`{PROJECT_ROOT.name}`")
    st.divider()

    PAGE = st.radio("Navigate", [
        "🏠  Run Pipeline",
        "📝  Annotation Editor",
        "🔍  Stage 1 · OCR",
        "✂️  Stage 2 · Corrector",
        "📄  Stage 3A · Serializer",
        "🏷️  Stage 3 · Classifier",
        "🧬  Stage 3.5 · Token Enricher",
        "🕸️  Stage 4 · Graph",
        "🔗  Stage 5 · Association",
        "🧪  Evaluation",
        "📊  Previous Experiments",
        "📓  Notes",
    ], label_visibility="collapsed")

    st.divider()

    if st.session_state.get("pipeline_image"):
        img_run = st.session_state["pipeline_image"]
        clf_run = st.session_state.get("pipeline_clf_key","")
        st.success(f"✅ Pipeline ran: `{img_run}`")
        st.markdown(_clf_badge(clf_run), unsafe_allow_html=True)
        diag_s = st.session_state.get("pipeline_diag", {})
        for sid, key in diag_s.get("stages", {}).items():
            st.caption(f"  {sid}: {key}")
    else:
        st.warning("No pipeline run yet.")

    st.divider()
    images_list = get_images()
    if not images_list:
        st.error(f"No images in `{DATA_RAW}`")
    else:
        st.caption(f"📷 {len(images_list)} images · formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

if "🏠" in PAGE:
    st.title("🏠 Run Pipeline")

    if not images_list:
        st.stop()

    col_img_sel, col_eval = st.columns([2, 1])
    with col_img_sel:
        selected_image = st.selectbox("📷 Image", images_list)
    with col_eval:
        use_llm = st.toggle("🤖 LLM evaluation", value=False,
                            help="Toggle on for Qwen LLM-assisted per-image eval. "
                                 "For full fast+LLM dual evaluation, use the "
                                 "🧪 Evaluation page.")

    st.markdown("### ⚙️ Pipeline Configuration")

    def _sel(stage_id, label, col_key=None):
        keys   = _keys(stage_id)
        labels = [_fmt(stage_id,k) for k in keys]
        default = next((i for i,k in enumerate(keys) if _ready(stage_id,k)),0)
        chosen_label = st.selectbox(label, labels, index=default,
                                    key=col_key or f"sel_{stage_id}")
        chosen_key   = keys[labels.index(chosen_label)]
        note = STAGE_REGISTRY[stage_id][chosen_key].get("note","")
        if note: st.markdown(f"<div class='note'>{note}</div>",unsafe_allow_html=True)
        return chosen_key

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: sel_ocr   = _sel("ocr",         "Stage 1 — OCR")
    with c2: sel_corr  = _sel("corrector",   "Stage 2 — Corrector")
    with c3:
        sel_clf = _sel("classifier", "Stage 3 — Classifier")
        if STAGE_REGISTRY["classifier"][sel_clf].get("uses_serializer"):
            st.markdown(
                "<div class='note'>⚡ text_serializer runs automatically before this classifier</div>",
                unsafe_allow_html=True
            )
    with c4: sel_enr   = _sel("enricher",    "Stage 3.5 — Enricher")
    with c5: sel_graph = _sel("graph",       "Stage 4 — Graph")
    with c6: sel_assoc = _sel("association", "Stage 5 — Association")

    # Warn if V2 graph/assoc selected without enricher
    _graph_v2 = STAGE_REGISTRY["graph"][sel_graph].get("version") == "v2"
    _assoc_v2 = STAGE_REGISTRY["association"][sel_assoc].get("version") == "v2"
    _enr_bypass = STAGE_REGISTRY["enricher"][sel_enr]["kind"] == "builtin-bypass"
    if (_graph_v2 or _assoc_v2) and _enr_bypass:
        st.warning("⚠️ V2 Graph/Association require the **Geometry-Aware Token Enricher**. "
                   "Select it in Stage 3.5 or results will be incorrect.")

    unready = [f"S{i+1}:{k}" for i,(sid,k) in enumerate([
        ("ocr",sel_ocr),("corrector",sel_corr),("classifier",sel_clf),
        ("enricher",sel_enr),("graph",sel_graph),("association",sel_assoc)]) if not _ready(sid,k)]
    if unready: st.warning(f"🚧 Not implemented: {', '.join(unready)}")

    if st.button("▶ Run Pipeline", type="primary", disabled=bool(unready)):
        with st.spinner("Running stages 1–5 …"):
            t0 = time.time()
            tuples, diag, inter = run_pipeline(
                selected_image, sel_ocr, sel_corr, sel_clf, sel_enr, sel_graph, sel_assoc)
            diag["total_s"] = round(time.time()-t0,2)
        st.session_state.update({
            "pipeline_image":    selected_image,
            "pipeline_diag":     diag,
            "pipeline_inter":    inter,
            "pipeline_tuples":   tuples,
            "pipeline_metrics":  None,
            "pipeline_pair_df":  None,
            "pipeline_clf_key":  sel_clf,
        })
        if not diag.get("error"):
            with st.spinner("Evaluating …"):
                gt_df = load_gt_df()
                m, pdf = run_evaluation(gt_df, tuples, selected_image, use_llm=use_llm)
            st.session_state["pipeline_metrics"] = m
            st.session_state["pipeline_pair_df"]  = pdf
            if m: st.success(f"Done in {diag['total_s']}s  |  Full Tuple F1: {m.get('full_tuple_f1',0):.1%}")
        else:
            st.error(diag["error"])

    st.divider()

    gt_df   = load_gt_df()
    sel_img = st.session_state.get("pipeline_image", selected_image)
    gt_rows = gt_df[gt_df["image_id"]==sel_img][GT_COLS].to_dict(orient="records")
    tuples  = st.session_state.get("pipeline_tuples")
    metrics = st.session_state.get("pipeline_metrics")
    pair_df = st.session_state.get("pipeline_pair_df")

    ci, cg = st.columns([2,3])
    with ci:
        img = _open_img(sel_img)
        if img: st.image(img, caption=sel_img, use_column_width=True)
    with cg:
        st.subheader(f"📗 Ground Truth  ({len(gt_rows)} tuples)")
        if gt_rows:
            st.dataframe(pd.DataFrame(gt_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No GT annotations for this image.")

        st.subheader("🤖 Predicted")
        if tuples is None:
            st.info("Press ▶ Run Pipeline above.")
        elif not tuples:
            st.warning("Pipeline returned 0 tuples.")
        else:
            pred_df = pd.DataFrame([{k:t.get(k,"") for k in GT_COLS} for t in tuples])
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

    if metrics:
        st.markdown("### 📊 Metrics  (vs Exp 1 baseline)")
        metrics_row(metrics)

    if pair_df is not None and not pair_df.empty:
        st.markdown("### 🔍 Side-by-Side Diff")
        disp, sty = build_diff_df(pair_df)
        if not disp.empty:
            st.dataframe(disp.style.apply(lambda _: sty.values, axis=None),
                         use_container_width=True, hide_index=True)

    diag = st.session_state.get("pipeline_diag")
    if diag:
        with st.expander("🔧 Pipeline Diagnostics"):
            st.json({k:v for k,v in diag.items() if k!="stages"})

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANNOTATION EDITOR
# ════════════════════════════════════════════════════════════════════════════

elif "📝" in PAGE:
    st.title("📝 Annotation Editor")
    st.caption("Edit ground-truth annotations and save back to the original JSON file.")

    if not images_list: st.stop()

    if "ann_img_idx" not in st.session_state:
        st.session_state["ann_img_idx"] = 0

    st.session_state["ann_img_idx"] = max(
        0, min(st.session_state["ann_img_idx"], len(images_list) - 1)
    )

    nav_left, nav_mid, nav_right = st.columns([1, 6, 1])
    with nav_left:
        st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
        if st.button("◀", key="ann_prev", use_container_width=True,
                     disabled=(st.session_state["ann_img_idx"] == 0)):
            st.session_state["ann_img_idx"] -= 1; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with nav_mid:
        chosen = st.selectbox("📷 Image", images_list,
                              index=st.session_state["ann_img_idx"], key="ann_img_sel")
        new_idx = images_list.index(chosen)
        if new_idx != st.session_state["ann_img_idx"]:
            st.session_state["ann_img_idx"] = new_idx; st.rerun()
    with nav_right:
        st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
        if st.button("▶", key="ann_next", use_container_width=True,
                     disabled=(st.session_state["ann_img_idx"] == len(images_list) - 1)):
            st.session_state["ann_img_idx"] += 1; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    selected_image = images_list[st.session_state["ann_img_idx"]]
    idx = st.session_state["ann_img_idx"]
    st.caption(f"Image **{idx + 1}** of **{len(images_list)}** — `{selected_image}`")
    st.progress((idx + 1) / len(images_list))

    c_img, c_edit = st.columns([2, 3])
    with c_img:
        img = _open_img(selected_image)
        if img: st.image(img, use_column_width=True)
    with c_edit:
        ann_data = load_annotation_json(selected_image)
        if ann_data is None:
            st.warning(f"No annotation file found for `{selected_image}`.")
            if st.button("➕ Create new annotation file"):
                new_data = {"image_id": selected_image, "nutrients": []}
                if save_annotation_json(selected_image, new_data):
                    st.success("Empty annotation file created. Reload page."); st.stop()
        else:
            st.markdown(f"**File:** `data/annotations/{Path(selected_image).stem}.json`")
            nutrients  = ann_data.get("nutrients", [])
            edit_cols  = ["nutrient","quantity","unit","context","nrv_percent","serving_size"]
            rows = [{c: str(n.get(c,"") or "").strip() for c in edit_cols} for n in nutrients] or [{c:"" for c in edit_cols}]
            edit_df = pd.DataFrame(rows, columns=edit_cols)
            st.markdown("**Edit annotations below:**")
            edited = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True,
                                    hide_index=True, key=f"ann_editor_{selected_image}",
                                    column_config={
                                        "nutrient":     st.column_config.TextColumn("Nutrient",     width="large"),
                                        "quantity":     st.column_config.TextColumn("Quantity",     width="small"),
                                        "unit":         st.column_config.TextColumn("Unit",         width="small"),
                                        "context":      st.column_config.TextColumn("Context",      width="medium"),
                                        "nrv_percent":  st.column_config.TextColumn("NRV %",        width="small"),
                                        "serving_size": st.column_config.TextColumn("Serving size", width="medium"),
                                    })
            col_save, col_stat = st.columns([1, 3])
            with col_save:
                if st.button("💾 Save to JSON", type="primary", key="ann_save"):
                    new_nutrients = [{c: str(row.get(c,"")).strip() for c in edit_cols}
                                     for _, row in edited.iterrows()
                                     if any(str(v).strip() for v in row)]
                    new_data = dict(ann_data); new_data["nutrients"] = new_nutrients
                    if save_annotation_json(selected_image, new_data):
                        st.success(f"✅ Saved {len(new_nutrients)} tuples.")
            with col_stat:
                st.caption(f"Rows: **{len(edited)}** · Non-empty: **{sum(1 for _,r in edited.iterrows() if any(str(v).strip() for v in r))}**")
            with st.expander("🔎 Preview raw JSON"):
                st.json(ann_data)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 1 · OCR
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 1" in PAGE:
    image_id = st.session_state.get("pipeline_image")
    inter    = st.session_state.get("pipeline_inter", {})
    s1       = inter.get("stage1_tokens", [])

    stage_banner(
        "🔍 Stage 1 · OCR",
        "Raw image (pixel data)",
        f"List of text tokens with bounding boxes and confidence scores  ({len(s1)} tokens)"
    )
    if not image_id: pipeline_required(); st.stop()

    c_vis, c_data = st.columns([2, 3])
    with c_vis:
        st.markdown("**Bounding box overlay** — colour = confidence tier")
        def conf_colour(tok):
            c = tok.get("conf", 0)
            if c >= 0.80: return (60, 179, 113)
            elif c >= 0.50: return (240, 173, 78)
            else: return (231, 76, 60)
        ov = overlay_bboxes(image_id, s1, colour_fn=conf_colour, label_key="token")
        if ov: st.image(ov, use_column_width=True)
        st.markdown("""<div style="font-size:11px; margin-top:6px">
          <span style="color:#3CB371">■</span> conf ≥ 0.80 &nbsp;
          <span style="color:#F0AD4E">■</span> conf 0.50–0.80 &nbsp;
          <span style="color:#E74C3C">■</span> conf &lt; 0.50
        </div>""", unsafe_allow_html=True)
    with c_data:
        st.markdown("**OUTPUT — OCR tokens**")
        if s1:
            df1 = pd.DataFrame([{"token":t["token"],"conf":round(t.get("conf",0),4),
                                  "x1":t.get("x1"),"y1":t.get("y1"),"x2":t.get("x2"),"y2":t.get("y2")} for t in s1])
            st.dataframe(df1, use_container_width=True, hide_index=True, height=420)
            below = sum(1 for t in s1 if t.get("conf",1) < CONF_THRESH)
            st.caption(f"Total: **{len(s1)}** · Below threshold: **{below}** · Kept: **{len(s1)-below}**")
        else:
            st.info("No tokens — pipeline may not have run yet.")
    note_widget(image_id, "Stage 1 · OCR")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 2 · CORRECTOR
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 2" in PAGE:
    image_id  = st.session_state.get("pipeline_image")
    inter     = st.session_state.get("pipeline_inter", {})
    s1        = inter.get("stage1_tokens",    [])
    s2        = inter.get("stage2_tokens",    [])
    pad_log   = inter.get("stage2_paddle_log", [])
    diag      = st.session_state.get("pipeline_diag", {})

    stage_banner(
        "✂️ Stage 2 · Corrector",
        f"OCR tokens ({len(s1)} raw)",
        f"Corrected/split tokens ({len(s2)} output)"
    )
    if not image_id: pipeline_required(); st.stop()

    c_vis, c_data = st.columns([2, 3])
    with c_vis:
        st.markdown("**Before correction**")
        ov_raw = overlay_bboxes(image_id, s1, colour_fn=lambda t:(200,200,200), label_key="token")
        if ov_raw: st.image(ov_raw, use_column_width=True, caption="Raw tokens")
        st.markdown("**After correction**")
        ov_cor = overlay_bboxes(image_id, s2, colour_fn=lambda t:(92,184,92), label_key="token")
        if ov_cor: st.image(ov_cor, use_column_width=True, caption="Corrected tokens")
    with c_data:
        corr_summary = diag.get("corrector","—")
        st.info(f"Corrector report: `{corr_summary}`")
        if pad_log:
            st.markdown("**Paddle Corrector audit log**")
            log_rows = []
            for e in pad_log:
                orig  = e.get("original",""); corr = e.get("corrected",[])
                rules_lst = e.get("rules_fired",[])
                corr_str = " | ".join(c["token"] if isinstance(c,dict) else str(c) for c in corr) if isinstance(corr,list) else str(corr)
                rules    = ", ".join(rules_lst) or e.get("rule","")
                is_split = isinstance(corr,list) and len(corr)>1
                is_snap  = "C11_nutrient_snap" in rules_lst
                is_regex = "C12_regex_fix" in rules_lst
                tok_type = "✂️ Split" if is_split else ("🔤 Snap" if is_snap else ("🔢 Regex" if is_regex else "✏️ Sub"))
                log_rows.append({"original":orig,"corrected":corr_str,"rules":rules,"type":tok_type})
            df_log = pd.DataFrame(log_rows)
            _TYPE_BG = {"✂️ Split":"#d6eaff","🔤 Snap":"#d4f5d4","🔢 Regex":"#e8d5f5","✏️ Sub":"#fff9c4"}
            def _sl(r): return [f"background-color:{_TYPE_BG.get(r['type'],'')}"] * 4
            st.dataframe(df_log.style.apply(_sl,axis=1), use_container_width=True, hide_index=True, height=360)
        elif s1 and s2:
            diff_rows = [{"original":a["token"],"corrected":b["token"],"changed":"✏️" if a["token"]!=b["token"] else "—"}
                          for a,b in zip(s1,s2)]
            df2 = pd.DataFrame(diff_rows)
            def _sc(r):
                if r["changed"]=="✏️": return ["background-color:#fff9c4"]*3
                return [""]*3
            st.dataframe(df2.style.apply(_sc,axis=1), use_container_width=True, hide_index=True, height=360)
    note_widget(image_id, "Stage 2 · Corrector")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 3A · SERIALIZER
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 3A" in PAGE or "Serializer" in PAGE:
    st.title("📄 Stage 3A · Text Serializer")
    st.markdown("""
    **text_serializer.py** converts corrected OCR tokens into a plain-text string for
    GLiNER and Qwen. This page lets you inspect the exact text these models receive and
    verify that line grouping, token ordering, and character offsets are correct.
    """)

    if not images_list: st.stop()

    cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 2, 2])
    with cfg_col1:
        ser_image = st.selectbox("📷 Image", images_list, key="ser_img")
    with cfg_col2:
        ocr_keys   = _keys("ocr")
        ocr_labels = [_fmt("ocr",k) for k in ocr_keys]
        sel_ocr_s  = st.selectbox("Stage 1 — OCR", ocr_labels, key="ser_ocr")
        sel_ocr_sk = ocr_keys[ocr_labels.index(sel_ocr_s)]
    with cfg_col3:
        corr_keys   = _keys("corrector")
        corr_labels = [_fmt("corrector",k) for k in corr_keys]
        sel_corr_s  = st.selectbox("Stage 2 — Corrector", corr_labels, key="ser_corr")
        sel_corr_sk = corr_keys[corr_labels.index(sel_corr_s)]

    pipeline_img   = st.session_state.get("pipeline_image")
    pipeline_inter = st.session_state.get("pipeline_inter", {})
    ser_from_pipeline = (pipeline_img == ser_image and pipeline_inter.get("stage3a_used"))

    run_btn = st.button("🔄 Run Serializer", type="primary", key="ser_run")

    if run_btn or ser_from_pipeline:
        if ser_from_pipeline and not run_btn:
            serialized = pipeline_inter.get("stage3a_serialized", {})
            tokens_for_ser = pipeline_inter.get("stage2_tokens", [])
            st.info(f"Showing serializer output from the last pipeline run on `{pipeline_img}`.")
        else:
            with st.spinner("Running OCR + Corrector + Serializer …"):
                result = run_serializer_standalone(ser_image, sel_ocr_sk, sel_corr_sk)
                if result[0] is None:
                    st.stop()
                serialized, tokens_for_ser = result

        text        = serialized.get("text", "")
        token_spans = serialized.get("token_spans", [])
        lines       = serialized.get("lines", [])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OCR tokens (after corrector)", len(tokens_for_ser))
        m2.metric("Serialized tokens",            len(token_spans))
        m3.metric("Visual lines detected",        len(lines))
        m4.metric("Text length (chars)",          len(text))

        st.divider()

        img_col, detail_col = st.columns([2, 3])

        with img_col:
            st.markdown("**Label image**")
            img = _open_img(ser_image)
            if img: st.image(img, use_column_width=True)

        with detail_col:
            tab1, tab2, tab3 = st.tabs(["📝 Serialized Text", "📐 Line Groupings", "🔢 Token Spans"])

            with tab1:
                st.markdown("**Exact text sent to GLiNER / Qwen — one visual line per row:**")
                text_lines = text.split("\n")
                for i, line_text in enumerate(text_lines):
                    highlighted = re.sub(
                        r'(\b\d+\.?\d*\b)',
                        r'<span style="color:#5cb85c;font-weight:bold">\1</span>',
                        line_text
                    )
                    st.markdown(
                        f"<div class='ser-line'><span class='ln'>L{i+1:02d}</span>{highlighted}</div>",
                        unsafe_allow_html=True
                    )
                st.caption(f"Green numbers = detected numeric tokens · {len(text_lines)} lines")

                st.markdown("---")
                st.markdown("**Copy-ready raw text:**")
                st.text_area("Raw serialized text", value=text, height=200,
                             key="ser_raw_text", label_visibility="collapsed")

            with tab2:
                st.markdown("**Line grouping — which tokens landed on which visual line:**")
                tok_by_idx = {t.get("token_index", i): t
                              for i, t in enumerate(tokens_for_ser)}
                lg_rows = []
                for line in lines:
                    token_texts = []
                    for idx in line["token_indices"]:
                        tok = tok_by_idx.get(idx, {})
                        token_texts.append(str(tok.get("token","?")))
                    lg_rows.append({
                        "Line ID":     line["line_id"] + 1,
                        "y_center px": round(line["y_center"], 1),
                        "N tokens":    len(line["token_indices"]),
                        "Tokens →":    " | ".join(token_texts),
                    })
                df_lg = pd.DataFrame(lg_rows)

                def _line_flag(r):
                    if r["N tokens"] == 1 and any(
                        kw in r["Tokens →"].lower()
                        for kw in ["fett","kohlen","eiwei","ballaststoff","fettsäuren"]
                    ):
                        return ["background-color:#fff3cd"] * len(r)
                    if r["N tokens"] > 8:
                        return ["background-color:#ffd6d6"] * len(r)
                    return [""] * len(r)

                st.dataframe(df_lg.style.apply(_line_flag, axis=1),
                             use_container_width=True, hide_index=True, height=400)
                st.markdown("""
                <div style="font-size:11px;margin-top:4px">
                  <span style="background:#fff3cd;padding:1px 6px">■</span> Single nutrient token — may be split from its quantity<br>
                  <span style="background:#ffd6d6;padding:1px 6px">■</span> &gt;8 tokens on one line — may be a line-merge error
                </div>""", unsafe_allow_html=True)

            with tab3:
                st.markdown("**Character offsets — used for GLiNER/Qwen entity remapping:**")
                sp_rows = []
                for ts in token_spans:
                    sp_rows.append({
                        "token_index": ts["token_index"],
                        "line_id":     ts["line_id"] + 1,
                        "start_char":  ts["start_char"],
                        "end_char":    ts["end_char"],
                        "token_text":  ts["token_text"],
                        "length":      ts["end_char"] - ts["start_char"],
                    })
                df_sp = pd.DataFrame(sp_rows)

                line_ids = sorted(df_sp["line_id"].unique().tolist())
                sel_line = st.selectbox("Filter by line", ["All"] + [str(l) for l in line_ids],
                                        key="ser_line_filter")
                if sel_line != "All":
                    df_sp = df_sp[df_sp["line_id"] == int(sel_line)]

                st.dataframe(df_sp, use_container_width=True, hide_index=True, height=380)
                st.caption(
                    f"Showing {len(df_sp)} of {len(token_spans)} token spans  ·  "
                    f"Containment condition for entity mapping: "
                    f"`span.start ≤ token.start_char AND token.end_char ≤ span.end`"
                )

        st.divider()
        st.markdown("### 🔎 Potential Issues Detected")

        issues = []
        toks_by_line: Dict[int, List[str]] = {}
        for ts in token_spans:
            toks_by_line.setdefault(ts["line_id"], []).append(ts["token_text"])

        for lid, toks in toks_by_line.items():
            joined = " ".join(toks).lower()
            if "gesattigte" in joined and "fettsauren" not in joined:
                next_toks = toks_by_line.get(lid+1, [])
                if any("fettsauren" in t.lower() for t in next_toks):
                    issues.append(f"⚠️ **Line {lid+1}→{lid+2}:** `davon gesattigte` split from `Fettsauren` — multi-token nutrient span crosses line boundary. GLiNER cannot match this.")

        for ts in token_spans:
            t = ts["token_text"]
            if re.search(r'\d+\.?\d*(mg|µg|g|kJ|kcal)\d+', t, re.IGNORECASE):
                issues.append(f"⚠️ **Token `{t}`:** fused quantity+unit+NRV — corrector did not split. Qwen will receive this as one entity.")

        for ts in token_spans:
            if len(ts["token_text"]) > 80:
                issues.append(f"⚠️ **Token (line {ts['line_id']+1}):** length {len(ts['token_text'])} chars — multilingual slash-variant may confuse entity boundaries.")

        ctx_patterns = ["tagesdosis","100g","per serving","portion","pro 100","je 100"]
        ctx_found = []
        for ts in token_spans:
            if any(p in ts["token_text"].lower() for p in ctx_patterns):
                ctx_found.append(ts["token_text"])

        if ctx_found:
            issues.append(f"✅ **Context headers visible:** {', '.join([f'`{c}`' for c in ctx_found[:5]])} — Qwen can detect these (German multilingual support).")
        else:
            issues.append("⚠️ **No context headers detected** — CONTEXT Acc will be 0% for this image with GLiNER/Qwen.")

        if not issues:
            st.success("No issues detected — serializer output looks clean.")
        else:
            for iss in issues:
                if iss.startswith("✅"):
                    st.success(iss)
                else:
                    st.warning(iss)

    else:
        st.info("Select an image and click **Run Serializer** — or run the full pipeline with a GLiNER/Qwen classifier first.")
        st.markdown("""
        **What this page shows:**
        - The exact text string sent to GLiNER / Qwen (Section 1 — Serialized Text)
        - Which tokens were grouped onto each visual line (Section 2 — Line Groupings)
        - Character offsets used for entity-to-token remapping (Section 3 — Token Spans)
        - Automatic issue detection (split nutrient names, fused tokens, context visibility)
        """)

    note_widget(ser_image if 'ser_image' in locals() else (images_list[0] if images_list else ""), "Stage 3A · Serializer")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 3 · CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 3" in PAGE and "3A" not in PAGE and "3.5" not in PAGE:
    image_id   = st.session_state.get("pipeline_image")
    inter      = st.session_state.get("pipeline_inter", {})
    s2         = inter.get("stage2_tokens",    [])
    s3         = inter.get("stage3_classified",[])
    diag       = st.session_state.get("pipeline_diag",{})
    clf_key    = st.session_state.get("pipeline_clf_key","")
    clf_type   = diag.get("classifier_type","rule")

    counts = diag.get("classifier_counts",{})
    summary = "  ".join(f"{l}:{n}" for l,n in sorted(counts.items()))

    stage_banner(
        "🏷️ Stage 3 · Classifier",
        f"Corrected tokens ({len(s2)})" + (" → text_serializer → " if inter.get("stage3a_used") else " → "),
        f"Labelled tokens — {summary}"
    )
    if not image_id: pipeline_required(); st.stop()

    st.markdown(_clf_badge(clf_key), unsafe_allow_html=True)
    if inter.get("stage3a_used"):
        ser = inter.get("stage3a_serialized", {})
        st.info(
            f"📄 text_serializer ran before this classifier — "
            f"{len(ser.get('token_spans',[]))} tokens serialized into "
            f"{len(ser.get('lines',[]))} lines. "
            f"View full serializer output on the **Stage 3A · Serializer** page."
        )

    c_vis, c_data = st.columns([2, 3])
    with c_vis:
        st.markdown("**Colour overlay** — label per token")
        ov = overlay_bboxes(image_id, s3,
                            colour_fn=lambda t: LABEL_COLOURS.get(t.get("label","UNKNOWN"),(200,200,200)),
                            label_key="label")
        if ov: st.image(ov, use_column_width=True)
        _CLF_HEX = {"NUTRIENT":"#d6eaff","QUANTITY":"#d4f5d4","UNIT":"#fff3cd",
                    "CONTEXT":"#ead6ff","NOISE":"#f0f0f0","UNKNOWN":"#ffd6d6"}
        lc = st.columns(3)
        for i,(lbl,c) in enumerate(_CLF_HEX.items()):
            lc[i%3].markdown(f"<span style='background:{c};padding:2px 8px;border-radius:4px;font-size:11px'>{lbl}</span>",
                             unsafe_allow_html=True)

    with c_data:
        if counts:
            st.markdown("**Label distribution**")
            total_t = max(sum(counts.values()),1)
            st.dataframe(
                pd.DataFrame([{"Label":l,"Count":n,"Share":f"{n/total_t:.0%}"}
                               for l,n in sorted(counts.items(),key=lambda x:-x[1])]),
                use_container_width=True, hide_index=True,
            )

        # GLiNER-specific stats
        if clf_type == "gliner" and s3:
            gliner_matched = [t for t in s3 if t.get("gliner_score") is not None]
            if gliner_matched:
                avg_score = sum(t["gliner_score"] for t in gliner_matched) / len(gliner_matched)
                st.info(f"GLiNER: **{len(gliner_matched)}** tokens matched by spans · "
                        f"mean confidence **{avg_score:.3f}** · "
                        f"**{counts.get('UNKNOWN',0)}** tokens fell back to UNKNOWN")

        # Qwen-specific stats
        if clf_type == "qwen" and s3:
            qwen_matched = [t for t in s3 if t.get("qwen_entity") is not None]
            unique_entities = {t.get("qwen_entity") for t in qwen_matched}
            if qwen_matched:
                avg_score = sum(t["qwen_score"] for t in qwen_matched) / len(qwen_matched)
                st.info(f"Qwen: **{len(unique_entities)}** unique entities extracted · "
                        f"**{len(qwen_matched)}** tokens matched · "
                        f"mean fuzzy score **{avg_score:.3f}** · "
                        f"**{counts.get('UNKNOWN',0)}** tokens fell back to UNKNOWN")

        # Embedding-specific stats
        if clf_type == "embedding" and s3:
            emb_tokens = [t for t in s3 if t.get("classification_method") == "embedding"]
            rule_tokens = [t for t in s3 if t.get("classification_method") == "rule"]
            if emb_tokens or rule_tokens:
                st.info(
                    f"Embedding (BGE-M3): **{len(rule_tokens)}** tokens decided by rules · "
                    f"**{len(emb_tokens)}** tokens decided by embeddings · "
                    f"**{counts.get('UNKNOWN',0)}** UNKNOWN"
                )

        if s3:
            st.markdown("**Token table**")
            rows_s3 = []
            for t in sorted(s3, key=lambda x:(x.get("y1",0),x.get("x1",0))):
                row = {
                    "token":  t["token"],
                    "label":  t.get("label","?"),
                    "norm":   t.get("norm",""),
                    "conf":   round(t.get("conf",0),3),
                }
                if clf_type == "gliner":
                    row["gliner_score"] = round(t.get("gliner_score",0) or 0, 3)
                    row["span_text"]    = t.get("gliner_span_text","") or ""
                elif clf_type == "qwen":
                    row["qwen_score"]   = round(t.get("qwen_score",0) or 0, 3)
                    row["entity_text"]  = t.get("qwen_entity","") or ""
                elif clf_type == "embedding":
                    row["method"]     = t.get("classification_method","?")
                    row["emb_conf"]   = round(t.get("embedding_confidence",0) or 0, 3)
                    row["emb_ctx"]    = (t.get("embedding_context","") or "")[:40]
                rows_s3.append(row)

            df3 = pd.DataFrame(rows_s3)
            def _sl(r): return [f"background-color:{_CLF_HEX.get(r['label'],'')}"]*len(r)
            st.dataframe(df3.style.apply(_sl,axis=1), use_container_width=True,
                         hide_index=True, height=340)

    note_widget(image_id, "Stage 3 · Classifier")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 3.5 · TOKEN ENRICHER
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 3.5" in PAGE or "Token Enricher" in PAGE:
    image_id = st.session_state.get("pipeline_image")
    inter    = st.session_state.get("pipeline_inter", {})
    s3       = inter.get("stage3_classified", [])
    enriched = inter.get("stage35_enriched", [])
    enr_diag = inter.get("stage35_diag", {})
    enr_used = inter.get("stage35_used", False)

    stage_banner(
        "🧬 Stage 3.5 · Token Enricher",
        f"Classified tokens ({len(s3)})",
        f"Enriched tokens ({len([t for t in enriched if t.get('is_enriched')])} active)"
    )
    if not image_id:
        pipeline_required(); st.stop()

    if not enr_used:
        st.warning("Token Enricher was **bypassed** in the last pipeline run. "
                   "Re-run with **Geometry-Aware** enricher selected in Stage 3.5.")
        note_widget(image_id, "Stage 3.5 · Token Enricher")
        st.stop()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Logical rows",    enr_diag.get("num_rows", 0))
    m2.metric("Logical columns", enr_diag.get("num_columns", 0))
    m3.metric("Dosage streams",  enr_diag.get("dosage_streams", 0))
    m4.metric("Headers detected",enr_diag.get("headers_detected", 0))
    m5.metric("Rank consistent", "✅" if enr_diag.get("rank_consistent") else "❌")

    c_vis, c_data = st.columns([2, 3])

    with c_vis:
        ROLE_COLOURS = {
            "NUTRIENT":  (74,  144, 217),
            "DOSAGE":    (92,  184,  92),
            "UNIT":      (240, 173,  78),
            "NRV":       (155,  89, 182),
            "MIXED":     (170, 170, 170),
            "SINGLETON": (120, 120, 120),
            "UNKNOWN":   (200, 200, 200),
        }
        st.markdown("**Column role overlay** — colour = column role")
        ov = overlay_bboxes(
            image_id, [t for t in enriched if t.get("is_enriched")],
            colour_fn=lambda t: ROLE_COLOURS.get(t.get("column_role", "UNKNOWN"), (200,200,200)),
            label_key="column_role"
        )
        if ov:
            st.image(ov, use_column_width=True)

        rc_html = "  ".join(
            f"<span style='background:rgb({c[0]},{c[1]},{c[2]});color:#fff;"
            f"padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600'>{r}</span>"
            for r, c in ROLE_COLOURS.items() if r in {t.get("column_role") for t in enriched}
        )
        st.markdown(rc_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Row membership overlay** — same colour = same logical row")
        import colorsys
        active_enriched = [t for t in enriched if t.get("is_enriched")]
        max_row = max((t.get("row_id", 0) for t in active_enriched), default=0)
        def row_colour(tok):
            rid = tok.get("row_id", 0)
            hue = (rid * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
            return (int(r*255), int(g*255), int(b*255))
        ov_row = overlay_bboxes(image_id, active_enriched,
                                colour_fn=row_colour, label_key="token")
        if ov_row:
            st.image(ov_row, use_column_width=True, caption=f"{max_row+1} logical rows")

    with c_data:
        tab_cols, tab_tokens, tab_diag = st.tabs(["📊 Column Table", "🔢 Token Table", "🔧 Diagnostics"])

        with tab_cols:
            st.markdown("**Detected columns — roles, angles, contexts, and data ranks**")
            col_roles    = enr_diag.get("column_roles", {})
            col_angles   = enr_diag.get("column_angles", {})
            col_contexts = enr_diag.get("column_contexts", {})
            rank_counts  = enr_diag.get("rank_counts", {})

            col_rows = []
            for col_id in sorted(col_roles.keys()):
                col_rows.append({
                    "Col ID":    col_id,
                    "Role":      col_roles.get(col_id, "?"),
                    "Angle (°)": col_angles.get(col_id, "?"),
                    "Data ranks":rank_counts.get(col_id, "?"),
                    "Context":   col_contexts.get(col_id, "—"),
                })
            if col_rows:
                df_cols = pd.DataFrame(col_rows)
                ROLE_BG = {"NUTRIENT":"#d6eaff","DOSAGE":"#d4f5d4","UNIT":"#fff3cd",
                           "NRV":"#ead6ff","MIXED":"#f0f0f0","SINGLETON":"#e8e8e8"}
                def _role_style(r):
                    bg = ROLE_BG.get(r["Role"], "")
                    return [f"background-color:{bg}"] * len(r)
                st.dataframe(df_cols.style.apply(_role_style, axis=1),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No columns detected.")

            streams = enr_diag.get("dosage_streams", 0)
            if streams > 0:
                st.markdown(f"**Dosage streams: {streams}** — each mapped independently to nutrient column")
                stream_rows = []
                for t in active_enriched:
                    sid = t.get("dosage_stream_id", -1)
                    if sid >= 0:
                        ctx = t.get("column_context_id", "—")
                        if not any(s["Stream"] == sid for s in stream_rows):
                            col_id = t.get("column_id", "?")
                            n_tokens = sum(1 for x in active_enriched
                                           if x.get("dosage_stream_id") == sid)
                            stream_rows.append({
                                "Stream": sid,
                                "Column": col_id,
                                "Context": ctx,
                                "Tokens": n_tokens,
                            })
                if stream_rows:
                    st.dataframe(pd.DataFrame(stream_rows),
                                 use_container_width=True, hide_index=True)

        with tab_tokens:
            st.markdown("**Enriched token table** — all structural fields")
            if active_enriched:
                tok_rows = []
                for t in sorted(active_enriched, key=lambda x: (x.get("row_id",0), x.get("rank_in_row",0))):
                    tok_rows.append({
                        "token":     t.get("token",""),
                        "label":     t.get("label",""),
                        "row":       t.get("row_id", -1),
                        "col":       t.get("column_id", -1),
                        "rank_row":  t.get("rank_in_row", -1),
                        "nut_rank":  t.get("nutrient_rank_in_column", -1),
                        "qty_rank":  t.get("qty_rank_in_column", -1),
                        "data_rank": t.get("data_rank_in_column", -1),
                        "col_role":  t.get("column_role",""),
                        "header":    "✓" if t.get("is_header") else "",
                        "context":   t.get("column_context_id","") or "",
                        "stream":    t.get("dosage_stream_id", -1),
                        "angle":     round(t.get("angle_deg", 0), 1),
                        "row_conf":  t.get("row_confidence", 0),
                        "col_conf":  t.get("column_confidence", 0),
                    })
                df_tok = pd.DataFrame(tok_rows)
                _CLF_HEX = {"NUTRIENT":"#d6eaff","QUANTITY":"#d4f5d4","UNIT":"#fff3cd",
                             "CONTEXT":"#ead6ff","NOISE":"#f0f0f0","UNKNOWN":"#ffd6d6"}
                def _tok_style(r):
                    bg = _CLF_HEX.get(r.get("label",""), "")
                    return [f"background-color:{bg}"] * len(r)
                st.dataframe(df_tok.style.apply(_tok_style, axis=1),
                             use_container_width=True, hide_index=True, height=420)
                st.caption(f"Total: **{len(tok_rows)}** enriched tokens")
            else:
                st.info("No enriched tokens.")

        with tab_diag:
            st.markdown("**Full enricher diagnostics**")
            st.json(enr_diag)

    note_widget(image_id, "Stage 3.5 · Token Enricher")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 4 · GRAPH  (with edge-type explanations)
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 4" in PAGE:
    image_id = st.session_state.get("pipeline_image")
    inter    = st.session_state.get("pipeline_inter",{})
    s3       = inter.get("stage3_classified",[])
    G        = inter.get("stage4_graph",{})

    _all_etypes = set(e["type"] for e in G.get("edges", []))
    _is_v2_graph = bool(_all_etypes & {"ROW_COMPAT", "COL_COMPAT", "DIRECTIONAL_ADJ", "HEADER_SCOPE"})

    if _is_v2_graph:
        EDGE_TYPES = ["ROW_COMPAT", "COL_COMPAT", "DIRECTIONAL_ADJ", "HEADER_SCOPE"]
        EDGE_COLOURS_GRAPH = {
            "ROW_COMPAT":       (100, 149, 237, 180),
            "COL_COMPAT":       (255, 165,   0, 180),
            "DIRECTIONAL_ADJ":  ( 60, 179, 113, 180),
            "HEADER_SCOPE":     (220,  80,  80, 220),
        }
        EDGE_HEX = {
            "ROW_COMPAT":       "#6495ED",
            "COL_COMPAT":       "#FFA500",
            "DIRECTIONAL_ADJ":  "#3CB371",
            "HEADER_SCOPE":     "#DC5050",
        }
    else:
        EDGE_TYPES = ["SAME_ROW", "SAME_COL", "ADJACENT", "CONTEXT_SCOPE"]
        EDGE_COLOURS_GRAPH = {
            "SAME_ROW":      (100, 149, 237, 180),
            "SAME_COL":      (255, 165,   0, 180),
            "ADJACENT":      ( 60, 179, 113, 180),
            "CONTEXT_SCOPE": (220,  80,  80, 220),
        }
        EDGE_HEX = {
            "SAME_ROW":      "#6495ED",
            "SAME_COL":      "#FFA500",
            "ADJACENT":      "#3CB371",
            "CONTEXT_SCOPE": "#DC5050",
        }

    edge_counts = Counter(e["type"] for e in G.get("edges",[]))
    summary     = "  ".join(f"{et}:{n}" for et,n in sorted(edge_counts.items()))

    _graph_ver = "V2 Geometry-Aware" if _is_v2_graph else "V1"

    stage_banner(
        f"🕸️ Stage 4 · Graph Constructor ({_graph_ver})",
        f"{'Enriched' if _is_v2_graph else 'Labelled'} tokens ({len(s3)} nodes)",
        f"Typed semantic graph — {G.get('num_nodes',0)} nodes, {G.get('num_edges',0)} edges  [{summary}]"
    )
    if not image_id: pipeline_required(); st.stop()

    # ── Tab layout: overlay / explanations ────────────────────────────────────
    tab_overlay, tab_explain = st.tabs(["🖼 Overlay & Filter", "📖 Edge-Type Explanations"])

    with tab_overlay:
        available_types = [et for et in EDGE_TYPES if edge_counts.get(et, 0) > 0]
        selected_types  = st.multiselect(
            "🔍 Show edge types",
            options   = available_types,
            default   = available_types,
            key       = "graph_edge_filter",
            help      = "Select one or more edge types to display on the overlay. "
                        "Deselect all others to isolate a single relation.",
            format_func = lambda et: f"{et}  ({edge_counts.get(et,0)})",
        )

        legend_html = "  ".join(
            f"<span style='background:{EDGE_HEX[et]};color:#fff;"
            f"padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600'>{et}</span>"
            for et in EDGE_TYPES
        )
        st.markdown(legend_html, unsafe_allow_html=True)
        st.caption("Use the filter above to isolate individual edge types — e.g. select only one to inspect its alignment. "
                   "Click **📖 Edge-Type Explanations** tab for what each relation means.")

        def overlay_graph_filtered(image_name: str, G: Dict,
                                   show_types: List[str]) -> Optional[Image.Image]:
            from PIL import ImageDraw
            img = _open_img(image_name)
            if img is None: return None
            draw     = ImageDraw.Draw(img, "RGBA")
            node_map = {n["id"]: n for n in G.get("nodes", [])}

            for edge in G.get("edges", []):
                if edge["type"] not in show_types:
                    continue
                src = node_map.get(edge["src"]); dst = node_map.get(edge["dst"])
                if not src or not dst: continue
                sc  = ((src["x1"]+src["x2"])//2, (src["y1"]+src["y2"])//2)
                dc  = ((dst["x1"]+dst["x2"])//2, (dst["y1"]+dst["y2"])//2)
                col = EDGE_COLOURS_GRAPH.get(edge["type"], (180,180,180,120))
                w   = 3 if edge["type"] in ("CONTEXT_SCOPE","HEADER_SCOPE") else 2
                draw.line([sc, dc], fill=col, width=w)

            for node in G.get("nodes", []):
                col    = LABEL_COLOURS.get(node.get("label","UNKNOWN"),(200,200,200))
                x1,y1,x2,y2 = node["x1"],node["y1"],node["x2"],node["y2"]
                draw.rectangle([x1,y1,x2,y2], fill=(*col,30), outline=(*col,200), width=1)
            return img

        c_vis, c_data = st.columns([2, 3])
        with c_vis:
            if selected_types:
                ov = overlay_graph_filtered(image_id, G, selected_types)
            else:
                ov = overlay_graph_filtered(image_id, G, [])
            if ov:
                caption = ("All edge types" if set(selected_types) == set(available_types)
                           else f"Showing: {', '.join(selected_types)}" if selected_types
                           else "Nodes only — no edge types selected")
                st.image(ov, use_column_width=True, caption=caption)
            else:
                st.info("Run the pipeline first to see the graph overlay.")

        with c_data:
            cn, ce = st.columns(2)
            with cn:
                st.markdown("**Node labels**")
                lc = Counter(n["label"] for n in G.get("nodes",[]))
                st.dataframe(
                    pd.DataFrame([{"Label":l,"Count":c} for l,c in sorted(lc.items(),key=lambda x:-x[1])]),
                    use_container_width=True, hide_index=True)
            with ce:
                st.markdown("**Edge types**")
                edge_rows = []
                for et in EDGE_TYPES:
                    cnt = edge_counts.get(et, 0)
                    active = "✅" if et in selected_types else "⬜"
                    edge_rows.append({"":active, "Type":et, "Count":cnt})
                st.dataframe(pd.DataFrame(edge_rows), use_container_width=True,
                             hide_index=True)

            scope_type = "HEADER_SCOPE" if _is_v2_graph else "CONTEXT_SCOPE"
            ctx_edges = [e for e in G.get("edges",[]) if e["type"]==scope_type]
            if ctx_edges:
                st.markdown(f"**{scope_type} edges ({len(ctx_edges)}) — context header → governed tokens**")
                nm = {n["id"]:n for n in G.get("nodes",[])}
                ctx_rows = [
                    {
                        "Header token": nm.get(e["src"],{}).get("token","?"),
                        "norm/context": nm.get(e["src"],{}).get("column_context_id","") or nm.get(e["src"],{}).get("norm",""),
                        "→ Target":     nm.get(e["dst"],{}).get("token","?"),
                        "Dst label":    nm.get(e["dst"],{}).get("label","?"),
                    }
                    for e in ctx_edges[:60]
                ]
                st.dataframe(pd.DataFrame(ctx_rows), use_container_width=True, hide_index=True)
                if len(ctx_edges) > 60:
                    st.caption(f"Showing 60 of {len(ctx_edges)}")
            else:
                st.warning(f"{scope_type} edges = 0 — context will not propagate to any tuples. "
                           "Check Stage 3 classifier CONTEXT token count.")

            col_etype = "COL_COMPAT" if _is_v2_graph else "SAME_COL"
            if selected_types == [col_etype]:
                col_edges = [e for e in G.get("edges",[]) if e["type"]==col_etype]
                if col_edges:
                    st.markdown(f"**{col_etype} edges ({len(col_edges)}) — vertically aligned tokens**")
                    nm = {n["id"]:n for n in G.get("nodes",[])}
                    col_rows = [
                        {
                            "Token A":       nm.get(e["src"],{}).get("token","?"),
                            "Label A":       nm.get(e["src"],{}).get("label","?"),
                            "Token B":       nm.get(e["dst"],{}).get("token","?"),
                            "Label B":       nm.get(e["dst"],{}).get("label","?"),
                        }
                        for e in col_edges[:60]
                    ]
                    st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    with tab_explain:
        render_edge_explanation(EDGE_TYPES, _is_v2_graph)

    note_widget(image_id, "Stage 4 · Graph")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: STAGE 5 · ASSOCIATION
# ════════════════════════════════════════════════════════════════════════════

elif "Stage 5" in PAGE:
    image_id = st.session_state.get("pipeline_image")
    inter    = st.session_state.get("pipeline_inter",{})
    G        = inter.get("stage4_graph",{})
    tuples   = st.session_state.get("pipeline_tuples",[]) or []
    metrics  = st.session_state.get("pipeline_metrics")
    pair_df  = st.session_state.get("pipeline_pair_df")
    gt_df    = load_gt_df()
    gt_rows  = gt_df[gt_df["image_id"]==image_id][GT_COLS].to_dict(orient="records") if image_id else []

    stage_banner(
        "🔗 Stage 5 · Association",
        f"Typed semantic graph ({G.get('num_nodes',0)} nodes, {G.get('num_edges',0)} edges)",
        f"Structured tuples — {len(tuples)} extracted  (GT: {len(gt_rows)})"
    )
    if not image_id: pipeline_required(); st.stop()

    # Show fallback mode badges if triggered
    diag = st.session_state.get("pipeline_diag", {})
    if diag.get("paragraph_mode_triggered"):
        st.info(f"📝 **Paragraph mode activated** — graph produced "
                f"{diag.get('paragraph_tuples_before',0)} tuples, "
                f"paragraph extractor recovered "
                f"**{diag.get('paragraph_tuples_after',0)}** tuples from fused OCR text.")
    elif diag.get("sentence_mode_triggered"):
        st.info(f"💬 **Sentence mode activated** — graph produced 0 tuples, "
                f"sentence extractor recovered **{diag.get('sentence_tuples',0)}** "
                f"tuples from running-text patterns.")

    c_vis, c_data = st.columns([2, 3])
    with c_vis:
        from PIL import ImageDraw
        img = _open_img(image_id)
        if img and tuples:
            draw = ImageDraw.Draw(img, "RGBA")
            node_map = {n["id"]:n for n in G.get("nodes",[])}
            for tup in tuples:
                nid = tup.get("node_id") or tup.get("nutrient_node_id")
                if nid and nid in node_map:
                    nd = node_map[nid]
                    has_all = tup.get("quantity") and tup.get("unit") and tup.get("context")
                    col = (60,179,113) if has_all else (240,173,78)
                    draw.rectangle([nd["x1"],nd["y1"],nd["x2"],nd["y2"]],
                                   fill=(*col,50), outline=(*col,220), width=2)
        if img: st.image(img, use_column_width=True)
        st.markdown("""<div style="font-size:11px">
        <span style="color:#3CB371">■</span> full tuple &nbsp;
        <span style="color:#F0AD4E">■</span> partial tuple
        </div>""", unsafe_allow_html=True)

    with c_data:
        cg, cp = st.columns(2)
        with cg:
            st.markdown(f"**Ground Truth ({len(gt_rows)})**")
            if gt_rows: st.dataframe(pd.DataFrame(gt_rows), use_container_width=True, hide_index=True, height=240)
        with cp:
            st.markdown(f"**Predicted ({len(tuples)})**")
            if tuples:
                pf = pd.DataFrame([{k:t.get(k,"") for k in GT_COLS} for t in tuples])
                st.dataframe(pf, use_container_width=True, hide_index=True, height=240)

        if metrics:
            st.markdown("**Metrics vs Exp 1 baseline**")
            metrics_row(metrics)

        if pair_df is not None and not pair_df.empty:
            st.markdown("**Diff table**")
            disp, sty = build_diff_df(pair_df)
            if not disp.empty:
                st.dataframe(disp.style.apply(lambda _: sty.values, axis=None),
                             use_container_width=True, hide_index=True, height=340)
    note_widget(image_id, "Stage 5 · Association")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION  (NEW — dual fast + LLM pass)
# ════════════════════════════════════════════════════════════════════════════

elif "🧪" in PAGE and "Evaluation" in PAGE:
    st.title("🧪 Evaluation")
    st.caption("Run **both** the fast rule-based evaluator AND the LLM-assisted evaluator on "
               "a predicted-tuples source, and compare them side-by-side.")

    # ── Source selection ─────────────────────────────────────────────────────
    src_col1, src_col2 = st.columns([2, 3])
    with src_col1:
        source_mode = st.radio(
            "Prediction source",
            ["Current pipeline run (single image)", "Previous experiment (full dataset)"],
            key="eval_src"
        )
    with src_col2:
        llm_model = st.text_input("LLM model (Ollama)", value="qwen2.5:7b",
                                  help="Must be pulled locally via `ollama pull <model>`")

    pred_tuples: List[Dict] = []
    gt_rows: List[Dict]     = []
    eval_scope              = ""

    if source_mode.startswith("Current"):
        img_sel = st.session_state.get("pipeline_image")
        tup_sel = st.session_state.get("pipeline_tuples")
        if not img_sel or tup_sel is None:
            st.info("No pipeline run yet — run the pipeline on the **🏠 Run Pipeline** page first.")
            st.stop()
        pred_tuples = tup_sel
        gt_df_all = load_gt_df()
        img_gt_df = gt_df_all[gt_df_all["image_id"] == img_sel]
        gt_rows = [
            {"image_id": img_sel, **{k: r[k] for k in GT_COLS}}
            for _, r in img_gt_df.iterrows()
        ]
        eval_scope = f"single image `{img_sel}` ({len(pred_tuples)} predicted, {len(gt_rows)} GT)"
    else:
        outputs_dir = PROJECT_ROOT / "outputs"
        exp_dirs = (sorted([d.name for d in outputs_dir.iterdir() if d.is_dir()])
                    if outputs_dir.exists() else [])
        if not exp_dirs:
            st.warning(f"No experiment outputs found in `{outputs_dir}`"); st.stop()
        sel_exp = st.selectbox("📁 Experiment", exp_dirs, index=len(exp_dirs)-1, key="eval_exp_sel")
        exp_path = outputs_dir / sel_exp
        tup_csv = exp_path / "tuples.csv"
        if not tup_csv.exists():
            st.warning(f"`tuples.csv` not found in `{exp_path}`"); st.stop()
        df_pred = pd.read_csv(tup_csv, encoding="utf-8")
        pred_tuples = df_pred.to_dict(orient="records")
        gt_df_all = load_gt_df()
        gt_rows = gt_df_all.to_dict(orient="records")
        eval_scope = f"experiment `{sel_exp}` ({len(pred_tuples)} predicted, {len(gt_rows)} GT)"

    st.info(f"📦 **Scope:** {eval_scope}")

    # ── Action buttons ───────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    run_fast = btn_col1.button("⚡ Run Fast (rule-based only)", type="primary", key="eval_run_fast")
    run_llm  = btn_col2.button("🤖 Run LLM-assisted", key="eval_run_llm")
    run_both = btn_col3.button("🔬 Run BOTH + compare", key="eval_run_both")

    # ── Execute ──────────────────────────────────────────────────────────────
    if run_fast or run_both:
        with st.spinner("Running fast evaluator …"):
            t0 = time.time()
            fast_m, fast_pdf = run_full_evaluation(gt_rows, pred_tuples, use_llm=False, model=llm_model)
            fast_time = round(time.time()-t0, 1)
        st.session_state["eval_fast_metrics"] = fast_m
        st.session_state["eval_fast_pdf"]     = fast_pdf
        st.session_state["eval_fast_time"]    = fast_time

    if run_llm or run_both:
        with st.spinner("Running LLM-assisted evaluator (this may take minutes) …"):
            t0 = time.time()
            llm_m, llm_pdf = run_full_evaluation(gt_rows, pred_tuples, use_llm=True, model=llm_model)
            llm_time = round(time.time()-t0, 1)
        st.session_state["eval_llm_metrics"] = llm_m
        st.session_state["eval_llm_pdf"]     = llm_pdf
        st.session_state["eval_llm_time"]    = llm_time

    # ── Display ──────────────────────────────────────────────────────────────
    fast_m  = st.session_state.get("eval_fast_metrics")
    llm_m   = st.session_state.get("eval_llm_metrics")
    fast_pdf = st.session_state.get("eval_fast_pdf")
    llm_pdf  = st.session_state.get("eval_llm_pdf")

    if fast_m or llm_m:
        st.divider()
        st.markdown("### 📊 Evaluation Metrics Comparison")

        # Side-by-side metric table
        metric_rows = []
        for k, lbl in METRIC_LABELS.items():
            row = {"Metric": lbl}
            if fast_m: row["Fast-only"]   = f"{fast_m.get(k,0):.1%}"
            if llm_m:  row["LLM-assisted"] = f"{llm_m.get(k,0):.1%}"
            if fast_m and llm_m:
                delta = llm_m.get(k,0) - fast_m.get(k,0)
                sign = "+" if delta >= 0 else ""
                row["Δ (LLM − Fast)"] = f"{sign}{delta*100:.1f}pp"
            metric_rows.append(row)
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        # Top-level stats
        col1, col2, col3, col4 = st.columns(4)
        if fast_m:
            col1.metric("Fast: matched", fast_m.get("matched_pairs",0))
            col2.metric("Fast: time", f"{st.session_state.get('eval_fast_time','?')}s")
        if llm_m:
            col3.metric("LLM: matched",  llm_m.get("matched_pairs",0))
            col4.metric("LLM calls",     llm_m.get("llm_calls",0))

    # ── Tabs: fast details / LLM details / diff ─────────────────────────────
    if fast_m or llm_m:
        st.divider()
        tabs = []
        if fast_m: tabs.append("⚡ Fast-pass details")
        if llm_m:  tabs.append("🤖 LLM-pass details")
        if fast_m and llm_m: tabs.append("🔬 Upgraded by LLM (Fast❌→LLM✅)")

        tab_objs = st.tabs(tabs)
        idx = 0

        if fast_m:
            with tab_objs[idx]:
                st.markdown(f"**{len(fast_pdf) if fast_pdf is not None else 0} pairs** — all decided by rule-based matcher.")
                if fast_pdf is not None and not fast_pdf.empty:
                    disp_f, sty_f = build_diff_df(fast_pdf)
                    if not disp_f.empty:
                        st.dataframe(disp_f.style.apply(lambda _: sty_f.values, axis=None),
                                     use_container_width=True, hide_index=True, height=500)
                    st.download_button(
                        "⬇️ Download fast pair details CSV",
                        data=fast_pdf.to_csv(index=False, encoding="utf-8"),
                        file_name="fast_pair_details.csv", mime="text/csv",
                        key="eval_dl_fast"
                    )
            idx += 1

        if llm_m:
            with tab_objs[idx]:
                st.markdown(f"**{len(llm_pdf) if llm_pdf is not None else 0} pairs** — "
                           f"{llm_m.get('fast_hits',0)} fast-hits + {llm_m.get('llm_calls',0)} LLM calls.")
                if llm_pdf is not None and not llm_pdf.empty:
                    disp_l, sty_l = build_diff_df(llm_pdf)
                    if not disp_l.empty:
                        st.dataframe(disp_l.style.apply(lambda _: sty_l.values, axis=None),
                                     use_container_width=True, hide_index=True, height=500)
                    st.download_button(
                        "⬇️ Download LLM pair details CSV",
                        data=llm_pdf.to_csv(index=False, encoding="utf-8"),
                        file_name="llm_pair_details.csv", mime="text/csv",
                        key="eval_dl_llm"
                    )
            idx += 1

        if fast_m and llm_m:
            with tab_objs[idx]:
                st.markdown("**Pairs where the LLM evaluator rescued a match the fast evaluator missed.** "
                           "These are the cases where the LLM added value over rule-based matching.")
                if fast_pdf is not None and llm_pdf is not None and \
                   not fast_pdf.empty and not llm_pdf.empty:
                    # Join on (image_id, gt_nutrient, pred_nutrient) to find upgrades
                    fp = fast_pdf.copy()
                    lp = llm_pdf.copy()

                    # Normalize qty col names
                    qty_gt_f = "gt_qty" if "gt_qty" in fp.columns else "gt_quantity"
                    qty_gt_l = "gt_qty" if "gt_qty" in lp.columns else "gt_quantity"

                    key_cols = ["image_id", "gt_nutrient", "pred_nutrient", qty_gt_f]
                    # Get fast-4f-false rows
                    fp_fail = fp[fp.get("full_match_4f", False) != True].copy()
                    # Get llm-4f-true rows
                    lp_win  = lp[lp.get("full_match_4f", False) == True].copy()

                    # Try merging
                    try:
                        upgraded = fp_fail.merge(
                            lp_win,
                            left_on=key_cols,
                            right_on=["image_id","gt_nutrient","pred_nutrient", qty_gt_l],
                            suffixes=("_fast","_llm"),
                            how="inner"
                        )
                        if not upgraded.empty:
                            st.success(f"🎉 LLM upgraded **{len(upgraded)}** pairs that fast evaluator marked as mismatch!")
                            show_cols = ["image_id","gt_nutrient","pred_nutrient",
                                         "gt_unit_llm","pred_unit_llm",
                                         "gt_context_llm","pred_context_llm"]
                            show_cols = [c for c in show_cols if c in upgraded.columns]
                            st.dataframe(upgraded[show_cols],
                                         use_container_width=True, hide_index=True, height=400)
                        else:
                            st.info("No pairs were upgraded by the LLM — fast evaluator caught everything.")
                    except Exception as e:
                        st.warning(f"Could not compute upgrade diff: {e}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREVIOUS EXPERIMENTS  (renamed from "LLM Evaluator")
# ════════════════════════════════════════════════════════════════════════════

elif "📊" in PAGE:
    st.title("📊 Previous Experiments")
    st.caption("Load any experiment output to inspect its metrics and per-pair field evaluation details.")

    outputs_dir = PROJECT_ROOT / "outputs"
    exp_dirs    = (sorted([d.name for d in outputs_dir.iterdir() if d.is_dir()])
                   if outputs_dir.exists() else [])
    if not exp_dirs:
        st.warning(f"No experiment outputs found in `{outputs_dir}`"); st.stop()

    selected_exp = st.selectbox("📁 Experiment output", exp_dirs,
                                index=len(exp_dirs)-1, key="llm_exp_sel")
    exp_path     = outputs_dir / selected_exp
    metrics_file = exp_path / "evaluation_results.json"
    detail_file  = exp_path / "llm_evaluation_details.csv"

    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            exp_metrics = _json.load(f)
        hc1, hc2, hc3 = st.columns([4, 1, 1])
        with hc1:
            st.markdown(f"**Evaluator:** `{exp_metrics.get('evaluator','—')}`")
            st.markdown(f"**Timestamp:** `{exp_metrics.get('timestamp','—')}`")
            if exp_metrics.get("notes"): st.caption(exp_metrics["notes"])
        with hc2:
            st.metric("LLM calls",      exp_metrics.get("llm_calls",0))
            st.metric("Fast-pass hits", exp_metrics.get("fast_hits",0))
        with hc3:
            st.metric("GT tuples",   exp_metrics.get("gt_tuples",0))
            st.metric("Pred tuples", exp_metrics.get("predicted_tuples",0))

        st.markdown("#### Metrics")
        mc = st.columns(6)
        for col, (k, lbl) in zip(mc, [("nutrient_f1","Nutrient F1"),("quantity_acc","Qty Acc"),
                                       ("unit_acc","Unit Acc"),("context_acc","Context Acc"),
                                       ("full3f_f1","Full F1 (3f)"),("full_tuple_f1","Full F1 (4f)")]):
            col.metric(lbl, f"{exp_metrics.get(k,0):.1%}")
        cost = exp_metrics.get("context_cost_pp")
        if cost is not None:
            st.info(f"Context cost 3f→4f: **−{cost}pp** · 3f={exp_metrics.get('full3f_f1',0):.1%} → 4f={exp_metrics.get('full_tuple_f1',0):.1%}")
    else:
        st.warning(f"`evaluation_results.json` not found in `{exp_path}`")

    st.markdown("---")
    if not detail_file.exists():
        st.warning(f"`llm_evaluation_details.csv` not found in `{exp_path}`"); st.stop()

    detail_df = pd.read_csv(detail_file, encoding="utf-8")
    _cols = detail_df.columns.tolist()
    _has_4f_col = "full_match_4f" in _cols
    _has_3f_col = "full_match_3f" in _cols
    _has_fm_col = "full_match"    in _cols

    fa, fb, fc, fd = st.columns([2,2,2,2])
    with fa:
        img_opts = ["All"] + sorted(detail_df["image_id"].dropna().unique().tolist())
        sel_img  = st.selectbox("Image", img_opts, key="llm_img")
    with fb:
        m_opts = ["All"] + sorted(detail_df["eval_method"].dropna().unique().tolist()) if "eval_method" in _cols else ["All"]
        sel_method = st.selectbox("Eval method", m_opts, key="llm_method")
    with fc:
        match_opts = ["All","✅ Full 4f","🟨 Full 3f","🟡 Partial","❌ Mismatch"]
        sel_match  = st.selectbox("Match outcome", match_opts, key="llm_match")
    with fd:
        show_reasoning = st.toggle("Show field reasoning", value=True, key="llm_reasoning")

    def _b(v):
        if isinstance(v,bool): return v
        if isinstance(v,float): return bool(v) if v==v else False
        return str(v).lower() in ("true","1","yes")

    filtered = detail_df.copy()
    if sel_img    != "All": filtered = filtered[filtered["image_id"]==sel_img]
    if sel_method != "All" and "eval_method" in filtered.columns:
        filtered = filtered[filtered["eval_method"]==sel_method]

    MATCH_BG = {"✅ Full 4f":"#d4f5d4","🟨 Full 3f":"#fff9c4","🟡 Partial":"#ffe8cc","❌ Mismatch":"#ffd6d6"}
    CHECK_C  = {"✓":"#d4f5d4","✗":"#ffd6d6"}

    display_rows = []
    for _, r in filtered.iterrows():
        nm = _b(r.get("nutrient_match",False)); qm = _b(r.get("quantity_match",False))
        um = _b(r.get("unit_match",False));     cm = _b(r.get("context_match",False))
        mth = str(r.get("eval_method","") or "")
        is_4f = _b(r.get("full_match_4f",False)) if _has_4f_col else (_b(r.get("full_match",False)) if _has_fm_col else (nm and qm and um and cm))
        is_3f = _b(r.get("full_match_3f",False)) if _has_3f_col else (nm and qm and um)
        icon = ("✅ Full 4f" if is_4f else "🟨 Full 3f" if is_3f else "🟡 Partial" if nm else "❌ Mismatch")
        qty_gt = "gt_qty" if "gt_qty" in _cols else "gt_quantity"
        qty_pd = "pred_qty" if "pred_qty" in _cols else "pred_quantity"
        row = {"Image":str(r.get("image_id","")),
               "GT Nutrient":str(r.get("gt_nutrient","")), "Pred Nutrient":str(r.get("pred_nutrient","")),
               "GT Qty":str(r.get(qty_gt,"")),             "Pred Qty":str(r.get(qty_pd,"")),
               "GT Unit":str(r.get("gt_unit","")),         "Pred Unit":str(r.get("pred_unit","")),
               "GT Context":str(r.get("gt_context","")),   "Pred Context":str(r.get("pred_context","")),
               "N":"✓" if nm else "✗","Q":"✓" if qm else "✗",
               "U":"✓" if um else "✗","C":"✓" if cm else "✗",
               "Match":icon,"Method":mth}
        if show_reasoning:
            row["N reason"] = str(r.get("nutrient_reason","") or "")
            row["Q reason"] = str(r.get("quantity_reason","") or "")
            row["U reason"] = str(r.get("unit_reason","") or "")
            row["C reason"] = str(r.get("context_reason","") or "")
        display_rows.append(row)

    if sel_match != "All":
        display_rows = [r for r in display_rows if r["Match"]==sel_match]

    if not display_rows:
        st.info("No pairs match the current filters."); st.stop()

    disp_df = pd.DataFrame(display_rows)
    counts  = {k: sum(1 for r in display_rows if r["Match"]==k) for k in match_opts[1:]}
    s1c,s2c,s3c,s4c,s5c = st.columns(5)
    s1c.metric("Showing",     len(display_rows))
    s2c.metric("✅ Full 4f",  counts.get("✅ Full 4f",0))
    s3c.metric("🟨 Full 3f",  counts.get("🟨 Full 3f",0))
    s4c.metric("🟡 Partial",  counts.get("🟡 Partial",0))
    s5c.metric("❌ Mismatch", counts.get("❌ Mismatch",0))

    def _style(r):
        bg = MATCH_BG.get(r.get("Match",""),""); cols = r.index.tolist()
        out = []
        for c in cols:
            if c=="Match":     out.append(f"background-color:{bg}")
            elif c in("N","Q","U","C"): out.append(f"background-color:{CHECK_C.get(r[c],'')}")
            elif "reason" in c: out.append("font-size:11px;color:#444;font-style:italic")
            else: out.append("")
        return out

    all_img_ids = sorted(detail_df["image_id"].dropna().unique().tolist())
    if sel_img != "All":
        _display_img_id = sel_img
    else:
        _display_img_id = st.selectbox("📷 Label image", all_img_ids, key="llm_display_img") if all_img_ids else None

    img_col, tbl_col = st.columns([1,3])
    with img_col:
        if _display_img_id:
            lbl_img = _open_img(_display_img_id)
            if lbl_img: st.image(lbl_img, caption=_display_img_id, use_container_width=True)
    with tbl_col:
        st.dataframe(disp_df.style.apply(_style,axis=1), use_container_width=True, hide_index=True, height=520)

    st.download_button("⬇️ Download filtered CSV",
                       data=disp_df.to_csv(index=False,encoding="utf-8"),
                       file_name=f"{selected_exp}_filtered.csv", mime="text/csv",
                       key="llm_download")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: NOTES
# ════════════════════════════════════════════════════════════════════════════

elif "📓" in PAGE:
    st.title("📓 Pipeline Notes")
    st.caption(f"All notes saved to `{NOTES_FILE.relative_to(PROJECT_ROOT)}`")

    notes_df = load_notes()
    if notes_df.empty:
        st.info("No notes saved yet. Add notes from any Stage page.")
    else:
        fc, sc, _ = st.columns([2,2,3])
        with fc:
            img_filter = st.multiselect("Filter by image", sorted(notes_df["image_id"].unique()), default=[])
        with sc:
            stage_filter = st.multiselect("Filter by stage", sorted(notes_df["stage"].unique()), default=[])
        filtered = notes_df.copy()
        if img_filter:   filtered = filtered[filtered["image_id"].isin(img_filter)]
        if stage_filter: filtered = filtered[filtered["stage"].isin(stage_filter)]
        st.markdown(f"**{len(filtered)} notes** (of {len(notes_df)} total)")
        st.dataframe(filtered[["timestamp","image_id","stage","note"]].reset_index(drop=True),
                     use_container_width=True, hide_index=True,
                     column_config={"timestamp":st.column_config.TextColumn("Time",width="small"),
                                    "image_id": st.column_config.TextColumn("Image",width="small"),
                                    "stage":    st.column_config.TextColumn("Stage",width="medium"),
                                    "note":     st.column_config.TextColumn("Note", width="large")})
        st.download_button("⬇️ Download notes as TSV",
                           data=filtered.to_csv(sep="\t",index=False),
                           file_name="pipeline_notes.tsv", mime="text/tab-separated-values")

    st.divider()
    st.markdown("**➕ Add a note directly**")
    na, nb, nc = st.columns([2,2,4])
    with na: note_img   = st.selectbox("Image",  images_list or ["—"], key="notes_img")
    with nb: note_stage = st.selectbox("Stage",  STAGE_NAMES, key="notes_stage")
    with nc: note_text  = st.text_input("Note",  key="notes_text", placeholder="Write note…")
    if st.button("💾 Save", key="notes_save_btn"):
        if note_text.strip() and note_img != "—":
            save_note(note_img, note_stage, note_text); st.success("Saved."); st.rerun()
        else:
            st.warning("Please select an image and write a note.")