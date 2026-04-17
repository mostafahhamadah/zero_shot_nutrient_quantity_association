"""
gemma_classifier.py
===================
Stage 3 — Gemma Semantic Classifier (Experiment 5)

PURPOSE
-------
Replace Qwen 2.5 (Experiment 4) with Google's Gemma model via Ollama
for zero-shot semantic classification of OCR tokens from supplement labels.

Gemma addresses the same structural failures as Qwen:
  1. German context headers (Je Tagesdosis, Pro 100g) — Gemma is multilingual
  2. Isolated unit tokens (g, mg, µg) — Gemma sees full row context

This classifier is architecturally identical to qwen_classifier.py.
The only differences are the model ID, system prompt phrasing, and
the Ollama model name constant. All fuzzy remapping, CONTEXT_MAP,
serializer integration, and output contract are unchanged.

DESIGN DECISIONS (mirrors qwen_classifier.py — locked)
-------------------------------------------------------
Strategy        : Option B — full serialized text per image sent to Gemma.
                  One Ollama call per image.

Coverage        : Full replacement — Gemma classifies all 5 labels.
                  NUTRIENT, QUANTITY, UNIT, CONTEXT, NOISE (5-label scheme).

JSON schema     : Schema A — entity text + label only.
                  {"entities": [{"text": "...", "label": "..."}]}
                  No character offsets — LLMs hallucinate positions.

Remapping       : Fuzzy text match between Gemma entity text and token list.
                  Single-word entities: SequenceMatcher per token.
                  Multi-word entities: n-gram window over consecutive token spans.
                  Threshold: 0.65 (permissive — catches OCR variants).

Serving size    : Labelled CONTEXT directly by Gemma.
                  Prompt instructs: "2 Kapseln", "70g" etc. → CONTEXT.

Fallback label  : UNKNOWN — tokens not matched by any Gemma entity.

CONTEXT handling: CONTEXT_MAP normalises CONTEXT token text to canonical form
                  (per_100g / per_serving / per_daily_dose).
                  Same map as experiment_01 classifier and qwen_classifier.

PIPELINE POSITION
-----------------
Stage 2 (PaddleOCR corrector, split_fused_token applied)
    → text_serializer.serialize_tokens_for_gliner()   (reused as text source)
    → [this module] gemma_classifier.py  →  labelled token list
    → graph_constructor.py               (Stage 4, unchanged)

OUTPUT CONTRACT
---------------
Returns list of token dicts. Each dict is the original Stage 2 token dict with:
    "label"          : str    NUTRIENT|QUANTITY|UNIT|CONTEXT|UNKNOWN
    "norm"           : str|None  canonical context form for CONTEXT tokens
    "gemma_entity"   : str|None  Gemma entity text that matched this token
    "gemma_score"    : float|None  SequenceMatcher ratio of the match

USAGE
-----
    from gemma_classifier import GemmaClassifier
    classifier = GemmaClassifier()
    labelled   = classifier.classify(tokens)

    # Custom model (e.g. gemma3:12b)
    classifier = GemmaClassifier(model_id="gemma3:12b")

SUPPORTED OLLAMA MODELS
-----------------------
    gemma3:1b    — fastest, lowest accuracy
    gemma3:4b    — recommended default (light, good multilingual)
    gemma3:12b   — higher accuracy, slower
    gemma3:27b   — highest accuracy, requires strong GPU

THESIS CITATION
---------------
    Google DeepMind (2025). Gemma 3 Technical Report.
    Model: gemma3:4b via Ollama local inference.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests

from src.utils.text_serializer import serialize_tokens_for_gliner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ollama API endpoint — must be running locally
OLLAMA_URL: str = "http://localhost:11434/api/chat"

#: Default Gemma model name as registered in Ollama
#: gemma3:4b is recommended — light enough for CPU, good multilingual support
MODEL_ID: str = "gemma4:e4b"

#: Temperature — 0.0 for fully deterministic output
TEMPERATURE: float = 0.0

#: Fuzzy match threshold for entity text → token remapping
#: 0.65 = permissive, catches OCR variants (Magnesiumcitraat vs Magnesiumcitrat)
FUZZY_THRESHOLD: float = 0.65

#: Label for tokens not matched by any Gemma entity
FALLBACK_LABEL: str = "UNKNOWN"

#: Valid labels Gemma may return — any other value is rejected
VALID_LABELS: frozenset = frozenset({"NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE"})

#: Ollama request timeout in seconds
REQUEST_TIMEOUT: int = 600

#: CONTEXT_MAP — normalises CONTEXT token text to canonical pipeline form.
#: Identical to qwen_classifier.py and experiment_01_final_semantic_classifier.py.
CONTEXT_MAP: Dict[str, str] = {
    # German — per daily dose
    "je tagesdosis":         "per_daily_dose",
    "pro tagesdosis":        "per_daily_dose",
    "tagesdosis":            "per_daily_dose",
    "tagesbedarf":           "per_daily_dose",
    "je 2 kapseln":          "per_daily_dose",
    "je 3 kapseln":          "per_daily_dose",
    "je 4 kapseln":          "per_daily_dose",
    # German — per 100g
    "je 100 g":              "per_100g",
    "pro 100 g":             "per_100g",
    "pro 100g":              "per_100g",
    "per 100g":              "per_100g",
    "per 100 g":             "per_100g",
    "je 100g":               "per_100g",
    "1009":                  "per_100g",
    "je 1009":               "per_100g",
    "pro 1009":              "per_100g",
    "100g":                  "per_100g",
    # German — per serving
    "pro portion":           "per_serving",
    "je portion":            "per_serving",
    "portion":               "per_serving",
    "pro106":                "per_serving",
    "portion**":             "per_serving",
    # English — per daily dose
    "per daily dose":        "per_daily_dose",
    "daily dose":            "per_daily_dose",
    "daily value":           "per_daily_dose",
    "per day":               "per_daily_dose",
    # English — per serving
    "per serving":           "per_serving",
    "serving":               "per_serving",
    "per portion":           "per_serving",
    # English — per 100g
    "per 100 g":             "per_100g",
    "per 100g":              "per_100g",
    # French
    "par dose journalière":  "per_daily_dose",
    "par portion":           "per_serving",
    "pour 100 g":            "per_100g",
    # Dutch
    "per dagelijkse dosis":  "per_daily_dose",
    "per portie":            "per_serving",
    "per 100 g":             "per_100g",
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a nutritional supplement label entity extractor. Your task is to \
identify and classify entities in OCR-extracted text from supplement labels.

Classify each entity into exactly one of these labels:

NUTRIENT  — Any nutritional ingredient, vitamin, mineral, fatty acid, or active \
compound. Examples: Magnesium, Vitamin B12, Kohlenhydrate, Fat, Protein, Calcium, \
davon gesättigte Fettsäuren, Ballaststoffe, Eiweiß, Natrium, Energie, Brennwert, \
Fett, Salz, Zucker, Thiamin, Niacin, Biotin, Folsäure, Cholecalciferol, Creatine.

QUANTITY  — Any standalone numeric value representing an amount or dosage. \
Examples: 400, 2.5, 0.8, 1000, 86.1, 0.03. Do NOT include the unit.

UNIT      — Any unit of measurement. Examples: mg, µg, g, kJ, kcal, ml, I.E., \
mg α-TE, mg NE, µg RE, x 10^9 KBE.

CONTEXT   — Any serving reference header OR serving size descriptor. \
Context headers: Je Tagesdosis, Pro 100g, Per serving, Pro Portion, per_daily_dose, \
per_100g, Je 1 Kapsel. \
Serving size descriptors (label these CONTEXT too): 2 Kapseln, 70g, 4 Tabletten, \
1 Sachet, 30ml — these describe the physical amount constituting one serving.

NOISE     — Table borders, percent signs, NRV percentage numbers, formatting \
characters, asterisks, page numbers, or any non-semantic structural token.

OUTPUT RULES:
1. Return ONLY a valid JSON object — no preamble, no explanation, no markdown.
2. Only include NUTRIENT, QUANTITY, UNIT, and CONTEXT entities. Do NOT include NOISE.
3. Use the exact text as it appears in the input — do not translate or normalise.
4. Multi-word nutrient names must be one entity: {"text": "davon gesättigte Fettsäuren", "label": "NUTRIENT"}.
5. Do not invent entities not present in the input text.
6. Numbers that are clearly NRV percentages (e.g. 107, 100%, 50%) are NOISE — omit them.

REQUIRED JSON FORMAT:
{"entities": [{"text": "<exact text>", "label": "<LABEL>"}, ...]}
"""


def _build_prompt(serialized_text: str) -> str:
    """Build the user-turn prompt with the serialized OCR text."""
    return (
        "Extract all nutritional entities from this supplement label OCR text. "
        "Return only the JSON object.\n\n"
        f"OCR TEXT:\n{serialized_text}"
    )


# ---------------------------------------------------------------------------
# Ollama API caller
# ---------------------------------------------------------------------------

def _call_gemma(
    serialized_text: str,
    model:           str   = MODEL_ID,
    temperature:     float = TEMPERATURE,
    timeout:         int   = REQUEST_TIMEOUT,
) -> Optional[str]:
    """
    Call Gemma via Ollama chat endpoint.

    Uses format="json" to enforce JSON output mode.
    Temperature=0.0 for deterministic results across runs.

    Returns raw response string, or None on any error.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(serialized_text)},
        ],
        "stream":  False,
        "format":  "json",
        "options": {
            "temperature": temperature,
            "seed":        42,
            "num_predict": 2048,   # ← add this, default is ~128 which is too small
            "num_ctx":     8192,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        logger.error(
            "Ollama not reachable at %s — is it running? "
            "Start with: ollama serve", OLLAMA_URL,
        )
        return None
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %ds.", timeout)
        return None
    except Exception as exc:
        logger.error("Gemma API call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> List[Dict[str, str]]:
    """
    Parse Gemma's JSON response into a list of validated entity dicts.
    Drops any entity with an invalid label or empty text.
    """
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$",           "", clean)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        # Attempt recovery — truncated JSON: find last complete entity and close the array
        logger.warning("Gemma JSON parse error: %s | raw: %s", exc, raw[:200])
        try:
            last_close = clean.rfind("},")
            if last_close == -1:
                last_close = clean.rfind("}")
            if last_close > 0:
                recovered = clean[:last_close + 1] + "]}"
                data = json.loads(recovered)
                logger.info("Gemma JSON recovered: truncated response partially parsed.")
            else:
                return []
        except Exception:
            return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        logger.warning("Gemma response missing 'entities' list.")
        return []

    valid = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        text  = str(ent.get("text",  "")).strip()
        label = str(ent.get("label", "")).strip().upper()
        if not text:
            continue
        if label not in VALID_LABELS:
            logger.debug("Gemma returned unknown label '%s' for '%s' — skipped.", label, text)
            continue
        if label == "NOISE":
            continue
        valid.append({"text": text, "label": label})

    logger.debug("Gemma entities parsed: %d valid from %d raw.", len(valid), len(entities))
    return valid


# ---------------------------------------------------------------------------
# Fuzzy token remapper  (identical algorithm to qwen_classifier.py)
# ---------------------------------------------------------------------------

def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_tokens_for_entity(
    entity_text: str,
    token_spans: List[Dict[str, Any]],
    threshold:   float = FUZZY_THRESHOLD,
) -> List[int]:
    """
    Find token indices in token_spans that best match entity_text.

    Single-word entity  → best single-token fuzzy match.
    Multi-word entity   → best N-token sliding window fuzzy match.

    Returns list of token_index values, or empty list if no match >= threshold.
    """
    if not entity_text or not token_spans:
        return []

    words = entity_text.split()

    # ---- Single-word ----
    if len(words) == 1:
        best_ratio, best_idx = 0.0, -1
        for ts in token_spans:
            r = _seq_ratio(entity_text, ts["token_text"])
            if r > best_ratio:
                best_ratio, best_idx = r, ts["token_index"]
        return [best_idx] if best_ratio >= threshold else []

    # ---- Multi-word ----
    n            = len(words)
    sorted_spans = sorted(token_spans, key=lambda x: (x["line_id"], x["start_char"]))
    m            = len(sorted_spans)
    if n > m:
        return []

    best_ratio, best_indices = 0.0, []
    for i in range(m - n + 1):
        window      = sorted_spans[i: i + n]
        window_text = " ".join(w["token_text"] for w in window)
        r           = _seq_ratio(entity_text, window_text)
        if r > best_ratio:
            best_ratio   = r
            best_indices = [w["token_index"] for w in window]

    return best_indices if best_ratio >= threshold else []


def _map_entities_to_tokens(
    entities:    List[Dict[str, str]],
    token_spans: List[Dict[str, Any]],
    threshold:   float = FUZZY_THRESHOLD,
) -> Dict[int, Dict[str, Any]]:
    """
    Map Gemma entity list to token indices.
    Higher score wins when multiple entities match the same token.
    """
    assignments: Dict[int, Dict[str, Any]] = {}

    for ent in entities:
        entity_text     = ent["text"]
        label           = ent["label"]
        matched_indices = _find_tokens_for_entity(entity_text, token_spans, threshold)
        if not matched_indices:
            logger.debug("No token match for Gemma entity '%s' (label=%s).", entity_text, label)
            continue

        first_ts    = next((ts for ts in token_spans
                            if ts["token_index"] == matched_indices[0]), None)
        match_score = (_seq_ratio(entity_text, first_ts["token_text"])
                       if first_ts else 1.0)

        for idx in matched_indices:
            existing = assignments.get(idx)
            if existing is None or match_score >= existing.get("score", 0.0):
                assignments[idx] = {
                    "label":       label,
                    "entity_text": entity_text,
                    "score":       match_score,
                }

    return assignments


# ---------------------------------------------------------------------------
# CONTEXT normalisation
# ---------------------------------------------------------------------------

def _normalise_context(token_text: str) -> Optional[str]:
    """Map CONTEXT token text to canonical form. Returns None if not in map."""
    return CONTEXT_MAP.get(token_text.strip().lower(), None)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class GemmaClassifier:
    """
    Zero-shot semantic token classifier using Gemma via Ollama.

    Architecturally identical to QwenClassifier — only the model and
    diagnostic field names differ (gemma_entity / gemma_score).

    Parameters
    ----------
    model_id   : Ollama model name (default: gemma3:4b).
    threshold  : fuzzy match threshold (default: 0.65).
    ollama_url : Ollama API URL (default: http://localhost:11434/api/chat).
    """

    def __init__(
        self,
        model_id:   str   = MODEL_ID,
        threshold:  float = FUZZY_THRESHOLD,
        ollama_url: str   = OLLAMA_URL,
    ) -> None:
        self.model_id   = model_id
        self.threshold  = threshold
        self.ollama_url = ollama_url
        logger.info("GemmaClassifier initialised: model=%s, threshold=%.2f",
                    model_id, threshold)

    def classify(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a list of corrected OCR tokens using Gemma via Ollama.

        Steps
        -----
        1. Serialize tokens → text + token_spans via text_serializer.
        2. Call Gemma via Ollama with the serialized text.
        3. Parse JSON response → entity list.
        4. Fuzzy-map entity texts → token indices.
        5. Assign FALLBACK_LABEL to unmatched tokens.
        6. Normalise CONTEXT tokens via CONTEXT_MAP.
        7. Return augmented token list.

        Returns
        -------
        List of token dicts with added fields:
            "label"        : str
            "norm"         : str | None
            "gemma_entity" : str | None
            "gemma_score"  : float | None
        """
        if not tokens:
            return []

        # Step 1 — Serialise
        serialized  = serialize_tokens_for_gliner(tokens)
        text        = serialized["text"]
        token_spans = serialized["token_spans"]

        if not text.strip():
            logger.warning("Empty serialized text — all tokens labelled %s.", FALLBACK_LABEL)
            return self._all_fallback(tokens)

        # Step 2 — Call Gemma
        raw = _call_gemma(
            text,
            model=self.model_id,
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT,
        )

        if raw is None:
            logger.warning("Gemma call failed — all tokens labelled %s.", FALLBACK_LABEL)
            return self._all_fallback(tokens)

        # Step 3 — Parse response
        entities = _parse_response(raw)
        logger.debug("Gemma returned %d valid entities for %d tokens.",
                     len(entities), len(tokens))

        # Step 4 — Map entities → token indices
        assignments        = _map_entities_to_tokens(entities, token_spans, self.threshold)
        serialized_indices = {ts["token_index"] for ts in token_spans}

        # Steps 5-6 — Build output list
        result: List[Dict[str, Any]] = []

        for orig_idx, tok in enumerate(tokens):
            out = dict(tok)

            if orig_idx not in serialized_indices:
                out["label"]        = FALLBACK_LABEL
                out["norm"]         = None
                out["gemma_entity"] = None
                out["gemma_score"]  = None
                result.append(out)
                continue

            match = assignments.get(orig_idx)

            if match is None:
                out["label"]        = FALLBACK_LABEL
                out["norm"]         = None
                out["gemma_entity"] = None
                out["gemma_score"]  = None
            else:
                out["label"]        = match["label"]
                out["gemma_entity"] = match["entity_text"]
                out["gemma_score"]  = match["score"]

                if match["label"] == "CONTEXT":
                    out["norm"] = _normalise_context(str(tok.get("token", "")))
                    if out["norm"] is None:
                        logger.debug(
                            "CONTEXT token '%s' not in CONTEXT_MAP (serving size or variant).",
                            tok.get("token", ""),
                        )
                else:
                    out["norm"] = None

            result.append(out)

        assert len(result) == len(tokens), (
            f"classify() output length {len(result)} != input length {len(tokens)}"
        )

        self._log_label_summary(result)
        return result

    def _all_fallback(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for tok in tokens:
            out                = dict(tok)
            out["label"]       = FALLBACK_LABEL
            out["norm"]        = None
            out["gemma_entity"] = None
            out["gemma_score"]  = None
            result.append(out)
        return result

    def _log_label_summary(self, labelled: List[Dict[str, Any]]) -> None:
        from collections import Counter
        counts = Counter(t.get("label", "UNKNOWN") for t in labelled)
        logger.info("Gemma label distribution: %s", dict(counts))