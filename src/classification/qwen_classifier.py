"""
qwen_classifier.py
==================
Stage 3 — Qwen 2.5:7b Semantic Classifier (Experiment 4)

PURPOSE
-------
Replace the GLiNER span extractor (Experiment 3) with a general-purpose LLM
for zero-shot semantic classification of OCR tokens from supplement labels.

Qwen 2.5:7b addresses the two structural failures of GLiNER biomed:
  1. German context headers (Je Tagesdosis, Pro 100g) — Qwen is multilingual
  2. Isolated unit tokens (g, mg, µg) — Qwen sees full row context

DESIGN DECISIONS (locked 2025-04-06)
--------------------------------------
Strategy        : Option B — full serialized text per image sent to Qwen.
                  57 Ollama calls total (one per image). ~5-10 min runtime.

Coverage        : Full replacement — Qwen classifies all 5 labels.
                  NUTRIENT, QUANTITY, UNIT, CONTEXT, NOISE (5-label scheme).

JSON schema     : Schema A — entity text + label only.
                  {"entities": [{"text": "...", "label": "..."}]}
                  No character offsets — LLMs hallucinate positions.

Remapping       : Fuzzy text match between Qwen entity text and token list.
                  Single-word entities: SequenceMatcher per token.
                  Multi-word entities: n-gram window over consecutive token spans.
                  Threshold: 0.65 (permissive — catches OCR variants).

Serving size    : Labelled CONTEXT directly by Qwen.
                  Prompt instructs: "2 Kapseln", "70g" etc. → CONTEXT.
                  Applied only when adjacent context is per_serving/per_daily_dose.
                  Post-processing: CONTEXT tokens without CONTEXT_MAP match
                  near a per_100g header → reassigned NOISE.

Fallback label  : UNKNOWN — tokens not matched by any Qwen entity are passed
                  to the rule-based fallback chain in run_experiment.

CONTEXT handling: CONTEXT_MAP normalises CONTEXT token text to canonical form
                  (per_100g / per_serving / per_daily_dose) after Qwen assigns
                  the CONTEXT label. Same map as experiment_01 classifier.

PIPELINE POSITION
-----------------
Stage 2 (PaddleOCR corrector, split_fused_token applied)
    → text_serializer.serialize_tokens_for_gliner()   (reused as text source)
    → [this module] qwen_classifier.py  →  labelled token list
    → graph_constructor.py              (Stage 4, unchanged)

OUTPUT CONTRACT
---------------
Returns list of token dicts. Each dict is the original Stage 2 token dict with:
    "label"         : str    NUTRIENT|QUANTITY|UNIT|CONTEXT|UNKNOWN
    "norm"          : str|None  canonical context form for CONTEXT tokens
    "qwen_entity"   : str|None  Qwen entity text that matched this token
    "qwen_score"    : float|None  SequenceMatcher ratio of the match

USAGE
-----
    from qwen_classifier import QwenClassifier
    classifier = QwenClassifier()
    labelled   = classifier.classify(tokens)

THESIS CITATION
---------------
    Qwen Team (2025). Qwen2.5 Technical Report.
    Model: qwen2.5:7b via Ollama local inference.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils.text_serializer import serialize_tokens_for_gliner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ollama API endpoint — must be running locally
OLLAMA_URL: str = "http://localhost:11434/api/chat"

#: Qwen model name as registered in Ollama
MODEL_ID: str = "qwen2.5:7b"

#: Temperature — 0.0 for fully deterministic output
TEMPERATURE: float = 0.0

#: Fuzzy match threshold for entity text → token remapping
#: 0.65 = permissive, catches OCR variants (Magnesiumcitraat vs Magnesiumcitrat)
FUZZY_THRESHOLD: float = 0.65

#: Label for tokens not matched by any Qwen entity
FALLBACK_LABEL: str = "UNKNOWN"

#: Valid labels Qwen may return — any other value is rejected
VALID_LABELS: frozenset = frozenset({"NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE"})

#: Ollama request timeout in seconds
REQUEST_TIMEOUT: int = 300

#: CONTEXT_MAP — normalises CONTEXT token text to canonical pipeline form.
#: Preserved from experiment_01_final_semantic_classifier.py.
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
    """
    Build the user-turn prompt containing the serialized OCR text.

    Parameters
    ----------
    serialized_text : output of text_serializer — tokens joined by spaces
                      within lines, lines separated by newlines.

    Returns
    -------
    User-turn prompt string.
    """
    return (
        "Extract all nutritional entities from this supplement label OCR text. "
        "Return only the JSON object.\n\n"
        f"OCR TEXT:\n{serialized_text}"
    )


# ---------------------------------------------------------------------------
# Ollama API caller
# ---------------------------------------------------------------------------

def _call_qwen(
    serialized_text: str,
    model: str = MODEL_ID,
    temperature: float = TEMPERATURE,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[str]:
    """
    Call Qwen 2.5:7b via Ollama chat endpoint.

    Uses format="json" to enforce JSON output mode in Ollama.
    Temperature=0.0 for deterministic results across runs.

    Parameters
    ----------
    serialized_text : the full serialized OCR text for one image.
    model           : Ollama model name.
    temperature     : sampling temperature.
    timeout         : HTTP request timeout in seconds.

    Returns
    -------
    Raw response string from Qwen, or None on connection error.
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
        logger.error("Qwen API call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> List[Dict[str, str]]:
    """
    Parse Qwen's JSON response into a list of entity dicts.

    Validates that each entity has "text" and "label" fields and that
    the label is in VALID_LABELS. Invalid entities are dropped with a warning.

    Parameters
    ----------
    raw : raw string from Qwen (should be valid JSON).

    Returns
    -------
    List of {"text": str, "label": str} dicts. Empty list on parse failure.
    """
    # Strip markdown code fences if Qwen added them despite format=json
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.warning("Qwen JSON parse error: %s | raw: %s", exc, raw[:200])
        return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        logger.warning("Qwen response missing 'entities' list.")
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
            logger.debug("Qwen returned unknown label '%s' for '%s' — skipped.", label, text)
            continue
        if label == "NOISE":
            continue  # NOISE entities are not needed — unmatched tokens → UNKNOWN

        valid.append({"text": text, "label": label})

    logger.debug("Qwen entities parsed: %d valid from %d raw.", len(valid), len(entities))
    return valid


# ---------------------------------------------------------------------------
# Fuzzy token remapper
# ---------------------------------------------------------------------------

def _seq_ratio(a: str, b: str) -> float:
    """Case-insensitive SequenceMatcher ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_tokens_for_entity(
    entity_text:  str,
    token_spans:  List[Dict[str, Any]],
    threshold:    float = FUZZY_THRESHOLD,
) -> List[int]:
    """
    Find the token index/indices in token_spans that best match entity_text.

    Algorithm
    ---------
    Single-word entities (no spaces in entity_text):
        Fuzzy match entity_text against each token's text.
        Return the token_index with the highest ratio >= threshold.

    Multi-word entities (contains spaces):
        Split entity_text into N words.
        Slide an N-token window over token_spans (sorted by start_char).
        Compute ratio between entity_text and concatenated window text.
        Return all N token indices of the best window >= threshold.

    Parameters
    ----------
    entity_text  : entity text string from Qwen (exact text it returned).
    token_spans  : token_spans list from serialize_tokens_for_gliner().
    threshold    : minimum SequenceMatcher ratio to accept a match.

    Returns
    -------
    List of token_index values (may be empty if no match above threshold).
    """
    if not entity_text or not token_spans:
        return []

    words = entity_text.split()

    # ---- Single-word entity ----
    if len(words) == 1:
        best_ratio = 0.0
        best_idx   = -1
        for ts in token_spans:
            r = _seq_ratio(entity_text, ts["token_text"])
            if r > best_ratio:
                best_ratio = r
                best_idx   = ts["token_index"]
        if best_ratio >= threshold:
            return [best_idx]
        return []

    # ---- Multi-word entity ----
    n = len(words)
    # Sort by character position (left-to-right, top-to-bottom)
    sorted_spans = sorted(token_spans, key=lambda x: (x["line_id"], x["start_char"]))
    m = len(sorted_spans)

    if n > m:
        # More words than tokens — impossible to match
        return []

    best_ratio   = 0.0
    best_indices: List[int] = []

    for i in range(m - n + 1):
        window      = sorted_spans[i: i + n]
        window_text = " ".join(w["token_text"] for w in window)
        r           = _seq_ratio(entity_text, window_text)
        if r > best_ratio:
            best_ratio   = r
            best_indices = [w["token_index"] for w in window]

    if best_ratio >= threshold:
        return best_indices
    return []


def _map_entities_to_tokens(
    entities:     List[Dict[str, str]],
    token_spans:  List[Dict[str, Any]],
    threshold:    float = FUZZY_THRESHOLD,
) -> Dict[int, Dict[str, str]]:
    """
    Map Qwen entity list to token indices.

    When multiple Qwen entities match the same token, the entity with the
    higher SequenceMatcher ratio wins (last-write-wins for equal ratio).

    Parameters
    ----------
    entities    : parsed Qwen entities [{"text": str, "label": str}].
    token_spans : from serialize_tokens_for_gliner().
    threshold   : fuzzy match threshold.

    Returns
    -------
    Dict mapping token_index → {"label": str, "entity_text": str, "score": float}
    """
    assignments: Dict[int, Dict[str, str]] = {}

    for ent in entities:
        entity_text = ent["text"]
        label       = ent["label"]

        # Find matching token indices
        matched_indices = _find_tokens_for_entity(entity_text, token_spans, threshold)
        if not matched_indices:
            logger.debug("No token match for Qwen entity '%s' (label=%s).", entity_text, label)
            continue

        # Compute match score (use ratio of full entity text against first matched token
        # as a representative score for diagnostics)
        first_ts    = next((ts for ts in token_spans if ts["token_index"] == matched_indices[0]), None)
        match_score = (
            _seq_ratio(entity_text, first_ts["token_text"])
            if first_ts else 1.0
        )

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
    """
    Map CONTEXT token text to canonical form via CONTEXT_MAP.
    Returns None if not in map (serving size descriptor or unknown variant).
    """
    return CONTEXT_MAP.get(token_text.strip().lower(), None)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class QwenClassifier:
    """
    Zero-shot semantic token classifier using Qwen 2.5:7b via Ollama.

    Sends the full serialized OCR text for each image to Qwen in one call.
    Maps Qwen's entity predictions back to token indices via fuzzy matching.

    Parameters
    ----------
    model_id   : Ollama model name (default: qwen2.5:7b).
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
        logger.info("QwenClassifier initialised: model=%s, threshold=%.2f",
                    model_id, threshold)

    def classify(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a list of corrected OCR tokens using Qwen 2.5:7b.

        Steps
        -----
        1. Serialize tokens → text + token_spans via text_serializer.
        2. Call Qwen via Ollama with the serialized text.
        3. Parse JSON response → entity list.
        4. Fuzzy-map entity texts → token indices.
        5. Assign FALLBACK_LABEL to unmatched tokens.
        6. Normalise CONTEXT tokens via CONTEXT_MAP.
        7. Return augmented token list.

        Parameters
        ----------
        tokens : token list from Stage 2 corrector.

        Returns
        -------
        List of token dicts with added fields:
            "label"       : str
            "norm"        : str | None   (canonical context form)
            "qwen_entity" : str | None   (matched Qwen entity text)
            "qwen_score"  : float | None (SequenceMatcher match score)
        """
        if not tokens:
            return []

        # ---- Step 1: Serialise ----
        serialized   = serialize_tokens_for_gliner(tokens)
        text         = serialized["text"]
        token_spans  = serialized["token_spans"]

        if not text.strip():
            logger.warning("Empty serialized text — all tokens labelled %s.", FALLBACK_LABEL)
            return self._all_fallback(tokens)

        # ---- Step 2: Call Qwen ----
        raw = _call_qwen(
            text,
            model=self.model_id,
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT,
        )

        if raw is None:
            logger.warning("Qwen call failed — all tokens labelled %s.", FALLBACK_LABEL)
            return self._all_fallback(tokens)

        # ---- Step 3: Parse response ----
        entities = _parse_response(raw)
        logger.debug("Qwen returned %d valid entities for %d tokens.",
                     len(entities), len(tokens))

        # ---- Step 4: Map entities → token indices ----
        assignments = _map_entities_to_tokens(entities, token_spans, self.threshold)

        # Build set of serialized token indices for fast lookup
        serialized_indices = {ts["token_index"] for ts in token_spans}

        # ---- Steps 5-6: Build output list ----
        result: List[Dict[str, Any]] = []

        for orig_idx, tok in enumerate(tokens):
            out = dict(tok)  # preserve all original fields

            if orig_idx not in serialized_indices:
                # Token filtered by serializer (empty text / missing coords)
                out["label"]       = FALLBACK_LABEL
                out["norm"]        = None
                out["qwen_entity"] = None
                out["qwen_score"]  = None
                result.append(out)
                continue

            match = assignments.get(orig_idx)

            if match is None:
                out["label"]       = FALLBACK_LABEL
                out["norm"]        = None
                out["qwen_entity"] = None
                out["qwen_score"]  = None
            else:
                out["label"]       = match["label"]
                out["qwen_entity"] = match["entity_text"]
                out["qwen_score"]  = match["score"]

                if match["label"] == "CONTEXT":
                    out["norm"] = _normalise_context(str(tok.get("token", "")))
                    if out["norm"] is None:
                        # Serving size or unknown context variant — keep CONTEXT,
                        # norm=None. Stage 5 _get_context_for_qty handles None norm
                        # by looking at adjacent context nodes.
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
            out = dict(tok)
            out["label"]       = FALLBACK_LABEL
            out["norm"]        = None
            out["qwen_entity"] = None
            out["qwen_score"]  = None
            result.append(out)
        return result

    def _log_label_summary(self, labelled: List[Dict[str, Any]]) -> None:
        from collections import Counter
        counts = Counter(t.get("label", "UNKNOWN") for t in labelled)
        logger.info("Qwen label distribution: %s", dict(counts))