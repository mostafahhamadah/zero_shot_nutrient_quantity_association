"""
gliner_classifier.py
====================
Stage 3 — GLiNER Semantic Classifier (Experiment 3)

PURPOSE
-------
Replace the rule-based SemanticClassifier (experiment_01_final_semantic_classifier.py)
with a zero-shot span-extraction approach using GLiNER biomed bi-base.

For each image's corrected OCR tokens (Stage 2 output), this module:
  1. Serializes tokens into a GLiNER-ready text string via text_serializer.py
  2. Runs Ihor/gliner-biomed-bi-base-v1.0 with descriptive Set B entity labels
  3. Applies a 0.3 confidence threshold (permissive — maximise recall)
  4. Maps predicted spans back to tokens using tag-all strategy
  5. Maps GLiNER label strings to internal pipeline labels via LABEL_MAP
  6. Assigns UNKNOWN to tokens not covered by any accepted span
  7. Normalises CONTEXT tokens to canonical form via CONTEXT_MAP

DESIGN DECISIONS (locked 2025-04-06)
--------------------------------------
Model            : Ihor/gliner-biomed-bi-base-v1.0
                   Bi-encoder: DeBERTa-v3-base (text) + BAAI/bge-small-en-v1.5 (labels).
                   Chosen for: best zero-shot recall among bi-encoders (58.31% micro F1),
                   stable label embeddings under German OCR input, 3.5x faster than bi-large.
                   Reference: Yazdani, Stepanov & Teodoro (2025) arXiv:2504.00676

Entity labels    : Set B — descriptive natural-language strings.
                   Chosen for: richer embedding signal than minimal Set A; closer to
                   biomedical pre-training vocabulary than Set C.

Confidence       : 0.3 — permissive threshold to maximise recall.
                   False positives from low-confidence spans are tolerated at this stage;
                   the graph association step (Stage 5) provides a second filter.

Span assignment  : Tag-all — every token fully contained in a GLiNER span receives
                   that span's label. Zero changes to Stage 4 graph_constructor.py.
                   Containment condition: span.start <= token.start_char AND
                                          token.end_char  <= span.end

Fallback label   : UNKNOWN — tokens not covered by any accepted span.
                   UNKNOWN tokens are passed to the rule-based fallback chain in
                   run_experiment.py for QUANTITY/UNIT/CONTEXT secondary classification.
                   (Not NOISE — NOISE suppresses downstream association entirely.)

Context handling : Option C — GLiNER handles everything including context labels.
                   GLiNER assigns the CONTEXT label; CONTEXT_MAP then normalises
                   the token text to canonical form (per_100g / per_serving /
                   per_daily_dose). German context headers risk low recall here —
                   this is documented as a known failure mode for Experiment 3.

split_fused_token: Runs UPSTREAM in PaddleOCR Stage 2 corrector.
                   This module always receives already-split tokens.

PIPELINE POSITION
-----------------
Stage 2 (PaddleOCR corrector, split_fused_token already applied)
    → text_serializer.serialize_tokens_for_gliner()
    → [this module] gliner_classifier.py  →  labelled token list
    → graph_constructor.py               (Stage 4, unchanged)

OUTPUT CONTRACT (matches Stage 4 graph_constructor.py expectations)
--------------------------------------------------------------------
Returns a list of token dicts. Each dict is the original Stage 2 token dict
with the following fields added:

    "label"            : str   — NUTRIENT | QUANTITY | UNIT | CONTEXT | UNKNOWN
    "norm"             : str | None
                         For CONTEXT tokens: canonical form from CONTEXT_MAP
                         (per_100g | per_serving | per_daily_dose | None)
                         For all other labels: None
    "gliner_score"     : float | None
                         Confidence score of the GLiNER span that covered this token.
                         None for UNKNOWN tokens (no span matched).
    "gliner_span_text" : str | None
                         Full text of the GLiNER span that covered this token.
                         Useful for diagnostics — distinguishes multi-token spans.
                         None for UNKNOWN tokens.

USAGE
-----
    from gliner_classifier import GLiNERClassifier

    classifier = GLiNERClassifier()                 # loads model once
    labelled   = classifier.classify(tokens)        # tokens = Stage 2 output

    # Or use the module-level convenience function:
    labelled   = classify_tokens(tokens)            # creates a default instance

THESIS CITATION
---------------
    Yazdani, A., Stepanov, I., and Teodoro, D. (2025).
    GLiNER-biomed: A Suite of Efficient Models for Open Biomedical Named Entity
    Recognition. arXiv:2504.00676. doi:10.48550/arXiv.2504.00676
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from gliner import GLiNER

from src.utils.text_serializer import serialize_tokens_for_gliner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all locked decisions live here, nowhere else
# ---------------------------------------------------------------------------

#: Hugging Face model ID for the bi-encoder base variant.
MODEL_ID: str = "Ihor/gliner-biomed-bi-large-v1.0"

#: Set B — descriptive entity label strings passed to GLiNER at inference.
#: Used with Ihor/gliner-biomed-bi-large-v1.0 (bi-encoder large).
#: Label encoder: BAAI/bge-base-en-v1.5 (larger than bi-base's bge-small).
#: These are encoded once and cached at model load time.
GLINER_LABELS: List[str] = [
    "nutritional ingredient or vitamin or mineral",
    "numeric amount or dosage value",
    "measurement unit",
    "serving size header or reference amount label",
]

#: Explicit mapping from GLiNER label strings → internal pipeline labels.
#: Must contain one entry per string in GLINER_LABELS.
LABEL_MAP: Dict[str, str] = {
    "nutritional ingredient or vitamin or mineral":  "NUTRIENT",
    "numeric amount or dosage value":                "QUANTITY",
    "measurement unit":                              "UNIT",
    "serving size header or reference amount label": "CONTEXT",
}

#: Minimum GLiNER confidence score to accept a span.
#: 0.3 = permissive, maximise recall. False positives filtered by Stage 5.
CONFIDENCE_THRESHOLD: float = 0.3

#: Label assigned to tokens not covered by any accepted GLiNER span.
#: UNKNOWN passes tokens to rule-based fallback; NOISE would suppress them.
FALLBACK_LABEL: str = "UNKNOWN"

#: Canonical context normalisation map.
#: Keys are lowercased OCR token variants; values are canonical pipeline forms.
#: Preserved from experiment_01_final_semantic_classifier.py (Stage 3, Exp 1).
#: Applied AFTER GLiNER assigns CONTEXT label — GLiNER detects the span,
#: CONTEXT_MAP resolves the canonical meaning.
CONTEXT_MAP: Dict[str, str] = {
    # German — per daily dose
    "je tagesdosis":          "per_daily_dose",
    "pro tagesdosis":         "per_daily_dose",
    "tagesdosis":             "per_daily_dose",
    "tagesbedarf":            "per_daily_dose",
    "je 2 kapseln":           "per_daily_dose",
    "je 3 kapseln":           "per_daily_dose",
    "je 4 kapseln":           "per_daily_dose",
    # German — per 100g
    "je 100 g":               "per_100g",
    "pro 100 g":              "per_100g",
    "pro 100g":               "per_100g",
    "per 100g":               "per_100g",
    "per 100 g":              "per_100g",
    "je 100g":                "per_100g",
    "1009":                   "per_100g",    # OCR artefact
    "je 1009":                "per_100g",    # OCR artefact
    "pro 1009":               "per_100g",    # OCR artefact
    "100g":                   "per_100g",
    # German — per serving
    "pro portion":            "per_serving",
    "je portion":             "per_serving",
    "portion":                "per_serving",
    "pro106":                 "per_serving",  # OCR artefact
    "portion**":              "per_serving",  # OCR artefact
    # English — per daily dose
    "per daily dose":         "per_daily_dose",
    "daily dose":             "per_daily_dose",
    "daily value":            "per_daily_dose",
    "per day":                "per_daily_dose",
    # English — per serving
    "per serving":            "per_serving",
    "serving":                "per_serving",
    "per portion":            "per_serving",
    # English — per 100g
    "per 100 g":              "per_100g",
    "per 100g":               "per_100g",
    # French
    "par dose journalière":   "per_daily_dose",
    "par portion":            "per_serving",
    "pour 100 g":             "per_100g",
    # Dutch
    "per dagelijkse dosis":   "per_daily_dose",
    "per portie":             "per_serving",
    "per 100 g":              "per_100g",
}


# ---------------------------------------------------------------------------
# Helper: CONTEXT_MAP normalisation
# ---------------------------------------------------------------------------

def _normalise_context(token_text: str) -> Optional[str]:
    """
    Map a token text to its canonical context form using CONTEXT_MAP.

    Lookup is case-insensitive and strips leading/trailing whitespace.
    Returns None if no mapping found — GLiNER may have detected a CONTEXT span
    for an OCR variant not yet in CONTEXT_MAP.

    Parameters
    ----------
    token_text : raw token text from the OCR corrector.

    Returns
    -------
    Canonical string (per_100g | per_serving | per_daily_dose) or None.
    """
    return CONTEXT_MAP.get(token_text.strip().lower(), None)


# ---------------------------------------------------------------------------
# Core classifier class
# ---------------------------------------------------------------------------

class GLiNERClassifier:
    """
    Zero-shot semantic token classifier using GLiNER biomed bi-base.

    The model is loaded once on instantiation. All classify() calls reuse the
    same loaded model. For batch processing across multiple images in one
    experiment run, create a single GLiNERClassifier instance and call
    classify() per image.

    Parameters
    ----------
    model_id    : HF model ID. Default = MODEL_ID constant.
    labels      : Entity label strings. Default = GLINER_LABELS constant.
    label_map   : Mapping from GLiNER labels to internal labels.
                  Default = LABEL_MAP constant.
    threshold   : Confidence threshold for span acceptance.
                  Default = CONFIDENCE_THRESHOLD constant.
    flat_ner    : If True, use flat (non-overlapping) NER mode.
                  If False, allow nested spans. Default True for supplement labels
                  where nested nutrient spans are rare and flat is more predictable.
    """

    def __init__(
        self,
        model_id:  str             = MODEL_ID,
        labels:    List[str]       = None,
        label_map: Dict[str, str]  = None,
        threshold: float           = CONFIDENCE_THRESHOLD,
        flat_ner:  bool            = True,
    ) -> None:
        self.model_id  = model_id
        self.labels    = labels    if labels    is not None else GLINER_LABELS
        self.label_map = label_map if label_map is not None else LABEL_MAP
        self.threshold = threshold
        self.flat_ner  = flat_ner
        import torch

        logger.info("Loading GLiNER model: %s", self.model_id)
        self._model = GLiNER.from_pretrained(self.model_id)

        # Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self._model = self._model.to(device)
            logger.info(f"✓ GLiNER running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠ GLiNER running on CPU (no GPU available)")

        self._model.eval()
        logger.info("GLiNER model loaded. Label encoder caches %d label embeddings.",
                    len(self.labels))

    # ------------------------------------------------------------------
    # Private: span-to-token assignment (tag-all)
    # ------------------------------------------------------------------

    def _assign_spans_to_tokens(
        self,
        predicted_spans: List[Dict[str, Any]],
        token_spans:     List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Map GLiNER predicted spans to token indices using tag-all strategy.

        Containment condition (half-open intervals):
            span["start"] <= token["start_char"]
            AND token["end_char"] <= span["end"]

        When a token is covered by multiple overlapping spans (possible at
        threshold=0.3), the span with the HIGHER confidence score wins.

        Parameters
        ----------
        predicted_spans : output of model.predict_entities(), filtered by threshold.
                          Each dict has keys: start, end, text, label, score.
        token_spans     : token_spans list from serialize_tokens_for_gliner().

        Returns
        -------
        Dict mapping token_index → winning span dict.
        Only tokens covered by at least one accepted span appear in the dict.
        """
        assignments: Dict[int, Dict[str, Any]] = {}

        for span in predicted_spans:
            s_start = span["start"]
            s_end   = span["end"]
            s_score = span["score"]

            for tok in token_spans:
                t_idx        = tok["token_index"]
                t_start_char = tok["start_char"]
                t_end_char   = tok["end_char"]

                # Tag-all containment: token fully inside span
                if s_start <= t_start_char and t_end_char <= s_end:
                    existing = assignments.get(t_idx)
                    if existing is None or s_score > existing["score"]:
                        assignments[t_idx] = span

        return assignments

    # ------------------------------------------------------------------
    # Private: map GLiNER label string to internal pipeline label
    # ------------------------------------------------------------------

    def _map_label(self, gliner_label: str) -> str:
        """
        Convert a GLiNER entity label string to an internal pipeline label.

        Uses LABEL_MAP (explicit dict). If the string is not in the map
        (should not happen with fixed GLINER_LABELS), falls back to FALLBACK_LABEL
        with a warning.

        Parameters
        ----------
        gliner_label : the label string returned by GLiNER (must match a key in LABEL_MAP).

        Returns
        -------
        One of: NUTRIENT | QUANTITY | UNIT | CONTEXT | UNKNOWN
        """
        mapped = self.label_map.get(gliner_label)
        if mapped is None:
            logger.warning(
                "GLiNER returned unexpected label '%s' — assigning %s.",
                gliner_label, FALLBACK_LABEL,
            )
            return FALLBACK_LABEL
        return mapped

    # ------------------------------------------------------------------
    # Public: classify
    # ------------------------------------------------------------------

    def classify(
        self,
        tokens: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Classify a list of corrected OCR tokens using GLiNER biomed bi-base.

        Steps
        -----
        1. Serialize tokens → text + token_spans via text_serializer.
        2. Run GLiNER prediction on the serialized text with GLINER_LABELS.
        3. Filter predicted spans by CONFIDENCE_THRESHOLD.
        4. Assign spans to tokens via tag-all containment.
        5. Map GLiNER label strings to internal labels via LABEL_MAP.
        6. Assign FALLBACK_LABEL to tokens not covered by any span.
        7. Normalise CONTEXT tokens via CONTEXT_MAP (add "norm" field).
        8. Return augmented token list (original fields + label/norm/diagnostics).

        Parameters
        ----------
        tokens : list of token dicts from Stage 2 corrector.
                 Each dict must contain: token, x1, y1, x2, y2, cx, cy, conf.
                 split_fused_token() has already run upstream.

        Returns
        -------
        List of token dicts. Each dict is the original Stage 2 dict with
        the following fields added:
            "label"            : str
            "norm"             : str | None
            "gliner_score"     : float | None
            "gliner_span_text" : str | None

        Notes
        -----
        - Tokens with empty text after stripping (NOISE candidates) receive
          FALLBACK_LABEL since the serializer excludes them from the text and
          they therefore have no token_span entry.
        - The returned list preserves the original token order and length —
          one output dict per input dict, index-aligned.
        """
        if not tokens:
            return []

        # ---- Step 1: Serialise ----
        serialized    = serialize_tokens_for_gliner(tokens)
        text          = serialized["text"]
        token_spans   = serialized["token_spans"]   # [{token_index, start_char, end_char, ...}]

        if not text.strip():
            logger.warning("Serializer produced empty text — all tokens labelled %s.",
                           FALLBACK_LABEL)
            return self._all_fallback(tokens)

        # ---- Step 2: GLiNER prediction ----
        logger.debug("Running GLiNER on %d chars, %d token spans.",
                     len(text), len(token_spans))
        raw_predictions: List[Dict[str, Any]] = self._model.predict_entities(
            text,
            self.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
        )

        # ---- Step 3: Filter by threshold ----
        # predict_entities already filters by threshold internally in most GLiNER
        # versions, but we apply it explicitly for safety and logging.
        accepted = [s for s in raw_predictions if s["score"] >= self.threshold]

        logger.debug(
            "GLiNER raw predictions: %d | accepted (≥%.2f): %d",
            len(raw_predictions), self.threshold, len(accepted),
        )

        # ---- Step 4: Span-to-token assignment (tag-all) ----
        assignments = self._assign_spans_to_tokens(accepted, token_spans)

        # Build a fast lookup: token_index → token_span entry
        # (for tokens that appear in the serialized text)
        serialized_indices = {ts["token_index"] for ts in token_spans}

        # ---- Steps 5-7: Build output list ----
        result: List[Dict[str, Any]] = []

        for orig_idx, tok in enumerate(tokens):
            out = dict(tok)  # shallow copy — preserve all original fields

            if orig_idx not in serialized_indices:
                # Token was filtered out by serializer (empty text / missing coords)
                # Treat as NOISE-equivalent: assign FALLBACK but mark as not serialized
                out["label"]            = FALLBACK_LABEL
                out["norm"]             = None
                out["gliner_score"]     = None
                out["gliner_span_text"] = None
                result.append(out)
                continue

            winning_span = assignments.get(orig_idx)

            if winning_span is None:
                # No GLiNER span covered this token → fallback
                out["label"]            = FALLBACK_LABEL
                out["norm"]             = None
                out["gliner_score"]     = None
                out["gliner_span_text"] = None

            else:
                internal_label = self._map_label(winning_span["label"])
                out["label"]            = internal_label
                out["gliner_score"]     = winning_span["score"]
                out["gliner_span_text"] = winning_span["text"]

                # Step 7: CONTEXT normalisation
                if internal_label == "CONTEXT":
                    out["norm"] = _normalise_context(str(tok.get("token", "")))
                    if out["norm"] is None:
                        logger.debug(
                            "CONTEXT token '%s' not in CONTEXT_MAP — norm=None.",
                            tok.get("token", ""),
                        )
                else:
                    out["norm"] = None

            result.append(out)

        # Sanity check: output length must equal input length
        assert len(result) == len(tokens), (
            f"classify() output length {len(result)} != input length {len(tokens)}"
        )

        self._log_label_summary(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_fallback(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return all tokens labelled FALLBACK_LABEL — used on empty text."""
        result = []
        for tok in tokens:
            out = dict(tok)
            out["label"]            = FALLBACK_LABEL
            out["norm"]             = None
            out["gliner_score"]     = None
            out["gliner_span_text"] = None
            result.append(out)
        return result

    def _log_label_summary(self, labelled: List[Dict[str, Any]]) -> None:
        """Log per-label token counts at DEBUG level for pipeline diagnostics."""
        counts: Dict[str, int] = {}
        for tok in labelled:
            lbl = tok.get("label", "UNKNOWN")
            counts[lbl] = counts.get(lbl, 0) + 1
        logger.debug("GLiNER label distribution: %s", counts)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_classifier: Optional[GLiNERClassifier] = None


def classify_tokens(
    tokens:    List[Dict[str, Any]],
    model_id:  str            = MODEL_ID,
    threshold: float          = CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Module-level convenience function for single-call classification.

    Creates a default GLiNERClassifier instance on first call and reuses it
    for subsequent calls within the same Python process. This avoids reloading
    the model for each image in run_experiment.py.

    Parameters
    ----------
    tokens    : Stage 2 corrector output (list of token dicts).
    model_id  : HF model ID. Override to test other variants.
    threshold : Confidence threshold override.

    Returns
    -------
    Labelled token list — same contract as GLiNERClassifier.classify().
    """
    global _default_classifier

    if _default_classifier is None:
        _default_classifier = GLiNERClassifier(
            model_id=model_id,
            threshold=threshold,
        )

    return _default_classifier.classify(tokens)