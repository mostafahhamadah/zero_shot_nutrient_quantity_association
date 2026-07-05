"""
gliner_classifier.py
====================
Stage 3 — GLiNER-biomed Semantic Classifier  (Ablation 2, third variant)
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

WHAT THIS FILE REALISES (Methodology §4.3, esp. §4.3.6)
-------------------------------------------------------
The semantic classifier factors into two parts:

  (1) a *shared deterministic cascade* that resolves the syntactic roles
      UNIT, NRV/NOISE, CONTEXT and QUANTITY, and
  (2) a single *nutrient / not-nutrient node* that makes the one hard,
      open-vocabulary decision.

The cascade is identical across all three classifier variants and is
supplied by the rule-based SemanticClassifier
(experiment_01_final_semantic_classifier.py).  The variants differ ONLY in
how they realise node (2):

  • rule-based variant      → curated nutrient lexicon
  • embedding variant       → BGE-M3 cosine similarity      (embedding_semantic_classifier.py)
  • GLiNER variant (here)   → GLiNER-biomed span acceptance

This file is therefore the structural sibling of
EmbeddingSemanticClassifier: it wraps the SAME cascade, runs it first,
and only re-decides the tokens the cascade leaves as nutrient
candidates.  The cascade remains authoritative for QUANTITY / UNIT /
CONTEXT / NOISE — GLiNER never overrides them, and CONTEXT canonicalisation
(per_100g / per_serving / per_daily_dose) is done by the cascade, not here.

This is a deliberate departure from the earlier Set-B "tag-all" design,
which let GLiNER assign all four roles under four labels.  That design did
not match the methodology; this one does.

THE NUTRIENT NODE (Methodology §4.3.6)
--------------------------------------
Model       : Ihor/gliner-biomed-bi-large-v1.0  (bi-encoder; DeBERTa-v3
              text encoder + separate sentence encoder for the labels, so
              arbitrary natural-language labels are supplied at inference
              without retraining).  Used zero-shot, no fine-tuning.
Label       : a SINGLE descriptive label,
              "nutritional ingredient or vitamin or mineral".
Decision    : a nutrient candidate is labelled NUTRIENT when it falls
              within an accepted entity span (span score >= threshold),
              and UNKNOWN otherwise.
Threshold   : the acceptance threshold is a CALIBRATED operating point,
              fixed by a token-level threshold sweep over the nutrient
              candidates (see gliner_threshold_sweep.py and Methodology
              §4.3.4 for the analogous embedding-node calibration).  It is
              exposed here as `threshold`; NUTRIENT_THRESHOLD below is the
              operating point the sweep selects.

HOW A CANDIDATE MAPS TO "WITHIN A SPAN"
---------------------------------------
GLiNER needs surrounding text to recognise an entity, so the whole label
is serialised once (text_serializer.serialize_tokens_for_gliner) and GLiNER
is run a single time on that text with the single label.  Every token keeps
its half-open character span [start_char, end_char) in that text.  A
candidate token is "within an accepted span" when some accepted nutrient
span fully contains it:

        span.start <= token.start_char  AND  token.end_char <= span.end

The score attached to the candidate is the maximum score over the spans
that contain it (0.0 / None when no span does).  Only candidate tokens are
consulted; tokens the cascade already resolved keep their label regardless
of what spans cover them.

MODES (mirroring the embedding node)
------------------------------------
mode="gliner_only" (default, the variant reported in the thesis)
    The lexicon plays no part: every candidate the cascade leaves as
    NUTRIENT *or* UNKNOWN is decided by GLiNER.  This is the genuinely
    zero-shot realisation of the node.
mode="hybrid"
    The lexicon's positive (NUTRIENT) calls are kept and GLiNER is used
    only to rescue tokens the lexicon left UNKNOWN.

OUTPUT CONTRACT  (drop-in for SemanticClassifier / EmbeddingSemanticClassifier)
------------------------------------------------------------------------------
classify_all(tokens) returns a list of token dicts, one per input token,
index-aligned, each being the cascade's output dict with these fields set:

    "label"               : NUTRIENT | QUANTITY | UNIT | CONTEXT | NOISE | UNKNOWN
    "norm"                : canonical form set by the cascade (unchanged here)
    "gliner_score"        : float | None   max covering nutrient-span score
                                            (None for non-candidates / no span)
    "gliner_span_text"    : str | None     text of that covering span
    "classification_method": "rule" | "gliner"

THESIS CITATION
---------------
    Yazdani, A., Stepanov, I., and Teodoro, D. (2025).
    GLiNER-biomed: A Suite of Efficient Models for Open Biomedical Named
    Entity Recognition. arXiv:2504.00676.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

# Shared deterministic cascade (same object the embedding node wraps).
from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier
# Reused serializer: builds the GLiNER input text + per-token char spans.
from src.utils.text_serializer import serialize_tokens_for_gliner

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Locked design constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#: Hugging Face model id — bi-encoder large variant (Methodology §4.3.6).
MODEL_ID: str = "Ihor/gliner-biomed-bi-large-v1.0"

#: The SINGLE descriptive label presented to GLiNER for the nutrient node.
NUTRIENT_LABEL: str = "nutritional ingredient or vitamin or mineral"

#: Calibrated acceptance threshold — the operating point chosen by the
#: token-level threshold sweep (gliner_threshold_sweep.py).  The value below
#: is the permissive pre-sweep default from the methodology text; replace it
#: with the swept optimum once the sweep has run.
NUTRIENT_THRESHOLD: float = 0.18

#: Internal floor passed to GLiNER so the SAME code path can both classify
#: (apply `threshold`) and feed the sweep (return raw scores >= floor).
#: Must be <= the smallest threshold the sweep will ever test.
SCORE_FLOOR: float = 0.001

#: Roles the cascade owns outright — GLiNER never touches these.
RULE_AUTHORITATIVE_LABELS: frozenset = frozenset({"QUANTITY", "UNIT", "CONTEXT", "NOISE"})

#: Roles that make a token a *nutrient candidate* eligible for the GLiNER node.
CANDIDATE_LABELS: frozenset = frozenset({"NUTRIENT", "UNKNOWN"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure helpers (no model required — unit-testable in isolation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _decide_candidate(gliner_score: Optional[float], threshold: float) -> str:
    """Nutrient-node decision for a single candidate.

    NUTRIENT iff the candidate falls within an accepted span, i.e. it has a
    covering nutrient-span score that meets the acceptance threshold.
    """
    if gliner_score is None:
        return "UNKNOWN"
    return "NUTRIENT" if gliner_score >= threshold else "UNKNOWN"


def _assign_max_span_score(
    spans: List[Dict[str, Any]],
    token_spans: List[Dict[str, Any]],
) -> Dict[int, Tuple[float, str]]:
    """Map each token index to (max covering span score, that span's text).

    Containment (half-open intervals): a span covers a token when
        span["start"] <= token["start_char"]  and  token["end_char"] <= span["end"].
    Only tokens covered by at least one span appear in the result; when
    several spans cover a token the highest score wins.
    """
    best: Dict[int, Tuple[float, str]] = {}
    for span in spans:
        s_start = span["start"]
        s_end = span["end"]
        s_score = float(span["score"])
        s_text = span.get("text", "")
        for tok in token_spans:
            if s_start <= tok["start_char"] and tok["end_char"] <= s_end:
                t_idx = tok["token_index"]
                prev = best.get(t_idx)
                if prev is None or s_score > prev[0]:
                    best[t_idx] = (s_score, s_text)
    return best


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLiNERSemanticClassifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GLiNERSemanticClassifier:
    """Cascade + GLiNER-biomed nutrient node (Methodology §4.3.6).

    Parameters
    ----------
    mode : {"gliner_only", "hybrid"}
        gliner_only — GLiNER decides every nutrient candidate (lexicon ignored).
        hybrid      — lexicon NUTRIENT calls kept; GLiNER rescues UNKNOWN only.
    model_id : str
        Hugging Face model id (default: bi-large variant).
    threshold : float
        Span-acceptance threshold (the swept operating point).
    confidence_threshold : float
        Passed to the underlying rule cascade (its low-confidence -> NOISE knob).
    flat_ner : bool
        Flat (non-overlapping) NER. True is appropriate for label panels.
    device : str or None
        'cpu', 'cuda', or None to auto-detect.
    """

    VALID_MODES = {"gliner_only", "hybrid"}

    def __init__(
        self,
        mode: str = "gliner_only",
        model_id: str = MODEL_ID,
        threshold: float = NUTRIENT_THRESHOLD,
        confidence_threshold: float = 0.3,
        flat_ner: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.model_id = model_id
        self.threshold = threshold
        self.flat_ner = flat_ner
        self.labels = [NUTRIENT_LABEL]

        # Shared cascade — identical object the other two variants use.
        self.rule_classifier = SemanticClassifier(confidence_threshold=confidence_threshold)

        # Lazy heavy imports so this module is importable (and the sweep's
        # type hints usable) without the model installed.
        from gliner import GLiNER
        import torch

        print(f"[GLiNERClassifier] mode={mode} | Loading model: {model_id}")
        self._model = GLiNER.from_pretrained(model_id)

        if device is not None:
            self._model = self._model.to(torch.device(device))
            print(f"[GLiNERClassifier] device={device}")
        elif torch.cuda.is_available():
            self._model = self._model.to(torch.device("cuda"))
            print(f"[GLiNERClassifier] device=cuda ({torch.cuda.get_device_name(0)})")
        else:
            print("[GLiNERClassifier] device=cpu (no GPU available)")

        self._model.eval()
        print(f"[GLiNERClassifier] Ready — single label: '{NUTRIENT_LABEL}'")

    # ── candidate membership ──────────────────────────────────────────

    def _is_candidate(self, rule_label: str) -> bool:
        """Is this cascade label a nutrient candidate eligible for GLiNER?"""
        if rule_label in RULE_AUTHORITATIVE_LABELS:
            return False
        if self.mode == "hybrid" and rule_label == "NUTRIENT":
            return False  # trust the lexicon's positive call
        return rule_label in CANDIDATE_LABELS

    # ── shared scoring core (serves both classify_all and the sweep) ──

    def _score_tokens(self, tokens: List[dict]) -> List[dict]:
        """Run the cascade, run GLiNER once, attach raw per-candidate scores.

        Returns the cascade's output dicts (index-aligned to `tokens`) with
        three extra fields on every dict:
            "is_candidate"     : bool
            "gliner_score"     : float | None   max covering nutrient-span score
            "gliner_span_text" : str | None
        The acceptance threshold is NOT applied here — that is left to the
        caller, which is what lets the threshold sweep reuse this method.
        """
        # 1) Shared deterministic cascade on every token.
        rule_results = [self.rule_classifier.classify_token(t) for t in tokens]

        # Initialise the diagnostic fields.
        for r in rule_results:
            r["is_candidate"] = self._is_candidate(r.get("label", "UNKNOWN"))
            r["gliner_score"] = None
            r["gliner_span_text"] = None

        # 2) Serialise once for GLiNER (full-label context preserved).
        serialized = serialize_tokens_for_gliner(tokens)
        text = serialized["text"]
        token_spans = serialized["token_spans"]

        if not text.strip() or not token_spans:
            logger.warning("Serializer produced empty text — no GLiNER scores.")
            return rule_results

        # 3) Single GLiNER pass at the floor, single label.
        spans: List[Dict[str, Any]] = self._model.predict_entities(
            text, self.labels, threshold=SCORE_FLOOR, flat_ner=self.flat_ner,
        )

        # 4) Max covering span score per token.
        scored = _assign_max_span_score(spans, token_spans)

        # 5) Attach scores to candidate tokens only.
        for orig_idx, r in enumerate(rule_results):
            if not r["is_candidate"]:
                continue
            hit = scored.get(orig_idx)
            if hit is not None:
                r["gliner_score"] = hit[0]
                r["gliner_span_text"] = hit[1]

        return rule_results

    # ── public API (drop-in replacement) ──────────────────────────────

    def classify_all(self, tokens: List[dict]) -> List[dict]:
        """Classify every token.

        Cascade owns QUANTITY/UNIT/CONTEXT/NOISE; GLiNER decides the
        nutrient candidates at the calibrated acceptance threshold.
        """
        if not tokens:
            return []

        scored = self._score_tokens(tokens)

        for r in scored:
            if r["is_candidate"]:
                r["label"] = _decide_candidate(r["gliner_score"], self.threshold)
                r["classification_method"] = "gliner"
            else:
                r["classification_method"] = "rule"
            r.pop("is_candidate", None)  # internal flag, not part of the contract

        return scored

    def classify(self, tokens: List[dict]) -> List[dict]:
        """Alias for classify_all (back-compat with the old API)."""
        return self.classify_all(tokens)

    def score_candidates(self, tokens: List[dict]) -> List[dict]:
        """Per-candidate raw scores for the threshold sweep.

        Returns one record per *candidate* token (cascade-eligible for the
        node), with the raw GLiNER score (0.0 when no span covers it).  The
        sweep pairs each record with a ground-truth nutrient label and
        thresholds the score.  Non-candidate tokens are excluded because the
        node is never asked about them.
        """
        scored = self._score_tokens(tokens)
        records: List[dict] = []
        for r in scored:
            if not r.get("is_candidate"):
                continue
            records.append({
                "token": r.get("token", ""),
                "norm": r.get("norm", ""),
                "rule_label": r.get("label", "UNKNOWN"),
                "gliner_score": r["gliner_score"] if r["gliner_score"] is not None else 0.0,
                "gliner_span_text": r["gliner_span_text"],
                "x1": r.get("x1"), "y1": r.get("y1"),
                "x2": r.get("x2"), "y2": r.get("y2"),
                "cx": r.get("cx"), "cy": r.get("cy"),
            })
        return records

    def summary(self, labeled_tokens: List[dict]) -> dict:
        """Print and return a label / method breakdown."""
        from collections import Counter

        label_counts = Counter(t.get("label", "UNKNOWN") for t in labeled_tokens)
        method_counts = Counter(t.get("classification_method", "?") for t in labeled_tokens)
        total = len(labeled_tokens)

        print(f"\n{'='*60}")
        print("  GLiNER SEMANTIC CLASSIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Total tokens : {total}")
        print(f"  Mode         : {self.mode}   threshold={self.threshold}")
        print("\n  --- Labels ---")
        for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE", "UNKNOWN"]:
            n = label_counts.get(label, 0)
            pct = n / total * 100 if total else 0
            print(f"  {label:<10}  {n:>4}  ({pct:>5.1f}%)")
        print("\n  --- Method ---")
        for method in ["rule", "gliner"]:
            n = method_counts.get(method, 0)
            pct = n / total * 100 if total else 0
            print(f"  {method:<10}  {n:>4}  ({pct:>5.1f}%)")
        print(f"{'='*60}\n")
        return {"labels": dict(label_counts), "methods": dict(method_counts)}


# Back-compat alias: existing imports of GLiNERClassifier keep working.
GLiNERClassifier = GLiNERSemanticClassifier


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-level convenience (caches one instance per process)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_default_classifier: Optional[GLiNERSemanticClassifier] = None


def classify_tokens(
    tokens: List[dict],
    mode: str = "gliner_only",
    model_id: str = MODEL_ID,
    threshold: float = NUTRIENT_THRESHOLD,
) -> List[dict]:
    """Single-call classification reusing a cached default instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = GLiNERSemanticClassifier(
            mode=mode, model_id=model_id, threshold=threshold,
        )
    return _default_classifier.classify_all(tokens)