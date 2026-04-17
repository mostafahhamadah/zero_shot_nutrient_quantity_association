"""
embedding_semantic_classifier.py
================================
Stage 3 (Hybrid) — Embedding-Augmented Semantic Token Classification
Zero-Shot Nutrient Extraction Pipeline | Moustafa Hamada | THD + USB

PURPOSE
-------
Replaces the NUTRIENT_LEXICON with zero-shot embedding similarity.
The rule-based classifier still handles QUANTITY, UNIT, CONTEXT, NOISE
(where deterministic rules are perfect).  But for NUTRIENT detection,
we use a pre-trained multilingual sentence encoder to compute cosine
similarity against semantic category prototypes.

WHY NOT PURE RULES?
-------------------
The NUTRIENT_LEXICON approach has fundamental limitations:
  - False negatives: any nutrient not in the 230-entry list → UNKNOWN
  - False positives: substring matching catches unrelated words
  - Maintenance burden: manually adding entries per language
  - Not zero-shot: the lexicon IS training data, just hand-coded

WHY NOT PURE EMBEDDINGS?
-------------------------
Numbers (QUANTITY), abbreviations (UNIT), and structural patterns
(CONTEXT, NOISE) are better handled by rules:
  - "400" is a quantity by syntax, not semantics
  - "mg" is too short for meaningful embedding similarity
  - "per 100g" has OCR variants that rules handle perfectly

ARCHITECTURE
------------
  1. Rules: QUANTITY? → UNIT? → CONTEXT? → NOISE? (deterministic)
  2. Everything else → embedding model → NUTRIENT or UNKNOWN
  3. No NUTRIENT_LEXICON used at all

DEPENDENCIES
------------
  pip install sentence-transformers torch
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Import the existing rule-based classifier
from src.classification.experiment_01_final_semantic_classifier import SemanticClassifier


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CATEGORY PROTOTYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Each category has multiple prototype descriptions.  The encoder maps
# these to embedding space; at inference time, a token's embedding is
# compared against ALL prototypes, and we take the max similarity per
# category.  More prototypes = better coverage of the semantic region.

CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    "NUTRIENT": [
        # ── Descriptive prototypes (semantic region) ──────────────────
        "vitamin or mineral nutrient name",
        "dietary nutrient like protein fat carbohydrate fiber",
        "amino acid supplement ingredient",
        "trace element like zinc iron selenium copper",
        "plant extract or herbal compound",
        "energy value or caloric content",
        "biological molecule found in food or supplements",

        # ── Real nutrient names as anchors (DE) ───────────────────────
        "Fett", "Eiweiß", "Salz", "Brennwert", "Energie",
        "Kohlenhydrate", "Ballaststoffe", "Fettsäuren",
        "davon Zucker", "davon gesättigte Fettsäuren",
        "Magnesium", "Kalzium", "Eisen", "Zink", "Selen",
        "Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D",
        "Folsäure", "Pantothensäure", "Thiamin", "Riboflavin",
        "Niacin", "Biotin", "Jod", "Kupfer", "Mangan",
        "Natrium", "Kalium", "Phosphor", "Chrom", "Molybdän",
        "Koffein", "Kreatin", "Inositol", "Cholin",

        # ── Real nutrient names as anchors (EN) ───────────────────────
        "Fat", "Protein", "Salt", "Energy",
        "Carbohydrate", "Fibre", "Sugars",
        "of which saturates", "of which sugars",
        "Calcium", "Iron", "Zinc", "Selenium",
        "Folic acid", "Pantothenic acid",
        "Omega 3", "DHA", "EPA",
        "Glucosamine", "Chondroitin", "Hyaluronic acid",
        "Taurine", "Curcuma", "Resveratrol", "Quercetin",
        "L-Carnitine", "Coenzyme Q10", "Lutein",
        "Creatine", "MSM", "Astaxanthin",

        # ── Real nutrient names as anchors (FR/IT/ES/NL) ──────────────
        "Protéines", "Glucides", "Lipides", "Sel",
        "Proteine", "Carboidrati", "Grassi",
        "Proteínas", "Grasas",
        "Eiwitten", "Koolhydraten", "Vetten",

        # ── Row-context patterns (nutrient + number + unit) ───────────
        "Magnesium 400 mg", "Vitamin C 80 mg", "Eisen 14 mg",
        "Fett 12.5 g", "Eiweiß 8.2 g", "Salz 0.3 g",
        "Brennwert 1200 kJ", "Kohlenhydrate 45 g",
        "Calcium 800 mg", "Vitamin B12 2.5 µg",
        "Zink 10 mg", "Selen 55 µg", "Jod 150 µg",
    ],
    "QUANTITY": [
        "numeric measurement value or amount",
        "decimal number representing dosage or weight",
        "count or quantity of milligrams grams micrograms",
    ],
    "UNIT": [
        "unit of measurement like milligrams grams kilojoules",
        "weight volume or energy measurement abbreviation",
        "mg g µg kcal kJ mL international units",
    ],
    "CONTEXT": [
        "per serving size or portion description",
        "per 100 grams or per 100 milliliters reference",
        "daily dose or recommended daily intake",
        "nutritional reference context column header",
    ],
    "NOISE": [
        "ingredient list or allergen warning text",
        "storage instructions or batch number",
        "manufacturer contact or company name",
        "regulatory disclaimer or legal notice",
        "percentage of daily reference value NRV",
        "barcode or product identification code",
        "packaging recycling disposal instructions",
        "Verpackung", "Zutaten", "Haltbar bis",
        "GmbH", "AG", "Hergestellt",
        "Mindestens haltbar", "Kühl lagern",
    ],
}

# Labels where rules are authoritative — never override with embeddings
# NUTRIENT is NOT here — embeddings own nutrient classification now
RULE_AUTHORITATIVE_LABELS = {"QUANTITY", "UNIT", "CONTEXT", "NOISE"}

# Labels that should go to embedding resolution (rule said these but
# we don't trust rule-based nutrient detection)
EMBEDDING_RESOLUTION_LABELS = {"NUTRIENT", "UNKNOWN"}

# Minimum similarity to accept NUTRIENT label from embeddings
DEFAULT_NUTRIENT_THRESHOLD = 0.30

# Minimum similarity gap between NUTRIENT and second-best category
DEFAULT_MARGIN = 0.05


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EmbeddingSemanticClassifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingSemanticClassifier:
    """
    Hybrid rule + embedding semantic classifier.

    Rules own: QUANTITY, UNIT, CONTEXT, NOISE (deterministic, 100% accurate).
    Embeddings handle NUTRIENT detection — but HOW depends on `mode`:

    mode="embedding_only"  (experiment A)
        NUTRIENT_LEXICON is bypassed.  ALL tokens that pass rule filters
        go to embeddings.  Embeddings alone decide NUTRIENT vs UNKNOWN.

    mode="hybrid"  (experiment B)
        Rule-based NUTRIENT_LEXICON runs first.  If lexicon says NUTRIENT,
        keep it.  Only UNKNOWN tokens go to embeddings for a second chance.
        Best of both worlds: lexicon precision + embedding recall.

    Parameters
    ----------
    mode : str
        "embedding_only" or "hybrid"
    model_name : str
        Sentence-transformers model to load.
    confidence_threshold : float
        Passed to the underlying rule-based classifier.
    nutrient_threshold : float
        Minimum cosine similarity for NUTRIENT label from embeddings.
    margin : float
        Required gap between best and second-best category similarity.
    device : str or None
        'cpu', 'cuda', or None (auto-detect).
    """

    VALID_MODES = {"embedding_only", "hybrid"}

    def __init__(
        self,
        mode: str = "hybrid",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        confidence_threshold: float = 0.30,
        nutrient_threshold: float = DEFAULT_NUTRIENT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        device: Optional[str] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode
        self.rule_classifier = SemanticClassifier(
            confidence_threshold=confidence_threshold
        )
        self.nutrient_threshold = nutrient_threshold
        self.margin = margin

        # Load model
        print(f"[EmbeddingClassifier] mode={mode} | Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # Pre-compute prototype embeddings
        self._prototype_labels: List[str] = []
        self._prototype_embeddings: np.ndarray = self._build_prototypes()
        print(
            f"[EmbeddingClassifier] Ready — "
            f"{len(self._prototype_labels)} prototypes across "
            f"{len(CATEGORY_PROTOTYPES)} categories"
        )

    def _build_prototypes(self) -> np.ndarray:
        """Encode all prototype phrases and store category mapping."""
        all_texts = []
        self._prototype_category_map: List[str] = []

        for category, phrases in CATEGORY_PROTOTYPES.items():
            for phrase in phrases:
                all_texts.append(phrase)
                self._prototype_category_map.append(category)

        self._prototype_labels = self._prototype_category_map
        embeddings = self.model.encode(
            all_texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(embeddings)

    def _compute_category_scores(self, token_text: str) -> Dict[str, float]:
        """
        Compute max cosine similarity between token and each category's
        prototype embeddings.

        Returns dict: {category: max_similarity}
        """
        token_emb = self.model.encode(
            [token_text], normalize_embeddings=True, show_progress_bar=False
        )
        # Cosine similarity (embeddings are L2-normalized)
        similarities = (token_emb @ self._prototype_embeddings.T).flatten()

        # Aggregate: max similarity per category
        scores: Dict[str, float] = {}
        for i, sim in enumerate(similarities):
            cat = self._prototype_category_map[i]
            if cat not in scores or sim > scores[cat]:
                scores[cat] = float(sim)

        return scores

    def _batch_compute_scores(
        self, texts: List[str]
    ) -> List[Dict[str, float]]:
        """
        Batch version of _compute_category_scores for efficiency.
        Encodes all texts in one forward pass.
        """
        if not texts:
            return []

        token_embs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=64,
        )
        # (N, D) @ (P, D).T → (N, P)
        all_sims = np.array(token_embs) @ self._prototype_embeddings.T

        results = []
        for row in all_sims:
            scores: Dict[str, float] = {}
            for j, sim in enumerate(row):
                cat = self._prototype_category_map[j]
                if cat not in scores or sim > scores[cat]:
                    scores[cat] = float(sim)
            results.append(scores)

        return results

    def _resolve_embedding_label(
        self, scores: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Decide if this token is NUTRIENT based on embedding scores.

        Since rules already handle QUANTITY/UNIT/CONTEXT/NOISE, the only
        real question here is: is this a NUTRIENT or not?

        Returns (label, confidence).
        """
        if not scores:
            return "UNKNOWN", 0.0

        nutrient_score = scores.get("NUTRIENT", 0.0)

        # Get the best NON-NUTRIENT score for margin check
        other_best = max(
            (v for k, v in scores.items() if k != "NUTRIENT"), default=0.0
        )

        if nutrient_score < self.nutrient_threshold:
            return "UNKNOWN", nutrient_score

        if (nutrient_score - other_best) < self.margin:
            return "UNKNOWN", nutrient_score  # too close to noise/other

        return "NUTRIENT", nutrient_score

    @staticmethod
    def _build_row_contexts(
        tokens: List[dict], cy_tolerance: float = 15.0
    ) -> Dict[int, List[int]]:
        """
        Group tokens into approximate rows by cy (y-center).
        Returns {row_id: [token_indices]} sorted left-to-right.
        """
        if not tokens:
            return {}

        # Sort by cy
        indexed = sorted(enumerate(tokens), key=lambda x: x[1].get("cy", 0))
        rows: Dict[int, List[int]] = {}
        row_id = 0
        current_cy = indexed[0][1].get("cy", 0)
        rows[row_id] = [indexed[0][0]]

        for orig_idx, tok in indexed[1:]:
            tok_cy = tok.get("cy", 0)
            if abs(tok_cy - current_cy) > cy_tolerance:
                row_id += 1
                current_cy = tok_cy
            else:
                # Running average
                current_cy = (current_cy * len(rows.get(row_id, [])) + tok_cy) / (
                    len(rows.get(row_id, [])) + 1
                )
            rows.setdefault(row_id, []).append(orig_idx)

        # Sort each row left-to-right by cx
        for rid in rows:
            rows[rid].sort(key=lambda i: tokens[i].get("cx", 0))

        return rows

    def _build_context_string(
        self, token_idx: int, tokens: List[dict],
        row_map: Dict[int, List[int]], idx_to_row: Dict[int, int]
    ) -> str:
        """
        Build a context-enriched string for a token by including its
        row neighbors. Example: "Fett" in row [Fett, 12.5, g] becomes
        "Fett 12.5 g".

        This gives the embedding model enough context to understand
        that "Fett" is a nutrient (it appears next to a number + unit).
        """
        rid = idx_to_row.get(token_idx)
        if rid is None:
            return tokens[token_idx].get("token", "")

        row_indices = row_map[rid]
        # Build row text from all tokens in this row
        row_tokens = [tokens[i].get("token", "") for i in row_indices]
        row_text = " ".join(row_tokens)

        # If row is very long (>10 tokens), use only nearby neighbors
        if len(row_indices) > 10:
            pos = row_indices.index(token_idx)
            start = max(0, pos - 3)
            end = min(len(row_indices), pos + 4)
            nearby = [tokens[row_indices[j]].get("token", "") for j in range(start, end)]
            row_text = " ".join(nearby)

        return row_text

    def _needs_embedding(self, rule_label: str) -> bool:
        """Check if this rule label should be sent to embeddings."""
        if rule_label in RULE_AUTHORITATIVE_LABELS:
            return False
        if self.mode == "hybrid" and rule_label == "NUTRIENT":
            return False  # lexicon said NUTRIENT → trust it
        # embedding_only: both NUTRIENT and UNKNOWN go to embeddings
        # hybrid: only UNKNOWN goes to embeddings
        return True

    # ── Public API ───────────────────────────────────────────────────

    def classify_token(self, token: dict) -> dict:
        """
        Classify a single token.
        Rules decide QUANTITY/UNIT/CONTEXT/NOISE.
        mode="embedding_only": embeddings decide ALL nutrient detection.
        mode="hybrid": lexicon keeps its NUTRIENT calls, embeddings
                       only rescue UNKNOWN tokens.
        """
        result = self.rule_classifier.classify_token(token)
        rule_label = result.get("label", "UNKNOWN")

        if not self._needs_embedding(rule_label):
            result["classification_method"] = "rule"
            result["embedding_scores"] = {}
            return result

        # Send to embeddings
        text = token.get("token", "")
        scores = self._compute_category_scores(text)
        emb_label, emb_conf = self._resolve_embedding_label(scores)

        result["embedding_scores"] = scores
        result["label"] = emb_label
        result["embedding_confidence"] = emb_conf
        result["classification_method"] = "embedding"

        return result

    def classify_all(self, tokens: List[dict]) -> List[dict]:
        """
        Classify all tokens with batched, context-aware embedding computation.

        Context-aware encoding: instead of encoding "Fett" alone, we encode
        "Fett 12.5 g" (the full row). This gives the model enough signal
        to recognize that short tokens like "Fett" are nutrients.
        """
        # Step 1: rule-based pass
        rule_results = [
            self.rule_classifier.classify_token(t) for t in tokens
        ]

        # Step 2: build row structure for context-aware encoding
        row_map = self._build_row_contexts(tokens)
        idx_to_row = {}
        for rid, indices in row_map.items():
            for idx in indices:
                idx_to_row[idx] = rid

        # Step 3: identify tokens needing embeddings (mode-dependent)
        needs_embedding_texts = []
        needs_embedding_idx = []

        for i, result in enumerate(rule_results):
            rule_label = result.get("label", "UNKNOWN")
            if not self._needs_embedding(rule_label):
                result["classification_method"] = "rule"
                result["embedding_scores"] = {}
                continue

            # Build context-aware text for this token
            context_text = self._build_context_string(i, tokens, row_map, idx_to_row)
            needs_embedding_texts.append(context_text)
            needs_embedding_idx.append(i)

        # Step 4: batch encode and resolve
        if needs_embedding_texts:
            all_scores = self._batch_compute_scores(needs_embedding_texts)

            for k, idx in enumerate(needs_embedding_idx):
                scores = all_scores[k]
                emb_label, emb_conf = self._resolve_embedding_label(scores)

                rule_results[idx]["embedding_scores"] = scores
                rule_results[idx]["label"] = emb_label
                rule_results[idx]["embedding_confidence"] = emb_conf
                rule_results[idx]["classification_method"] = "embedding"
                rule_results[idx]["embedding_context"] = needs_embedding_texts[k]

        return rule_results

    def summary(self, labeled_tokens: list) -> dict:
        """Print classification summary with method breakdown."""
        from collections import Counter

        label_counts = Counter(t.get("label", "UNKNOWN") for t in labeled_tokens)
        method_counts = Counter(
            t.get("classification_method", "unknown") for t in labeled_tokens
        )
        total = len(labeled_tokens)

        print(f"\n{'='*60}")
        print("  HYBRID SEMANTIC CLASSIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Total tokens : {total}")
        print(f"\n  --- Labels ---")
        for label in ["NUTRIENT", "QUANTITY", "UNIT", "CONTEXT", "NOISE", "UNKNOWN"]:
            n = label_counts.get(label, 0)
            pct = n / total * 100 if total else 0
            print(f"  {label:<10}  {n:>4}  ({pct:>5.1f}%)")
        print(f"\n  --- Classification Method ---")
        for method in ["rule", "embedding", "rule_fallback", "hybrid"]:
            n = method_counts.get(method, 0)
            pct = n / total * 100 if total else 0
            print(f"  {method:<15}  {n:>4}  ({pct:>5.1f}%)")
        print(f"{'='*60}\n")

        return {"labels": dict(label_counts), "methods": dict(method_counts)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys

    test_tokens = [
        # Row 0 (cy=50): context header
        {"token": "per 100g", "conf": 0.95, "x1": 200, "y1": 40, "x2": 300, "y2": 60, "cx": 250, "cy": 50},
        # Row 1 (cy=100): Brennwert 1200 kJ  (nutrient row)
        {"token": "Brennwert", "conf": 0.95, "x1": 10, "y1": 90, "x2": 100, "y2": 110, "cx": 55, "cy": 100},
        {"token": "1200", "conf": 0.95, "x1": 200, "y1": 90, "x2": 250, "y2": 110, "cx": 225, "cy": 100},
        {"token": "kJ", "conf": 0.95, "x1": 260, "y1": 90, "x2": 280, "y2": 110, "cx": 270, "cy": 100},
        # Row 2 (cy=140): Fett 12.5 g
        {"token": "Fett", "conf": 0.95, "x1": 10, "y1": 130, "x2": 60, "y2": 150, "cx": 35, "cy": 140},
        {"token": "12.5", "conf": 0.95, "x1": 200, "y1": 130, "x2": 240, "y2": 150, "cx": 220, "cy": 140},
        {"token": "g", "conf": 0.95, "x1": 250, "y1": 130, "x2": 260, "y2": 150, "cx": 255, "cy": 140},
        # Row 3 (cy=180): davon Zucker 4.2 g
        {"token": "davon Zucker", "conf": 0.95, "x1": 20, "y1": 170, "x2": 130, "y2": 190, "cx": 75, "cy": 180},
        {"token": "4.2", "conf": 0.95, "x1": 200, "y1": 170, "x2": 230, "y2": 190, "cx": 215, "cy": 180},
        {"token": "g", "conf": 0.95, "x1": 240, "y1": 170, "x2": 250, "y2": 190, "cx": 245, "cy": 180},
        # Row 4 (cy=220): Salz 0.3 g
        {"token": "Salz", "conf": 0.95, "x1": 10, "y1": 210, "x2": 60, "y2": 230, "cx": 35, "cy": 220},
        {"token": "0.3", "conf": 0.95, "x1": 200, "y1": 210, "x2": 230, "y2": 230, "cx": 215, "cy": 220},
        {"token": "g", "conf": 0.95, "x1": 240, "y1": 210, "x2": 250, "y2": 230, "cx": 245, "cy": 220},
        # Row 5 (cy=260): EiweiB 8.2 g  (OCR corruption)
        {"token": "EiweiB", "conf": 0.95, "x1": 10, "y1": 250, "x2": 80, "y2": 270, "cx": 45, "cy": 260},
        {"token": "8.2", "conf": 0.95, "x1": 200, "y1": 250, "x2": 230, "y2": 270, "cx": 215, "cy": 260},
        {"token": "g", "conf": 0.95, "x1": 240, "y1": 250, "x2": 250, "y2": 270, "cx": 245, "cy": 260},
        # Row 6 (cy=300): Magnesium 400 mg
        {"token": "Magnesium", "conf": 0.95, "x1": 10, "y1": 290, "x2": 110, "y2": 310, "cx": 60, "cy": 300},
        {"token": "400", "conf": 0.95, "x1": 200, "y1": 290, "x2": 240, "y2": 310, "cx": 220, "cy": 300},
        {"token": "mg", "conf": 0.95, "x1": 250, "y1": 290, "x2": 270, "y2": 310, "cx": 260, "cy": 300},
        # Row 7 (cy=340): Vitamin B12 2.5 µg
        {"token": "B12", "conf": 0.95, "x1": 10, "y1": 330, "x2": 50, "y2": 350, "cx": 30, "cy": 340},
        {"token": "2.5", "conf": 0.95, "x1": 200, "y1": 330, "x2": 230, "y2": 350, "cx": 215, "cy": 340},
        {"token": "µg", "conf": 0.95, "x1": 240, "y1": 330, "x2": 260, "y2": 350, "cx": 250, "cy": 340},
        # Row 8 (cy=380): Glucosamine 500 mg (not in lexicon)
        {"token": "Glucosamine", "conf": 0.95, "x1": 10, "y1": 370, "x2": 120, "y2": 390, "cx": 65, "cy": 380},
        {"token": "500", "conf": 0.95, "x1": 200, "y1": 370, "x2": 240, "y2": 390, "cx": 220, "cy": 380},
        {"token": "mg", "conf": 0.95, "x1": 250, "y1": 370, "x2": 270, "y2": 390, "cx": 260, "cy": 380},
        # Row 9 (cy=420): Hyaluronsäure 50 mg (not in lexicon)
        {"token": "Hyaluronsäure", "conf": 0.95, "x1": 10, "y1": 410, "x2": 140, "y2": 430, "cx": 75, "cy": 420},
        {"token": "50", "conf": 0.95, "x1": 200, "y1": 410, "x2": 230, "y2": 430, "cx": 215, "cy": 420},
        {"token": "mg", "conf": 0.95, "x1": 250, "y1": 410, "x2": 270, "y2": 430, "cx": 260, "cy": 420},
        # Row 10 (cy=460): Taurin 100 mg
        {"token": "Taurin", "conf": 0.95, "x1": 10, "y1": 450, "x2": 70, "y2": 470, "cx": 40, "cy": 460},
        {"token": "100", "conf": 0.95, "x1": 200, "y1": 450, "x2": 240, "y2": 470, "cx": 220, "cy": 460},
        {"token": "mg", "conf": 0.95, "x1": 250, "y1": 450, "x2": 270, "y2": 470, "cx": 260, "cy": 460},
        # Row 11 (cy=500): Curcuma 250 mg
        {"token": "Curcuma", "conf": 0.95, "x1": 10, "y1": 490, "x2": 90, "y2": 510, "cx": 50, "cy": 500},
        {"token": "250", "conf": 0.95, "x1": 200, "y1": 490, "x2": 240, "y2": 510, "cx": 220, "cy": 500},
        {"token": "mg", "conf": 0.95, "x1": 250, "y1": 490, "x2": 270, "y2": 510, "cx": 260, "cy": 500},
        # Row 12 (cy=600): noise tokens (different area)
        {"token": "GmbH", "conf": 0.95, "x1": 10, "y1": 590, "x2": 70, "y2": 610, "cx": 40, "cy": 600},
        {"token": "Haltbar bis", "conf": 0.95, "x1": 100, "y1": 590, "x2": 200, "y2": 610, "cx": 150, "cy": 600},
        {"token": "Verpackung", "conf": 0.95, "x1": 220, "y1": 590, "x2": 320, "y2": 610, "cx": 270, "cy": 600},
    ]

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    def run_mode(m):
        print(f"\n{'#'*60}")
        print(f"  MODE: {m}")
        print(f"{'#'*60}")
        clf = EmbeddingSemanticClassifier(mode=m)
        results = clf.classify_all(test_tokens)
        print(f"\n{'token':<20} {'label':<10} {'method':<12} {'context used':<35} {'top scores'}")
        print("-" * 120)
        for r in results:
            scores_str = ""
            if r.get("embedding_scores"):
                top = sorted(r["embedding_scores"].items(), key=lambda x: -x[1])[:3]
                scores_str = " | ".join(f"{k}={v:.3f}" for k, v in top)
            ctx = r.get("embedding_context", "—")
            if len(ctx) > 33:
                ctx = ctx[:30] + "..."
            print(
                f"  {r['token']:<20} {r['label']:<10} "
                f"{r.get('classification_method', '?'):<12} "
                f"{ctx:<35} {scores_str}"
            )
        clf.summary(results)
        return results

    if mode == "both":
        r_emb = run_mode("embedding_only")
        r_hyb = run_mode("hybrid")

        # Disagreement report
        print(f"\n{'='*60}")
        print("  DISAGREEMENTS (embedding_only vs hybrid)")
        print(f"{'='*60}")
        diffs = 0
        for e, h, t in zip(r_emb, r_hyb, test_tokens):
            if e["label"] != h["label"]:
                diffs += 1
                print(
                    f"  {t['token']:<20} "
                    f"emb_only={e['label']:<10} "
                    f"hybrid={h['label']:<10}"
                )
        if diffs == 0:
            print("  (none)")
        print(f"\n  Total disagreements: {diffs}/{len(test_tokens)}\n")
    else:
        run_mode(mode)