# The `role_rank` Row-Edge Mode — Experiment Log & Failure Analysis

**Project:** Zero-Shot Nutrient and Quantity Association using OCR-Guided Semantic Graph Matching
**Component under test:** Stage 4 graph construction — `ROW_COMPAT` edge construction
**Pipeline (held fixed):** PaddleOCR PP-OCRv5 → Corrector V2 (C15) → Embedding classifier (BGE-M3, `embedding_only`, t = 0.59, m = 0.04) → Token Enricher → **Graph V2** → Association V2 (+ paragraph fallback) → fast rule-based evaluator
**Ground truth:** `test_set_normalized.csv` (866 tuples, 57 images)
**Runner:** `run_graph_v3_experiment.py`

> All configurations in this document share the *same* classifier, threshold, GT, and evaluator. The **only** variable is how `ROW_COMPAT` edges are built. Numbers are therefore directly comparable to one another, but **not** to earlier rule-based-classifier results (e.g. the older 41.5%), which used a different Stage 3.

---

## 1. What is `role_rank`?


### 1.1 Background: the row-edge relation

In the semantic graph, a `ROW_COMPAT` edge asserts that two tokens belong to the **same table row** — i.e. the same nutrient record. Resolving a tuple `(nutrient, quantity, unit, context)` depends on this edge: the association stage walks from a nutrient along its `ROW_COMPAT` edges to find its quantities and their units. How the "same row" relation is *detected* is a design choice with several options:

| Row mode | Detection signal | Nature |
|---|---|---|
| `cy` (default) | `|cy_a − cy_b| ≤ threshold` — tokens on the same pixel line | Geometric |
| `rank` | equal `data_rank_in_column`, across **different** columns only | Structural |
| **`role_rank`** | **equal label-specific rank (see below), same or different column** | **Structural** |

`role_rank` was introduced to repair a specific defect in plain `rank` (Section 1.3).

### 1.2 Definition

`role_rank` links two tokens with a `ROW_COMPAT` edge when they occupy **the same ordinal position within their own semantic role**. Each token is assigned a *label-specific rank* — a counter that increments only over tokens of the same label, computed per column in the enricher:

- `NUTRIENT` → `nutrient_rank_in_column`
- `QUANTITY` → `qty_rank_in_column`
- `UNIT` → `unit_rank_in_column`  *(counter added to the enricher specifically for this mode)*
- any other label (CONTEXT, NRV, …) → not linked by `role_rank`

The edge rule:

> Two enriched nodes *a* and *b* receive a bidirectional `ROW_COMPAT` edge **iff** `role_rank(a) == role_rank(b)` and both are ≥ 0.

Concretely, this links the **k-th nutrient ↔ the k-th quantity ↔ the k-th unit**:

```
NUTRIENT (nutrient_rank = k)  ──ROW──  QUANTITY (qty_rank = k)  ──ROW──  UNIT (unit_rank = k)
```

For a clean two-stream table this reconstructs each row by ordinal position rather than by pixel coordinate:

```
Magnesium  (nut_rank 0)   400 (qty_rank 0)  mg (unit_rank 0)   120 (qty_rank 0, stream 1)  mg (unit_rank 0, stream 1)
Vitamin C  (nut_rank 1)    80 (qty_rank 1)  mg (unit_rank 1)    24 (qty_rank 1, stream 1)  mg (unit_rank 1, stream 1)
```

### 1.3 What it was designed to fix (vs plain `rank`)

Plain `rank` linked equal `data_rank_in_column` across **different** columns only. `data_rank` counts *all* non-header tokens in a column, so when a dosage column interleaves quantities and inline units (`400`, `mg`, `80`, `mg`, …), the quantity ranks no longer align with the nutrient ranks, and — because plain `rank` excludes same-column pairs — an **inline unit is never linked to its quantity at all**. In testing, plain `rank` collapsed unit accuracy to **26.9%** (exp39b).

`role_rank` addresses both points:

1. **Label-specific counters** (`qty_rank`, `unit_rank`) skip the interleaving, so the k-th quantity and the k-th unit align even when units sit between quantities in the column.
2. **Same-column links are permitted**, so a quantity and its *inline* unit (same `column_id`, `qty_rank = k` / `unit_rank = k`) are connected — the case plain `rank` structurally could not reach.

This was verified at the edge level before evaluation: under `role_rank`, an inline `400`↔`mg` edge is created where plain `rank` produces none.

---

## 2. Experiment A — Testing `role_rank` in isolation (exp39c)

### 2.1 Configuration

```
--row-edge-mode role_rank --col-edge-mode cx --no-llm
```

### 2.2 Results

| Metric | plain `rank` (exp39b) | **`role_rank` (exp39c)** | `cy` baseline (exp39d) |
|---|---|---|---|
| Quantity Match Acc | 33.0% | 29.0% | 76.3% |
| **Unit Match Acc** | 26.9% | **62.1%** | 82.4% |
| Context Match Acc | 75.2% | 79.2% | 85.0% |
| Nutrient F1 | 0.645 | 0.673 | 0.666 |
| 3F-F1 | 5.7% | 14.4% | 40.8% |
| **4F-F1 (primary)** | 5.7% | **13.2%** | **38.6%** |
| Predicted tuples | 749 | 694 | 961 |
| True hits (4F) | 46 | 103 | 353 |

### 2.3 Interpretation

- **The unit fix worked as designed.** Unit accuracy rose 26.9% → 62.1% (≈ 2.3×), and 4F-F1 rose 5.7% → 13.2% (≈ 2.3×). True hits more than doubled (46 → 103). The quantity↔unit linkage that plain `rank` destroyed was restored. Per-image confirmation: the "Quantity 100% / Unit 0%" images under plain `rank` (e.g. 109, 115) became Unit 88% / 86% under `role_rank`.
- **Quantity remained the bottleneck.** Quantity accuracy did **not** improve (33.0% → 29.0%); it is now the single field capping 4F-F1. The arithmetic ceiling of the per-field accuracies (≈ 0.29 × 0.62 × 0.79) is consistent with the observed 13.2%.
- **`role_rank` trails the `cy` baseline by ~25 pp** (13.2% vs 38.6%), and **the entire gap is quantity** (29% vs 76%). Unit and context are within ~6 pp and ~6 pp of `cy` respectively.

The conclusion at this stage: `role_rank` is a *correct structural row mode for units*, but its quantity association is fundamentally weaker than geometry. Section 4 explains why.

---

## 3. Experiment B — Merging `role_rank` with the default (exp39f_merged)

### 3.1 Rationale

Since `role_rank` recovered some structure and `cy` was strong, a **late-fusion union** was tested: run the default (`cy`) association first and keep all of its tuples, then run the `role_rank` association and **add only the tuples `role_rank` produced that `cy` did not** (dedup on the full 5-field key, default surface form preferred). The hypothesis was that `role_rank` might recover a few rows `cy` missed, raising recall without harming the rest.

This was implemented as a dedicated component (`merged_graph_associator.py`, exposed via the runner's `--merge-modes` flag); each mode gets its own clean graph and association pass, and only the final tuple sets are unioned. The default (`cy`) result is never altered.

### 3.2 Configuration

```
--merge-modes --no-llm
```

### 3.3 Results

| Metric | `cy` baseline (exp39d) | **merge (exp39f)** | Δ |
|---|---|---|---|
| 4F Recall | 40.8% | 42.3% | **+1.5 pp** |
| 4F Precision | 36.7% | 25.6% | **−11.1 pp** |
| **4F-F1** | **38.6%** | **31.9%** | **−6.7 pp** |
| Nutrient Precision | 0.633 | 0.462 | −0.171 |
| Nutrient Recall | 0.702 | 0.762 | +0.060 |
| Quantity Match Acc | 76.3% | 73.9% | −2.4 pp |
| Predicted tuples | 961 | 1428 | +467 |
| True hits (4F) | 353 | 366 | +13 |

### 3.4 Interpretation — the merge hurt

The union added **467** unique `role_rank` tuples. Of those, only **13** were correct full-tuple matches (353 → 366 true hits); the remaining **454** were false positives.

- **Yield of the additions: ≈ 2.8% correct (≈ 97.2% wrong).**
- Recall rose marginally (+1.5 pp) because a handful of genuine rows were recovered; precision collapsed (−11.1 pp) because of the 454 false positives.
- Because F1 is the harmonic mean of precision and recall, a small recall gain cannot offset a large precision loss — net **−6.7 pp** (38.6% → 31.9%).

Per-image, the multi-stream tables inflated with false positives (e.g. 20.png: 48 GT → 114 predicted; 34.png: 40 GT → 86 predicted), and the two per-serving images (118/119) stayed at **0% quantity** — unaffected, because their problem is not the row mode (Section 4.5).

**The union did not fail due to a bug.** It failed for a structural reason: **a high-precision tuple set cannot be improved by adding a low-precision one.** `role_rank`'s *unique* tuples are precisely its *wrong* ones — any `role_rank` tuple that agreed with `cy` was removed as a duplicate, leaving the misaligned-quantity tuples as the additions.

---

## 4. Failure Analysis

### 4.1 The core mechanism — ordinal pairing assumes equal, ordered cardinality

`role_rank` pairs the **k-th nutrient with the k-th quantity by ordinal position**. This is correct **only if** the nutrient column and the quantity column contain the **same number of entries, in the same vertical order**. The ranks are independent down-column counters; if the two counters fall out of step at position *j*, then **every** pairing from *j* onward is shifted onto a neighbour's value.

Real OCR on these labels violates the equal-cardinality assumption in four recurring ways:

1. **Dropped token.** A missed value in the dosage column gives it *N − 1* quantities for *N* nutrients. From the gap downward, nutrient *k* is paired with quantity *k − 1* (its predecessor's value). One missing token corrupts the entire tail of the table.
2. **Fused token.** `400mg` read as a single `QUANTITY` token (no separate `UNIT`) shortens the unit column relative to the quantity column, so `unit_rank` stops aligning with `qty_rank`.
3. **Extra / spurious token.** An NRV `%` value, a footnote number, or a stray glyph classified as `QUANTITY` *adds* an entry, shifting the counter the other way.
4. **Mis-split / merged column.** If column induction places values in the wrong band, `qty_rank` is counted over the wrong set of tokens entirely — the ranks are meaningless before pairing begins.

### 4.2 Why **quantity** fails while **unit** looks fine

In all four break modes the nutrient is usually still correct and the **unit often still appears correct** — because units repeat down a column (`mg`, `mg`, `mg`, …), so a shifted row still lands on `mg`. The **quantity** is the only field whose every value is *distinct* (`400`, `80`, `333`, …), so a one-row shift is immediately wrong. This is exactly the observed signature in exp39c: Unit 62.1% but Quantity 29.0%. **High unit accuracy under `role_rank` does not imply correct rows — it only implies a homogeneous unit column.**

### 4.3 Why `cy` does not have this problem

`cy` links tokens by shared vertical position — they are physically on the same printed line. A dropped token, a fused unit, a spurious `%`, or a mis-induced column does **not** move the *remaining* tokens off their line: `400` is at the same height as `Magnesium` regardless of how many tokens sit above it. `cy` therefore reads the value directly off the page geometry, whereas `role_rank` must *reconstruct* the row by counting — and counting is fragile to precisely the noise OCR introduces. This invariance is the entire ~25 pp quantity gap.

### 4.4 Why the merge amplified the failure

Section 4.1–4.2 explain why `role_rank`'s tuples are wrong; the merge then **added** them. Deduplication removed every `role_rank` tuple that matched `cy` (consensus), so the surviving additions were dominated by the misaligned-quantity tuples. Pouring a ~3%-correct set into a 36.7%-precision set drove precision down far faster than recall rose — the harmonic mean fell. Late fusion is only beneficial when the second source is **high-precision and complementary**; `role_rank` is neither.

### 4.5 The residual case not explained by row mode — 118/119

Images 118 and 119 sit at **0% quantity under every configuration**, including `cy`. Their failure is break-mode #4 (column mis-induction): the per-serving quantities are placed in the wrong column band, so neither geometry nor rank can associate them correctly. This is an **enricher / column-induction** issue (candidate fix: tuning `col_max_parallel_px`), **not** a row-edge issue, and it is the only quantity loss that remains open after this experiment. The enricher's `rank_consistent` diagnostic flags these images.

---

## 5. Conclusion & Thesis Framing

### 5.1 Verdict

- **Do not adopt `role_rank` as the production row mode**, and **do not ship the merge.** The geometric default (`cy`, 38.6% 4F-F1) is the best graph configuration.
- `role_rank` (13.2%), plain `rank` (5.7%), and the late-fusion merge (31.9%) are retained as an **ablation**, not as candidates.

### 5.2 The defensible claim

> We replaced geometric row edges with structural ordinal-rank edges, holding the OCR, classifier, enricher, association, and evaluator fixed. Performance fell from 38.6% to 13.2% 4F-F1, with the loss concentrated almost entirely in quantity (76% → 29%). The failure is mechanistic: ordinal rank assumes equal, ordered column cardinality, which OCR violates through dropped tokens, fused units, spurious detections, and column mis-induction; geometric position is invariant to all four. A late-fusion union of the two (default + unique structural tuples) reduced F1 further (to 31.9%), because ~97% of the structural mode's unique tuples are false positives and cannot improve a higher-precision set. This empirically establishes that **geometric row detection is a necessary component of the graph, not an interchangeable design choice.**

This is a stronger result for the defense than reporting the winning number alone: it demonstrates *why* the chosen design wins, and quantifies the cost of the principled alternative.

---

## Appendix — Configuration provenance & exact figures

| Experiment | Row mode | Col mode | Pred | Matched | NutF1 | Qty | Unit | Ctx | 3F-F1 | 4F-F1 | True hits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp39b | `rank` | `column_id` | 749 | 521 | 0.645 | 33.0% | 26.9% | 75.2% | 5.7% | 5.7% | 46 |
| exp39c | `role_rank` | `cx` | 694 | 525 | 0.673 | 29.0% | 62.1% | 79.2% | 14.4% | 13.2% | 103 |
| exp39d | `cy` | `cx` | 961 | 608 | 0.666 | 76.3% | 82.4% | 85.0% | 40.8% | 38.6% | 353 |
| exp39f_merged | `cy` ∪ `role_rank` (unique-add) | `cx` | 1428 | 660 | 0.575 | 73.9% | 79.2% | 82.1% | 34.0% | 31.9% | 366 |

**Common settings:** classifier = `EmbeddingSemanticClassifier(mode="embedding_only", nutrient_threshold=0.59, margin=0.04, confidence_threshold=0.30)`; evaluator = fast rule-based (`--no-llm`, 0 LLM calls); GT = `test_set_normalized.csv` (866 tuples, 57 images); runner = `run_graph_v3_experiment.py`.

**Merge accounting (exp39f vs exp39d):** +467 predicted tuples, +13 true hits, +454 false positives ⇒ ≈ 2.8% of additions correct. 4F precision 36.7% → 25.6%; 4F recall 40.8% → 42.3%; 4F-F1 38.6% → 31.9%.

**Note on comparability:** these figures use the embedding classifier and are **not** comparable to earlier rule-based-classifier results (e.g. 41.5% 4F-F1), which differ at Stage 3.
