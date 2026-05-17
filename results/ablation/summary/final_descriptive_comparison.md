# Final descriptive comparison

## 1. OCR engine

- `EXP-01` (easyocr): F1 = 0.1789
- `EXP-02` (paddleocr): F1 = 0.4072

**Winner: `paddleocr`** (EXP-02) — F1 = 0.4072, Δ +0.2283 over the other engine.

## 2. Classifier ranking (mean F1 across association methods)

- `A` (rule-based): mean F1 = 0.3794
- `B` (embedding (BGE-M3)): mean F1 = 0.3666
- `C` (GLiNER): mean F1 = 0.0000

## 3. Association method ranking (mean F1 across classifiers)

- `VLM` (vlm): mean F1 = 0.4844
- `G2` (graph_v2): mean F1 = 0.3968
- `G1` (graph_v1): mean F1 = 0.1585

## 4. Graph v2 vs Graph v1 (head-to-head, per classifier)

- `A` (rule-based): G1 0.2431 → G2 0.4072 (improved by +0.1641)
- `B` (embedding (BGE-M3)): G1 0.2325 → G2 0.3864 (improved by +0.1539)

## 5. VLM vs Graph v2 (head-to-head, per classifier)

- `A` (rule-based): G2 0.4072 → VLM 0.4879 (improved by +0.0807)
- `B` (embedding (BGE-M3)): G2 0.3864 → VLM 0.4810 (improved by +0.0946)

## 6. Best experiment by metric

- Highest F1: `EXP-05` (paddleocr + AVLM) = 0.4879
- Highest precision: `EXP-08` (paddleocr + BVLM) = 0.5820
- Highest recall: `EXP-05` (paddleocr + AVLM) = 0.4319

## 7. Stability across images (lower std = more stable)

- Most stable:   `EXP-09` (std F1 = 0.0000)
- Most volatile: `EXP-07` (std F1 = 0.3507)

## 8. Final pipeline recommendation

- **Recommended thesis pipeline: `EXP-05`** — OCR = paddleocr, classifier = A (rule-based), association = VLM (vlm). 4-field F1 = 0.4879.