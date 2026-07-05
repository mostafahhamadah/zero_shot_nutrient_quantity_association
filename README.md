# Zero-Shot Nutrient and Quantity Association using OCR-Guided Semantic Graph Matching

M.Sc. thesis project — **Moustafa Hamada**
AI & Data Science, Deggendorf Institute of Technology (THD) + University of South Bohemia
Supervisor: Prof. Dr. Andreas Fischer

---

## What this is

A modular, interpretable pipeline that reads a supplement or food-label image and
extracts structured four-field tuples:

```
⟨ nutrient , quantity , unit , context ⟩
```

for example:

```
Magnesium (gesamt) , 400 , mg , per_daily_dose
```

The task is **zero-shot**: no model is trained on the target labels. OCR only
supplies tokens with coordinates; the scientific contribution is the reasoning
that turns those tokens into correct tuples.

The headline novelty is **OCR-guided semantic graph matching** — an interpretable
Stage 4/5 that links a nutrient to its own quantity/unit/context using geometric
and structural edges over the OCR tokens. A vision-language model (Gemma 3 4B) is
included **only** as a black-box comparison point in the association ablation; it
is not the primary method.

Evaluation is on a fixed test set of **57 images / 866 gold tuples**
(`test_set_normalized.csv`).

---

## Pipeline

Six stages. Stage 3 (classifier) and Stage 5 (associator) each have interchangeable
back-ends, which is what the thesis ablates.

```
 Stage 1   OCR                PaddleOCR PP-OCRv5        src/ocr/paddleocr_runner.py
                              (EasyOCR = beaten baseline src/ocr/ocr_runner.py)
    │
 Stage 2   Corrector          C1–C17, C15 English       src/utils/paddleOCR_corrector_v2.py
                              canonicalisation
    │
 Stage 3   Semantic classifier   one of:
             • rule-based      SemanticClassifier            src/classification/experiment_01_final_semantic_classifier.py
             • embedding       EmbeddingSemanticClassifier   src/classification/embedding_semantic_classifier.py   (BGE-M3, t=0.59 / m=0.04)
             • GLiNER          GLiNERSemanticClassifier      src/classification/gliner_classifier.py               (biomed, τ*=0.18)
    │
 Stage 3.5 Token enricher     TokenEnricher             src/utils/token_enricher.py
    │
 Stage 4   Graph construction GraphConstructorV2        src/graph/graph_constructor_v2.py
                              (overlap edges by default; cy/cx/rank/role_rank/column_id selectable)
    │
 Stage 5   Association        one of:
             • graph matcher   TupleAssociatorV2         src/matching/association_v2.py         ← headline
             • VLM             VLMAssociator             src/matching/vlm_association.py        (Gemma 3 4B, comparison)
             • late fusion     MergedModeAssociator      src/matching/merged_graph_associator.py (ablation only)
                              + paragraph/sentence fallback (src/utils/paragraph_extractor.py, sentence_extractor.py)
    │
 Stage 6   Evaluation         LLMTupleEvaluator         src/evaluation/llm_evaluator.py
                              (use_llm=False → deterministic, zero LLM calls, exactly reproducible)
```

Semantic role colours used across the figures: **nutrient = blue, quantity =
green, unit = orange, context = purple**.

---

## Repository structure

```
zero_shot_nutrient_association/
├── README.md
├── .gitignore
├── .gitattributes
├── requirements.txt                    # generate locally: pip freeze > requirements.txt
├── test_set_normalized.csv             # gold: 866 tuples / 57 images (V2 evaluation)
├── app.py                              # Streamlit app for the pipeline (per-stage inspection/debugging; also edits data/annotations/*.json)
├── thesis_defense_interactive.html     # interactive defence deck
│
├── run_experiment_paddle.py            # baseline runner : PaddleOCR + rule clf + Graph V1  (GT = JSON)
├── run_embedding_only_experiment_v2.py # V2 runner       : PaddleOCR + BGE-M3 + Graph V2     (GT = CSV)
├── run_gliner_experiment.py            # V2 runner       : PaddleOCR + GLiNER + Graph V2      (GT = CSV)
│
├── ablations.ipynb                     # ablation + pipeline-configuration studies (notebook)
├── ablation_experiments_notebook.ipynb # ablation + pipeline-configuration studies (notebook)
│
├── data/
│   ├── raw/                            # 57 label images (input)
│   └── annotations/                    # per-image gold JSON (used by run_experiment_paddle.py)
│
├── outputs/                            # generated per run — git-ignored
│   └── <experiment_name>/
│       ├── tuples.csv                  # predicted tuples
│       ├── evaluation_results.json     # metrics
│       ├── pipeline_diagnostics.csv    # per-stage counts
│       └── run_log.txt
│
└── src/
    ├── ocr/          paddleocr_runner.py, ocr_runner.py
    ├── utils/        paddleOCR_corrector_v2.py, ocr_corrector.py, token_enricher.py,
    │                 paragraph_extractor.py, sentence_extractor.py, text_serializer.py
    ├── classification/ experiment_01_final_semantic_classifier.py,
    │                    embedding_semantic_classifier.py, gliner_classifier.py
    ├── graph/        graph_constructor_v2.py, graph_constructor.py
    ├── matching/     association_v2.py, vlm_association.py, merged_graph_associator.py,
    │                 experiment_01_final_association.py, association.py
    └── evaluation/   llm_evaluator.py
```

> Module paths above follow the imports the runners use (`from src.<pkg>.<module> import …`).
> Each `src/` sub-folder is an importable package — keep an empty `__init__.py` in each,
> or rely on namespace packages. Adjust the tree if your local layout differs.

---

## Installation

**Prerequisites**

- Python 3.11 (Windows)
- For the VLM comparison only: [LM Studio](https://lmstudio.ai) serving
  `Gemma 3 4B Instruct` (Q4_K_M GGUF) on `http://127.0.0.1:1234`.

**Environment**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install --upgrade pip
```

**Core dependencies** (the versions this project is pinned to):

```bash
# PyTorch — nightly cu128 is required for RTX 5070 / Blackwell GPU support
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# OCR, embedding, NER, and glue
pip install paddlepaddle==2.6.2          # CPU build
pip install paddleocr
pip install sentence-transformers==5.4.1
pip install gliner
pip install "numpy==1.26.4"              # pinned; do not upgrade
pip install streamlit                    # for app.py
```

Then freeze your exact environment so the repo is reproducible:

```bash
pip freeze > requirements.txt
```

**Models** download automatically on first use:

- PaddleOCR PP-OCRv5 weights (PaddleOCR cache)
- `BAAI/BGE-M3` embedding model (Hugging Face)
- `Ihor/gliner-biomed-bi-large-v1.0` NER model (Hugging Face)
- Gemma 3 4B is loaded inside LM Studio, not via `pip`.

**Hardware notes**

- Target machine: NVIDIA RTX 5070 Laptop (Blackwell `sm_120`, 8 GB VRAM).
- PaddleOCR is **CPU-forced** — the current PaddlePaddle build does not support
  Blackwell. BGE-M3 and GLiNER are also CPU-pinned.
- The VLM runs on the GPU through LM Studio (full offload, Flash Attention, q8_0
  KV cache).
- None of the above is required to read the code or reproduce the deterministic
  evaluation numbers; it only affects run speed.

---

## Data

| Source | Used by | Contents |
|---|---|---|
| `data/raw/` | all runners | 57 supplement/food-label images (the pipeline input) |
| `test_set_normalized.csv` | V2 runners | gold, 866 tuples; columns: `image_id, nutrient, quantity, unit, context, nrv_percent, serving_size` |
| `data/annotations/*.json` | `run_experiment_paddle.py` | per-image gold, edited through `app.py` |

The evaluator scores four fields. In the V2 runners `context` is combined with
`serving_size` as `"context (serving_size)"` before matching; `nrv_percent` is
carried in the gold but not part of the scored tuple.

---

## Usage

Every runner writes to `outputs/<experiment_name>/`. Add `--no-llm` to run the
**deterministic** evaluator (no LLM calls, exactly reproducible) — this is the
mode used to report the locked numbers below.

**1. GLiNER + Graph V2 — best interpretable configuration (run E2-C)**

```bash
python run_gliner_experiment.py --experiment e2c_gliner_graphv2 --no-llm
```

Defaults are the tuned overlap graph (row/col/context = overlap). Useful flags:

```bash
--threshold 0.18            # GLiNER span-acceptance τ*  (default 0.18)
--mode hybrid               # keep lexicon nutrients, GLiNER rescues UNKNOWN only
--row-edge-mode cy --col-edge-mode cx --context-scope-mode cy   # geometric-centroid graph
--col-edge-mode column_id   # structural column edges (ablation)
--merge-modes               # late-fusion: cy + role_rank unique-add (ablation)
--audit --images 1,108,118  # dump every stage's output for the listed images
--compare e2b_embedding_graphv2   # diff metrics against a prior run folder
```

**2. Embedding (BGE-M3) + Graph V2 (run E2-B)**

```bash
python run_embedding_only_experiment_v2.py --experiment e2b_embedding_graphv2 --no-llm
```

```bash
--threshold 0.59 --margin 0.04   # tuned NUTRIENT cosine threshold / margin (defaults)
--images 1,101,118               # restrict to specific image stems
--compare e2c_gliner_graphv2
```

**3. PaddleOCR baseline — rule classifier + Graph V1 (GT from JSON)**

```bash
python run_experiment_paddle.py --no-llm
```

**VLM-association comparison (best overall, run E3-EMB-VLM = 0.4945)**

This swaps Stage 5 for `VLMAssociator` and requires LM Studio to be running at
`http://127.0.0.1:1234`. `vlm_association.py` supports three back-ends
(`openai_compat` for LM Studio — default, `hf` for the Hugging Face router,
`ollama` legacy). Use the VLM runner script that wires `src.matching.vlm_association`
into the V2 cascade.

**Ablation & pipeline-configuration studies**

The full ablation and pipeline-configuration studies (classifier back-ends, graph
edge modes, threshold calibration, association engines) are carried out in two
Jupyter notebooks in the repo root:

- `ablations.ipynb`
- `ablation_experiments_notebook.ipynb`

These drive the runners/modules above across configurations and collate the
per-configuration metrics reported in the results table and in Chapter 6.

```bash
jupyter lab            # then open either notebook
```

**Inspect the pipeline visually**

`app.py` is a Streamlit app for the pipeline — it runs each stage and shows the
per-stage output for a chosen image (per-stage debugging), and is also where the
`data/annotations/*.json` gold files are edited.

```bash
streamlit run app.py
```

---

## Outputs

Each run produces, under `outputs/<experiment_name>/`:

- `tuples.csv` — predicted `image_id, nutrient, quantity, unit, context`
- `evaluation_results.json` — nutrient F1, quantity/unit/context accuracy,
  3-field and 4-field tuple F1, precision/recall
- `pipeline_diagnostics.csv` — per-stage token, edge and tuple counts
- `run_log.txt` — full run log

`outputs/` is git-ignored. To keep a curated results folder under version control,
force-add specific files: `git add -f outputs/e2c_gliner_graphv2/evaluation_results.json`.

---

## Results (locked, deterministic evaluator)

Four-field tuple F1 on the 57-image / 866-tuple test set:

| Run | Configuration | 4F-F1 |
|---|---|---:|
| EXP-01 | EasyOCR baseline | 0.1773 |
| E2-A | PaddleOCR · rule · Graph V2 | 0.4231 |
| E2-B | PaddleOCR · BGE-M3 · Graph V2 | 0.4025 |
| **E2-C** | **PaddleOCR · GLiNER · Graph V2** | **0.4304** ← best interpretable |
| A_VLM | rule · VLM association | 0.4819 |
| **E3-EMB-VLM** | **BGE-M3 · VLM association** | **0.4945** ← best overall |
| E3-GLiNER-VLM | GLiNER · VLM association | 0.4446 |
| E4 | end-to-end Gemma 3 4B | 0.3597 |

Headline findings:

- **OCR is the single largest lever.** Switching EasyOCR → PaddleOCR adds
  +24.6 pp 4F-F1, almost entirely in quantity (+31.4 pp) and unit (+29.5 pp);
  nutrient detection is nearly flat (+4.7 pp).
- **Geometric row edges (`cy`) beat structural rank.** Same-line pairing is
  robust to OCR drops, fusions and stray NRV%/footnote tokens; ordinal
  `role_rank` breaks when a column's cardinality shifts by one token.
- **Late fusion of `cy` + `role_rank` hurts** (~97 % false positives on the
  role_rank-unique additions), so the union is not used.

The full ablation grid (graph edge-mode sweep, threshold calibration, etc.) is
produced in `ablations.ipynb` and `ablation_experiments_notebook.ipynb`, and
written up in Chapter 6 of the thesis.

---

## Reproducibility & notes

- Reported numbers use `--no-llm`: the evaluator makes **zero LLM calls** and is
  byte-for-byte reproducible across runs and machines.
- Classifier thresholds (`τ*=0.18` for GLiNER, `t=0.59 / m=0.04` for BGE-M3) are
  this project's own calibration results on its data — not the models' published
  benchmark scores.
- Version control uses a single `origin` with two push URLs (GitHub + THD GitLab);
  `git push origin main` pushes to both. HTTPS only.
- **AI assistance disclosure:** AI tools were used during development with
  institutional declaration, as permitted by THD; the GitLab commit history serves
  as the timestamped authenticity record.

---

## Citation

```bibtex
@mastersthesis{hamada2026zeroshot,
  author = {Moustafa Hamada},
  title  = {Zero-Shot Nutrient and Quantity Association using OCR-Guided Semantic Graph Matching},
  school = {Deggendorf Institute of Technology and University of South Bohemia},
  year   = {2026}
}
```
