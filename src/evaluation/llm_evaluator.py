"""
llm_evaluator.py  —  LLM-Assisted Tuple Evaluator
Zero-Shot Nutrient Extraction | Moustafa Hamada | THD + USB

DUAL FULL-MATCH REPORTING
--------------------------
  full_match_3f  (3-field) : nutrient + quantity + unit — context ignored
  full_match_4f  (4-field) : all four fields — primary thesis metric
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

MODEL         = "qwen2.5:7b"
OLLAMA_URL    = "http://localhost:11434"
TEMPERATURE   = 0.0
LLM_TIMEOUT_S = 30

SYSTEM_PROMPT = (
    "You are a precise evaluator for a nutritional supplement label extraction pipeline. "
    "Your task is to judge whether a predicted nutritional tuple matches a ground truth tuple. "
    "Respond with ONLY a valid JSON object -- no preamble, no explanation, no markdown."
)

UNIT_ALIASES: Dict[str, str] = {
    "µg": "µg", "ug": "µg", "mcg": "µg", "μg": "µg",
    "µg re": "µg re", "µg ne": "µg ne", "µg te": "µg te",
    "mg": "mg", "g": "g", "kg": "kg",
    "kj": "kj", "kcal": "kcal", "cal": "kcal",
    "ie": "iu", "iu": "iu",
    "mg/tag": "mg/tag", "mg/day": "mg/tag",
    "µg/tag": "µg/tag", "µg/day": "µg/tag",
}

CONTEXT_ALIASES: Dict[str, str] = {
    "per_100g": "per_100g", "per 100g": "per_100g", "per 100 g": "per_100g",
    "je 100g": "per_100g", "pro 100g": "per_100g", "100g": "per_100g",
    "per_100ml": "per_100ml", "per 100ml": "per_100ml",
    "per_serving": "per_serving", "per serving": "per_serving",
    "per portion": "per_serving", "pro portion": "per_serving",
    "per_daily_dose": "per_daily_dose", "per daily dose": "per_daily_dose",
    "je tagesdosis": "per_daily_dose", "pro tagesdosis": "per_daily_dose",
}


def _norm_str(v: str) -> str:
    return str(v or "").strip().lower()

def _fold_german(s: str) -> str:
    """Fold German special characters for matching: ß→ss, ä→a, ö→o, ü→u."""
    return (s.replace("ß", "ss")
             .replace("ä", "a").replace("ö", "o").replace("ü", "u")
             .replace("é", "e").replace("è", "e")
             .replace("á", "a").replace("à", "a")
             .replace("ł", "l").replace("ř", "r").replace("í", "i")
             .replace("ú", "u").replace("ů", "u").replace("ý", "y"))

def _norm_qty(v) -> Optional[float]:
    s = str(v or "").strip().replace(",", ".").lstrip("<>~")
    try:
        return float(s)
    except ValueError:
        return None

def _norm_unit(v: str) -> str:
    return UNIT_ALIASES.get(_norm_str(v), _norm_str(v))

def _norm_context(v: str) -> str:
    return CONTEXT_ALIASES.get(_norm_str(v), _norm_str(v))

def _fast_nutrient(gt: str, pred: str) -> Optional[bool]:
    g, p = _norm_str(gt), _norm_str(pred)
    if g == p: return True
    # Fold German/accented chars for matching
    gf, pf = _fold_german(g), _fold_german(p)
    if gf == pf: return True
    g2 = re.sub(r"[-\u2013]\s*", "", gf).strip()
    p2 = re.sub(r"[-\u2013]\s*", "", pf).strip()
    if g2 and p2 and g2 == p2: return True
    if gf in pf or pf in gf:
        shorter = min(len(gf), len(pf))
        longer  = max(len(gf), len(pf))
        if longer > 0 and shorter / longer >= 0.65:
            return True
    return None

def _fast_quantity(gt: str, pred: str) -> bool:
    g, p = _norm_qty(gt), _norm_qty(pred)
    if g is None or p is None:
        return _norm_str(gt) == _norm_str(pred)
    return abs(g - p) < 1e-9

def _fast_unit(gt: str, pred: str) -> bool:
    return _norm_unit(gt) == _norm_unit(pred)

def _strip_serving(ctx: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(ctx or "").strip()).strip()

def _fast_context(gt: str, pred: str) -> bool:
    return _norm_context(_strip_serving(gt)) == _norm_context(_strip_serving(pred))


def _build_prompt(gt_nutrient, pred_nutrient, gt_quantity, pred_quantity,
                  gt_unit, pred_unit, gt_context, pred_context) -> str:
    return f"""Compare this ground truth nutritional tuple against the predicted tuple.
Apply every rule below strictly and in order.

GROUND TRUTH:
  nutrient : {gt_nutrient!r}
  quantity : {gt_quantity!r}
  unit     : {gt_unit!r}
  context  : {gt_context!r}

PREDICTION:
  nutrient : {pred_nutrient!r}
  quantity : {pred_quantity!r}
  unit     : {pred_unit!r}
  context  : {pred_context!r}

================================
NUTRIENT MATCHING RULES
================================

R1 -- SPECIFICITY: A sub-nutrient NEVER matches its parent.
  false: "Fett" vs "Fettsaeuren", "Magnesium" vs "Magnesiumoxid"
  true:  "Brennwert" vs "Energie", "Vitamin C" vs "VitaminC"

R2 -- OCR NOISE: strip spaces, normalise ß=ss, dash-prefixes, colon suffixes.

R3 -- MULTILINGUAL EQUIVALENCES:
  Energie=Energy=Brennwert | Eiweiss=Protein=Eiwitten | Fett=fat=Lipides=Vetten
  Kohlenhydrate=carbohydrates | Zucker=sugars | Salz=salt
  Ballaststoffe=dietary fibre | "davon gesaettigte Fettsaeuren"="of which saturates"

R4 -- PARTIAL MATCH: overlap >= 0.65 AND R1 not violated.

R5 -- GARBAGE: unrecognisable OCR string -> false.

================================
QUANTITY MATCHING RULES
================================

R6 -- TRACE: 0=0.0="<0.1"="<0.1" are traces; 0.5 and 7.0 are NOT traces.
R7 -- NUMERIC: float comparison, tolerance +/-0.05. "1.4" vs "14" -> false.
R8 -- MISSING: pred empty & GT nonzero -> false. GT zero/trace & pred empty -> true.

================================
UNIT MATCHING RULES
================================

R10 -- ALIASES: ug=mcg=µg; kJ != kcal; I.E.=IU; KBE=CFU; ml!=g.
R11 -- MISSING unit -> false.

================================
CONTEXT MATCHING RULES
================================

R12 -- STRIP serving-size suffix before comparing.
R13 -- per_100g=je100g=pro100g | per_serving=pro Portion=je Kapsel | per_daily_dose=je Tagesdosis
R14 -- per_100g != per_serving != per_daily_dose. None/empty context -> false.

================================
OUTPUT — ONLY this JSON:
================================

{{
  "nutrient_match": true or false,
  "quantity_match": true or false,
  "unit_match":     true or false,
  "context_match":  true or false,
  "nutrient_reason": "one sentence",
  "quantity_reason": "one sentence",
  "unit_reason":     "one sentence",
  "context_reason":  "one sentence"
}}"""


_LLM_PARSE_FAILURES = 0

def _call_ollama(prompt: str) -> Optional[Dict]:
    global _LLM_PARSE_FAILURES
    try:
        import ollama as _ollama
        response = _ollama.generate(
            model   = MODEL,
            system  = SYSTEM_PROMPT,
            prompt  = prompt,
            options = {"temperature": 0, "num_predict": 300, "stop": ["\n\n\n"]},
        )
        raw = response["response"].strip()

        # Strip <think>...</think> blocks (qwen3 thinking mode)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        # Strip markdown code fences
        raw = re.sub(r'```(?:json)?\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw).strip()

        # Extract JSON object
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if not m:
            # Fallback: try greedy match
            m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            _LLM_PARSE_FAILURES += 1
            if _LLM_PARSE_FAILURES <= 3:
                print(f"    [LLM] No JSON found in response: {raw[:200]!r}")
            return None

        try:
            result = json.loads(m.group())
        except json.JSONDecodeError:
            # Try fixing common issues: trailing commas, single quotes
            cleaned = m.group()
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                _LLM_PARSE_FAILURES += 1
                if _LLM_PARSE_FAILURES <= 3:
                    print(f"    [LLM] JSON parse failed: {m.group()[:200]!r}")
                return None

        return result
    except Exception as e:
        print(f"    [LLM] Ollama call failed: {e}")
        return None


class FieldEvaluator:
    def __init__(self, use_llm: bool = True):
        self.use_llm   = use_llm
        self.llm_calls = 0
        self.fast_hits = 0

    def evaluate(self, gt: Dict, pred: Dict) -> Dict:
        gt_n, gt_q, gt_u, gt_c = (gt.get(k,"") for k in ("nutrient","quantity","unit","context"))
        pr_n, pr_q, pr_u, pr_c = (pred.get(k,"") for k in ("nutrient","quantity","unit","context"))

        fast_n = _fast_nutrient(gt_n, pr_n)
        fast_q = _fast_quantity(gt_q, pr_q)
        fast_u = _fast_unit(gt_u, pr_u)
        fast_c = _fast_context(gt_c, pr_c)

        if fast_n is True and fast_q and fast_u and fast_c:
            self.fast_hits += 1
            return {
                "nutrient_match": True,  "quantity_match": True,
                "unit_match":     True,  "context_match":  True,
                "full_match": True, "method": "fast",
                "nutrient_reason": "exact/substring match",
                "quantity_reason": "numeric match",
                "unit_reason":     "alias match",
                "context_reason":  "alias match",
            }

        llm_result = None
        if self.use_llm and fast_n is None:
            llm_result = _call_ollama(
                _build_prompt(gt_n, pr_n, gt_q, pr_q, gt_u, pr_u, gt_c, pr_c))
            self.llm_calls += 1

        if llm_result:
            n_match  = bool(llm_result.get("nutrient_match", False))
            q_match  = fast_q or bool(llm_result.get("quantity_match", False))
            u_match  = fast_u or bool(llm_result.get("unit_match",     False))
            c_match  = fast_c or bool(llm_result.get("context_match",  False))
            method   = "fast+llm"
            n_reason = llm_result.get("nutrient_reason", "")
            q_reason = llm_result.get("quantity_reason", "")
            u_reason = llm_result.get("unit_reason",     "")
            c_reason = llm_result.get("context_reason",  "")
        else:
            n_match, q_match, u_match, c_match = fast_n is True, fast_q, fast_u, fast_c
            method   = "fast"
            n_reason = "substring match" if n_match else "no match"
            q_reason = "numeric match"   if q_match else "mismatch"
            u_reason = "alias match"     if u_match else "mismatch"
            c_reason = "alias match"     if c_match else "mismatch"

        return {
            "nutrient_match": n_match, "quantity_match": q_match,
            "unit_match":     u_match, "context_match":  c_match,
            "full_match": n_match and q_match and u_match and c_match,
            "method": method,
            "nutrient_reason": n_reason, "quantity_reason": q_reason,
            "unit_reason": u_reason,     "context_reason":  c_reason,
        }


class LLMTupleEvaluator:
    """
    Dual full-match reporting:
      full_match_3f — nutrient + qty + unit (context ignored)
      full_match_4f — all four fields (primary thesis metric)
    """

    def __init__(self, gt_csv: str = "data/annotations/gold_annotations_4field.csv",
                 gt_rows: list = None, use_llm: bool = True, model: str = MODEL):
        self.gt_csv        = gt_csv
        self._gt_preloaded = gt_rows
        self.use_llm       = use_llm
        self.model         = model

    def _load_gt(self) -> List[Dict]:
        def _ctx(row):
            ctx = str(row.get("context","") or "").strip()
            srv = str(row.get("serving_size","") or "").strip()
            if srv and srv.lower() not in ("","none","nan") and "(" not in ctx:
                ctx = f"{ctx} ({srv})"
            return ctx

        if self._gt_preloaded is not None:
            return [{"image_id": str(r.get("image_id","")).strip(),
                     "nutrient": str(r.get("nutrient","")).strip(),
                     "quantity": str(r.get("quantity","")).strip(),
                     "unit":     str(r.get("unit","")).strip(),
                     "context":  _ctx(r)} for r in self._gt_preloaded]

        rows = []
        with open(self.gt_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({"image_id": row.get("image_id","").strip(),
                              "nutrient": row.get("nutrient","").strip(),
                              "quantity": row.get("quantity","").strip(),
                              "unit":     row.get("unit","").strip(),
                              "context":  _ctx(row)})
        return rows

    def run(self, predictions: List[Dict], experiment: str, out_dir: Path,
            notes: str = "", diagnostics: dict = None) -> Dict:

        out_dir   = Path(out_dir)
        gt_rows   = self._load_gt()
        evaluator = FieldEvaluator(use_llm=self.use_llm)
        global _LLM_PARSE_FAILURES
        _LLM_PARSE_FAILURES = 0

        gt_by_img: Dict[str, List]   = {}
        pred_by_img: Dict[str, List] = {}
        for row in gt_rows:
            gt_by_img.setdefault(row["image_id"], []).append(row)
        for t in predictions:
            pred_by_img.setdefault(t["image_id"], []).append(t)

        all_images = sorted(set(list(gt_by_img) + list(pred_by_img)))
        total_gt   = len(gt_rows)
        total_pred = len(predictions)

        tp = fp = fn = 0
        full_correct_3f = full_correct_4f = 0
        per_image: List[Dict] = []
        pair_rows: List[Dict] = []
        analysis_rows: List[Dict] = []
        detail_rows: List[Dict]   = []
        t_start = time.time()

        for img in all_images:
            gt_list   = gt_by_img.get(img, [])
            pred_list = pred_by_img.get(img, [])
            matched = qty_m = unit_m = ctx_m = full_m_3f = full_m_4f = 0

            # ── PASS 1: Fast N×M — lock perfect matches immediately ───
            # Compare every GT against every pred using fast checks only.
            # If all 4 fields match → lock both, remove from pool.
            # Zero LLM calls in this pass.
            used_gt:   set = set()
            used_pred: set = set()
            assignments: list = []  # (gt_idx, pred_idx)

            # Build all fast-match candidates with scores
            fast_candidates = []  # (score, g_idx, p_idx)
            for g_idx, gt in enumerate(gt_list):
                for p_idx, pred in enumerate(pred_list):
                    fn_result = _fast_nutrient(gt["nutrient"], pred.get("nutrient",""))
                    if fn_result is not True:
                        continue
                    fq = _fast_quantity(gt.get("quantity",""), pred.get("quantity",""))
                    fu = _fast_unit(gt.get("unit",""), pred.get("unit",""))
                    fc = _fast_context(gt.get("context",""), pred.get("context",""))
                    if fq and fu and fc:
                        # Perfect 4-field fast match — compute score for ordering
                        qty_bonus  = 1.0
                        unit_bonus = 0.5
                        fast_candidates.append((5.5, g_idx, p_idx))

            # Lock perfect matches in score-descending order
            fast_candidates.sort(key=lambda x: -x[0])
            for score, g_idx, p_idx in fast_candidates:
                if g_idx in used_gt or p_idx in used_pred:
                    continue
                used_gt.add(g_idx)
                used_pred.add(p_idx)
                assignments.append((g_idx, p_idx))

            # ── PASS 2: N×M comparison on remaining unlocked tuples ───
            # Only tuples not locked in Pass 1 enter this comparison.
            # LLM calls happen only for these harder cases.
            score_pairs = []  # (score, gt_idx, pred_idx)
            for g_idx, gt in enumerate(gt_list):
                if g_idx in used_gt:
                    continue
                for p_idx, pred in enumerate(pred_list):
                    if p_idx in used_pred:
                        continue
                    g = _fold_german(_norm_str(gt["nutrient"]))
                    p = _fold_german(_norm_str(pred.get("nutrient","")))

                    # name_score : 2.0 = exact  1.0 = substring  0.0 = no overlap
                    # Also check with dashes stripped
                    g2 = re.sub(r"[-\u2013]\s*", "", g).strip()
                    p2 = re.sub(r"[-\u2013]\s*", "", p).strip()

                    if g == p or (g2 and p2 and g2 == p2):
                        name_score = 2.0
                    elif g in p or p in g:
                        name_score = 1.0
                    else:
                        name_score = 0.0

                    # Segment matching for multilingual labels (/ and | separators)
                    if name_score == 0.0 and ('/' in g or '/' in p or '|' in g or '|' in p):
                        g_segs = [s.strip() for s in re.split(r'[/|]', g) if len(s.strip()) > 2]
                        p_segs = [s.strip() for s in re.split(r'[/|]', p) if len(s.strip()) > 2]
                        for gs in g_segs:
                            for ps in p_segs:
                                if gs == ps or (len(gs) > 3 and len(ps) > 3 and (gs in ps or ps in gs)):
                                    name_score = 1.0
                                    break
                            if name_score > 0:
                                break
                            if len(gs) > 3 and gs in p:
                                name_score = 1.0
                                break

                    qty_bonus  = 1.0 if _fast_quantity(
                                     gt.get("quantity",""), pred.get("quantity","")) else 0.0
                    unit_bonus = 0.5 if _fast_unit(
                                     gt.get("unit",""), pred.get("unit","")) else 0.0
                    combined   = name_score * 2 + qty_bonus + unit_bonus

                    if combined >= 2.0:  # at least substring name match
                        score_pairs.append((combined, g_idx, p_idx))

            # Assign remaining matches in score-descending order
            score_pairs.sort(key=lambda x: -x[0])
            for score, g_idx, p_idx in score_pairs:
                if g_idx in used_gt or p_idx in used_pred:
                    continue
                used_gt.add(g_idx)
                used_pred.add(p_idx)
                assignments.append((g_idx, p_idx))

            # ── Process all matched pairs (Pass 1 + Pass 2) ──────────
            for g_idx, p_idx in assignments:
                gt   = gt_list[g_idx]
                pred = pred_list[p_idx]
                tp += 1; matched += 1
                result = evaluator.evaluate(gt, pred)

                is_3f = result["nutrient_match"] and result["quantity_match"] and result["unit_match"]
                is_4f = is_3f and result["context_match"]

                if result["quantity_match"]: qty_m  += 1
                if result["unit_match"]:     unit_m += 1
                if result["context_match"]:  ctx_m  += 1
                if is_3f: full_m_3f += 1; full_correct_3f += 1
                if is_4f: full_m_4f += 1; full_correct_4f += 1

                pair_rows.append({
                    "image_id": img,
                    "gt_nutrient": gt["nutrient"],   "pred_nutrient": pred.get("nutrient",""),
                    "gt_qty": gt["quantity"],         "pred_qty": pred.get("quantity",""),
                    "gt_unit": gt["unit"],            "pred_unit": pred.get("unit",""),
                    "gt_context": gt["context"],      "pred_context": pred.get("context",""),
                    "qty_match": result["quantity_match"],
                    "unit_match": result["unit_match"],
                    "ctx_match": result["context_match"],
                    "full_match_3f": is_3f, "full_match_4f": is_4f,
                })
                detail_rows.append({
                    "image_id": img,
                    "gt_nutrient": gt["nutrient"],   "pred_nutrient": pred.get("nutrient",""),
                    "gt_quantity": gt["quantity"],    "pred_quantity": pred.get("quantity",""),
                    "gt_unit": gt["unit"],            "pred_unit": pred.get("unit",""),
                    "gt_context": gt["context"],      "pred_context": pred.get("context",""),
                    "nutrient_match": result["nutrient_match"],
                    "quantity_match": result["quantity_match"],
                    "unit_match":     result["unit_match"],
                    "context_match":  result["context_match"],
                    "full_match_3f":  is_3f, "full_match_4f": is_4f,
                    "eval_method":    result["method"],
                    "nutrient_reason": result["nutrient_reason"],
                    "quantity_reason": result["quantity_reason"],
                    "unit_reason":     result["unit_reason"],
                    "context_reason":  result["context_reason"],
                })

                if not is_4f:
                    reasons = []
                    if not result["nutrient_match"]:
                        reasons.append(f"nutrient: {result['nutrient_reason']}")
                    if not result["quantity_match"]:
                        reasons.append(f"qty: got {pred.get('quantity','')} expected {gt['quantity']}")
                    if not result["unit_match"]:
                        reasons.append(f"unit: got {pred.get('unit','')} expected {gt['unit']}")
                    if not result["context_match"]:
                        reasons.append(f"context: got {pred.get('context','')} expected {gt['context']}")
                    analysis_rows.append({
                        "tag": "incorrect", "image_id": img,
                        "gt_nutrient": gt["nutrient"], "gt_quantity": gt["quantity"],
                        "gt_unit": gt["unit"], "gt_context": gt["context"],
                        "pred_nutrient": pred.get("nutrient",""),
                        "pred_quantity": pred.get("quantity",""),
                        "pred_unit": pred.get("unit",""),
                        "pred_context": pred.get("context",""),
                        "reason": " | ".join(reasons),
                    })

            # ── Unmatched GT tuples (false negatives) ────────────────
            for g_idx, gt in enumerate(gt_list):
                if g_idx not in used_gt:
                    fn += 1
                    analysis_rows.append({
                        "tag": "missing", "image_id": img,
                        "gt_nutrient": gt["nutrient"], "gt_quantity": gt["quantity"],
                        "gt_unit": gt["unit"], "gt_context": gt["context"],
                        "pred_nutrient": "", "pred_quantity": "",
                        "pred_unit": "", "pred_context": "",
                        "reason": "nutrient not found in predictions",
                    })

            fp += len(pred_list) - len(used_pred)
            nm = matched
            per_image.append({
                "image_id": img, "gt": len(gt_list), "pred": len(pred_list), "match": nm,
                "qty_pct":    round(qty_m     / nm * 100) if nm else 0,
                "unit_pct":   round(unit_m    / nm * 100) if nm else 0,
                "ctx_pct":    round(ctx_m     / nm * 100) if nm else 0,
                "full3f_pct": round(full_m_3f / nm * 100) if nm else 0,
                "full4f_pct": round(full_m_4f / nm * 100) if nm else 0,
            })

        elapsed = round(time.time() - t_start, 1)
        np_ = len(pair_rows)

        nutr_prec = tp / (tp + fp) if (tp + fp) else 0
        nutr_rec  = tp / (tp + fn) if (tp + fn) else 0
        nutr_f1   = 2*nutr_prec*nutr_rec / (nutr_prec+nutr_rec) if (nutr_prec+nutr_rec) else 0
        qty_acc   = sum(1 for r in pair_rows if r["qty_match"])  / max(np_, 1)
        unit_acc  = sum(1 for r in pair_rows if r["unit_match"]) / max(np_, 1)
        ctx_acc   = sum(1 for r in pair_rows if r["ctx_match"])  / max(np_, 1)

        full3f_prec = full_correct_3f / total_pred if total_pred else 0
        full3f_rec  = full_correct_3f / total_gt   if total_gt   else 0
        full3f_f1   = 2*full3f_prec*full3f_rec / (full3f_prec+full3f_rec) if (full3f_prec+full3f_rec) else 0

        full4f_prec = full_correct_4f / total_pred if total_pred else 0
        full4f_rec  = full_correct_4f / total_gt   if total_gt   else 0
        full4f_f1   = 2*full4f_prec*full4f_rec / (full4f_prec+full4f_rec) if (full4f_prec+full4f_rec) else 0

        ctx_cost_f1 = round((full3f_f1 - full4f_f1) * 100, 1)

        W = 65
        print(f"{'='*W}")
        print(f"  EXPERIMENT RESULTS : {experiment}")
        print(f"  EVALUATOR          : LLM-Assisted ({self.model} via Ollama)")
        print(f"{'='*W}")
        print(f"  Images evaluated   : {len(all_images)}")
        print(f"  GT tuples          : {total_gt}")
        print(f"  Predicted tuples   : {total_pred}")
        print(f"  Matched pairs      : {np_}")
        print(f"{'─'*W}")
        print(f"  Nutrient Precision : {nutr_prec:.3f}")
        print(f"  Nutrient Recall    : {nutr_rec:.3f}")
        print(f"  Nutrient F1        : {nutr_f1:.3f}")
        print(f"{'─'*W}")
        print(f"  Quantity Match Acc : {qty_acc*100:.1f}%")
        print(f"  Unit Match Acc     : {unit_acc*100:.1f}%")
        print(f"  Context Match Acc  : {ctx_acc*100:.1f}%")
        print(f"{'─'*W}")
        print(f"  3-FIELD (no context) — nutrient + qty + unit")
        print(f"  ★ Full Tuple Precision : {full3f_prec*100:.1f}%")
        print(f"  ★ Full Tuple Recall    : {full3f_rec*100:.1f}%")
        print(f"  ★ Full Tuple F1        : {full3f_f1*100:.1f}%")
        print(f"{'─'*W}")
        print(f"  4-FIELD (with context) — primary thesis metric")
        print(f"  ★ Full Tuple Precision : {full4f_prec*100:.1f}%")
        print(f"  ★ Full Tuple Recall    : {full4f_rec*100:.1f}%")
        print(f"  ★ Full Tuple F1        : {full4f_f1*100:.1f}%")
        print(f"{'─'*W}")
        print(f"  Context cost (3f - 4f F1) : -{ctx_cost_f1}pp")
        print(f"  LLM calls made     : {evaluator.llm_calls}")
        print(f"  Fast-pass hits     : {evaluator.fast_hits}")
        print(f"  LLM parse failures : {_LLM_PARSE_FAILURES}")
        print(f"  Total eval time    : {elapsed}s")
        print(f"{'='*W}")
        print()
        print(f"  {'IMAGE':<17} {'GT':>4}  {'PRED':>5}  {'MATCH':>6}  "
              f"{'QTY%':>5}  {'UNIT%':>6}  {'CTX%':>5}  {'3F%':>5}  {'4F%':>5}")
        print(f"  {'─'*W}")
        for row in per_image:
            print(f"  {row['image_id']:<17} {row['gt']:>4}  {row['pred']:>5}  "
                  f"{row['match']:>6}  {row['qty_pct']:>4}%  "
                  f"{row['unit_pct']:>5}%  {row['ctx_pct']:>4}%  "
                  f"{row['full3f_pct']:>4}%  {row['full4f_pct']:>4}%")

        metrics = {
            "experiment": experiment, "timestamp": datetime.now().isoformat(),
            "notes": notes, "evaluator": f"LLM-Assisted ({self.model})",
            "schema": "nutrient|quantity|unit|context",
            "images_evaluated": len(all_images),
            "gt_tuples": total_gt, "predicted_tuples": total_pred, "matched_pairs": np_,
            "nutrient_precision": round(nutr_prec,4), "nutrient_recall": round(nutr_rec,4),
            "nutrient_f1":        round(nutr_f1,  4),
            "quantity_acc":       round(qty_acc,  4),
            "unit_acc":           round(unit_acc, 4),
            "context_acc":        round(ctx_acc,  4),
            "full3f_correct":     full_correct_3f,
            "full3f_precision":   round(full3f_prec,4),
            "full3f_recall":      round(full3f_rec, 4),
            "full3f_f1":          round(full3f_f1,  4),
            "full4f_correct":     full_correct_4f,
            "full_tuple_precision": round(full4f_prec,4),
            "full_tuple_recall":    round(full4f_rec, 4),
            "full_tuple_f1":        round(full4f_f1,  4),
            "context_cost_pp":    ctx_cost_f1,
            "llm_calls":          evaluator.llm_calls,
            "fast_hits":          evaluator.fast_hits,
            "eval_time_s":        elapsed,
        }

        metrics_json = out_dir / "evaluation_results.json"
        summary_csv  = out_dir / "evaluation_summary.csv"
        per_img_csv  = out_dir / "per_image_results.csv"
        pairs_csv    = out_dir / "pair_details.csv"
        analysis_csv = out_dir / f"{experiment}_analysis.csv"
        detail_csv   = out_dir / "llm_evaluation_details.csv"

        with open(metrics_json,"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2,ensure_ascii=False)
        with open(summary_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f,fieldnames=list(metrics.keys())); w.writeheader(); w.writerow(metrics)
        with open(per_img_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f,fieldnames=["image_id","gt","pred","match","qty_pct","unit_pct","ctx_pct","full3f_pct","full4f_pct"])
            w.writeheader(); w.writerows(per_image)
        with open(pairs_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f,fieldnames=["image_id","gt_nutrient","pred_nutrient","gt_qty","pred_qty","gt_unit","pred_unit","gt_context","pred_context","qty_match","unit_match","ctx_match","full_match_3f","full_match_4f"])
            w.writeheader(); w.writerows(pair_rows)
        with open(analysis_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f,fieldnames=["tag","image_id","gt_nutrient","gt_quantity","gt_unit","gt_context","pred_nutrient","pred_quantity","pred_unit","pred_context","reason"])
            w.writeheader(); w.writerows(analysis_rows)
        with open(detail_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f,fieldnames=["image_id","gt_nutrient","pred_nutrient","gt_quantity","pred_quantity","gt_unit","pred_unit","gt_context","pred_context","nutrient_match","quantity_match","unit_match","context_match","full_match_3f","full_match_4f","eval_method","nutrient_reason","quantity_reason","unit_reason","context_reason"])
            w.writeheader(); w.writerows(detail_rows)

        print(f"\n  Saved:")
        for p in [metrics_json,summary_csv,per_img_csv,pairs_csv,analysis_csv,detail_csv]:
            print(f"    {p}")

        return metrics