"""
sweep_threshold_margin.py
=========================
Phase 2 of embedding-classifier threshold/margin tuning.

INPUT
-----
CSV from collect_embedding_scores.py with one row per token that
reached the embedding pathway:
    nutrient_score, second_best_score, margin_actual, gt_label, ...

DECISION RULE
-------------
Predict NUTRIENT iff
    nutrient_score      >= threshold
    margin_actual       >= margin    (= nutrient_score - second_best_score)

OUTPUT
------
outputs/threshold_sweep/
    sweep_results.csv          full grid of (t, m) → P/R/F1/TP/FP/FN
    plots/score_distribution.png    nutrient_score histogram by gt_label
    plots/margin_distribution.png   margin_actual histogram by gt_label
    plots/threshold_sweep.png       counts + P/R/F1 vs threshold (m fixed)
    plots/margin_sweep.png          counts + P/R/F1 vs margin (t fixed)
    plots/heatmap_f1.png            F1 across full (t, m) grid
    plots/pr_curve.png              precision/recall curve (margin=0)
    summary.txt                     best operating point + comparison

USAGE
-----
    python src/analysis/sweep_threshold_margin.py \
        --scores-csv outputs/threshold_sweep/embedding_scores_embedding_only.csv

    # Override the fixed-axis defaults for the 1D sweeps:
    python src/analysis/sweep_threshold_margin.py \
        --scores-csv outputs/threshold_sweep/embedding_scores_hybrid.csv \
        --fix-margin 0.05 --fix-threshold 0.45
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Core sweep ────────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, threshold: float, margin: float) -> dict:
    """Return TP/FP/FN/TN/P/R/F1 for a single (t, m)."""
    pred = (df["nutrient_score"].values >= threshold) & \
           (df["margin_actual"].values  >= margin)
    gt = df["gt_label"].values.astype(bool)

    tp = int(( pred &  gt).sum())
    fp = int(( pred & ~gt).sum())
    fn = int((~pred &  gt).sum())
    tn = int((~pred & ~gt).sum())

    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r / (p + r) if (p + r) else 0.0
    return {"threshold": threshold, "margin": margin,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": p, "recall": r, "f1": f1}


def sweep_grid(df: pd.DataFrame, thresholds: np.ndarray,
               margins: np.ndarray) -> pd.DataFrame:
    rows = [evaluate(df, t, m) for t in thresholds for m in margins]
    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame, path: Path):
    pos = df.loc[df.gt_label == 1, "nutrient_score"]
    neg = df.loc[df.gt_label == 0, "nutrient_score"]
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(neg, bins=bins, alpha=0.55, label=f"non-nutrient (n={len(neg)})",
            color="#d62728")
    ax.hist(pos, bins=bins, alpha=0.55, label=f"nutrient (n={len(pos)})",
            color="#2ca02c")
    ax.axvline(0.40, ls="--", color="gray", lw=1, label="default t=0.40")
    ax.set_xlabel("NUTRIENT cosine similarity")
    ax.set_ylabel("count")
    ax.set_title("Distribution of NUTRIENT score by GT label")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_margin_distribution(df: pd.DataFrame, path: Path):
    pos = df.loc[df.gt_label == 1, "margin_actual"]
    neg = df.loc[df.gt_label == 0, "margin_actual"]
    lo, hi = float(df.margin_actual.min()), float(df.margin_actual.max())
    bins = np.linspace(min(lo, -0.2), max(hi, 0.4), 41)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(neg, bins=bins, alpha=0.55, label=f"non-nutrient (n={len(neg)})",
            color="#d62728")
    ax.hist(pos, bins=bins, alpha=0.55, label=f"nutrient (n={len(pos)})",
            color="#2ca02c")
    ax.axvline(0.10, ls="--", color="gray", lw=1, label="default m=0.10")
    ax.axvline(0.00, ls=":",  color="black", lw=0.7)
    ax.set_xlabel("margin = NUTRIENT − second-best category")
    ax.set_ylabel("count")
    ax.set_title("Distribution of margin by GT label")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_threshold_sweep(grid: pd.DataFrame, fix_margin: float, path: Path):
    sub = grid[np.isclose(grid.margin, fix_margin)].sort_values("threshold")
    if sub.empty:
        print(f"[warn] no rows for margin={fix_margin}; skipping threshold sweep plot")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sub.threshold, sub.TP, label="TP (correct nutrient)",  color="#2ca02c", lw=2)
    ax1.plot(sub.threshold, sub.FP, label="FP (false nutrient)",     color="#d62728", lw=2)
    ax1.plot(sub.threshold, sub.FN, label="FN (missed nutrient)",    color="#ff7f0e", lw=2)
    ax1.set_xlabel("nutrient_threshold")
    ax1.set_ylabel("token count")
    ax1.set_title(f"Counts vs threshold  (margin={fix_margin})")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(sub.threshold, sub.precision, label="precision", color="#1f77b4", lw=2)
    ax2.plot(sub.threshold, sub.recall,    label="recall",    color="#9467bd", lw=2)
    ax2.plot(sub.threshold, sub.f1,        label="F1",        color="#000000", lw=2.5)
    best_t = sub.loc[sub.f1.idxmax(), "threshold"]
    ax2.axvline(best_t, ls="--", color="gray", lw=1,
                label=f"best F1 @ t={best_t:.2f}")
    ax2.set_xlabel("nutrient_threshold")
    ax2.set_ylabel("score")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"P / R / F1 vs threshold  (margin={fix_margin})")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_margin_sweep(grid: pd.DataFrame, fix_threshold: float, path: Path):
    sub = grid[np.isclose(grid.threshold, fix_threshold)].sort_values("margin")
    if sub.empty:
        print(f"[warn] no rows for threshold={fix_threshold}; skipping margin sweep plot")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sub.margin, sub.TP, label="TP", color="#2ca02c", lw=2)
    ax1.plot(sub.margin, sub.FP, label="FP", color="#d62728", lw=2)
    ax1.plot(sub.margin, sub.FN, label="FN", color="#ff7f0e", lw=2)
    ax1.set_xlabel("margin")
    ax1.set_ylabel("token count")
    ax1.set_title(f"Counts vs margin  (threshold={fix_threshold})")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(sub.margin, sub.precision, label="precision", color="#1f77b4", lw=2)
    ax2.plot(sub.margin, sub.recall,    label="recall",    color="#9467bd", lw=2)
    ax2.plot(sub.margin, sub.f1,        label="F1",        color="#000000", lw=2.5)
    best_m = sub.loc[sub.f1.idxmax(), "margin"]
    ax2.axvline(best_m, ls="--", color="gray", lw=1,
                label=f"best F1 @ m={best_m:.2f}")
    ax2.set_xlabel("margin")
    ax2.set_ylabel("score")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"P / R / F1 vs margin  (threshold={fix_threshold})")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_heatmap(grid: pd.DataFrame, path: Path):
    pivot = grid.pivot(index="margin", columns="threshold", values="f1")
    pivot = pivot.sort_index(ascending=False)   # high margin at top
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="upper",
                   cmap="viridis",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(),   pivot.index.max()])
    cb = fig.colorbar(im, ax=ax); cb.set_label("F1")

    # Mark best
    best = grid.loc[grid.f1.idxmax()]
    ax.scatter(best.threshold, best.margin, s=180,
               facecolors="none", edgecolors="white", lw=2,
               label=f"best  t={best.threshold:.2f}  m={best.margin:.2f}  "
                     f"F1={best.f1:.3f}")
    # Mark default
    ax.scatter(0.40, 0.10, s=120, marker="x", color="red", lw=2,
               label="default  t=0.40  m=0.10")

    ax.set_xlabel("nutrient_threshold")
    ax.set_ylabel("margin")
    ax.set_title("F1 across (threshold, margin) grid")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pr_curve(df: pd.DataFrame, path: Path):
    """Precision-recall curve sweeping only nutrient_threshold (margin=0)."""
    ts = np.linspace(0.0, 1.0, 101)
    ps, rs, f1s = [], [], []
    for t in ts:
        e = evaluate(df, t, 0.0)
        ps.append(e["precision"]); rs.append(e["recall"]); f1s.append(e["f1"])
    ps, rs, f1s = np.array(ps), np.array(rs), np.array(f1s)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rs, ps, color="#1f77b4", lw=2)
    # Annotate a few thresholds along the curve
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        i = int(round(t * 100))
        if 0 <= i < len(ts):
            ax.scatter(rs[i], ps[i], s=40, color="#1f77b4", zorder=5)
            ax.annotate(f"t={t:.2f}", (rs[i], ps[i]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    best_i = int(np.argmax(f1s))
    ax.scatter(rs[best_i], ps[best_i], s=160, marker="*",
               color="black", zorder=6,
               label=f"best F1={f1s[best_i]:.3f} @ t={ts[best_i]:.2f}")
    ax.set_xlabel("recall");    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02);       ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve (margin=0)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", required=True,
                    help="Path to embedding_scores_*.csv from Phase 1")
    ap.add_argument("--out",  default=None,
                    help="Output dir (default: alongside scores CSV)")
    ap.add_argument("--threshold-min",  type=float, default=0.20)
    ap.add_argument("--threshold-max",  type=float, default=0.65)
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument("--margin-min",     type=float, default=0.00)
    ap.add_argument("--margin-max",     type=float, default=0.25)
    ap.add_argument("--margin-step",    type=float, default=0.01)
    ap.add_argument("--fix-margin",     type=float, default=0.10)
    ap.add_argument("--fix-threshold",  type=float, default=0.40)
    args = ap.parse_args()

    scores_csv = Path(args.scores_csv)
    out_dir = Path(args.out) if args.out else scores_csv.parent
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    print(f"[sweep] loaded {len(df)} rows from {scores_csv}")
    print(f"[sweep] gt_label distribution: "
          f"pos={int(df.gt_label.sum())}  "
          f"neg={int((1 - df.gt_label).sum())}")

    # Round for clean grid join
    thresholds = np.round(np.arange(args.threshold_min,
                                    args.threshold_max + 1e-9,
                                    args.threshold_step), 4)
    margins    = np.round(np.arange(args.margin_min,
                                    args.margin_max + 1e-9,
                                    args.margin_step), 4)

    print(f"[sweep] grid: {len(thresholds)} thresholds × "
          f"{len(margins)} margins = {len(thresholds)*len(margins)} cells")

    grid = sweep_grid(df, thresholds, margins)
    grid_path = out_dir / "sweep_results.csv"
    grid.to_csv(grid_path, index=False)
    print(f"[sweep] saved grid → {grid_path}")

    # ── Plots
    plot_score_distribution (df,   plots_dir / "score_distribution.png")
    plot_margin_distribution(df,   plots_dir / "margin_distribution.png")
    plot_threshold_sweep(grid, args.fix_margin,    plots_dir / "threshold_sweep.png")
    plot_margin_sweep   (grid, args.fix_threshold, plots_dir / "margin_sweep.png")
    plot_heatmap        (grid,                     plots_dir / "heatmap_f1.png")
    plot_pr_curve       (df,                       plots_dir / "pr_curve.png")
    print(f"[sweep] saved 6 plots → {plots_dir}/")

    # ── Summary
    best     = grid.loc[grid.f1.idxmax()]
    default  = evaluate(df, 0.40, 0.10)
    summary = []
    summary.append(f"Source: {scores_csv}")
    summary.append(f"Tokens evaluated: {len(df)}  "
                   f"(positives: {int(df.gt_label.sum())}, "
                   f"negatives: {int((1 - df.gt_label).sum())})")
    summary.append("")
    summary.append("DEFAULT (t=0.40, m=0.10):")
    summary.append(f"  TP={default['TP']:>4d}  FP={default['FP']:>4d}  "
                   f"FN={default['FN']:>4d}  TN={default['TN']:>4d}")
    summary.append(f"  precision={default['precision']:.4f}  "
                   f"recall={default['recall']:.4f}  "
                   f"F1={default['f1']:.4f}")
    summary.append("")
    summary.append(f"BEST (grid search):")
    summary.append(f"  threshold={best.threshold:.3f}   margin={best.margin:.3f}")
    summary.append(f"  TP={int(best.TP):>4d}  FP={int(best.FP):>4d}  "
                   f"FN={int(best.FN):>4d}  TN={int(best.TN):>4d}")
    summary.append(f"  precision={best.precision:.4f}  "
                   f"recall={best.recall:.4f}  "
                   f"F1={best.f1:.4f}")
    summary.append("")
    summary.append(f"Δ vs default:  F1 {best.f1 - default['f1']:+.4f}")
    summary.append("")
    summary.append("Top-10 (t, m) configurations by F1:")
    for _, row in grid.nlargest(10, "f1").iterrows():
        summary.append(
            f"  t={row.threshold:.2f}  m={row.margin:.2f}  "
            f"F1={row.f1:.4f}  P={row.precision:.3f}  R={row.recall:.3f}  "
            f"TP={int(row.TP)}  FP={int(row.FP)}  FN={int(row.FN)}"
        )

    text = "\n".join(summary)
    print("\n" + text)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(f"\n[sweep] saved summary → {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()