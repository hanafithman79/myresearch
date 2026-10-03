# -*- coding: utf-8 -*-
"""
Objective 1 - fair early-stopping comparison of the deep models
===============================================================
Compares the 5x5 repeated cross-validation results of every deep model trained for a
fixed 15 epochs with the same models trained with early stopping (same folds, same seeds,
same rule for all: 15% of the TRAINING fold held out, patience 5 on Macro-F1, up to 50
epochs, best weights restored). Classical models are not affected by early stopping and
serve as fixed reference points.

For each target (E6 = 4 tracks, P2 = Saber Pro top quartile vs below):
  1. per deep model: Macro-F1 and accuracy with fixed epochs vs early stopping, the paired
     difference with a Nadeau-Bengio corrected t-test, and the epochs actually used;
  2. the ranking under early stopping: Proposed minus every other model, corrected p.

Inputs (RESULTS_DIR): objective1_repeated_cv[_P2][_ES]_folds.csv
Outputs (OUT_DIR):    objective1_early_stopping_comparison.csv,
                      objective1_early_stopping_ranking.csv, fig20_early_stopping.png
Usage: python objective1_early_stopping.py [RESULTS_DIR] [OUT_DIR]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import objective1_experiments as base  # noqa: E402
from objective1_repeated_cv import corrected_t  # noqa: E402

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
RATIO = 1 / (base.N_SPLITS - 1)
TARGETS = {"E6": ("Macro-Track", "", "4 macro-tracks"),
           "P2": ("Saber Pro top quartile", "_P2", "Saber Pro top quartile vs below")}
DEEP = [m for m, (cat, kind, _) in base.MODEL_REGISTRY.items() if kind == "torch"]
SHORT = {"Static MLP Only": "Static MLP", "Standalone LSTM": "LSTM",
         "Hybrid 1: MLP + CNN + LSTM": "H1 MLP+CNN+LSTM", "Hybrid 2: MLP + BiLSTM": "H2 MLP+BiLSTM",
         "Hybrid 3: MLP + GRU": "H3 MLP+GRU", "Hybrid 4: MLP + Transformer": "H4 MLP+Transf.",
         base.PROPOSED: "Proposed TCN+LSTM"}


def load(tag, es):
    label, suffix, _ = TARGETS[tag]
    path = os.path.join(RESULTS, f"objective1_repeated_cv{suffix}{'_ES' if es else ''}_folds.csv")
    return pd.read_csv(path).set_index(["Repeat", "Fold", "Model"]), label


def main():
    comp, rank = [], []
    for tag, (_, _, title) in TARGETS.items():
        fixed, label = load(tag, False)
        es, _ = load(tag, True)
        f1, acc = f"{label} Macro-F1", f"{label} Top-1 Acc (%)"
        for m in DEEP:
            a = fixed.xs(m, level="Model")
            b = es.xs(m, level="Model").loc[a.index]
            d = (b[f1] - a[f1]).values
            _, p = corrected_t(d, RATIO)
            comp.append({"Target": title, "Model": m,
                         "Macro-F1 fixed 15 epochs": round(a[f1].mean(), 4),
                         "Macro-F1 early stopping": round(b[f1].mean(), 4),
                         "Change": round(d.mean(), 4), "p (corrected)": f"{p:.3g}",
                         "Accuracy fixed (%)": round(a[acc].mean(), 2),
                         "Accuracy early stopping (%)": round(b[acc].mean(), 2),
                         "Epochs used (mean)": round(b["Epochs"].mean(), 1),
                         "Epochs used (range)": f"{int(b['Epochs'].min())}-{int(b['Epochs'].max())}"})
        prop = es.xs(base.PROPOSED, level="Model")
        for m in base.MODEL_REGISTRY:
            if m == base.PROPOSED:
                continue
            o = es.xs(m, level="Model").loc[prop.index]
            d = (prop[f1] - o[f1]).values
            _, p = corrected_t(d, RATIO)
            rank.append({"Target": title, "Model": m, "Model Macro-F1 (early stopping)": round(o[f1].mean(), 4),
                         "Proposed minus model": round(d.mean(), 4), "p (corrected)": f"{p:.3g}",
                         "Verdict": ("Proposed better" if p < 0.05 and d.mean() > 0 else
                                     "Proposed worse" if p < 0.05 else "No significant difference")})
    comp, rank = pd.DataFrame(comp), pd.DataFrame(rank)
    comp.to_csv(os.path.join(OUT, "objective1_early_stopping_comparison.csv"), index=False)
    rank.to_csv(os.path.join(OUT, "objective1_early_stopping_ranking.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(comp.to_string(index=False))
    print()
    print(rank.to_string(index=False))
    figure(comp)


def figure(comp):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#e4e3df",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, (tag, (label, suffix, title)) in zip(axes, TARGETS.items()):
        c = comp[comp["Target"] == title].set_index("Model").loc[DEEP[::-1]]
        y = np.arange(len(c))
        for yi, (_, r) in zip(y, c.iterrows()):
            ax.annotate("", xy=(r["Macro-F1 early stopping"], yi), xytext=(r["Macro-F1 fixed 15 epochs"], yi),
                        arrowprops={"arrowstyle": "->", "color": "#8a8983", "lw": 1.2})
        ax.scatter(c["Macro-F1 fixed 15 epochs"], y, s=40, color="#b9b8b2", zorder=3,
                   label="Fixed 15 epochs")
        ax.scatter(c["Macro-F1 early stopping"], y, s=48, color="#2a78d6", zorder=4,
                   edgecolors="white", linewidths=1.2, label="Early stopping")
        fixed, _ = load(tag, False)
        lr = fixed.xs("Logistic Regression", level="Model")[f"{label} Macro-F1"].mean()
        ax.axvline(lr, color="#0b0b0b", linestyle="--", linewidth=1.1)
        ax.text(lr, len(c) - 0.45, f" Logistic Regression {lr:.3f}", fontsize=7.5, va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[m] for m in c.index])
        ax.get_yticklabels()[list(c.index).index(base.PROPOSED)].set_fontweight("bold")
        ax.set_title(f"{title}: Macro-F1", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(c) - 0.1)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("Early stopping vs fixed 15 epochs for every deep model (5 x 5 repeated CV means)",
                 x=0.0, ha="left", fontsize=11, fontweight="bold")
    out = os.path.join(OUT, "fig20_early_stopping.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
