# -*- coding: utf-8 -*-
"""
Objective 1 - repeated cross-validation of the main result (4 macro-tracks)
==========================================================================
Final protocol of E6 (4 tracks trained directly, no class weights, Saber Pro inputs
removed), repeated R = 5 times with a different fold split each time (5 x 5 = 25 test
folds per model). Repeat 0 uses the original split and seeds, so it reproduces E6.

Because the 25 folds share training data they are not independent, so intervals and
tests use the Nadeau & Bengio (2003) corrected resampled t statistic:
    var_corrected = (1/J + n_test/n_train) * s^2,   df = J - 1   (J = 25)

Outputs (working directory):
  objective1_repeated_cv_folds.csv    every model x repeat x fold
  objective1_repeated_cv_summary.csv  mean, corrected 95% CI, gain over the majority
                                      class, corrected p-values vs majority and vs Proposed
  fig12_repeated_cv.png               25 fold scores per model, mean and corrected 95% CI
Usage: python objective1_repeated_cv.py [R] [E6|P2]   (default: 5 repeats, 4 macro-tracks)
       python objective1_repeated_cv.py plot [E6|P2]   (redraw the figure from the saved CSVs)
P2 repeats the same procedure for the Saber Pro top-quartile target (Objective 1b).
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import objective1_experiments as base
from objective1_ablations import SABER_PRO_FEATURES

NO_PRO = [f for f in base.TEMPORAL_FEATURES if f not in SABER_PRO_FEATURES]
TARGETS = {
    # tag: (metric label, function building the target from the data, file suffix, title)
    "E6": ("Macro-Track", None, "", "4 macro-tracks"),
    "P2": ("Saber Pro top quartile", "P2", "_P2", "Saber Pro top quartile vs below"),
}
LABEL, MAKE, SUFFIX, TITLE = TARGETS["E6"]
ACC = f"{LABEL} Top-1 Acc (%)"
F1 = f"{LABEL} Macro-F1"


def configure(tag):
    """Switch the module to another target (E6 = 4 tracks, P2 = Saber Pro top quartile)."""
    global LABEL, MAKE, SUFFIX, TITLE, ACC, F1
    LABEL, MAKE, SUFFIX, TITLE = TARGETS[tag]
    ACC, F1 = f"{LABEL} Top-1 Acc (%)", f"{LABEL} Macro-F1"


def corrected_t(diffs, ratio):
    """Nadeau-Bengio corrected resampled t-test of mean(diffs) = 0; returns (t, p)."""
    j = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return np.inf if np.mean(diffs) != 0 else 0.0, 0.0 if np.mean(diffs) != 0 else 1.0
    t = np.mean(diffs) / np.sqrt((1 / j + ratio) * var)
    return t, 2 * stats.t.sf(abs(t), df=j - 1)


def corrected_ci(values, ratio):
    j = len(values)
    half = stats.t.ppf(0.975, j - 1) * np.sqrt((1 / j + ratio) * np.var(values, ddof=1))
    return np.mean(values) - half, np.mean(values) + half


def main(repeats=5):
    df = base.load_and_prepare(base.DATA_PATH)
    if MAKE:
        from objective1b_performance import EXPERIMENTS as PERF
        df = df[df["QUARTILE"].notna()].reset_index(drop=True)
        df[base.MACRO_TARGET] = PERF[MAKE]["make"](df["QUARTILE"])
    frames = []
    for r in range(repeats):
        _, folds = base.run_experiment(
            df, label_col=base.MACRO_TARGET, weighted=False, temporal_features=NO_PRO,
            time_steps=2, title=f"REPEATED CV - repeat {r + 1}/{repeats} ({TITLE})",
            target_label=LABEL,
            cv_seed=base.SEED + r, seed_offset=100 * r,
            summary_csv=os.devnull, fold_csv=os.devnull)
        frames.append(folds.assign(Repeat=r + 1))
    folds = pd.concat(frames, ignore_index=True)
    folds.to_csv(f"objective1_repeated_cv{SUFFIX}_folds.csv", index=False)

    ratio = 1 / (base.N_SPLITS - 1)  # n_test / n_train for k-fold
    key = ["Repeat", "Fold"]
    ref = folds[folds["Model"] == base.REFERENCE].set_index(key)
    prop = folds[folds["Model"] == base.PROPOSED].set_index(key)
    rows = []
    for name, (category, _, _) in base.MODEL_REGISTRY.items():
        m = folds[folds["Model"] == name].set_index(key)
        row = {"Category": category, "Model": name}
        for col, label, nd in [(ACC, "Accuracy (%)", 2), (F1, "Macro-F1", 4)]:
            lo, hi = corrected_ci(m[col].values, ratio)
            row[label] = round(m[col].mean(), nd)
            row[f"{label} 95% CI"] = f"{lo:.{nd}f}-{hi:.{nd}f}"
        gain = m[F1] - ref.loc[m.index, F1]
        row["Macro-F1 gain over majority"] = round(gain.mean(), 4)
        row["Macro-F1 ratio to majority"] = round(m[F1].mean() / ref[F1].mean(), 2)
        if name != base.REFERENCE:
            _, p = corrected_t(gain.values, ratio)
            row["p vs majority (corrected)"] = f"{p:.3g}"
        if name != base.PROPOSED:
            _, p = corrected_t((prop[F1] - m.loc[prop.index, F1]).values, ratio)
            row["Proposed minus model (Macro-F1)"] = round((prop[F1] - m.loc[prop.index, F1]).mean(), 4)
            row["p vs Proposed (corrected)"] = f"{p:.3g}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(f"objective1_repeated_cv{SUFFIX}_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"\nREPEATED CV SUMMARY - {TITLE} ({repeats} x {base.N_SPLITS}-fold, corrected resampled t)")
    print(summary.to_string(index=False))


def plot(folds_csv=None, out=None):
    folds_csv = folds_csv or f"objective1_repeated_cv{SUFFIX}_folds.csv"
    out = out or ("fig12_repeated_cv.png" if not SUFFIX else f"fig15_repeated_cv{SUFFIX}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = pd.read_csv(folds_csv)
    ratio = 1 / (base.N_SPLITS - 1)
    order = list(base.MODEL_REGISTRY)
    short = {base.REFERENCE: "Majority class", "Logistic Regression": "Logistic Reg.",
             "Random Forest": "Random Forest", "HistGradientBoosting": "HistGradBoost",
             "Static MLP Only": "Static MLP", "Standalone LSTM": "LSTM",
             "Hybrid 1: MLP + CNN + LSTM": "H1 MLP+CNN+LSTM", "Hybrid 2: MLP + BiLSTM": "H2 MLP+BiLSTM",
             "Hybrid 3: MLP + GRU": "H3 MLP+GRU", "Hybrid 4: MLP + Transformer": "H4 MLP+Transf.",
             base.PROPOSED: "Proposed TCN+LSTM"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#e4e3df",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (col, title) in zip(axes, [(F1, "Macro-F1"), (ACC, "Accuracy (%)")]):
        ref = folds.loc[folds["Model"] == base.REFERENCE, col].mean()
        for i, m in enumerate(order[::-1]):
            v = folds.loc[folds["Model"] == m, col].values
            colr = "#2a78d6" if m == base.PROPOSED else "#8a8983"
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
            ax.scatter(v, i + jitter, s=9, color=colr, alpha=0.45, linewidths=0)
            lo, hi = corrected_ci(v, ratio)
            ax.plot([lo, hi], [i, i], color="#0b0b0b", linewidth=2)
            ax.plot(v.mean(), i, marker="o", markersize=7, color=colr,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.axvline(ref, color="#0b0b0b", linestyle="--", linewidth=1.1)
        ax.text(ref, len(order) - 0.4, f" majority class {ref:.3g}", fontsize=7.5, va="bottom")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([short[m] for m in order[::-1]])
        ax.get_yticklabels()[order[::-1].index(base.PROPOSED)].set_fontweight("bold")
        ax.set_title(f"{TITLE}: {title}", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(order) - 0.1)
    fig.suptitle("Repeated cross-validation (5 x 5 folds): dots = folds, circle = mean, "
                 "bar = corrected 95% CI (Nadeau-Bengio)", x=0.0, ha="left", fontsize=11,
                 fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    tags = [a.upper() for a in args if a.upper() in TARGETS]
    configure(tags[0] if tags else "E6")
    if "plot" in args:
        plot()
    else:
        nums = [int(a) for a in args if a.isdigit()]
        main(nums[0] if nums else 5)
        plot()
