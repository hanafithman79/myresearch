# -*- coding: utf-8 -*-
"""
Objective 1 - controlled fusion-boundary experiment (semi-synthetic labels)
==========================================================================
Question: how does the benefit of dual-branch fusion depend on WHERE the predictive
signal lies (background variables vs Saber 11 scores)?

Real inputs, controlled outcome. Every student keeps his or her real background and
Saber 11 features; only the label is synthetic:
  S = background signal  (from the 14 real categorical background variables)
  A = academic signal    (from the 8 real Saber 11 features)
  z = sqrt(alpha) * S + sqrt(1 - alpha) * A          (S, A standardised)
  y = 1[z + noise > median]                          (balanced binary label)
so alpha is the share of the signal variance that comes from background. Noise SD 0.5
caps the accuracy of a perfect model at about 85%.

Two signal forms:
  linear     S, A = random-weight linear indices (Logistic Regression is near-optimal)
  nonlinear  S = random lookup tables over pairs of background variables (interactions);
             A = random pairwise products and squared terms of the scores
For each form, alpha in ALPHAS and N_DRAWS random weight draws. Models: majority class,
Logistic Regression, static MLP (background branch only), standalone LSTM (academic branch
only) and the proposed dual-branch model, all deep models with the same early stopping,
stratified 5-fold CV with leakage-safe preprocessing (the label is the data-generating
truth, so it is built once on the full data).

Outputs: objective1_fusion_boundary_folds.csv (every fold), _summary.csv,
         fig21_fusion_boundary.png
Usage:   python objective1_fusion_boundary.py            (run + summarise + plot)
         python objective1_fusion_boundary.py plot       (summarise + plot saved folds)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

import objective1_experiments as base  # noqa: E402
from objective1_ablations import SABER_PRO_FEATURES  # noqa: E402

NO_PRO = [f for f in base.TEMPORAL_FEATURES if f not in SABER_PRO_FEATURES]
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
FORMS = ["linear", "nonlinear"]
N_DRAWS = 2
NOISE_SD = 0.5
STATIC_ONLY, SEQ_ONLY = "Static MLP Only", "Standalone LSTM"
MODELS = [base.REFERENCE, "Logistic Regression", STATIC_ONLY, SEQ_ONLY, base.PROPOSED]
FOLDS_CSV = "objective1_fusion_boundary_folds.csv"
SUMMARY_CSV = "objective1_fusion_boundary_summary.csv"
F1 = "Synthetic Macro-F1"
# Approximate position of the real tasks on the alpha axis: SHAP share of importance from
# background variables across Logistic Regression, gradient boosting and the proposed model.
REAL_TASKS = {"Real task: study track": (0.50, 0.74), "Real task: Saber Pro": (0.15, 0.43)}


def _standardise(v):
    return (v - v.mean()) / v.std()


def signal_components(df, form, rng):
    """Return standardised background (S) and academic (A) signals for every student."""
    static = df[base.STATIC_FEATURES].copy()
    for c in base.STATIC_FEATURES:
        static[c] = static[c].where(static[c].notna(), "MISSING").astype(str).str.strip()
    acad = df[NO_PRO].astype(float)
    acad = ((acad - acad.median()) / acad.std()).fillna(0.0).values  # median-imputed, scaled

    if form == "linear":
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(static)
        s = onehot @ rng.normal(size=onehot.shape[1])
        a = acad @ rng.normal(size=acad.shape[1])
    else:
        pairs = [("GENDER", "STRATUM"), ("EDU_FATHER", "REVENUE"), ("OCC_MOTHER", "SCHOOL_TYPE"),
                 ("SISBEN", "INTERNET")]
        s = np.zeros(len(df))
        for c1, c2 in pairs:
            combo = static[c1] + "|" + static[c2]
            table = {k: rng.normal() for k in combo.unique()}
            s += combo.map(table).values
        a = np.zeros(len(df))
        n = acad.shape[1]
        for _ in range(4):
            i, j = rng.choice(n, 2, replace=False)
            a += rng.normal() * acad[:, i] * acad[:, j]
        a += (acad ** 2 - 1) @ (rng.normal(size=n) / np.sqrt(2))
    return _standardise(s), _standardise(a)


def make_label(s, a, alpha, rng):
    z = np.sqrt(alpha) * s + np.sqrt(1 - alpha) * a + rng.normal(scale=NOISE_SD, size=len(s))
    return np.where(z > np.median(z), "High", "Low")


def run():
    df = base.load_and_prepare(base.DATA_PATH)
    frames = []
    if os.path.exists(FOLDS_CSV):  # resume
        frames.append(pd.read_csv(FOLDS_CSV))
    done = set() if not frames else set(map(tuple, frames[0][["Form", "Draw", "Alpha"]]
                                            .drop_duplicates().values.tolist()))
    for form in FORMS:
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(1000 * (FORMS.index(form) + 1) + draw)
            s, a = signal_components(df, form, rng)
            print(f"[{form} draw {draw}] corr(S, A) = {np.corrcoef(s, a)[0, 1]:.3f}", flush=True)
            for alpha in ALPHAS:
                if (form, draw, alpha) in done:
                    continue
                data = df.copy()
                data[base.MACRO_TARGET] = make_label(
                    s, a, alpha, np.random.default_rng(int(alpha * 100) + 17 * draw))
                _, folds = base.run_experiment(
                    data, label_col=base.MACRO_TARGET, weighted=False, temporal_features=NO_PRO,
                    time_steps=2, title=f"FUSION BOUNDARY {form} draw {draw} alpha {alpha}",
                    target_label="Synthetic", models=MODELS, early_stopping=True,
                    summary_csv=os.devnull, fold_csv=os.devnull)
                frames.append(folds.assign(Form=form, Draw=draw, Alpha=alpha,
                                           CorrSA=np.corrcoef(s, a)[0, 1]))
                pd.concat(frames, ignore_index=True).to_csv(FOLDS_CSV, index=False)


def summarise():
    folds = pd.read_csv(FOLDS_CSV)
    wide = folds.pivot_table(index=["Form", "Alpha", "Draw", "Fold"], columns="Model", values=F1)
    rows = []
    for (form, alpha), g in wide.groupby(level=["Form", "Alpha"]):
        best_single = g[[STATIC_ONLY, SEQ_ONLY]].max(axis=1)
        gain = g[base.PROPOSED] - best_single
        vs_lr = g[base.PROPOSED] - g["Logistic Regression"]
        t_gain = stats.ttest_1samp(gain, 0.0)
        t_lr = stats.ttest_1samp(vs_lr, 0.0)
        rows.append({"Form": form, "Alpha (background share)": alpha,
                     **{f"{m} Macro-F1": round(g[m].mean(), 4) for m in MODELS},
                     "Fusion gain over best single branch": round(gain.mean(), 4),
                     "Gain 95% CI": "{:.4f} to {:.4f}".format(*stats.t.interval(
                         0.95, len(gain) - 1, gain.mean(), stats.sem(gain))),
                     "p (gain)": f"{t_gain.pvalue:.3g}",
                     "Proposed minus Logistic Regression": round(vs_lr.mean(), 4),
                     "p (vs LR)": f"{t_lr.pvalue:.3g}", "Folds": len(gain)})
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(out.to_string(index=False))
    return folds, wide


def plot(wide):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#e4e3df",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    colours = {"Logistic Regression": "#0b0b0b", STATIC_ONLY: "#eb6834", SEQ_ONLY: "#1baf7a",
               base.PROPOSED: "#2a78d6"}
    names = {"Logistic Regression": "Logistic Regression", STATIC_ONLY: "Background branch only (MLP)",
             SEQ_ONLY: "Academic branch only (LSTM)", base.PROPOSED: "Fusion (proposed)"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for col, form in enumerate(FORMS):
        g = wide.xs(form, level="Form")
        ax = axes[0, col]
        for m, c in colours.items():
            mean = g[m].groupby(level="Alpha").mean()
            ax.plot(mean.index, mean.values, color=c, linewidth=2.2 if m == base.PROPOSED else 1.6,
                    linestyle="--" if m == "Logistic Regression" else "-", marker="o", markersize=4,
                    label=names[m])
        ax.set_title(f"{form.capitalize()} signal: Macro-F1 by model", loc="left", fontweight="bold")
        ax.set_ylabel("Macro-F1 (5 folds x 2 weight draws)")
        ax2 = axes[1, col]
        best = g[[STATIC_ONLY, SEQ_ONLY]].max(axis=1)
        gain = (g[base.PROPOSED] - best).groupby(level="Alpha")
        mean, sem = gain.mean(), gain.sem()
        half = stats.t.ppf(0.975, gain.size().values - 1) * sem
        ax2.fill_between(mean.index, mean - half, mean + half, color="#2a78d6", alpha=0.18, linewidth=0)
        ax2.plot(mean.index, mean.values, color="#2a78d6", linewidth=2.2, marker="o", markersize=4)
        ax2.axhline(0, color="#52514e", linewidth=1)
        ax2.set_title(f"{form.capitalize()} signal: fusion gain over the best single branch",
                      loc="left", fontweight="bold")
        ax2.set_ylabel("Macro-F1 difference (95% CI)")
        ax2.set_xlabel("Share of the signal from background variables (alpha)")
        for ax_ in (ax, ax2):
            for (label, (lo, hi)), shade in zip(REAL_TASKS.items(), ["#fdeee6", "#e3f5ee"]):
                ax_.axvspan(lo, hi, color=shade, zorder=0)
            ax_.grid(color="#e4e3df", linewidth=0.8)
            ax_.set_axisbelow(True)
            for sp in ("top", "right"):
                ax_.spines[sp].set_visible(False)
            ax_.set_xlim(-0.03, 1.03)
        ylo, yhi = ax2.get_ylim()
        for label, (lo, hi) in REAL_TASKS.items():
            ax2.text((lo + hi) / 2, yhi, label.replace("Real task: ", "real: "), ha="center",
                     va="top", fontsize=7.5, color="#52514e")
    axes[0, 0].legend(loc="lower center", frameon=False, fontsize=8)
    fig.suptitle("Controlled fusion-boundary experiment: real inputs, semi-synthetic labels "
                 "(shaded = approximate position of the real tasks)", x=0.0, ha="left",
                 fontsize=11, fontweight="bold")
    fig.savefig("fig21_fusion_boundary.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig21_fusion_boundary.png")


if __name__ == "__main__":
    if "plot" not in sys.argv[1:]:
        run()
    _, wide = summarise()
    plot(wide)
