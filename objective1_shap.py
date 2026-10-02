# -*- coding: utf-8 -*-
"""
Objective 1 - SHAP explanations for the main result (4 macro-tracks)
====================================================================
Final protocol of E6 (4 tracks, no class weights, Saber Pro inputs removed). In each of
the 5 folds the model is trained on the training part and explained on held-out students:

  Logistic Regression   shap.LinearExplainer   (log-odds scale), all held-out students
  HistGradientBoosting  shap.TreeExplainer     (raw margin), all held-out students
  Proposed hybrid       shap.GradientExplainer (class probabilities), SAMPLE_PER_FOLD students

One-hot columns are summed (signed, as SHAP is additive) back to their original variable. Because the three models
explain different scales, importance is reported as each variable's SHARE of the model's
total mean |SHAP| (averaged over the 4 classes), which makes rankings comparable.

Outputs: objective1_shap_importance.csv, objective1_shap_by_class.csv,
         fig10_shap_importance.png, fig11_shap_by_class.png
Usage:   python objective1_shap.py [OUT_DIR]   (dataset.csv in the working directory)
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import objective1_experiments as base  # noqa: E402
from objective1_ablations import SABER_PRO_FEATURES  # noqa: E402

warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
NO_PRO = [f for f in base.TEMPORAL_FEATURES if f not in SABER_PRO_FEATURES]
TIME_STEPS = 2
SAMPLE_PER_FOLD = 400      # held-out students explained per fold for the deep model
BACKGROUND = 200           # background students for the deep model's expected gradients

BLUE = "#2a78d6"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                     "axes.titleweight": "bold", "axes.titlesize": 10})


def fold_arrays(df_tr, df_va, continuous, nominal):
    """Same transforms as base.preprocess_fold, plus the original variable of every column."""
    static_tf = base.build_static_transformer(continuous, nominal)
    temporal_tf = base.build_temporal_transformer()
    tr_s, va_s = df_tr[base.STATIC_FEATURES].copy(), df_va[base.STATIC_FEATURES].copy()
    for frame in (tr_s, va_s):
        for c in nominal:
            frame[c] = frame[c].where(frame[c].notna(), "MISSING").astype(str).str.strip()
    xs_tr = static_tf.fit_transform(tr_s).astype(np.float32)
    xs_va = static_tf.transform(va_s).astype(np.float32)
    xt_tr = temporal_tf.fit_transform(df_tr[NO_PRO]).astype(np.float32)
    xt_va = temporal_tf.transform(df_va[NO_PRO]).astype(np.float32)
    static_vars = []
    for name in static_tf.get_feature_names_out():
        raw = name.split("__", 1)[1]
        static_vars.append(max((f for f in base.STATIC_FEATURES if raw == f or raw.startswith(f + "_")),
                               key=len))
    return xs_tr, xs_va, xt_tr, xt_va, static_vars


def per_variable(vals, variables):
    """vals: (n, columns, classes) signed SHAP. SHAP values are additive, so a variable's
    contribution is the SIGNED sum of its one-hot columns; importance is the mean |sum|.
    (Summing absolute values per column instead would inflate many-category variables.)
    Returns the variable names and a V x K matrix of mean |SHAP|."""
    var_names = list(dict.fromkeys(variables))
    idx = {v: [i for i, x in enumerate(variables) if x == v] for v in var_names}
    summed = np.stack([vals[:, idx[v], :].sum(axis=1) for v in var_names], axis=1)  # n,V,K
    return var_names, np.abs(summed).mean(axis=0)  # V x K


def as_nck(values, n_classes):
    """Normalise SHAP output to (n, columns, classes)."""
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:  # binary-style output
        return values[:, :, None]
    if values.shape[-1] != n_classes and values.shape[1] == n_classes:
        return np.moveaxis(values, 1, -1)
    return values


class ProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xs, xt):
        return torch.softmax(self.model(xs, xt), dim=1)


def main():
    base.set_seed(base.SEED)
    df = base.load_and_prepare(base.DATA_PATH)
    classes = sorted(df[base.MACRO_TARGET].unique())
    y = df[base.MACRO_TARGET].map({c: i for i, c in enumerate(classes)}).values
    k = len(classes)
    continuous, nominal = base.split_static_types(df)
    skf = StratifiedKFold(base.N_SPLITS, shuffle=True, random_state=base.SEED)
    acc = {"Logistic Regression": [], "HistGradientBoosting": [], base.PROPOSED: []}
    variables = None

    for fold, (tr, va) in enumerate(skf.split(df, y), start=1):
        xs_tr, xs_va, xt_tr, xt_va, static_vars = fold_arrays(df.iloc[tr], df.iloc[va],
                                                              continuous, nominal)
        variables = static_vars + NO_PRO
        xf_tr, xf_va = np.hstack([xs_tr, xt_tr]), np.hstack([xs_va, xt_va])

        lr = LogisticRegression(max_iter=3000, random_state=base.SEED).fit(xf_tr, y[tr])
        sv = shap.LinearExplainer(lr, xf_tr).shap_values(xf_va)
        acc["Logistic Regression"].append(per_variable(as_nck(sv, k), variables)[1])

        hgb = HistGradientBoostingClassifier(random_state=base.SEED).fit(xf_tr, y[tr])
        sv = shap.TreeExplainer(hgb).shap_values(xf_va)
        acc["HistGradientBoosting"].append(per_variable(as_nck(sv, k), variables)[1])

        seq_tr = xt_tr.reshape(-1, TIME_STEPS, len(NO_PRO) // TIME_STEPS)
        seq_va = xt_va.reshape(-1, TIME_STEPS, len(NO_PRO) // TIME_STEPS)
        _, model = base.train_torch_model(
            base.ProposedDualBranch, xs_tr, seq_tr, y[tr], xs_va, seq_va, k,
            np.ones(k, dtype=np.float32), seed=base.SEED + fold, return_model=True)
        model.eval()
        rng = np.random.default_rng(base.SEED + fold)
        bg = rng.choice(len(tr), BACKGROUND, replace=False)
        ex = rng.choice(len(va), min(SAMPLE_PER_FOLD, len(va)), replace=False)
        explainer = shap.GradientExplainer(
            ProbWrapper(model).to(base.DEVICE),
            [torch.from_numpy(xs_tr[bg]).to(base.DEVICE), torch.from_numpy(seq_tr[bg]).to(base.DEVICE)])
        sv = explainer.shap_values([torch.from_numpy(xs_va[ex]).to(base.DEVICE),
                                    torch.from_numpy(seq_va[ex]).to(base.DEVICE)])
        if isinstance(sv, list) and len(sv) == 2 and np.asarray(sv[0]).ndim >= 3 \
                and np.asarray(sv[0]).shape[-1] == k:      # [static (n,S,K), seq (n,T,F,K)]
            s_part, t_part = np.asarray(sv[0]), np.asarray(sv[1])
        else:                                              # per class: [[static, seq], ...]
            s_part = np.stack([np.asarray(c[0]) for c in sv], axis=-1)
            t_part = np.stack([np.asarray(c[1]) for c in sv], axis=-1)
        t_part = t_part.reshape(t_part.shape[0], -1, k)    # steps x features, row-major = NO_PRO order
        full = np.concatenate([s_part, t_part], axis=1)
        acc[base.PROPOSED].append(per_variable(full, variables)[1])
        print(f"[SHAP] fold {fold} done", flush=True)

    var_names = list(dict.fromkeys(variables))
    rows, by_class = [], []
    for model_name, mats in acc.items():
        mat = np.mean(mats, axis=0)               # V x K, averaged over folds
        overall = mat.mean(axis=1)
        share = overall / overall.sum()
        for v, a, s in zip(var_names, overall, share):
            rows.append({"Model": model_name, "Feature": v, "Mean |SHAP|": a, "Share of total": s})
        if model_name == base.PROPOSED:
            norm = mat / mat.sum(axis=0, keepdims=True)
            for vi, v in enumerate(var_names):
                by_class.append({"Feature": v, **{c: norm[vi, ci] for ci, c in enumerate(classes)}})
    imp = pd.DataFrame(rows).sort_values(["Model", "Share of total"], ascending=[True, False])
    imp.round(5).to_csv(os.path.join(OUT, "objective1_shap_importance.csv"), index=False)
    bc = pd.DataFrame(by_class)
    bc.round(5).to_csv(os.path.join(OUT, "objective1_shap_by_class.csv"), index=False)

    # rank agreement between the three models
    piv = imp.pivot(index="Feature", columns="Model", values="Share of total")
    print("\nSpearman rank agreement of SHAP importance between models:")
    print(piv.corr(method="spearman").round(2).to_string())
    print(imp.groupby("Model").head(6).round(3).to_string(index=False))
    figures(imp, bc, classes)


def figures(imp, bc, classes):
    models = ["Logistic Regression", "HistGradientBoosting", base.PROPOSED]
    titles = ["Logistic Regression", "HistGradientBoosting", "Proposed (MLP + TCN + LSTM)"]
    top = 12
    lim = imp["Share of total"].max() * 1.12
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ax, m, t in zip(axes, models, titles):
        d = imp[imp["Model"] == m].nlargest(top, "Share of total")
        y = np.arange(len(d))[::-1]
        ax.barh(y, d["Share of total"] * 100, color=BLUE, height=0.68, edgecolor="white", linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(d["Feature"])
        ax.set_xlim(0, lim * 100)
        ax.set_title(t, loc="left")
        ax.set_xlabel("Share of total mean |SHAP| (%)")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("SHAP feature importance for the 4-track model (held-out students, 5 folds, top 12)",
                 x=0.0, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig10_shap_importance.png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    order = imp[imp["Model"] == base.PROPOSED].nlargest(top, "Share of total")["Feature"].tolist()
    mat = bc.set_index("Feature").loc[order, classes].values * 100
    short = [c.replace("Mechanical, Electrical & Tech", "Mech/Elec/Tech")
             .replace("Industrial & Management", "Industrial").replace("Civil & Infrastructure", "Civil")
             .replace("Chemical & Process", "Chemical") for c in classes]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=mat.max() * 1.05, aspect="auto")
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xticks(np.arange(-.5, len(short), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] >= mat.max() * 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("% of the class's total mean |SHAP|")
    cb.outline.set_visible(False)
    ax.set_title("Proposed model: which features drive each track", loc="left")
    fig.savefig(os.path.join(OUT, "fig11_shap_by_class.png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved fig10_shap_importance.png, fig11_shap_by_class.png")


if __name__ == "__main__":
    main()
