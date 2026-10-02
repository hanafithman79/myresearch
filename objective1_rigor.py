# -*- coding: utf-8 -*-
"""
Objective 1 - robustness checks for the final protocols
=======================================================
  1. Bootstrap 95% CIs (1,000 resamples of students) for accuracy and Macro-F1 of every
     model, from the saved out-of-fold predictions, plus a paired CI for
     Proposed minus the best baseline.                     -> objective1_bootstrap_ci.csv
  2. Seed stability: the deep models re-trained with two more seed sets on the same folds
     (Logistic Regression and the majority class are included as deterministic anchors).
                                                           -> objective1_seed_stability.csv
  3. Permutation importance (Logistic Regression, 5 folds, Macro-F1 drop when one raw
     column is shuffled) for the track target and the Saber Pro quartile target.
                                                           -> objective1_permutation_importance.csv

Inputs: dataset.csv and objective1_{E5,E6,P1,P2}_oof.npz in the working directory.
Usage:  python objective1_rigor.py [ci seeds importance]   (no arguments runs all three)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import objective1_experiments as base
from objective1_ablations import SABER_PRO_FEATURES
from objective1b_performance import EXPERIMENTS as PERF

NO_PRO = [f for f in base.TEMPORAL_FEATURES if f not in SABER_PRO_FEATURES]
N_BOOT = 1000
CI_TAGS = {"E5": "Grouped majors (16)", "E6": "Macro-track (4)",
           "P1": "Saber Pro quartile (4)", "P2": "Saber Pro top quartile (2)"}


def bootstrap_ci():
    rng = np.random.default_rng(base.SEED)
    rows = []
    for tag, target in CI_TAGS.items():
        path = f"objective1_{tag}_oof.npz"
        if not os.path.exists(path):
            print(f"[CI] skip {tag}: {path} not found")
            continue
        d = np.load(path)
        models = list(d["models"])
        # E5 trains on grouped majors (y); the others train on the MACRO_TARGET column (y_macro == y)
        y = d["y"]
        labels = np.arange(d["probs"].shape[2])
        preds = d["probs"].argmax(axis=2)
        idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
        boot = {}
        for mi, m in enumerate(models):
            acc = np.array([accuracy_score(y[i], preds[mi, i]) for i in idx]) * 100
            f1 = np.array([f1_score(y[i], preds[mi, i], average="macro", labels=labels,
                                    zero_division=0) for i in idx])
            boot[m] = f1
            rows.append({"Target": target, "Model": m,
                         "Accuracy (%)": 100 * accuracy_score(y, preds[mi]),
                         "Accuracy 95% CI": f"{np.percentile(acc, 2.5):.1f}-{np.percentile(acc, 97.5):.1f}",
                         "Macro-F1": f1_score(y, preds[mi], average="macro", labels=labels,
                                              zero_division=0),
                         "Macro-F1 95% CI": f"{np.percentile(f1, 2.5):.3f}-{np.percentile(f1, 97.5):.3f}"})
        # paired: Proposed minus the best non-reference baseline (by point Macro-F1)
        cand = [r for r in rows if r["Target"] == target and r["Model"] not in
                (base.PROPOSED, base.REFERENCE)]
        best = max(cand, key=lambda r: r["Macro-F1"])["Model"]
        diff = boot[base.PROPOSED] - boot[best]
        rows.append({"Target": target, "Model": f"Proposed minus {best}",
                     "Macro-F1": diff.mean(),
                     "Macro-F1 95% CI": f"{np.percentile(diff, 2.5):.3f} to {np.percentile(diff, 97.5):.3f}"})
        print(f"[CI] {target} done")
    out = pd.DataFrame(rows).round(4)
    out.to_csv("objective1_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


def target_frames():
    df = base.load_and_prepare(base.DATA_PATH)
    df = df[df["QUARTILE"].notna()].reset_index(drop=True)
    track = df.copy()
    quart = df.copy()
    quart[base.MACRO_TARGET] = PERF["P1"]["make"](quart["QUARTILE"])
    return {"E6": ("Macro-Track", track), "P1": ("Saber Pro quartile", quart)}


def seed_stability():
    models = [base.REFERENCE, "Logistic Regression", "Hybrid 3: MLP + GRU", base.PROPOSED]
    rows = []
    for tag, (label, data) in target_frames().items():
        for offset in (0, 1000, 2000):
            _, folds = base.run_experiment(
                data, label_col=base.MACRO_TARGET, weighted=False, temporal_features=NO_PRO,
                time_steps=2, title=f"SEED STABILITY {tag} seed offset {offset}",
                target_label=label, models=models, seed_offset=offset,
                summary_csv=os.devnull, fold_csv=os.devnull)
            for m, g in folds.groupby("Model"):
                rows.append({"Experiment": tag, "Model": m, "Seed offset": offset,
                             "Accuracy (%)": g[f"{label} Top-1 Acc (%)"].mean(),
                             "Macro-F1": g[f"{label} Macro-F1"].mean()})
    out = pd.DataFrame(rows)
    summ = out.groupby(["Experiment", "Model"]).agg(
        acc_mean=("Accuracy (%)", "mean"), acc_sd=("Accuracy (%)", "std"),
        f1_mean=("Macro-F1", "mean"), f1_sd=("Macro-F1", "std")).reset_index().round(4)
    out.round(4).to_csv("objective1_seed_stability.csv", index=False)
    summ.to_csv("objective1_seed_stability_summary.csv", index=False)
    print(summ.to_string(index=False))


def importance():
    rows = []
    for tag, (label, data) in target_frames().items():
        y = data[base.MACRO_TARGET].values
        cont, nom = base.split_static_types(data)
        X = data[base.STATIC_FEATURES + NO_PRO].copy()
        for c in nom:
            X[c] = X[c].where(X[c].notna(), "MISSING").astype(str)
        skf = StratifiedKFold(base.N_SPLITS, shuffle=True, random_state=base.SEED)
        imps = []
        for tr, va in skf.split(X, y):
            pipe = make_pipeline(ColumnTransformer([
                ("nom", make_pipeline(SimpleImputer(strategy="constant", fill_value="MISSING"),
                                      OneHotEncoder(handle_unknown="ignore")), nom),
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                 cont + NO_PRO)]), LogisticRegression(max_iter=3000))
            pipe.fit(X.iloc[tr], y[tr])
            r = permutation_importance(pipe, X.iloc[va], y[va], scoring="f1_macro",
                                       n_repeats=5, random_state=base.SEED, n_jobs=-1)
            imps.append(r.importances_mean)
        imps = np.array(imps)
        for j, col in enumerate(X.columns):
            rows.append({"Experiment": tag, "Target": label, "Feature": col,
                         "Macro-F1 drop (mean)": imps[:, j].mean(),
                         "Macro-F1 drop (sd over folds)": imps[:, j].std(ddof=1)})
        print(f"[Importance] {tag} done")
    out = pd.DataFrame(rows).round(5).sort_values(["Experiment", "Macro-F1 drop (mean)"],
                                                   ascending=[True, False])
    out.to_csv("objective1_permutation_importance.csv", index=False)
    print(out.groupby("Experiment").head(8).to_string(index=False))


if __name__ == "__main__":
    parts = set(a.lower() for a in sys.argv[1:]) or {"ci", "seeds", "importance"}
    if "ci" in parts:
        bootstrap_ci()
    if "importance" in parts:
        importance()
    if "seeds" in parts:
        seed_stability()
