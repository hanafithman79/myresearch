# -*- coding: utf-8 -*-
"""
Objective 1b - predicting post-secondary performance (Saber Pro)
================================================================
Same models, folds, seeds and leakage-safe preprocessing as objective1_ablations.py
(final protocol: no class weights), but the target is the student's Saber Pro result,
measured at the END of university, predicted from information available at the START:
Saber 11 scores (+ derived STEM/HUM features) and socioeconomic background.

  P1  Saber Pro national quartile (4 classes: Q1..Q4)
  P2  Top national quartile (Q4) vs below (Q1-Q3)  - near-balanced binary task

No Saber Pro variable (G_SC, PERCENTILE, 2ND_DECILE, QUARTILE, *_PRO) is used as an input.
Outputs: objective1_<tag>_results.csv, objective1_<tag>_folds.csv, objective1_<tag>_oof.npz.

Usage: python objective1b_performance.py [P1 P2]   (dataset.csv in the working directory)
"""

import sys

import numpy as np
import torch

import objective1_experiments as base
from objective1_ablations import SABER_PRO_FEATURES

NO_PRO = [f for f in base.TEMPORAL_FEATURES if f not in SABER_PRO_FEATURES]

EXPERIMENTS = {
    "P1": dict(title="P1 SABER PRO QUARTILE (4 classes) from Saber 11 + background",
               target_label="Saber Pro quartile",
               make=lambda q: "Q" + q.astype(int).astype(str)),
    "P2": dict(title="P2 SABER PRO TOP QUARTILE vs BELOW from Saber 11 + background",
               target_label="Saber Pro top quartile",
               make=lambda q: np.where(q == 4, "Top quartile", "Below top quartile")),
}


def main(selected):
    print(f"[Setup] Device: {base.DEVICE} | torch {torch.__version__}")
    df = base.load_and_prepare(base.DATA_PATH)
    df = df[df["QUARTILE"].notna()].reset_index(drop=True)
    for tag, cfg in EXPERIMENTS.items():
        if selected and tag not in selected:
            continue
        data = df.copy()
        # run_experiment trains on MACRO_TARGET when label_col == MACRO_TARGET; here that
        # column holds the Saber Pro target, and target_label names the metrics after it.
        data[base.MACRO_TARGET] = cfg["make"](data["QUARTILE"])
        print(f"\n[{tag}] class distribution:\n{data[base.MACRO_TARGET].value_counts().to_string()}")
        base.run_experiment(
            data, label_col=base.MACRO_TARGET, weighted=False,
            temporal_features=NO_PRO, time_steps=2, title=cfg["title"],
            target_label=cfg["target_label"],
            summary_csv=f"objective1_{tag}_results.csv",
            fold_csv=f"objective1_{tag}_folds.csv",
            oof_npz=f"objective1_{tag}_oof.npz")


if __name__ == "__main__":
    main(set(a.upper() for a in sys.argv[1:]))
