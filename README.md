# Objective 1: student study pathway prediction

Benchmark of a dual-branch hybrid model (static MLP + causal TCN + LSTM) against
classical, single-branch and hybrid baselines, using 12,411 Colombian engineering
students (Saber 11 and Saber Pro records). Stratified 5-fold CV with all preprocessing
fitted on the training folds only.

## Scripts

| Script | What it runs |
| --- | --- |
| `objective1_experiments.py` | Baseline protocol: 21 majors + 4 macro-tracks, 11 models, paired t-tests |
| `objective1_ablations.py` | E0-E6: direct macro-track, no class weights, grouped majors, no Saber Pro, final protocols |
| `objective1b_performance.py` | Contrast outcome P1/P2/P3: Saber Pro performance from Saber 11 + background (not a study-pathway target) |
| `objective1_rigor.py` | Bootstrap 95% CIs, seed stability, permutation importance |
| `objective1_repeated_cv.py` | 5x5 repeated CV with Nadeau-Bengio corrected tests: 4 tracks (figure 12) or `P2` Saber Pro (figure 15) |
| `objective1_early_stopping.py` | Fair early-stopping comparison of all deep models on both targets, figure 20 |
| `objective1_fusion_boundary.py` | Controlled fusion-boundary experiment: real inputs, semi-synthetic labels with a controlled background share of signal, figure 21 |
| `objective1_shap.py` | SHAP importance (Logistic Regression, gradient boosting, proposed model): 4 tracks (figures 10-11) or `P2` (figure 13) |
| `objective1_figures.py` | Figures 1-9 |

`dataset.csv` (12,411 students, 45 variables) is included. Run any script from the repository root with `python <script>.py`.
To redraw the figures:
`python objective1_figures.py results/ablations results/figures results/objective1b results/rigor`

## Results

- `results/` - baseline run (original specification)
- `results/ablations/` - E0-E6 summaries, per-fold metrics, out-of-fold predictions
- `results/objective1b/` - P1/P2/P3 Saber Pro performance results (confusion matrices: figures 7, 16, 17)
- `results/rigor/` - confidence intervals, seed stability, feature importance
- `results/repeated_cv/` - 5x5 repeated cross-validation of the main result
- `results/shap/` - SHAP importance overall and per track
- `results/fusion_boundary/` - semi-synthetic fusion-boundary experiment (every fold and the summary)
- `results/early_stopping/` - repeated CV with early stopping for all deep models, and the comparison
- `results/figures/` - confusion matrices, heatmaps and metric charts

## Target variable

The research predicts the study pathway, recorded as `ACADEMIC_PROGRAM`:
- **Main target:** `ACADEMIC_PROGRAM` grouped into 4 study tracks (`MACRO_TRACK`).
- **Granular target:** the 21 programs (16 after grouping programs with fewer than 30 students).

Saber Pro performance (`QUARTILE`, scripts and folders labelled "1b" / P1-P3) is not a
study-pathway target. It is used only as a contrast outcome, to show how the value of
dual-branch fusion depends on where the predictive signal lies. The fusion-boundary
experiment uses semi-synthetic labels on the real inputs for the same purpose.

## Headline findings

- Main result (4 tracks, 5x5 repeated CV): the proposed model doubles Macro-F1 over the
  majority-class reference (0.304 vs 0.152, corrected p < 1e-15) but not accuracy (43.6%).
  It is statistically tied with Logistic Regression (0.312) and the other hybrids.
- SHAP: gender is the largest driver of track in every model (17-19% of importance).
- Early stopping (same rule for all deep models, inner validation split of the training
  fold): the hybrids overfit at 15 epochs on the Saber Pro task (best epoch about 3).
  With early stopping the proposed model reaches Macro-F1 0.313 (tracks) and 0.805
  (Saber Pro) and is not significantly worse than any model on either task.

- Major choice: best macro-track accuracy 45.7%, against 43.6% for always predicting
  the most common track. The features carry little information about the major.
- Contrast outcome, Saber Pro performance (5x5 repeated CV): top national quartile vs below is
  predicted with 79-81% accuracy against a 50.9% majority guess (LSTM 80.9%, Logistic
  Regression 80.7%, proposed 79.1%). Top half vs bottom half reaches 85.0% against a 76.2%
  baseline. The English Saber 11 score is the strongest predictor.
- The dual-branch hybrid matches or trails simpler models on every target.
- Fusion-boundary experiment (real inputs, semi-synthetic labels): the gain of fusion over
  the best single branch follows an inverted U in the share of signal from background
  variables, about 0 at both extremes and +0.09 to +0.10 Macro-F1 at 40-50%. With a
  nonlinear signal the proposed model beats Logistic Regression by about 0.09 at every
  share; with a linear signal Logistic Regression stays slightly ahead.
