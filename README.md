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
| `objective1b_performance.py` | P1/P2/P3: predicting Saber Pro performance from Saber 11 + background |
| `objective1_rigor.py` | Bootstrap 95% CIs, seed stability, permutation importance |
| `objective1_repeated_cv.py` | 5x5 repeated CV with Nadeau-Bengio corrected tests: 4 tracks (figure 12) or `P2` Saber Pro (figure 15) |
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
- `results/figures/` - confusion matrices, heatmaps and metric charts

## Headline findings

- Main result (4 tracks, 5x5 repeated CV): the proposed model doubles Macro-F1 over the
  majority-class reference (0.304 vs 0.152, corrected p < 1e-15) but not accuracy (43.6%).
  It is statistically tied with Logistic Regression (0.312) and the other hybrids.
- SHAP: gender is the largest driver of track in every model (17-19% of importance).

- Major choice: best macro-track accuracy 45.7%, against 43.6% for always predicting
  the most common track. The features carry little information about the major.
- Objective 1b, Saber Pro performance (5x5 repeated CV): top national quartile vs below is
  predicted with 79-81% accuracy against a 50.9% majority guess (LSTM 80.9%, Logistic
  Regression 80.7%, proposed 79.1%). Top half vs bottom half reaches 85.0% against a 76.2%
  baseline. The English Saber 11 score is the strongest predictor.
- The dual-branch hybrid matches or trails simpler models on every target.
