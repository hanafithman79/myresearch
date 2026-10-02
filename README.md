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
| `objective1b_performance.py` | P1/P2: predicting Saber Pro performance from Saber 11 + background |
| `objective1_rigor.py` | Bootstrap 95% CIs, seed stability, permutation importance |
| `objective1_figures.py` | Figures 1-9 |

Put `dataset.csv` in the working directory, then run a script with `python <script>.py`.
To redraw the figures:
`python objective1_figures.py results/ablations results/figures results/objective1b results/rigor`

## Results

- `results/` - baseline run (original specification)
- `results/ablations/` - E0-E6 summaries, per-fold metrics, out-of-fold predictions
- `results/objective1b/` - P1/P2 Saber Pro performance results
- `results/rigor/` - confidence intervals, seed stability, feature importance
- `results/figures/` - confusion matrices, heatmaps and metric charts

## Headline findings

- Major choice: best macro-track accuracy 45.7%, against 43.6% for always predicting
  the most common track. The features carry little information about the major.
- Saber Pro performance: top national quartile vs below is predicted with 81.0%
  accuracy (LSTM on Saber 11 scores), against 50.9% for the majority guess.
- The dual-branch hybrid matches or trails simpler models on every target.
