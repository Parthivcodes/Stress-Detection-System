# Feature engineering, broader XGBoost tuning and calibration

## Outcome

Performance gates passed: **False**. Passing is not itself permission to activate: runtime checks are also required.

This experiment does not modify production artifacts. The other five model families were not retrained in this round. No cloud deployment.

## Feature comparison (training-only grouped CV)

| Representation | Dimensions | Mean log-loss (lower is better) |
|---|---:|---:|
| original | 69 | 0.719740 |
| ordered | 31 | 0.719603 |
| engineered | 38 | 0.719496 |

Selected: **engineered**. Comparison used identical baseline settings; no feature representation was forced into production.

Ordered encoding keeps six ordered scales numeric. Engineered encoding adds a seven-factor social-risk count, unhealthy-days flag/log, sleep bands and a clipped BMI copy. The original BMI remains. Clipping/scaling is fitted within training folds. These transformations add no respondents or new information; the risk count is not a validated clinical scale.

## Search and calibration

50 sequential Optuna TPE trials; two stratified grouped outer folds, with training-only inner early stopping; 700-round cap and patience 30. Objective: unweighted log-loss. No validation or holdout labels used for hyperparameter search.

Log-loss can improve partly because smaller class weights bring probabilities closer to the observed class distribution. This does not by itself demonstrate stronger discrimination, accuracy at protected recall, or better high-stress detection.

12,832 original training respondents reserved before tuning for disjoint temperature fitting/checking. The final model excludes this reserve. This reduces fitting data deliberately to protect calibration independence.

Calibration accepted: **False**; applied temperature 1.000000. Independent check log-loss 0.688514 → 0.688547; Brier 0.399875 → 0.399892.

A scalar temperature cannot change argmax labels. A separate high-class cutoff is selected on validation. The ensemble retains LR 40%, XGB 50%, LSTM 10%, with one cutoff; it uses candidate calibrated XGB probabilities only in this isolated experiment.

Selected hyperparameters:

```json
{
  "n_estimators": 700,
  "learning_rate": 0.06800981021604165,
  "max_depth": 2,
  "min_child_weight": 7.397930995578851,
  "subsample": 0.7464666411823344,
  "colsample_bytree": 0.8569248655919637,
  "reg_alpha": 0.005029298116217086,
  "reg_lambda": 3.4621066431849417,
  "gamma": 0.2747336593377415,
  "weight_power": 0.006129106017468772
}
```

Final best iteration (zero-based): 698; fitting time 5.2s.

## Validation decisions

| Model / decision | Accuracy | High recall | High missed | Sometimes recall | Macro-F1 | QWK |
|---|---:|---:|---:|---:|---:|---:|
| xgb / original_actual | 67.45% | 65.04% | 622 | 21.41% | 0.5307 | 0.5063 |
| xgb / baseline | 67.51% | 65.04% | 622 | 20.98% | 0.5303 | 0.5097 |
| xgb / candidate | 68.97% | 65.04% | 622 | 5.41% | 0.4766 | 0.4857 |
| xgb / raw_argmax | 70.70% | 39.97% | 1068 | 19.17% | 0.5243 | 0.4717 |
| xgb / calibrated_argmax | 70.70% | 39.97% | 1068 | 19.17% | 0.5243 | 0.4717 |
| ensemble / original_actual | 68.43% | 60.03% | 711 | 23.86% | 0.5433 | 0.5143 |
| ensemble / baseline | 68.45% | 60.26% | 707 | 23.80% | 0.5434 | 0.5139 |
| ensemble / candidate | 69.56% | 60.03% | 711 | 13.91% | 0.5172 | 0.4964 |
| Always rarely (majority baseline) | 65.68% | 0% | All high cases | 0% | — | — |

High-recall floors: XGB 65.04%; ensemble 60.03%. Safeguards require no loss in macro-F1, QWK or sometimes recall against both current and matched-threshold baselines.

## One frozen final evaluation

| Model / decision | Accuracy | High recall | High missed | Sometimes recall | Macro-F1 | QWK |
|---|---:|---:|---:|---:|---:|---:|
| xgb / original_actual | 67.54% | 65.33% | 1255 | 20.95% | 0.5295 | 0.5090 |
| xgb / original_matched | 67.67% | 65.14% | 1262 | 20.59% | 0.5298 | 0.5141 |
| xgb / candidate | 68.75% | 64.72% | 1277 | 4.72% | 0.4717 | 0.4857 |
| xgb / raw_argmax | 70.45% | 40.58% | 2151 | 18.44% | 0.5228 | 0.4706 |
| xgb / calibrated_argmax | 70.45% | 40.58% | 2151 | 18.44% | 0.5228 | 0.4706 |
| ensemble / original_actual | 68.39% | 60.14% | 1443 | 22.68% | 0.5393 | 0.5202 |
| ensemble / original_matched | 68.36% | 60.19% | 1441 | 22.54% | 0.5386 | 0.5201 |
| ensemble / candidate | 69.35% | 60.50% | 1430 | 12.84% | 0.5124 | 0.5032 |
| ensemble / raw_argmax | 70.27% | 45.44% | 1975 | 23.26% | 0.5414 | 0.4995 |
| ensemble / calibrated_argmax | 70.27% | 45.44% | 1975 | 23.26% | 0.5414 | 0.4995 |
| Always rarely (majority baseline) | 65.78% | 0% | All high cases | 0% | — | — |

The holdout was reused by historical experiments. This is NOT untouched external evidence; its uncertainty intervals do not account for prior adaptive reuse. Never retune against these results. Historical always-rarely accuracy: 65.78%, with 0% high recall.

Gate explanations:

```json
{
  "xgb": [
    "No validation threshold passed all declared accuracy, recall, macro-F1, QWK and middle-recall gates; candidate threshold is a reporting-only fallback.",
    "Final high-stress recall below the current-model/60% floor.",
    "macro_f1 below current decision.",
    "qwk below current decision.",
    "Sometimes recall below current decision.",
    "macro_f1 below matched decision.",
    "qwk below matched decision.",
    "Sometimes recall below matched decision."
  ],
  "ensemble": [
    "No validation threshold passed all declared accuracy, recall, macro-F1, QWK and middle-recall gates; candidate threshold is a reporting-only fallback.",
    "macro_f1 below current decision.",
    "qwk below current decision.",
    "Sometimes recall below current decision.",
    "macro_f1 below matched decision.",
    "qwk below matched decision.",
    "Sometimes recall below matched decision."
  ]
}
```

## Files and reproduction

- `protocol.json`, `original_hashes.json`, `source_hashes.json`: frozen design and original integrity. `backup/`: protected copies.
- `training_roles.csv`, `training_cv_roles.csv`: respondent roles and grouped splits.
- `trials/`, `optuna.sqlite3`, `optuna_trials.csv`, `best_trial.json`: all 50 searches.
- `candidate.joblib`, `encoder.json`, `calibration.json`: isolated model and preprocessing/calibration.
- `frozen_selection.json`: thresholds and hashes frozen before evaluation.
- `test_report.json`, `comparison.csv`, `gate_explanations.json`: detailed metrics, confusion matrices and paired intervals.
- `validation_reliability.csv`, `accuracy_recall_curve.csv`, `accuracy_recall.svg`: reliability bins and operating trade-off.
- `Feature_Engineering_and_XGBoost.ipynb`: executed explanation of saved evidence; does not rerun the holdout.

Run `tools/tune_xgb_features.py init/search/select/test --project PROJECT --output NEW_EXPERIMENT` in that order. Each command needs its phase; choose a NEW output for replication. `test` is one-shot and deliberately refuses to rerun. Supply `--optuna-path` only when using an isolated package installation. Keep `feature_pipeline.py` and `tune_six_models.py` beside the tuning script.

Runtime used: existing CPU ML environment plus isolated Optuna 5.0.0. No production dependency replacements. Refer to `environment.json` for exact versions. Existing experiment files must not be overwritten.

## Rollback and viva summary

If the candidate is rejected, no rollback is needed: current data/models/configuration were never replaced. Do not copy candidate.joblib into current_models; its encoder and calibration differ. Backups retain original relative paths and SHA-256 hashes.

“We tested better representations of the same real survey inputs, searched 50 XGBoost configurations using training-only grouped cross-validation, checked probability calibration on separate respondents, and chose a validation threshold that protects high-stress detection. We froze the candidate before the final reused-holdout evaluation and kept the existing model whenever the safety gates failed.”

## Runtime and preservation

Completed trials: 50; total trial fit/evaluation time 195.1s. Candidate artifact 1.81 MiB.

Unchanged active backend: 512 validation requests / 3072 model outputs passed; 34.4s for the check; peak 350.0 MiB under a 512 MiB cap with swap disabled. No OOM kill. Invalid input/authentication checks and explicit overload backpressure passed. This is local verification, not a cloud deployment.
