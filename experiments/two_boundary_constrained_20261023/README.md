# Two-boundary decisions and constrained XGBoost tuning

Performance gate passed: **False**. Frozen configuration: **boundaries_only**. Passing this gate still requires runtime/integration checks; this report does not itself activate anything.

## Protection and limitations

Current models, data and relevant inference sources were copied into backup/ and SHA-256 recorded before trials. No source respondents were dropped, duplicated or relabeled. Other five models were not retrained. No cloud deployment.

The final holdout has been reused by earlier experiments. It is NOT an independent test set; bootstrap intervals do not account for adaptive historical reuse. No guarantee about future-person recall is made.

This is a bounded search, not proof that no better model or threshold exists. The high boundary is searched on a finite grid and the two-boundary policy does not generally reproduce the original XGBoost class-multiplier decision regions. Rejected candidates do not establish a ceiling on achievable accuracy.

## Method

First, evaluate new decision boundaries on existing raw model probabilities. Predict high if P(high) >= h; otherwise predict sometimes if P(sometimes)/(P(rarely)+P(sometimes)) >= s, else rarely. Probabilities are not altered to match labels. Ties at either boundary go to the upper class. The high grid uses 81 probability quantiles; the lower boundary search is exact for every distinct conditional probability.

Then search 40 predeclared XGBoost configurations using seeded sklearn ParameterSampler (no new packages). Each of two grouped outer folds has separate model-fitting, early-stopping, threshold-calibration and scoring respondents. Preprocessing is fitted only on the model-fitting portion. The original XGBoost is refitted within each fold, so baseline predictions do not leak full-training labels into scoring.

Primary search objective: highest mean outer-fold accuracy among candidates preserving high recall (at least 60% or that fold baseline), sometimes recall, macro-F1 and QWK on BOTH scoring folds. Calibrate boundaries on the inner calibration subset, never on the scoring fold. Search includes original/engineered features and separate sometimes/high training class weights. Training-only weighting does not change the evaluation class distribution.

Feasible CV configurations: 1 / 40. Chosen trial: 0 (zero-based). Diagnostic-only CV selection: False.

Final fitting excludes a reserved 20% training subset and uses an internal early-stop subset. Existing validation selects the final boundaries. Complete configuration selection requires BOTH XGB and ensemble to improve with no safeguard loss. If none qualifies, the declared least-violation configuration is evaluated only for documentation, never activation.

## Validation comparisons

| Model / version | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | High missed |
|---|---:|---:|---:|---:|---:|---:|
| xgb / current | 67.45% | 65.04% | 21.41% | 0.5307 | 0.5063 | 622 |
| xgb / boundaries_only | 66.36% | 67.00% | 23.45% | 0.5308 | 0.5073 | 587 |
| xgb / tuned_xgb | 66.34% | 66.67% | 23.92% | 0.5313 | 0.5051 | 593 |
| ensemble / current | 68.43% | 60.03% | 23.86% | 0.5433 | 0.5143 | 711 |
| ensemble / boundaries_only | 68.44% | 60.03% | 23.86% | 0.5434 | 0.5144 | 711 |
| ensemble / tuned_xgb | 67.79% | 60.15% | 25.98% | 0.5442 | 0.5151 | 709 |
| Always rarely (majority baseline) | 65.68% | 0% | 0% | — | — | All high cases |

## Frozen final comparison

| Model / version | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | High missed |
|---|---:|---:|---:|---:|---:|---:|
| xgb / original | 67.54% | 65.33% | 20.95% | 0.5295 | 0.5090 | 1255 |
| xgb / candidate | 66.47% | 66.85% | 23.70% | 0.5314 | 0.5077 | 1200 |
| xgb / raw_argmax | 68.91% | 52.40% | 31.69% | 0.5607 | 0.5269 | 1723 |
| ensemble / original | 68.39% | 60.14% | 22.68% | 0.5393 | 0.5202 | 1443 |
| ensemble / candidate | 68.39% | 60.14% | 22.65% | 0.5392 | 0.5203 | 1443 |
| ensemble / raw_argmax | 68.96% | 50.86% | 29.75% | 0.5543 | 0.5216 | 1779 |
| Always rarely (majority baseline) | 65.78% | 0% | 0% | — | — | All high cases |

Historical majority baseline: 65.78% accuracy, zero high or sometimes recall. No alternative was selected using these holdout results.

Gate explanations:

```json
{
  "xgb": [
    "Frozen candidate did not qualify on validation; diagnostic only.",
    "Accuracy did not improve.",
    "qwk decreased.",
    "Paired 95% interval does not establish a positive accuracy gain."
  ],
  "ensemble": [
    "Frozen candidate did not qualify on validation; diagnostic only.",
    "Sometimes recall decreased.",
    "macro_f1 decreased.",
    "Paired 95% interval does not establish a positive accuracy gain."
  ]
}
```

Selected training parameters:

```json
{
  "n_estimators": 350,
  "max_depth": 5,
  "learning_rate": 0.04,
  "min_child_weight": 12,
  "reg_lambda": 5,
  "reg_alpha": 0,
  "subsample": 0.85,
  "colsample_bytree": 0.85,
  "gamma": 0,
  "weight_power": 0.5,
  "middle_weight": 1.0,
  "high_weight": 1.0,
  "mode": "original"
}
```

Frozen boundaries:

```json
{
  "xgb": {
    "high": 0.27792591452598575,
    "middle": 0.4678294798942472
  },
  "ensemble": {
    "high": 0.33022436197988114,
    "middle": 0.5001174221478011
  }
}
```

## Evidence and reproduction

- `protocol.json`, `configurations.json`, `source_hashes.json`: design and search frozen before trials. `pretrial_fix.md` documents the implementation correction made before any results.
- `backup/`, `original_hashes.json`: safe copies and original hashes.
- `cv_roles.csv`, `final_roles.csv`: respondent roles, no overlap between fit/calibration/scoring groups.
- `trials/`, `cv_baselines.json`, `trial_history.csv`, `cv_selection.json`: complete CV results, including rejected configurations.
- `boundary_validation.json`, `frozen_selection.json`, `test_report.json`, `comparison.csv`: decisions and metrics. The test report includes per-class precision/recall/F1, confusion matrices, AUC, detected/missed counts and paired intervals.
- `candidate.joblib`, `encoder.json`: isolated checkpoint; do not copy into current_models without its compatible inference support.
- `Two_Boundary_Constrained_Tuning.ipynb`: executed notebook reading saved evidence; does not rerun the holdout.

Use the existing WSL ML environment and run tools/tune_two_boundaries.py with phases init, boundaries, search, select, test in order, each with --project PROJECT --output NEW_EXPERIMENT. The init phase refuses an existing output; the test phase has an exclusive one-shot guard. Never overwrite prior experiments or rerun test to choose an alternative.

Total trial fit/scoring time: 313.4s. Candidate checkpoint: 2.57 MiB.

If rejected, rollback is unnecessary because no active artifact changed. The original models and policy remain authoritative. Runtime verification is reported separately; there has been no Render or Vercel deployment.

## Viva explanation

“We tuned the decision rules and training parameters together, while requiring the model to keep detecting both high-stress and sometimes-stress respondents. We used separate training, calibration and scoring folds to avoid choosing and judging cutoffs on the same people. We only allow a replacement when the frozen results and runtime checks satisfy the declared safeguards.”
