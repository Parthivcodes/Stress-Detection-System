# Expanded BRFSS experiment

## Completed held-out comparison

All values below use the same 30,718 previously unused respondents, including 3,620 usually/always responses. This is not the earlier 5,998-person test set.

| Model | Original accuracy | Tuned accuracy | Expanded accuracy | Original high recall | Tuned high recall | Expanded high recall | High cases missed, original → expanded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 58.84% | 64.54% | 67.46% | 62.27% | 53.09% | 64.14% | 1366 → 1298 |
| Random Forest | 60.75% | 63.81% | 66.58% | 58.78% | 52.90% | 61.10% | 1492 → 1408 |
| XGBoost | 59.39% | 66.17% | 67.54% | 60.72% | 46.60% | 65.33% | 1422 → 1255 |
| 1D CNN | 64.34% | 64.34% | 68.83% | 46.16% | 46.16% | 52.46% | 1949 → 1721 |
| LSTM | 61.59% | 63.96% | 68.96% | 54.61% | 50.99% | 56.74% | 1643 → 1566 |
| CNN-LSTM | 60.85% | 66.40% | 67.84% | 58.70% | 48.95% | 60.03% | 1495 → 1447 |

All six individual models passed the observed accuracy/high-recall gate. None reached 85% accuracy. High-frequency cases still get missed; this is not a diagnostic tool. Recall for the middle (sometimes) category decreased in every expanded model compared with its original version. Full precision, recall, macro F1, balanced accuracy and confusion matrices remain in comparison.json. The results do not establish external validity or guarantee future recall.

The majority-vote ensemble also passed: original accuracy/high recall 62.06%/56.49%, tuned 66.21%/49.17%, expanded **68.62%/59.89%**. Independent checks reproduced all 18 model/version metrics, spot-checked 100 records against the raw CDC file, verified 96 saved-model predictions, and confirmed the original data/model hashes remained unchanged.

Run `Start-BRFSS-Expanded.ps1` for the verified nineteen-input local preview on port 8004. The original active-version setting and earlier launchers are preserved. The form automatically adapts to the version served by the backend, and the evidence table includes high-frequency recall.


This is a separate experiment. The original 374-row dataset, the 30,000-row BRFSS dataset, both BRFSS model versions, and the active-model setting are preserved and hash-checked. No synthetic or duplicated respondents are added.

## Sources and extra inputs

Source: [CDC BRFSS 2022 public-use annual data](https://www.cdc.gov/brfss/annual_data/annual_2022.html), using the previously downloaded official ASCII archive and codebook. The target remains SDHSTRE1: rarely/never, sometimes, usually/always feeling stressed in the past 30 days. The highest category is a self-reported frequency, not a clinical diagnosis.

The original twelve predictors are supplemented with:

| Predictor | CDC field | ASCII column | Valid codes |
| --- | --- | ---: | --- |
| Social isolation frequency | SDHISOLT | 363 | 1 always to 5 never |
| Lost employment or reduced hours in past year | SDHEMPLY | 364 | 1 yes, 2 no |
| Food ran out and could not afford more in past year | SDHFOOD1 | 366 | 1 always to 5 never |
| Unable to pay housing or utility bills in past year | SDHBILLS | 367 | 1 yes, 2 no |
| Threatened utility shutoff in past year | SDHUTILS | 368 | 1 yes, 2 no |
| Transportation prevented necessary activities in past year | SDHTRNSP | 369 | 1 yes, 2 no |
| Could not afford a needed doctor visit in past year | MEDCOST1 | 111 | 1 yes, 2 no |

We exclude unknown, refused and missing responses rather than inventing values. The expanded CSV retains original codes, source row numbers, composite respondent IDs and derived fields separately. It contains every eligible complete case, not a selected easy-to-predict subset. Exclusions are applied before model evaluation and recorded in the audit.

Social isolation is correlated with stress, but is a separate contemporaneous survey question, not a value computed from the target. It can share response-style bias. Financial and access questions cover the prior year and do not establish causation. Stress-answer derivatives, mentally unhealthy days, depression diagnosis, life satisfaction, state and respondent identifiers are not model inputs.

## Splits and baseline comparison

Group by the original twelve-feature profile, using a stable canonical SHA-256 hash. Identical profiles cannot cross splits even if added answers differ. Profiles used for previous training remain training-only. All previous validation/test profiles are quarantined from this experiment. Previously unused profiles are assigned deterministically to 70% training, 10% validation and 20% test by hash bucket. The proportions are approximate, and previous-profile handling changes overall totals.

The new validation and test partitions contain no IDs or twelve-feature profiles from the original 30,000 respondents. The original and tuned models are evaluated on exactly the same new people as the expanded models. This is a new internal same-year holdout, not independent external or prospective validation. Earlier scores on 5,998 different people should not be used as a direct before/after comparison here.

## Recall-constrained selection

For each model family, validation usually/always recall must exceed the better of its two prior versions by 0.02. Validation macro F1 must at least match the better previous value. Among qualifying candidates, maximize accuracy, then macro F1. A fixed small class-score multiplier grid is part of validation selection. These adjusted estimates are not calibrated clinical probabilities. All six selections are frozen before final evaluation; final outcomes never cause retuning.

Candidates that fail validation are retained only for transparent reporting. The final deployment gate additionally requires no lower observed usually/always recall and no lower accuracy than **either** prior version on the same test cohort. This sample-level gate cannot guarantee no missed cases on future people, or that every previously detected individual remains detected. All per-class precision/recall and confusion matrices are retained, including any trade-offs.

Neural architectures operate on tabular feature order, not time series. Class weighting, early stopping and training-only scaling are used. No new dependencies or fabricated labels are needed. Unweighted complete-case selection and optional-module coverage limit representativeness. An 85% accuracy target is aspirational, not a reporting rule.

## Evidence files

- `dataset_audit.json`: counts, exclusions, sources and split membership summary.
- `protocol.json`: predeclared search and evaluation rules.
- `validation_trials.json`: all candidate/threshold trials.
- `comparison.json`: same-cohort results for all three versions.
- `test_predictions.csv`: probabilities and predictions for independent reproduction.
- `preserved_hashes.json`: original-data/model integrity checks.

The live dashboard is not switched automatically by this experiment. Promotion requires passing the gates and updating the input form for all nineteen predictors. Before final results were inspected, an additional safety check was defined: the majority-vote ensemble must also match or exceed both earlier ensembles in observed accuracy and high-frequency recall. The independent verification writes `deployment_gate.json`; inference refuses to activate the expanded version unless all individual models and the ensemble pass.
