# Excluded BRFSS feature review: depression history

Decision: **rejected_on_final_evaluation**. Production data, all six active models and inference remain unchanged.

## Feature review

| Field | Decision | Reason | Training usable coverage |
|---|---|---|---:|
| ADDEPEV3 | test | Ever told depressive disorder; not calculated from SDHSTRE1. Available as self-report at assessment. Diagnosis timing versus current stress unknown; not valid evidence for future prediction. | 99.63% |
| MENTHLTH | exclude | Explicitly includes stress in the same30-day question; strong conceptual overlap. | 99.05% |
| LSATISFY | exclude | Current life satisfaction is not literal target leakage, but overlapping concurrent well-being; keep task conservative. | 99.49% |
| POORHLTH | exclude | Concurrent mental/physical limitation; skip depends on mental-health days. | 57.01% |
| DECIDE | exclude | Concurrent mental/emotional concentration symptoms; attribution and timing ambiguous. | 99.60% |
| DIFFALON | exclude | Concurrent mental/emotional functional difficulty; attribution and timing ambiguous. | 99.84% |
| ACEDEPRS | exclude | Historical household exposure is not literal target leakage, but optional-module missingness and sensitive retrospective context make it unsuitable for this bounded experiment. | 7.93% |

Review source: the project’s locally supplied official BRFSS 2022 codebook (USCODE22_LLCP_102523.HTML), preserved as codebook metadata in feature_review.json. Correlation alone did not determine eligibility.

ADDEPEV3 asks whether the respondent was ever told they had a depressive disorder. It is not computed from the stress-frequency label. It is suitable to test for concurrent assessment when the user can supply that history. The survey does not establish diagnosis timing relative to current stress, so this experiment cannot certify a future-prediction task as leakage-free.

Codes 1 and 2 represent yes/no; unknown, refusal and missing remain an explicit third category. No respondent was dropped, and no unknown response was turned into no. Depression history is sensitive, and diagnosis access can differ between respondents. A future form would need an optional, accurately worded question and an unknown response.

Life satisfaction and childhood household exposure are not automatically target-derived leakage. They were excluded here to keep the experiment narrow and avoid concurrent well-being overlap or optional-module coverage problems. MENTHLTH explicitly includes stress and stays excluded.

## Fixed experiment

Original 128,248 training respondents, three stratified profile-group folds. Same fitting, inner calibration and scoring rows for original 19 inputs versus 20 inputs including depression history. Each outer training fold reserves 20% for calibration. Missing handling and preprocessing are fitted without scoring labels.

Fixed XGBoost settings: 350 trees, depth 5, learning rate 0.04, minimum child weight 12, L2 5, row/column sampling 0.85, four CPU threads. Training inverse-frequency weights have exponent 0.5. No broad hyperparameter search, oversampling or changed labels.

Thresholds are selected only on inner calibration. They maximize accuracy while preserving the reference high/sometimes recalls, macro-F1 and QWK, with a minimum 60% high recall. Infeasible threshold fallbacks are diagnostic, never eligible. The reference is a fold-refitted XGBoost using the active original decision multipliers, not a production model scored on its own training data.

## Training-only out-of-fold comparison

| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Original 19 inputs | 67.62% | 64.55% | 21.86% | 0.5325 | 0.5115 | 0.7853 |
| Plus depression history, calibration-selected rule | 68.97% | 65.55% | 22.21% | 0.5467 | 0.5392 | 0.7957 |
| Plus depression history, unchanged rule | 68.50% | 66.27% | 23.57% | 0.5466 | 0.5354 | 0.7957 |
| Always rarely | 65.86% | 0% | 0% | — | — | — |

Historical reused internal-holdout always-rarely baseline: 65.78% accuracy, zero high/sometimes recall. It is not directly comparable to these training-fold results.

With the calibration-selected rule, accuracy changed by +1.356 percentage points. High cases detected changed by +150; sometimes cases detected changed by +98.

| Fold | Baseline accuracy | New accuracy | High recall change (points) | Sometimes recall change (points) | Calibration feasible | Safeguards pass |
|---|---:|---:|---:|---:|---|---|
| 1 | 67.80% | 69.15% | +1.383 | +0.990 | True | True |
| 2 | 67.62% | 68.94% | +1.527 | +0.021 | True | True |
| 3 | 67.43% | 68.83% | +0.060 | +0.021 | True | True |

## Reused-validation selection

| Comparison | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Componentwise strict reference | 68.43% | 65.04% | 23.86% | 0.5433 | 0.5143 | 0.7839 |
| Candidate | 68.84% | 65.43% | 23.86% | 0.5509 | 0.5386 | 0.7927 |
| Always rarely | 65.68% | 0% | 0% | — | — | — |

The reference combines the stricter current XGBoost/ensemble acceptance requirements; it is not a single model. Thresholds were selected here, so this is not an independent evaluation.

## Frozen final evaluation on reused internal holdout

| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Current XGBoost | 67.54% | 65.33% | 20.95% | 0.5295 | 0.5090 | 0.7849 |
| Current Ensemble | 68.39% | 60.14% | 22.68% | 0.5393 | 0.5202 | 0.7852 |
| Frozen depression-history candidate | 68.68% | 65.14% | 23.10% | 0.5469 | 0.5368 | 0.7949 |
| Always rarely | 65.78% | 0% | 0% | — | — | — |

| Compared with | Accuracy change (points) | Extra high detected | Extra sometimes detected | Accuracy change 95% interval (points) | Pass? |
|---|---:|---:|---:|---|---|
| XGBoost | +1.133 | -7 | +148 | [+0.794, +1.471] | False |
| Ensemble | +0.290 | +181 | +29 | [-0.042, +0.606] | False |

Paired stratified bootstrap intervals are descriptive. They do not correct for historical reuse of this holdout. No thresholds or model settings were changed after this evaluation. A failed gate retains the original model even when another metric improves.

## Decision and limitations

A replacement requires strictly greater accuracy and preserved high/sometimes recalls, macro-F1 and QWK; training screening also requires safeguards in every fold. Any survivor still needs frozen validation/final evaluation and runtime/UI integration checks. Failure at an earlier gate keeps the current models and avoids opening the test set.

Validation evaluated: True. Reused holdout evaluated: True. Production changed: False.

This experiment tests one feature and one bounded modeling recipe. A failure does not establish a permanent accuracy ceiling. A gain does not guarantee unchanged recall on new respondents. Data are cross-sectional, complete-case US survey records; no causal, diagnostic or population-weighted claim is made.

## Saved evidence

- feature_review.json: exact codebook wording, exclusions and coverage.
- protocol.json, frozen_hashes.json: predeclared settings and frozen source.
- backup/, original_hashes.json: original active artifacts and identities.
- roles.csv, oof.npz: group-safe training-fold roles and predictions.
- fold_*.json, summary.json: precision/recall/F1, confusion matrices, AUC and detected/missed counts; raw argmax is diagnostic only.
- decision.json, verification.json: outcome and data/model integrity checks.

No rollback is needed because active models and data were not changed.
