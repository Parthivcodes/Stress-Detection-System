# Depression history with stronger high-stress weights and recall margins

Decision: **rejected_on_validation**. Current production data and models are unchanged. Previous experiments and their reported results were preserved.

## Frozen experiment

The same 19 original predictors plus ADDEPEV3 depression history were used. Its yes/no/unknown categories remain distinct. No mental-health-days, current life satisfaction or target-derived fields were added. This is concurrent stress-frequency assessment; diagnosis timing is unknown and no future-prediction claim is made.

Four high-class weight factors were declared before fitting: 1.00 (margin-only control), 1.10, 1.25 and 1.50. Each multiplies the high-class square-root inverse-frequency training weight. Other classes, XGBoost settings and labels are unchanged. Calibration, validation and holdout distributions are not reweighted.

During threshold calibration, high recall must reach max(60%, reference high recall) + 2 percentage points. Sometimes recall must reach reference sometimes recall + 0.5 percentage points. Targets are capped at 100%. Macro-F1 and QWK must also be preserved. These margins are cushions, not guarantees on new respondents.

Three stratified profile-group folds use all 128,248 original training respondents. Each outer training part reserves grouped20% calibration. Preprocessing fits only fitting rows. All candidates use identical role assignments. Outer scoring rows never choose thresholds. Fresh fold-refitted 19-input XGBoost with its existing decision multipliers is the reference.

Eligible candidates must improve pooled accuracy and preserve both recalls/F1/QWK, with feasible calibration and preserved safeguards in every scoring fold. Select one CV winner by accuracy, macro-F1, then lower weight; no switching after validation or test.

## Training-only out-of-fold comparison

| Model / high-class weight factor | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Original19 reference | 67.62% | 64.55% | 21.86% | 0.5325 | 0.5115 | 0.7853 |
| History, factor 1.0 | 68.35% | 67.84% | 22.96% | 0.5453 | 0.5375 | 0.7957 |
| History, factor 1.1 | 68.26% | 67.91% | 23.19% | 0.5456 | 0.5376 | 0.7950 |
| History, factor 1.25 | 68.34% | 67.79% | 22.96% | 0.5453 | 0.5370 | 0.7938 |
| History, factor 1.5 | 68.32% | 67.66% | 23.12% | 0.5454 | 0.5363 | 0.7920 |
| Always rarely | 65.86% | 0% | 0% | — | — | — |

Historical reused internal-holdout always-rarely baseline:65.78% accuracy and zero high/sometimes recall. It is not directly comparable with training-fold estimates.

| Factor | Accuracy change (points) | Extra high cases detected | Extra sometimes cases detected | Eligible for refit? |
|---|---:|---:|---:|---|
| 1.0 | +0.731 | +497 | +313 | True |
| 1.1 | +0.648 | +507 | +379 | True |
| 1.25 | +0.726 | +489 | +314 | True |
| 1.5 | +0.699 | +470 | +361 | True |

## Fold safeguards

| Fold | Factor | Calibration feasible | Accuracy change (points) | High recall change (points) | Sometimes recall change (points) | Scoring safeguards pass |
|---|---|---|---:|---:|---:|---|
| 1 | 1.0 | True | +0.507 | +3.605 | +1.211 | True |
| 1 | 1.1 | True | +0.430 | +3.683 | +1.653 | True |
| 1 | 1.25 | True | +0.550 | +3.566 | +1.327 | True |
| 1 | 1.5 | True | +0.468 | +3.585 | +1.169 | True |
| 2 | 1.0 | True | +0.809 | +3.515 | +0.725 | True |
| 2 | 1.1 | True | +0.690 | +3.777 | +1.409 | True |
| 2 | 1.25 | True | +0.772 | +3.475 | +0.778 | True |
| 2 | 1.5 | True | +0.800 | +3.415 | +0.841 | True |
| 3 | 1.0 | True | +0.875 | +2.740 | +1.335 | True |
| 3 | 1.1 | True | +0.824 | +2.600 | +0.911 | True |
| 3 | 1.25 | True | +0.856 | +2.660 | +1.180 | True |
| 3 | 1.5 | True | +0.828 | +2.320 | +1.759 | True |

## One frozen winner on reused validation

Selected high-class factor: 1.0. This validation set was historically reused and chooses the final threshold; it is not independent confirmation.

| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Current XGBoost | 67.45% | 65.04% | 21.41% | 0.5307 | 0.5063 | 0.7839 |
| Current Ensemble | 68.43% | 60.03% | 23.86% | 0.5433 | 0.5143 | 0.7842 |
| History plus weight/margin candidate | 68.24% | 67.79% | 24.41% | 0.5492 | 0.5381 | 0.7927 |
| Always rarely | 65.68% | 0% | 0% | — | — | — |

High recall calibration target: 67.04%. Sometimes target: 24.36%. Threshold feasible: True. Validation eligible: False.

## Decision and limitations

Validation evaluated: True. Holdout evaluated: False. Production changed: False.

A previous holdout result motivated this hypothesis. The new selection used training/validation only, but this does not make a historically reused holdout untouched. No particular missed respondents were targeted, and no scoring or test rows were dropped to raise accuracy.

A calibration recall cushion can trade accuracy for detection and may not survive new respondents. A failure retains the original model without trying a different candidate on the holdout. A passing internal result would still need input-schema, API/frontend, 512MB runtime and rollback checks before local use. No deployment is authorized here.

## Reproducible files

- protocol.json, frozen_hashes.json, source/: fixed candidates, selection rules and source snapshots.
- backup/, original_hashes.json: current and previous-experiment preservation.
- roles.csv and oof.npz: grouped memberships, probabilities and frozen per-fold decisions.
- fold_*.json and summary.json: raw argmax and decision metrics, confusion matrices, per-class precision/recall/F1, AUC and timing.
- selected.json and validation.json, when present: sole CV winner and subsequent validation outcome.
- decision.json and verification.json: final status and integrity checks.

No production rollback is necessary while production_changed is false.
