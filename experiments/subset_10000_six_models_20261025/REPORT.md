# Six models trained on the 10,000-respondent subset

Decision: **all_rejected**. Current data, all six active models and the website remain unchanged.

## Verification correction

Final verification found 4 comparison respondents sharing 4 profile groups with calibration. Those comparison rows were excluded solely by group identity. All models and thresholds stayed frozen. The tables below use the corrected 14,321-respondent comparison; initial results remain saved but are not authoritative.

Trained on 7,364 real respondents, calibrated thresholds on 872, and compared old/new on the same separate 14,321 respondents. The selected 1,764 test rows were not used for training or threshold selection. The full test set was not evaluated in this comparison.

## Accuracy and minority recall

| Model | Current accuracy | Subset accuracy | Current high recall | Subset high recall | Current sometimes recall | Subset sometimes recall | Pass? |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 67.77% | 65.46% | 64.88% | 67.62% | 18.80% | 20.71% | No |
| Random Forest | 66.55% | 59.99% | 62.43% | 67.56% | 25.52% | 33.67% | No |
| XGBoost | 67.59% | 60.79% | 65.59% | 69.71% | 21.20% | 32.90% | No |
| 1D CNN | 68.97% | 62.96% | 52.36% | 65.35% | 15.46% | 20.37% | No |
| LSTM | 69.16% | 63.10% | 57.60% | 72.45% | 21.54% | 21.85% | No |
| CNN-LSTM | 67.97% | 55.96% | 61.18% | 64.34% | 18.30% | 21.08% | No |
| Ensemble | 68.58% | 64.49% | 60.52% | 67.56% | 23.77% | 25.52% | No |
| Always rarely | 65.67% | 65.67% | 0% | 0% | 0% | 0% | No |

Arithmetic mean accuracy across the six individual models: current **68.00%**, subset **61.37%**. This mean is not ensemble accuracy. Historical reused-holdout always-rarely accuracy is 65.78%, with zero minority recall; the split-specific baseline above is the relevant comparison.

| Model | Change in high cases detected | Change in sometimes cases detected | Threshold calibration feasible? |
|---|---:|---:|---|
| Logistic Regression | +46 | +62 | False |
| Random Forest | +86 | +264 | False |
| XGBoost | +69 | +379 | False |
| 1D CNN | +218 | +159 | False |
| LSTM | +249 | +10 | False |
| CNN-LSTM | +53 | +90 | False |
| Ensemble | +118 | +57 | False |

## Training and safeguards

- Same 19 predictors and original target construction. IDs, stress_raw, raw_* and provenance columns are never model inputs.
- Classical models retain active hyperparameter configurations. Logistic regression and XGBoost training weights are recomputed from the subset training labels. Random forest retains its active weighting setting.
- Neural architectures are unchanged. A grouped internal 10% training development split selects epochs (maximum 14, patience 4), then each network is reinitialized and fitted on all 7,364 training respondents for that many epochs. One fixed seed; no best-seed selection.
- Scaling and encoding are fitted on the corresponding training rows only. Calibration and comparison rows are disjoint; profile groups never cross roles.
- The 872 calibration rows select one two-boundary rule per candidate, requiring high recall at least max(60%, current model recall), and no reduction in sometimes recall, macro-F1 or QWK. Any infeasible fallback is diagnostic only.
- Comparison rows do not choose thresholds. A candidate passes only with strictly better accuracy and preserved high/sometimes recall, macro-F1 and QWK. Each individual model has its own current-model reference.
- Ensemble weights stay LR 40%, XGBoost 50%, LSTM 10%. New raw probabilities are averaged before applying one calibration-selected rule; no majority vote.
- Current predictions are reused from saved probabilities only after dataset and checkpoint identity checks. These respondents belong to historically reused validation, not an untouched external test.

## Raw probability argmax diagnostics

| Model | Current raw accuracy | Subset raw accuracy | Subset raw high recall | Subset raw sometimes recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.05% | 68.36% | 49.79% | 27.35% |
| Random Forest | 63.33% | 62.90% | 55.34% | 39.38% |
| XGBoost | 68.82% | 67.90% | 48.12% | 30.15% |
| 1D CNN | 67.22% | 67.74% | 47.05% | 14.07% |
| LSTM | 69.54% | 68.22% | 54.26% | 25.06% |
| CNN-LSTM | 68.59% | 64.72% | 38.70% | 12.38% |

Raw argmax accuracy cannot qualify a replacement when minority recall falls. The primary comparison above uses the frozen calibration-selected decision rules.

## Interpretation

A smaller dataset does not create stronger predictive information. This experiment checks one predeclared training recipe per model; it does not prove that every possible smaller-data model will fail. Results do not establish 80% average accuracy. No records were fabricated, relabeled or selected by prediction correctness.

Selection on reused validation can be optimistic. Any qualifying candidate would require a frozen one-pass comparison on the reused internal holdout plus inference compatibility, memory and rollback checks before replacing production. A rejected candidate is not retuned using its comparison results.

## Files and verification

- protocol.json and frozen_hashes.json: frozen training, threshold and evaluation rules.
- backup/ and original_hashes.json: preserved active artifacts and source identities.
- Six new model files and preprocessor.joblib: isolated trained candidates, not active models.
- model_0.json through model_5.json: training histories, epoch choices, timing, file sizes and metrics.
- group_disjoint_comparison/summary.json: authoritative corrected accuracy, balanced accuracy, macro-F1, QWK, AUC, per-class metrics, confusion matrices and detected/missed counts.
- group_disjoint_comparison/roles.csv and saved probability files: corrected membership and inference outputs. Original top-level results are retained for audit only.
- group_disjoint_comparison/decision.json and verification.json: eligibility and checks. All six saved checkpoints reload and reproduce predictions within 1e-6; original and backup hashes match.

No production rollback is necessary: active data/model paths and inference code were not modified.
