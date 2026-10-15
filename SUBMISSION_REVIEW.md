# University submission review

Reviewed 14 October 2026. Verdict: a solid demonstration and comparative ML project, but not an unconditional claim of submission readiness. The university rubric and original proposal were not supplied. Use the corrections below in the final report and viva.

## What was delivered and verified

- Replaced the dense sci-fi cockpit with the Still dashboard: sage palette, responsive cards, interactive Three.js particle sphere/orbits, reduced-motion preference, pause/reset controls, 12 profile inputs, presets, actual six-model outputs, benchmark table, and JSON download.
- Removed simulated predictions. Offline/backend failures produce an explicit unavailable state. Editing inputs invalidates old results; responses for changed profiles are discarded.
- Added API health endpoint, bounded finite numeric inputs, valid categorical inputs, blood-pressure consistency, 400 responses for bad input, and a local-only server binding.
- Formula-based heuristic indicators are explicitly labeled as illustrations, not feature attributions. They are not shown as explainability in the new UI.
- Reproduced all six saved-model accuracy and macro-F1 values. Verified training/inference preprocessing parity on a dataset row, scaler statistics, class probabilities, and invalid input rejection.
- HTTP page, health, and prediction smoke tests passed. JavaScript syntax passed. Browser rendering, mobile layout, keyboard interaction, and download interaction remain unverified because no browser connection was available.
- Model artifacts were preserved. Existing scikit-learn artifacts identify version 1.6.1; verification was rerun with that version and requirements now pin it.

## Data audit: the main concern

374 records, 132 unique predictor profiles, 242 additional repeated rows. Low/moderate/high class counts: 141/113/120. No identical predictor group has conflicting class labels.

The original seed-42 split contains 261 training, 38 validation, and 75 test rows. **58 of 75 test rows (77.3%) exactly match a training feature profile.** This is not target-column leakage: Stress Level and Person ID are excluded, and the scaler is fitted on training data. It is a dependence/duplicate-profile concern that the previous “zero leakage” wording overlooked. Unique Person IDs do not remove identical-feature overlap. Identical profiles do not prove the same person was measured repeatedly, so group-aware splitting is a conservative sensitivity analysis, not proof of participant independence.

See `reports/data_audit.json` for machine-readable evidence.

## Is more hyperparameter tuning required?

**A larger search is not supported by this experiment.** The original code uses fixed model settings plus neural early stopping; it does not contain a recorded systematic hyperparameter search.

The added experiment uses five outer StratifiedGroupKFold splits and three inner group-aware folds. Exact normalized predictor profiles define groups without using labels. Scaling for logistic regression is fitted within each training fold. Inner selection uses macro F1. Outer scores compare fixed settings and tuned settings on the same folds. Models are retrained for each fold; saved deployment artifacts are untouched.

| Model | Fixed macro F1, mean ± fold SD | Tuned macro F1, mean ± fold SD | Fixed accuracy |
|---|---:|---:|---:|
| Dummy majority | 0.1579 ± 0.0552 | — | 32.31% |
| Logistic Regression | 0.9437 ± 0.0434 | 0.9330 ± 0.0551 | 95.47% |
| Random Forest | 0.9454 ± 0.0452 | 0.9339 ± 0.0477 | 95.51% |
| XGBoost | 0.9533 ± 0.0559 | 0.9544 ± 0.0474 | 96.15% |

Search spaces: LR C={0.1,1,10}; RF depth={4,8,None} and minimum leaf size={1,3}; XGBoost depth={2,5} and learning rate={0.03,0.08}. Other parameters are fixed in the script.

Tuning reduced mean F1 for LR and RF; XGBoost improved by only 0.0010, small compared with fold variation. This is not a significance test and does not prove no better settings exist. Preserve the simple baselines; prioritize validation and documentation. Neural models were verified on the original holdout but were NOT retrained under grouped folds or tuned in this experiment. Do not use these new classical results to claim superiority over neural models evaluated under a different protocol.

Fold scores are dependent, fold sizes/class proportions vary, and SD is not a confidence interval. This is one seed on one small dataset; near-duplicate profiles may remain across groups. External validation is still absent. The old test set has already been inspected, so it is no longer a pristine future model-selection holdout.

## Corrections needed in the final academic narrative

1. Describe the task as classification from tabular physiological/lifestyle attributes. There are no measured waveform sequences in this dataset. LSTM/1D CNN operate over feature columns; attention does not make the model invariant to feature order. Avoid claims of validated temporal dynamics or real-time PPG measurement.
2. The thresholds ≤4, 5–6, ≥7 are chosen class bins, not standardized clinical categories. The custom cardiovascular index and quality/hours ratio are engineered features, not independently validated medical markers or conventional sleep efficiency.
3. Do not claim “100% proposal compliance” without the rubric. Reconcile all metrics with CSV outputs: the archived CNN-LSTM ROC-AUC is 0.9884 in `reports/model_comparison.csv`; older README/UI text used 0.9747. The new UI reads an export of the CSV.
4. Original ablation uses an 80/20 training/test split and different settings from the 70/10/20 main comparison. Its activity-only subset includes HR/activity and steps/awake-hours, which incorporate other modalities. Redefine pure modalities and run paired grouped folds before claiming that fusion itself improves performance.
5. Original permutation importance uses the test set. It may describe that fitted model, but do not reuse it to choose features and then present that test score as untouched. Report importance variability and use training-side folds for feature selection.
6. Mean probabilities and vote agreement are different quantities. Uncalibrated probabilities and the six-model voting rule need their own evaluation before claims about ensemble accuracy or reliability.
7. Document dataset source, license, collection method, duplicate handling, target definition, and consent/anonymization facts that can actually be verified. Add limitations, related work, architecture diagram, setup instructions, screenshots, references, and required university cover/declaration pages.
8. The old defense guides and report contain stronger claims than the code supports. This review supersedes those claims, but those documents have not been comprehensively rewritten. Integrate these corrections before handing them in.

## Reproducibility and next steps

Run `verify_project.py` and `validate_submission.py` using the README instructions. See `reports/grouped_validation_folds.csv` for selected parameters and per-fold results; `reports/reproduced_model_metrics.csv` for saved-model checks. Preserve dependency versions with the submission. Validation used Python 3.13.14, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.6.1, scipy 1.18.1, torch 2.14.1+cpu, and xgboost 3.4.1. scipy emits a harmless “unknown solver option: iprint” warning from the older scikit-learn optimizer wrapper; no fit failures occurred. A preliminary run with sklearn 1.9.1 produced different group allocations; the final CSVs use the artifact-compatible 1.6.1 environment.

For a classroom demo, the implementation and reproducible baseline comparison are substantial. For a research-focused submission, complete the grouped neural comparison, corrected ablation, repeated seeds, and uncertainty reporting if the rubric requires all six architectures to be compared fairly. Prefer these steps over an expensive blind parameter search.

Method references: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [group-aware cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
