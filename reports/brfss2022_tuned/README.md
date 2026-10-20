# BRFSS tuning experiment

## Held-out results

The same 5,998 respondents were evaluated before and after tuning. Accuracy improved for five models; 1D CNN was retained unchanged because validation did not support a replacement.

| Model | Accuracy before | Accuracy after | Macro F1 before | Macro F1 after | Usually/always recall before | Usually/always recall after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 58.85% | 64.52% | 0.5015 | 0.5222 | 63.56% | 53.81% |
| Random Forest | 60.85% | 63.92% | 0.5234 | 0.5310 | 60.45% | 53.25% |
| XGBoost | 59.65% | 66.41% | 0.5202 | 0.5281 | 63.42% | 48.02% |
| 1D CNN | 63.62% | 63.62% | 0.5076 | 0.5076 | 48.59% | 48.59% |
| LSTM | 62.17% | 63.79% | 0.5309 | 0.5312 | 57.77% | 52.82% |
| CNN-LSTM | 61.74% | 66.02% | 0.5251 | 0.5089 | 60.03% | 49.72% |

These are **accuracy-oriented trade-offs**, not improvements on every measure. Balanced accuracy declined in all five changed models. CNN-LSTM's macro F1 declined and its sometimes-class recall fell from 36.36% to 21.21%. Only XGBoost and CNN-LSTM exceed the 64.85% majority-class baseline in accuracy, while every model substantially exceeds its macro F1. Differences have not been established as statistically significant or externally validated. No model was reselected after seeing these outcomes.

## Why accuracy changed

The original small lifestyle dataset and BRFSS use different targets and populations. Their accuracies are not directly comparable. BRFSS predicts three categories of self-reported stress frequency from twelve cross-sectional survey answers; those answers do not determine stress exactly.

In the first BRFSS run, all six models used inverse-frequency class weights. The neural checkpoints were selected by macro F1, not accuracy. This favors detecting uncommon answers rather than always choosing the most frequent response. The test-set majority-class baseline has 64.85% accuracy but only 0.2623 macro F1 and zero recall for both minority classes.

## What this experiment changes

The 30,000-row CSV, labels, respondent IDs, input columns, preprocessing and train/validation/test membership remain unchanged. Their hashes and the original model hashes are retained in `preserved_hashes.json`. No synthetic samples, oversampling, response edits, label changes, or test-set filtering are used.

We compare three settings per classical model, varying regularization, tree complexity and class weighting. Neural networks are fine-tuned from the first-run checkpoints with square-root class weights, a smaller learning rate, gradient clipping and early stopping. Every family also includes the unchanged original checkpoint as a candidate.

For each candidate, a small fixed grid adjusts predicted probabilities using the **training-set** class proportions: normalize(probability × class_proportion^exponent). This corrects part of the class-frequency distortion introduced by balanced training. It is a decision trade-off, not new evidence about a respondent and not clinical probability calibration. The exponent is chosen only on validation data. A selected original checkpoint with this adjustment is reported as such, not falsely described as a newly trained model.

## Selection rule fixed before the run

Maximize the mean of validation accuracy and macro F1, subject to validation macro F1 at least matching the original checkpoint, balanced accuracy declining by no more than 0.03, and usually/always recall remaining at least 0.30. The original version is the fallback. These are development criteria, not medically validated safety thresholds. Higher accuracy may still cost minority-class recall; report both.

All six selections are saved before test evaluation begins. `validation_trials.json` retains every attempted candidate/epoch/adjustment. `selection.json` records the final choices. `comparison.json` reports both versions, including per-class recall. `test_predictions.csv` permits independent reproduction of held-out metrics.

## Evaluation limitations

The same 5,998-person internal holdout was reported in the prior project iteration. It is reused for a paired before/after comparison, **not a newly untouched external test cohort**. We do not select or rerun models based on its outcomes. Validation reuse across candidates can cause selection optimism. New independently collected or separately reserved respondents would be needed for stronger confirmation. Dataset and population limitations from the original BRFSS report still apply.

Original BRFSS models remain in `models/brfss2022`; tuned artifacts are separate in `models/brfss2022_tuned`. Set `BRFSS_MODEL_VERSION=brfss2022` to use the original version. The active-version file controls the default only after verification. The server reports its model version and serves metrics from that same version.
