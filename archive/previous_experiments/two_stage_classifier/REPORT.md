# Two-stage stress classifier experiment

## Decision

Status: **rejected_in_training_cv**. No candidate met the predeclared accuracy and recall/safeguard requirements. Validation and holdout not evaluated.

The six active models, preprocessing, ensemble policy, original dataset and website remain unchanged. Active artifacts and relevant sources were backed up with matching SHA-256 hashes. No deployment was performed.

## What was tested

- Stage A: probability of sometimes or high stress.
- Stage B: probability of high stress conditional on sometimes/high; trained only on actual minority-class respondents.
- Combined scores: rarely = 1−A; sometimes = A×(1−B); high = A×B. Both models run for every respondent; no hard first-stage gate.
- Two fixed candidates: no binary class weighting, and inverse-frequency weights raised to 0.5 in each stage. These are raw model scores, not established calibrated population probabilities.
- Original 19 input features, unchanged respondents and class labels. No added predictors, fabricated cases, resampling of evaluation data or time-series claims.

## Evaluation safeguards

Three stratified profile-group folds use the original training split. Each outer training portion reserves a grouped 20% calibration split. Preprocessing is fitted only on fitting rows. All candidates and the multiclass baseline use identical rows. Stage-two fitting uses only the minority subset of the fit portion.

Decision thresholds are chosen on calibration rows only to maximize accuracy while preserving reference high/sometimes recall, macro-F1 and QWK; high recall must also be at least 60%. Scoring folds never select their own thresholds. A candidate must improve pooled accuracy, preserve pooled safeguards and meet safeguards in every scoring fold. Infeasible calibration fallbacks are diagnostic, never eligible.

The comparison baseline is a freshly refitted multiclass XGBoost using the current decision multipliers, not a production model evaluated on its own training rows. These results cannot be substituted for the website’s historical accuracy.

## Training-only out-of-fold results

| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Multiclass reference | 67.59% | 64.51% | 21.60% | 0.5313 | 0.5096 | 0.7852 |
| Two-stage unweighted | 67.44% | 65.20% | 21.41% | 0.5312 | 0.5132 | 0.7882 |
| Two-stage weighted | 67.57% | 65.19% | 21.24% | 0.5314 | 0.5130 | 0.7863 |
| Always rarely | 65.86% | 0% | 0% | — | — | — |

The historical reused internal-holdout always-rarely baseline is 65.78% accuracy with 0% high/sometimes recall. It is quoted for context only; that holdout was not reopened for rejected candidates.

| Candidate vs reference | Accuracy change (percentage points) | Extra high cases detected | Extra sometimes cases detected | Eligible? |
|---|---:|---:|---:|---|
| unweighted | -0.141 | +104 | -52 | No |
| weighted | -0.017 | +102 | -103 | No |

## Fold-level checks

| Fold | Candidate | Accuracy change (points) | High recall change (points) | Sometimes recall change (points) | Calibration feasible | Scoring safeguards |
|---|---|---:|---:|---:|---|---|
| 1 | unweighted | -0.243 | +0.579 | -0.640 | True | False |
| 1 | weighted | -0.108 | +0.499 | -0.371 | True | False |
| 2 | unweighted | +0.061 | -0.040 | +0.618 | True | False |
| 2 | weighted | +0.182 | -0.142 | +0.063 | True | False |
| 3 | unweighted | -0.241 | +1.493 | -0.520 | True | False |
| 3 | weighted | -0.126 | +1.629 | -0.775 | True | False |

## Raw argmax diagnostics

| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Two-stage unweighted | 70.44% | 40.05% | 18.84% | 0.5213 | 0.4716 | 0.7882 |
| Two-stage weighted | 69.92% | 50.17% | 21.47% | 0.5383 | 0.5066 | 0.7863 |

A higher raw argmax accuracy does not satisfy the request if minority recall falls. The thresholded comparison above is the primary experiment.

## Interpretation and limitations

This bounded experiment evaluates one conditional architecture with two weighting choices. Failure does not prove that no two-stage model can ever improve; it does mean there is no evidence here to replace production. Do not repeatedly tune against these scoring folds or report the best fold as overall performance.

Candidate fitting, calibration and scoring totaled 12.3 seconds across six pairs, excluding shared baseline fits, setup and file verification. No paid compute or new dependencies were used.

One seed and correlated folds do not establish a confidence guarantee. The baseline is a fold-refitted XGBoost, not the active ensemble. A surviving candidate would still require frozen comparison against the current XGBoost and ensemble on reused validation, one governed final evaluation and backend runtime checks.

## Saved files and reproduction

- protocol.json: predeclared settings, selection rules and limitations.
- backup/, original_hashes.json: preserved current artifacts and source hashes; CSV preserved in place.
- source/, frozen_source_hashes.json: frozen experiment source and protocol hashes.
- roles.csv: exact fitting, calibration and scoring membership.
- oof.npz: training-only probabilities, decisions and respondent IDs.
- fold_*.json, cv_summary.json: thresholds, precision/recall/F1, confusion matrices, detected/missed counts, AUC and timing.
- decision.json, verification.json: outcome and integrity checks.
- tools/experiment_two_stage.py and tools/report_two_stage.py: use the existing WSL ML environment and a new experiment output directory for a rerun. No existing run is overwritten.

No rollback is required because no active file changed. If considering a new experiment, define its hypothesis and safeguards before looking at its evaluation results.
