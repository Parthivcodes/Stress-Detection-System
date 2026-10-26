# BRFSS additional-predictor audit — 25 October 2026

## Outcome and scope

The current six models, active ensemble, website and datasets were not changed. This experiment audits original BRFSS variables and runs a fixed-setting XGBoost diagnostic on training data only. It does not establish a new production accuracy.

All 183,040 existing respondents matched their original records in the 445,132-row 2022 source. Respondent ID, source row and stress target agreed for every match. The separate extract preserves original codes; missing values are cleaned only in experiment memory.

## What the errors reveal

These are descriptive results on the historically reused validation set, not an untouched test. Each saved model uses its existing decision multipliers; the ensemble averages raw probabilities using its existing weights and cutoff.

| Current model | Accuracy | Sometimes recall | High recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 67.63% | 18.80% | 64.47% | 0.5226 | 0.4943 | 0.7790 |
| Random Forest | 66.46% | 25.72% | 62.00% | 0.5337 | 0.4954 | 0.7665 |
| XGBoost | 67.45% | 21.41% | 65.04% | 0.5307 | 0.5063 | 0.7839 |
| 1D CNN | 68.91% | 15.39% | 52.05% | 0.5116 | 0.4748 | 0.7693 |
| LSTM | 69.04% | 21.41% | 57.05% | 0.5374 | 0.5081 | 0.7835 |
| CNN-LSTM | 67.82% | 18.36% | 60.60% | 0.5208 | 0.4998 | 0.7722 |
| Ensemble | 68.43% | 23.86% | 60.03% | 0.5433 | 0.5143 | 0.7842 |
| Always rarely (this validation split) | 65.68% | 0% | 0% | — | — | — |

Historical reused internal-holdout reference: always rarely = 65.78% accuracy with 0% sometimes/high recall. That historical figure is not a new evaluation; the split-specific baselines above and below are the relevant comparisons.

| True class | Cases | Missed by all six | Shared miss rate |
|---|---:|---:|---:|
| Rarely | 9,981 | 723 | 7.24% |
| Sometimes | 3,437 | 2,055 | 59.79% |
| High | 1,779 | 486 | 27.32% |

Shared errors suggest missing predictive information or ambiguous classes. They do not prove an irreducible ceiling. An oracle choosing the correct model per respondent is not a deployable ensemble.

## Additional fields and training coverage

All 39 candidates were declared before their associations were examined. Two fixed groups were tested: six context fields (25 total inputs), then those six plus 14 physical-health and three lifestyle fields (42 total inputs). Coverage below means a usable official response, not merely a nonblank value.

| Variable | Meaning | Usable training responses | Group |
|---|---|---:|---|
| CHILDREN | Number of Children in Household | 99.70% | context |
| RENTHOM1 | Own or Rent Home | 99.80% | context |
| PERSDOC3 | Have Personal Health Care Provider? | 99.44% | context |
| CHECKUP1 | Length of time since last routine checkup | 99.21% | context |
| FOODSTMP | During the past 12 months have you received food stamps | 99.87% | context |
| PRIMINSR | What is Primary Source of Health Insurance? | 97.97% | context |
| CVDINFR4 | Ever Diagnosed with Heart Attack | 99.62% | physical |
| CVDCRHD4 | Ever Diagnosed with Angina or Coronary Heart Disease | 99.31% | physical |
| CVDSTRK3 | Ever Diagnosed with a Stroke | 99.78% | physical |
| ASTHMA3 | Ever Told Had Asthma | 99.74% | physical |
| CHCCOPD3 | Ever told you had C.O.P.D. emphysema or chronic bronchitis? | 99.73% | physical |
| CHCKDNY2 | Ever told you have kidney disease? | 99.72% | physical |
| DIABETE4 | (Ever told) you had diabetes | 99.89% | physical |
| HAVARTH4 | Told Had Arthritis | 99.62% | physical |
| CHCSCNC1 | (Ever told) (you had) skin cancer that is not melanoma? | 99.47% | physical |
| CHCOCNC1 | (Ever told) (you had) melanoma or any other types of cancer? | 99.61% | physical |
| DIFFWALK | Difficulty Walking or Climbing Stairs | 99.80% | physical |
| DIFFDRES | Difficulty Dressing or Bathing | 99.92% | physical |
| DEAF | Are you deaf or do you have serious difficulty hearing? | 99.78% | physical |
| BLIND | Blind or Difficulty seeing | 99.84% | physical |
| SMOKE100 | Smoked at Least 100 Cigarettes | 99.59% | lifestyle |
| USENOW3 | Use of Smokeless Tobacco Products | 99.93% | lifestyle |
| ECIGNOW2 | Do you now use e-cigarettes, or vaping products every day, some days, or not at all? | 99.84% | lifestyle |
| VETERAN3 | Are You A Veteran | 99.94% | audit_only |
| ASTHNOW | Still Have Asthma | 14.85% | audit_only |
| SMOKDAY2 | Frequency of Days Now Smoking | 40.60% | audit_only |
| ALCDAY4 | Days in past 30 had alcoholic beverage | 99.32% | audit_only |
| AVEDRNK3 | Avg alcoholic drinks per day in past 30 | 54.83% | audit_only |
| DRNK3GE5 | Binge Drinking | 54.70% | audit_only |
| MAXDRNKS | Most drinks on single occasion past 30 days | 54.03% | audit_only |
| CAREGIV1 | Provided regular care for family or friend | 24.68% | audit_only |
| CRGVHRS1 | How Many Hours Do You Provide Care For Person? | 4.66% | audit_only |
| MENTHLTH | Number of Days Mental Health Not Good | 99.05% | excluded_overlap |
| LSATISFY | Satisfaction with life | 99.49% | excluded_overlap |
| ADDEPEV3 | (Ever told) you had a depressive disorder | 99.63% | excluded_overlap |
| POORHLTH | Poor Physical or Mental Health | 57.01% | excluded_overlap |
| DECIDE | Difficulty Concentrating or Remembering | 99.60% | excluded_overlap |
| DIFFALON | Difficulty Doing Errands Alone | 99.84% | excluded_overlap |
| ACEDEPRS | Live With Anyone Depressed, Mentally Ill, Or Suicidal? | 7.93% | excluded_overlap |

### Interpretation and missingness

- Housing tenure, children, healthcare access, insurance and SNAP receipt supply additional household context. US insurance categories and SNAP are not directly transferable to an Indian questionnaire.
- Physical history and disability questions add sensitive information and form burden; they are available at survey time, but this cross-sectional dataset does not establish causality or prediction of future stress.
- All training respondents remain in the pilot. Categorical missing values receive an explicit category; the number of children uses a fit-fold median and a missing indicator. Complete-case filtering would unnecessarily discard respondents.
- Missing codes are variable-specific: insurance codes 7, 9 and 88 are valid; checkup code 8 means never; children code 88 means zero. Refusal/unknown and blank responses are not relabeled as healthy answers.
- Conditional asthma, smoking and alcohol follow-ups require skip-aware handling. Alcohol frequency uses weekly/monthly units and requires further plausibility checks. Caregiving has only 24.68% usable training responses and caregiving hours 4.66%; these are optional/conditional questions. These audit-only fields were not modeled.
- MENTHLTH explicitly includes stress in the same reporting period. It and previously excluded life satisfaction/depression fields remain excluded, along with concurrent mental/emotional limitation fields. Not every excluded variable is literally derived from the target: some raise overlap, sensitivity or task-definition concerns instead.
- The existing unhealthy_days input is PHYSHLTH (physical unhealthy days), not MENTHLTH. Strong univariate association is not proof of added predictive value.

## Fixed training-only pilot

Three stratified group folds use the original 128,248 training respondents. Each outer training portion reserves a grouped inner calibration split. Preprocessing is fitted only on the fit portion. All three variants use identical rows and XGBoost settings. Calibration chooses decision multipliers; outer scoring labels never choose them.

The reference is a freshly fitted, fixed-setting 19-input XGBoost with the original decision multipliers, not the deployed model evaluated on these same training rows. An added-fields result cannot be compared directly with the website’s historical holdout accuracy.

| Training OOF variant | Accuracy | Sometimes recall | High recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| 19 inputs, original decision rule (reference) | 67.62% | 21.86% | 64.55% | 0.5325 | 0.5115 | 0.7853 |
| baseline — calibration-selected rule | 67.62% | 21.86% | 64.55% | 0.5325 | 0.5115 | 0.7853 |
| context — calibration-selected rule | 67.16% | 23.77% | 65.03% | 0.5349 | 0.5109 | 0.7856 |
| all_safe — calibration-selected rule | 67.30% | 23.84% | 65.53% | 0.5369 | 0.5142 | 0.7879 |
| Always rarely (training OOF) | 65.86% | 0% | 0% | — | — | — |

| Variant vs fixed reference | Accuracy change (points) | Extra high cases detected | Extra sometimes cases detected | Every fold passes safeguards? |
|---|---:|---:|---:|---|
| baseline | +0.000 | +0 | +0 | Yes |
| context | -0.454 | +72 | +547 | No |
| all_safe | -0.315 | +147 | +566 | No |

A fold passes only when its inner calibration is feasible and outer high recall is at least the higher of 60% and the reference, with sometimes recall, macro-F1 and QWK also preserved. Infeasible fallback operating points are diagnostic failures, not eligible candidates.

### Same-rule comparison to separate new features from decision changes

| Variant, original decision multipliers | Accuracy | Sometimes recall | High recall | Macro-F1 | QWK | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 67.62% | 21.86% | 64.55% | 0.5325 | 0.5115 | 0.7853 |
| context | 67.65% | 21.91% | 64.47% | 0.5327 | 0.5120 | 0.7856 |
| all_safe | 67.80% | 21.92% | 64.88% | 0.5344 | 0.5155 | 0.7879 |

At the unchanged decision rule, 23 extra fields gave a pooled +0.184 percentage-point accuracy change, +50 high cases and +15 sometimes cases detected. This is a small exploratory signal, not a qualifying replacement. Fold 0 missed five more high cases; fold 1 missed eight more sometimes cases. The calibrated operating points did not retain the accuracy gain.

Raw argmax results, per-class precision/F1, confusion matrices, detected/missed counts and per-fold timing are in pilot_summary.json and pilot_<fold>_<variant>.json. These descriptive diagnostics have no population weighting or confidence guarantee. One seed and three correlated folds cannot establish a reliable small improvement.

## Decision and next step

No added-field group passed the declared improvement screen. Keep the current models. These results do not justify increasing the website’s advertised accuracy or adding 23 questions to its form. A further experiment would need a new hypothesis and frozen protocol; repeatedly adjusting against these scoring folds would overfit the analysis.

Mental-health/depression/satisfaction inputs would require an explicit change of task and questionnaire rationale. Their stronger associations alone do not justify adding them to inflate accuracy. Fewer respondents, deleting difficult cases, changing label balance or relabeling missing values would not be a fair improvement.

## Reproduction and files

- protocol.json and candidate_spec.json: frozen experimental rules and variable-specific coding.
- original_hashes.json and verification.json: protected-file hashes and checks.
- codebook_catalog.json: locally supplied official BRFSS codebook metadata.
- same_respondent_raw_candidates.csv and extraction.json: separate unchanged-code extract and join checks; the original dataset remains intact.
- coverage.csv, training_associations.csv: availability and descriptive training-only associations.
- validation_errors.json, validation_error_overlap.csv, validation_subgroups.csv: all-six error analysis on reused validation data. Small subgroup recalls are suppressed below 30 class cases.
- pilot_roles.csv: fit/calibration/score membership; pilot_oof.npz: training-only out-of-fold probabilities and decisions.
- pilot_summary.json and pilot_<fold>_<variant>.json: metrics and decision settings. pilot_feature_gain.csv is fitted gain importance, not causal evidence.
- Reproduction scripts: audit_brfss_predictors.py (catalog/extract), analyze_predictor_audit.py (audit/pilot), verify_predictor_audit.py, report_predictor_audit.py. Use a fresh experiment directory and the existing ML environment; extraction runs in Windows Python, analysis in the existing WSL environment.

No rollback is necessary: no production artifact or active pointer was changed. The audit kept raw observations separate from derived diagnostics, following the data-analysis skill’s preservation guidance.
