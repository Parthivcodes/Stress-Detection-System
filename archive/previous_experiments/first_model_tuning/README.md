# Six-model hyperparameter tuning

Completed: 36 grouped cross-validation fits and nine final fits (six model families,
with two fixed seeds for each neural family). No candidate passed all predeclared
performance safeguards. All current models and the current ensemble remain active.

Always rarely baseline: 65.78% accuracy and 0% high recall on the reused holdout.

| Model | Deployed accuracy | Matched original accuracy | Candidate accuracy | Deployed high recall | Candidate high recall | Performance gate |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 67.46% | 67.38% | 67.45% | 64.14% | 64.86% | False |
| Random Forest | 66.58% | 63.35% | 61.66% | 61.10% | 62.18% | False |
| XGBoost | 67.54% | 67.67% | 67.63% | 65.33% | 65.25% | False |
| 1D CNN | 68.83% | 66.33% | 67.33% | 52.46% | 60.47% | False |
| LSTM | 68.96% | 68.78% | 68.45% | 56.74% | 60.00% | False |
| CNN-LSTM | 67.84% | 67.88% | 68.66% | 60.03% | 60.91% | False |
| Ensemble | 68.39% | 68.39% | 68.39% | 60.14% | 60.14% | False |

## Interpretation
CNN-LSTM illustrates why accuracy alone is insufficient: its candidate improved
test accuracy from 67.84% to 68.66% and high recall from 60.03% to 60.91%, but
sometimes recall fell from 18.24% to 8.47%. Its macro-F1 and QWK also declined.
It was already ineligible on validation and was not activated.

XGBoost's small accuracy increase versus its deployed decision came with three
additional high-stress misses on test. It also failed validation safeguards.

The ensemble row remains the existing blend because no tuned member qualified;
the other predeclared blends did not beat all validation safeguards. It is not
evidence of a newly improved ensemble.

Candidates were ranked using training-only grouped cross-validation; final thresholds and eligibility used validation. Final test was opened once after freezing. It is a historically reused holdout, not independent evidence.
Strict safeguards require non-degradation in middle recall, macro-F1 and QWK as well as accuracy and high recall. A candidate can improve accuracy and still be rejected.
No model widths/layers or feature encoding were changed in this bounded pilot. Neural timing caps can stop training before 14 epochs; see the recorded histories. Results do not establish that further tuning cannot help.
Secondary neural seeds measure instability; they are never selected because they look better.
The survey is cross-sectional. An LSTM processes a fixed feature ordering, not patient history or genuine temporal signals.

## Reproduce and inspect
Use tools/tune_six_models.py with phases audit, search, select, test, report and --project/--output. Audit refuses existing directories; test refuses a second pass. Do not create new experiments simply to retune against test.
protocol.json: predeclared settings; audit.json: data checks; cv/: every CV result; cv_split_ids.csv/final_train_ids.csv: group-safe roles; candidates/: separate models and preprocessors; frozen_selection.json: frozen choices; test_report.json: detailed comparisons and paired intervals; comparison.csv: summary; backup/: protected originals.

`Hyperparameter_Tuning.ipynb` contains ten successfully executed code cells,
including actual training-only preprocessing reproduction and checks against the
saved candidate preprocessor. Open it from the project or experiment directory.
It does not automatically train models or evaluate test respondents again.
`trial_history.csv`, `detailed_comparison.csv`, and `rejection_reasons.json`
provide expanded evidence. See `SEARCH_LIMITATIONS.md` for deferred search
dimensions and statistical caveats; `environment.json` records library versions.

## Runtime and preservation checks

- Original data, model, schema, policy and source hashes remain unchanged.
- The historical layout check passed for all 47 original data/model files.
- Synthetic-data checks passed for cutoff boundaries, grouped partition isolation
  and non-degradation gates.
- The retained installed API served 512 validation-input requests, producing
  3,072 individual predictions with the documented soft-ensemble decision rule.
- The Linux API test ran under a 512 MiB memory limit with swap disabled and no
  OOM events. This run's final service peak was approximately 428 MiB; memory
  accounting/cache state varies, so it is not a guarantee of cloud behavior.
- Authentication, invalid input, schema and benchmarks checks passed. Concurrent
  overload returned deliberate 503 backpressure; the health endpoint remained up.
- `runtime_verification.json` contains the local evidence. No cloud deployment,
  paid compute, dependency installation or model activation was performed.

## Rollback
The active models and source are unchanged. There is nothing to roll back unless a separate installation step is performed. Backups and original hashes provide recovery and verification.

## Viva explanation
Hyperparameters are settings chosen before fitting, such as tree depth, regularization and learning rate. We tried a bounded set using grouped training folds, tuned stress cutoffs on separate calibration data, selected on validation, and evaluated once on the held-out respondents. Better accuracy is not accepted if it hides more high-stress cases.
