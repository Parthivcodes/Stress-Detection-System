# Validation-selected soft ensemble and ordinal experiment

## Active choice

Weighted **raw** probabilities: Logistic Regression 0.4, XGBoost 0.5, LSTM 0.1.
High-stress cutoff: **0.33022436197988114**. If P(high) reaches the cutoff, select
usually/always; otherwise choose the larger rarely/never or sometimes probability.
This is an explicit thresholded decision, not argmax over all three probabilities.
Individual model cards keep their previous multipliers and predictions.

## Final reused-holdout comparison (30,718 records)

| Method | Accuracy | Macro F1 | QWK | Sometimes recall | High recall | Macro OVR AUC |
|---|---:|---:|---:|---:|---:|---:|
| Always rarely/never | 65.78% | .2645 | .0000 | 0% | 0% | .5000 |
| Previous majority vote | 68.62% | .5263 | .5070 | 18.56% | 59.89% | .7829 |
| Frozen soft ensemble | 68.39% | .5393 | .5202 | 22.68% | 60.14% | .7852 |
| Normal XGB, validation-matched cutoff | 68.31% | .5462 | .5235 | 25.29% | 60.17% | .7849 |
| Ordinal XGB, validation-matched cutoff | 67.80% | .5435 | .5217 | 25.69% | 60.14% | .7843 |

The new ensemble improves macro F1, QWK and middle/high recall versus the old vote,
but accuracy declines by 0.23 percentage points. It detects 2,177 of 3,620 high cases
versus 2,168 previously (nine fewer misses), and 1,563 middle cases versus 1,279.
This is not proof of statistically significant improvement or external validity.
The ensemble is **not universally best**: thresholded standalone XGB has better
macro F1, QWK and middle recall in both validation and test. We did not switch to it
after seeing test results; the requested ensemble search and choice were frozen.

## Selection protocol

- Run the six existing models on validation without retraining them.
- Train only two new binary XGBs: P(y >= 1) and P(y = 2), with the selected normal
  XGB hyperparameters, up to 350 rounds and 25-round early stopping on validation
  logloss. Both use the original three-class square-root inverse-frequency sample
  weights from training. Validation logloss is unweighted.
- Enforce monotonic probabilities by B=min(B,A); return [1-A, A-B, B]. No crossing
  occurred on validation. These are ordinal-aware binary tasks, not a jointly
  trained proportional-odds model.
- For each candidate, use the highest validation cutoff achieving at least 60%
  high recall (60.0337% here). This matches recall up to tied scores; it does not
  guarantee exactly the same recall on the test population.
- Search positive weights in 0.1 steps: nine XGB+LSTM mixes and 36 LR+XGB+LSTM
  mixes. Rank by macro F1, then QWK, sometimes recall, AUC and accuracy.
- Ordinal replaces normal XGB in that search only if validation macro F1 improves
  and QWK does not decrease. Ordinal F1=.545377 vs .544520, but QWK=.519280 vs
  .519955. It failed that predeclared gate and is retained only as an experiment.
- Freeze `frozen_selection.json` before opening the final test pass. The
  `TEST_STARTED.json` hash identifies the frozen choice. The script refuses a
  second final test pass or an overwrite of the completed experiment.
- A predeclared deployment safety gate requires test high recall >=.60 and F1/QWK
  no worse than the previous vote. Failure retains the old default; it never
  selects an alternative using test performance. The frozen blend passed.

**Critical limitation:** previous test results informed the user's proposal.
This is a reused internal holdout, not an untouched or externally validated test.
No test-based retuning was performed in this experiment.

## Files and reproduction

`protocol.json` records the rules before fitting. `validation_trials.json` records
all allowed mixes. `validation_summary.json` includes the ordinal comparison.
`test_report.json` contains all comparisons, including each individual model.
`original_hashes.json` verifies the dataset, six original checkpoints, schema,
preprocessor, selections, metrics and active version are unchanged.

Run `tools/run_soft_ordinal_experiment.py validation --project <root> --output
<new-folder>` then `... test ...` only for a separately justified future experiment.
Do not create repeated experiment folders to tune against this holdout.

## Runtime and rollback

The active policy is copied to `models/current_models/ensemble/policy.json`.
The Docker allowlist already includes that subtree. No ordinal checkpoint is
loaded for this selected blend; all six original model cards remain available.
`/api/schema` exposes the ensemble settings and metrics. The API returns the
actual weighted probabilities, cutoff and decision rule. The UI explains the
cutoff and distinguishes individual model agreement from consensus selection.

Set `BRFSS_DISABLE_SOFT_ENSEMBLE=1` and restart the Python server to restore the old
majority-vote behavior without changing or deleting checkpoints. Original source
is also preserved under the timestamped backup created during installation.

## Installed runtime verification

- All 47 original data/model files and historical hash manifests remain intact.
- 128 validation-input parity checks preserved all six individual classifications;
  reconstructed ensemble probability error was below 2.3e-9.
- The installed Gunicorn API completed 512 validation-input requests (3,072
  individual predictions) under a 512 MiB cgroup limit with swap disabled and no
  OOM events. Memory pressure reached the limit, so this is not spare-capacity
  assurance or a claim of cloud deployment verification.
- Concurrent overload returned explicit HTTP 503 backpressure; health remained
  available. Authentication, schema, benchmarks and invalid-input checks passed.
- `runtime_memory_verification.json` contains the local runtime evidence.
