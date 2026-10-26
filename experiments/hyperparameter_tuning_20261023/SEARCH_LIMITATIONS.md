# How to interpret this bounded experiment

- Eighteen configurations were screened in two grouped folds: 36 fits. This is
  a small explicit search, not exhaustive or Bayesian optimization.
- Screening uses a deterministic subset of training profile groups. Neural
  screening permits four epochs; this can favor fast-learning configurations.
- Final fitting reserves 10% of the original training partition for early
  stopping. Original saved checkpoints used a different fitting/selection
  workflow. Their comparison measures the whole candidate workflow, not the
  isolated causal effect of one hyperparameter.
- Neural candidates have two predetermined seeds. The first is the candidate;
  the second reports stability and is never substituted because it scores better.
- Neural widths, layer counts, convolution kernels, feature order and scaling
  remain fixed. Logistic Regression solver/penalty and Random Forest class
  weighting remain fixed. These are deliberately deferred search dimensions,
  not dimensions proven unhelpful. Several settings change together in a trial.
- Inner calibration cutoffs are scored on a separate outer fold. The CV ranking
  permits a two-percentage-point sampling tolerance before penalizing recall
  shortfall. Final validation and promotion gates have no such tolerance.
- Full validation chooses final cutoffs and checks safeguards. Reported
  validation performance therefore includes selection optimism.
- Test was historically reused. This run freezes choices and opens it once, but
  that does not turn it into an untouched or external test. No failed candidate
  should be retuned against these final results.
- Paired stratified bootstrap intervals resample respondents within target
  classes. They are descriptive; they do not adjust for repeated historical
  experimentation, multiple comparisons, BRFSS survey design or profile-cluster
  dependence. They are not population-level confidence statements.
- All records remain real. Complete-case filtering can introduce selection bias;
  these results are not survey-weighted estimates of the general population.
- These are concurrent self-reported survey associations, not clinical diagnoses,
  causal effects, physiological time series or future-stress forecasts.
- Runtime installation is conditional on the frozen performance gates and
  memory/API checks. A rejected candidate remains an experiment, never an
  automatically substituted production model.
