# BRFSS 2022 stress-frequency research dataset

Source: [CDC 2022 BRFSS annual data](https://www.cdc.gov/brfss/annual_data/annual_2022.html), public-use ASCII annual file. The downloaded archive and official codebook are retained in `data/brfss2022/raw`.

## Cohort and provenance

The source contains 445,132 records. We verified all composite survey keys (2022, state, SEQNO) are unique. Of these, 185,816 have valid responses for the chosen twelve predictors and the stress item. A seeded random sample without replacement contains exactly **30,000 unique respondent records**. These are not fabricated, duplicated or augmented. Public-use IDs identify survey records; we cannot independently establish participants' biological identities.

The machine-readable audit records source SHA-256, missing/invalid counts, class distribution and split sizes. The CSV retains source row numbers, respondent keys, original codes, decoded predictors, labels and split assignments. Missing, refused and don't-know responses are excluded, not invented. CDC's age field is already imputed and top-coded at 80. BMI has two implied decimal places; physically unhealthy days code 88 becomes zero.

## Target and inputs

Target SDHSTRE1 is self-reported stress frequency: 4/5 = rarely/never (class 0), 3 = sometimes (class 1), 1/2 = usually/always (class 2). This is not the old dataset's stress-level scale, a clinical diagnosis, or a forecast of future stress.

Predictors: age, hours of sleep, physically unhealthy days in the previous 30 days, BMI, survey-recorded sex, exercise in the past month, general health, marital status, education, employment, annual household income in US dollars, and emotional/social support. The label and respondent identifiers are never model inputs. Heart rate, blood pressure, steps and occupation from the original model are not available here and are not synthesized.

## Evaluation

Identical twelve-input profiles are grouped before stratified ten-fold assignment: seven folds train (21,003), one validates (2,999), two test (5,998). No survey ID or identical input profile crosses these partitions. Scaling is fitted only on training records. Neural early stopping uses validation macro F1; the test set is evaluated only after selecting a checkpoint. Class weighting addresses imbalance. Reports include a training-prior baseline, accuracy, balanced accuracy, macro F1, macro ROC-AUC and confusion matrices (rows=true, columns=predicted).

This is an unweighted complete-case convenience sample of participating US survey respondents, not a nationally representative estimate. Income/support nonresponse and optional-module coverage introduce selection bias. Scores are internal held-out results, not external validation; probabilities are uncalibrated. LSTM/CNN-LSTM operate on an arbitrary ordered tabular vector, **not longitudinal measurements**. Concurrent emotional support and health associations are not evidence of causation. Generalization to other countries is unverified.

## Safety, reproduction and rollback

Original root-level data and models remain untouched. `legacy_backup.zip` contains copies and `legacy_hashes.json` records SHA-256 values, checked again after training. New models live only in `models/brfss2022`.

Run `brfss_pipeline.py` with the existing WSL `/opt/stress-venv/bin/python` environment to reproduce preparation and all six training runs (seed 42). The downloaded raw archive must be present. `READY.json` is written only after training and legacy hash verification succeed.

The local Python dashboard defaults to BRFSS. Set `STRESS_DATASET=legacy` to retain the old inference engine; its original physiological form is available in Git history, not the new BRFSS form. Existing Streamlit and JavaScript deployment paths are legacy experiences, not the BRFSS-trained backend. Do not present their outputs as these retrained models. No website has been published as part of this migration.
