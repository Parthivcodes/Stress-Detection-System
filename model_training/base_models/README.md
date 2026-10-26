# Separate lower-noise simulation experiment

**Generated-data educational benchmark. These scores do not establish accuracy on real people. Existing datasets, models and website selections remain unchanged.**

## Fixed experiment

Noise standard deviation was fixed at 0.25 before training, compared with 1.0 in the previous simulation. The 19 features, 10,000 profile IDs, all split assignments and fit/development row order are identical. The hidden-score equation and class boundaries did not change. Labels are different because the noise assumption changed; this is an easier simulation, not evidence that tuning improved prediction on the old ground truth. No noise value was adjusted after reading results.

There are 6,300 model-fitting profiles, 700 development profiles, 1,500 validation profiles and 1,500 test profiles. Labels changed for 2,548 of the 10,000 profiles relative to the old simulation. The test was evaluated once after all checkpoints were frozen. Development rows selected neural stopping epochs; validation metrics were diagnostic and selected no settings. All family decisions use raw-probability argmax.

## Measured test results

| Model | Previous noise 1.0 | New noise 0.25 | New macro-F1 | Sometimes recall | High recall | High detected / missed |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 64.93% | 90.00% | 0.9014 | 86.44% | 90.75% | 353 / 36 |
| Random Forest | 61.00% | 76.73% | 0.7704 | 67.46% | 79.18% | 308 / 81 |
| XGBoost | 63.27% | 86.93% | 0.8715 | 84.75% | 85.35% | 332 / 57 |
| 1D CNN | 62.00% | 78.07% | 0.7863 | 75.59% | 78.66% | 306 / 83 |
| LSTM | 66.27% | 91.40% | 0.9159 | 88.64% | 93.06% | 362 / 27 |
| CNN-LSTM | 63.67% | 84.60% | 0.8486 | 76.27% | 88.95% | 346 / 43 |

Six-family mean accuracy: **84.62%**, versus 63.52% in the previous higher-noise simulation. This difference reflects different generated labels, not an apples-to-apples training improvement.

Always predicting the training-majority class gives **39.33%** on the new test (majority class=1). The prior simulation's always-rarely baseline was **37.73%**. Baselines change when class proportions change.

The simulator's known conditional distribution supports approximately **90.24% expected optimal accuracy** over the training feature profiles at this noise level. This is a mathematical reference under invented assumptions, not a clinical accuracy estimate or strict finite-sample bound.

## Settings and stability

Three classical models use the same configuration-cloning recipe as before. Every neural family uses its existing architecture, two predetermined seeds averaged, AdamW learning rate 0.0007, weight decay 0.001, weighted crossentropy, batch256, maximum 40 epochs, patience 6 and a 120-second soft cap checked between epochs. All preprocessing fits only on fitting rows. There was no hyperparameter search or post-test adjustment. Neural inputs are ordered tabular feature positions, not temporal histories.

| Model | Validation accuracy per fixed seed | Total fitting seconds | New test accuracy95% interval |
|---|---|---:|---|
| Logistic Regression | 87.33% | 4.3 | 88.38% to 91.42% |
| Random Forest | 74.93% | 6.9 | 74.53% to 78.80% |
| XGBoost | 84.33% | 0.7 | 85.13% to 88.55% |
| 1D CNN | 75.53%, 76.87% | 37.4 | 75.90% to 80.09% |
| LSTM | 88.40%, 88.73% | 51.1 | 89.87% to 92.72% |
| CNN-LSTM | 82.93%, 81.33% | 42.9 | 82.69% to 86.34% |

Two seeds give limited stability evidence. Intervals describe generated-test sampling variation and do not cover uncertainty about whether the generator resembles real people. Full precision, recall, F1, QWK, AUC and confusion matrices remain in test_metrics.json.

## Safety and reproduction

protected_hashes.json verifies every previous simulation artifact plus the current real-data model bundle, active-model selection and original expanded dataset. frozen_source_hashes.json protects this experiment's protocol/source snapshot; frozen_models.json protects checkpoints/preprocessors. All nine checkpoints representing six families were loaded successfully for evaluation. Neither existing dataset was overwritten, no existing checkpoint was replaced and neither website was switched.

New dataset: model_training_data.csv in this experiment directory. Source, checkpoints, split roles, validation histories, final probabilities and results are stored alongside it. Choose a fresh output directory when reproducing.

```powershell
wsl -d Ubuntu -u nobody --exec /opt/stress-venv/bin/python -B /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection/tools/train_lower_noise_simulation.py --project /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection --output /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection/experiments/lower_noise_fresh
```

These new models remain experimental. A future website integration would need explicit selection of this dataset/version, input and prediction-parity verification and accurate simulation labelling. No rollback is currently necessary because the active versions were not changed.
