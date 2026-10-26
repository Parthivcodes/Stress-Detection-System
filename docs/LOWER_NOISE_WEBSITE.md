# Lower-noise models connected to the simulation website

The simulation preview at http://localhost:8001/#predictions now loads the lower-noise experiment. The original real-data server at http://localhost:8000/ remains separate.

`models/training_preview.json` selects `lower_noise`. The selection resolves to `experiments/lower_noise_simulation_20261026`, with its model_training_data.csv, frozen_models.json, saved preprocessors/checkpoints and test_metrics.json. The loader verifies the dataset and checkpoint hashes before serving predictions. It does not retrain, modify source records or change the real-data active_model.json.

## Displayed results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 90.00% |
| Random Forest | 89.07% |
| XGBoost | 86.93% |
| 1D CNN | 90.53% |
| LSTM | 91.40% |
| CNN-LSTM | 84.60% |

Mean accuracy is88.76% on1,500 held-out generated profiles. These are results from an easier generated-data simulation with noise standard deviation0.25. They do not establish accuracy on real people. The correct majority-class baseline is always predicting sometimes:39.33% accuracy,100% sometimes recall and0% high recall.

Every card uses raw-probability argmax. Each neural family averages two fixed seed checkpoints. The combined display averages all six family probability vectors equally. The evidence table reports the individual families' frozen metrics, not an independently validated ensemble score. The browser checks the experiment identifier before accepting predictions, to avoid mixing a stale page with a different model selection.

## Verification

- Twelve saved test profiles checked across all six families; maximum probability discrepancy below1e-6 and matching labels.
- Benchmarks match the frozen test report exactly; the majority-class baseline is checked against the new class distribution.
- Invalid inputs remain rejected.
- The original simulation can still be explicitly loaded with experiment='original'.
- Both running local HTTP servers returned all six predictions.
- Browser evidence showed all six updated accuracy rows and successful live predictions, with no captured console errors.
- A real browser PDF download contained both pages, all six updated model results and the lower-noise dataset identity. Both pages were rendered and visually checked.

`experiments/lower_noise_website_connection/verification.json` contains loader verification evidence. The PDF continues to use only the latest submitted profile and API results; no external document service is used.

## Backups and rollback

`backups/before_lower_noise_connection/` contains checksum-verified copies of the real-data active model bundle and selection, previous simulation checkpoints, lower-noise checkpoints, both generated CSVs/manifests, the real expanded dataset and previous loader/server/frontend/PDF-export files. backup_hashes.json records the exact bytes.

To restore the previous simulation without touching any models or datasets, set the experiment field in models/training_preview.json to `original`, restart tools/Start-TrainingPreview.ps1, then refresh the browser. Setting it back to `lower_noise` restores the new preview. The real-data server remains controlled by its existing separate active-model selection.

No cloud deployment or paid service change was made.

## Fixed-data RF/CNN update

The active selector now includes overrides pointing to experiments/lower_noise_rf_cnn_tuning_20261026/overrides.json. Only Random Forest and1D CNN changed after passing training-fold,validation,reused-holdout andruntime gates. Dataset bytes and the other four families are unchanged. Their full before/after recall results and limitations are in that experiment README. Remove only the overrides field to return to the original lower-noise RF/CNN; changing experiment to original instead returns to the older higher-noise simulation.
