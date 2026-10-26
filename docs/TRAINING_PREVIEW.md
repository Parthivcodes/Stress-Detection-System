# Website connected to model_training_data.csv

The six saved model families were already trained using exactly these 10,000 generated records before the CSV was renamed. The CSV hash is unchanged: `eae884b43a8e63507ba00429c7bf99f3fbf5276b5fe1ae7e9b0b6521b888db70`. Integration reuses and verifies those completed checkpoints; it does not claim a new training run.

The data describe an educational simulation, not observed people. The existing real-data website and its default model selection remain separate. No cloud deployment was performed.

## Open and restart

- Training-data preview: http://localhost:8001/#predictions
- Original real-data website: http://localhost:8000/#predictions
- From the project directory in PowerShell: `./tools/Start-TrainingPreview.ps1`
- Equivalent existing-runtime command: `wsl -d Ubuntu -u nobody --exec /opt/stress-venv/bin/python -B /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection/tools/start_training_preview.py --port 8001`

The normal server command remains unchanged. Both previews can run concurrently. The new launcher binds only to loopback. Stop its console with Ctrl+C when no longer needed.

## Actual accuracy

These are frozen raw-argmax results on 1,500 held-out generated profiles, not real-world accuracy. No new thresholds or weights were chosen during website integration.

| Model | Accuracy | High-class recall |
|---|---:|---:|
| Logistic Regression | 64.93% | 74.55% |
| Random Forest | 61.00% | 71.62% |
| XGBoost | 63.27% | 70.50% |
| 1D CNN | 62.00% | 70.05% |
| LSTM | 66.27% | 73.42% |
| CNN-LSTM | 63.67% | 76.13% |

Mean accuracy is63.52%. Always predicting rarely/never gives37.73% accuracy and0% high/sometimes recall in this generated test set. These metrics cannot establish an improvement over the original BRFSS results because the populations and label-generating mechanisms differ.

## Prediction rules

Each individual label is the argmax of its raw class probabilities. Every neural family averages the same two predetermined seeds used during its saved evaluation. The displayed combined result equally averages the six family probability vectors and takes argmax, with lower-class tie break. This combined display has not been independently validated; the evidence table reports individual frozen model metrics only. No previous real-data class multipliers are applied to these new models.

The website and downloaded PDF identify this as a generated-data simulation. The PDF includes the same displayed probabilities and input profile. No external PDF service or new backend dependency is used.

## Files and verification

- `src/training_preview_inference.py`: validated loading, checksum checks and the six prediction paths.
- `tools/start_training_preview.py`: independent local server that reuses the existing website.
- `experiments/synthetic_simulation_20261026/model_training_data.csv`: unchanged 10,000-row input table.
- `experiments/synthetic_simulation_20261026/models/`: existing trained artifacts and fit-only preprocessors.
- `experiments/synthetic_simulation_20261026/SYNTHETIC_test_metrics.json`: full frozen raw and optional operating-rule evaluation. The preview deliberately uses the documented raw rule.
- `experiments/training_preview_connection/adapter_verification.json`: twelve saved prediction comparisons across every model, seven rejected invalid inputs and actual latency.

Verification found maximum probability difference below1.1e-7 when comparing the new adapter with saved results. It checks exact feature order, checkpoint hashes, data hash, benchmark values and label logic. Both local HTTP servers returned all six results. Browser verification confirmed the accuracy table, completed predictions, a working two-page PDF download and no captured console errors. Both PDF pages were rendered and inspected.

## Backups and rollback

`backups/before_training_preview/` contains checksum-verified copies of the complete active model bundle, active selection file, original expanded real dataset, model_training_data.csv and its manifest, server.py and the prior frontend files. `backup_hashes.json` records those bytes. The new mode did not change active_model.json or any existing checkpoint or dataset.

To return to the original mode, use port8000. To revert the shared frontend changes as well, restore frontend/prediction.js, frontend/analysis-pdf.js and frontend/index.html from this backup. Stop port8001; the two added Python files can remain unused. No model rollback or data recovery is required.
