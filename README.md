# Stress Prediction Project

## Start the website

```powershell
.\Start-Website.ps1
```

Open **http://127.0.0.1:8004/#predictions** and keep the terminal running.
This uses your existing WSL Python environment and the six expanded models.
No reinstall is needed.

## Project structure — line by line

```text
stressleveldetection/
├── Start-Website.ps1                 Main launcher, port 8004
├── server.py                        Website and prediction API
├── README.md                        Start here
├── data/
│   ├── original_sleep_dataset/      Original 374-row sleep/lifestyle data
│   ├── baseline_survey_dataset/     First 30,000-person survey and raw CDC source
│   └── expanded_survey_dataset/     Current 183,040-person dataset with splits
├── models/
│   ├── active_model.json            Default model selection
│   └── current_models/              Six expanded models and preprocessing files
├── backups/
│   ├── original_models/             Original sleep/lifestyle models
│   ├── baseline_models/             First BRFSS model set
│   ├── tuned_models/                Tuned BRFSS model set
│   └── previous_active_model.json   Previous default selection
├── frontend/
│   ├── index.html                   Main page
│   ├── landing.js / landing.css     Main-page animation and appearance
│   ├── prediction.js / prediction.css  Form and prediction results
│   ├── fonts/                       Local fonts and licenses
│   └── dashboard/                   Additional React dashboard
├── src/
│   ├── model_paths.py               Central model locations and historical aliases
│   ├── brfss_inference.py           Load survey models and predict
│   ├── models.py                    Classifier/neural-network definitions
│   └── preprocessing.py             Original-data preprocessing
├── training/
│   ├── train_expanded_models.py     Current training workflow
│   ├── train_baseline_models.py     First BRFSS training workflow
│   ├── tune_baseline_models.py      Baseline tuning
│   ├── train_original_models.py     Original sleep/lifestyle training
│   └── download_original_dataset.py Original download helper
├── tests/                           Model, data, website and integrity checks
├── reports/                         Evaluation results and historical audits
├── docs/                            Earlier project report and README
├── tools/                           Setup, alternate launchers and migration helpers
├── legacy/                          Original Streamlit application
├── api/                             Older Vercel adapters, not local Python inference
├── figures/                         Research plots
└── output/                          Reference images and prior verification outputs
```

Hidden `.venv` and `.runtime` are retained environments. `.cleanup` contains
recoverable earlier cleanup files. These are not training datasets.

## The six current model files

| File in `models/current_models/` | Algorithm |
| --- | --- |
| `logistic_regression.joblib` | Logistic Regression |
| `random_forest.joblib` | Random Forest |
| `xgboost.joblib` | XGBoost |
| `conv1d_net.pt` | 1D CNN |
| `lstm_net.pt` | LSTM |
| `cnn_lstm_net.pt` | CNN-LSTM |

`preprocessor.joblib` transforms inputs; `schema.json` defines the 19 inputs;
`selection.json` stores prediction adjustments; `metrics.json` and `READY.json`
record evaluation and completion. Never mix these files between model versions.

## Verification and training

```powershell
wsl.exe -d Ubuntu -u nobody --cd $PWD.Path --exec /opt/stress-venv/bin/python -B tests/model_layout_validation.py --inference
```

Run training from the root using module names, for example
`python -m training.train_expanded_models`. Completed model sets are protected
against accidental retraining. Do not remove those guards to chase test scores.

Internal version IDs and historical report folders keep their BRFSS names so
existing evidence remains traceable. Friendly model folders are mapped in
`src/model_paths.py`. Old-to-new paths and original checksums are recorded in
`reports/project_cleanup/reorganization.json`.

This is a research demonstration, not a clinical diagnosis. Legacy documentation
in `docs/` contains older accuracy figures, not current expanded-model results.
Use the local Python website for actual predictions, not the older Vercel API.
