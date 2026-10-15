# Stress Level Detection — Still Research Dashboard

A university research prototype comparing six classifiers on tabular physiological and lifestyle features. The interface includes a responsive 3D particle sculpture, profile controls, actual model predictions, archived benchmarks, and JSON result export. The sculpture is decorative, not a measured biosignal.

## Run

### This PC: verified WSL backend

Windows blocks native Python model libraries on this PC. The working backend now runs in Ubuntu/WSL with an isolated Python 3.12 environment at `/opt/stress-venv`. From this project folder, run:

```powershell
.\Start-Dashboard.ps1
```

Or use `.\Start-Dashboard-WSL.ps1` explicitly. Keep the terminal open and visit http://localhost:8000/#predictions (not the static preview on port 8001). The server runs as the unprivileged Linux `nobody` user and reads the existing project/model files; no model retraining or security-policy changes are involved. WSL can print a systemd user-session warning for this service account, but direct Python execution and prediction tests pass.

Only CPU inference libraries are installed in the Linux environment. `requirements-inference.txt` excludes training/plotting/UI extras. PyTorch was installed first from its official CPU wheel index; XGBoost uses `xgboost-cpu`.

Verified on 15 October 2026: all six model artifacts and preprocessing parity (`verify_project.py`); live API health, three profiles, normalized probabilities, and invalid-input rejection (`tests/prediction-api.py`); dashboard Analyze button displays all six actual results with no browser console errors.

### Other PCs: standard Python setup

With Python 3.11–3.13 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python server.py
```

Open http://localhost:8000. The frontend requires this backend for predictions. There is no fake offline inference. Assets and fonts work locally without a CDN.

On this machine, Python 3.12 and a project-local `.venv` are now installed. Start the dashboard with:

```powershell
.\.venv\Scripts\python.exe server.py
```

The project environment includes scikit-learn 1.6.1 to match the saved models. `Start-Dashboard.ps1` also prefers this environment. The older `.runtime` directory is retained but is not needed by this command.

### Native Windows runtime limitation (resolved for the dashboard through WSL)

An independent NumPy import also failed because Application Control blocked its `_umath_linalg` native module. The restriction affects more than PyTorch; classical-model inference is not a verified fallback.

Dependency installation into `.venv` completed and `pip check` passed. However, Windows Application Control rejects `torch_python.dll` with WinError 4551, including in the fresh environment. This is an operating-system code-integrity restriction, not a missing Python package. All six predictions remain unavailable until an approved PyTorch runtime can load. Do not disable security controls or treat the static preview on port 8001 as a working prediction backend. Obtain an approved runtime or run the project on another authorized environment, then run the verification below before the demonstration.

## Verify and evaluate

```powershell
python verify_project.py
python validate_submission.py
```

On this machine replace `python` with `.\.venv\Scripts\python.exe`.

- `verify_project.py`: saved model evaluation, feature/scaler parity, valid probabilities, invalid input checks.
- `validate_submission.py`: exact-profile audit and five outer / three inner group-aware folds, comparing fixed versus tuned LR, RF, and XGBoost, plus a dummy baseline.
- `reports/grouped_validation_summary.csv`: new evaluation; does not replace original holdout metrics.
- `SUBMISSION_REVIEW.md`: findings, tuning recommendation, and remaining submission work.
- `train_pipeline.py`: original six-model row-split training workflow, retained for reproducibility. It is not the stricter grouped evaluation.
- `app.py`: legacy Streamlit dashboard (`streamlit run app.py`). Use the new frontend for the current demo.

## Important research limits

The dataset contains 374 rows but only 132 distinct predictor profiles. In the original split, 58 of 75 test rows match training profiles. Original 94.67% RF/XGBoost test accuracy is reproducible, but that split is insufficient evidence of independent-profile generalization.

These are tabular features, not ECG/PPG recordings or longitudinal sequences. Stress categories are project-defined bins, not clinically validated diagnoses. Probabilities are uncalibrated. The six-model vote has not been evaluated as a standalone ensemble. Original reports and defense guides must be read with the corrections in `SUBMISSION_REVIEW.md`.

Dataset reference: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset. Verify source provenance, license, and collection details before submitting.
