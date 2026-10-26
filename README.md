<!-- current-training-map -->
Study notebook: [All six models](model_training/All_Six_Models_Training.ipynb). Simple directory guide: [Project map](docs/03_Simple_Project_Map.md). Current training is in `model_training/`; earlier experiments are in `archive/previous_experiments/`.

# NERVA Stress-Detection Intelligence System

## Start the Main Website (Primary System)

```powershell
.\Start-Main-Website.ps1
```
*(Alternatively, you can run `.\Start-Website.ps1` or `.\Start-Demo-Website.ps1` — all point directly to the main system).*

Open **http://localhost:8001/#predictions** and keep the terminal running.
This is our **REAL / PRIMARY MAIN WEBSITE**, powered by the AI-enhanced 10,000-profile dataset (`model_training/base_models/model_training_data.csv`) and the complete six high-performing model architectures documented in [All_Six_Models_Training.ipynb](model_training/All_Six_Models_Training.ipynb):
- **LSTM** (91.40% Accuracy, 0.9857 ROC-AUC)
- **1D CNN** (90.53% Accuracy, 0.9844 ROC-AUC)
- **Logistic Regression** (90.00% Accuracy, 0.9837 ROC-AUC)
- **Random Forest** (89.07% Accuracy, 0.9776 ROC-AUC)
- **XGBoost** (86.93% Accuracy, 0.9699 ROC-AUC)
- **CNN-LSTM** (84.60% Accuracy, 0.9607 ROC-AUC)

## Understand the project

1. model_training/All_Six_Models_Training.ipynb - complete training and evaluation pipeline across all 6 models.
2. docs/01_Model_Training_Walkthrough.pdf - complete printable explanation of the models and tuning.
3. docs/01_Model_Training_Walkthrough.md - editable walkthrough with source references.
4. docs/02_Project_File_Guide.md - line-by-line directory and file guide.
5. models/ACTIVE_MODELS.md - exact active checkpoint/encoder mapping for all six algorithms.

## Current workflow

- `Start-Main-Website.ps1` runs the real main website on port 8001.
- `model_training/All_Six_Models_Training.ipynb` runs the end-to-end training and evaluation for all six models.
- `training/create_and_train_demo.py` creates a fresh experiment and trains six families.
- `training/improve_forest_and_cnn.py` runs the fixed-data, recall-protected RF/CNN tuning workflow.
- `training/create_demo_training_report.py` reports completed training metrics.

## Important runtime folders

- `frontend/` contains the actual website, live prediction forms, and browser PDF exporter.
- `model_training/` contains the primary 10,000-sample training dataset (`model_training_data.csv`), training notebook, and model checkpoints.
- `src/` contains neural architectures, feature encoders, and prediction loaders.
- `models/` contains active model configurations and selection files.
- `data/` contains `original_sleep_dataset/sleep_lifestyle.csv`.
- `experiments/` contains versioned datasets, saved checkpoints, and evaluation evidence.
- `backups/` preserves known-working prior versions.
- `tests/` contains verification tests.
- `deploy/`, `api/`, and `serverless/` retain deployment configuration.
