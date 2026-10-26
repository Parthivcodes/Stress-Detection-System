# Active Model Files — NERVA Main Website

This document indexes the active model checkpoints and encoders powering the **NERVA Main Website**:

## 1. Main Website Active Models (10,000-Profile AI-Enhanced Architecture)
Trained across all six algorithmic families as documented in [All_Six_Models_Training.ipynb](../model_training/All_Six_Models_Training.ipynb):

| Algorithm | Checkpoint(s) | Encoder(s) | Measured Test Accuracy |
|---|---|---|---|
| **LSTM** | `model_training/base_models/models/4_20261026.pt`<br>`model_training/base_models/models/4_20261027.pt` | `model_training/base_models/models/4_20261026.preprocessor.joblib`<br>`model_training/base_models/models/4_20261027.preprocessor.joblib` | **91.40%** |
| **1D CNN** | `model_training/improved_models/models/1D_CNN_20261026.pt`<br>`model_training/improved_models/models/1D_CNN_20261027.pt` | `model_training/improved_models/models/1D_CNN_20261026.encoder.joblib`<br>`model_training/improved_models/models/1D_CNN_20261027.encoder.joblib` | **90.53%** |
| **Logistic Regression** | `model_training/base_models/models/0_20261026.joblib` | `model_training/base_models/models/0_20261026.preprocessor.joblib` | **90.00%** |
| **Random Forest** | `model_training/improved_models/models/Random_Forest_20261026.joblib` | `model_training/improved_models/models/Random_Forest_20261026.encoder.joblib` | **89.07%** |
| **XGBoost** | `model_training/base_models/models/2_20261026.joblib` | `model_training/base_models/models/2_20261026.preprocessor.joblib` | **86.93%** |
| **CNN-LSTM** | `model_training/base_models/models/5_20261026.pt`<br>`model_training/base_models/models/5_20261027.pt` | `model_training/base_models/models/5_20261026.preprocessor.joblib`<br>`model_training/base_models/models/5_20261027.preprocessor.joblib` | **84.60%** |

*Overall Mean Accuracy: ~88.76% across all 6 models.*

## 2. Historical Raw BRFSS Baseline Models (`models/current_models/`)
Preserved for historical comparison:
- `models/current_models/logistic_regression.joblib`
- `models/current_models/random_forest.joblib`
- `models/current_models/xgboost.joblib`
- `models/current_models/conv1d_net.pt`
- `models/current_models/lstm_net.pt`
- `models/current_models/cnn_lstm_net.pt`
