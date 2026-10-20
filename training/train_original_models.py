"""
End-to-End Training Pipeline for Multimodal Stress Level Detection.
Executes training for all 6 models, performs hyperparameter validation,
saves trained artifacts, produces ablation study and explainability analysis,
and generates publication-quality figures.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from src.model_paths import LEGACY_ROOT
import joblib
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.preprocessing import load_and_preprocess_data, prepare_train_test_split, CLASS_NAMES
from src.models import (
    PhysiologicalDataset,
    Conv1DNet,
    LSTMNet,
    CNNLSTMNet,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    train_deep_model,
    predict_deep_model
)
from src.evaluate import (
    compute_all_metrics,
    plot_confusion_matrices_grid,
    plot_learning_curves,
    plot_metric_barchart
)
from src.explainability import (
    compute_and_plot_tree_feature_importance,
    compute_permutation_importance
)
from src.ablation import run_multimodal_ablation

def main():
    if LEGACY_ROOT.exists() and any(LEGACY_ROOT.iterdir()):
        raise SystemExit("Archived legacy models exist. Refusing to overwrite backups; select a new experiment output directory before retraining.")
    print("=" * 70)
    print("  MULTIMODAL PHYSIOLOGICAL STRESS LEVEL DETECTION — TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load and Preprocess Data
    print("\n[Step 1/6] Loading data and engineering physiological biomarkers...")
    df_clean = load_and_preprocess_data("data/original_sleep_dataset/sleep_lifestyle.csv")
    print(f"Dataset successfully prepared: {df_clean.shape[0]} samples, {df_clean.shape[1]} processed columns.")

    # 2. Strict Train/Val/Test Split (Zero Data Leakage)
    print("\n[Step 2/6] Performing Stratified Train / Validation / Test split (70% Train, 10% Val, 20% Test)...")
    data_dict = prepare_train_test_split(df_clean, test_size=0.20, val_size=0.10, random_state=42)
    feature_cols = data_dict['feature_cols']
    num_features = len(feature_cols)
    print(f"Features: {num_features} dimensions. Total test samples: {len(data_dict['y_test'])}.")

    os.makedirs(LEGACY_ROOT, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save preprocessing metadata and scaler
    joblib.dump(data_dict['scaler'], (LEGACY_ROOT / "scaler.pkl"))
    with open((LEGACY_ROOT / "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    X_train_s = data_dict['X_train_scaled']
    y_train = data_dict['y_train']
    X_val_s = data_dict['X_val_scaled']
    y_val = data_dict['y_val']
    X_test_s = data_dict['X_test_scaled']
    y_test = data_dict['y_test']

    # PyTorch DataLoaders
    train_ds = PhysiologicalDataset(X_train_s, y_train)
    val_ds = PhysiologicalDataset(X_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    results = {}

    # 3. Train Classical Machine Learning Models
    print("\n[Step 3/6] Training Classical Machine Learning Models...")
    
    # Model 1: Logistic Regression
    print("  -> Training Model 1/6: Logistic Regression (Baseline)...")
    lr_model = build_logistic_regression()
    lr_model.fit(X_train_s, y_train)
    joblib.dump(lr_model, (LEGACY_ROOT / "logistic_regression.joblib"))
    lr_preds = lr_model.predict(X_test_s)
    lr_proba = lr_model.predict_proba(X_test_s)
    results['Logistic Regression'] = compute_all_metrics(y_test, lr_preds, lr_proba, 'Logistic Regression')

    # Model 2: Random Forest
    print("  -> Training Model 2/6: Random Forest (Non-linear ensemble)...")
    rf_model = build_random_forest()
    rf_model.fit(X_train_s, y_train)
    joblib.dump(rf_model, (LEGACY_ROOT / "random_forest.joblib"))
    rf_preds = rf_model.predict(X_test_s)
    rf_proba = rf_model.predict_proba(X_test_s)
    results['Random Forest'] = compute_all_metrics(y_test, rf_preds, rf_proba, 'Random Forest')

    # Model 3: XGBoost
    print("  -> Training Model 3/6: XGBoost (Gradient boosting)...")
    xgb_model = build_xgboost()
    xgb_model.fit(X_train_s, y_train)
    joblib.dump(xgb_model, (LEGACY_ROOT / "xgboost.joblib"))
    xgb_preds = xgb_model.predict(X_test_s)
    xgb_proba = xgb_model.predict_proba(X_test_s)
    results['XGBoost'] = compute_all_metrics(y_test, xgb_preds, xgb_proba, 'XGBoost')

    # 4. Train Deep Learning Architectures (PyTorch)
    print("\n[Step 4/6] Training Deep Learning Models (PyTorch with Early Stopping)...")
    dl_histories = {}

    # Model 4: 1D CNN
    print("  -> Training Model 4/6: 1D CNN (Local physiological feature pattern learner)...")
    cnn_model = Conv1DNet(input_dim=num_features, num_classes=3)
    cnn_model, cnn_hist = train_deep_model(cnn_model, train_loader, val_loader, epochs=80, lr=0.001)
    torch.save(cnn_model.state_dict(), (LEGACY_ROOT / "conv1d_net.pt"))
    dl_histories['1D CNN'] = cnn_hist
    cnn_preds, cnn_proba = predict_deep_model(cnn_model, X_test_s)
    results['1D CNN'] = compute_all_metrics(y_test, cnn_preds, cnn_proba, '1D CNN')

    # Model 5: LSTM
    print("  -> Training Model 5/6: LSTM (Optimized Attention-Skip Recurrent Network)...")
    lstm_model = LSTMNet(input_dim=num_features, hidden_dim=32, num_layers=1, num_classes=3)
    lstm_model, lstm_hist = train_deep_model(lstm_model, train_loader, val_loader, epochs=100, lr=0.005, patience=15)
    torch.save(lstm_model.state_dict(), (LEGACY_ROOT / "lstm_net.pt"))
    dl_histories['LSTM'] = lstm_hist
    lstm_preds, lstm_proba = predict_deep_model(lstm_model, X_test_s)
    results['LSTM'] = compute_all_metrics(y_test, lstm_preds, lstm_proba, 'LSTM')

    # Model 6: CNN-LSTM
    print("  -> Training Model 6/6: CNN-LSTM (Hybrid hierarchical & temporal feature modeling)...")
    cnnlstm_model = CNNLSTMNet(input_dim=num_features, num_classes=3)
    cnnlstm_model, cnnlstm_hist = train_deep_model(cnnlstm_model, train_loader, val_loader, epochs=80, lr=0.001)
    torch.save(cnnlstm_model.state_dict(), (LEGACY_ROOT / "cnn_lstm_net.pt"))
    dl_histories['CNN-LSTM'] = cnnlstm_hist
    cnnlstm_preds, cnnlstm_proba = predict_deep_model(cnnlstm_model, X_test_s)
    results['CNN-LSTM'] = compute_all_metrics(y_test, cnnlstm_preds, cnnlstm_proba, 'CNN-LSTM')

    # 5. Export Metrics & Visualizations
    print("\n[Step 5/6] Generating Comparative Plots and Metrics Tables...")
    summary_list = []
    for m_name, res in results.items():
        summary_list.append({
            'Model': res['Model'],
            'Accuracy': round(res['Accuracy'], 4),
            'Precision (Macro)': round(res['Precision (Macro)'], 4),
            'Recall (Macro)': round(res['Recall (Macro)'], 4),
            'F1-Score (Macro)': round(res['F1-Score (Macro)'], 4),
            'Precision (Weighted)': round(res['Precision (Weighted)'], 4),
            'Recall (Weighted)': round(res['Recall (Weighted)'], 4),
            'F1-Score (Weighted)': round(res['F1-Score (Weighted)'], 4),
            'ROC-AUC (Macro)': round(res['ROC-AUC (Macro)'], 4) if not np.isnan(res['ROC-AUC (Macro)']) else "N/A"
        })

    df_summary = pd.DataFrame(summary_list).sort_values(by='F1-Score (Macro)', ascending=False)
    df_summary.to_csv("reports/model_comparison.csv", index=False)
    
    # Save Figures
    plot_confusion_matrices_grid(results, "figures/confusion_matrices_comparison.png")
    plot_learning_curves(dl_histories, "figures/deep_learning_curves.png")
    plot_metric_barchart(df_summary, "figures/model_metrics_comparison.png")

    # 6. Ablation Study & Explainability
    print("\n[Step 6/6] Executing Multimodal Ablation Study & Physiological Explainability...")
    run_multimodal_ablation(df_clean, save_path="figures/ablation_study.png")
    compute_and_plot_tree_feature_importance(rf_model, xgb_model, feature_cols, "figures/tree_feature_importance.png")
    compute_permutation_importance(xgb_model, X_test_s, y_test, feature_cols, "figures/permutation_importance.png")

    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)
    print("\nAll trained models, figures, and reports are saved successfully!")

if __name__ == "__main__":
    main()
