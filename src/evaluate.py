"""
Comprehensive Evaluation and Visualization Module for Stress Level Classification.
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix,
and generates publication-quality figures for university reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ['Low Stress', 'Moderate Stress', 'High Stress']

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> dict:
    """
    Computes all standard classification metrics as defined in Section 9 of the proposal.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Multiclass One-vs-Rest ROC-AUC calculation
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        if y_proba is not None and y_proba.shape[1] == 3:
            auc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro')
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)

    return {
        'Model': model_name,
        'Accuracy': float(acc),
        'Precision (Macro)': float(prec_macro),
        'Recall (Macro)': float(rec_macro),
        'F1-Score (Macro)': float(f1_macro),
        'Precision (Weighted)': float(prec_weighted),
        'Recall (Weighted)': float(rec_weighted),
        'F1-Score (Weighted)': float(f1_weighted),
        'ROC-AUC (Macro)': float(auc),
        'Confusion_Matrix': cm
    }

def plot_confusion_matrices_grid(results_dict: dict, save_path: str = "figures/confusion_matrices_comparison.png"):
    """
    Plots a multi-panel comparison of confusion matrices for all evaluated models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(results_dict)
    cols = 3
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, (name, res) in enumerate(results_dict.items()):
        cm = res['Confusion_Matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
        )
        acc_text = f"Acc: {res['Accuracy']:.2%} | F1: {res['F1-Score (Macro)']:.3f}"
        ax.set_title(f"{name}\n({acc_text})", fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Multimodal Stress Level Classification — Confusion Matrices Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices grid to {save_path}")

def plot_learning_curves(dl_histories: dict, save_path: str = "figures/deep_learning_curves.png"):
    """
    Generates training and validation loss and accuracy curves for 1D CNN, LSTM, and CNN-LSTM.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(len(dl_histories), 2, figsize=(12, 3.5 * len(dl_histories)))
    if len(dl_histories) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (model_name, hist) in enumerate(dl_histories.items()):
        epochs = range(1, len(hist['train_loss']) + 1)
        
        # Loss plot
        ax_loss = axes[i, 0]
        ax_loss.plot(epochs, hist['train_loss'], label='Train Loss', color='#2563EB', lw=2)
        ax_loss.plot(epochs, hist['val_loss'], label='Val Loss', color='#DC2626', lw=2, linestyle='--')
        ax_loss.set_title(f"{model_name} — Loss Curve", fontsize=11, fontweight='bold')
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("Cross Entropy Loss", fontsize=9)
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy plot
        ax_acc = axes[i, 1]
        ax_acc.plot(epochs, hist['train_acc'], label='Train Acc', color='#10B981', lw=2)
        ax_acc.plot(epochs, hist['val_acc'], label='Val Acc', color='#F59E0B', lw=2, linestyle='--')
        ax_acc.set_title(f"{model_name} — Accuracy Curve", fontsize=11, fontweight='bold')
        ax_acc.set_xlabel("Epoch", fontsize=9)
        ax_acc.set_ylabel("Accuracy", fontsize=9)
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    plt.suptitle("Deep Learning Models: Learning & Generalization Curves", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved DL learning curves to {save_path}")

def plot_metric_barchart(df_summary: pd.DataFrame, save_path: str = "figures/model_metrics_comparison.png"):
    """
    Generates comparison bar charts across models for Accuracy, Macro F1, and Macro Recall.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_to_plot = ['Accuracy', 'F1-Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']
    
    plt.figure(figsize=(12, 6))
    df_melt = df_summary.melt(id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    palette = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
    ax = sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette=palette)
    
    plt.title("Comparative Performance Analysis Across All 6 Machine Learning & Deep Learning Models", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=10)
    plt.xlabel("Model Architecture", fontsize=10)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add numeric labels on bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=7, color='black', xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metric bar chart to {save_path}")
