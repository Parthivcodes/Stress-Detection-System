"""
Explainability & Interpretability Module (Section 12 of Proposal):
Computes feature importance rankings, physiological biomarker contributions,
and generates explainability visualizations to satisfy academic evaluation criteria.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def compute_and_plot_tree_feature_importance(rf_model, xgb_model, feature_names: list, save_path: str = "figures/tree_feature_importance.png"):
    """
    Computes Gini / MDI and Gain feature importances from Random Forest and XGBoost,
    ranking the most influential physiological markers.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_imp,
        'XGBoost': xgb_imp
    })
    
    # Sort by average importance
    df_imp['Average'] = (df_imp['Random Forest'] + df_imp['XGBoost']) / 2.0
    df_imp = df_imp.sort_values(by='Average', ascending=False).head(12)

    plt.figure(figsize=(10, 6))
    df_plot = df_imp.melt(id_vars=['Feature'], value_vars=['Random Forest', 'XGBoost'], var_name='Model', value_name='Importance')
    
    sns.barplot(data=df_plot, y='Feature', x='Importance', hue='Model', palette=['#3B82F6', '#10B981'])
    plt.title("Top 12 Most Influential Physiological Signals & Biomarkers (MDI Importance)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Normalized Feature Importance", fontsize=10)
    plt.ylabel("Physiological Feature", fontsize=10)
    plt.grid(axis='x', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved tree feature importance plot to {save_path}")

    return df_imp

def compute_permutation_importance(model, X_test, y_test, feature_names: list, save_path: str = "figures/permutation_importance.png"):
    """
    Computes model-agnostic test-set permutation importance to measure genuine generalizable signal impact.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='f1_macro')
    
    indices = np.argsort(perm.importances_mean)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_means = perm.importances_mean[indices]
    top_stds = perm.importances_std[indices]

    plt.figure(figsize=(10, 5))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_means, xerr=top_stds, align='center', color='#8B5CF6', ecolor='gray', capsize=4)
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Decrease in Macro F1-Score when Feature is Shuffled", fontsize=10)
    plt.title("Test-Set Permutation Importance (Impact on Model Generalization)", fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation importance plot to {save_path}")
