"""
Multimodal Physiological Ablation Study (Section 10, RQ1 & RQ4):
Empirically assesses whether combining multimodal signals (Cardiovascular, Movement, Sleep)
outperforms individual unimodal physiological representations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def run_multimodal_ablation(df_encoded, random_state: int = 42, save_path: str = "figures/ablation_study.png"):
    """
    Evaluates unimodal vs. multimodal feature subsets using XGBoost and Random Forest.
    """
    from src.preprocessing import get_modality_feature_groups
    from sklearn.model_selection import train_test_split

    drop_cols = ['Person ID', 'Blood Pressure', 'Stress Level', 'Stress_Category']
    all_features = [c for c in df_encoded.columns if c not in drop_cols]
    modalities = get_modality_feature_groups(all_features)
    y = df_encoded['Stress_Category'].values

    ablation_results = []

    for mod_name, feat_list in modalities.items():
        if len(feat_list) == 0:
            continue
            
        X_sub = df_encoded[feat_list].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y, test_size=0.2, random_state=random_state, stratify=y)
        
        # Scaling
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        rf.fit(X_tr_s, y_tr)
        rf_preds = rf.predict(X_te_s)
        rf_acc = accuracy_score(y_te, rf_preds)
        rf_f1 = f1_score(y_te, rf_preds, average='macro', zero_division=0)

        # 2. XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state, eval_metric='mlogloss')
        xgb.fit(X_tr_s, y_tr)
        xgb_preds = xgb.predict(X_te_s)
        xgb_acc = accuracy_score(y_te, xgb_preds)
        xgb_f1 = f1_score(y_te, xgb_preds, average='macro', zero_division=0)

        ablation_results.append({
            'Modality Subset': mod_name.replace('_', ' '),
            'Num Features': len(feat_list),
            'RF Accuracy': rf_acc,
            'RF F1-Score': rf_f1,
            'XGB Accuracy': xgb_acc,
            'XGB F1-Score': xgb_f1,
            'Best Accuracy': max(rf_acc, xgb_acc),
            'Best F1-Score': max(rf_f1, xgb_f1)
        })

    df_ablation = pd.DataFrame(ablation_results).sort_values(by='Best F1-Score', ascending=False)
    
    # Save CSV
    os.makedirs("reports", exist_ok=True)
    df_ablation.to_csv("reports/ablation_study.csv", index=False)

    # Plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    df_plot = df_ablation.melt(id_vars=['Modality Subset'], value_vars=['Best Accuracy', 'Best F1-Score'], var_name='Metric', value_name='Score')
    
    ax = sns.barplot(data=df_plot, x='Modality Subset', y='Score', hue='Metric', palette=['#3B82F6', '#10B981'])
    plt.title("Ablation Study: Multimodal Fusion vs. Individual Physiological Modalities", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Score", fontsize=10)
    plt.xlabel("Physiological Signal Modality", fontsize=10)
    plt.xticks(rotation=20, ha='right')
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multimodal ablation plot to {save_path}")

    return df_ablation
