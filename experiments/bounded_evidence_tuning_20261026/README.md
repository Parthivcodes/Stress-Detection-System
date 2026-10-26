# Bounded real-data follow-up

Outcome: **rejected_in_training_cv**. None of the predeclared variants passed every fold's accuracy and minority-recall/F1/QWK safeguards. Production was retained. No additional real validation or test evaluation was performed.

This two-fold bounded run is not exhaustive5x2CV. Neural fits used two fixed seeds averaged,12epochs maximum and60-second soft caps; these are limited-budget experiments.

Always-rarely training-OOF baseline: 65.86% accuracy,0% high and sometimes recall. Historical reused holdout baseline65.78% is a different partition.

| Model | Configuration | Accuracy | Accuracy delta vs matched baseline (pp) | Macro-F1 delta | Sometimes recall delta (pp) | High recall delta (pp) | Qualifies |
|---|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | baseline | 67.78% | +0.000 | +0.0000 | +0.000 | +0.000 | No |
| Logistic Regression | derived | 67.64% | -0.145 | +0.0043 | +1.245 | +1.092 | No |
| Logistic Regression | focus | 67.00% | -0.780 | +0.0044 | +2.818 | +0.682 | No |
| Random Forest | baseline | 66.19% | +0.000 | +0.0000 | +0.000 | +0.000 | No |
| Random Forest | derived | 65.19% | -0.992 | +0.0055 | +4.154 | +0.973 | No |
| Random Forest | focus | 65.83% | -0.350 | +0.0008 | +0.718 | +0.708 | No |
| XGBoost | baseline | 67.45% | +0.000 | +0.0000 | +0.000 | +0.000 | No |
| XGBoost | derived | 65.89% | -1.562 | +0.0001 | +3.662 | +1.800 | No |
| XGBoost | focus | 66.37% | -1.085 | +0.0037 | +3.767 | +0.874 | No |
| 1D CNN | baseline | 68.09% | +0.000 | +0.0000 | +0.000 | +0.000 | No |
| 1D CNN | focus | 67.98% | -0.111 | +0.0102 | +2.131 | +0.371 | No |
| LSTM | baseline | 67.06% | +0.000 | +0.0000 | +0.000 | +0.000 | No |
| LSTM | focus | 65.51% | -1.552 | +0.0029 | +5.573 | +0.576 | No |
| CNN-LSTM | baseline | 67.84% | +0.000 | +0.0000 | +0.000 | +0.000 | No |
| CNN-LSTM | focus | 68.05% | +0.207 | -0.0008 | -0.631 | +0.615 | No |

Eligibility also compares the frozen original decision rule and requires passing each scoring fold; pooled gains alone cannot override a failed fold. Full precision,recall,F1,AUC,confusion matrices,seed reports,times and splitIDs are in the JSON/NPZ/roles artifacts. The registered stacking diagnostic was not executed after the user redirected work to synthetic data and PDF export.
