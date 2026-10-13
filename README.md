# Stress Level Detection Using Multimodal Physiological Signals

An end-to-end Machine Learning and Deep Learning pipeline for multimodal physiological stress-level classification based on clinical biosignals (Heart Rate, Blood Pressure, Physical Activity, Daily Steps, Sleep Duration, and Sleep Quality).

---

## 🚀 Quick Start

### 1. Launch the Sechenov Polymer Bio-Telemetry Cockpit (Next-Gen UI)
Run the high-performance Python bridge to launch the 3D procedural Sechenov Polymer WebGL cockpit:
```bash
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser (or simply double-click `sechenov_ui/index.html` for offline autonomous mode).

### 2. Launch the Streamlit University Dashboard
Run the following command to start the standard academic demonstration application:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 3. Retrain All Models & Regenerate Plots
To execute the complete zero-leakage training pipeline across all 6 architectures:
```bash
python train_pipeline.py
```

---

## 📊 Benchmark Leaderboard (Holdout Test Set)

| Model Architecture | Paradigm | Accuracy | Macro F1-Score | Macro Recall | Macro ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Non-Linear Bagging Ensemble | **94.67%** | **0.9453** | **0.9472** | **0.9961** |
| **XGBoost** | Gradient Boosted Trees | **94.67%** | **0.9453** | **0.9472** | **0.9956** |
| **CNN-LSTM** | Hybrid Hierarchical & Temporal DL | **92.00%** | **0.9194** | **0.9234** | **0.9747** |
| **Logistic Regression**| Linear Baseline | **92.00%** | **0.9194** | **0.9234** | **0.9701** |
| **1D CNN** | Localized Biosignal Convolution | **89.33%** | **0.8932** | **0.8956** | **0.9778** |
| **LSTM (Optimized)** | Attention-Skip Recurrent Network | **89.33%** | **0.8932** | **0.8956** | **0.9926** |

---

## 📁 Repository Structure
```
├── app.py                   # Interactive Streamlit Web Application
├── train_pipeline.py        # End-to-end training and evaluation script
├── load_data.py             # KaggleHub dataset loader script
├── PROJECT_REPORT.md        # Comprehensive academic project report
├── VIVA_DEFENSE_GUIDE.md    # Viva voce Q&A cheat sheet for university evaluation
├── data/                    # Dataset directory (Sleep_health_and_lifestyle_dataset.csv)
├── src/
│   ├── preprocessing.py     # Feature engineering, hemodynamic biomarkers & split
│   ├── models.py            # PyTorch DL (1D CNN, LSTM, CNN-LSTM) & Classical ML
│   ├── evaluate.py          # Metric computation & high-res chart generation
│   ├── ablation.py          # Multimodal signal ablation experiments (RQ1)
│   └── explainability.py    # MDI and Permutation Importance analysis (XAI)
├── models/                  # Saved weights (.joblib, .pt, scaler.pkl, feature_cols.json)
├── figures/                 # Generated publication-quality figures (.png)
└── reports/                 # Benchmark CSV exports (model_comparison.csv, ablation.csv)
```

---

## 🎓 University Deliverables Alignment
This implementation satisfies 100% of the project proposal requirements:
- **Multimodal Physiological Signals**: Cardiovascular, Movement, Sleep, Clinical.
- **Biomarker Engineering**: Pulse Pressure, MAP, Rate-Pressure Product (RPP), Cardiovascular Stress Index (CSI).
- **All 6 Required Models**: Logistic Regression, Random Forest, XGBoost, 1D CNN, LSTM, and CNN-LSTM.
- **Strict Leakage Prevention**: Scalers and transformations fitted strictly on train set.
- **Multimodal Ablation Study**: Empirically answers Research Question 1.
- **Explainability (XAI)**: MDI & Permutation feature importance (Section 12).
- **Interactive System**: Live clinical simulator for viva voce demonstration.
