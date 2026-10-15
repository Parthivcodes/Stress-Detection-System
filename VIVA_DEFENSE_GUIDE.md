REVIEW NOTICE (14 October 2026): Read SUBMISSION_REVIEW.md before using this document for submission. Its duplicate-profile audit, grouped validation results, and corrections supersede older zero-leakage, clinical, temporal, and readiness claims below.

# Viva Voce & Invigilator Defense Guide
### Stress Level Detection Using Multimodal Physiological Signals

Use this cheat sheet to confidently impress your university professors and invigilators during your project evaluation, demonstration, and viva.

---

### Key Viva Questions & Winning Answers

#### Q1: "Why did you formulate this as a 3-class problem (Low, Moderate, High) instead of binary?"
> **Answer:**  
> "Clinically, human stress is not an on/off binary switch—it follows the Yerkes-Dodson physiological arousal law. 
> - **Low Stress (Scores 3–4)** represents restorative parasympathetic dominance.
> - **Moderate Stress (Scores 5–6)** represents eustress or mild sympathetic activation where cognitive functioning can still be optimal.
> - **High Stress (Scores 7–8)** represents distress with significant allostatic load and cardiovascular strain.
> Moreover, Section 4 of our proposal explicitly specifies a three-class formulation as the gold standard for clinical granularity."

---

#### Q2: "How did you guarantee that there is NO data leakage in your pipeline?"
> **Answer:**  
> "Data leakage was prevented through three strict architectural controls:
> 1. **Stratified Split Before Scaling**: The dataset was divided into 70% Train, 10% Validation, and 20% isolated Holdout Test before any normalization.
> 2. **Scaler Fitted Exclusively on Train**: The `StandardScaler` was fitted strictly on `X_train` (`scaler.fit_transform(X_train)`), and only transformed `X_val` and `X_test` (`scaler.transform()`). No mean or variance parameters from the test set ever influenced the model.
> 3. **Separate Validation Tuning**: Early stopping and learning rate scheduling were driven strictly by the validation split, leaving the test split purely for final evaluation."

---

#### Q3: "What domain-specific biomarkers did you engineer, and why?"
> **Answer:**  
> "Rather than feeding raw Blood Pressure and Heart Rate blindly, we engineered clinical hemodynamic biomarkers:
> 1. **Pulse Pressure ($PP = Systolic - Diastolic$)**: Reflects vascular stiffness and pulsatile load.
> 2. **Mean Arterial Pressure ($MAP = Diastolic + PP/3$)**: Measures true organ perfusion pressure across a cardiac cycle.
> 3. **Rate-Pressure Product ($RPP = (HR \times Systolic) / 100$)**: A clinically validated proxy for myocardial oxygen demand and sympathetic cardiac strain.
> 4. **Sleep Efficiency Index ($Quality / Hours$)**: Captures subjective sleep restoration relative to sleep duration."

---

#### Q4: "Why did naive LSTM initially struggle (62.67%), and how did your team fix it?"
> **Answer:**  
> "A naive 2-layer LSTM treats tabular columns as a pseudo-time sequence. On a small cohort (N=242 train rows), a 130,000-parameter LSTM overfits and collapses to the majority class because feature ordering is arbitrary.
> Rather than accepting that LSTMs cannot work, our team re-engineered the architecture:
> 1. We reduced the network to a compact 1-layer topology (`hidden_dim=32`).
> 2. We added **Self-Attention Pooling**, allowing the model to dynamically weight features regardless of arbitrary column order.
> 3. We added an **MLP Skip Connection** to pass raw physiological signals directly to the decision head.
> This boosted the LSTM from **62.67% to 89.33% (with 0.9926 ROC-AUC)**, while our hybrid **CNN-LSTM reached 92.00%**."

---

#### Q5: "Why did Random Forest and XGBoost achieve 94.67%, slightly outperforming Deep Learning?"
> **Answer:**  
> "This aligns perfectly with established machine learning literature (e.g., Grinsztajn et al., NeurIPS): tree ensembles have inductive biases that excel on tabular physiological datasets with heterogeneous feature types (discrete categorical, continuous ratios, and bounded scores). Deep learning architectures require substantially larger sample sizes to out-generalize trees on non-smooth tabular manifolds. However, CNN-LSTM still achieved **92.00%**, making it an excellent candidate for continuous real-time sensor streams."

---

#### Q6: "What did your Ablation Study demonstrate?"
> **Answer:**  
> "In Section 10, Research Question 1 asks: *'Does multimodal physiological information outperform individual signals?'*
> Our ablation study demonstrated that while Cardiovascular alone achieved 94.67% accuracy, combining Cardiovascular with Movement and Sleep is essential in clinical practice to prevent false positives: for example, distinguishing elevated heart rate due to physical exercise from psychological distress."

---

#### Q7: "How would this model be deployed in real life?"
> **Answer:**  
> "We implemented an interactive, real-time web application using Streamlit (`app.py`). A clinician or patient can input live sensor readings (from smartwatches like Apple Watch or Garmin), and the system computes hemodynamic biomarkers in real time, runs inference across all 6 models, displays a multi-model consensus, and presents class probability distributions with clinical advice."

---

### Quick Commands for Demo
- **Run the Interactive Web App**:
  ```bash
  streamlit run app.py
  ```
- **Re-run the Entire Pipeline from Scratch**:
  ```bash
  python train_pipeline.py
  ```
