"""Checks actual saved models, inference preprocessing parity, and validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import server
from src.preprocessing import load_and_preprocess_data, prepare_train_test_split
from src.models import predict_deep_model

server.init_models()
root = Path(__file__).resolve().parent
frame = load_and_preprocess_data(str(root / 'data/Sleep_health_and_lifestyle_dataset.csv'))
split = prepare_train_test_split(frame, val_size=.10)
assert server.ARTIFACTS['feature_cols'] == split['feature_cols']
np.testing.assert_allclose(server.ARTIFACTS['scaler'].mean_, split['scaler'].mean_)
metrics=[]
for name in ['Logistic Regression', 'Random Forest', 'XGBoost', '1D CNN', 'LSTM', 'CNN-LSTM']:
    model = server.ARTIFACTS[name]
    if name in ['1D CNN', 'LSTM', 'CNN-LSTM']:
        pred, prob = predict_deep_model(model, split['X_test_scaled'])
    else:
        pred = model.predict(split['X_test_scaled'])
        prob = model.predict_proba(split['X_test_scaled'])
    assert prob.shape == (75, 3) and np.isfinite(prob).all()
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-5)
    metrics.append({'Model': name, 'Accuracy': accuracy_score(split['y_test'], pred), 'Macro F1': f1_score(split['y_test'], pred, average='macro')})
raw = pd.read_csv(root / 'data/Sleep_health_and_lifestyle_dataset.csv').iloc[0]
sys_bp, dia_bp = map(float, raw['Blood Pressure'].split('/'))
profile = dict(heart_rate=float(raw['Heart Rate']), systolic_bp=sys_bp, diastolic_bp=dia_bp, sleep_duration=float(raw['Sleep Duration']), quality_of_sleep=float(raw['Quality of Sleep']), physical_activity=float(raw['Physical Activity Level']), daily_steps=float(raw['Daily Steps']), age=float(raw['Age']), gender=raw.Gender, bmi_category=raw['BMI Category'].replace('Normal Weight','Normal'), sleep_disorder='None' if pd.isna(raw['Sleep Disorder']) else raw['Sleep Disorder'], occupation=raw.Occupation)
response=server.compute_prediction(profile)
assert len(response['models']) == 6
vector=server.ARTIFACTS['scaler'].transform(frame[split['feature_cols']].iloc[[0]].to_numpy(dtype=np.float32))
for name, details in response['models'].items():
    model=server.ARTIFACTS[name]
    probabilities=predict_deep_model(model, vector)[1][0] if name in ['1D CNN','LSTM','CNN-LSTM'] else model.predict_proba(vector)[0]
    np.testing.assert_allclose(probabilities, details['probabilities'], atol=1e-5)
for invalid in [None, {}, {**profile, 'heart_rate': float('nan')}, {**profile, 'gender': 'unsupported'}, {**profile, 'systolic_bp': 90, 'diastolic_bp': 100}]:
    try:
        server.compute_prediction(invalid)
        raise AssertionError('Invalid input was accepted')
    except ValueError:
        pass
pd.DataFrame(metrics).to_csv(root / 'reports/reproduced_model_metrics.csv', index=False)
print(json.dumps(metrics, indent=2))
print('PASS: six artifacts, scaler parity, inference feature parity, probability normalization, invalid inputs.')
