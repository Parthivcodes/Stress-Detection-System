"""Exercise the running server with synthetic profiles; no extra test packages."""
import json
import math
import urllib.request
import urllib.error

BASE = 'http://127.0.0.1:8000'
EXPECTED = {'Random Forest', 'XGBoost', 'Logistic Regression', '1D CNN', 'LSTM', 'CNN-LSTM'}
profile = dict(heart_rate=75, systolic_bp=126, diastolic_bp=83,
               sleep_duration=7.2, quality_of_sleep=7, physical_activity=60,
               daily_steps=7000, age=35, gender='Male', bmi_category='Normal',
               sleep_disorder='None', occupation='Software Engineer')

def predict(data):
    request = urllib.request.Request(BASE + '/api/predict',
        data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

with urllib.request.urlopen(BASE + '/api/health', timeout=15) as response:
    assert json.load(response)['models_loaded'] is True
for title, changes in [('Typical', {}), ('Rested', dict(heart_rate=62, sleep_duration=8.5,
        quality_of_sleep=9)), ('Strained', dict(heart_rate=95, sleep_duration=5,
        quality_of_sleep=3))]:
    result = predict({**profile, **changes})
    assert result['status'] == 'success'
    assert set(result['models']) == EXPECTED
    for name, model in result['models'].items():
        probabilities = model['probabilities']
        assert len(probabilities) == 3
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        assert abs(sum(probabilities) - 1) < 1e-5
        assert math.isfinite(model['confidence']) and 0 <= model['confidence'] <= 1
    print(title, result['consensus']['label'], '— six valid model responses')
try:
    predict({**profile, 'heart_rate': -1})
except urllib.error.HTTPError as error:
    assert error.code == 400
else:
    raise AssertionError('Invalid input was accepted')
print('PASS: health, three profiles, all six models, probabilities, invalid input')
