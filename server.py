"""
Sechenov Polymer Bio-Telemetry Bridge Server
Serves the futuristic Sechenov UI and handles real-time inference across all 6 models.
Zero third-party web framework requirement (runs on built-in http.server).
"""

import os
import sys
import json
import joblib
import torch
import numpy as np
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import threading
import urllib.parse
import math

INPUT_RANGES = {
    'heart_rate': (55, 110), 'systolic_bp': (90, 170),
    'diastolic_bp': (60, 110), 'sleep_duration': (4, 10),
    'quality_of_sleep': (1, 10), 'physical_activity': (0, 120),
    'daily_steps': (0, 15000), 'age': (18, 100),
}

def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object.')
    for key, (low, high) in INPUT_RANGES.items():
        if key not in data or isinstance(data[key], bool):
            raise ValueError(f'{key} is required and must be numeric.')
        try:
            value = float(data[key])
        except (ValueError, TypeError):
            raise ValueError(f'{key} must be numeric.')
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be between {low} and {high}.')
    if float(data['systolic_bp']) <= float(data['diastolic_bp']):
        raise ValueError('Systolic pressure must exceed diastolic pressure.')
    for key, prefix in [('gender', 'Gender'), ('bmi_category', 'BMI Category'),
                        ('sleep_disorder', 'Sleep Disorder'), ('occupation', 'Occupation')]:
        if f'{prefix}_{data.get(key)}' not in ARTIFACTS['feature_cols']:
            raise ValueError(f'Unsupported {key}.')

# Import project architecture
from src.model_paths import LEGACY_ROOT
from src.models import Conv1DNet, LSTMNet, CNNLSTMNet, predict_deep_model
from src.preprocessing import CLASS_NAMES

PORT = int(os.environ.get('PORT', '8000'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "frontend")

# Global model container
MODELS_LOADED = False
ARTIFACTS = {}
BRFSS = None
PREDICTION_LOCK = threading.Lock()

def init_models():
    global MODELS_LOADED, ARTIFACTS, BRFSS
    if MODELS_LOADED:
        return
    if os.environ.get('STRESS_DATASET', 'brfss2022') != 'legacy':
        from src.brfss_inference import BRFSSPredictor
        BRFSS = BRFSSPredictor()
        MODELS_LOADED = True
        print('[NERVA] Six BRFSS 2022 models loaded.')
        return
    
    print("[STILL RESEARCH] Loading neural and statistical model weights...")
    
    with open(os.path.join(LEGACY_ROOT, "feature_cols.json"), "r") as f:
        feature_cols = json.load(f)
        
    scaler = joblib.load(os.path.join(LEGACY_ROOT, "scaler.pkl"))
    
    # Classical Models
    lr_model = joblib.load(os.path.join(LEGACY_ROOT, "logistic_regression.joblib"))
    rf_model = joblib.load(os.path.join(LEGACY_ROOT, "random_forest.joblib"))
    xgb_model = joblib.load(os.path.join(LEGACY_ROOT, "xgboost.joblib"))
    
    num_features = len(feature_cols)

    # Deep Learning Models
    device = torch.device('cpu')
    
    cnn_model = Conv1DNet(input_dim=num_features, num_classes=3)
    cnn_model.load_state_dict(torch.load(os.path.join(LEGACY_ROOT, "conv1d_net.pt"), map_location=device, weights_only=True))
    cnn_model.eval()

    lstm_model = LSTMNet(input_dim=num_features, hidden_dim=32, num_layers=1, num_classes=3)
    lstm_model.load_state_dict(torch.load(os.path.join(LEGACY_ROOT, "lstm_net.pt"), map_location=device, weights_only=True))
    lstm_model.eval()

    cnnlstm_model = CNNLSTMNet(input_dim=num_features, num_classes=3)
    cnnlstm_model.load_state_dict(torch.load(os.path.join(LEGACY_ROOT, "cnn_lstm_net.pt"), map_location=device, weights_only=True))
    cnnlstm_model.eval()

    ARTIFACTS = {
        'feature_cols': feature_cols,
        'scaler': scaler,
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        '1D CNN': cnn_model,
        'LSTM': lstm_model,
        'CNN-LSTM': cnnlstm_model
    }
    MODELS_LOADED = True
    print("[STILL RESEARCH] All 6 models and scalers loaded successfully into memory.")

def compute_prediction(data: dict) -> dict:
    if BRFSS is not None:
        return BRFSS.predict(data)
    validate_input(data)
    feature_cols = ARTIFACTS['feature_cols']
    scaler = ARTIFACTS['scaler']
    
    # Extract raw parameters with fallbacks
    heart_rate = float(data.get('heart_rate', 75))
    systolic_bp = float(data.get('systolic_bp', 126))
    diastolic_bp = float(data.get('diastolic_bp', 83))
    sleep_duration = float(data.get('sleep_duration', 7.2))
    quality_of_sleep = float(data.get('quality_of_sleep', 7))
    physical_activity = float(data.get('physical_activity', 60))
    daily_steps = float(data.get('daily_steps', 7000))
    age = float(data.get('age', 35))
    gender = str(data.get('gender', 'Male'))
    bmi_category = str(data.get('bmi_category', 'Normal'))
    sleep_disorder = str(data.get('sleep_disorder', 'None'))
    occupation = str(data.get('occupation', 'Software Engineer'))

    # Calculate Hemodynamic & Autonomic Biomarkers
    pulse_pressure = systolic_bp - diastolic_bp
    map_bp = diastolic_bp + (pulse_pressure / 3.0)
    rpp = (heart_rate * systolic_bp) / 100.0
    csi = (heart_rate / 72.0) * (systolic_bp / 120.0)
    sleep_efficiency = quality_of_sleep / (sleep_duration + 0.1)
    hr_activity_ratio = heart_rate / (physical_activity + 1.0)
    awake_hours = max(1.0, 24.0 - sleep_duration)
    steps_awake = daily_steps / awake_hours

    # Construct complete input vector
    input_dict = {col: 0.0 for col in feature_cols}
    input_dict['Age'] = age
    input_dict['Sleep Duration'] = sleep_duration
    input_dict['Quality of Sleep'] = quality_of_sleep
    input_dict['Physical Activity Level'] = physical_activity
    input_dict['Heart Rate'] = heart_rate
    input_dict['Daily Steps'] = daily_steps
    input_dict['Systolic_BP'] = systolic_bp
    input_dict['Diastolic_BP'] = diastolic_bp
    input_dict['Pulse_Pressure'] = pulse_pressure
    input_dict['Mean_Arterial_Pressure'] = map_bp
    input_dict['Rate_Pressure_Product'] = rpp
    input_dict['Cardiovascular_Stress_Index'] = csi
    input_dict['Sleep_Efficiency'] = sleep_efficiency
    input_dict['HR_to_Activity_Ratio'] = hr_activity_ratio
    input_dict['Steps_per_Hour_Awake'] = steps_awake

    # One-hot categorical mappings
    if f"Gender_{gender}" in input_dict:
        input_dict[f"Gender_{gender}"] = 1.0
    if f"BMI Category_{bmi_category}" in input_dict:
        input_dict[f"BMI Category_{bmi_category}"] = 1.0
    if f"Sleep Disorder_{sleep_disorder}" in input_dict:
        input_dict[f"Sleep Disorder_{sleep_disorder}"] = 1.0
    if f"Occupation_{occupation}" in input_dict:
        input_dict[f"Occupation_{occupation}"] = 1.0

    raw_vector = np.array([input_dict[c] for c in feature_cols], dtype=np.float32).reshape(1, -1)
    scaled_vector = scaler.transform(raw_vector)

    # Model evaluation map
    models_spec = [
        ("Random Forest", ARTIFACTS['Random Forest'], False),
        ("XGBoost", ARTIFACTS['XGBoost'], False),
        ("CNN-LSTM", ARTIFACTS['CNN-LSTM'], True),
        ("1D CNN", ARTIFACTS['1D CNN'], True),
        ("Logistic Regression", ARTIFACTS['Logistic Regression'], False),
        ("LSTM", ARTIFACTS['LSTM'], True)
    ]

    predictions_detail = {}
    class_votes = []
    
    for name, model_obj, is_dl in models_spec:
        if is_dl:
            pred, prob = predict_deep_model(model_obj, scaled_vector)
            p_cls = int(pred[0])
            p_probs = prob[0].tolist()
        else:
            p_cls = int(model_obj.predict(scaled_vector)[0])
            p_probs = model_obj.predict_proba(scaled_vector)[0].tolist()
            
        class_votes.append(p_cls)
        predictions_detail[name] = {
            "predicted_class": p_cls,
            "label": CLASS_NAMES[p_cls],
            "confidence": float(p_probs[p_cls]),
            "probabilities": [float(p) for p in p_probs]
        }

    # Consensus verdict
    # Ties resolve to the lowest class ID; agreement is not confidence.
    consensus_class = max(range(3), key=class_votes.count)
    consensus_label = CLASS_NAMES[consensus_class]
    consensus_agreement = class_votes.count(consensus_class) / len(class_votes)

    # Average class probability across models
    avg_probabilities = [0.0, 0.0, 0.0]
    for m in predictions_detail.values():
        for i in range(3):
            avg_probabilities[i] += m["probabilities"][i] / len(predictions_detail)

    # Feature contribution factors (for HUD Explainability bars)
    contributions = [
        {"name": "Heart Rate & Strain", "value": round(float(min(100, max(10, (heart_rate - 55) * 1.6))), 1)},
        {"name": "Rate-Pressure Product", "value": round(float(min(100, max(10, (rpp - 60) * 1.1))), 1)},
        {"name": "Sleep Deficit Factor", "value": round(float(min(100, max(10, (10 - sleep_duration) * 14))), 1)},
        {"name": "Autonomic Ratio", "value": round(float(min(100, max(10, hr_activity_ratio * 35))), 1)},
        {"name": "Vascular Perfusion (MAP)", "value": round(float(min(100, max(10, (map_bp - 70) * 1.3))), 1)}
    ]

    return {
        "status": "success",
        "consensus": {
            "class_id": consensus_class,
            "label": consensus_label,
            "agreement_rate": round(consensus_agreement * 100, 1),
            "probabilities": [round(p, 4) for p in avg_probabilities]
        },
        "models": predictions_detail,
        "biomarkers": {
            "pulse_pressure": round(pulse_pressure, 1),
            "map_bp": round(map_bp, 1),
            "rpp": round(rpp, 1),
            "csi": round(csi, 2),
            "sleep_efficiency": round(sleep_efficiency, 2),
            "hr_activity_ratio": round(hr_activity_ratio, 2),
            "steps_awake": round(steps_awake, 1)
        },
        "illustrative_indicators": contributions,
        "indicator_note": "Formula-based illustrations, not model feature attributions.",
        "tie_break_rule": "Lowest class ID among equally frequent votes"
    }

class SechenovServerHandler(SimpleHTTPRequestHandler):
    # Idle speculative browser sockets must not block other browsers.
    timeout = 15
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def translate_path(self, path):
        translated = super().translate_path(path)
        dashboard = os.path.join(STATIC_DIR, 'dashboard')
        # Serve the relocated React build, not its TypeScript source index.
        if os.path.commonpath([dashboard, translated]) == dashboard:
            return os.path.join(dashboard, 'dist', os.path.relpath(translated, dashboard))
        return translated

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/schema' and BRFSS:
            expanded = BRFSS.version == 'brfss2022_expanded'
            report = 'brfss2022_expanded' if expanded else 'brfss2022'
            with open(os.path.join(BASE_DIR, 'reports', report, 'dataset_audit.json')) as f:
                audit = json.load(f)
            result = {'model_version': BRFSS.version, 'input_count': len(BRFSS.schema['numeric'])+len(BRFSS.schema['categorical']),
                      'respondents': audit['eligible_records'] if expanded else audit['sample_records'],
                      'test_respondents': audit['split_counts']['test'] if expanded else audit['splits']['test'],
                      'training_respondents': audit['split_counts']['train'] if expanded else audit['splits']['train'],
                      'ensemble': BRFSS.ensemble_info()}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            return
        if self.path == '/api/benchmarks':
            path = str(BRFSS.root / 'metrics.json') if BRFSS else os.path.join(STATIC_DIR, 'benchmarks.json')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            with open(path, 'rb') as f:
                self.wfile.write(f.read())
            return
        if self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'models_loaded': MODELS_LOADED, 'dataset': 'brfss2022' if BRFSS else 'legacy', 'model_version': BRFSS.version if BRFSS else 'legacy'}).encode())
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if not 0 < content_length <= 16384:
                    raise ValueError('Request body must be between 1 and 16384 bytes.')
                post_body = self.rfile.read(content_length)
                req_data = json.loads(post_body.decode('utf-8'))
                
                # Serialize model execution, not static files or health checks.
                with PREDICTION_LOCK:
                    result = compute_prediction(req_data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(400 if isinstance(e, (ValueError, TypeError)) else 500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                err_resp = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(err_resp).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    # Ensure stdout handles UTF-8 on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    init_models()
    print("\n=======================================================")
    print(" [*] STILL STRESS INTELLIGENCE — LOCAL RESEARCH SERVER")
    print(f" URL: http://localhost:{PORT}")
    print(f" Static Assets: {STATIC_DIR}")
    print("=======================================================\n")
    server = ThreadingHTTPServer(('127.0.0.1', PORT), SechenovServerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down research server.")
        server.server_close()
