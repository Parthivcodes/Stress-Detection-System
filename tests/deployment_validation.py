"""Validate production WSGI auth and real-model parity without provisioning."""
import csv
import io
import json
import os
import secrets
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["INFERENCE_API_KEY"] = secrets.token_urlsafe(32)
from deploy.backend import application, ENGINE, LOCK

def request(path, method="GET", body=None, authorized=True):
    raw = json.dumps(body).encode() if body is not None else b""
    env = {"PATH_INFO":path, "REQUEST_METHOD":method,
           "CONTENT_LENGTH":str(len(raw)), "wsgi.input":io.BytesIO(raw)}
    if authorized:
        env["HTTP_AUTHORIZATION"] = "Bearer " + os.environ["INFERENCE_API_KEY"]
    statuses = []
    result = b"".join(application(env, lambda status, headers: statuses.append(int(status.split()[0]))))
    return statuses[0], json.loads(result)

assert request("/healthz", authorized=False)[0] == 200
assert request("/api/health", authorized=False)[0] == 401
assert request("/models/current_models/lstm_net.pt")[0] == 404
assert request("/api/schema")[1]["input_count"] == 19
assert request("/api/benchmarks")[0] == 200
assert request("/api/predict", "POST", {})[0] == 400
with (ROOT/"data/expanded_survey_dataset/respondents.csv").open(newline="") as stream:
    row = next(csv.DictReader(stream))
payload = {k:float(row[k]) for k in ENGINE.schema["numeric"] + ENGINE.schema["categorical"]}
status, result = request("/api/predict", "POST", payload)
assert status == 200 and len(result["models"]) == 6
direct = ENGINE.predict(payload)
for name, prediction in result["models"].items():
    assert prediction["predicted_class"] == direct["models"][name]["predicted_class"]
    assert all(abs(a-b) < 1e-5 for a,b in zip(prediction["probabilities"], direct["models"][name]["probabilities"]))
with LOCK:
    assert request("/api/predict", "POST", payload)[0] == 503
    assert request("/api/health")[0] == 200
print("PASS: production API authentication, no file exposure, real six-model parity, validation, bounded inference and responsive health.")
