"""Production WSGI API. No static files, datasets, or checkpoint downloads."""
import hmac
import json
import os
import threading
from pathlib import Path

from src.brfss_inference import BRFSSPredictor

ROOT = Path(__file__).resolve().parents[1]
TOKEN = os.environ.get("INFERENCE_API_KEY", "")
if len(TOKEN) < 32:
    raise RuntimeError("Set a server-only INFERENCE_API_KEY of at least 32 characters.")
ENGINE = BRFSSPredictor("brfss2022_expanded")
LOCK = threading.Lock()
AUDIT = json.loads((ROOT / "reports/brfss2022_expanded/dataset_audit.json").read_text())
METRICS = json.loads((ENGINE.root / "metrics.json").read_text())
SCHEMA = {
    "model_version": ENGINE.version, "input_count": 19,
    "respondents": AUDIT["eligible_records"],
    "test_respondents": AUDIT["split_counts"]["test"],
    "training_respondents": AUDIT["split_counts"]["train"],
}


def application(environ, start_response):
    def reply(code, body):
        data = json.dumps(body, allow_nan=False).encode()
        labels = {200:"OK", 400:"Bad Request", 401:"Unauthorized", 404:"Not Found",
                  405:"Method Not Allowed", 413:"Content Too Large", 500:"Internal Server Error",
                  503:"Service Unavailable"}
        start_response(f"{code} {labels[code]}", [
            ("Content-Type", "application/json"), ("Content-Length", str(len(data))),
            ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
        ])
        return [data]

    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "")
    # Public, non-sensitive readiness endpoint for the hosting health check.
    if path == "/healthz" and method == "GET":
        return reply(200, {"ready": True})
    if not hmac.compare_digest(environ.get("HTTP_AUTHORIZATION", ""), f"Bearer {TOKEN}"):
        return reply(401, {"status":"error", "message":"Unauthorized"})
    if path not in ("/api/health", "/api/schema", "/api/benchmarks", "/api/predict"):
        return reply(404, {"status":"error", "message":"Not found"})
    if method != ("POST" if path == "/api/predict" else "GET"):
        return reply(405, {"status":"error", "message":"Method not allowed"})
    if path == "/api/health":
        return reply(200, {"models_loaded":True, "dataset":"brfss2022", "model_version":ENGINE.version})
    if path == "/api/schema":
        return reply(200, SCHEMA)
    if path == "/api/benchmarks":
        return reply(200, METRICS)
    try:
        length = int(environ.get("CONTENT_LENGTH") or "0")
        if not 0 < length <= 16384:
            return reply(413, {"status":"error", "message":"Invalid request size"})
        body = json.loads(environ["wsgi.input"].read(length))
        # Bound concurrent heavy inference; health checks remain responsive.
        if not LOCK.acquire(blocking=False):
            return reply(503, {"status":"error", "message":"Models are busy. Please retry shortly."})
        try:
            result = ENGINE.predict(body)
        finally:
            LOCK.release()
        return reply(200, result)
    except (ValueError, TypeError, UnicodeDecodeError):
        return reply(400, {"status":"error", "message":"Invalid survey inputs"})
    except Exception:
        return reply(500, {"status":"error", "message":"Prediction failed"})
