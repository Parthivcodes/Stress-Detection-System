"""Exercise the relocated website and six-model API on an ephemeral local port."""
import csv
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from http.server import ThreadingHTTPServer
import socket
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("BRFSS_MODEL_VERSION", None)
os.environ.pop("STRESS_DATASET", None)
import server


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        key = "src" if tag == "script" else "href" if tag == "link" else None
        value = attrs.get(key, "")
        if value and not value.startswith(("data:", "http", "#")):
            self.urls.add(value)


server.init_models()
httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.SechenovServerHandler)
worker = threading.Thread(target=httpd.serve_forever, daemon=True)
worker.start()
base = f"http://127.0.0.1:{httpd.server_port}"

def get(path):
    with urllib.request.urlopen(base + path, timeout=20) as response:
        assert response.status == 200
        return response.read()

try:
    # Reproduce an external browser preopening connections without sending requests.
    idle = [socket.create_connection(("127.0.0.1", httpd.server_port)) for _ in range(6)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(lambda _: json.loads(get("/api/health"))["models_loaded"], range(16)))
    for connection in idle:
        connection.close()
    health = json.loads(get("/api/health"))
    assert health["models_loaded"] and health["model_version"] == "brfss2022_expanded"
    assert json.loads(get("/api/schema"))["input_count"] == 19
    parser = Assets()
    parser.feed(get("/").decode())
    for asset in parser.urls:
        assert get("/" + asset.lstrip("/")), asset
    assert b"#predictions" in get("/workspace.html")
    # The additional dashboard's compiled files must remain available.
    dashboard_assets = Assets()
    dashboard_assets.feed(get("/dashboard/").decode())
    for asset in dashboard_assets.urls:
        assert get("/dashboard/" + asset.removeprefix("./").lstrip("/"))
    with (ROOT / "data/expanded_survey_dataset/respondents.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    keys = server.BRFSS.schema["numeric"] + server.BRFSS.schema["categorical"]
    payload = {key: float(row[key]) for key in keys}
    request = urllib.request.Request(base + "/api/predict",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert len(result["models"]) == 6
    assert result["model_version"] == "brfss2022_expanded"
    direct = server.BRFSS.predict(payload)
    for name, prediction in result["models"].items():
        assert prediction["predicted_class"] == direct["models"][name]["predicted_class"]
        # Threaded CPU kernels may differ by floating-point rounding.
        assert all(abs(a-b) < 1e-5 for a,b in zip(prediction["probabilities"], direct["models"][name]["probabilities"]))
    invalid = urllib.request.Request(base + "/api/predict", data=b"{}",
                                    headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(invalid, timeout=10)
    except urllib.error.HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Invalid input accepted")
    # Import relocated training modules without running/retraining anything.
    import training.train_baseline_models
    import training.tune_baseline_models
    import training.train_expanded_models
    print("PASS: relocated website/assets, health, 19-input schema, six real HTTP predictions, invalid-input rejection and training imports.")
finally:
    httpd.shutdown()
    httpd.server_close()
    worker.join(timeout=5)
