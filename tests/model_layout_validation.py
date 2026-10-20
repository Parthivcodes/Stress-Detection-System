"""Check relocation integrity; --inference additionally loads all model versions."""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_paths import LEGACY_ROOT, model_dir, preserved_path

parser = argparse.ArgumentParser()
parser.add_argument("--inference", action="store_true")
args = parser.parse_args()

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

checked = 0
for record in json.loads((ROOT / "reports/project_cleanup/before.json").read_text())["inventory"]:
    name = record["path"]
    if name.startswith(("models/", "data/")):
        assert digest(preserved_path(name)) == record["sha256"], name
        checked += 1
for directory, manifest in [
    ("brfss2022", "legacy_hashes.json"),
    ("brfss2022_tuned", "preserved_hashes.json"),
    ("brfss2022_expanded", "preserved_hashes.json"),
]:
    for name, expected in json.loads((ROOT / "reports" / directory / manifest).read_text()).items():
        assert digest(preserved_path(name)) == expected, name

assert json.loads((ROOT / "models/active_model.json").read_text())["version"] == "brfss2022_expanded"
assert {p.name for p in (ROOT / "models").iterdir()} == {
    "current_models", "active_model.json", "README.md"
}
for name in ["legacy/original_dashboard.py", "server.py", "training/train_original_models.py",
             "training/train_baseline_models.py", "training/tune_baseline_models.py",
             "training/train_expanded_models.py", "src/model_paths.py", "src/brfss_inference.py"]:
    ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), filename=name)
print(f"PASS: {checked} original data/model files and all historical manifests match; layout and syntax valid.")

if args.inference:
    from src.brfss_inference import BRFSSPredictor
    # No environment override: exercise the actual default configuration.
    os.environ.pop("BRFSS_MODEL_VERSION", None)
    for version in [None, "brfss2022", "brfss2022_tuned"]:
        engine = BRFSSPredictor(version)
        expected = version or "brfss2022_expanded"
        assert engine.version == expected
        csv_path = ROOT / ("data/expanded_survey_dataset/respondents.csv"
                           if version is None else "data/baseline_survey_dataset/respondents_30000.csv")
        with csv_path.open(newline="") as stream:
            row = next(csv.DictReader(stream))
        keys = engine.schema["numeric"] + engine.schema["categorical"]
        result = engine.predict({key: float(row[key]) for key in keys})
        assert len(result["models"]) == 6
        for prediction in result["models"].values():
            assert abs(sum(prediction["probabilities"]) - 1) < 1e-5
        print(f"PASS: {expected}: six real predictions from {engine.root.relative_to(ROOT)}.")
    os.environ["STRESS_DATASET"] = "legacy"
    import server
    server.init_models()
    assert server.MODELS_LOADED and server.BRFSS is None
    print(f"PASS: all six legacy models load from {LEGACY_ROOT.relative_to(ROOT)}.")
