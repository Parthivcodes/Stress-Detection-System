"""Read-only parity/memory harness, supports an isolated candidate checkout."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--count", type=int, default=512)
parser.add_argument("--compact")
args = parser.parse_args()
from src import model_paths
model_paths.PROJECT_ROOT = Path(args.project).resolve()
model_paths.MODEL_ROOT = model_paths.PROJECT_ROOT / "models"
if args.compact:
    os.environ["BRFSS_COMPACT_FOREST_DIR"] = args.compact
else:
    os.environ.pop("BRFSS_COMPACT_FOREST_DIR", None)
from src.brfss_inference import BRFSSPredictor

started = time.monotonic()
engine = BRFSSPredictor("brfss2022_expanded")
loaded_seconds = time.monotonic() - started
keys = engine.schema["numeric"] + engine.schema["categorical"]
results = []
# Spread records deterministically across the entire file, rather than only
# testing its first respondents. No training or sampling-based model selection.
with (model_paths.PROJECT_ROOT / "data/expanded_survey_dataset/respondents.csv").open(newline="") as stream:
    reader = csv.DictReader(stream)
    stride = max(1, 183040 // args.count)
    for index, row in enumerate(reader):
        if index % stride or len(results) >= args.count:
            continue
        payload = {k: float(row[k]) for k in keys}
        results.append(engine.predict(payload))
elapsed = time.monotonic() - started
report = {
    "results": results, "count": len(results),
    "load_seconds": loaded_seconds, "total_seconds": elapsed,
    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
with Path(args.output).open("x") as stream:
    json.dump(report, stream)
print(json.dumps({k:v for k,v in report.items() if k != "results"}), flush=True)
