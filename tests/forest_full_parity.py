"""Compare every real respondent without changing any models or data."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from src.compact_forest import CompactForest

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True)
parser.add_argument('--compact', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path(args.project)
models = root / 'models/current_models'
schema = json.loads((models / 'schema.json').read_text())
selection = json.loads((models / 'selection.json').read_text())
pre = joblib.load(models / 'preprocessor.joblib')
original = joblib.load(models / 'random_forest.joblib')
compact = CompactForest(args.compact, models / 'random_forest.joblib')
count = changes = 0
max_error = 0.0
start = time.monotonic()
for chunk in pd.read_csv(root / 'data/expanded_survey_dataset/respondents.csv', chunksize=4096):
    x = pre.transform(chunk[schema['numeric'] + schema['categorical']]).astype('float32')
    a, b = original.predict_proba(x), compact.predict_proba(x)
    max_error = max(max_error, float(np.max(np.abs(a-b))))
    for p in (a,b):
        p *= np.asarray(selection['Random Forest']['multipliers'])
        p /= p.sum(axis=1, keepdims=True)
    changes += int(np.count_nonzero(a.argmax(axis=1) != b.argmax(axis=1)))
    count += len(chunk)
    assert max_error < 1e-12 and changes == 0, (count, changes, max_error)
    print(f'PASS {count} respondents; max probability difference {max_error:.3g}', flush=True)
report = {'respondents': count, 'class_changes': changes, 'max_probability_error': max_error,
          'seconds': time.monotonic() - start, 'source_sha256': json.loads((Path(args.compact)/'manifest.json').read_text())['source_sha256']}
Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
