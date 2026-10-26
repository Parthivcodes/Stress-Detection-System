"""Validate production HTTP results against original model outputs."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--api', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
baseline = json.loads(Path(args.baseline).read_text())
api = json.loads(Path(args.api).read_text())
assert len(baseline['results']) == len(api['results']) == 512
errors = {}
for old, new in zip(baseline['results'], api['results']):
    assert old['model_version'] == new['model_version']
    assert old['consensus']['class_id'] == new['consensus']['class_id']
    assert old['consensus']['agreement_rate'] == new['consensus']['agreement_rate']
    assert old['models'].keys() == new['models'].keys()
    for name in old['models']:
        a,b = old['models'][name],new['models'][name]
        assert a['predicted_class'] == b['predicted_class']
        assert a['label'] == b['label']
        delta = max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))
        errors[name] = max(errors.get(name,0),delta)
        assert delta < (1e-12 if name == 'Random Forest' else 1e-6)
summary = dict(api['summary'], maximum_probability_error=errors, class_changes=0)
Path(args.output).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'memory_stat'},indent=2))
