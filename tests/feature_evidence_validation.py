"""Read-only integrity check; no fitting or prediction on any split."""
import csv, hashlib, json, sys
from pathlib import Path
out=Path(sys.argv[1]);project=Path(sys.argv[2])
read=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
original=read('original_hashes.json')
for name,digest in original.items():
    assert sha(project/name)==digest, name
    assert sha(out/'backup'/name)==digest, 'backup '+name
frozen=read('frozen_selection.json')
for name,digest in frozen['hashes'].items():assert sha(out/name)==digest,name
assert read('TEST_STARTED.json')['frozen_sha256']==sha(out/'frozen_selection.json')
trials=list((out/'trials').glob('*.json'))
assert len(trials)==50
assert {json.loads(p.read_text())['trial'] for p in trials}==set(range(50))
roles=list(csv.DictReader((out/'training_roles.csv').open()))
assert len(roles)==len({r['respondent_id'] for r in roles})==128248
cal=read('calibration.json');a=set(cal['fit_respondents']);b=set(cal['check_respondents'])
assert not a&b
assert a|b=={r['respondent_id'] for r in roles if r['role']=='calibration_only'}
nb=read('Feature_Engineering_and_XGBoost.ipynb')
cells=[c for c in nb['cells'] if c['cell_type']=='code']
assert cells and all(c['execution_count'] and all(o['output_type']!='error' for o in c['outputs']) for c in cells)
rt=read('runtime_verification.json')['summary']
assert rt['requests']==512 and rt['model_predictions']==3072
assert rt['memory_limit']=='536870912' and 'oom_kill 0' in rt['memory_events']
print(f'PASS: {len(original)} protected files and backups, frozen artifacts, 50 trials, disjoint calibration, {len(cells)} executed notebook cells, and six-model API evidence.')
