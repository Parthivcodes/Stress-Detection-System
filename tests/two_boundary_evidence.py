"""Read-only preservation, split-role and frozen-evidence checks."""
import csv,hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);project=Path(sys.argv[2])
read=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for name,digest in read('original_hashes.json').items():
    assert sha(project/name)==digest,name
    assert sha(out/'backup'/name)==digest,'backup '+name
f=read('frozen_selection.json')
for name,digest in f['hashes'].items():assert sha(out/name)==digest,name
for name,digest in f['source_hashes'].items():assert sha(Path(__file__).resolve().parents[1]/'tools'/name)==digest,name
assert read('TEST_STARTED.json')['frozen_sha256']==sha(out/'frozen_selection.json')
trials=list((out/'trials').glob('*.json'));assert len(trials)==40
assert {json.loads(p.read_text())['trial'] for p in trials}==set(range(40))
# Full rows never need to be loaded; verify group disjointness through source IDs.
with (out/'backup/data/expanded_survey_dataset/respondents.csv').open(newline='') as stream:
    groups={r['respondent_id']:(r['profile_group'],r['split']) for r in csv.DictReader(stream)}
foldgroups={};seen=set()
with (out/'cv_roles.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        rid=row['respondent_id'];fold=row['fold'];role=row['role'];group,split=groups[rid]
        assert split=='train';assert (fold,rid) not in seen;seen.add((fold,rid))
        key=(fold,group);assert key not in foldgroups or foldgroups[key]==role;foldgroups[key]=role
assert len(seen)==2*128248
finalgroups={};seen=set()
with (out/'final_roles.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        rid=row['respondent_id'];role=row['role'];group,split=groups[rid]
        assert split=='train' and rid not in seen;seen.add(rid)
        assert group not in finalgroups or finalgroups[group]==role;finalgroups[group]=role
assert len(seen)==128248
nb=read('Two_Boundary_Constrained_Tuning.ipynb');cells=[c for c in nb['cells'] if c['cell_type']=='code']
assert len(cells)==6 and all(c['execution_count'] and not any(o['output_type']=='error' for o in c['outputs']) for c in cells)
rt=read('runtime_verification.json')['summary'];assert rt['model_predictions']==3072 and rt['memory_limit']=='536870912' and 'oom_kill 0' in rt['memory_events']
print('PASS: active files/backups unchanged; 40 trials; disjoint respondent/profile-group roles; frozen hashes; six executed notebook cells; all-six-model runtime evidence.')
