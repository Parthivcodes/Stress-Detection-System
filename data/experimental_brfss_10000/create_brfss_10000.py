like """Create a reproducible real-respondent subset without outcome-based cherry-picking."""
import csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path('C:/Users/parth/OneDrive/Desktop/stressleveldetection')
OUT=Path('C:/Users/parth/OneDrive/Documents/ChatGPT/AI project/output/brfss_10000')
SOURCE=ROOT/'data/expanded_survey_dataset/respondents.csv'
SEED='brfss10000-20261025-v1'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,o):(OUT/n).write_text(json.dumps(o,indent=2)+'\n',encoding='utf-8')
if OUT.exists():raise RuntimeError('Refusing to overwrite existing subset.')
OUT.mkdir(parents=True)
protected=[SOURCE]+[p for p in (ROOT/'models/current_models').rglob('*') if p.is_file()]+[ROOT/'models/active_model.json']
hashes={p.relative_to(ROOT).as_posix():sha(p) for p in protected};save('original_hashes.json',hashes)
schema=json.loads((ROOT/'models/current_models/schema.json').read_text());features=schema['numeric']+schema['categorical']
save('protocol.json',dict(rows=10000,seed=SEED,features=features,method='Largest-remainder allocation proportional to existing split x target counts; select smallest SHA256(seed:respondent_id) within each stratum. No prediction/performance filtering. Preserve all source columns and original field strings.',eligibility='Existing non-quarantine train/validation/test rows only. No new exclusions or relabeling.',evaluation='Existing validation and test memberships retained. These are historically reused internal splits, not independent new evidence. No accuracy target guaranteed and no model training performed.'))
with SOURCE.open(newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
assert len({r['respondent_id'] for r in rows})==len(rows)
groups=defaultdict(list)
for r in rows:
    if r['split'] in ('train','validation','test'):groups[(r['split'],r['target'])].append(r)
total=sum(map(len,groups.values()));alloc={k:10000*len(v)//total for k,v in groups.items()}
remainder=10000-sum(alloc.values())
for k in sorted(groups,key=lambda k:(-(10000*len(groups[k])%total),k))[:remainder]:alloc[k]+=1
selected=[]
for k in sorted(groups):
    ranked=sorted(groups[k],key=lambda r:hashlib.sha256((SEED+':'+r['respondent_id']).encode()).hexdigest())
    selected.extend(ranked[:alloc[k]])
selected.sort(key=lambda r:int(r['source_row']))
assert len(selected)==10000 and len({r['respondent_id'] for r in selected})==10000
split_groups=defaultdict(set);profiles=defaultdict(list)
for r in selected:
    for key in features:
        value=float(r[key]);lo,hi=schema['limits'][key]
        assert math.isfinite(value) and lo<=value<=hi,(r['respondent_id'],key)
        if key!='bmi':assert value==int(value)
    stress=int(r['stress_raw']);assert int(r['target'])==(0 if stress in (4,5) else 1 if stress==3 else 2)
    split_groups[r['split']].add(r['profile_group'])
    profiles[tuple(r[k] for k in features)].append(r)
for a in split_groups:
    for b in split_groups:
        if a!=b:assert not split_groups[a]&split_groups[b]
dest=OUT/'brfss_10000_respondents.csv'
with dest.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(selected)
with dest.open(newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f);assert reader.fieldnames==columns;reloaded=list(reader)
assert reloaded==selected
for n,h in hashes.items():assert sha(ROOT/n)==h
dist=[dict(split=k[0],target=int(k[1]),eligible=len(groups[k]),subset=alloc[k]) for k in sorted(groups)]
report=dict(rows=10000,unique_respondents=10000,features=features,columns_preserved=columns,source_sha256=hashes[SOURCE.relative_to(ROOT).as_posix()],subset_sha256=sha(dest),distribution=dist,split_counts=dict(Counter(r['split'] for r in selected)),class_counts=dict(Counter(r['target'] for r in selected)),identical_feature_groups=sum(len(v)>1 for v in profiles.values()),extra_rows_with_identical_features=sum(len(v)-1 for v in profiles.values()),identical_feature_groups_with_different_labels=sum(len({r['target'] for r in v})>1 for v in profiles.values()),checks=dict(originals_unchanged=True,all_field_values_identical_to_source=True,no_missing_predictors=True,predictors_within_existing_schema=True,target_mapping_valid=True,no_profile_group_overlap_between_splits=True),accuracy=None,models_retrained=False)
save('quality_report.json',report)
lines=['# BRFSS 10,000-respondent experimental subset','',
'Exactly 10,000 real respondents from the existing BRFSS 2022 dataset. The 19 predictors, labels, original codes and provenance columns are unchanged. This is a smaller subset of the existing data, not a newly collected or independently validated dataset.', '',
'## How records were selected','',
'Proportional sampling within each original split and target class, using a single fixed SHA-256 ranking of respondent IDs. No model scores, correctness, confidence or desired accuracy were used. Previously quarantined records remain excluded. Different people with identical feature values were retained.', '',
'| Split | Rarely | Sometimes | High | Total |','|---|---:|---:|---:|---:|']
for split in ['train','validation','test']:
    ns=[alloc[(split,str(c))] for c in range(3)];lines.append(f'| {split} | {ns[0]} | {ns[1]} | {ns[2]} | {sum(ns)} |')
lines += ['', '## Safe use','',
'Use only the 19 predictors listed in protocol.json. Use target as the label. Respondent IDs, source rows, split, profile_group, stress_raw and raw_* columns are provenance/label metadata, not additional model inputs. Feeding stress_raw or raw_stress_raw into a classifier would leak the answer.', '',
'Keep the split column. Do not randomly reshuffle existing validation/test people into training. Fit preprocessing only on train. Use profile_group for training-only grouped cross-validation. The validation/test rows belong to historically reused internal splits; they are not a fresh untouched test.', '',
'The existing models have already seen the selected training rows. Evaluating those models on the entire CSV would produce an invalid estimate of generalization. A genuine subset experiment must retrain models on the subset training rows and evaluate against a fixed comparison set.', '',
'## Quality checks and limitations','',
f"All checks passed: 10,000 unique IDs, exact original values, valid feature ranges/target mapping, no missing predictors and no cross-split profile-group overlap. {report['extra_rows_with_identical_features']} rows share all 19 features with another selected respondent. {report['identical_feature_groups_with_different_labels']} such feature groups contain different labels; these records were not removed to make classification easier.", '',
'This subset preserves the quality limitations of the source survey. Reducing row count does not remove measurement uncertainty or create information absent from the predictors. No accuracy improvement or 80% average accuracy has been demonstrated. No models were retrained and no active data/model paths changed.', '',
'The original data and models remain intact. quality_report.json records checks and hashes; protocol.json records the selection method. The standalone selection script makes the extraction reproducible.', '']
(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['rows','split_counts','class_counts','extra_rows_with_identical_features','checks']}))
