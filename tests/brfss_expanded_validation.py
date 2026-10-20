"""Independent audit of expanded data and frozen model evaluation."""
import sys,json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model_paths import preserved_path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score
from training.train_expanded_models import DATA,OUT,REPORT,NUM,CAT,FEATURES,EXTRA,NAMES,FILES,FIELDS,groupkey
from training.tune_baseline_models import probability,adjusted
from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
import torch,joblib

for p,h in json.loads((REPORT/'preserved_hashes.json').read_text()).items():
    assert hashlib.sha256(preserved_path(p).read_bytes()).hexdigest()==h,p
frame=pd.read_csv(DATA/'respondents.csv');old=pd.read_csv(ROOT/'data/baseline_survey_dataset/respondents_30000.csv')
assert len(frame)>30000 and frame.respondent_id.is_unique
assert frame.groupby('profile_group')['split'].nunique().max()==1
new=frame[frame.split.isin(['validation','test'])]
assert not set(new.respondent_id)&set(old.respondent_id)
assert not set(new.profile_group)&set(groupkey(old))
assert not set(frame.loc[frame.split=='train','profile_group'])&set(groupkey(old[old.split!='train']))
sample=frame.sample(100,random_state=77).set_index('source_row');source_checks=0
with zipfile.ZipFile(ROOT/'data/baseline_survey_dataset/raw/LLCP2022ASC.zip') as z,z.open(z.namelist()[0]) as f:
    for n,line in enumerate(f,1):
        if n not in sample.index:continue
        row=sample.loc[n]
        for key,(s,size) in FIELDS.items():assert row['raw_'+key]==int(line[s-1:s+size-1])
        for key,(s,_,_,_) in EXTRA.items():assert row['raw_'+key]==int(line[s-1:s])
        assert row.target==(0 if row.stress_raw>=4 else 1 if row.stress_raw==3 else 2)
        source_checks+=1
assert source_checks==100
saved=pd.read_csv(REPORT/'test_predictions.csv').set_index('respondent_id');te=frame[frame.split=='test'].set_index('respondent_id')
assert set(saved.index)==set(te.index) and saved.index.is_unique
assert np.array_equal(saved.target,te.loc[saved.index,'target'])
comparison=json.loads((REPORT/'comparison.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
pre=joblib.load(OUT/'preprocessor.joblib');torch.set_num_threads(4)
sample_ids=saved.sample(16,random_state=55).index;x=pre.transform(te.loc[sample_ids,FEATURES]).astype('float32')
for i,row in enumerate(comparison):
    name=row['Model']
    for version in ['original','tuned','expanded']:
        p=saved[[f'{name} {version} probability {c}' for c in range(3)]].to_numpy();pred=saved[f'{name} {version}'];metric=row[version]
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
        assert np.array_equal(p.argmax(1),pred)
        assert np.isclose(accuracy_score(saved.target,pred),metric['Accuracy'])
        assert np.isclose(f1_score(saved.target,pred,average='macro'),metric['F1-Score (Macro)'])
        assert np.allclose(recall_score(saved.target,pred,average=None),metric['Recall by class'])
        assert np.isclose(roc_auc_score(saved.target,p,multi_class='ovr'),metric['ROC-AUC (Macro)'])
    if i<3:model=joblib.load(OUT/FILES[i])
    else:
        model=[Conv1DNet,LSTMNet,CNNLSTMNet][i-3](input_dim=x.shape[1]);model.load_state_dict(torch.load(OUT/FILES[i],map_location='cpu',weights_only=True))
    actual=adjusted(probability(model,x),selection[name]['multipliers'])
    assert np.allclose(actual,saved.loc[sample_ids,[f'{name} expanded probability {c}' for c in range(3)]].to_numpy(),atol=1e-5)
    gate=selection[name]['eligible'] and all(row['expanded']['Recall by class'][2]>=row[v]['Recall by class'][2] and row['expanded']['Accuracy']>=row[v]['Accuracy'] for v in ['original','tuned'])
    assert gate==row['passes_no_more_high_stress_misses_and_accuracy_gate']
ensemble={}
for version in ['original','tuned','expanded']:
    votes=saved[[f'{name} {version}' for name in NAMES]].to_numpy()
    counts=np.stack([(votes==c).sum(1) for c in range(3)],axis=1)
    prediction=counts.argmax(1)  # Same lowest-class-ID tie break as the live API.
    ensemble[version]={'accuracy':float(accuracy_score(saved.target,prediction)),
                       'high_recall':float(recall_score(saved.target,prediction,labels=[2],average=None)[0]),
                       'macro_f1':float(f1_score(saved.target,prediction,average='macro'))}
ensemble_pass=all(ensemble['expanded']['accuracy']>=ensemble[v]['accuracy'] and ensemble['expanded']['high_recall']>=ensemble[v]['high_recall'] for v in ['original','tuned'])
all_pass=all(row['passes_no_more_high_stress_misses_and_accuracy_gate'] for row in comparison) and ensemble_pass
gate={'all_models_and_ensemble_pass':bool(all_pass),'ensemble_pass':bool(ensemble_pass),'ensemble_metrics':ensemble,
      'source_spot_checks':100,'saved_model_parity_checks':96,'originals_unchanged':True,
      'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()}
(REPORT/'deployment_gate.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
print('PASS: source responses spot-checked; prior files unchanged; unique respondents and isolated profiles; all 18 version/model metrics reproduced; 96 saved-model predictions match; deployment gates verified.')
print('DEPLOYMENT GATE:',gate)
