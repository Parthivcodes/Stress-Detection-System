"""Validation-only selection; one frozen final test pass; no original writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ensemble_policy import classify, matched_threshold, weighted_probabilities, ordinal_probabilities
from src.models import Conv1DNet, LSTMNet, CNNLSTMNet

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=['validation','test'])
parser.add_argument('--project', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
ROOT = Path(args.project).resolve(); OUT = Path(args.output).resolve()
MODELS = ROOT/'models/current_models'
torch.set_num_threads(4)
torch.manual_seed(20261018); np.random.seed(20261018)
NAMES=['Logistic Regression','Random Forest','XGBoost','1D CNN','LSTM','CNN-LSTM']
FILES=['logistic_regression.joblib','random_forest.joblib','xgboost.joblib','conv1d_net.pt','lstm_net.pt','cnn_lstm_net.pt']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def metrics(y,p,pred):
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'qwk':float(cohen_kappa_score(y,pred,weights='quadratic')),
            'recall':recall_score(y,pred,labels=[0,1,2],average=None,zero_division=0).tolist(),
            'macro_auc_ovr':float(roc_auc_score(y,p,multi_class='ovr',average='macro')),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}

def neural(model,x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+256])).softmax(1).numpy() for i in range(0,len(x),256)])

def originals(x):
    result={}
    for i,(name,file) in enumerate(zip(NAMES,FILES)):
        if i<3:
            model=joblib.load(MODELS/file); p=model.predict_proba(x)
        else:
            model=[Conv1DNet,LSTMNet,CNNLSTMNet][i-3](input_dim=x.shape[1])
            model.load_state_dict(torch.load(MODELS/file,map_location='cpu',weights_only=True));p=neural(model,x)
        result[name]=p
        print('INFERENCE',name,len(x),flush=True)
        del model
    return result

def baseline(y,raw):
    selection=json.loads((MODELS/'selection.json').read_text())
    ps=[]; votes=[]; report={}
    for name in NAMES:
        p=raw[name]*np.asarray(selection[name]['multipliers']);p=p/p.sum(axis=1,keepdims=True)
        ps.append(p);votes.append(p.argmax(1));report[name+' existing adjusted']=metrics(y,p,p.argmax(1))
    vote=np.array(votes).T
    winner=np.stack([(vote==i).sum(axis=1) for i in range(3)],axis=1).argmax(1)
    report['Existing majority vote']=metrics(y,np.mean(ps,axis=0),winner)
    p=np.tile([1.,0.,0.],(len(y),1))
    report['Always rarely baseline']=metrics(y,p,np.zeros(len(y),dtype=int))
    return report

def rank(record):
    m=record['metrics']
    return (m['macro_f1'],m['qwk'],m['recall'][1],m['macro_auc_ovr'],m['accuracy'])

def candidate(y,raw,weights):
    p=weighted_probabilities(raw,weights);t=matched_threshold(y,p)
    return {'weights':weights,'high_threshold':t,'metrics':metrics(y,p,classify(p,t))}

def search(y,raw,xgb_name):
    found=[]
    for x in range(1,10):
        found.append(candidate(y,raw,{xgb_name:x/10,'LSTM':(10-x)/10}))
    for lr in range(1,9):
        for x in range(1,10-lr):
            found.append(candidate(y,raw,{'Logistic Regression':lr/10,xgb_name:x/10,'LSTM':(10-lr-x)/10}))
    return found

schema=json.loads((MODELS/'schema.json').read_text());keys=schema['numeric']+schema['categorical']
data_path=ROOT/'data/expanded_survey_dataset/respondents.csv'
pre=joblib.load(MODELS/'preprocessor.joblib')
protected=[data_path,MODELS/'preprocessor.joblib',MODELS/'schema.json',MODELS/'selection.json',MODELS/'metrics.json',ROOT/'models/active_model.json']+[MODELS/f for f in FILES]

if args.phase=='validation':
    if OUT.exists(): raise SystemExit('Refusing to overwrite an existing experiment. No test-driven reruns.')
    OUT.mkdir(parents=True)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    dump(OUT/'original_hashes.json',hashes)
    dump(OUT/'protocol.json',{
        'seed':20261018,'selection_split':'validation','target_high_recall':0.60,
        'probabilities':'Raw unadjusted model probabilities; one high cutoff after averaging.',
        'threshold':'Largest validation cutoff achieving >=0.60 high recall; ties included.',
        'weights':'Positive weights in 0.1 steps summing to 1; 9 two-model and 36 three-model combinations per XGB type.',
        'rank':['macro_f1','qwk','sometimes_recall','macro_auc_ovr','accuracy'],
        'ordinal_gate':'Higher validation macro-F1 and nonlower QWK than raw normal XGB, both at matched recall.',
        'ordinal_training':'Same selected XGB settings; 350 max rounds, depth 5, lr .04, early stopping 25 on validation logloss; original three-class sqrt inverse-frequency training sample weights for both binaries.',
        'ordinal_projection':'B=min(B,A); probabilities=[1-A,A-B,B].',
        'test_policy':'Freeze thresholds and selected candidate before one test pass. No test-based retuning.',
        'activation_gate':'Frozen candidate test recall >=0.60 and macro-F1/QWK >= existing vote; otherwise retain original default, no alternative selected using test.',
        'caveat':'Reused internal holdout: prior test results informed the proposal, so not an untouched external evaluation.'})
    frame=pd.read_csv(data_path);train=frame.loc[frame.split=='train'];val=frame.loc[frame.split=='validation']
    x=pre.transform(train[keys]).astype('float32'); y=train.target.to_numpy()
    v=pre.transform(val[keys]).astype('float32');vy=val.target.to_numpy()
    assert not set(train.profile_group)&set(val.profile_group)
    raw=originals(v)
    weights=len(y)/(3*np.bincount(y,minlength=3)); sample_weight=weights[y]**.5
    binary=[]
    for label,ty,ey in [('at_least_sometimes',y>=1,vy>=1),('high',y==2,vy==2)]:
        print('TRAIN ordinal',label,flush=True)
        model=XGBClassifier(n_estimators=350,max_depth=5,learning_rate=.04,min_child_weight=12,
              reg_lambda=5,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=20261018,
              objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=25)
        model.fit(x,ty.astype(int),sample_weight=sample_weight,eval_set=[(v,ey.astype(int))],verbose=False)
        joblib.dump(model,OUT/f'ordinal_{label}.joblib')
        binary.append(model.predict_proba(v)[:,1])
        dump(OUT/f'ordinal_{label}_training.json',{'best_iteration':model.best_iteration,'best_score':float(model.best_score),'evals':model.evals_result()})
    raw['Ordinal XGBoost']=ordinal_probabilities(*binary)
    report=baseline(vy,raw)
    normal=candidate(vy,raw,{'XGBoost':1.0}); ordinal=candidate(vy,raw,{'Ordinal XGBoost':1.0})
    ordinal_pass=ordinal['metrics']['macro_f1']>normal['metrics']['macro_f1'] and ordinal['metrics']['qwk']>=normal['metrics']['qwk']
    matched={name:candidate(vy,raw,{name:1.0}) for name in NAMES+['Ordinal XGBoost']}
    trials=search(vy,raw,'XGBoost')
    ordinal_trials=search(vy,raw,'Ordinal XGBoost') if ordinal_pass else []
    eligible=trials+ordinal_trials
    selected=max(eligible,key=rank)
    frozen={'format':1,'selection_split':'validation','weights':selected['weights'],'high_threshold':selected['high_threshold'],
            'validation_metrics':selected['metrics'],'ordinal_eligible':bool(ordinal_pass),
            'matched_recall_baselines':matched,
            'source_sha256':{file:digest(MODELS/file) for file in ['preprocessor.joblib','schema.json','logistic_regression.joblib','xgboost.joblib','lstm_net.pt']},
            'ordinal_sha256':{file:digest(OUT/file) for file in ['ordinal_at_least_sometimes.joblib','ordinal_high.joblib']}}
    dump(OUT/'validation_trials.json',trials+ordinal_trials)
    dump(OUT/'validation_existing_baselines.json',report)
    dump(OUT/'frozen_selection.json',frozen)
    dump(OUT/'validation_summary.json',{'normal_xgb_matched':normal,'ordinal_xgb_matched':ordinal,'ordinal_eligible':bool(ordinal_pass),
         'ordinal_crossing_fraction':float(np.mean(binary[1]>binary[0])), 'selected':selected,'candidates':len(eligible)})
    print(json.dumps({'ordinal_eligible':bool(ordinal_pass),'selected':selected},indent=2),flush=True)
else:
    if (OUT/'TEST_STARTED.json').exists(): raise SystemExit('Final test already started; refusing another test evaluation.')
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    hashes=json.loads((OUT/'original_hashes.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    for f,h in frozen['ordinal_sha256'].items(): assert digest(OUT/f)==h
    dump(OUT/'TEST_STARTED.json',{'frozen_selection_sha256':digest(OUT/'frozen_selection.json'),'started_unix':time.time()})
    frame=pd.read_csv(data_path);test=frame.loc[frame.split=='test']
    t=pre.transform(test[keys]).astype('float32');ty=test.target.to_numpy()
    raw=originals(t)
    binaries=[joblib.load(OUT/f'ordinal_{label}.joblib').predict_proba(t)[:,1] for label in ['at_least_sometimes','high']]
    raw['Ordinal XGBoost']=ordinal_probabilities(*binaries)
    report=baseline(ty,raw)
    for name,record in frozen['matched_recall_baselines'].items():
        p=raw[name];report[name+' validation-matched threshold']=metrics(ty,p,classify(p,record['high_threshold']))
    p=weighted_probabilities(raw,frozen['weights']);chosen=metrics(ty,p,classify(p,frozen['high_threshold']))
    report['Frozen soft ensemble']=chosen
    old=report['Existing majority vote']
    approved=chosen['recall'][2]>=.60 and chosen['macro_f1']>=old['macro_f1'] and chosen['qwk']>=old['qwk']
    dump(OUT/'test_report.json',{'records':len(ty),'comparisons':report,'ordinal_crossing_fraction':float(np.mean(binaries[1]>binaries[0])),
                               'approved':bool(approved),'test_caveat':'Reused internal holdout; no tuning performed after evaluation.'})
    dump(OUT/'policy.json',{'enabled':bool(approved),'method':'weighted_raw_probabilities_with_high_cutoff',
           'weights':frozen['weights'],'high_threshold':frozen['high_threshold'],'source_sha256':frozen['source_sha256'],
           'ordinal_sha256':frozen['ordinal_sha256'],'frozen_selection_sha256':digest(OUT/'frozen_selection.json'),
           'test_metrics':chosen,'majority_baseline_accuracy':report['Always rarely baseline']['accuracy']})
    print(json.dumps({'approved':bool(approved),'test_metrics':chosen,'old_vote':old,
                      'always_rarely':report['Always rarely baseline']},indent=2),flush=True)
assert all(digest(ROOT/p)==h for p,h in hashes.items())
print('PASS original data, models and config checksums unchanged.',flush=True)
