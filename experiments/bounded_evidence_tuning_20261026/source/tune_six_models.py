"""Bounded, grouped, leakage-aware tuning. Production artifacts are read-only.

Phases: audit -> search -> select -> test -> report. Test is a one-shot phase.
Resume completed search/final fits, never rerun the holdout for tuning.
"""
import argparse, copy, gc, hashlib, json, os, platform, resource, shutil, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, log_loss
from xgboost import XGBClassifier

NAMES=['Logistic Regression','Random Forest','XGBoost','1D CNN','LSTM','CNN-LSTM']
FILES=['logistic_regression.joblib','random_forest.joblib','xgboost.joblib','conv1d_net.pt','lstm_net.pt','cnn_lstm_net.pt']
SEED=20261023

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')

def read(p): return json.loads(Path(p).read_text())

def metric(y,p,pred=None):
    p=np.asarray(p,dtype=float);p=p/p.sum(1,keepdims=True)
    pred=p.argmax(1) if pred is None else pred
    cm=confusion_matrix(y,pred,labels=[0,1,2]);r=np.diag(cm)/np.maximum(cm.sum(1),1);pr=np.diag(cm)/np.maximum(cm.sum(0),1)
    f=2*r*pr/np.maximum(r+pr,1e-15)
    return dict(accuracy=float(np.trace(cm)/cm.sum()),balanced_accuracy=float(r.mean()),macro_f1=float(f.mean()),qwk=float(cohen_kappa_score(y,pred,weights='quadratic')),recall=r.tolist(),precision=pr.tolist(),f1=f.tolist(),macro_auc=float(roc_auc_score(y,p,multi_class='ovr')),confusion_matrix=cm.tolist(),high_detected=int(cm[2,2]),high_missed=int(cm[2,:2].sum()))

def cutoff(y,p,floor):
    h=np.sort(p[y==2,2]);return float(h[max(0,len(h)-int(np.ceil(floor*len(h)-1e-10)))])

def decide(p,t): return np.where(p[:,2]>=t,2,p[:,:2].argmax(1))

def prob(model,x):
    if not isinstance(model,torch.nn.Module): return model.predict_proba(x)
    model.eval()
    with torch.no_grad(): return np.concatenate([model(torch.from_numpy(x[i:i+256])).softmax(1).numpy() for i in range(0,len(x),256)])

def preprocessor(schema):
    cats=[list(range(schema['limits'][k][0],schema['limits'][k][1]+1)) for k in schema['categorical']]
    return ColumnTransformer([('numeric',StandardScaler(),schema['numeric']),('categorical',OneHotEncoder(categories=cats,sparse_output=False,handle_unknown='error'),schema['categorical'])])

def partition(frame,n,seed):
    a,b=next(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(frame,frame.target,frame.profile_group))
    assert not set(frame.iloc[a].profile_group)&set(frame.iloc[b].profile_group)
    return frame.iloc[a],frame.iloc[b]

def original(i,x):
    if i<3: m=joblib.load(MODELS/FILES[i])
    else:
        m=CLASSES[i-3](input_dim=x.shape[1]);m.load_state_dict(torch.load(MODELS/FILES[i],map_location='cpu',weights_only=True))
    p=prob(m,x);del m;gc.collect();return p

def protect():
    for name,digest in read(OUT/'original_hashes.json').items():
        assert sha(ROOT/name)==digest, 'Original changed: '+name

def get_configs():
    configs={}
    for i in range(3):
        model=joblib.load(MODELS/FILES[i]);params=model.get_params();del model
        params['n_jobs']=4
        if i==0:
            params.pop('n_jobs',None);params['class_weight']=None
            variants=[{},dict(C=.01),dict(C=10.,tol=1e-5)]
        elif i==1:
            variants=[{},dict(n_estimators=180,max_depth=12,min_samples_leaf=30,max_features=.6,max_samples=.85),dict(n_estimators=220,max_depth=16,min_samples_leaf=20,max_features=.8,max_samples=.9)]
        else:
            # XGBoost's default NaN sentinel is omitted from JSON and restored
            # by the constructor default; cleaned input contains no NaNs.
            if 'missing' in params and np.isnan(params['missing']):params.pop('missing')
            variants=[{},dict(n_estimators=500,max_depth=3,learning_rate=.03,min_child_weight=20,reg_lambda=10,reg_alpha=.5),dict(n_estimators=450,max_depth=4,learning_rate=.04,min_child_weight=20,reg_lambda=10,reg_alpha=1.,gamma=.1)]
        configs[NAMES[i]]=[dict(id=j,params={**params,**v},weight_power=.5 if i!=1 else 1.) for j,v in enumerate(variants)]
    for i in range(3,6):
        configs[NAMES[i]]=[dict(id=0,lr=.0007,weight_decay=.001,batch_size=512,weight_power=.5,dropout=None,scheduler=False),dict(id=1,lr=.001,weight_decay=.003,batch_size=256,weight_power=.5,dropout=.35,scheduler=True),dict(id=2,lr=.0004,weight_decay=.0003,batch_size=512,weight_power=.35,dropout=.2,scheduler=True)]
    return configs

def fit(i,c,fitframe,dev,seed,stage):
    began=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
    pre=preprocessor(SCHEMA);x=pre.fit_transform(fitframe[FEATURES]).astype('float32');v=pre.transform(dev[FEATURES]).astype('float32')
    y=fitframe.target.to_numpy();vy=dev.target.to_numpy();weights=(len(y)/(3*np.bincount(y,minlength=3)))**c['weight_power']
    history=[]
    if i<3:
        params=copy.deepcopy(c['params']);params['random_state']=seed
        if i==0: params['class_weight']=dict(enumerate(weights));m=LogisticRegression(**params);m.fit(x,y)
        elif i==1: m=RandomForestClassifier(**params);m.fit(x,y)
        else:
            # Baseline config reproduces its original fixed boosting budget;
            # alternatives early-stop on a training-only development partition.
            if c['id']!=0: params['early_stopping_rounds']=25
            m=XGBClassifier(**params);m.fit(x,y,sample_weight=weights[y],eval_set=[(v,vy)],verbose=False)
            history=m.evals_result()
    else:
        m=CLASSES[i-3](input_dim=x.shape[1])
        if c['dropout'] is not None:
            for layer in m.modules():
                if isinstance(layer,torch.nn.Dropout):layer.p=c['dropout']
        optimizer=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2) if c['scheduler'] else None
        lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=c['batch_size'],shuffle=True)
        best=float('inf');state=None;stale=0;limit=4 if stage=='screen' else 14;seconds=60 if stage=='screen' else 240
        for epoch in range(1,limit+1):
            m.train();total=0.
            for bx,by in loader:
                optimizer.zero_grad();loss=lossfn(m(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);optimizer.step();total+=float(loss.detach())*len(by)
            val_loss=float(log_loss(vy,prob(m,v),labels=[0,1,2]))
            history.append(dict(epoch=epoch,train_weighted_loss=total/len(y),dev_unweighted_logloss=val_loss))
            if val_loss<best-1e-5:best=val_loss;state=copy.deepcopy(m.state_dict());stale=0
            else:stale+=1
            if scheduler:scheduler.step(val_loss)
            print('EPOCH',NAMES[i],c['id'],stage,seed,epoch,round(val_loss,5),flush=True)
            if stale>=4 or time.monotonic()-began>=seconds:break
        m.load_state_dict(state)
    info=dict(seconds=time.monotonic()-began,history=history,train_rows=len(y),dev_rows=len(vy),seed=seed,config=c,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return pre,m,info

def audit():
    if (OUT/'protocol.json').exists():raise RuntimeError('Audit refuses to overwrite a frozen experiment.')
    OUT.mkdir(parents=True,exist_ok=True)
    protected=list(MODELS.rglob('*'))+[ROOT/'models/active_model.json',ROOT/'data/expanded_survey_dataset/respondents.csv']
    protected += [ROOT/p for p in ['src/models.py','src/brfss_inference.py','src/ensemble_policy.py','src/model_paths.py','server.py','deploy/backend.py','frontend/prediction.js','training/train_expanded_models.py','training/train_baseline_models.py']]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    save(OUT/'original_hashes.json',manifest)
    for p in protected:
        if p.is_file():
            dest=OUT/'backup'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==sha(p),'Incomplete backup differs; inspect manually.'
            else:shutil.copyfile(p,dest)
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv')
    assert df.respondent_id.is_unique
    assert df.groupby('profile_group').split.nunique().max()==1
    assert not df[FEATURES].isna().any().any()
    assert np.array_equal(df.target,np.where(df.stress_raw>=4,0,np.where(df.stress_raw==3,1,2)))
    assert not set(FEATURES)&{'stress_raw','target','respondent_id','source_row','profile_group','split','MENTHLTH','LSATISFY','ADDEPEV3'}
    for col in FEATURES:
        lo,hi=SCHEMA['limits'][col];assert df[col].between(lo,hi).all()
    assert np.array_equal(df.unhealthy_days,np.where(df.raw_unhealthy_days==88,0,df.raw_unhealthy_days))
    assert np.allclose(df.bmi,df.raw_bmi/100)
    tr=df[df.split=='train'];pre=preprocessor(SCHEMA).fit(tr[FEATURES]);old=joblib.load(MODELS/'preprocessor.joblib')
    for key in ['mean_','scale_','var_']:
        assert np.allclose(getattr(pre.named_transformers_['numeric'],key),getattr(old.named_transformers_['numeric'],key),rtol=1e-12,atol=1e-12)
    assert np.array_equal(pre.transform(tr[FEATURES].iloc[:100]),old.transform(tr[FEATURES].iloc[:100]))
    a=dict(rows=len(df),unique_respondents=int(df.respondent_id.nunique()),split_counts=df.split.value_counts().to_dict(),duplicate_full_feature_rows=int(df.duplicated(FEATURES).sum()),duplicate_feature_target_rows=int(df.duplicated(FEATURES+['target']).sum()),profile_groups_crossing_splits=0,features=FEATURES,target_mapping={'4/5':0,'3':1,'1/2':2},train_preprocessor_matches=True,raw_missing_policy='Complete cases: reject invalid/refused/dont-know codes; physical unhealthy days 88->0; BMI /100.',limitations=['Complete-case US survey; not survey-weighted population estimates.','Predictors and stress are concurrent associations, not causal or future forecasts.','Support/isolation/general health can overlap conceptually with distress; not direct target-derived fields.','Test was previously reused and cannot establish independent generalization.'])
    save(OUT/'audit.json',a)
    configs=get_configs()
    protocol=dict(seed=SEED,secondary_seed=SEED+1,features=FEATURES,configs=configs,cpu_threads=4,cuda=torch.cuda.is_available(),python=platform.python_version(),memory=Path('/proc/meminfo').read_text().splitlines()[:3],data_sha256=manifest['data/expanded_survey_dataset/respondents.csv'],screen='Training profiles whose hash integer modulo 4 is zero (~25% training); 2 stratified grouped outer folds; each outer training fold split into 80% fit, 10% early-stop development, 10% threshold calibration using grouped folds. Same fold IDs across trials.',screen_epochs=4,screen_neural_seconds_per_fit=60,final='Best training-CV configuration per model; fit 90% of full training split, early stop on remaining grouped 10%; two predetermined seeds for each neural network; first seed is candidate, second is stability evidence only.',final_max_epochs=14,final_neural_seconds_per_fit=240,selection='CV rank: minimum recall-constraint violation, then highest mean accuracy, macro-F1 and QWK. Final validation floor=max(.60,original adjusted recall); ensemble floor=max(.60,current ensemble recall). Highest cutoff meeting floor. Candidates must improve accuracy and not reduce macro-F1, QWK or sometimes recall versus matched-threshold original; must also not reduce currently deployed accuracy or high recall. No tolerance except 1e-12 numerical comparisons.',ensemble_search='Current LR/XGB/LSTM weights, .5 XGB+.5 LSTM, and .5 LR+.5 XGB on provisionally qualifying members; original members otherwise. Threshold on validation only. Compare standalone explicitly.',test='Freeze all candidates, baseline cutoffs, eligibility and blend before one test pass. Existing internal holdout reused. Never choose another candidate using test.',promotion='Must be validation eligible, and final test accuracy/macro-F1/QWK/sometimes recall no worse than matched original, high recall >=max(.60,actual original recall), accuracy >= actual original, paired accuracy improvement CI lower bound >0. All runtime checks required before activation. A failed gate retains existing version without further test tuning.',confidence='1000 paired stratified bootstrap draws of prediction pairs for accuracy/high-recall deltas; descriptive intervals, not correction for repeated historical experiments.',baseline='Always rarely historical holdout accuracy65.78% and high recall0%; also report each partition majority baseline.',ordinal='Previous ordinal artifacts/results retained as historical comparison, not trained or selected here.',search_scope='Bounded pilot, not exhaustive. Architecture widths/layers and scaling remain fixed for compatible inference. No oversampling or probability calibration. Hyperparameters explored as explicit bundled configurations; effects are not isolated causal estimates.')
    save(OUT/'protocol.json',protocol);save(OUT/'script_sha256.json',{'sha256':sha(__file__)})
    print('AUDIT',json.dumps(a),flush=True);protect()

def load_train():
    # Read only train/validation into downstream fitting. Test predictions are
    # never loaded or used until the guarded test phase.
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv')
    return df[df.split=='train'].copy(),df[df.split=='validation'].copy()

def validation_originals(va):
    path=OUT/'original_validation.npz'
    if path.exists():return dict(np.load(path))
    x=joblib.load(MODELS/'preprocessor.joblib').transform(va[FEATURES]).astype('float32')
    raw={name:original(i,x) for i,name in enumerate(NAMES)};np.savez_compressed(path,**raw);return raw

def search():
    protect();protocol=read(OUT/'protocol.json');tr,va=load_train();raw=validation_originals(va);selection=read(MODELS/'selection.json');y=va.target.to_numpy()
    floors={name:max(.6,metric(y,raw[name]*selection[name]['multipliers'])['recall'][2]) for name in NAMES}
    save(OUT/'recall_floors.json',floors)
    subset=tr[tr.profile_group.str[:8].map(lambda s:int(s,16)%4==0)]
    splits=list(StratifiedGroupKFold(2,shuffle=True,random_state=SEED).split(subset,subset.target,subset.profile_group))
    assignment=[]
    for fold,(trainidx,validx) in enumerate(splits):
        outer=subset.iloc[trainidx];evalframe=subset.iloc[validx];rest,cal=partition(outer,10,SEED+fold);fitframe,dev=partition(rest,9,SEED+fold+20)
        for role,frame in [('fit',fitframe),('early_stop',dev),('calibrate',cal),('score',evalframe)]:
            assignment.extend(dict(respondent_id=r,fold=fold,role=role) for r in frame.respondent_id)
        for i,name in enumerate(NAMES):
            for c in protocol['configs'][name]:
                path=OUT/'cv'/f'{i}_{c["id"]}_{fold}.json'
                if path.exists():continue
                print('FIT',name,'config',c['id'],'fold',fold,flush=True)
                pre,m,info=fit(i,c,fitframe,dev,SEED+fold,'screen')
                cp=prob(m,pre.transform(cal[FEATURES]).astype('float32'));p=prob(m,pre.transform(evalframe[FEATURES]).astype('float32'));t=cutoff(cal.target.to_numpy(),cp,floors[name])
                record=dict(model=name,fold=fold,config_id=c['id'],threshold=t,floor=floors[name],raw=metric(evalframe.target.to_numpy(),p),constrained=metric(evalframe.target.to_numpy(),p,decide(p,t)),fit=info,majority_accuracy=float(evalframe.target.value_counts(normalize=True).max()),historical_rarely_baseline_accuracy=.6578)
                save(path,record);print('CV',name,c['id'],fold,json.dumps(record['constrained']),flush=True);del m,pre;gc.collect()
    pd.DataFrame(assignment).to_csv(OUT/'cv_split_ids.csv',index=False)
    selected={}
    for name in NAMES:
        candidates=[]
        for c in protocol['configs'][name]:
            rr=[read(OUT/'cv'/f'{NAMES.index(name)}_{c["id"]}_{fold}.json') for fold in range(2)]
            avg={k:float(np.mean([r['constrained'][k] for r in rr])) for k in ['accuracy','macro_f1','qwk']}
            avg['high_recall']=float(np.mean([r['constrained']['recall'][2] for r in rr]))
            # Fixed tolerance for finite small calibration samples only; final
            # validation eligibility enforces the actual floor with no tolerance.
            violation=max(0,floors[name]-.02-avg['high_recall'])
            candidates.append(dict(config=c,mean=avg,recall_violation=violation))
        selected[name]=max(candidates,key=lambda c:(-c['recall_violation'],c['mean']['accuracy'],c['mean']['macro_f1'],c['mean']['qwk']))
    save(OUT/'cv_selected.json',selected);protect()

def qualifies(m,b,actual,floor):
    return bool(m['accuracy']>b['accuracy']+1e-12 and all(m[k]+1e-12>=b[k] for k in ['macro_f1','qwk']) and m['recall'][1]+1e-12>=b['recall'][1] and m['recall'][2]+1e-12>=floor and m['accuracy']+1e-12>=actual['accuracy'])

def ensemble(raw,weights):
    p=sum(w*raw[n] for n,w in weights.items());return p/p.sum(1,keepdims=True)

def select():
    if (OUT/'frozen_selection.json').exists():raise RuntimeError('Selection is already frozen.')
    protect();tr,va=load_train();vy=va.target.to_numpy();base=validation_originals(va);selections=read(MODELS/'selection.json');floors=read(OUT/'recall_floors.json');chosen=read(OUT/'cv_selected.json')
    fitframe,dev=partition(tr,10,SEED+100);pd.concat([fitframe[['respondent_id']].assign(role='fit'),dev[['respondent_id']].assign(role='early_stop')]).to_csv(OUT/'final_train_ids.csv',index=False)
    result={};newraw={};modelhashes={}
    for i,name in enumerate(NAMES):
        c=chosen[name]['config'];folder=OUT/'candidates'/str(i);folder.mkdir(parents=True,exist_ok=True);seed_reports=[]
        for seed in ([SEED] if i<3 else [SEED,SEED+1]):
            prefix=folder/str(seed);meta=Path(str(prefix)+'.json');probs=Path(str(prefix)+'.npz');checkpoint=Path(str(prefix)+('.joblib' if i<3 else '.pt'));prep=Path(str(prefix)+'_preprocessor.joblib')
            if not meta.exists():
                print('FINAL FIT',name,seed,flush=True);pre,m,info=fit(i,c,fitframe,dev,seed,'final');start=time.monotonic();p=prob(m,pre.transform(va[FEATURES]).astype('float32'));info['validation_inference_seconds']=time.monotonic()-start
                if i<3:joblib.dump(m,checkpoint)
                else:torch.save(m.state_dict(),checkpoint)
                joblib.dump(pre,prep);np.savez_compressed(probs,p=p);info['checkpoint_bytes']=checkpoint.stat().st_size;save(meta,info);del m,pre;gc.collect()
            p=np.load(probs)['p'];t=cutoff(vy,p,floors[name]);s=metric(vy,p,decide(p,t));seed_reports.append(dict(seed=seed,raw=metric(vy,p),constrained=s,threshold=t,fit=read(meta)))
            if seed==SEED:newraw[name]=p
            for f in [checkpoint,prep,meta,probs]:modelhashes[str(f.relative_to(OUT))]=sha(f)
        original_t=cutoff(vy,base[name],floors[name]);original_m=metric(vy,base[name],decide(base[name],original_t));actual=metric(vy,base[name]*selections[name]['multipliers'])
        first=seed_reports[0];eligible=qualifies(first['constrained'],original_m,actual,floors[name])
        result[name]=dict(config=c,seed=SEED,threshold=first['threshold'],baseline_threshold=original_t,floor=floors[name],original_raw=metric(vy,base[name]),original_actual=actual,original_matched=original_m,candidate=first,seed_reports=seed_reports,validation_eligible=eligible)
        print('VALIDATION',name,'eligible',eligible,'old accuracy',actual['accuracy'],'candidate',first['constrained']['accuracy'],flush=True)
    policy=read(MODELS/'ensemble/policy.json');oldp=ensemble(base,policy['weights']);actual=metric(vy,oldp,decide(oldp,policy['high_threshold']));efloor=max(.6,actual['recall'][2]);bt=cutoff(vy,oldp,efloor);bm=metric(vy,oldp,decide(oldp,bt))
    pool={n:newraw[n] if result[n]['validation_eligible'] else base[n] for n in NAMES}
    etrials=[]
    for weights in [policy['weights'],{'XGBoost':.5,'LSTM':.5},{'Logistic Regression':.5,'XGBoost':.5}]:
        p=ensemble(pool,weights);t=cutoff(vy,p,efloor);s=metric(vy,p,decide(p,t));etrials.append(dict(weights=weights,threshold=t,metrics=s,eligible=qualifies(s,bm,actual,efloor)))
    ebest=max(etrials,key=lambda r:(r['eligible'],r['metrics']['accuracy'],r['metrics']['macro_f1']))
    frozen=dict(models=result,ensemble=dict(selected=ebest,trials=etrials,floor=efloor,baseline_threshold=bt,original_actual=actual,original_matched=bm,member_sources={n:'candidate' if result[n]['validation_eligible'] else 'original' for n in NAMES}),candidate_hashes=modelhashes,protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),majority_validation_accuracy=float(va.target.value_counts(normalize=True).max()),historical_rarely_baseline_accuracy=.6578)
    save(OUT/'frozen_selection.json',frozen);protect()

def interval(y,old,new):
    rng=np.random.default_rng(SEED);groups=[np.flatnonzero(y==c) for c in range(3)];d=[];h=[]
    for _ in range(1000):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);d.append(float(np.mean(new[idx]==y[idx])-np.mean(old[idx]==y[idx])));hi=idx[y[idx]==2];h.append(float(np.mean(new[hi]==2)-np.mean(old[hi]==2)))
    return dict(accuracy_delta_95ci=np.quantile(d,[.025,.975]).tolist(),high_recall_delta_95ci=np.quantile(h,[.025,.975]).tolist())

def test():
    protect();frozen=read(OUT/'frozen_selection.json')
    assert frozen['protocol_sha256']==sha(OUT/'protocol.json') and frozen['script_sha256']==sha(__file__)
    for f,h in frozen['candidate_hashes'].items():assert sha(OUT/f)==h
    guard=OUT/'TEST_STARTED.json'
    with guard.open('x') as f:json.dump({'frozen_sha256':sha(OUT/'frozen_selection.json'),'note':'One-shot reused internal holdout. Do not retune.'},f)
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');te=df[df.split=='test'];y=te.target.to_numpy();x=joblib.load(MODELS/'preprocessor.joblib').transform(te[FEATURES]).astype('float32');base={};new={};report={};selections=read(MODELS/'selection.json')
    for i,name in enumerate(NAMES):
        b=original(i,x);base[name]=b;f=frozen['models'][name];prefix=OUT/'candidates'/str(i)/str(SEED);pre=joblib.load(str(prefix)+'_preprocessor.joblib');tx=pre.transform(te[FEATURES]).astype('float32');start=time.monotonic()
        if i<3:m=joblib.load(str(prefix)+'.joblib')
        else:
            m=CLASSES[i-3](input_dim=tx.shape[1]);m.load_state_dict(torch.load(str(prefix)+'.pt',map_location='cpu',weights_only=True))
        p=prob(m,tx);new[name]=p;seconds=time.monotonic()-start;pred=decide(p,f['threshold']);op=decide(b,f['baseline_threshold']);actual=metric(y,b*selections[name]['multipliers']);bm=metric(y,b,op);nm=metric(y,p,pred);ci=interval(y,op,pred)
        passed=bool(f['validation_eligible'] and qualifies(nm,bm,actual,max(.6,actual['recall'][2])) and ci['accuracy_delta_95ci'][0]>0)
        report[name]=dict(original_raw=metric(y,b),original_actual=actual,original_matched=bm,candidate_raw=metric(y,p),candidate_constrained=nm,paired_vs_matched=ci,validation_eligible=f['validation_eligible'],performance_gate_passed=passed,test_load_and_inference_seconds=seconds)
        print('TEST',name,json.dumps(report[name]),flush=True);del m,pre;gc.collect()
    policy=read(MODELS/'ensemble/policy.json');oldp=ensemble(base,policy['weights']);ef=frozen['ensemble'];pool={n:new[n] if ef['member_sources'][n]=='candidate' else base[n] for n in NAMES};p=ensemble(pool,ef['selected']['weights']);pred=decide(p,ef['selected']['threshold']);op=decide(oldp,ef['baseline_threshold']);actual=metric(y,oldp,decide(oldp,policy['high_threshold']));bm=metric(y,oldp,op);nm=metric(y,p,pred);ci=interval(y,op,pred)
    passed=bool(ef['selected']['eligible'] and qualifies(nm,bm,actual,max(.6,actual['recall'][2])) and ci['accuracy_delta_95ci'][0]>0)
    report['Ensemble']=dict(original_actual=actual,original_matched=bm,candidate_raw=metric(y,p),candidate_constrained=nm,paired_vs_matched=ci,validation_eligible=ef['selected']['eligible'],performance_gate_passed=passed)
    rarely=np.tile([1.,0.,0.],(len(y),1));major=int(np.bincount(y).argmax());mp=np.eye(3)[np.repeat(major,len(y))]
    report['baselines']=dict(historical_rarely_accuracy=.6578,always_rarely=metric(y,rarely),majority_class=major,majority=metric(y,mp),rows=len(y),evaluation='Previously reused internal holdout, not independent validation.')
    save(OUT/'test_report.json',report);np.savez_compressed(OUT/'test_predictions.npz',target=y,respondent_id=te.respondent_id.to_numpy(dtype=str),**{'original_'+str(i):base[n] for i,n in enumerate(NAMES)},**{'candidate_'+str(i):new[n] for i,n in enumerate(NAMES)})
    protect();print('TEST COMPLETE. No deployment performed.',flush=True)

def report():
    f=read(OUT/'frozen_selection.json');r=read(OUT/'test_report.json');rows=[]
    for name in NAMES+['Ensemble']:
        a=r[name]['original_actual'];b=r[name]['original_matched'];c=r[name]['candidate_constrained']
        rows.append(dict(model=name,old_accuracy=100*a['accuracy'],matched_old_accuracy=100*b['accuracy'],candidate_accuracy=100*c['accuracy'],old_high_recall=100*a['recall'][2],matched_old_high_recall=100*b['recall'][2],candidate_high_recall=100*c['recall'][2],candidate_macro_f1=c['macro_f1'],candidate_qwk=c['qwk'],candidate_middle_recall=100*c['recall'][1],candidate_auc=c['macro_auc'],validation_eligible=r[name]['validation_eligible'],performance_gate_passed=r[name]['performance_gate_passed']))
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    lines=['# Six-model hyperparameter tuning','', 'No active model has been changed by this experiment. Promotion requires separate runtime verification.','', 'Always rarely baseline: 65.78% accuracy and 0% high recall on the reused holdout.','', '| Model | Deployed accuracy | Matched original accuracy | Candidate accuracy | Deployed high recall | Candidate high recall | Performance gate |','|---|---:|---:|---:|---:|---:|---|']
    for row in rows:lines.append('| '+row['model']+' | '+' | '.join(f'{row[k]:.2f}%' for k in ['old_accuracy','matched_old_accuracy','candidate_accuracy','old_high_recall','candidate_high_recall'])+' | '+str(row['performance_gate_passed'])+' |')
    lines+=['','## Interpretation','Candidates were ranked using training-only grouped cross-validation; final thresholds and eligibility used validation. Final test was opened once after freezing. It is a historically reused holdout, not independent evidence.','Strict safeguards require non-degradation in middle recall, macro-F1 and QWK as well as accuracy and high recall. A candidate can improve accuracy and still be rejected.','No model widths/layers or feature encoding were changed in this bounded pilot. Neural timing caps can stop training before 14 epochs; see the recorded histories. Results do not establish that further tuning cannot help.','Secondary neural seeds measure instability; they are never selected because they look better.','The survey is cross-sectional. An LSTM processes a fixed feature ordering, not patient history or genuine temporal signals.','','## Reproduce and inspect','Use tools/tune_six_models.py with phases audit, search, select, test, report and --project/--output. Audit refuses existing directories; test refuses a second pass. Do not create new experiments simply to retune against test.','protocol.json: predeclared settings; audit.json: data checks; cv/: every CV result; cv_split_ids.csv/final_train_ids.csv: group-safe roles; candidates/: separate models and preprocessors; frozen_selection.json: frozen choices; test_report.json: detailed comparisons and paired intervals; comparison.csv: summary; backup/: protected originals.','','## Rollback','The active models and source are unchanged. There is nothing to roll back unless a separate installation step is performed. Backups and original hashes provide recovery and verification.','','## Viva explanation','Hyperparameters are settings chosen before fitting, such as tree depth, regularization and learning rate. We tried a bounded set using grouped training folds, tuned stress cutoffs on separate calibration data, selected on validation, and evaluated once on the held-out respondents. Better accuracy is not accepted if it hides more high-stress cases.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    # Readable, safe notebook: inspect real outputs; do not silently retrain or
    # reopen the test. Execution outputs are generated by executing its cells.
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))
    md('# Stress prediction: preprocessing and hyperparameter tuning\nStudy notebook for the actual bounded experiment. All outputs are loaded from saved experiment evidence. No training or test evaluation runs automatically.')
    code("from pathlib import Path\nimport json\nimport pandas as pd\nE = Path.cwd()\nif not (E/'protocol.json').exists():\n    raise RuntimeError('Open this notebook with the experiment folder as the working directory.')\ndef load(name): return json.loads((E/name).read_text())\nprint('Experiment:', E.name)")
    md('## 1. Dataset and target\nOne row represents one respondent. Different respondents can legitimately share all predictor values. Stress frequency is ordered: rarely/never < sometimes < usually/always. Complete-case filtering and unweighted sampling limit generalization.')
    code("audit=load('audit.json')\nprint(json.dumps(audit,indent=2))")
    md('## 2. Preprocessing\nRaw invalid/refused/dont-know codes were excluded; physical unhealthy days 88 becomes 0; BMI is divided by 100. Four numeric features use StandardScaler and 15 categorical features use fixed-category OneHotEncoder, producing 69 features. IDs, raw stress and target are not inputs. Every CV preprocessing object fits only its fitting partition; early-stop, calibration and scoring partitions only transform. The saved original scaler was independently checked against training-only statistics.')
    code("protocol=load('protocol.json')\nprint('Features:', protocol['features'])\nprint('Original preprocessing matches training:',audit['train_preprocessor_matches'])\nroles=pd.read_csv(E/'cv_split_ids.csv')\nprint(roles.groupby(['fold','role']).size().to_string())\nassert not roles.duplicated(['fold','respondent_id']).any()")
    md('## 3. Tuning without leakage\nTraining-only outer folds score each configuration. Inner development data choose early-stopping epochs; a separate inner calibration partition chooses the cutoff. Validation is reserved for final thresholds and eligibility. The test was used historically and is explicitly not pristine.')
    code("print(json.dumps({k:protocol[k] for k in ['screen','final','selection','promotion','search_scope']},indent=2))")
    md('## 4. Search spaces and selected settings\nThree configurations per model, including the original settings. This is a bounded pilot, not an exhaustive search. Existing architecture sizes are fixed; neural learning rate, regularization, batch size, weighting, dropout and scheduler are investigated.')
    code("print(json.dumps(load('cv_selected.json'),indent=2))")
    md('## 5. Neural stability\nThe first seed was predetermined as the candidate. The second seed is reported for stability, not cherry-picked. These are cross-sectional features, not temporal patient measurements.')
    code("frozen=load('frozen_selection.json')\nfor name, item in frozen['models'].items():\n    print(name,[(s['seed'],round(s['constrained']['accuracy'],4),len(s['fit']['history']) if isinstance(s['fit']['history'],list) else 'tree') for s in item['seed_reports']])")
    md('## 6. One frozen final evaluation\nOriginal deployed decisions and matched-threshold original decisions are both shown. Always rarely: 65.78% historical holdout accuracy, 0% high recall. See test_report.json for all precision/recall/F1 values, confusion matrices, paired intervals and AUC. Intervals are descriptive and do not correct for historical test reuse.')
    code("table=pd.read_csv(E/'comparison.csv')\nprint(table.to_string(index=False))\nresults=load('test_report.json')\nprint(json.dumps(results['baselines'],indent=2))")
    code("for name,item in results.items():\n    if name=='baselines': continue\n    print(name,'paired intervals:',item['paired_vs_matched'],'performance gate:',item['performance_gate_passed'])")
    md('## 7. Preservation and limitations\nThe script never installs candidate models. Any accepted candidate still needs inference parity, input validation and memory-limit checks before activation. Failed candidates remain experiments. Exact backups and hashes are preserved. A higher validation recall does not guarantee fewer misses on future respondents.')
    code("print('Protected files:',len(load('original_hashes.json')))\nprint('Frozen selection recorded:',load('TEST_STARTED.json')['frozen_sha256'])\nprint('Read README.md for interpretation and rollback.')")
    save(OUT/'Hyperparameter_Tuning.ipynb',dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':platform.python_version()}},nbformat=4,nbformat_minor=5))
    protect();print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['audit','search','select','test','report']);parser.add_argument('--project',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    ROOT=Path(args.project).resolve();OUT=Path(args.output).resolve();MODELS=ROOT/'models/current_models';sys.path.insert(0,str(ROOT))
    from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];SCHEMA=read(MODELS/'schema.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical'];torch.set_num_threads(4)
    globals()[args.phase]()
