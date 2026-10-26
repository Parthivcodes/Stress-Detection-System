"""Features -> 50 Optuna trials -> disjoint calibration -> frozen evaluation.
No active data/models are written. Independent external validation remains needed.
"""
import argparse,copy,gc,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from feature_pipeline import FeatureEncoder,temperature
from tune_six_models import sha,save,read,metric,decide,partition,prob,interval

SEED=202610231

def verify():
    for path,h in read(OUT/'original_hashes.json').items():assert sha(ROOT/path)==h,path

def data():
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv')
    return df[df.split=='train'].copy(),df[df.split=='validation'].copy()

def reserve(tr):
    search,cal=partition(tr,10,SEED)
    return search,cal

def baseparams():
    return dict(n_estimators=350,max_depth=5,learning_rate=.04,min_child_weight=12,reg_lambda=5,reg_alpha=0,subsample=.85,colsample_bytree=.85,gamma=0,weight_power=.5)

def fit(params,x,y,dx,dy):
    p=copy.deepcopy(params);power=p.pop('weight_power');weights=(len(y)/(3*np.bincount(y,minlength=3)))**power
    model=XGBClassifier(**p,n_jobs=4,random_state=SEED,eval_metric='mlogloss',tree_method='hist',early_stopping_rounds=30)
    model.fit(x,y,sample_weight=weights[y],eval_set=[(dx,dy)],verbose=False)
    return model

def folds(search,mode):
    result=[];roles=[]
    for k,(a,b) in enumerate(StratifiedGroupKFold(2,shuffle=True,random_state=SEED+1).split(search,search.target,search.profile_group)):
        fitframe,dev=partition(search.iloc[a],10,SEED+k+10);score=search.iloc[b]
        assert not set(fitframe.profile_group)&set(score.profile_group)
        enc=FeatureEncoder(SCHEMA,mode).fit(fitframe)
        result.append((enc.transform(fitframe),fitframe.target.to_numpy(copy=True),enc.transform(dev),dev.target.to_numpy(copy=True),enc.transform(score),score.target.to_numpy(copy=True)))
        for role,frame in [('fit',fitframe),('early_stop',dev),('score',score)]:roles.extend(dict(respondent_id=r,fold=k,role=role) for r in frame.respondent_id)
    pd.DataFrame(roles).to_csv(OUT/'training_cv_roles.csv',index=False)
    return result

def score_params(params,cache):
    rows=[];start=time.monotonic()
    for x,y,dx,dy,sx,sy in cache:
        model=fit(params,x,y,dx,dy);p=model.predict_proba(sx)
        rows.append(dict(log_loss=float(log_loss(sy,p)),best_iteration=model.best_iteration,raw=metric(sy,p)))
        del model
    return dict(params=params,folds=rows,mean_log_loss=float(np.mean([r['log_loss'] for r in rows])),seconds=time.monotonic()-start)

def initialize():
    if OUT.exists():raise RuntimeError('Choose a new experiment directory; no overwrite.')
    OUT.mkdir(parents=True)
    protected=list(MODELS.rglob('*'))+[ROOT/'data/expanded_survey_dataset/respondents.csv',ROOT/'models/active_model.json',ROOT/'src/brfss_inference.py',ROOT/'src/models.py',ROOT/'server.py',ROOT/'deploy/backend.py',ROOT/'frontend/prediction.js']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()};save(OUT/'original_hashes.json',hashes)
    for name in hashes:
        dest=OUT/'backup'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    tr,va=data();assert tr.respondent_id.is_unique and va.respondent_id.is_unique
    assert not set(tr.profile_group)&set(va.profile_group)
    search,cal=reserve(tr)
    pd.concat([search[['respondent_id']].assign(role='search_and_fit'),cal[['respondent_id']].assign(role='calibration_only')]).to_csv(OUT/'training_roles.csv',index=False)
    protocol=dict(seed=SEED,feature_sets=['original','ordered','engineered'],feature_selection='Compare identical baseline XGBoost settings on two grouped training CV folds; choose lowest mean unweighted log-loss. Original control retained; no forced engineered-feature promotion.',features='Numeric originals retained. Six true ordered scales as numeric, other categories one-hot with binary reference dropped. Engineered: seven-factor social risk count, unhealthy days >=14 and log1p, three sleep bands, BMI 1st/99th percentile clipped copy. All fit-dependent values learn within fold.',risk_rule='job_loss/bill_difficulty/utility_threat/transport_barrier ==1; food_insecurity/isolation <=3; support>=4. Count is an engineered predictor, not a validated clinical score.',source_codebook='data/baseline_survey_dataset/raw/codebook/USCODE22_LLCP_102523.HTML',trials=50,search='Optuna TPESampler seeded; sequential, no pruning, two grouped training folds; inner 10% early-stop split per fold; full training search partition, not a small sample.',search_ranges=dict(learning_rate=[.015,.15],max_depth=[2,7],min_child_weight=[2,80],subsample=[.65,1],colsample_bytree=[.6,1],reg_alpha=[1e-5,10],reg_lambda=[.1,50],gamma=[0,2],weight_power=[0,1],n_estimators=700),calibration='Reserve grouped 10% of original training before all feature selection/tuning. Final model fit excludes calibration data. Split calibration reserve into fit/check halves; fit positive scalar temperature on first half, enable only if independent calibration-check logloss improves and Brier does not worsen. Fixed accepted temperature; no refit on check.',final_fit='90% of search partition fit, 10% internal early-stop. Calibration reserve never used in fitting or hyperparameter search.',operating_point='On existing validation choose highest accuracy with high recall >=max(.60,current original XGB validation recall); no decreases in macro-F1, QWK or middle recall vs original matched-cutoff baseline and deployed baseline. Threshold chosen separately from calibration and tuning. Ensemble existing weights with candidate XGB and one separately selected cutoff, same safeguards vs current ensemble.',promotion='Atomic XGB+ensemble update only if both validation eligible, both final test meet non-degradation safeguards vs current and matched originals, each high recall >=max(.60,current respective test recall), paired accuracy delta CI lower>0, plus runtime512MiB/API checks. Otherwise retain current versions. No alternative chosen using test.',evaluation='Existing internal test is historically reused, NOT independent validation. Freeze configuration, model/encoder/calibration hashes and cutoffs before one final evaluation. No test tuning.',baseline='Always rarely: historical holdout65.78% accuracy and 0% high recall; report actual partition baseline too.',cpu_threads=4,paid_services=False)
    save(OUT/'protocol.json',protocol);save(OUT/'source_hashes.json',{Path(__file__).name:sha(__file__),'feature_pipeline.py':sha(Path(__file__).with_name('feature_pipeline.py'))})
    print('PROTOCOL FROZEN',len(search),'search rows;',len(cal),'reserved calibration rows',flush=True);verify()

def search_phase():
    verify();tr,_=data();search,_=reserve(tr);features={}
    for mode in ['original','ordered','engineered']:
        path=OUT/f'features_{mode}.json'
        if not path.exists():
            print('FEATURE COMPARISON',mode,flush=True);cache=folds(search,mode);r=score_params(baseparams(),cache);r['dimensions']=cache[0][0].shape[1];save(path,r);del cache;gc.collect()
        features[mode]=read(path)
    mode=min(features,key=lambda m:features[m]['mean_log_loss']);save(OUT/'feature_selection.json',{'selected':mode,'comparisons':features})
    print('SELECTED FEATURES',mode,flush=True)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # JSON trial history resumes a deterministic sampler by replaying completed
    # trials into a fresh study. No changes to any historical experiment.
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED),storage='sqlite:///'+str(OUT/'optuna.sqlite3'),study_name='xgb_features',load_if_exists=True)
    if not study.trials:
        p=baseparams();p.pop('n_estimators');p['reg_alpha']=1e-5;study.enqueue_trial(p)
    cache=folds(search,mode)
    def objective(t):
        p=dict(n_estimators=700,learning_rate=t.suggest_float('learning_rate',.015,.15,log=True),max_depth=t.suggest_int('max_depth',2,7),min_child_weight=t.suggest_float('min_child_weight',2,80,log=True),subsample=t.suggest_float('subsample',.65,1),colsample_bytree=t.suggest_float('colsample_bytree',.6,1),reg_alpha=t.suggest_float('reg_alpha',1e-5,10,log=True),reg_lambda=t.suggest_float('reg_lambda',.1,50,log=True),gamma=t.suggest_float('gamma',0,2),weight_power=t.suggest_float('weight_power',0,1))
        r=score_params(p,cache);r['trial']=t.number;save(OUT/'trials'/f'{t.number:03d}.json',r)
        print('TRIAL',t.number+1,'/50 logloss',round(r['mean_log_loss'],6),'seconds',round(r['seconds'],1),flush=True);return r['mean_log_loss']
    remaining=50-len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
    if remaining>0:study.optimize(objective,n_trials=remaining)
    save(OUT/'best_trial.json',read(OUT/'trials'/f'{study.best_trial.number:03d}.json'));study.trials_dataframe().to_csv(OUT/'optuna_trials.csv',index=False)
    save(OUT/'optuna_version.json',{'optuna':optuna.__version__});verify()

def original_probs(frame):
    x=joblib.load(MODELS/'preprocessor.joblib').transform(frame[SCHEMA['numeric']+SCHEMA['categorical']]).astype('float32');r={}
    for name,file in [('Logistic Regression','logistic_regression.joblib'),('XGBoost','xgboost.joblib')]:
        m=joblib.load(MODELS/file);r[name]=np.asarray(m.predict_proba(x),dtype=float);del m
    sys.path.insert(0,str(ROOT));from src.models import LSTMNet
    m=LSTMNet(input_dim=x.shape[1]);m.load_state_dict(torch.load(MODELS/'lstm_net.pt',map_location='cpu',weights_only=True));r['LSTM']=np.asarray(prob(m,x),dtype=float)
    return r

def blend(raw,weights):
    p=sum(w*np.asarray(raw[n],dtype=float) for n,w in weights.items());return p/p.sum(1,keepdims=True)

def brier(y,p):return float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))

def sufficient(m,b,floor,strict=True):
    return bool((m['accuracy']>b['accuracy']+1e-12 if strict else m['accuracy']+1e-12>=b['accuracy']) and m['recall'][2]+1e-12>=floor and m['macro_f1']+1e-12>=b['macro_f1'] and m['qwk']+1e-12>=b['qwk'] and m['recall'][1]+1e-12>=b['recall'][1])

def operating(y,p,floor,reference=None):
    # Exact O(n log n) accuracy across all distinct high-probability cutoffs.
    lower=p[:,:2].argmax(1);order=np.argsort(-p[:,2],kind='stable');sortedp=p[order,2];yy=y[order]
    correct=int(np.sum(lower==y));delta=(yy==2).astype(int)-(lower[order]==yy).astype(int)
    acc=(correct+np.cumsum(delta))/len(y);rec=np.cumsum(yy==2)/np.sum(y==2)
    end=np.r_[sortedp[:-1]!=sortedp[1:],True];indices=np.flatnonzero(end & (rec+1e-12>=floor));rank=indices[np.lexsort((-sortedp[indices],-acc[indices]))]
    # Compute all confusion matrices cumulatively; do not recompute AUC for
    # thousands of thresholds. AUC does not depend on the chosen decision cutoff.
    initial=np.bincount(3*y+lower,minlength=9)
    changes=np.zeros((len(y),9),dtype=int);ix=np.arange(len(y))
    np.add.at(changes,(ix,3*yy+lower[order]),-1)
    np.add.at(changes,(ix,3*yy+2),1)
    cms=(initial+np.cumsum(changes,axis=0)).reshape(-1,3,3)
    diagonal=np.diagonal(cms,axis1=1,axis2=2);rows=cms.sum(2);cols=cms.sum(1)
    rr=diagonal/np.maximum(rows,1);pp=diagonal/np.maximum(cols,1)
    f1=(2*rr*pp/np.maximum(rr+pp,1e-15)).mean(1)
    w=(np.arange(3)[:,None]-np.arange(3)[None,:])**2
    qwk=1-(cms*w).sum((1,2))*len(y)/np.maximum((rows[:,:,None]*cols[:,None,:]*w).sum((1,2)),1)
    allowed=np.ones(len(rank),dtype=bool)
    if reference is not None:
        allowed=(acc[rank]>reference['accuracy']+1e-12)&(f1[rank]+1e-12>=reference['macro_f1'])&(qwk[rank]+1e-12>=reference['qwk'])&(rr[rank,1]+1e-12>=reference['recall'][1])
    eligible=bool(allowed.any());j=int(rank[np.flatnonzero(allowed)[0]]) if eligible else int(rank[0])
    t=float(sortedp[j]);m=metric(y,p,decide(p,t))
    return dict(threshold=t,metrics=m,eligible=eligible)

def curve(y,p):
    rows=[]
    for t in np.linspace(0,1,201):
        pred=decide(p,t);rows.append({'threshold':float(t),'accuracy':float(np.mean(pred==y)),'high_recall':float(np.mean(pred[y==2]==2))})
    return rows

def select_phase():
    if (OUT/'frozen_selection.json').exists():raise RuntimeError('Already frozen.')
    verify();tr,va=data();search,cal=reserve(tr);fitframe,dev=partition(search,10,SEED+100);mode=read(OUT/'feature_selection.json')['selected'];params=read(OUT/'best_trial.json')['params']
    enc=FeatureEncoder(SCHEMA,mode).fit(fitframe);start=time.monotonic();m=fit(params,enc.transform(fitframe),fitframe.target.to_numpy(copy=True),enc.transform(dev),dev.target.to_numpy(copy=True))
    joblib.dump(m,OUT/'candidate.joblib');save(OUT/'encoder.json',enc.config());save(OUT/'fit.json',dict(seconds=time.monotonic()-start,best_iteration=m.best_iteration,fit_rows=len(fitframe),dev_rows=len(dev),calibration_rows=len(cal),params=params))
    cf,cc=partition(cal,2,SEED+200);cp=np.asarray(m.predict_proba(enc.transform(cf)),dtype=float);qp=np.asarray(m.predict_proba(enc.transform(cc)),dtype=float);cy=cf.target.to_numpy();qy=cc.target.to_numpy()
    opt=minimize_scalar(lambda logt:log_loss(cy,temperature(cp,np.exp(logt))),bounds=(-2,2),method='bounded');t=float(np.exp(opt.x));qcal=temperature(qp,t)
    enabled=log_loss(qy,qcal)<log_loss(qy,qp) and brier(qy,qcal)<=brier(qy,qp)
    calibration=dict(temperature=t if enabled else 1.,fitted_temperature=t,enabled=bool(enabled),check_raw_logloss=float(log_loss(qy,qp)),check_calibrated_logloss=float(log_loss(qy,qcal)),check_raw_brier=brier(qy,qp),check_calibrated_brier=brier(qy,qcal),fit_respondents=cf.respondent_id.tolist(),check_respondents=cc.respondent_id.tolist())
    save(OUT/'calibration.json',calibration)
    y=va.target.to_numpy();raw=original_probs(va);praw=np.asarray(m.predict_proba(enc.transform(va)),dtype=float);p=temperature(praw,calibration['temperature']);np.savez_compressed(OUT/'validation_probs.npz',candidate_raw=praw,candidate_calibrated=p,**raw)
    mult=read(MODELS/'selection.json')['XGBoost']['multipliers'];actual=metric(y,raw['XGBoost']*mult);floor=max(.6,actual['recall'][2]);base=operating(y,raw['XGBoost'],floor)
    # Require preservation against both original policies for middle-class/F1/QWK.
    ref=copy.deepcopy(base['metrics'])
    for k in ['accuracy','macro_f1','qwk']:ref[k]=max(ref[k],actual[k])
    ref['recall'][1]=max(ref['recall'][1],actual['recall'][1]);choice=operating(y,p,floor,ref)
    policy=read(MODELS/'ensemble/policy.json');bp=blend(raw,policy['weights']);ba=metric(y,bp,decide(bp,policy['high_threshold']));ef=max(.6,ba['recall'][2]);eb=operating(y,bp,ef);newraw={**raw,'XGBoost':p};ep=blend(newraw,policy['weights']);eref=copy.deepcopy(eb['metrics'])
    for k in ['accuracy','macro_f1','qwk']:eref[k]=max(eref[k],ba[k])
    eref['recall'][1]=max(eref['recall'][1],ba['recall'][1]);ec=operating(y,ep,ef,eref)
    pd.DataFrame(curve(y,p)).to_csv(OUT/'accuracy_recall_curve.csv',index=False)
    reliability=[]
    for label in range(3):
        for lo in np.arange(0,1,.1):
            for version,pp in [('raw',praw),('calibrated',p)]:
                ix=(pp[:,label]>=lo)&(pp[:,label]<(lo+.1 if lo<.9 else 1.000001))
                if ix.any():reliability.append(dict(version=version,label=label,lower_bin=float(lo),count=int(ix.sum()),mean_predicted=float(pp[ix,label].mean()),observed=float((y[ix]==label).mean())))
    pd.DataFrame(reliability).to_csv(OUT/'validation_reliability.csv',index=False)
    frozen=dict(xgb=dict(candidate=choice,baseline=base,original_actual=actual,floor=floor,raw_argmax=metric(y,praw),calibrated_argmax=metric(y,p)),ensemble=dict(candidate=ec,baseline=eb,original_actual=ba,floor=ef,weights=policy['weights']),validation_eligible=bool(choice['eligible'] and ec['eligible']),majority_validation_accuracy=float(va.target.value_counts(normalize=True).max()),historical_rarely_accuracy=.6578,hashes={n:sha(OUT/n) for n in ['candidate.joblib','encoder.json','calibration.json','protocol.json']},source_hashes={Path(__file__).name:sha(__file__),'feature_pipeline.py':sha(Path(__file__).with_name('feature_pipeline.py'))})
    save(OUT/'frozen_selection.json',frozen);verify();print('FROZEN eligible',frozen['validation_eligible'],'XGB',choice['metrics'],'ensemble',ec['metrics'],flush=True)

def final_test():
    verify();f=read(OUT/'frozen_selection.json')
    for n,h in f['hashes'].items():assert sha(OUT/n)==h
    for n,h in f['source_hashes'].items():assert sha(Path(__file__).with_name(n))==h
    with (OUT/'TEST_STARTED.json').open('x') as stream:json.dump({'frozen_sha256':sha(OUT/'frozen_selection.json')},stream)
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');te=df[df.split=='test'];y=te.target.to_numpy();raw=original_probs(te);m=joblib.load(OUT/'candidate.joblib');enc=FeatureEncoder.restore(read(OUT/'encoder.json'));praw=m.predict_proba(enc.transform(te));p=temperature(praw,read(OUT/'calibration.json')['temperature']);policy=read(MODELS/'ensemble/policy.json');bp=blend(raw,policy['weights']);ep=blend({**raw,'XGBoost':p},policy['weights']);report={}
    for name,old,new,actual_pred in [('xgb',raw['XGBoost'],p,(raw['XGBoost']*read(MODELS/'selection.json')['XGBoost']['multipliers']).argmax(1)),('ensemble',bp,ep,decide(bp,policy['high_threshold']))]:
        cfg=f[name];op=decide(old,cfg['baseline']['threshold']);pred=decide(new,cfg['candidate']['threshold']);actual=metric(y,old,actual_pred);base=metric(y,old,op);candidate=metric(y,new,pred);ci=interval(y,op,pred)
        ok=cfg['candidate']['eligible'] and sufficient(candidate,base,max(.6,actual['recall'][2])) and sufficient(candidate,actual,max(.6,actual['recall'][2]),strict=False) and ci['accuracy_delta_95ci'][0]>0
        report[name]=dict(original_actual=actual,original_matched=base,candidate=candidate,raw_argmax=metric(y,praw if name=='xgb' else ep),calibrated_argmax=metric(y,new),paired_vs_matched=ci,performance_gate_passed=bool(ok))
    report['calibration']=dict(raw_logloss=float(log_loss(y,praw)),calibrated_logloss=float(log_loss(y,p)),raw_brier=brier(y,praw),calibrated_brier=brier(y,p))
    report['approved_for_runtime_verification']=all(report[n]['performance_gate_passed'] for n in ['xgb','ensemble']);report['always_rarely']=dict(accuracy=float(np.mean(y==0)),high_recall=0.);report['limitation']='Historically reused internal holdout; no external validation.'
    save(OUT/'test_report.json',report);np.savez_compressed(OUT/'test_predictions.npz',respondent_id=te.respondent_id.to_numpy(dtype=str),target=y,candidate_raw=praw,candidate_calibrated=p,old_xgb=raw['XGBoost'],old_ensemble=bp,candidate_ensemble=ep)
    verify();print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['init','search','select','test']);parser.add_argument('--project',required=True);parser.add_argument('--output',required=True);parser.add_argument('--optuna-path');args=parser.parse_args()
    if args.optuna_path:sys.path.insert(0,args.optuna_path)
    ROOT=Path(args.project);OUT=Path(args.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');torch.set_num_threads(4)
    {'init':initialize,'search':search_phase,'select':select_phase,'test':final_test}[args.phase]()
