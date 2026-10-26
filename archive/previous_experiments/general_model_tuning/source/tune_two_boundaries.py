"""Isolated two-boundary policies and recall-constrained grouped XGBoost search."""
import argparse,copy,gc,json,os,platform,shutil,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedGroupKFold,ParameterSampler
from scipy.stats import loguniform,uniform,randint
from xgboost import XGBClassifier
from feature_pipeline import FeatureEncoder
from tune_six_models import sha,save,read,metric,partition,interval
import tune_xgb_features as previous

SEED=202610232
def predict(p,t):
    q=p[:,1]/np.maximum(p[:,0]+p[:,1],1e-15)
    return np.where(p[:,2]>=t['high'],2,np.where(q>=t['middle'],1,0))

def cmstats(cm):
    cm=np.asarray(cm,dtype=float);n=cm.sum(axis=(-2,-1));d=np.diagonal(cm,axis1=-2,axis2=-1)
    rows=cm.sum(-1);cols=cm.sum(-2);rec=d/np.maximum(rows,1);pr=d/np.maximum(cols,1)
    f1=np.mean(2*rec*pr/np.maximum(rec+pr,1e-15),axis=-1)
    w=(np.arange(3)[:,None]-np.arange(3)[None,:])**2
    qwk=1-(cm*w).sum(axis=(-2,-1))*n/np.maximum((rows[..., :,None]*cols[...,None,:]*w).sum(axis=(-2,-1)),1)
    return d.sum(-1)/n,rec,f1,qwk

def boundaries(y,p,reference,extra_high=None):
    """Deterministic 81-quantile high grid, exact lower boundary per high cutoff.
    Thresholds selected ONLY on calibration/validation, never scoring/test folds.
    """
    q=p[:,1]/np.maximum(p[:,0]+p[:,1],1e-15)
    hs=np.unique(np.r_[0.,np.quantile(p[:,2],np.linspace(0,1,81)),1.000001,[] if extra_high is None else [extra_high]])
    order=np.argsort(q,kind='stable');sq=q[order];sy=y[order]
    # At threshold q[j], earlier elements predict rarely, current/later sometimes.
    starts=np.r_[0,np.flatnonzero(sq[1:]!=sq[:-1])+1,len(y)]
    lows=np.r_[sq[starts[:-1]],1.000001]
    floorh=max(.6,reference['recall'][2]);floorm=reference['recall'][1]
    best=None;fallback=None
    for h in hs:
        ishigh=p[order,2]>=h
        high=np.bincount(sy[ishigh],minlength=3)
        if high[2]/max(1,np.sum(y==2))+1e-12<floorh:continue
        cum=np.vstack([np.zeros(3,dtype=int),np.cumsum(np.eye(3,dtype=int)[sy]*(~ishigh)[:,None],axis=0)])
        zero=cum[starts];one=cum[-1]-zero
        cms=np.stack([zero,one,np.broadcast_to(high,zero.shape)],axis=-1)
        acc,rec,f1,kappa=cmstats(cms)
        valid=(rec[:,1]+1e-12>=floorm)&(f1+1e-12>=reference['macro_f1'])&(kappa+1e-12>=reference['qwk'])
        # Keep a diagnostic fallback only; it can never become eligible.
        violation=np.maximum(0,floorm-rec[:,1])+np.maximum(0,reference['macro_f1']-f1)+np.maximum(0,reference['qwk']-kappa)
        j=int(np.lexsort((-acc,violation))[0]);key=(float(violation[j]),-float(acc[j]))
        if fallback is None or key<fallback[0]:fallback=(key,float(h),float(lows[j]))
        if valid.any():
            ids=np.flatnonzero(valid);j=int(ids[np.argmax(acc[ids])]);key=(float(acc[j]),float(f1[j]),float(kappa[j]))
            if best is None or key>best[0]:best=(key,float(h),float(lows[j]))
    chosen=best if best else fallback
    assert chosen is not None
    t=dict(high=chosen[1],middle=chosen[2]);m=metric(y,p,predict(p,t))
    return dict(thresholds=t,metrics=m,feasible=bool(best is not None),improves=bool(best is not None and m['accuracy']>reference['accuracy']+1e-12))

def violations(m,b):
    return [max(.6,b['recall'][2])-m['recall'][2],b['recall'][1]-m['recall'][1],b['macro_f1']-m['macro_f1'],b['qwk']-m['qwk']]

def verify():
    for name,digest in read(OUT/'original_hashes.json').items():assert sha(ROOT/name)==digest,name

def frames():
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv')
    return df[df.split=='train'].copy(),df[df.split=='validation'].copy()

def original(frame):
    previous.ROOT=ROOT;previous.MODELS=MODELS;previous.SCHEMA=SCHEMA
    return previous.original_probs(frame)

def actual(y,raw):
    xp=raw['XGBoost'];ep=previous.blend(raw,POLICY['weights'])
    return dict(xgb=metric(y,xp,(xp*MULT).argmax(1)),ensemble=metric(y,ep,previous.decide(ep,POLICY['high_threshold'])))

def policies(y,raw,refs):
    return dict(xgb=boundaries(y,raw['XGBoost'],refs['xgb']),ensemble=boundaries(y,previous.blend(raw,POLICY['weights']),refs['ensemble'],POLICY['high_threshold']))

def baseline_params():
    return dict(n_estimators=350,max_depth=5,learning_rate=.04,min_child_weight=12,reg_lambda=5,reg_alpha=0,subsample=.85,colsample_bytree=.85,gamma=0,weight_power=.5,middle_weight=1.,high_weight=1.,mode='original')

def fit(params,x,y,dx,dy,baseline=False):
    params=copy.deepcopy(params);params.pop('mode');exponent=params.pop('weight_power');middle=params.pop('middle_weight');high=params.pop('high_weight')
    weights=(len(y)/(3*np.bincount(y,minlength=3)))**exponent;weights*=np.array([1.,middle,high])
    m=XGBClassifier(**params,n_jobs=4,random_state=SEED,eval_metric='mlogloss',tree_method='hist',**({} if baseline else {'early_stopping_rounds':30}))
    m.fit(x,y,sample_weight=weights[y],eval_set=[(dx,dy)],verbose=False)
    return m

def initialize():
    if OUT.exists():raise RuntimeError('New experiment path required.')
    OUT.mkdir(parents=True)
    paths=[p for p in MODELS.rglob('*') if p.is_file()]+[ROOT/p for p in ['models/active_model.json','data/expanded_survey_dataset/respondents.csv','src/brfss_inference.py','src/ensemble_policy.py','src/models.py','server.py','deploy/backend.py','frontend/prediction.js']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};save(OUT/'original_hashes.json',hashes)
    for name in hashes:
        dest=OUT/'backup'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    tr,va=frames();assert tr.respondent_id.is_unique and va.respondent_id.is_unique
    assert not set(tr.profile_group)&set(va.profile_group)
    save(OUT/'protocol.json',dict(seed=SEED,trials=40,search='Seeded ParameterSampler from existing sklearn; sequential four-thread XGBoost. No extra dependencies.',
        boundary='High when P2>=h; otherwise sometimes when P1/(P0+P1)>=s, else rarely. High grid81 quantiles plus current ensemble cutoff; exact distinct lower cutoffs. Raw probabilities unchanged.',
        cv='Two stratified group outer folds within original training. Each outer training part: grouped 20% threshold calibration; remaining grouped 12.5% early-stop and 87.5% fit. Four disjoint roles; preprocessing fit only on fit role. Original XGB baseline refitted per fold; NEVER use original full-training checkpoints to score training folds.',
        baseline='Original XGB 350 fixed rounds, original features and training weights; original deployed multipliers. Threshold baseline is its own original decision within each fold.',
        objective='Maximize mean outer-fold accuracy among candidates satisfying BOTH high and sometimes recalls, macro-F1 and QWK vs refitted baseline on EVERY scoring fold. High floor max(60%,baseline); sometimes floor baseline. Thresholds chosen on inner calibration with same constraints. No thresholds fitted on scoring fold.',
        selection='First evaluate two-boundary-only policies on validation with saved models. Then final-fit single best feasible CV candidate. If none feasible, fit lowest summed violation for diagnostics only, ineligible for promotion. No alternate candidates selected from holdout.',
        final_fit='Entire training:20% grouped threshold reserve not used for model fitting; remaining12.5% earlystop; remainingfit. Existing validation selects final thresholds. Both original and engineered encodings in search.',
        promotion='Freeze one whole candidate configuration (boundaries-only or tuned XGB+boundaries) before holdout. Require both XGB and ensemble accuracy improvements with no loss of both recalls, macro-F1,QWK on validation. Rank by ensemble validation accuracy. If none, diagnostic choice by least violation, never eligible. Test once; require both improvements, no recall or macro-F1/QWK loss vs current, and paired accuracy95CI lower>0. Then full512MiB/API/rollback checks before any local activation. No deployment.',
        test='Historically reused internal holdout, not untouched or independent. No retuning after evaluation.',
        baseline_accuracy='Historical always-rarely65.78%, high/sometimes recall0%; report split-specific baseline.',
        search_space=dict(depth=[2,6],learning_rate=[.02,.12],min_child_weight=[3,60],subsample=[.7,1.],colsample=[.7,1.],reg_alpha=[1e-5,5.],reg_lambda=[.5,30.],gamma=[0,1.5],weight_power=[.15,.65],middle_weight=[.8,2.5],high_weight=[.8,1.5],rounds=550),
        changes='No real respondents added/removed/relabeled. No resampling. Training-only class weights documented. Source data, active checkpoints and prior experiments never overwritten.'))
    save(OUT/'source_hashes.json',{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('feature_pipeline.py'),Path(__file__).with_name('tune_six_models.py'),Path(__file__).with_name('tune_xgb_features.py')]})
    save(OUT/'resources.json',dict(cpu_count=os.cpu_count(),memory=Path('/proc/meminfo').read_text(),disk_free=shutil.disk_usage(OUT).free,python=platform.python_version(),threads=4))
    # Declare all candidate settings before inspecting any candidate outcome.
    space=dict(max_depth=randint(2,7),learning_rate=loguniform(.02,.12),min_child_weight=loguniform(3,60),subsample=uniform(.7,.3),colsample_bytree=uniform(.7,.3),reg_alpha=loguniform(1e-5,5),reg_lambda=loguniform(.5,30),gamma=uniform(0,1.5),weight_power=uniform(.15,.5),middle_weight=uniform(.8,1.7),high_weight=uniform(.8,.7),mode=['original','engineered'])
    configs=[baseline_params()]+[dict(n_estimators=550,**p) for p in ParameterSampler(space,n_iter=39,random_state=SEED)]
    save(OUT/'configurations.json',configs);verify();print('PROTOCOL AND 40 CONFIGURATIONS FROZEN',flush=True)

def boundary_phase():
    if (OUT/'boundary_validation.json').exists():raise RuntimeError('Already completed.')
    verify();_,va=frames();y=va.target.to_numpy();raw=original(va);refs=actual(y,raw)
    save(OUT/'boundary_validation.json',dict(original=refs,candidate=policies(y,raw,refs),majority_accuracy=float(np.mean(y==0))))
    np.savez_compressed(OUT/'original_validation.npz',**raw)
    print('BOUNDARIES',json.dumps(read(OUT/'boundary_validation.json')),flush=True)

def make_folds(tr):
    caches=[];roles=[]
    for k,(a,b) in enumerate(StratifiedGroupKFold(2,shuffle=True,random_state=SEED).split(tr,tr.target,tr.profile_group)):
        working,cal=partition(tr.iloc[a],5,SEED+k+1);ft,dev=partition(working,8,SEED+k+11);score=tr.iloc[b]
        groups=[set(x.profile_group) for x in [ft,dev,cal,score]]
        assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
        d={}
        for mode in ['original','engineered']:
            enc=FeatureEncoder(SCHEMA,mode).fit(ft);d[mode]=[enc.transform(x) for x in [ft,dev,cal,score]]
        d['ys']=[x.target.to_numpy() for x in [ft,dev,cal,score]]
        for role,frame in zip(['fit','early_stop','threshold_calibration','score'],[ft,dev,cal,score]):
            roles.extend(dict(respondent_id=r,fold=k,role=role) for r in frame.respondent_id)
        x,dx,cx,sx=d['original'];y,dy,cy,sy=d['ys'];model=fit(baseline_params(),x,y,dx,dy,baseline=True)
        cp=model.predict_proba(cx);sp=model.predict_proba(sx)
        d['reference_cal']=metric(cy,cp,(cp*MULT).argmax(1));d['reference_score']=metric(sy,sp,(sp*MULT).argmax(1))
        d['baseline_two_boundary']=boundaries(cy,cp,d['reference_cal']);d['baseline_two_score']=metric(sy,sp,predict(sp,d['baseline_two_boundary']['thresholds']))
        caches.append(d);print('PREPARED FOLD',k,flush=True)
    pd.DataFrame(roles).to_csv(OUT/'cv_roles.csv',index=False)
    save(OUT/'cv_baselines.json',[{k:v for k,v in d.items() if k.startswith('reference') or k.startswith('baseline')} for d in caches])
    return caches

def search():
    verify();tr,_=frames();cache=make_folds(tr);configs=read(OUT/'configurations.json');results=[]
    for number,cfg in enumerate(configs):
        path=OUT/'trials'/f'{number:03d}.json'
        if path.exists():results.append(read(path));continue
        start=time.monotonic();folds=[]
        for d in cache:
            x,dx,cx,sx=d[cfg['mode']];y,dy,cy,sy=d['ys'];m=fit(cfg,x,y,dx,dy,baseline=(number==0))
            cp=np.asarray(m.predict_proba(cx),dtype=float);sp=np.asarray(m.predict_proba(sx),dtype=float)
            policy=boundaries(cy,cp,d['reference_cal']);sc=metric(sy,sp,predict(sp,policy['thresholds']));v=violations(sc,d['reference_score'])
            folds.append(dict(calibration=policy,score=sc,raw_score=metric(sy,sp),violations=v,best_iteration=getattr(m,'best_iteration',cfg['n_estimators']-1)))
            del m;gc.collect()
        r=dict(trial=number,params=cfg,folds=folds,feasible=bool(all(f['calibration']['feasible'] and max(f['violations'])<=1e-12 for f in folds)),violation=float(sum(max(0,v) for f in folds for v in f['violations'])),accuracy=float(np.mean([f['score']['accuracy'] for f in folds])),seconds=time.monotonic()-start)
        save(path,r);results.append(r);print('TRIAL',number+1,'/40','accuracy',round(r['accuracy'],5),'feasible',r['feasible'],'violation',round(r['violation'],5),flush=True)
    eligible=[r for r in results if r['feasible']];winner=max(eligible,key=lambda r:r['accuracy']) if eligible else min(results,key=lambda r:(r['violation'],-r['accuracy']))
    save(OUT/'cv_selection.json',dict(candidate=winner,feasible_trials=len(eligible),diagnostic_only=not bool(eligible)));verify()

def select():
    if (OUT/'frozen_selection.json').exists():raise RuntimeError('Already frozen.')
    verify();tr,va=frames();working,cal=partition(tr,5,SEED+100);ft,dev=partition(working,8,SEED+101)
    cv=read(OUT/'cv_selection.json');cfg=cv['candidate']['params'];enc=FeatureEncoder(SCHEMA,cfg['mode']).fit(ft);start=time.monotonic()
    m=fit(cfg,enc.transform(ft),ft.target.to_numpy(),enc.transform(dev),dev.target.to_numpy(),baseline=(cv['candidate']['trial']==0))
    joblib.dump(m,OUT/'candidate.joblib');save(OUT/'encoder.json',enc.config())
    save(OUT/'final_fit.json',dict(seconds=time.monotonic()-start,fit_rows=len(ft),earlystop_rows=len(dev),reserved_rows=len(cal),best_iteration=getattr(m,'best_iteration',cfg['n_estimators']-1)))
    pd.concat([frame[['respondent_id']].assign(role=role) for frame,role in [(ft,'fit'),(dev,'early_stop'),(cal,'reserved')]]).to_csv(OUT/'final_roles.csv',index=False)
    raw=dict(np.load(OUT/'original_validation.npz'));p=np.asarray(m.predict_proba(enc.transform(va)),dtype=float);np.save(OUT/'candidate_validation.npy',p)
    y=va.target.to_numpy();refs=read(OUT/'boundary_validation.json')['original'];newraw={**raw,'XGBoost':p}
    candidates={'boundaries_only':read(OUT/'boundary_validation.json')['candidate'],'tuned_xgb':policies(y,newraw,refs)}
    eligible=[]
    for name,pol in candidates.items():
        if all(v['improves'] for v in pol.values()) and (name!='tuned_xgb' or not cv['diagnostic_only']):eligible.append(name)
    chosen=max(eligible,key=lambda n:candidates[n]['ensemble']['metrics']['accuracy']) if eligible else min(candidates,key=lambda n:(sum(max(0,v) for model in ['xgb','ensemble'] for v in violations(candidates[n][model]['metrics'],refs[model])), -candidates[n]['ensemble']['metrics']['accuracy']))
    save(OUT/'frozen_selection.json',dict(chosen=chosen,eligible=bool(eligible),candidates=candidates,original=refs,cv_eligible=not cv['diagnostic_only'],weights=POLICY['weights'],hashes={n:sha(OUT/n) for n in ['candidate.joblib','encoder.json','protocol.json','configurations.json','cv_selection.json']},source_hashes=read(OUT/'source_hashes.json')))
    verify();print('FROZEN',chosen,'eligible',bool(eligible),flush=True)

def evaluate():
    verify();f=read(OUT/'frozen_selection.json')
    for name,digest in f['hashes'].items():assert sha(OUT/name)==digest,name
    for name,digest in f['source_hashes'].items():assert sha(Path(__file__).with_name(name))==digest,name
    with (OUT/'TEST_STARTED.json').open('x') as stream:json.dump({'frozen_sha256':sha(OUT/'frozen_selection.json')},stream)
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');te=df[df.split=='test'];y=te.target.to_numpy();raw=original(te);old=actual(y,raw)
    # Only the single frozen candidate is evaluated. No test-based alternative selection.
    newraw=raw
    if f['chosen']=='tuned_xgb':
        enc=FeatureEncoder.restore(read(OUT/'encoder.json'));m=joblib.load(OUT/'candidate.joblib');newraw={**raw,'XGBoost':m.predict_proba(enc.transform(te))}
    selected=f['candidates'][f['chosen']];report={}
    for name,op,np_ in [('xgb',raw['XGBoost'],newraw['XGBoost']),('ensemble',previous.blend(raw,POLICY['weights']),previous.blend(newraw,POLICY['weights']))]:
        original_pred=(op*MULT).argmax(1) if name=='xgb' else previous.decide(op,POLICY['high_threshold'])
        pred=predict(np_,selected[name]['thresholds']);metrics=metric(y,np_,pred);ci=interval(y,original_pred,pred)
        v=violations(metrics,old[name]);passed=f['eligible'] and metrics['accuracy']>old[name]['accuracy'] and max(v)<=1e-12 and ci['accuracy_delta_95ci'][0]>0
        report[name]=dict(original=old[name],candidate=metrics,raw_argmax=metric(y,np_),paired=ci,violations=v,passes=bool(passed))
        np.savez_compressed(OUT/f'{name}_test_predictions.npz',target=y,respondent_id=te.respondent_id.to_numpy(dtype=str),original=original_pred,candidate=pred,probabilities=np_)
    report.update(chosen=f['chosen'],approved_for_runtime=all(report[n]['passes'] for n in ['xgb','ensemble']),majority_accuracy=float(np.mean(y==0)),note='Previously reused internal holdout, no independent generalization claim.')
    save(OUT/'test_report.json',report);verify();print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['init','boundaries','search','select','test']);ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    ROOT=Path(args.project);OUT=Path(args.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');POLICY=read(MODELS/'ensemble/policy.json');MULT=np.asarray(read(MODELS/'selection.json')['XGBoost']['multipliers']);torch.set_num_threads(4)
    {'init':initialize,'boundaries':boundary_phase,'search':search,'select':select,'test':evaluate}[args.phase]()
