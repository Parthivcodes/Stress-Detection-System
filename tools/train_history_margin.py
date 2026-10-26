"""Training-only weight search with predeclared recall margins; isolated artifacts."""
import argparse,copy,gc,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
import analyze_predictor_audit as audit
from test_depression_history import PARAMS,SEED
from tune_six_models import read,save,sha,metric,partition
from tune_two_boundaries import boundaries,predict,violations

FACTORS=[1.,1.1,1.25,1.5]
def protect():
    for n,h in read(OUT/'original_hashes.json').items():assert sha(ROOT/n)==h,n
def margin(ref):
    r=copy.deepcopy(ref);r['recall'][2]=min(1,max(.6,r['recall'][2])+.02);r['recall'][1]=min(1,r['recall'][1]+.005);return r
def sufficient(m,b):return bool(m['accuracy']>b['accuracy']+1e-12 and max(violations(m,b))<=1e-12)
def fit(frame,cols,factor):
    enc=audit.ExtraEncoder(cols).fit(frame);x=enc.transform(frame);y=frame.target.to_numpy();w=(len(y)/(3*np.bincount(y,minlength=3)))**.5;w[2]*=factor
    m=XGBClassifier(**PARAMS);m.fit(x,y,sample_weight=w[y]);return enc,m

def initialize():
    if OUT.exists():raise RuntimeError('Refusing overwrite.')
    OUT.mkdir(parents=True)
    paths=[p for p in MODELS.rglob('*') if p.is_file()]+[ROOT/n for n in ['models/active_model.json','data/expanded_survey_dataset/respondents.csv','src/brfss_inference.py','src/ensemble_policy.py','experiments/predictor_audit_20261025/same_respondent_raw_candidates.csv','experiments/depression_history_20261025/summary.json','experiments/depression_history_20261025/final_evaluation.json','experiments/depression_history_20261025/decision.json','experiments/depression_history_20261025/frozen_candidate.json','experiments/depression_history_20261025/candidate_xgb.json']]
    save(OUT/'original_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for p in paths:
        if p.suffix=='.csv':continue
        dest=OUT/'backup'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    save(OUT/'protocol.json',dict(seed=SEED,params=PARAMS,weight_factors=FACTORS,features=SCHEMA['numeric']+SCHEMA['categorical']+['ADDEPEV3'],
      scope='Only depression history added; explicit1/2/unknown encoding. Four predeclared high-class weight factors multiplying existing square-root inverse-frequency weights.1.0 is margin-only control. No changes to other hyperparameters or labels.',
      margin='During calibration/validation: high recall target=max(0.60,referencehigh)+0.02; sometimes target=reference+0.005, both capped1.0. MacroF1/QWK no lower. Margin is not a guarantee on unseen respondents.',
      cv='Three stratified profile-group folds of original128248 training respondents; grouped20% inner calibration; preprocessing fit on fit only. Same fit/cal/score for all trials. Fresh19feature XGB baseline uses active multipliers. No original full-training model scores on training folds.',
      selection='Eligible requires all inner-calibration policies feasible, every outerfold preserves baseline high/sometimes recalls,F1,QWK, pooledaccuracy strictly higher with pooled safeguards. Select highest pooledaccuracy,then macroF1,then lower weight. Score folds do not pick thresholds.',
      validation='Refit one CV winner on full training; full historically reused validation selects threshold with margins relative to componentwise strict currentXGB and ensemble references. Accuracy must improve over both and base recalls/F1/QWK be preserved. No alternate candidate if winner fails.',
      final='Freeze checkpoint,encoder andthresholds. Only after validation gate run one final full30718 reused internalholdout. Same final gates as prior: accuracy strictly higher and recalls/F1/QWK preserved vs both currentXGB and ensemble; pairedaccuracy95%CI lower>0. No retuning after test.',
      caveat='This hypothesis was motivated by previous holdout results. The holdout is reused and not independent confirmation. Do not target particular previously missed cases; no identity-based weight/filter.',
      promotion='No deployment. Only qualifying candidate after512MB/API/UI/rollback checks; otherwise originals remain. All past experiments preserved.',
      budget='15foldfits(3baselines+12candidates),optional1refit. Four CPU threads sequential. No paid compute or new dependencies.',
      baseline='Alwaysrarely at each split. Historical internalholdout65.78% accuracy,0minorityrecalls.'))
    for name in ['train_history_margin.py','test_depression_history.py','analyze_predictor_audit.py','feature_pipeline.py','tune_two_boundaries.py','tune_six_models.py']:
        target=OUT/'source'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/name,target)
    save(OUT/'frozen_hashes.json',{p.relative_to(OUT).as_posix():sha(p) for p in [OUT/'protocol.json']+list((OUT/'source').glob('*.py'))});print('PROTOCOL FROZEN',flush=True)

def run():
    initialize();protect();audit.ROOT=ROOT;audit.OUT=AUDIT;audit.SPEC=read(AUDIT/'candidate_spec.json');audit.SCHEMA=SCHEMA;audit.verify()
    df,raw,extra=audit.load();assert sha(AUDIT/'same_respondent_raw_candidates.csv')==read(AUDIT/'extraction.json')['candidate_sha256']
    frame=pd.concat([df,extra[['ADDEPEV3']]],axis=1);tr=frame[frame.split.eq('train')].reset_index(drop=True);y=tr.target.to_numpy();assert df.groupby('profile_group').split.nunique().max()==1
    keys=['baseline']+[str(f) for f in FACTORS];oof={n:np.zeros((len(tr),3)) for n in keys};preds={n:np.zeros(len(tr),dtype=int) for n in keys};records=[];roles=[]
    for fold,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(tr,y,tr.profile_group)):
        ft,cal=partition(tr.iloc[a],5,SEED+fold);sc=tr.iloc[b];cy=cal.target.to_numpy();sy=sc.target.to_numpy()
        assert not set(ft.profile_group)&set(cal.profile_group) and not set(tr.iloc[a].profile_group)&set(sc.profile_group)
        for role,f in [('fit',ft),('calibration',cal),('score',sc)]:roles.extend(dict(respondent_id=i,profile_group=g,fold=fold,role=role) for i,g in zip(f.respondent_id,f.profile_group))
        enc,m=fit(ft,[],1);cp=m.predict_proba(enc.transform(cal));sp=m.predict_proba(enc.transform(sc));br=metric(cy,cp,(cp*MULT).argmax(1));bs=metric(sy,sp,(sp*MULT).argmax(1));oof['baseline'][b]=sp;preds['baseline'][b]=(sp*MULT).argmax(1);del m
        for factor in FACTORS:
            start=time.monotonic();key=str(factor);enc,m=fit(ft,['ADDEPEV3'],factor);cp=m.predict_proba(enc.transform(cal));sp=m.predict_proba(enc.transform(sc));ref=margin(br);policy=boundaries(cy,cp,ref);pp=predict(sp,policy['thresholds']);mm=metric(sy,sp,pp)
            r=dict(fold=fold,factor=factor,baseline=bs,calibration_reference=br,margin_reference=ref,policy=policy,candidate=mm,raw_argmax=metric(sy,sp),passes=bool(policy['feasible'] and max(violations(mm,bs))<=1e-12),seconds=time.monotonic()-start)
            save(OUT/f'fold_{fold}_weight_{key}.json',r);records.append(r);oof[key][b]=sp;preds[key][b]=pp
            print('FOLD',fold,'factor',factor,'accuracy',round(mm['accuracy'],5),'high',round(mm['recall'][2],5),'sometimes',round(mm['recall'][1],5),'pass',r['passes'],flush=True);del m;gc.collect()
    pd.DataFrame(roles).to_csv(OUT/'roles.csv',index=False);np.savez_compressed(OUT/'oof.npz',respondent_id=tr.respondent_id.to_numpy(dtype=str),target=y,**oof,**{n+'_pred':v for n,v in preds.items()})
    base=metric(y,oof['baseline'],preds['baseline']);summary={};eligible=[]
    for factor in FACTORS:
        key=str(factor);mm=metric(y,oof[key],preds[key]);ok=sufficient(mm,base) and all(r['passes'] for r in records if r['factor']==factor)
        summary[key]=dict(metrics=mm,raw_argmax=metric(y,oof[key]),eligible=ok)
        if ok:eligible.append(key)
    save(OUT/'summary.json',dict(baseline=base,candidates=summary,rows=len(y),majority_accuracy=float(np.mean(y==0))))
    if eligible:
        winner=max(eligible,key=lambda k:(summary[k]['metrics']['accuracy'],summary[k]['metrics']['macro_f1'],-float(k)));save(OUT/'selected.json',dict(factor=float(winner),basis='Training CV only'));final(frame,float(winner))
    else:save(OUT/'decision.json',dict(status='rejected_in_training_cv',production_changed=False,validation_evaluated=False,test_evaluated=False,reason='No predeclared candidate passed training-only screen.'))
    protect();print('DONE',json.dumps(read(OUT/'decision.json')),flush=True)

def final(frame,factor):
    tr=frame[frame.split.eq('train')];va=frame[frame.split.eq('validation')];enc,m=fit(tr,['ADDEPEV3'],factor);p=m.predict_proba(enc.transform(va));y=va.target.to_numpy();olddir=ROOT/'experiments/hyperparameter_tuning_20261023'
    for n,h in read(olddir/'original_hashes.json').items():
        if n.startswith('models/current_models/') or n=='data/expanded_survey_dataset/respondents.csv':assert sha(ROOT/n)==h
    old=dict(np.load(olddir/'original_validation.npz'));policy=read(MODELS/'ensemble/policy.json');ep=sum(w*old[n].astype(float) for n,w in policy['weights'].items());ep/=ep.sum(1,keepdims=True)
    refs={'XGBoost':metric(y,old['XGBoost'],(old['XGBoost']*MULT).argmax(1)),'Ensemble':metric(y,ep,np.where(ep[:,2]>=policy['high_threshold'],2,ep[:,:2].argmax(1)))}
    ref=copy.deepcopy(refs['XGBoost'])
    for k in ['accuracy','macro_f1','qwk']:ref[k]=max(r[k] for r in refs.values())
    ref['recall']=[max(r['recall'][i] for r in refs.values()) for i in range(3)];chosen=boundaries(y,p,margin(ref));passed=chosen['feasible'] and sufficient(chosen['metrics'],ref)
    save(OUT/'validation.json',dict(factor=factor,baselines=refs,reference=ref,margin_reference=margin(ref),policy=chosen,eligible=passed,majority_accuracy=float(np.mean(y==0))))
    m.save_model(OUT/'candidate_xgb.json');save(OUT/'encoder.json',dict(base=enc.base.config(),addition='ADDEPEV3',categories=[1,2,-1]))
    save(OUT/'frozen_candidate.json',dict(factor=factor,thresholds=chosen['thresholds'],hashes={n:sha(OUT/n) for n in ['candidate_xgb.json','encoder.json']},eligible=passed))
    save(OUT/'decision.json',dict(status='awaiting_final_evaluation' if passed else 'rejected_on_validation',production_changed=False,validation_evaluated=True,test_evaluated=False,reason='Frozen winner only; no alternate candidate or threshold search on comparison/test results.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';AUDIT=ROOT/'experiments/predictor_audit_20261025';SCHEMA=read(MODELS/'schema.json');MULT=np.asarray(read(MODELS/'selection.json')['XGBoost']['multipliers']);run()
