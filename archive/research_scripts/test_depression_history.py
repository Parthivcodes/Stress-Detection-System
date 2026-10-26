"""Predeclared BRFSS historical-depression ablation; no production mutation."""
import argparse,copy,gc,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
import analyze_predictor_audit as audit
from tune_six_models import read,save,sha,metric,partition
from tune_two_boundaries import boundaries,predict,violations

SEED=20261025
PARAMS=dict(n_estimators=350,max_depth=5,learning_rate=.04,min_child_weight=12,reg_lambda=5,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=SEED,eval_metric='mlogloss',tree_method='hist')
def protect():
    for n,h in read(OUT/'original_hashes.json').items():assert sha(ROOT/n)==h,n
def sufficient(m,b):return bool(m['accuracy']>b['accuracy']+1e-12 and max(violations(m,b))<=1e-12)
def fit(frame,cols):
    enc=audit.ExtraEncoder(cols).fit(frame);x=enc.transform(frame);y=frame.target.to_numpy();weights=(len(y)/(3*np.bincount(y,minlength=3)))**.5
    m=XGBClassifier(**PARAMS);m.fit(x,y,sample_weight=weights[y]);return enc,m
def initialize():
    if OUT.exists():raise RuntimeError('Refusing overwrite.')
    OUT.mkdir(parents=True)
    paths=[p for p in MODELS.rglob('*') if p.is_file()]+[ROOT/n for n in ['models/active_model.json','data/expanded_survey_dataset/respondents.csv','src/brfss_inference.py','src/ensemble_policy.py','experiments/predictor_audit_20261025/same_respondent_raw_candidates.csv','experiments/predictor_audit_20261025/codebook_catalog.json']]
    save(OUT/'original_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for p in paths:
        if p.suffix=='.csv':continue
        dest=OUT/'backup'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    book=read(AUDIT/'codebook_catalog.json');coverage=pd.read_csv(AUDIT/'coverage.csv');coverage=coverage[coverage.split.eq('train')].set_index('variable')
    reasons={
      'ADDEPEV3':('test','Ever told depressive disorder; not calculated from SDHSTRE1. Available as self-report at assessment. Diagnosis timing versus current stress unknown; not valid evidence for future prediction.'),
      'MENTHLTH':('exclude','Explicitly includes stress in the same30-day question; strong conceptual overlap.'),
      'LSATISFY':('exclude','Current life satisfaction is not literal target leakage, but overlapping concurrent well-being; keep task conservative.'),
      'POORHLTH':('exclude','Concurrent mental/physical limitation; skip depends on mental-health days.'),
      'DECIDE':('exclude','Concurrent mental/emotional concentration symptoms; attribution and timing ambiguous.'),
      'DIFFALON':('exclude','Concurrent mental/emotional functional difficulty; attribution and timing ambiguous.'),
      'ACEDEPRS':('exclude','Historical household exposure is not literal target leakage, but optional-module missingness and sensitive retrospective context make it unsuitable for this bounded experiment.')}
    save(OUT/'feature_review.json',{n:dict(decision=v[0],reason=v[1],codebook=book[n],training_valid_rate=float(coverage.loc[n,'valid_rate'])) for n,v in reasons.items()})
    save(OUT/'protocol.json',dict(seed=SEED,parameters=PARAMS,features=SCHEMA['numeric']+SCHEMA['categorical'],addition='ADDEPEV3 only, 1=yes,2=no; all other codes explicit unknown category. No imputing unknown as no, no dropping respondents.',
      task='Concurrent self-reported stress-frequency assessment, not diagnosis or future stress prediction. History is not a target-derived field; absolute freedom from contextual/temporal overlap cannot be established.',
      cv='Original full training128248,3 stratified profile-group folds, same seed/folds as predictor audit. Each outer train reserves grouped20% inner calibration. Same fit/cal/score rows for baseline19 vs added20. Fit preprocessing only on fit. No search/early stopping.',
      thresholds='Two-boundary rule selected only on inner calibration with81quantile high grid and exact middle grid. Maxaccuracy preserving original fixed-policy refit baseline high/sometimes recall,F1,QWK; high>=max(.60,baseline). Infeasible fallbacks ineligible.',
      eligibility='Pooled accuracy strictly better and high/sometimes recall,F1,QWK preserved. Every scoring fold must preserve recalls and safeguards and each calibration must be feasible. This conservative screen can reject small improvements.',
      validation='If CV passes, refit once on full original training, choose thresholds on historically reused full validation against componentwise stricter activeXGB/currentensemble requirements. Freeze before any final holdout. No switching candidates.',
      final='Only if validation passes, evaluate once on full30718 reused internal holdout. Require accuracy improvement and both recalls,F1,QWK preserved versus currentXGB and ensemble; paired accuracy improvement evidence required. No retuning after test.',
      promotion='No deployment. Only after final performance plus API/frontend/schema20input,unknownhandling,512MB runtime and rollback checks. Until then production unchanged.',
      baseline='Alwaysrarely per split, historical reusedholdout65.78% and0minorityrecall.',budget='6foldfits and optional1finalfit.4threads sequential; no paid compute/new dependencies.'))
    shutil.copyfile(__file__,OUT/'test_depression_history.py');save(OUT/'frozen_hashes.json',{n:sha(OUT/n) for n in ['protocol.json','feature_review.json','test_depression_history.py']})
    print('PROTOCOL FROZEN',flush=True)

def run():
    initialize();protect()
    audit.ROOT=ROOT;audit.OUT=AUDIT;audit.SPEC=read(AUDIT/'candidate_spec.json');audit.SCHEMA=SCHEMA
    audit.verify();df,raw,extra=audit.load()
    assert sha(AUDIT/'same_respondent_raw_candidates.csv')==read(AUDIT/'extraction.json')['candidate_sha256']
    frame=pd.concat([df,extra[['ADDEPEV3']]],axis=1);tr=frame[frame.split.eq('train')].reset_index(drop=True);y=tr.target.to_numpy()
    assert df.respondent_id.is_unique and df.groupby('profile_group').split.nunique().max()==1
    outputs={n:np.zeros((len(tr),3)) for n in ['baseline','depression_history']};preds={n:np.zeros(len(tr),dtype=int) for n in outputs};records=[];roles=[]
    for fold,(aa,bb) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(tr,y,tr.profile_group)):
        ft,cal=partition(tr.iloc[aa],5,SEED+fold);score=tr.iloc[bb];cy=cal.target.to_numpy();sy=score.target.to_numpy()
        assert not set(ft.profile_group)&set(cal.profile_group) and not set(tr.iloc[aa].profile_group)&set(score.profile_group)
        for role,f in [('fit',ft),('calibration',cal),('score',score)]:roles.extend(dict(respondent_id=i,profile_group=g,fold=fold,role=role) for i,g in zip(f.respondent_id,f.profile_group))
        be,bm=fit(ft,[]);bc=bm.predict_proba(be.transform(cal));bs=bm.predict_proba(be.transform(score));br=metric(cy,bc,(bc*MULT).argmax(1));ref=metric(sy,bs,(bs*MULT).argmax(1))
        outputs['baseline'][bb]=bs;preds['baseline'][bb]=(bs*MULT).argmax(1)
        start=time.monotonic();enc,m=fit(ft,['ADDEPEV3']);cp=m.predict_proba(enc.transform(cal));sp=m.predict_proba(enc.transform(score));policy=boundaries(cy,cp,br);pp=predict(sp,policy['thresholds']);mm=metric(sy,sp,pp)
        r=dict(fold=fold,baseline=ref,policy=policy,candidate=mm,raw_argmax=metric(sy,sp),fixed_rule=metric(sy,sp,(sp*MULT).argmax(1)),passes=bool(policy['feasible'] and max(violations(mm,ref))<=1e-12),seconds=time.monotonic()-start,fit_rows=len(ft),calibration_rows=len(cal),score_rows=len(score),encoded_features=enc.transform(score.iloc[:1]).shape[1])
        save(OUT/f'fold_{fold}.json',r);records.append(r);outputs['depression_history'][bb]=sp;preds['depression_history'][bb]=pp
        print('FOLD',fold,'old',ref['accuracy'],'new',mm['accuracy'],'high',mm['recall'][2],'sometimes',mm['recall'][1],'pass',r['passes'],flush=True);del bm,m;gc.collect()
    pd.DataFrame(roles).to_csv(OUT/'roles.csv',index=False)
    np.savez_compressed(OUT/'oof.npz',respondent_id=tr.respondent_id.to_numpy(dtype=str),target=y,**outputs,**{n+'_pred':v for n,v in preds.items()})
    b=metric(y,outputs['baseline'],preds['baseline']);m=metric(y,outputs['depression_history'],preds['depression_history']);eligible=sufficient(m,b) and all(r['passes'] for r in records)
    save(OUT/'summary.json',dict(baseline=b,candidate=m,raw_argmax=metric(y,outputs['depression_history']),fixed_rule=metric(y,outputs['depression_history'],(outputs['depression_history']*MULT).argmax(1)),eligible=eligible,majority_accuracy=float(np.mean(y==0)),rows=len(y)))
    if eligible:final(frame)
    else:save(OUT/'decision.json',dict(status='rejected_in_training_cv',production_changed=False,validation_evaluated=False,test_evaluated=False,reason='Failed predeclared accuracy/recall/safeguard screen. No validation or holdout evaluation.'))
    protect();print('DONE',json.dumps(read(OUT/'decision.json')),flush=True)

def final(frame):
    tr=frame[frame.split.eq('train')];va=frame[frame.split.eq('validation')];enc,m=fit(tr,['ADDEPEV3']);p=m.predict_proba(enc.transform(va));y=va.target.to_numpy()
    olddir=ROOT/'experiments/hyperparameter_tuning_20261023'
    for n,h in read(olddir/'original_hashes.json').items():
        if n.startswith('models/current_models/') or n=='data/expanded_survey_dataset/respondents.csv':assert sha(ROOT/n)==h
    old=dict(np.load(olddir/'original_validation.npz'));policy=read(MODELS/'ensemble/policy.json');ep=sum(w*old[n].astype(float) for n,w in policy['weights'].items());ep/=ep.sum(1,keepdims=True)
    refs=[metric(y,old['XGBoost'],(old['XGBoost']*MULT).argmax(1)),metric(y,ep,np.where(ep[:,2]>=policy['high_threshold'],2,ep[:,:2].argmax(1)))]
    ref=copy.deepcopy(refs[0])
    for k in ['accuracy','macro_f1','qwk']:ref[k]=max(r[k] for r in refs)
    ref['recall']=[max(r['recall'][i] for r in refs) for i in range(3)]
    chosen=boundaries(y,p,ref);save(OUT/'validation.json',dict(reference=ref,policy=chosen,raw_argmax=metric(y,p),majority_accuracy=float(np.mean(y==0))))
    m.save_model(OUT/'candidate_xgb.json');save(OUT/'encoder.json',dict(base=enc.base.config(),addition='ADDEPEV3',categories=[1,2,-1]))
    passed=chosen['feasible'] and sufficient(chosen['metrics'],ref)
    save(OUT/'frozen_candidate.json',dict(thresholds=chosen['thresholds'],hashes={n:sha(OUT/n) for n in ['candidate_xgb.json','encoder.json']},eligible=passed))
    save(OUT/'decision.json',dict(status='awaiting_final_evaluation' if passed else 'rejected_on_validation',production_changed=False,validation_evaluated=True,test_evaluated=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';AUDIT=ROOT/'experiments/predictor_audit_20261025';SCHEMA=read(MODELS/'schema.json');MULT=np.asarray(read(MODELS/'selection.json')['XGBoost']['multipliers']);run()
