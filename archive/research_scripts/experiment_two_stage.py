"""Bounded conditional two-stage classifier experiment; production stays read-only."""
import argparse,copy,gc,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from feature_pipeline import FeatureEncoder
from tune_six_models import sha,save,read,metric,partition
from tune_two_boundaries import boundaries,predict,violations

SEED=20261026
PARAMS=dict(n_estimators=350,max_depth=5,learning_rate=.04,min_child_weight=12,reg_lambda=5,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=SEED,eval_metric='logloss',tree_method='hist')

def protect():
    for n,h in read(OUT/'original_hashes.json').items():assert sha(ROOT/n)==h,n

def initialize():
    if OUT.exists():raise RuntimeError('Experiment already exists; no overwrite.')
    OUT.mkdir(parents=True)
    paths=list(p for p in MODELS.rglob('*') if p.is_file())+[ROOT/n for n in ['models/active_model.json','data/expanded_survey_dataset/respondents.csv','src/brfss_inference.py','src/ensemble_policy.py','training/train_expanded_models.py']]
    save(OUT/'original_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for p in paths:
        if p.suffix=='.csv':continue
        dest=OUT/'backup'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    save(OUT/'protocol.json',dict(seed=SEED,features=SCHEMA['numeric']+SCHEMA['categorical'],target={'0':'rarely','1':'sometimes','2':'high'},
      scope='Two predeclared candidates: binary class-weight exponents 0 and 0.5. Original19 inputs only, same respondents. No architecture/hyperparameter search after scoring.',
      probability='A=P(Y>0); B=P(Y=2 | Y>0); probabilities=[1-A,A*(1-B),A*B]. Stage2 trains only on actual Y>0. All respondents receive both predictions; no hard routing gate.',
      cv='Three stratified profile-group folds of original train only. Within each outer train reserve grouped20% calibration. Preprocessing fit on fit subset. Identical fit/calibration/score rows for baseline and both candidates.',
      baseline='Fresh multiclass XGB on fit rows with weight exponent0.5 and active XGB multipliers. Never score training folds using production checkpoints trained on them.',
      decision='Inner calibration only: high P2 threshold (81 quantiles), then conditional middle threshold. Max accuracy preserving reference sometimes/high recall, macroF1,QWK; high floor max(0.60,reference). Imported frozen two-boundary implementation.',
      eligibility='Pooled OOF accuracy strictly higher and both recalls,macroF1,QWK no lower than fold-refit baseline; all fold calibration feasible and each score fold preserves both recalls and safeguards. No candidate switching after final evaluation.',
      selection='If any CV eligible choose pooledaccuracy then macroF1, then lower exponent. Refit once on original training. Existing validation selects one threshold against componentwise strictest current XGB and ensemble recalls/F1/QWK, and greater of their accuracies. Validation is historically reused, not untouched.',
      final='Only if validation qualifies: freeze checkpoints/preprocessing/thresholds, perform one final reused internal-holdout evaluation. Require accuracy higher and both recalls,F1,QWK no lower than both active XGB and ensemble. No retuning or alternate candidate selection after holdout.',
      promotion='No deployment. Keep current production unless final performance,512MB runtime,API parity and rollback checks pass. Candidate can be rejected earlier without opening holdout.',
      parameters=PARAMS,trial_budget='2 binary pairs x3 folds plus3 multiclass baselines; conditional final refit of one pair. Sequential 4 threads, no new dependencies.',
      limitations='Cross-sectional survey, not temporal prediction. Class weights used only in fit subsets. Confidence in small gains limited by reused data and correlated folds. Majority baseline shown for each split; historical holdout majority65.78%.'))
    for name in ['experiment_two_stage.py','feature_pipeline.py','tune_six_models.py','tune_two_boundaries.py']:
        source=Path(__file__).parent/name;target=OUT/'source'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    save(OUT/'frozen_source_hashes.json',{n:sha(OUT/n) for n in ['protocol.json']+[f'source/{n}' for n in ['experiment_two_stage.py','feature_pipeline.py','tune_six_models.py','tune_two_boundaries.py']]})
    protect();print('PROTOCOL FROZEN; current models backed up',flush=True)

def fit(x,y,power,binary=True):
    p=copy.deepcopy(PARAMS)
    if not binary:p['eval_metric']='mlogloss'
    k=2 if binary else 3;weights=(len(y)/(k*np.bincount(y,minlength=k)))**power
    m=XGBClassifier(**p);m.fit(x,y,sample_weight=weights[y]);return m

def pair(x,y,power):return fit(x,(y>0).astype(int),power),fit(x[y>0],(y[y>0]==2).astype(int),power)

def probabilities(models,x):
    a=models[0].predict_proba(x)[:,1].astype(float);b=models[1].predict_proba(x)[:,1].astype(float)
    p=np.column_stack([1-a,a*(1-b),a*b]);assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
    return p

def sufficient(m,b,accuracy=True):return bool((not accuracy or m['accuracy']>b['accuracy']+1e-12) and max(violations(m,b))<=1e-12)

def run():
    protect();df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');assert df.respondent_id.is_unique
    tr=df[df.split.eq('train')].reset_index(drop=True);va=df[df.split.eq('validation')].reset_index(drop=True)
    assert not set(tr.profile_group)&set(va.profile_group)
    y=tr.target.to_numpy();folds=StratifiedGroupKFold(3,shuffle=True,random_state=SEED)
    names=['unweighted','weighted'];powers=[0.,.5];oof={n:np.zeros((len(tr),3)) for n in ['baseline']+names};dec={n:np.zeros(len(tr),dtype=int) for n in oof};records=[];roles=[]
    for fold,(train_idx,score_idx) in enumerate(folds.split(tr,y,tr.profile_group)):
        ft,cal=partition(tr.iloc[train_idx],5,SEED+fold);sc=tr.iloc[score_idx];fy=ft.target.to_numpy();cy=cal.target.to_numpy();sy=sc.target.to_numpy()
        assert not set(ft.profile_group)&set(cal.profile_group) and not set(tr.iloc[train_idx].profile_group)&set(sc.profile_group)
        for role,f in [('fit',ft),('calibration',cal),('score',sc)]:roles.extend(dict(respondent_id=i,fold=fold,role=role) for i in f.respondent_id)
        enc=FeatureEncoder(SCHEMA,'original').fit(ft);x=enc.transform(ft);cx=enc.transform(cal);sx=enc.transform(sc)
        baseline=fit(x,fy,.5,False);bp=baseline.predict_proba(sx);bc=baseline.predict_proba(cx)
        bpred=(bp*MULT).argmax(1);bs=metric(sy,bp,bpred);br=metric(cy,bc,(bc*MULT).argmax(1));oof['baseline'][score_idx]=bp;dec['baseline'][score_idx]=bpred
        for n,power in zip(names,powers):
            start=time.monotonic();models=pair(x,fy,power);cp=probabilities(models,cx);sp=probabilities(models,sx)
            policy=boundaries(cy,cp,br);pred=predict(sp,policy['thresholds']);m=metric(sy,sp,pred)
            r=dict(fold=fold,candidate=n,weight_power=power,baseline=bs,calibration_reference=br,policy=policy,score=m,raw_argmax=metric(sy,sp),passes=bool(policy['feasible'] and sufficient(m,bs,False)),seconds=time.monotonic()-start,fit_rows=len(ft),stage2_fit_rows=int((fy>0).sum()),calibration_rows=len(cal),score_rows=len(sc))
            records.append(r);save(OUT/f'fold_{fold}_{n}.json',r);oof[n][score_idx]=sp;dec[n][score_idx]=pred
            print('FOLD',fold,n,'accuracy',round(m['accuracy'],5),'high',round(m['recall'][2],5),'sometimes',round(m['recall'][1],5),'passes',r['passes'],flush=True)
            del models;gc.collect()
    pd.DataFrame(roles).to_csv(OUT/'roles.csv',index=False)
    np.savez_compressed(OUT/'oof.npz',respondent_id=tr.respondent_id.to_numpy(dtype=str),target=y,**oof,**{n+'_pred':p for n,p in dec.items()})
    base=metric(y,oof['baseline'],dec['baseline']);summ={};eligible=[]
    for n in names:
        m=metric(y,oof[n],dec[n]);ok=sufficient(m,base) and all(r['passes'] for r in records if r['candidate']==n)
        summ[n]=dict(metrics=m,raw_argmax=metric(y,oof[n]),eligible=ok,accuracy_change_points=100*(m['accuracy']-base['accuracy']),high_detected_delta=m['high_detected']-base['high_detected'],sometimes_detected_delta=m['confusion_matrix'][1][1]-base['confusion_matrix'][1][1])
        if ok:eligible.append(n)
    save(OUT/'cv_summary.json',dict(baseline=base,candidates=summ,majority_accuracy=float(np.mean(y==0)),rows=len(y)))
    if not eligible:
        save(OUT/'decision.json',dict(status='rejected_in_training_cv',production_changed=False,test_evaluated=False,reason='No candidate met the predeclared accuracy and recall/safeguard requirements. Validation and holdout not evaluated.'))
    else:
        chosen=max(eligible,key=lambda n:(summ[n]['metrics']['accuracy'],summ[n]['metrics']['macro_f1'],-powers[names.index(n)]))
        final(chosen,powers[names.index(chosen)],tr,va)
    protect();print('COMPLETED',json.dumps(read(OUT/'decision.json')),flush=True)

def final(name,power,tr,va):
    # This branch runs only after the predeclared training gate succeeds.
    enc=FeatureEncoder(SCHEMA,'original').fit(tr);models=pair(enc.transform(tr),tr.target.to_numpy(),power)
    p=probabilities(models,enc.transform(va));saved=ROOT/'experiments/hyperparameter_tuning_20261023'
    for n,h in read(saved/'original_hashes.json').items():
        if n.startswith('models/current_models/') or n=='data/expanded_survey_dataset/respondents.csv':assert sha(ROOT/n)==h
    raw=dict(np.load(saved/'original_validation.npz'));yp=va.target.to_numpy();policy=read(MODELS/'ensemble/policy.json')
    ep=sum(w*raw[n].astype(float) for n,w in policy['weights'].items());ep/=ep.sum(1,keepdims=True)
    refs=[metric(yp,raw['XGBoost'],(raw['XGBoost']*MULT).argmax(1)),metric(yp,ep,np.where(ep[:,2]>=policy['high_threshold'],2,ep[:,:2].argmax(1)))]
    reference=copy.deepcopy(refs[0])
    for key in ['accuracy','macro_f1','qwk']:reference[key]=max(r[key] for r in refs)
    reference['recall']=[max(r['recall'][i] for r in refs) for i in range(3)]
    selected=boundaries(yp,p,reference);save(OUT/'validation.json',dict(candidate=name,reference=reference,policy=selected,raw_argmax=metric(yp,p),majority_accuracy=float(np.mean(yp==0)),note='Historically reused validation, not untouched.'))
    if not (selected['feasible'] and sufficient(selected['metrics'],reference)):
        save(OUT/'decision.json',dict(status='rejected_on_validation',production_changed=False,test_evaluated=False,reason='Frozen CV winner did not improve accuracy while preserving both recalls and safeguards on reused validation.'));return
    for i,m in enumerate(models):m.save_model(OUT/f'stage_{i+1}.json')
    save(OUT/'encoder.json',enc.config());save(OUT/'frozen_candidate.json',dict(name=name,weight_power=power,thresholds=selected['thresholds'],files={n:sha(OUT/n) for n in ['stage_1.json','stage_2.json','encoder.json']}))
    save(OUT/'decision.json',dict(status='validation_passed_awaiting_final_evaluation',production_changed=False,test_evaluated=False,reason='Frozen candidate ready for single final evaluation and runtime checks; not yet approved for use.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');MULT=np.asarray(read(MODELS/'selection.json')['XGBoost']['multipliers'])
    initialize();run()
