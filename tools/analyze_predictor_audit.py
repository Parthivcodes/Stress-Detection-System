"""Audit diagnostics and fixed training-only ablations. No production writes/test selection."""
import argparse,copy,gc,itertools,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from feature_pipeline import FeatureEncoder
from tune_six_models import metric,partition,sha,save,read,NAMES
from tune_two_boundaries import cmstats

SEED=20261025
def verify():
    for name,digest in read(OUT/'original_hashes.json').items():assert sha(ROOT/Path(name.replace('\\','/')))==digest,name

def clean(raw):
    out=pd.DataFrame(index=raw.index)
    for name,s in SPEC.items():
        x=pd.to_numeric(raw[name],errors='coerce');ok=x.isin(s.get('valid',[]))
        for lo,hi in s.get('ranges',[])+([s['range']] if 'range' in s else []):ok|=x.between(lo,hi)
        for k in s.get('map',{}):ok|=x.eq(int(k))
        z=x.where(ok)
        for k,v in s.get('map',{}).items():z=z.mask(x.eq(int(k)),v)
        out[name]=z
    return out

def load():
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv')
    raw=pd.read_csv(OUT/'same_respondent_raw_candidates.csv',dtype=str,keep_default_na=False).set_index('respondent_id')
    assert raw.index.is_unique and df.respondent_id.is_unique
    raw=raw.loc[df.respondent_id].reset_index()
    assert np.array_equal(raw.split,df.split) and np.array_equal(raw.profile_group,df.profile_group)
    return df,raw,clean(raw)

def audit():
    verify();df,raw,extra=load();train=df.split.eq('train');book=read(OUT/'codebook_catalog.json');coverage=[];associations=[]
    for name,s in SPEC.items():
        for split in ['train','validation','test','quarantine']:
            ix=df.split.eq(split);x=extra.loc[ix,name];r=raw.loc[ix,name]
            coverage.append(dict(variable=name,label=book[name]['metadata']['Label'],group=s['group'],split=split,n=int(ix.sum()),valid=int(x.notna().sum()),valid_rate=float(x.notna().mean()),blank=int(r.eq('').sum()),invalid_nonblank=int((r.ne('')&x.isna()).sum()),note=s['note']))
        x=extra.loc[train,name].fillna(-1)
        if s.get('numeric'):x=pd.cut(x,[-2,-.5,0,1,2,4,10,31,100],include_lowest=True).astype(str)
        tab=pd.crosstab(x,df.loc[train,'target']);a=tab.to_numpy();expected=a.sum(1)[:,None]*a.sum(0)[None,:]/a.sum()
        chi=float(np.sum((a-expected)**2/np.maximum(expected,1e-15)));v=float(np.sqrt(chi/(a.sum()*max(1,min(a.shape)-1))))
        associations.append(dict(variable=name,group=s['group'],cramers_v=v,interpretation='Training-only univariate association including explicit missing; not incremental value or causation.'))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(associations).sort_values('cramers_v',ascending=False).to_csv(OUT/'training_associations.csv',index=False)
    sets={'context':[n for n,s in SPEC.items() if s['group']=='context'],'all_safe':[n for n,s in SPEC.items() if s['group'] in ['context','physical','lifestyle']]}
    save(OUT/'coverage_summary.json',dict(rows=len(df),train_rows=int(train.sum()),candidate_count=len(SPEC),sets=sets,complete_case_retention={k:float(extra.loc[train,cols].notna().all(axis=1).mean()) for k,cols in sets.items()},note='Retention is diagnostic only. Pilot retains everyone using explicit missing handling. Test coverage uses no outcome associations.'))
    # Reuse saved probabilities only after verifying data/checkpoint identity.
    olddir=ROOT/'experiments/hyperparameter_tuning_20261023';oldhash=read(olddir/'original_hashes.json')
    for name,digest in oldhash.items():
        if name.startswith('models/current_models/') or name=='data/expanded_survey_dataset/respondents.csv':assert sha(ROOT/name)==digest,name
    va=df[df.split=='validation'].reset_index(drop=True);y=va.target.to_numpy();probs=dict(np.load(olddir/'original_validation.npz'))
    selections=read(MODELS/'selection.json');policy=read(MODELS/'ensemble/policy.json');preds={};reports={}
    for name in NAMES:
        p=probs[name];assert p.shape==(len(va),3)
        pred=(p*np.asarray(selections[name]['multipliers'])).argmax(1);preds[name]=pred;reports[name]=metric(y,p,pred)
    ep=sum(w*probs[n].astype(float) for n,w in policy['weights'].items());ep/=ep.sum(1,keepdims=True)
    pred=np.where(ep[:,2]>=policy['high_threshold'],2,ep[:,:2].argmax(1));preds['Ensemble']=pred;reports['Ensemble']=metric(y,ep,pred)
    matrix=np.stack([preds[n] for n in NAMES],axis=1);wrong=matrix!=y[:,None];pairs=[]
    for i,j in itertools.combinations(range(6),2):
        for label in ['all',1,2]:
            ix=np.ones(len(y),dtype=bool) if label=='all' else y==label
            a=wrong[ix,i];b=wrong[ix,j];pairs.append(dict(model_a=NAMES[i],model_b=NAMES[j],true_class=str(label),error_jaccard=float(np.sum(a&b)/max(1,np.sum(a|b))),both_wrong=int(np.sum(a&b)),only_a_wrong=int(np.sum(a&~b)),only_b_wrong=int(np.sum(~a&b))))
    pd.DataFrame(pairs).to_csv(OUT/'validation_error_overlap.csv',index=False)
    error_summary={}
    for c in range(3):
        ix=y==c;allwrong=wrong[ix].all(1);unanimous=(matrix[ix]==matrix[ix,:1]).all(1)
        error_summary[str(c)]=dict(n=int(ix.sum()),all_six_wrong=int(allwrong.sum()),all_six_wrong_rate=float(allwrong.mean()),unanimous_wrong=int(np.sum(unanimous&allwrong)),any_model_correct_rate=float((~allwrong).mean()),note='Any-model-correct is an oracle diagnostic, not an achievable classifier accuracy.')
    grouping=dict(age=pd.cut(va.age,[17,34,49,64,81],labels=['18-34','35-49','50-64','65-80']),sex=va.sex.astype(str),sleep=pd.cut(va.sleep_duration,[0,6,9,24],labels=['under7','7to9','over9']),physical_days=pd.cut(va.unhealthy_days,[-1,0,13,30],labels=['0','1to13','14plus']),general_health=va.general_health.astype(str))
    subgroup=[]
    for variable,values in grouping.items():
        for value in values.dropna().unique():
            ix=(values==value).to_numpy();yy=y[ix];pp=pred[ix]
            if ix.sum()<100:continue
            subgroup.append(dict(variable=variable,value=str(value),n=int(ix.sum()),accuracy=float(np.mean(yy==pp)),sometimes_n=int(np.sum(yy==1)),sometimes_recall=float(np.mean(pp[yy==1]==1)) if np.sum(yy==1)>=30 else None,high_n=int(np.sum(yy==2)),high_recall=float(np.mean(pp[yy==2]==2)) if np.sum(yy==2)>=30 else None))
    pd.DataFrame(subgroup).to_csv(OUT/'validation_subgroups.csv',index=False)
    save(OUT/'validation_errors.json',dict(metrics=reports,class_overlap=error_summary,majority_accuracy=float(np.mean(y==0)),saved_prediction_sha256=sha(olddir/'original_validation.npz'),limitation='Descriptive reused-validation results, affected by historical model selection. Not untouched evaluation.'))
    verify();print('COVERAGE',json.dumps(read(OUT/'coverage_summary.json')),flush=True);print('ERROR OVERLAP',json.dumps(error_summary),flush=True)

class ExtraEncoder:
    def __init__(self,cols):self.cols=cols
    def fit(self,frame):
        self.base=FeatureEncoder(SCHEMA,'original').fit(frame)
        self.medians={n:float(frame[n].median()) if frame[n].notna().any() else 0. for n in self.cols if SPEC[n].get('numeric')}
        return self
    def transform(self,frame):
        parts=[self.base.transform(frame)]
        for n in self.cols:
            x=frame[n]
            if SPEC[n].get('numeric'):parts.extend([x.fillna(self.medians[n]).to_numpy()[:,None],x.isna().to_numpy()[:,None]])
            else:
                cats=SPEC[n]['valid']+[-1];xx=x.fillna(-1).to_numpy()
                parts.extend([(xx==c)[:,None] for c in cats])
        return np.concatenate(parts,axis=1).astype('float32')

def choose(y,p,reference):
    best=None;fallback=None
    for a in np.unique(np.r_[np.arange(.5,2.001,.05),MULT[1]]):
        for b in np.unique(np.r_[np.arange(.75,2.501,.05),MULT[2]]):
            pred=(p*np.array([1,a,b])).argmax(1);cm=np.bincount(3*y+pred,minlength=9).reshape(3,3);acc,rec,f1,qwk=cmstats(cm)
            violations=[max(.6,reference['recall'][2])-rec[2],reference['recall'][1]-rec[1],reference['macro_f1']-f1,reference['qwk']-qwk]
            v=sum(max(0,z) for z in violations);key=(float(acc),float(f1),float(qwk))
            if fallback is None or (v,-acc)<fallback[0]:fallback=((float(v),-float(acc)),[1,float(a),float(b)])
            if max(violations)<=1e-12 and (best is None or key>best[0]):best=(key,[1,float(a),float(b)])
    weights=(best or fallback)[1]
    return dict(multipliers=weights,feasible=bool(best is not None),metrics=metric(y,p,(p*weights).argmax(1)))

def pilot():
    if (OUT/'pilot_summary.json').exists():raise RuntimeError('Pilot already complete; no rerun.')
    verify();df,_,extra=load();tr=pd.concat([df,extra],axis=1).loc[df.split=='train'].reset_index(drop=True);y=tr.target.to_numpy()
    sets={'baseline':[],'context':[n for n,s in SPEC.items() if s['group']=='context'],'all_safe':[n for n,s in SPEC.items() if s['group'] in ['context','physical','lifestyle']]}
    outputs={k:np.zeros((len(tr),3)) for k in sets};decisions={k:np.zeros(len(tr),dtype=int) for k in sets};fixed={k:np.zeros(len(tr),dtype=int) for k in sets};roles=[];records=[];all_gain=[]
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(tr,y,tr.profile_group))
    for fold,(a,b) in enumerate(folds):
        ft,cal=partition(tr.iloc[a],5,SEED+fold);score=tr.iloc[b];sy=score.target.to_numpy();cy=cal.target.to_numpy();fy=ft.target.to_numpy()
        assert not set(ft.profile_group)&set(cal.profile_group) and not set(tr.iloc[a].profile_group)&set(score.profile_group)
        for role,frame in [('fit',ft),('calibration',cal),('score',score)]:roles.extend(dict(respondent_id=r,fold=fold,role=role) for r in frame.respondent_id)
        refcal=None;refscore=None
        for name,cols in sets.items():
            start=time.monotonic();enc=ExtraEncoder(cols).fit(ft);x=enc.transform(ft);cx=enc.transform(cal);sx=enc.transform(score)
            m=XGBClassifier(n_estimators=350,max_depth=5,learning_rate=.04,min_child_weight=12,reg_lambda=5,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=SEED,eval_metric='mlogloss',tree_method='hist')
            weights=(len(fy)/(3*np.bincount(fy,minlength=3)))**.5;m.fit(x,fy,sample_weight=weights[fy])
            cp=m.predict_proba(cx).astype(float);sp=m.predict_proba(sx).astype(float)
            if name=='baseline':refcal=metric(cy,cp,(cp*MULT).argmax(1));refscore=metric(sy,sp,(sp*MULT).argmax(1))
            policy=choose(cy,cp,refcal);pp=(sp*policy['multipliers']).argmax(1);mm=metric(sy,sp,pp);fm=metric(sy,sp,(sp*MULT).argmax(1))
            ok=policy['feasible'] and mm['recall'][2]+1e-12>=max(.6,refscore['recall'][2]) and mm['recall'][1]+1e-12>=refscore['recall'][1] and mm['macro_f1']+1e-12>=refscore['macro_f1'] and mm['qwk']+1e-12>=refscore['qwk']
            row=dict(fold=fold,variant=name,inputs=19+len(cols),encoded=sx.shape[1],fit_rows=len(ft),calibration_rows=len(cal),score_rows=len(score),policy=policy,score=mm,fixed_policy_score=fm,raw_score=metric(sy,sp),baseline_fixed_score=refscore,passes_recall_and_safeguards=bool(ok),seconds=time.monotonic()-start)
            records.append(row);save(OUT/f'pilot_{fold}_{name}.json',row);outputs[name][b]=sp;decisions[name][b]=pp;fixed[name][b]=(sp*MULT).argmax(1)
            # Gain attribution is descriptive fitted-model importance, not causal/permutation evidence.
            gain=m.feature_importances_;offset=69
            for c in cols:
                width=2 if SPEC[c].get('numeric') else len(SPEC[c]['valid'])+1
                all_gain.append(dict(fold=fold,variant=name,variable=c,normalized_gain=float(gain[offset:offset+width].sum())));offset+=width
            print('PILOT',fold,name,'accuracy',round(mm['accuracy'],5),'high',round(mm['recall'][2],5),'sometimes',round(mm['recall'][1],5),'safeguards',ok,flush=True)
            del m,x,cx,sx;gc.collect()
    pd.DataFrame(roles).to_csv(OUT/'pilot_roles.csv',index=False);pd.DataFrame(all_gain).to_csv(OUT/'pilot_feature_gain.csv',index=False)
    baseline=metric(y,outputs['baseline'],fixed['baseline']);summary={}
    for name in sets:
        m=metric(y,outputs[name],decisions[name]);byfold=[r for r in records if r['variant']==name]
        summary[name]=dict(constrained=m,fixed_policy=metric(y,outputs[name],fixed[name]),raw_argmax=metric(y,outputs[name]),all_folds_pass=all(r['passes_recall_and_safeguards'] for r in byfold),accuracy_delta_vs_fixed_baseline=m['accuracy']-baseline['accuracy'],high_detected_delta=m['high_detected']-baseline['high_detected'],sometimes_detected_delta=m['confusion_matrix'][1][1]-baseline['confusion_matrix'][1][1])
    save(OUT/'pilot_summary.json',dict(variants=summary,baseline_fixed=baseline,majority_accuracy=float(np.mean(y==0)),rows=len(tr),scope='Training-only3-fold out-of-fold diagnostic. No validation/test candidate selection or activation.'))
    np.savez_compressed(OUT/'pilot_oof.npz',respondent_id=tr.respondent_id.to_numpy(dtype=str),target=y,**outputs,**{k+'_pred':v for k,v in decisions.items()},baseline_fixed=fixed['baseline'])
    verify();print('PILOT COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','pilot']);p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';SPEC=read(OUT/'candidate_spec.json');SCHEMA=read(MODELS/'schema.json');MULT=np.array(read(MODELS/'selection.json')['XGBoost']['multipliers']);globals()[a.phase]()
