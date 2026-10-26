"""Nested held-out stacking diagnostic, not a production replacement."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from tune_six_models import read,save,sha,partition,metric,NAMES
from tune_two_boundaries import boundaries,predict
from bounded_evidence_tuning import strictref,better,SEEDS

def main(root,out,initialize=False):
    protocol=out/'stack_protocol.json'
    if initialize:
        if protocol.exists():raise RuntimeError('Already registered.')
        save(protocol,dict(source_sha256=sha(__file__),method='LR meta-learner C=.1,maxiter2000,sqrt inverse class weights,18rawprobabilities. Neural2seedmean for baseline and stack.',splits='Within each outer fold base models use fit/dev. Split calibration groups50/50 into meta-fit and threshold-calibration. Outer score never fits meta learner or thresholds.',reference='Refitted current LR.4/XGB.5/LSTM.1 blend with currentcutoff and calibration-matched policy. Protectbothrecalls,F1,QWK and improveaccuracy on eachscorefold and pooled.',scope='Diagnostic only,no test. Positive result requires fulltraining/validation/runtime confirmation.'))
        return
    assert sha(__file__)==read(protocol)['source_sha256']
    df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')].set_index('respondent_id');policy=read(root/'models/current_models/ensemble/policy.json');results=[];outputs=[]
    for fold in range(2):
        cps={};sps={}
        for i,n in enumerate(NAMES):
            files=[np.load(out/f'fit_{fold}_{i}_baseline_{seed}.npz') for seed in SEEDS if i>=3 or seed==SEEDS[0]]
            ids=files[0]['cal_ids'];sids=files[0]['score_ids'];cps[n]=np.mean([f['cal'] for f in files],axis=0);sps[n]=np.mean([f['score'] for f in files],axis=0)
            if i==0:calids=ids;scoreids=sids
            else:assert np.array_equal(calids,ids) and np.array_equal(scoreids,sids)
        cal=tr.loc[calids].reset_index();cal['position']=np.arange(len(cal));mf,tc=partition(cal,2,20261026+fold);fitidx=mf.position.to_numpy();ci=tc.position.to_numpy();sy=tr.loc[scoreids].target.to_numpy();cy=tc.target.to_numpy();my=mf.target.to_numpy()
        x=np.concatenate([cps[n] for n in NAMES],axis=1);sx=np.concatenate([sps[n] for n in NAMES],axis=1);w=(len(my)/(3*np.bincount(my,minlength=3)))**.5
        m=LogisticRegression(C=.1,max_iter=2000,class_weight=dict(enumerate(w)),random_state=20261026);m.fit(x[fitidx],my);cp=m.predict_proba(x[ci]);sp=m.predict_proba(sx)
        bc=sum(float(w)*cps[n][ci] for n,w in policy['weights'].items());bs=sum(float(w)*sps[n] for n,w in policy['weights'].items())
        original=lambda p:np.where(p[:,2]>=policy['high_threshold'],2,p[:,:2].argmax(1))
        oc=metric(cy,bc,original(bc));os=metric(sy,bs,original(bs));bp=boundaries(cy,bc,oc);bsp=predict(bs,bp['thresholds']) if bp['feasible'] else original(bs);bm=metric(sy,bs,bsp)
        ref=strictref(oc,bp['metrics'] if bp['feasible'] else oc);cpolicy=boundaries(cy,cp,ref);pred=predict(sp,cpolicy['thresholds']);mm=metric(sy,sp,pred)
        results.append(dict(fold=fold,candidate=mm,raw=metric(sy,sp),original_blend=os,matched_blend=bm,policy=cpolicy,eligible=bool(cpolicy['feasible'] and better(mm,strictref(os,bm))),majority_accuracy=float(np.mean(sy==0))))
        outputs.append(dict(y=sy,p=sp,pred=pred,bp=bs,bpred=bsp,opred=original(bs)))
    cat=lambda k:np.concatenate([v[k] for v in outputs]);y=cat('y');mm=metric(y,cat('p'),cat('pred'));bm=metric(y,cat('bp'),cat('bpred'));om=metric(y,cat('bp'),cat('opred'));eligible=bool(all(r['eligible'] for r in results) and better(mm,strictref(bm,om)))
    save(out/'stack_results.json',dict(folds=results,candidate=mm,original_blend=om,matched_blend=bm,eligible_for_further_study=eligible,production_changed=False,test_evaluated=False,majority_accuracy=float(np.mean(y==0))))
    print('STACK',json.dumps(dict(accuracy=mm['accuracy'],high=mm['recall'][2],sometimes=mm['recall'][1],eligible=eligible)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);p.add_argument('--initialize',action='store_true');a=p.parse_args();main(Path(a.project),Path(a.output),a.initialize)
