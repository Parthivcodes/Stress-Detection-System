"""Freeze training-CV winners on validation, then one guarded reused-holdout pass."""
import argparse,copy,gc,json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
import bounded_evidence_tuning as run
from tune_six_models import read,save,sha,metric,partition,prob,NAMES,FILES,interval
from tune_two_boundaries import boundaries,predict

def originals(frame):
    x=joblib.load(MODELS/'preprocessor.joblib').transform(frame[FEATURES]).astype('float32');raw={}
    for i,n in enumerate(NAMES):
        if i<3:m=joblib.load(MODELS/FILES[i])
        else:m=run.CLASSES[i-3](input_dim=SCHEMA['input_dim']);m.load_state_dict(torch.load(MODELS/FILES[i],weights_only=True,map_location='cpu'))
        raw[n]=prob(m,x);del m;gc.collect()
    return raw

def candidate(i,c,ft,dev,frame):
    predictions=[];records=[];paths=[]
    for seed in run.SEEDS if i>=3 else run.SEEDS[:1]:
        enc,m,info=run.fit(i,c,ft,dev,seed);predictions.append(prob(m,enc.transform(frame)));records.append(info)
        stem=OUT/'final_candidates'/f'{i}_{c}_{seed}';stem.parent.mkdir(exist_ok=True);ep=stem.with_suffix('.encoder.joblib');joblib.dump(enc,ep);paths.append(ep)
        mp=stem.with_suffix('.model.joblib' if i<3 else '.pt')
        if i<3:joblib.dump(m,mp)
        else:torch.save(m.state_dict(),mp)
        paths.append(mp);del m,enc;gc.collect()
    return np.mean(predictions,axis=0),records,paths

def load_candidate(i,c,frame):
    pp=[]
    for seed in run.SEEDS if i>=3 else run.SEEDS[:1]:
        stem=OUT/'final_candidates'/f'{i}_{c}_{seed}';enc=joblib.load(stem.with_suffix('.encoder.joblib'));x=enc.transform(frame)
        if i<3:m=joblib.load(stem.with_suffix('.model.joblib'))
        else:
            m=run.CLASSES[i-3](input_dim=x.shape[1]);m.load_state_dict(torch.load(stem.with_suffix('.pt'),weights_only=True,map_location='cpu'))
            if c=='focus':
                for l in m.modules():
                    if isinstance(l,torch.nn.Dropout):l.p=.35
        pp.append(prob(m,x));del m,enc;gc.collect()
    return np.mean(pp,axis=0)

def main():
    run.protect();decision=read(OUT/'decision.json');selected=decision['selected']
    if not selected:print('No CV-qualified candidates; validation and holdout remain closed.');return
    if (OUT/'validation_final.json').exists():raise RuntimeError('Refusing repeat final-selection run.')
    save(OUT/'final_protocol.json',dict(scope='Implementation of already-frozen final gates. No alternate trials after validation/test. Two neural seeds averaged as inCV. Final fit90% training/dev10%. Test singlepass only for validation-qualified candidates.',promotion='Performance alone does not activate; ensemble compatibility and512MB/API/rollback remain required. If none qualify retain originals.',source_sha256=sha(__file__)))
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')];va=df[df.split.eq('validation')];ft,dev=partition(tr,10,run.SEEDS[0]);y=va.target.to_numpy();old=originals(va);results={};frozen={};candidate_probs={}
    for n,c in selected.items():
        i=NAMES.index(n);print('FINAL FIT',n,c,flush=True);p,fitinfo,paths=candidate(i,c,ft,dev,va);candidate_probs[n]=p
        actual=metric(y,old[n],(old[n]*SEL[n]['multipliers']).argmax(1));bp=boundaries(y,old[n],actual)
        matched=bp['metrics'] if bp['feasible'] else actual;ref=run.strictref(actual,matched);policy=boundaries(y,p,ref);ok=bool(policy['feasible'] and run.better(policy['metrics'],ref))
        results[n]=dict(config=c,original=actual,matched_original=matched,reference=ref,raw=metric(y,p),candidate=policy['metrics'],policy=policy,baseline_policy=bp,eligible=ok,fit=fitinfo)
        frozen[n]=dict(config=c,eligible=ok,thresholds=policy['thresholds'],baseline_thresholds=bp['thresholds'] if bp['feasible'] else None,hashes={v.relative_to(OUT).as_posix():sha(v) for v in paths})
        print('VALIDATION',n,'accuracy',policy['metrics']['accuracy'],'high',policy['metrics']['recall'][2],'sometimes',policy['metrics']['recall'][1],'eligible',ok,flush=True)
    save(OUT/'validation_final.json',dict(models=results,majority_accuracy=float(np.mean(y==0)),rows=len(y)))
    save(OUT/'frozen_final.json',frozen);np.savez_compressed(OUT/'final_validation_predictions.npz',respondent_id=va.respondent_id.to_numpy(dtype=str),**candidate_probs)
    qualified=[n for n,v in frozen.items() if v['eligible']]
    if not qualified:
        save(OUT/'decision.json',dict(status='rejected_on_validation',selected=selected,production_changed=False,validation_evaluated=True,test_evaluated=False));run.protect();return
    if (OUT/'TEST_STARTED.json').exists():raise RuntimeError('Test pass already started; no automatic repetition.')
    save(OUT/'TEST_STARTED.json',dict(frozen_sha256=sha(OUT/'frozen_final.json'),caveat='Historically reused internalholdout,not untouched. No retuning after this.'))
    te=df[df.split.eq('test')];ty=te.target.to_numpy();oldtest=originals(te);testresults={};testp={};passed=[]
    for n in qualified:
        f=frozen[n];i=NAMES.index(n)
        for name,h in f['hashes'].items():assert sha(OUT/name)==h
        p=load_candidate(i,f['config'],te);testp[n]=p;pred=predict(p,f['thresholds']);op=(oldtest[n]*SEL[n]['multipliers']).argmax(1);bp=predict(oldtest[n],f['baseline_thresholds']) if f['baseline_thresholds'] is not None else op
        cm=metric(ty,p,pred);om=metric(ty,oldtest[n],op);bm=metric(ty,oldtest[n],bp);ref=run.strictref(om,bm)
        # Paired stratified bootstrap with fixed seed, reporting accuracy deltas.
        rng=np.random.default_rng(20261026);idx=[np.flatnonzero(ty==j) for j in range(3)];deltas=[]
        for _ in range(1000):
            chosen=np.concatenate([rng.choice(a,len(a),replace=True) for a in idx]);deltas.append([float(np.mean(pred[chosen]==ty[chosen])-np.mean(base[chosen]==ty[chosen])) for base in [op,bp]])
        cis=np.quantile(deltas,[.025,.975],axis=0).T.tolist();ok=bool(run.better(cm,ref) and min(v[0] for v in cis)>0)
        testresults[n]=dict(candidate=cm,raw=metric(ty,p),original=om,matched_original=bm,paired_accuracy_ci_vs_original=cis[0],paired_accuracy_ci_vs_matched=cis[1],passes_performance=ok)
        if ok:passed.append(n)
    np.savez_compressed(OUT/'final_test_predictions.npz',respondent_id=te.respondent_id.to_numpy(dtype=str),**testp)
    save(OUT/'test_final.json',dict(models=testresults,rows=len(te),majority_accuracy=float(np.mean(ty==0)),historical_baseline_accuracy=.6578,warning='Reused internal holdout,not independent confirmation.'))
    save(OUT/'decision.json',dict(status='performance_pass_pending_runtime_and_ensemble' if passed else 'rejected_on_reused_holdout',selected=selected,performance_passed=passed,production_changed=False,validation_evaluated=True,test_evaluated=True));run.protect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);p.add_argument('--diagnosis',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');SEL=read(MODELS/'selection.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical']
    sys.path.insert(0,str(ROOT));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    run.ROOT=ROOT;run.OUT=OUT;run.DIAG=Path(a.diagnosis);run.MODELS=MODELS;run.SCHEMA=SCHEMA;run.SEL=SEL;run.CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];torch.set_num_threads(4);main()
