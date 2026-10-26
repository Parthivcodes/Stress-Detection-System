"""One-shot final evaluation of a frozen qualifying history model. No deployment."""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from xgboost import XGBClassifier
from feature_pipeline import FeatureEncoder
from tune_six_models import read,save,sha,metric,interval
from tune_two_boundaries import predict,violations
import tune_xgb_features as original

def run(root,out):
    assert read(out/'decision.json')['status']=='awaiting_final_evaluation'
    frozen=read(out/'frozen_candidate.json');assert frozen['eligible']
    for n,h in frozen['hashes'].items():assert sha(out/n)==h
    for n,h in read(out/'original_hashes.json').items():assert sha(root/n)==h
    for n,h in read(out/'frozen_hashes.json').items():assert sha(out/n)==h
    with (out/'TEST_STARTED.json').open('x') as f:json.dump(dict(frozen_candidate_sha256=sha(out/'frozen_candidate.json'),note='One pass only; no candidate reselection or retuning.'),f)
    df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');te=df[df.split.eq('test')].copy();y=te.target.to_numpy()
    raw=pd.read_csv(root/'experiments/predictor_audit_20261025/same_respondent_raw_candidates.csv',dtype=str,keep_default_na=False).set_index('respondent_id')
    dep=pd.to_numeric(raw.loc[te.respondent_id,'ADDEPEV3'],errors='coerce').to_numpy();dep=np.where(np.isin(dep,[1,2]),dep,-1)
    enc=FeatureEncoder.restore(read(out/'encoder.json')['base']);base=enc.transform(te);x=np.column_stack([base]+[(dep==c).astype('float32') for c in [1,2,-1]]).astype('float32')
    m=XGBClassifier();m.load_model(out/'candidate_xgb.json');start=time.monotonic();p=m.predict_proba(x);seconds=time.monotonic()-start;new=predict(p,frozen['thresholds']);nm=metric(y,p,new)
    original.ROOT=root;original.MODELS=root/'models/current_models';original.SCHEMA=read(original.MODELS/'schema.json')
    old=original.original_probs(te);policy=read(original.MODELS/'ensemble/policy.json');sel=read(original.MODELS/'selection.json');ep=original.blend(old,policy['weights'])
    preds={'XGBoost':(old['XGBoost']*sel['XGBoost']['multipliers']).argmax(1),'Ensemble':original.decide(ep,policy['high_threshold'])}
    comparisons={};passed=True
    for n,op in preds.items():
        bp=old['XGBoost'] if n=='XGBoost' else ep;b=metric(y,bp,op);ci=interval(y,op,new)
        ok=bool(nm['accuracy']>b['accuracy']+1e-12 and max(violations(nm,b))<=1e-12 and ci['accuracy_delta_95ci'][0]>0)
        comparisons[n]=dict(baseline=b,accuracy_change_points=100*(nm['accuracy']-b['accuracy']),high_detected_delta=nm['high_detected']-b['high_detected'],sometimes_detected_delta=nm['confusion_matrix'][1][1]-b['confusion_matrix'][1][1],paired_bootstrap=ci,passes=ok);passed &= ok
    save(out/'final_evaluation.json',dict(candidate=nm,comparisons=comparisons,passes_performance=bool(passed),rows=len(te),majority_accuracy=float(np.mean(y==0)),inference_seconds=seconds,artifact_bytes=(out/'candidate_xgb.json').stat().st_size,note='Previously reused internal holdout, not untouched external validation. Paired stratified bootstrap is descriptive and does not correct historical selection. No retuning.'))
    np.savez_compressed(out/'final_predictions.npz',respondent_id=te.respondent_id.to_numpy(dtype=str),target=y,candidate=p,candidate_pred=new,baseline_xgb=old['XGBoost'],baseline_ensemble=ep)
    save(out/'decision.json',dict(status='performance_passed_runtime_pending' if passed else 'rejected_on_final_evaluation',production_changed=False,validation_evaluated=True,test_evaluated=True,reason='Runtime/API/UI/rollback checks required before use.' if passed else 'Frozen candidate failed final accuracy/recall/safeguard/uncertainty requirements; retain current production.'))
    for n,h in read(out/'original_hashes.json').items():assert sha(root/n)==h
    print(json.dumps(read(out/'final_evaluation.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4);run(Path(a.project),Path(a.output))
