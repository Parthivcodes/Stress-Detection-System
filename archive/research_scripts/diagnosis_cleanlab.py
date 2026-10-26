"""Review-only confident-learning flags from existing training-only OOF scores."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,'/tmp/stress-diagnosis-20261026')
import cleanlab,numpy as np,pandas as pd
from cleanlab.filter import find_label_issues
from tune_six_models import read,save,sha

def run(root,out):
    if (out/'cleanlab_review.json').exists():raise RuntimeError('Do not overwrite completed review.')
    save(out/'cleanlab_protocol.json',dict(version=cleanlab.__version__,source='Previously saved3-fold group-OOF baseline XGBoost probabilities from original training only.',method='find_label_issues default prune_by_noise_rate,frac_noise1,n_jobs1,rankbyself_confidence. Run raw scores and a fixed training-class-weight reversal sensitivity check. No label/row changes.',limitation='Class-weighted model scores are not calibrated true probabilities. Reversing fit-fold class weights is only an approximate sensitivity analysis, not calibration. Flag fractions are model-dependent review candidates, NOT verified mislabeled fractions or an accuracy ceiling.',docs='https://docs.cleanlab.ai/stable/cleanlab/filter.html'))
    original=root/'experiments/predictor_audit_20261025';tr=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=tr[tr.split.eq('train')].reset_index(drop=True)
    source=np.load(original/'pilot_oof.npz');assert np.array_equal(source['respondent_id'],tr.respondent_id);y=tr.target.to_numpy();p=source['baseline'].astype(float);p/=p.sum(1,keepdims=True)
    roles=pd.read_csv(original/'pilot_roles.csv');lookup=pd.Series(np.arange(len(tr)),index=tr.respondent_id);adjusted=p.copy()
    for fold in range(3):
        fitids=roles[(roles.fold==fold)&roles.role.eq('fit')].respondent_id;scoreids=roles[(roles.fold==fold)&roles.role.eq('score')].respondent_id
        a=lookup.loc[fitids].to_numpy();b=lookup.loc[scoreids].to_numpy();w=(len(a)/(3*np.bincount(y[a],minlength=3)))**.5;adjusted[b]/=w
    adjusted/=adjusted.sum(1,keepdims=True);results={};sets={}
    for name,probs in [('raw_class_weighted',p),('approximate_weight_reversal',adjusted)]:
        ids=find_label_issues(labels=y,pred_probs=probs,return_indices_ranked_by='self_confidence',n_jobs=1)
        sets[name]=set(ids.tolist());results[name]=dict(flagged=int(len(ids)),fraction=float(len(ids)/len(y)),class_counts={str(c):int(np.sum(y[ids]==c)) for c in range(3)})
        sample=tr.iloc[ids[:30]].copy();sample['review_model_class']=probs[ids[:30]].argmax(1);sample['observed_label_probability']=probs[ids[:30],y[ids[:30]]];sample.to_csv(out/f'cleanlab_{name}_sample.csv',index=False)
    results['overlap']=len(sets['raw_class_weighted']&sets['approximate_weight_reversal']);results['rows']=len(y);results['source_sha256']=sha(original/'pilot_oof.npz');results['labels_changed']=0;results['verified_label_error_rate']=None;results['estimated_accuracy_ceiling']=None
    save(out/'cleanlab_review.json',results);print(results,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
