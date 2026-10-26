"""Exclude calibration profile groups from scoring; never refit or change thresholds."""
import argparse,copy,json
from pathlib import Path
import numpy as np,pandas as pd
from tune_six_models import read,save,metric,NAMES,sha
from tune_two_boundaries import predict,violations

def run(root,out):
    dest=out/'group_disjoint_comparison'
    if dest.exists():raise RuntimeError('Do not overwrite correction.')
    dest.mkdir()
    # This correction is determined solely from IDs/groups, never scores or labels.
    roles=pd.read_csv(out/'roles.csv');calgroups=set(roles[roles.role.eq('threshold_calibration')].profile_group)
    excluded=roles[roles.role.eq('comparison') & roles.profile_group.isin(calgroups)]
    safe=roles[~roles.respondent_id.isin(excluded.respondent_id)].copy()
    assert safe.respondent_id.is_unique and safe.groupby('profile_group').role.nunique().max()==1
    safe.to_csv(dest/'roles.csv',index=False);excluded.to_csv(dest/'excluded_comparison_ids.csv',index=False)
    pp=dict(np.load(out/'comparison_predictions.npz'));keep=~np.isin(pp['respondent_id'],excluded.respondent_id.to_numpy())
    pp={n:p[keep] for n,p in pp.items()};np.savez_compressed(dest/'comparison_predictions.npz',**pp)
    full=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');va=full[full.split.eq('validation')].reset_index(drop=True)
    index=pd.Index(va.respondent_id).get_indexer(pp['respondent_id']);assert (index>=0).all()
    old=dict(np.load(root/'experiments/hyperparameter_tuning_20261023/original_validation.npz'));old={n:p[index] for n,p in old.items()}
    sel=read(root/'models/current_models/selection.json');policy=read(root/'models/current_models/ensemble/policy.json')
    old['Ensemble']=sum(w*old[n].astype(float) for n,w in policy['weights'].items());old['Ensemble']/=old['Ensemble'].sum(1,keepdims=True)
    s=copy.deepcopy(read(out/'summary.json'));y=pp['target'];eligible=[]
    for n,r in s['results'].items():
        pred=(old[n]*sel[n]['multipliers']).argmax(1) if n!='Ensemble' else np.where(old[n][:,2]>=policy['high_threshold'],2,old[n][:,:2].argmax(1))
        b=metric(y,old[n],pred);m=metric(y,pp[n],predict(pp[n],r['policy']['thresholds']));r['original']=b;r['candidate']=m
        r['eligible']=bool(r['policy']['feasible'] and m['accuracy']>b['accuracy']+1e-12 and max(violations(m,b))<=1e-12)
        if n!='Ensemble':r['original_raw']=metric(y,old[n]);r['candidate_raw']=metric(y,pp[n])
        if r['eligible']:eligible.append(n)
    s['comparison_rows']=len(y);s['majority_accuracy']=float(np.mean(y==0))
    s['original_six_mean_accuracy']=float(np.mean([s['results'][n]['original']['accuracy'] for n in NAMES]));s['candidate_six_mean_accuracy']=float(np.mean([s['results'][n]['candidate']['accuracy'] for n in NAMES]))
    save(dest/'summary.json',s);save(dest/'decision.json',dict(status='frozen_candidates_need_final_checks' if eligible else 'all_rejected',eligible=eligible,production_changed=False,test_evaluated=False))
    save(dest/'correction.json',dict(reason='Verification detected shared profile groups between calibration and comparison. Removed comparison rows in any calibration group; preserved initial results. No label/performance-based removal, refit, threshold change or alternate seed.',excluded_rows=len(excluded),excluded_groups=excluded.profile_group.nunique(),initial_rows=len(keep),corrected_rows=len(y),thresholds_unchanged=True,models_unchanged=True,initial_summary_sha256=sha(out/'summary.json')))
    print(json.dumps(read(dest/'correction.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
