"""Verify coding, identities, fold separation, saved metrics and source preservation."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import analyze_predictor_audit as a

def run(root,out):
    a.ROOT=root;a.OUT=out;a.SPEC=a.read(out/'candidate_spec.json')
    a.SCHEMA=a.read(root/'models/current_models/schema.json')
    a.verify();df,raw,extra=a.load()
    assert a.sha(out/'same_respondent_raw_candidates.csv')==a.read(out/'extraction.json')['candidate_sha256']
    synthetic=pd.DataFrame({n:['','77','99','1','2'] for n in a.SPEC})
    synthetic['CHILDREN']=['','88','99','1','2'];synthetic['PRIMINSR']=['7','9','88','77','99'];synthetic['CHECKUP1']=['8','','7','9','1']
    c=a.clean(synthetic)
    assert c.CHILDREN.isna().tolist()==[True,False,True,False,False] and c.CHILDREN.iloc[1]==0
    assert c.PRIMINSR.iloc[:3].tolist()==[7,9,88] and c.PRIMINSR.iloc[3:].isna().all()
    assert c.CHECKUP1.iloc[0]==8 and c.CHECKUP1.iloc[1:4].isna().all()
    tr=df[df.split.eq('train')].reset_index(drop=True);ids=set(tr.respondent_id)
    roles=pd.read_csv(out/'pilot_roles.csv');assert set(roles.respondent_id)==ids
    assert len(roles)==3*len(tr) and not roles.duplicated(['respondent_id','fold']).any()
    assert roles[roles.role.eq('score')].respondent_id.value_counts().eq(1).all()
    joined=roles.merge(tr[['respondent_id','profile_group']],on='respondent_id',validate='many_to_one')
    assert joined.groupby(['fold','profile_group']).role.nunique().eq(1).all()
    frame=pd.concat([df,extra],axis=1).loc[df.split.eq('train')].head(50).copy()
    e=a.ExtraEncoder(['CHILDREN','PRIMINSR']).fit(frame);before=e.medians.copy()
    altered=frame.copy();altered.CHILDREN=np.nan;altered.PRIMINSR=np.nan
    encoded=e.transform(altered);assert np.isfinite(encoded).all() and e.medians==before
    assert e.base.transform(frame).shape[1]==69
    p=np.load(out/'pilot_oof.npz');assert np.array_equal(p['respondent_id'],tr.respondent_id) and np.array_equal(p['target'],tr.target)
    summary=a.read(out/'pilot_summary.json');mult=np.asarray(a.read(root/'models/current_models/selection.json')['XGBoost']['multipliers'])
    for n in ['baseline','context','all_safe']:
        probs=p[n];assert probs.shape==(len(tr),3) and np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-6)
        m=a.metric(p['target'],probs,p[n+'_pred']);expected=summary['variants'][n]['constrained']
        assert m['confusion_matrix']==expected['confusion_matrix'] and abs(m['accuracy']-expected['accuracy'])<1e-12
        fixed=a.metric(p['target'],probs,(probs*mult).argmax(1))
        assert fixed['confusion_matrix']==summary['variants'][n]['fixed_policy']['confusion_matrix']
    a.verify()
    result=dict(passed=True,protected_files=len(a.read(out/'original_hashes.json')),unique_respondents=len(df),training_oof_rows=len(tr),candidate_extract_hash_verified=True,originals_unchanged=True,checks=['Variable-specific valid/missing codes','Same-person raw extract integrity','Fit-only numeric imputation','69 original encoded features','Training-only pilot roles','Each respondent scored exactly once','No group overlaps across fit/calibration/score within a fold','Saved probabilities finite and normalized','Saved confusion matrices and accuracy reproduced'],limitations='No production changes or deployment; no new test evaluation. Training-only diagnostic is not proof of generalization.')
    a.save(out/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);v=p.parse_args();run(Path(v.project),Path(v.output))
