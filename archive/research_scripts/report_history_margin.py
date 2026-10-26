"""Verify and report the fixed history/recall-margin experiment."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tune_six_models import read,save,sha,metric
from tune_two_boundaries import violations
from tune_two_boundaries import predict
from feature_pipeline import FeatureEncoder
from xgboost import XGBClassifier

def run(root,out):
    for n,h in read(out/'original_hashes.json').items():
        assert sha(root/n)==h,n
        if not n.endswith('.csv'):assert sha(out/'backup'/n)==h,n
    for n,h in read(out/'frozen_hashes.json').items():assert sha(out/n)==h,n
    roles=pd.read_csv(out/'roles.csv');assert not roles.duplicated(['fold','respondent_id']).any() and roles.groupby(['fold','profile_group']).role.nunique().eq(1).all()
    scored=roles[roles.role.eq('score')];assert scored.respondent_id.is_unique
    df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')]
    assert set(scored.respondent_id)==set(tr.respondent_id) and len(roles)==3*len(tr)
    p=np.load(out/'oof.npz');assert np.array_equal(p['respondent_id'],tr.respondent_id) and np.array_equal(p['target'],tr.target)
    s=read(out/'summary.json');d=read(out/'decision.json');protocol=read(out/'protocol.json');rows=[]
    for n in ['baseline']+list(s['candidates']):
        probs=p[n];assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-6)
        m=metric(p['target'],probs,p[n+'_pred']);expected=s['baseline'] if n=='baseline' else s['candidates'][n]['metrics'];assert m['confusion_matrix']==expected['confusion_matrix']
    for f in sorted(out.glob('fold_*.json')):
        r=read(f);rows.append(r)
        if r['policy']['feasible']:assert max(violations(r['policy']['metrics'],r['margin_reference']))<=1e-12
        for i,extra in [(1,.005),(2,.02)]:
            expected=min(1,(max(.6,r['calibration_reference']['recall'][2]) if i==2 else r['calibration_reference']['recall'][1])+extra)
            assert abs(r['margin_reference']['recall'][i]-expected)<1e-12
    checkpoint_verified=False
    if (out/'frozen_candidate.json').exists():
        frozen=read(out/'frozen_candidate.json')
        for n,h in frozen['hashes'].items():assert sha(out/n)==h,n
        va=df[df.split.eq('validation')];raw=pd.read_csv(root/'experiments/predictor_audit_20261025/same_respondent_raw_candidates.csv',dtype=str,keep_default_na=False).set_index('respondent_id')
        dep=pd.to_numeric(raw.loc[va.respondent_id,'ADDEPEV3'],errors='coerce').to_numpy();dep=np.where(np.isin(dep,[1,2]),dep,-1)
        enc=FeatureEncoder.restore(read(out/'encoder.json')['base']);x=np.column_stack([enc.transform(va)]+[(dep==c).astype('float32') for c in [1,2,-1]]).astype('float32')
        model=XGBClassifier();model.load_model(out/'candidate_xgb.json');vp=model.predict_proba(x);vm=metric(va.target.to_numpy(),vp,predict(vp,frozen['thresholds']))
        assert vm['confusion_matrix']==read(out/'validation.json')['policy']['metrics']['confusion_matrix'];checkpoint_verified=True
    save(out/'verification.json',dict(passed=True,originals_and_prior_experiment_unchanged=True,backups_verified=True,frozen_protocol_verified=True,no_group_overlap=True,training_only_cv=True,each_training_respondent_scored_once=True,margin_targets_verified=True,saved_metrics_reproduced=True,candidate_reload_validation_confusion_matrix_matches=checkpoint_verified))
    pct=lambda x:f'{100*x:.2f}%'
    def row(n,m):return f"| {n} | {pct(m['accuracy'])} | {pct(m['recall'][2])} | {pct(m['recall'][1])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['macro_auc']:.4f} |"
    lines=['# Depression history with stronger high-stress weights and recall margins','',
      f"Decision: **{d['status']}**. Current production data and models are unchanged. Previous experiments and their reported results were preserved.", '',
      '## Frozen experiment','',
      'The same 19 original predictors plus ADDEPEV3 depression history were used. Its yes/no/unknown categories remain distinct. No mental-health-days, current life satisfaction or target-derived fields were added. This is concurrent stress-frequency assessment; diagnosis timing is unknown and no future-prediction claim is made.', '',
      'Four high-class weight factors were declared before fitting: 1.00 (margin-only control), 1.10, 1.25 and 1.50. Each multiplies the high-class square-root inverse-frequency training weight. Other classes, XGBoost settings and labels are unchanged. Calibration, validation and holdout distributions are not reweighted.', '',
      'During threshold calibration, high recall must reach max(60%, reference high recall) + 2 percentage points. Sometimes recall must reach reference sometimes recall + 0.5 percentage points. Targets are capped at 100%. Macro-F1 and QWK must also be preserved. These margins are cushions, not guarantees on new respondents.', '',
      'Three stratified profile-group folds use all 128,248 original training respondents. Each outer training part reserves grouped20% calibration. Preprocessing fits only fitting rows. All candidates use identical role assignments. Outer scoring rows never choose thresholds. Fresh fold-refitted 19-input XGBoost with its existing decision multipliers is the reference.', '',
      'Eligible candidates must improve pooled accuracy and preserve both recalls/F1/QWK, with feasible calibration and preserved safeguards in every scoring fold. Select one CV winner by accuracy, macro-F1, then lower weight; no switching after validation or test.', '',
      '## Training-only out-of-fold comparison','', '| Model / high-class weight factor | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|',row('Original19 reference',s['baseline'])]
    for n,r in s['candidates'].items():lines.append(row('History, factor '+n,r['metrics']))
    lines += [f"| Always rarely | {pct(s['majority_accuracy'])} | 0% | 0% | — | — | — |",'',
      'Historical reused internal-holdout always-rarely baseline:65.78% accuracy and zero high/sometimes recall. It is not directly comparable with training-fold estimates.', '',
      '| Factor | Accuracy change (points) | Extra high cases detected | Extra sometimes cases detected | Eligible for refit? |','|---|---:|---:|---:|---|']
    b=s['baseline']
    for n,r in s['candidates'].items():
        m=r['metrics'];lines.append(f"| {n} | {100*(m['accuracy']-b['accuracy']):+.3f} | {m['high_detected']-b['high_detected']:+d} | {m['confusion_matrix'][1][1]-b['confusion_matrix'][1][1]:+d} | {r['eligible']} |")
    lines += ['', '## Fold safeguards','', '| Fold | Factor | Calibration feasible | Accuracy change (points) | High recall change (points) | Sometimes recall change (points) | Scoring safeguards pass |','|---|---|---|---:|---:|---:|---|']
    for r in rows:
        m=r['candidate'];b=r['baseline'];lines.append(f"| {r['fold']+1} | {r['factor']} | {r['policy']['feasible']} | {100*(m['accuracy']-b['accuracy']):+.3f} | {100*(m['recall'][2]-b['recall'][2]):+.3f} | {100*(m['recall'][1]-b['recall'][1]):+.3f} | {r['passes']} |")
    if (out/'validation.json').exists():
        v=read(out/'validation.json');lines += ['', '## One frozen winner on reused validation','',f"Selected high-class factor: {v['factor']}. This validation set was historically reused and chooses the final threshold; it is not independent confirmation.", '', '| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|']
        for n,r in v['baselines'].items():lines.append(row('Current '+n,r))
        lines += [row('History plus weight/margin candidate',v['policy']['metrics']),f"| Always rarely | {pct(v['majority_accuracy'])} | 0% | 0% | — | — | — |",'',f"High recall calibration target: {pct(v['margin_reference']['recall'][2])}. Sometimes target: {pct(v['margin_reference']['recall'][1])}. Threshold feasible: {v['policy']['feasible']}. Validation eligible: {v['eligible']}."]
    if (out/'final_evaluation.json').exists():
        f=read(out/'final_evaluation.json');lines += ['', '## One-shot frozen final evaluation','', '| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|']
        for n,r in f['comparisons'].items():lines.append(row('Current '+n,r['baseline']))
        lines += [row('Frozen candidate',f['candidate']),f"| Always rarely | {pct(f['majority_accuracy'])} | 0% | 0% | — | — | — |",'', 'Paired stratified-bootstrap differences and final gates are in final_evaluation.json. This is a repeatedly reused internal holdout, not external confirmation.']
    lines += ['', '## Decision and limitations','',
      f"Validation evaluated: {d['validation_evaluated']}. Holdout evaluated: {d['test_evaluated']}. Production changed: {d['production_changed']}.", '',
      'A previous holdout result motivated this hypothesis. The new selection used training/validation only, but this does not make a historically reused holdout untouched. No particular missed respondents were targeted, and no scoring or test rows were dropped to raise accuracy.', '',
      'A calibration recall cushion can trade accuracy for detection and may not survive new respondents. A failure retains the original model without trying a different candidate on the holdout. A passing internal result would still need input-schema, API/frontend, 512MB runtime and rollback checks before local use. No deployment is authorized here.', '',
      '## Reproducible files','',
      '- protocol.json, frozen_hashes.json, source/: fixed candidates, selection rules and source snapshots.',
      '- backup/, original_hashes.json: current and previous-experiment preservation.',
      '- roles.csv and oof.npz: grouped memberships, probabilities and frozen per-fold decisions.',
      '- fold_*.json and summary.json: raw argmax and decision metrics, confusion matrices, per-class precision/recall/F1, AUC and timing.',
      '- selected.json and validation.json, when present: sole CV winner and subsequent validation outcome.',
      '- decision.json and verification.json: final status and integrity checks.', '',
      'No production rollback is necessary while production_changed is false.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(verification='passed',decision=d,summary=s)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
