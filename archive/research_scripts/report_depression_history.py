"""Verify the historical-depression feature experiment and explain its limits."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tune_six_models import read,save,sha,metric
from feature_pipeline import FeatureEncoder
from tune_two_boundaries import predict
from xgboost import XGBClassifier

def run(root,out):
    for n,h in read(out/'original_hashes.json').items():
        assert sha(root/n)==h,n
        if not n.endswith('.csv'):assert sha(out/'backup'/n)==h,n
    for n,h in read(out/'frozen_hashes.json').items():assert sha(out/n)==h,n
    roles=pd.read_csv(out/'roles.csv');assert not roles.duplicated(['fold','respondent_id']).any()
    assert roles.groupby(['fold','profile_group']).role.nunique().eq(1).all()
    score=roles[roles.role.eq('score')];assert score.respondent_id.is_unique
    frame=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=frame[frame.split.eq('train')]
    assert set(score.respondent_id)==set(tr.respondent_id) and len(roles)==3*len(tr)
    pp=np.load(out/'oof.npz');assert np.array_equal(pp['respondent_id'],tr.respondent_id) and np.array_equal(pp['target'],tr.target)
    s=read(out/'summary.json');d=read(out/'decision.json');review=read(out/'feature_review.json')
    for n,key in [('baseline','baseline'),('depression_history','candidate')]:
        p=pp[n];assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
        m=metric(pp['target'],p,pp[n+'_pred']);assert m['confusion_matrix']==s[key]['confusion_matrix']
    raw=pd.read_csv(root/'experiments/predictor_audit_20261025/same_respondent_raw_candidates.csv',dtype=str,keep_default_na=False).set_index('respondent_id')
    v=raw.loc[tr.respondent_id,'ADDEPEV3'];counts=v.value_counts().to_dict()
    if (out/'final_predictions.npz').exists():
        finalp=np.load(out/'final_predictions.npz');frozen=read(out/'frozen_candidate.json')
        for n,h in frozen['hashes'].items():assert sha(out/n)==h
        te=frame[frame.split.eq('test')];assert np.array_equal(finalp['respondent_id'],te.respondent_id)
        z=te.iloc[:64];dd=pd.to_numeric(raw.loc[z.respondent_id,'ADDEPEV3'],errors='coerce').to_numpy();dd=np.where(np.isin(dd,[1,2]),dd,-1)
        enc=FeatureEncoder.restore(read(out/'encoder.json')['base']);x=np.column_stack([enc.transform(z)]+[(dd==c).astype('float32') for c in [1,2,-1]]).astype('float32')
        model=XGBClassifier();model.load_model(out/'candidate_xgb.json');assert np.allclose(model.predict_proba(x),finalp['candidate'][:64],atol=1e-6)
        assert np.array_equal(predict(finalp['candidate'],frozen['thresholds']),finalp['candidate_pred'])
        assert metric(finalp['target'],finalp['candidate'],finalp['candidate_pred'])['confusion_matrix']==read(out/'final_evaluation.json')['candidate']['confusion_matrix']
    save(out/'verification.json',dict(passed=True,originals_and_backups_unchanged=True,frozen_protocol_verified=True,respondent_ids_verified=True,cv_training_only=True,no_group_overlap=True,each_training_respondent_scored_once=True,confusion_matrices_reproduced=True,frozen_final_checkpoint_parity_verified=d['test_evaluated'],depression_training_raw_counts=counts))
    pct=lambda x:f'{100*x:.2f}%'
    def row(n,m):return f"| {n} | {pct(m['accuracy'])} | {pct(m['recall'][2])} | {pct(m['recall'][1])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['macro_auc']:.4f} |"
    lines=['# Excluded BRFSS feature review: depression history','',
      f"Decision: **{d['status']}**. Production data, all six active models and inference remain unchanged.", '',
      '## Feature review','', '| Field | Decision | Reason | Training usable coverage |','|---|---|---|---:|']
    for n,r in review.items():lines.append(f"| {n} | {r['decision']} | {r['reason']} | {pct(r['training_valid_rate'])} |")
    lines += ['', 'Review source: the project’s locally supplied official BRFSS 2022 codebook (USCODE22_LLCP_102523.HTML), preserved as codebook metadata in feature_review.json. Correlation alone did not determine eligibility.', '',
      'ADDEPEV3 asks whether the respondent was ever told they had a depressive disorder. It is not computed from the stress-frequency label. It is suitable to test for concurrent assessment when the user can supply that history. The survey does not establish diagnosis timing relative to current stress, so this experiment cannot certify a future-prediction task as leakage-free.', '',
      'Codes 1 and 2 represent yes/no; unknown, refusal and missing remain an explicit third category. No respondent was dropped, and no unknown response was turned into no. Depression history is sensitive, and diagnosis access can differ between respondents. A future form would need an optional, accurately worded question and an unknown response.', '',
      'Life satisfaction and childhood household exposure are not automatically target-derived leakage. They were excluded here to keep the experiment narrow and avoid concurrent well-being overlap or optional-module coverage problems. MENTHLTH explicitly includes stress and stays excluded.', '',
      '## Fixed experiment','',
      'Original 128,248 training respondents, three stratified profile-group folds. Same fitting, inner calibration and scoring rows for original 19 inputs versus 20 inputs including depression history. Each outer training fold reserves 20% for calibration. Missing handling and preprocessing are fitted without scoring labels.', '',
      'Fixed XGBoost settings: 350 trees, depth 5, learning rate 0.04, minimum child weight 12, L2 5, row/column sampling 0.85, four CPU threads. Training inverse-frequency weights have exponent 0.5. No broad hyperparameter search, oversampling or changed labels.', '',
      'Thresholds are selected only on inner calibration. They maximize accuracy while preserving the reference high/sometimes recalls, macro-F1 and QWK, with a minimum 60% high recall. Infeasible threshold fallbacks are diagnostic, never eligible. The reference is a fold-refitted XGBoost using the active original decision multipliers, not a production model scored on its own training data.', '',
      '## Training-only out-of-fold comparison','', '| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|',row('Original 19 inputs',s['baseline']),row('Plus depression history, calibration-selected rule',s['candidate']),row('Plus depression history, unchanged rule',s['fixed_rule']),
      f"| Always rarely | {pct(s['majority_accuracy'])} | 0% | 0% | — | — | — |", '',
      'Historical reused internal-holdout always-rarely baseline: 65.78% accuracy, zero high/sometimes recall. It is not directly comparable to these training-fold results.', '',
      f"With the calibration-selected rule, accuracy changed by {100*(s['candidate']['accuracy']-s['baseline']['accuracy']):+.3f} percentage points. High cases detected changed by {s['candidate']['high_detected']-s['baseline']['high_detected']:+d}; sometimes cases detected changed by {s['candidate']['confusion_matrix'][1][1]-s['baseline']['confusion_matrix'][1][1]:+d}.", '',
      '| Fold | Baseline accuracy | New accuracy | High recall change (points) | Sometimes recall change (points) | Calibration feasible | Safeguards pass |','|---|---:|---:|---:|---:|---|---|']
    for file in sorted(out.glob('fold_*.json')):
        r=read(file);b=r['baseline'];m=r['candidate'];lines.append(f"| {r['fold']+1} | {pct(b['accuracy'])} | {pct(m['accuracy'])} | {100*(m['recall'][2]-b['recall'][2]):+.3f} | {100*(m['recall'][1]-b['recall'][1]):+.3f} | {r['policy']['feasible']} | {r['passes']} |")
    if (out/'validation.json').exists():
        val=read(out/'validation.json');lines += ['', '## Reused-validation selection','', '| Comparison | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|',row('Componentwise strict reference',val['reference']),row('Candidate',val['policy']['metrics']),f"| Always rarely | {pct(val['majority_accuracy'])} | 0% | 0% | — | — | — |",'', 'The reference combines the stricter current XGBoost/ensemble acceptance requirements; it is not a single model. Thresholds were selected here, so this is not an independent evaluation.']
    if (out/'final_evaluation.json').exists():
        final=read(out/'final_evaluation.json')
        lines += ['', '## Frozen final evaluation on reused internal holdout','', '| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|']
        for name,r in final['comparisons'].items():lines.append(row('Current '+name,r['baseline']))
        lines += [row('Frozen depression-history candidate',final['candidate']),f"| Always rarely | {pct(final['majority_accuracy'])} | 0% | 0% | — | — | — |",'', '| Compared with | Accuracy change (points) | Extra high detected | Extra sometimes detected | Accuracy change 95% interval (points) | Pass? |','|---|---:|---:|---:|---|---|']
        for name,r in final['comparisons'].items():
            ci=r['paired_bootstrap']['accuracy_delta_95ci'];lines.append(f"| {name} | {r['accuracy_change_points']:+.3f} | {r['high_detected_delta']:+d} | {r['sometimes_detected_delta']:+d} | [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] | {r['passes']} |")
        lines += ['', 'Paired stratified bootstrap intervals are descriptive. They do not correct for historical reuse of this holdout. No thresholds or model settings were changed after this evaluation. A failed gate retains the original model even when another metric improves.']
    lines += ['', '## Decision and limitations','',
      'A replacement requires strictly greater accuracy and preserved high/sometimes recalls, macro-F1 and QWK; training screening also requires safeguards in every fold. Any survivor still needs frozen validation/final evaluation and runtime/UI integration checks. Failure at an earlier gate keeps the current models and avoids opening the test set.', '',
      f"Validation evaluated: {d['validation_evaluated']}. Reused holdout evaluated: {d['test_evaluated']}. Production changed: {d['production_changed']}.", '',
      'This experiment tests one feature and one bounded modeling recipe. A failure does not establish a permanent accuracy ceiling. A gain does not guarantee unchanged recall on new respondents. Data are cross-sectional, complete-case US survey records; no causal, diagnostic or population-weighted claim is made.', '',
      '## Saved evidence','',
      '- feature_review.json: exact codebook wording, exclusions and coverage.',
      '- protocol.json, frozen_hashes.json: predeclared settings and frozen source.',
      '- backup/, original_hashes.json: original active artifacts and identities.',
      '- roles.csv, oof.npz: group-safe training-fold roles and predictions.',
      '- fold_*.json, summary.json: precision/recall/F1, confusion matrices, AUC and detected/missed counts; raw argmax is diagnostic only.',
      '- decision.json, verification.json: outcome and data/model integrity checks.', '',
      'No rollback is needed because active models and data were not changed.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(verification='passed',decision=d,summary=s)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
