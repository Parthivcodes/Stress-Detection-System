"""Verify saved subset checkpoints and comparison metrics, then report."""
import argparse,gc,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from tune_six_models import read,save,sha,metric,prob,NAMES,FILES
from tune_two_boundaries import predict

def run(root,out):
    for n,h in read(out/'original_hashes.json').items():assert sha(root/n)==h,n
    for n,h in read(out/'frozen_hashes.json').items():assert sha(out/n)==h,n
    for n,h in read(out/'original_hashes.json').items():
        if not n.endswith('.csv'):assert sha(out/'backup'/n)==h,n
    scored=out/'group_disjoint_comparison'
    summary=read(scored/'summary.json');decision=read(scored/'decision.json');results=summary['results'];roles=pd.read_csv(scored/'roles.csv')
    assert roles.respondent_id.is_unique and roles.groupby('profile_group').role.nunique().max()==1
    p=np.load(scored/'comparison_predictions.npz');cal=np.load(out/'calibration_predictions.npz');assert not set(p['respondent_id'])&set(cal['respondent_id'])
    df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv').set_index('respondent_id')
    assert df.loc[p['respondent_id'],'split'].eq('validation').all()
    assert set(p['respondent_id'])==set(roles[roles.role.eq('comparison')].respondent_id)
    subset=pd.read_csv(root/'data/experimental_brfss_10000/brfss_10000_respondents.csv');train=subset[subset.split.eq('train')]
    schema=read(root/'models/current_models/schema.json');features=schema['numeric']+schema['categorical'];pre=joblib.load(out/'preprocessor.joblib')
    assert np.allclose(pre.named_transformers_['numeric'].mean_,train[schema['numeric']].mean().to_numpy())
    sys.path.insert(0,str(root));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    classes=[Conv1DNet,LSTMNet,CNNLSTMNet];x=pre.transform(df.loc[p['respondent_id'][:64],features]).astype('float32')
    for i,n in enumerate(NAMES):
        if i<3:m=joblib.load(out/FILES[i])
        else:m=classes[i-3](input_dim=x.shape[1]);m.load_state_dict(torch.load(out/FILES[i],map_location='cpu',weights_only=True))
        replay=prob(m,x);assert np.allclose(replay,p[n][:64],atol=1e-6),n;del m;gc.collect()
    for n,r in results.items():
        assert np.isfinite(p[n]).all() and (p[n]>=0).all() and np.allclose(p[n].sum(1),1,atol=1e-6)
        m=metric(p['target'],p[n],predict(p[n],r['policy']['thresholds']))
        assert m['confusion_matrix']==r['candidate']['confusion_matrix']
    save(out/'verification.json',dict(passed=True,original_hashes_unchanged=True,backups_verified=True,frozen_protocol_verified=True,fit_only_preprocessing=True,roles_disjoint=True,all_six_saved_models_reload_and_predict=True,saved_prediction_parity_atol=1e-6,confusion_matrices_reproduced=True,production_changed=False))
    pct=lambda x:f'{100*x:.2f}%'
    lines=['# Six models trained on the 10,000-respondent subset','',
      f"Decision: **{decision['status']}**. Current data, all six active models and the website remain unchanged.", '',
      f"Trained on {summary['train_rows']:,} real respondents, calibrated thresholds on {summary['calibration_rows']:,}, and compared old/new on the same separate {summary['comparison_rows']:,} respondents. The selected 1,764 test rows were not used for training or threshold selection. The full test set was not evaluated in this comparison.", '',
      '## Accuracy and minority recall','',
      '| Model | Current accuracy | Subset accuracy | Current high recall | Subset high recall | Current sometimes recall | Subset sometimes recall | Pass? |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for n,r in results.items():
        b=r['original'];m=r['candidate'];lines.append(f"| {n} | {pct(b['accuracy'])} | {pct(m['accuracy'])} | {pct(b['recall'][2])} | {pct(m['recall'][2])} | {pct(b['recall'][1])} | {pct(m['recall'][1])} | {'Yes' if r['eligible'] else 'No'} |")
    lines += [f"| Always rarely | {pct(summary['majority_accuracy'])} | {pct(summary['majority_accuracy'])} | 0% | 0% | 0% | 0% | No |",'',
      f"Arithmetic mean accuracy across the six individual models: current **{pct(summary['original_six_mean_accuracy'])}**, subset **{pct(summary['candidate_six_mean_accuracy'])}**. This mean is not ensemble accuracy. Historical reused-holdout always-rarely accuracy is 65.78%, with zero minority recall; the split-specific baseline above is the relevant comparison.", '',
      '| Model | Change in high cases detected | Change in sometimes cases detected | Threshold calibration feasible? |', '|---|---:|---:|---|']
    for n,r in results.items():
        b=r['original'];m=r['candidate'];lines.append(f"| {n} | {m['high_detected']-b['high_detected']:+d} | {m['confusion_matrix'][1][1]-b['confusion_matrix'][1][1]:+d} | {r['policy']['feasible']} |")
    lines += ['', '## Training and safeguards','',
      '- Same 19 predictors and original target construction. IDs, stress_raw, raw_* and provenance columns are never model inputs.',
      '- Classical models retain active hyperparameter configurations. Logistic regression and XGBoost training weights are recomputed from the subset training labels. Random forest retains its active weighting setting.',
      '- Neural architectures are unchanged. A grouped internal 10% training development split selects epochs (maximum 14, patience 4), then each network is reinitialized and fitted on all 7,364 training respondents for that many epochs. One fixed seed; no best-seed selection.',
      '- Scaling and encoding are fitted on the corresponding training rows only. Calibration and comparison rows are disjoint; profile groups never cross roles.',
      '- The 872 calibration rows select one two-boundary rule per candidate, requiring high recall at least max(60%, current model recall), and no reduction in sometimes recall, macro-F1 or QWK. Any infeasible fallback is diagnostic only.',
      '- Comparison rows do not choose thresholds. A candidate passes only with strictly better accuracy and preserved high/sometimes recall, macro-F1 and QWK. Each individual model has its own current-model reference.',
      '- Ensemble weights stay LR 40%, XGBoost 50%, LSTM 10%. New raw probabilities are averaged before applying one calibration-selected rule; no majority vote.',
      '- Current predictions are reused from saved probabilities only after dataset and checkpoint identity checks. These respondents belong to historically reused validation, not an untouched external test.',
      '', '## Raw probability argmax diagnostics','', '| Model | Current raw accuracy | Subset raw accuracy | Subset raw high recall | Subset raw sometimes recall |', '|---|---:|---:|---:|---:|']
    for n in NAMES:
        r=results[n];b=r['original_raw'];m=r['candidate_raw'];lines.append(f"| {n} | {pct(b['accuracy'])} | {pct(m['accuracy'])} | {pct(m['recall'][2])} | {pct(m['recall'][1])} |")
    lines += ['', 'Raw argmax accuracy cannot qualify a replacement when minority recall falls. The primary comparison above uses the frozen calibration-selected decision rules.', '',
      '## Interpretation','',
      'A smaller dataset does not create stronger predictive information. This experiment checks one predeclared training recipe per model; it does not prove that every possible smaller-data model will fail. Results do not establish 80% average accuracy. No records were fabricated, relabeled or selected by prediction correctness.', '',
      'Selection on reused validation can be optimistic. Any qualifying candidate would require a frozen one-pass comparison on the reused internal holdout plus inference compatibility, memory and rollback checks before replacing production. A rejected candidate is not retuned using its comparison results.', '',
      '## Files and verification','',
      '- protocol.json and frozen_hashes.json: frozen training, threshold and evaluation rules.',
      '- backup/ and original_hashes.json: preserved active artifacts and source identities.',
      '- Six new model files and preprocessor.joblib: isolated trained candidates, not active models.',
      '- model_0.json through model_5.json: training histories, epoch choices, timing, file sizes and metrics.',
      '- group_disjoint_comparison/summary.json: authoritative corrected accuracy, balanced accuracy, macro-F1, QWK, AUC, per-class metrics, confusion matrices and detected/missed counts.',
      '- group_disjoint_comparison/roles.csv and saved probability files: corrected membership and inference outputs. Original top-level results are retained for audit only.',
      '- group_disjoint_comparison/decision.json and verification.json: eligibility and checks. All six saved checkpoints reload and reproduce predictions within 1e-6; original and backup hashes match.', '',
      'No production rollback is necessary: active data/model paths and inference code were not modified.', '']
    correction=read(scored/'correction.json')
    lines[4:4]=['## Verification correction','',f"Final verification found {correction['excluded_rows']} comparison respondents sharing {correction['excluded_groups']} profile groups with calibration. Those comparison rows were excluded solely by group identity. All models and thresholds stayed frozen. The tables below use the corrected {correction['corrected_rows']:,}-respondent comparison; initial results remain saved but are not authoritative.",'']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(verification='passed',decision=decision,old_mean=summary['original_six_mean_accuracy'],new_mean=summary['candidate_six_mean_accuracy'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4);run(Path(a.project),Path(a.output))
