"""Verify completed two-stage diagnostics and write the human-readable report."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tune_six_models import read,save,sha,metric
from experiment_two_stage import probabilities

def run(root,out):
    hashes=read(out/'original_hashes.json')
    for n,h in hashes.items():assert sha(root/n)==h,n
    for n,h in read(out/'frozen_source_hashes.json').items():assert sha(out/n)==h,n
    for n,h in hashes.items():
        if n.endswith('.csv'):continue
        assert sha(out/'backup'/n)==h,n
    # Check soft composition, including first-stage extrema. No hard gate.
    class Fake:
        def __init__(self,p):self.p=np.asarray(p)
        def predict_proba(self,x):return np.column_stack([1-self.p,self.p])
    combined=probabilities((Fake([0,1,.4]),Fake([.7,.25,.75])),None)
    assert np.allclose(combined,[[1,0,0],[0,.75,.25],[.6,.1,.3]])
    df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')].reset_index(drop=True)
    roles=pd.read_csv(out/'roles.csv');assert len(roles)==len(tr)*3 and set(roles.respondent_id)==set(tr.respondent_id)
    assert not roles.duplicated(['fold','respondent_id']).any()
    scored=roles[roles.role.eq('score')];assert len(scored)==len(tr) and scored.respondent_id.is_unique
    joined=roles.merge(tr[['respondent_id','profile_group']],on='respondent_id',validate='many_to_one')
    assert joined.groupby(['fold','profile_group']).role.nunique().eq(1).all()
    p=np.load(out/'oof.npz');assert np.array_equal(p['respondent_id'],tr.respondent_id) and np.array_equal(p['target'],tr.target)
    summary=read(out/'cv_summary.json');decision=read(out/'decision.json');base=summary['baseline']
    for n in ['baseline','unweighted','weighted']:
        probs=p[n];assert probs.shape==(len(tr),3) and np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-6)
        m=metric(p['target'],probs,p[n+'_pred']);expected=base if n=='baseline' else summary['candidates'][n]['metrics']
        assert m['confusion_matrix']==expected['confusion_matrix'] and abs(m['accuracy']-expected['accuracy'])<1e-12
    save(out/'verification.json',dict(passed=True,protected_files_unchanged=len(hashes),backup_hashes_verified=True,frozen_protocol_source_verified=True,probability_composition_verified=True,training_only_roles_verified=True,no_group_leakage=True,saved_metrics_reproduced=True))
    pct=lambda x:f'{100*x:.2f}%'
    def row(label,m):return f"| {label} | {pct(m['accuracy'])} | {pct(m['recall'][2])} | {pct(m['recall'][1])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['macro_auc']:.4f} |"
    lines=['# Two-stage stress classifier experiment','', '## Decision','',
      f"Status: **{decision['status']}**. {decision['reason']}", '',
      'The six active models, preprocessing, ensemble policy, original dataset and website remain unchanged. Active artifacts and relevant sources were backed up with matching SHA-256 hashes. No deployment was performed.',
      '', '## What was tested','',
      '- Stage A: probability of sometimes or high stress.',
      '- Stage B: probability of high stress conditional on sometimes/high; trained only on actual minority-class respondents.',
      '- Combined scores: rarely = 1−A; sometimes = A×(1−B); high = A×B. Both models run for every respondent; no hard first-stage gate.',
      '- Two fixed candidates: no binary class weighting, and inverse-frequency weights raised to 0.5 in each stage. These are raw model scores, not established calibrated population probabilities.',
      '- Original 19 input features, unchanged respondents and class labels. No added predictors, fabricated cases, resampling of evaluation data or time-series claims.',
      '', '## Evaluation safeguards','',
      'Three stratified profile-group folds use the original training split. Each outer training portion reserves a grouped 20% calibration split. Preprocessing is fitted only on fitting rows. All candidates and the multiclass baseline use identical rows. Stage-two fitting uses only the minority subset of the fit portion.',
      '', 'Decision thresholds are chosen on calibration rows only to maximize accuracy while preserving reference high/sometimes recall, macro-F1 and QWK; high recall must also be at least 60%. Scoring folds never select their own thresholds. A candidate must improve pooled accuracy, preserve pooled safeguards and meet safeguards in every scoring fold. Infeasible calibration fallbacks are diagnostic, never eligible.',
      '', 'The comparison baseline is a freshly refitted multiclass XGBoost using the current decision multipliers, not a production model evaluated on its own training rows. These results cannot be substituted for the website’s historical accuracy.',
      '', '## Training-only out-of-fold results','',
      '| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|',row('Multiclass reference',base)]
    for n,r in summary['candidates'].items():lines.append(row('Two-stage '+n,r['metrics']))
    lines += [f"| Always rarely | {pct(summary['majority_accuracy'])} | 0% | 0% | — | — | — |",'',
      'The historical reused internal-holdout always-rarely baseline is 65.78% accuracy with 0% high/sometimes recall. It is quoted for context only; that holdout was not reopened for rejected candidates.',
      '', '| Candidate vs reference | Accuracy change (percentage points) | Extra high cases detected | Extra sometimes cases detected | Eligible? |','|---|---:|---:|---:|---|']
    for n,r in summary['candidates'].items():lines.append(f"| {n} | {r['accuracy_change_points']:+.3f} | {r['high_detected_delta']:+d} | {r['sometimes_detected_delta']:+d} | {'Yes' if r['eligible'] else 'No'} |")
    lines += ['', '## Fold-level checks','', '| Fold | Candidate | Accuracy change (points) | High recall change (points) | Sometimes recall change (points) | Calibration feasible | Scoring safeguards |','|---|---|---:|---:|---:|---|---|']
    total_time=0
    for path in sorted(out.glob('fold_*.json')):
        r=read(path);b=r['baseline'];m=r['score'];total_time+=r['seconds']
        lines.append(f"| {r['fold']+1} | {r['candidate']} | {100*(m['accuracy']-b['accuracy']):+.3f} | {100*(m['recall'][2]-b['recall'][2]):+.3f} | {100*(m['recall'][1]-b['recall'][1]):+.3f} | {r['policy']['feasible']} | {r['passes']} |")
    lines += ['', '## Raw argmax diagnostics','', '| Model | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for n,r in summary['candidates'].items():lines.append(row('Two-stage '+n,r['raw_argmax']))
    lines += ['', 'A higher raw argmax accuracy does not satisfy the request if minority recall falls. The thresholded comparison above is the primary experiment.',
      '', '## Interpretation and limitations','',
      'This bounded experiment evaluates one conditional architecture with two weighting choices. Failure does not prove that no two-stage model can ever improve; it does mean there is no evidence here to replace production. Do not repeatedly tune against these scoring folds or report the best fold as overall performance.',
      '', f"Candidate fitting, calibration and scoring totaled {total_time:.1f} seconds across six pairs, excluding shared baseline fits, setup and file verification. No paid compute or new dependencies were used.",
      '', 'One seed and correlated folds do not establish a confidence guarantee. The baseline is a fold-refitted XGBoost, not the active ensemble. A surviving candidate would still require frozen comparison against the current XGBoost and ensemble on reused validation, one governed final evaluation and backend runtime checks.',
      '', '## Saved files and reproduction','',
      '- protocol.json: predeclared settings, selection rules and limitations.',
      '- backup/, original_hashes.json: preserved current artifacts and source hashes; CSV preserved in place.',
      '- source/, frozen_source_hashes.json: frozen experiment source and protocol hashes.',
      '- roles.csv: exact fitting, calibration and scoring membership.',
      '- oof.npz: training-only probabilities, decisions and respondent IDs.',
      '- fold_*.json, cv_summary.json: thresholds, precision/recall/F1, confusion matrices, detected/missed counts, AUC and timing.',
      '- decision.json, verification.json: outcome and integrity checks.',
      '- tools/experiment_two_stage.py and tools/report_two_stage.py: use the existing WSL ML environment and a new experiment output directory for a rerun. No existing run is overwritten.',
      '', 'No rollback is required because no active file changed. If considering a new experiment, define its hypothesis and safeguards before looking at its evaluation results.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(verification='passed',decision=decision,summary=summary)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
