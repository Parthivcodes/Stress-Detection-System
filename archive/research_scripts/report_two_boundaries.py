"""Summarize frozen evidence; never retrain, select thresholds or run inference."""
import contextlib,csv,io,json,sys
from pathlib import Path
out=Path(sys.argv[1]).resolve();read=lambda n:json.loads((out/n).read_text())
f=read('frozen_selection.json');t=read('test_report.json');cv=read('cv_selection.json');b=read('boundary_validation.json')
trials=[json.loads(p.read_text()) for p in sorted((out/'trials').glob('*.json'))]
rows=[]
for model in ['xgb','ensemble']:
    for version,m in [('current',f['original'][model])]+[(name,policies[model]['metrics']) for name,policies in f['candidates'].items()]:rows.append(dict(split='validation',model=model,version=version,**m))
    for version in ['original','candidate','raw_argmax']:rows.append(dict(split='reused_holdout',model=model,version=version,**t[model][version]))
columns=['split','model','version','accuracy','balanced_accuracy','macro_f1','qwk','macro_auc','sometimes_recall','high_recall','high_detected','high_missed']
with (out/'comparison.csv').open('w',newline='') as stream:
    wr=csv.DictWriter(stream,fieldnames=columns);wr.writeheader()
    for row in rows:wr.writerow({k:row['recall'][1] if k=='sometimes_recall' else row['recall'][2] if k=='high_recall' else row[k] for k in columns})
with (out/'trial_history.csv').open('w',newline='') as stream:
    wr=csv.DictWriter(stream,fieldnames=['trial','accuracy','feasible','violation','seconds','mode']);wr.writeheader()
    for row in trials:wr.writerow({k:row['params']['mode'] if k=='mode' else row[k] for k in wr.fieldnames})
def table(split):
    lines=['| Model / version | Accuracy | High recall | Sometimes recall | Macro-F1 | QWK | High missed |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['split']==split:lines.append(f"| {row['model']} / {row['version']} | {row['accuracy']:.2%} | {row['recall'][2]:.2%} | {row['recall'][1]:.2%} | {row['macro_f1']:.4f} | {row['qwk']:.4f} | {row['high_missed']} |")
    base=b['majority_accuracy'] if split=='validation' else t['majority_accuracy']
    lines.append(f'| Always rarely (majority baseline) | {base:.2%} | 0% | 0% | — | — | All high cases |')
    return '\n'.join(lines)
reasons={}
for model in ['xgb','ensemble']:
    old=t[model]['original'];new=t[model]['candidate'];why=[]
    if not f['eligible']:why.append('Frozen candidate did not qualify on validation; diagnostic only.')
    if new['accuracy']<=old['accuracy']:why.append('Accuracy did not improve.')
    for idx,label in [(2,'High-stress'),(1,'Sometimes')]:
        if new['recall'][idx]<old['recall'][idx]:why.append(label+' recall decreased.')
    for key in ['macro_f1','qwk']:
        if new[key]<old[key]:why.append(key+' decreased.')
    if t[model]['paired']['accuracy_delta_95ci'][0]<=0:why.append('Paired 95% interval does not establish a positive accuracy gain.')
    reasons[model]=why
(out/'gate_reasons.json').write_text(json.dumps(reasons,indent=2)+'\n')
lines=['# Two-boundary decisions and constrained XGBoost tuning','',
    f"Performance gate passed: **{t['approved_for_runtime']}**. Frozen configuration: **{f['chosen']}**. Passing this gate still requires runtime/integration checks; this report does not itself activate anything.",
    '', '## Protection and limitations','',
    'Current models, data and relevant inference sources were copied into backup/ and SHA-256 recorded before trials. No source respondents were dropped, duplicated or relabeled. Other five models were not retrained. No cloud deployment.',
    '', 'The final holdout has been reused by earlier experiments. It is NOT an independent test set; bootstrap intervals do not account for adaptive historical reuse. No guarantee about future-person recall is made.',
    '', 'This is a bounded search, not proof that no better model or threshold exists. The high boundary is searched on a finite grid and the two-boundary policy does not generally reproduce the original XGBoost class-multiplier decision regions. Rejected candidates do not establish a ceiling on achievable accuracy.',
    '', '## Method','',
    'First, evaluate new decision boundaries on existing raw model probabilities. Predict high if P(high) >= h; otherwise predict sometimes if P(sometimes)/(P(rarely)+P(sometimes)) >= s, else rarely. Probabilities are not altered to match labels. Ties at either boundary go to the upper class. The high grid uses 81 probability quantiles; the lower boundary search is exact for every distinct conditional probability.',
    '', 'Then search 40 predeclared XGBoost configurations using seeded sklearn ParameterSampler (no new packages). Each of two grouped outer folds has separate model-fitting, early-stopping, threshold-calibration and scoring respondents. Preprocessing is fitted only on the model-fitting portion. The original XGBoost is refitted within each fold, so baseline predictions do not leak full-training labels into scoring.',
    '', 'Primary search objective: highest mean outer-fold accuracy among candidates preserving high recall (at least 60% or that fold baseline), sometimes recall, macro-F1 and QWK on BOTH scoring folds. Calibrate boundaries on the inner calibration subset, never on the scoring fold. Search includes original/engineered features and separate sometimes/high training class weights. Training-only weighting does not change the evaluation class distribution.',
    '', f"Feasible CV configurations: {cv['feasible_trials']} / {len(trials)}. Chosen trial: {cv['candidate']['trial']} (zero-based). Diagnostic-only CV selection: {cv['diagnostic_only']}.",
    '', 'Final fitting excludes a reserved 20% training subset and uses an internal early-stop subset. Existing validation selects the final boundaries. Complete configuration selection requires BOTH XGB and ensemble to improve with no safeguard loss. If none qualifies, the declared least-violation configuration is evaluated only for documentation, never activation.',
    '', '## Validation comparisons','',table('validation'),
    '', '## Frozen final comparison','',table('reused_holdout'),
    '', 'Historical majority baseline: 65.78% accuracy, zero high or sometimes recall. No alternative was selected using these holdout results.',
    '', 'Gate explanations:', '', '```json',json.dumps(reasons,indent=2),'```',
    '', 'Selected training parameters:', '', '```json',json.dumps(cv['candidate']['params'],indent=2),'```',
    '', 'Frozen boundaries:', '', '```json',json.dumps({k:v['thresholds'] for k,v in f['candidates'][f['chosen']].items()},indent=2),'```',
    '', '## Evidence and reproduction','',
    '- `protocol.json`, `configurations.json`, `source_hashes.json`: design and search frozen before trials. `pretrial_fix.md` documents the implementation correction made before any results.',
    '- `backup/`, `original_hashes.json`: safe copies and original hashes.',
    '- `cv_roles.csv`, `final_roles.csv`: respondent roles, no overlap between fit/calibration/scoring groups.',
    '- `trials/`, `cv_baselines.json`, `trial_history.csv`, `cv_selection.json`: complete CV results, including rejected configurations.',
    '- `boundary_validation.json`, `frozen_selection.json`, `test_report.json`, `comparison.csv`: decisions and metrics. The test report includes per-class precision/recall/F1, confusion matrices, AUC, detected/missed counts and paired intervals.',
    '- `candidate.joblib`, `encoder.json`: isolated checkpoint; do not copy into current_models without its compatible inference support.',
    '- `Two_Boundary_Constrained_Tuning.ipynb`: executed notebook reading saved evidence; does not rerun the holdout.',
    '', 'Use the existing WSL ML environment and run tools/tune_two_boundaries.py with phases init, boundaries, search, select, test in order, each with --project PROJECT --output NEW_EXPERIMENT. The init phase refuses an existing output; the test phase has an exclusive one-shot guard. Never overwrite prior experiments or rerun test to choose an alternative.',
    '', f"Total trial fit/scoring time: {sum(row['seconds'] for row in trials):.1f}s. Candidate checkpoint: {(out/'candidate.joblib').stat().st_size/1048576:.2f} MiB.",
    '', 'If rejected, rollback is unnecessary because no active artifact changed. The original models and policy remain authoritative. Runtime verification is reported separately; there has been no Render or Vercel deployment.',
    '', '## Viva explanation','',
    '“We tuned the decision rules and training parameters together, while requiring the model to keep detecting both high-stress and sometimes-stress respondents. We used separate training, calibration and scoring folds to avoid choosing and judging cutoffs on the same people. We only allow a replacement when the frozen results and runtime checks satisfy the declared safeguards.”']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

cells=[];scope={'EXPERIMENT':out};count=0
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):
    global count
    count+=1;stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(s,scope)
    cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=count,outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(True))]))
md('# Two-boundary tuning and constrained XGBoost\nThis notebook reads real saved results; no training or holdout inference is repeated.')
code("import json\nfrom pathlib import Path\nEXPERIMENT=Path.cwd() if (Path.cwd()/'protocol.json').exists() else globals().get('EXPERIMENT')\nassert EXPERIMENT is not None, 'Set EXPERIMENT to the saved experiment folder.'\nload=lambda n:json.loads((EXPERIMENT/n).read_text())\nprint(load('protocol.json')['objective'])")
md('## Decision rule\nHigh if P(high) ≥ h; otherwise sometimes if P(sometimes)/(P(rarely)+P(sometimes)) ≥ s; otherwise rarely. Thresholds do not alter the reported probabilities. Both class recalls are constrained.')
code("f=load('frozen_selection.json')\nfor name,policies in f['candidates'].items():\n    print(name)\n    for model,row in policies.items(): print(model,row['thresholds'],'validation accuracy:',row['metrics']['accuracy'],'recalls:',row['metrics']['recall'],'feasible:',row['feasible'])")
md('## Leakage controls\nEvery fold refits its own preprocessing and baseline. Training, early stopping, threshold calibration and scoring have disjoint profile groups. The score fold does not choose its own thresholds.')
code("import csv\nfrom collections import Counter\nroles=list(csv.DictReader((EXPERIMENT/'cv_roles.csv').open()))\nprint(Counter((r['fold'],r['role']) for r in roles))\nprint('Trials:',len(list((EXPERIMENT/'trials').glob('*.json'))))")
md('## Constrained search\nWe maximize accuracy only among configurations satisfying the recall, macro-F1 and QWK conditions in both outer folds. The default baseline is an explicit trial, with original features and fixed training settings.')
code("c=load('cv_selection.json')\nprint('Feasible trials:',c['feasible_trials'])\nprint('Diagnostic only:',c['diagnostic_only'])\nprint(json.dumps(c['candidate']['params'],indent=2))")
md('## Frozen final evaluation\nThis internal holdout has been reused historically. Treat results as internal evidence, not a claim of external generalization.')
code("t=load('test_report.json')\nfor model in ['xgb','ensemble']:\n    for version in ['original','candidate']:\n        m=t[model][version]\n        print(model,version,'accuracy',round(m['accuracy'],4),'high recall',round(m['recall'][2],4),'sometimes recall',round(m['recall'][1],4),'high missed',m['high_missed'])\nprint('Always rarely accuracy:',t['majority_accuracy'],'both minority recalls: 0')\nprint('Approved for runtime:',t['approved_for_runtime'])")
code("print(json.dumps(load('gate_reasons.json'),indent=2))")
(out/'Two_Boundary_Constrained_Tuning.ipynb').write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=4),indent=2)+'\n')
print('Saved report, CSV comparisons and six executed notebook cells.')
