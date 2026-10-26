"""Build readable evidence from frozen results; no training or test inference."""
import contextlib, csv, hashlib, io, json, platform, sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
read = lambda n: json.loads((out / n).read_text())
f = read('frozen_selection.json')
r = read('test_report.json')
features = read('feature_selection.json')
cal = read('calibration.json')
fit = read('fit.json')
rows = []
for split, source in [('validation', f), ('reused_internal_holdout', r)]:
    for model in ['xgb', 'ensemble']:
        item = source[model]
        for version in ['original_actual', 'original_matched', 'baseline', 'candidate', 'raw_argmax', 'calibrated_argmax']:
            if version not in item: continue
            m = item[version]
            if 'metrics' in m: m = m['metrics']
            rows.append(dict(split=split, model=model, version=version,
                **{k:m[k] for k in ['accuracy','balanced_accuracy','macro_f1','qwk','macro_auc','high_detected','high_missed']},
                sometimes_recall=m['recall'][1], high_recall=m['recall'][2]))
with (out/'comparison.csv').open('w', newline='') as stream:
    writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def table(split):
    lines=['| Model / decision | Accuracy | High recall | High missed | Sometimes recall | Macro-F1 | QWK |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['split']!=split:continue
        lines.append(f"| {row['model']} / {row['version']} | {row['accuracy']:.2%} | {row['high_recall']:.2%} | {row['high_missed']} | {row['sometimes_recall']:.2%} | {row['macro_f1']:.4f} | {row['qwk']:.4f} |")
    base=f['majority_validation_accuracy'] if split=='validation' else r['always_rarely']['accuracy']
    lines.append(f'| Always rarely (majority baseline) | {base:.2%} | 0% | All high cases | 0% | — | — |')
    return '\n'.join(lines)

reasons={}
for name in ['xgb','ensemble']:
    reasons[name]=[]
    v=f[name]; a=r[name]['original_actual']; b=r[name]['original_matched']; c=r[name]['candidate']
    if not v['candidate']['eligible']:reasons[name].append('No validation threshold passed all declared accuracy, recall, macro-F1, QWK and middle-recall gates; candidate threshold is a reporting-only fallback.')
    if c['recall'][2]<max(.6,a['recall'][2]):reasons[name].append('Final high-stress recall below the current-model/60% floor.')
    for baseline,label in [(a,'current decision'),(b,'matched decision')]:
        for key in ['accuracy','macro_f1','qwk']:
            if c[key]<baseline[key]:reasons[name].append(f'{key} below {label}.')
        if c['recall'][1]<baseline['recall'][1]:reasons[name].append(f'Sometimes recall below {label}.')
    if r[name]['paired_vs_matched']['accuracy_delta_95ci'][0]<=0:reasons[name].append('Paired accuracy interval does not establish a positive gain over the matched baseline.')
(out/'gate_explanations.json').write_text(json.dumps(reasons,indent=2)+'\n')
lines=['# Feature engineering, broader XGBoost tuning and calibration','',
    '## Outcome','',f"Performance gates passed: **{r['approved_for_runtime_verification']}**. Passing is not itself permission to activate: runtime checks are also required.",
    '', 'This experiment does not modify production artifacts. The other five model families were not retrained in this round. No cloud deployment.',
    '', '## Feature comparison (training-only grouped CV)','',
    '| Representation | Dimensions | Mean log-loss (lower is better) |','|---|---:|---:|']
for mode,item in features['comparisons'].items():lines.append(f"| {mode} | {item['dimensions']} | {item['mean_log_loss']:.6f} |")
lines += ['',f"Selected: **{features['selected']}**. Comparison used identical baseline settings; no feature representation was forced into production.",
    '', 'Ordered encoding keeps six ordered scales numeric. Engineered encoding adds a seven-factor social-risk count, unhealthy-days flag/log, sleep bands and a clipped BMI copy. The original BMI remains. Clipping/scaling is fitted within training folds. These transformations add no respondents or new information; the risk count is not a validated clinical scale.',
    '', '## Search and calibration','',
    '50 sequential Optuna TPE trials; two stratified grouped outer folds, with training-only inner early stopping; 700-round cap and patience 30. Objective: unweighted log-loss. No validation or holdout labels used for hyperparameter search.',
    '', 'Log-loss can improve partly because smaller class weights bring probabilities closer to the observed class distribution. This does not by itself demonstrate stronger discrimination, accuracy at protected recall, or better high-stress detection.',
    '', '12,832 original training respondents reserved before tuning for disjoint temperature fitting/checking. The final model excludes this reserve. This reduces fitting data deliberately to protect calibration independence.',
    '',f"Calibration accepted: **{cal['enabled']}**; applied temperature {cal['temperature']:.6f}. Independent check log-loss {cal['check_raw_logloss']:.6f} → {cal['check_calibrated_logloss']:.6f}; Brier {cal['check_raw_brier']:.6f} → {cal['check_calibrated_brier']:.6f}.",
    '', 'A scalar temperature cannot change argmax labels. A separate high-class cutoff is selected on validation. The ensemble retains LR 40%, XGB 50%, LSTM 10%, with one cutoff; it uses candidate calibrated XGB probabilities only in this isolated experiment.',
    '', 'Selected hyperparameters:', '', '```json',json.dumps(fit['params'],indent=2),'```',
    '',f"Final best iteration (zero-based): {fit['best_iteration']}; fitting time {fit['seconds']:.1f}s.",
    '', '## Validation decisions','',table('validation'),
    '',f"High-recall floors: XGB {f['xgb']['floor']:.2%}; ensemble {f['ensemble']['floor']:.2%}. Safeguards require no loss in macro-F1, QWK or sometimes recall against both current and matched-threshold baselines.",
    '', '## One frozen final evaluation','',table('reused_internal_holdout'),
    '', 'The holdout was reused by historical experiments. This is NOT untouched external evidence; its uncertainty intervals do not account for prior adaptive reuse. Never retune against these results. Historical always-rarely accuracy: 65.78%, with 0% high recall.',
    '', 'Gate explanations:', '', '```json',json.dumps(reasons,indent=2),'```',
    '', '## Files and reproduction','',
    '- `protocol.json`, `original_hashes.json`, `source_hashes.json`: frozen design and original integrity. `backup/`: protected copies.',
    '- `training_roles.csv`, `training_cv_roles.csv`: respondent roles and grouped splits.',
    '- `trials/`, `optuna.sqlite3`, `optuna_trials.csv`, `best_trial.json`: all 50 searches.',
    '- `candidate.joblib`, `encoder.json`, `calibration.json`: isolated model and preprocessing/calibration.',
    '- `frozen_selection.json`: thresholds and hashes frozen before evaluation.',
    '- `test_report.json`, `comparison.csv`, `gate_explanations.json`: detailed metrics, confusion matrices and paired intervals.',
    '- `validation_reliability.csv`, `accuracy_recall_curve.csv`, `accuracy_recall.svg`: reliability bins and operating trade-off.',
    '- `Feature_Engineering_and_XGBoost.ipynb`: executed explanation of saved evidence; does not rerun the holdout.',
    '', 'Run `tools/tune_xgb_features.py init/search/select/test --project PROJECT --output NEW_EXPERIMENT` in that order. Each command needs its phase; choose a NEW output for replication. `test` is one-shot and deliberately refuses to rerun. Supply `--optuna-path` only when using an isolated package installation. Keep `feature_pipeline.py` and `tune_six_models.py` beside the tuning script.',
    '', 'Runtime used: existing CPU ML environment plus isolated Optuna 5.0.0. No production dependency replacements. Refer to `environment.json` for exact versions. Existing experiment files must not be overwritten.',
    '', '## Rollback and viva summary','',
    'If the candidate is rejected, no rollback is needed: current data/models/configuration were never replaced. Do not copy candidate.joblib into current_models; its encoder and calibration differ. Backups retain original relative paths and SHA-256 hashes.',
    '', '“We tested better representations of the same real survey inputs, searched 50 XGBoost configurations using training-only grouped cross-validation, checked probability calibration on separate respondents, and chose a validation threshold that protects high-stress detection. We froze the candidate before the final reused-holdout evaluation and kept the existing model whenever the safety gates failed.”']
trials=[json.loads(p.read_text()) for p in (out/'trials').glob('*.json')]
lines += ['', '## Runtime and preservation','',
    f"Completed trials: {len(trials)}; total trial fit/evaluation time {sum(x['seconds'] for x in trials):.1f}s. Candidate artifact {(out/'candidate.joblib').stat().st_size/1048576:.2f} MiB."]
if (out/'runtime_verification.json').exists():
    rt=read('runtime_verification.json')['summary']
    lines += ['',f"Unchanged active backend: {rt['requests']} validation requests / {rt['model_predictions']} model outputs passed; {rt['seconds']:.1f}s for the check; peak {rt['memory_peak_bytes']/1048576:.1f} MiB under a 512 MiB cap with swap disabled. No OOM kill. Invalid input/authentication checks and explicit overload backpressure passed. This is local verification, not a cloud deployment."]
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Plot saved validation curve, not new inference.
curve=list(csv.DictReader((out/'accuracy_recall_curve.csv').open()))
points=' '.join(f"{60+float(x['high_recall'])*600:.2f},{350-float(x['accuracy'])*300:.2f}" for x in curve)
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430"><rect width="760" height="430" fill="white"/><g font-family="sans-serif" fill="#222"><text x="60" y="28" font-size="19">Validation: accuracy versus high-stress recall</text><path d="M60 50 V350 H660" fill="none" stroke="#555"/><polyline points="{points}" fill="none" stroke="#17677e" stroke-width="2"/><text x="280" y="405">High-stress recall (0 to 1)</text><text x="8" y="45">Accuracy</text><text x="40" y="355">0</text><text x="40" y="55">1</text><text x="58" y="375">0</text><text x="650" y="375">1</text></g></svg>'''
(out/'accuracy_recall.svg').write_text(svg)

import importlib, importlib.metadata as md
versions={}
for package in ['numpy','pandas','scipy','scikit-learn','xgboost','torch','joblib']:
    try: versions[package]=md.version(package)
    except md.PackageNotFoundError:
        versions[package]=importlib.import_module('sklearn' if package=='scikit-learn' else package).__version__
versions.update(python=platform.python_version(),optuna=read('optuna_version.json')['optuna'])
(out/'environment.json').write_text(json.dumps(versions,indent=2)+'\n')

# Execute explanatory notebook cells in order and capture actual outputs.
cells=[];scope={'EXPERIMENT':out};count=0
def markdown(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(source):
    global count
    count+=1;stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(source,scope)
    cells.append(dict(cell_type='code',metadata={},source=source.splitlines(True),execution_count=count,
        outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(True))]))
markdown('# Feature engineering → XGBoost tuning → calibration\nReal saved results. This notebook reads evidence; it does not train again or retune on the reused holdout.')
code("from pathlib import Path\nimport json\nEXPERIMENT = Path.cwd() if (Path.cwd()/'protocol.json').exists() else globals().get('EXPERIMENT')\nassert EXPERIMENT is not None, 'Set EXPERIMENT to the experiment directory.'\nload = lambda name: json.loads((EXPERIMENT/name).read_text())\nprint('Seed:', load('protocol.json')['seed'])\nprint('Trials:', len(list((EXPERIMENT/'trials').glob('*.json'))))")
markdown('## Preprocessing\nOriginal: standardized numeric + one-hot categorical. Ordered: numeric ordered scales, other categories one-hot (drop binary reference). Engineered: seven derived fields. Every learned value is fitted within a training fold. See `tools/feature_pipeline.py` for implementation.')
code("d=load('feature_selection.json')\nfor mode,row in d['comparisons'].items(): print(mode, row['dimensions'], 'columns; CV logloss:', round(row['mean_log_loss'],6))\nprint('Selected:', d['selected'])")
code("import sys\nimport pandas as pd\nsys.path.insert(0,str(EXPERIMENT.parents[1]/'tools'))\nfrom feature_pipeline import FeatureEncoder\nencoder=FeatureEncoder.restore(load('encoder.json'))\nframe=pd.read_csv(EXPERIMENT/'backup/data/expanded_survey_dataset/respondents.csv')\nexamples=frame.loc[frame.split=='train'].head(3)\nx=encoder.transform(examples)\nprint('Three original respondents -> shape:',x.shape)\nprint('Feature order:',encoder.names)\nprint('Training-fitted BMI clipping bounds:',encoder.bmi_limits)\nif encoder.mode=='engineered': print(pd.DataFrame(x[:,-7:],columns=encoder.names[-7:]).to_string(index=False))\ndel frame")
markdown('## Split protection\nSearch/fitting and calibration roles are disjoint. Grouped folds prevent identical feature profiles straddling partitions. Validation selects the final cutoff; the historically reused test is evaluated once after freezing.')
code("import csv\nfrom collections import Counter\nroles=list(csv.DictReader((EXPERIMENT/'training_roles.csv').open()))\nprint(Counter(row['role'] for row in roles))\nprint('Unique respondents:', len({row['respondent_id'] for row in roles}))")
markdown('## Broader tuning\nTwo grouped outer folds; inner early stopping. Unweighted log-loss objective, not accuracy. Training-only class-weight exponent is tuned. No fabricated data or validation resampling.')
code("print(json.dumps(load('best_trial.json')['params'],indent=2))\nprint('Best CV logloss:', load('best_trial.json')['mean_log_loss'])")
markdown('## Calibration and decision rule\nTemperature is fitted on one training-reserve half and checked on the other. It preserves argmax. Predict high if P(high) >= cutoff; otherwise choose the more probable lower class.')
code("c=load('calibration.json')\nfor key in ['enabled','temperature','check_raw_logloss','check_calibrated_logloss','check_raw_brier','check_calibrated_brier']: print(key,c[key])\nf=load('frozen_selection.json')\nfor name in ['xgb','ensemble']: print(name,'floor',f[name]['floor'],'cutoff',f[name]['candidate']['threshold'],'eligible',f[name]['candidate']['eligible'])")
markdown('## Frozen reused-holdout results\nAlways-rarely baseline is included. A validation recall floor is not a guarantee for unseen respondents. Do not rerun selection after inspecting this table.')
code("r=load('test_report.json')\nfor name in ['xgb','ensemble']:\n    for version in ['original_actual','original_matched','candidate']:\n        m=r[name][version]\n        print(name,version,'accuracy',round(m['accuracy'],4),'high recall',round(m['recall'][2],4),'high missed',m['high_missed'])\nprint('Always rarely:',r['always_rarely'])\nprint('Gates passed:',r['approved_for_runtime_verification'])")
code("print(json.dumps(load('gate_explanations.json'),indent=2))")
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version=platform.python_version())),nbformat=4,nbformat_minor=4)
(out/'Feature_Engineering_and_XGBoost.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
print('Report, comparison, plot and executed notebook saved.')
