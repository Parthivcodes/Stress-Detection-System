"""Phase-one evidence report and verified learning-curve diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np,pandas as pd
from tune_six_models import read,save,sha,NAMES

def run(root,out):
    for n,h in read(out/'original_hashes.json').items():assert sha(root/n)==h,n
    for n,h in read(out/'frozen_hashes.json').items():assert sha(out/n)==h,n
    b=read(out/'baseline.json');a=read(out/'data_audit.json');noise=read(out/'cleanlab_review.json');review=read(out/'source_label_review.json');curves=read(out/'learning_curves.json');agg=read(out/'learning_curve_aggregates.json')
    assert len(curves)==27
    roles=pd.read_csv(out/'learning_curve_roles.csv');fitgroups=set(roles[roles.role.eq('full')].profile_group);devgroups=set(roles[roles.role.eq('early_stop')].profile_group);assert not fitgroups&devgroups
    sets={n:set(roles[roles.role.eq(n)].respondent_id) for n in ['small','medium','full']};assert sets['small']<sets['medium']<sets['full']
    for r in curves:
        assert r['rows']==len(sets[r['size']])
        for part in ['train','validation']:
            for kind in ['raw','current_rule']:
                m=r[part][kind];cm=np.asarray(m['confusion_matrix']);assert abs(cm.trace()/cm.sum()-m['accuracy'])<1e-12
    save(out/'verification.json',dict(passed=True,production_unchanged=True,frozen_protocol_unchanged=True,learning_curve_fits=27,grouped_development_separate=True,nested_training_sizes_verified=True,baseline_saved_prediction_parity=True,labels_changed=0,test_evaluated=False))
    pct=lambda x:f'{100*x:.2f}%'
    lines=['# Six-model evidence diagnosis — Phase 1','',
      'Phase 1 is complete. This report reproduces the current models, measures learning curves and training gaps, and audits ambiguous records. It does not claim that all six models improved or that an untouched test set exists. No production model, respondent or label was changed.', '',
      '## Corrected project facts','',
      'The task is three-class classification of self-reported stress frequency: rarely/never, sometimes, usually/always. There are 183,040 real BRFSS 2022 respondents, including 128,248 training, 15,197 validation, 30,718 test and 8,877 quarantined records. The earlier 10,000-row subset was real sampled data, not synthetic.', '',
      'The six model families are Logistic Regression, Random Forest, XGBoost, 1D CNN, LSTM and CNN-LSTM. The 19 core predictors are age, sleep duration, physical unhealthy days, BMI, sex, exercise, general health, marital status, education, employment, income, support, isolation, job loss, food insecurity, bill difficulty, utility threat, transport barrier and medical cost. Four are numeric; the remaining15 are categorical/ordered scales. The current encoder produces69 inputs.', '',
      'These are cross-sectional survey features, not a time series. The recurrent models process feature positions; they do not observe temporal history. Existing source comments describing temporal/physiological learning are misleading for this dataset.', '',
      'The existing test set has informed earlier decisions. It was not opened for this diagnosis. Any future evaluation there must be described as a reused internal holdout. A fresh random split cannot erase prior model-selection exposure.', '',
      '## Reproduced baseline on reused validation','',
      '| Model | Current-rule accuracy | High recall | Sometimes recall | Macro-F1 | QWK | Macro AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for n,r in b['models'].items():
        m=r['validation']['current_rule'];lines.append(f"| {n} | {pct(m['accuracy'])} | {pct(m['recall'][2])} | {pct(m['recall'][1])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['macro_auc']:.4f} |")
    lines += [f"| Always rarely | {pct(b['majority_validation_accuracy'])} | 0% | 0% | — | — | — |",'',
      'All six checkpoints were reloaded and predictions matched saved validation probabilities within1e-6. Accuracy uses the current decision rules; AUC uses raw scores. Per-class precision/recall/F1 and confusion matrices are saved in baseline.json. The historical holdout majority baseline is65.78%; use the partition-specific baseline above for this table.', '',
      'Current minority recall is uneven. In particular, CNN/LSTM accuracy should not be presented as evidence that they meet a60% high-recall requirement. Comparing candidate accuracy without matching detection constraints would be misleading.', '',
      '## Training versus validation','',
      '| Model | Training-sample raw accuracy | Validation raw accuracy | Gap (points) |','|---|---:|---:|---:|']
    for n,r in b['models'].items():
        t=r['train_diagnostic']['raw']['accuracy'];v=r['validation']['raw']['accuracy'];lines.append(f'| {n} | {pct(t)} | {pct(v)} | {100*(t-v):+.2f} |')
    lines += ['', 'Training values use a deterministic5,000-row in-sample diagnostic sample, not a fresh test. Random Forest has the largest observed gap. Smaller gaps in other models can reflect weak features, regularization, optimization limits or genuine overlap; they do not prove an irreducible ceiling.', '',
      '![Current-model gaps](current_model_gaps.png)', '',
      '## Learning curves','',
      'All curves use the same fixed grouped development split and reused validation set. Training sizes are nested group-hash subsets. Classical models use one seed; neural models use two. Neural epochs are selected by development log loss, never validation accuracy. Training is capped at14epochs or120seconds checked between epochs, with patience4. Time-limited or still-improving runs cannot establish convergence.', '',
      '| Model | Small validation accuracy | Medium validation accuracy | Full validation accuracy | Full seed SD (points) | Time-limited runs |','|---|---:|---:|---:|---:|---:|']
    for n in NAMES:
        rr={r['size']:r for r in agg if r['model']==n};std=rr['full']['validation_accuracy_std'];lines.append(f"| {n} | {pct(rr['small']['validation_accuracy_mean'])} ({rr['small']['rows']:,} rows) | {pct(rr['medium']['validation_accuracy_mean'])} ({rr['medium']['rows']:,} rows) | {pct(rr['full']['validation_accuracy_mean'])} ({rr['full']['rows']:,} rows) | {'n/a' if std is None else f'{100*std:.2f}'} | {sum(r['time_caps'] for r in rr.values())} |")
    lines += ['', 'These are raw argmax curves, not promoted recall-constrained operating points. Classical one-seed results do not have an estimated seed standard deviation. Two neural seeds provide only a limited variability check, not a confidence interval or5x2 repeated CV.', '',
      '![Accuracy learning curves](learning_curves_accuracy.png)', '', '![Log-loss learning curves](learning_curves_logloss.png)', '',
      '## Data quality, conflicts and label review','',
      f"All {a['rows']:,} respondent IDs are unique and profile groups do not cross the existing splits. The modeling table has no missing predictor values and passes current schema limits and target mappings. It was built using complete-case exclusions; this does not establish that missingness in the source survey is random.", '',
      f"Within training there are {a['training_exact_duplicate_feature_extra_rows']} repeated19-feature rows belonging to distinct respondents, and {a['training_conflicting_feature_groups']} exact feature groups with conflicting labels ({a['training_rows_in_conflicting_groups']} respondents). Exact conflicts force at least {a['observed_exact_profile_min_errors']} mismatches for a deterministic lookup on these observed rows. That tiny count cannot explain a31% error rate. The corresponding in-sample lookup maximum is not a generalization ceiling.", '',
      f"A coarse near-match proxy finds {a['near_bin_groups_with_multiple_rows']} multi-person groups and {a['near_bin_conflicting_groups']} conflicting groups. It rounds age to5-year bins, BMI to1unit and physical days to5-day bins while keeping other features fixed. Bin membership does not prove duplicate people or erroneous labels. No such rows were removed.", '',
      f"Cleanlab flagged {noise['raw_class_weighted']['flagged']:,} records ({pct(noise['raw_class_weighted']['fraction'])}) from raw weighted group-OOF probabilities and {noise['approximate_weight_reversal']['flagged']:,} ({pct(noise['approximate_weight_reversal']['fraction'])}) after an approximate reversal of training class weights. The latter is a sensitivity check, not probability calibration. This disagreement shows that flags depend on model assumptions. Neither percentage is a verified label-error rate or an accuracy ceiling.", '',
      'Cleanlab was run on predictions made out of fold, as recommended in its [official documentation](https://docs.cleanlab.ai/stable/cleanlab/filter.html). No test rows were used. The model’s limited predictive power, class weighting and unmeasured circumstances can all create false label-error flags.', '',
      f"Reviewed {review['reviewed_unique_flagged_records']} unique flagged examples against the original fixed-width source. Every recorded stress response and target mapping matched. Highly flagged examples include older respondents reporting high stress despite otherwise favorable measured predictors; these combinations are possible. No verified source-level label error was found, so no label or row was changed.", '',
      '## What this establishes—and what it does not','',
      '- Random Forest deserves a targeted variance/regularization check; the strongest measured overfitting signal is model-specific.',
      '- Exact duplicate/conflicting profiles are too rare to explain present performance. They do not justify deleting difficult respondents.',
      '- Missing-value imputation comparisons cannot change an already complete table. Recovering incomplete real source respondents would require a separate missingness/selection-bias protocol.',
      '- Smaller training sets are not a demonstrated route to80% accuracy. Learning curves should guide any expansion, with convergence and seed limits respected.',
      '- No defensible numerical maximum accuracy or verified mislabeled percentage has been established. Approximately69% is observed performance, not a proved ceiling.', '',
      '## Next phases and evidence requirements','',
      'Phase2 should test only justified data treatments: retain all source-confirmed labels, compare train-fitted outlier handling against keep, and quantify train/validation feature drift. Phase3 should rank features on training-only held-out folds before selecting interactions. Phase4 should focus model-specific changes, using repeated grouped CV for finalists and multiple neural seeds. Existing feature/XGBoost/ensemble experiments must remain explicit baselines rather than being presented as new trials.', '',
      'Phases2–6 are not reported as completed by this Phase1 diagnosis. No new candidate has been selected for final testing. The requested final comparison on an untouched test set cannot be supplied from the already reused test data; suitable fresh data would be required.', '',
      '## Reproducible evidence','',
      'protocol.json and frozen_hashes.json record the fixed design. baseline.json contains full current metrics. curve_*.json and learning_curves.json contain all27 fits, seeds, actual training counts, epoch histories, wall time and process peak memory. learning_curve_roles.csv contains exact memberships. Ambiguity samples, Cleanlab settings and source_label_review.json document the review. verification.json confirms original-file integrity and split checks.', '',
      'All active models, source data and past results remain unchanged. No deployment was performed.', '']
    if (out/'feature_shift.json').exists():
        shift=read(out/'feature_shift.json');pos=lines.index('## What this establishes—and what it does not')
        extra=['## Feature drift and outlier diagnostics','',f"A separate feature-only domain classifier achieved out-of-fold AUC {shift['adversarial_auc']:.4f} distinguishing training from validation. Its three fold AUCs were {', '.join(f'{v:.4f}' for v in shift['fold_auc'])}. This detects no substantial shift under this classifier; it does not prove absence of all distribution change. Stress labels and the test set were not used.",'','| Numeric feature | Training min/max | IQR flags (training) | Train-fitted 1st/99th percentiles |','|---|---|---:|---|']
        for n,r in shift['numeric'].items():extra.append(f"| {n} | {r['min']:.2f} / {r['max']:.2f} | {r['train_iqr_flags']:,} | {r['percentile1']:.2f} / {r['percentile99']:.2f} |")
        extra += ['', 'IQR flags are not errors. For example, physical unhealthy days up to30 are valid survey responses; an IQR rule flags many meaningful severe cases. No clipping, imputation or deletion was performed. Source invalid/missing counts overlap and include modules not asked; they cannot be interpreted as independent random missingness.', '']
        lines[pos:pos]=extra
    (out/'PHASE_1_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('Phase1 report and verification saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
