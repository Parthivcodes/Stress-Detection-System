"""Create a readable audit report from completed diagnostics, without refitting."""
import argparse,csv,json
from pathlib import Path

def build(out):
    read=lambda n:json.loads((out/n).read_text())
    cov=list(csv.DictReader((out/'coverage.csv').open(newline='')))
    cov={r['variable']:r for r in cov if r['split']=='train'}
    spec=read('candidate_spec.json');err=read('validation_errors.json');pilot=read('pilot_summary.json')
    pct=lambda x:f'{100*x:.2f}%'
    lines=['# BRFSS additional-predictor audit — 25 October 2026','',
      '## Outcome and scope','',
      'The current six models, active ensemble, website and datasets were not changed. This experiment audits original BRFSS variables and runs a fixed-setting XGBoost diagnostic on training data only. It does not establish a new production accuracy.',
      '', 'All 183,040 existing respondents matched their original records in the 445,132-row 2022 source. Respondent ID, source row and stress target agreed for every match. The separate extract preserves original codes; missing values are cleaned only in experiment memory.',
      '', '## What the errors reveal','',
      'These are descriptive results on the historically reused validation set, not an untouched test. Each saved model uses its existing decision multipliers; the ensemble averages raw probabilities using its existing weights and cutoff.',
      '', '| Current model | Accuracy | Sometimes recall | High recall | Macro-F1 | QWK | Macro AUC |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for n,m in err['metrics'].items():
        lines.append(f"| {n} | {pct(m['accuracy'])} | {pct(m['recall'][1])} | {pct(m['recall'][2])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['macro_auc']:.4f} |")
    lines += [f"| Always rarely (this validation split) | {pct(err['majority_accuracy'])} | 0% | 0% | — | — | — |",'',
      'Historical reused internal-holdout reference: always rarely = 65.78% accuracy with 0% sometimes/high recall. That historical figure is not a new evaluation; the split-specific baselines above and below are the relevant comparisons.',
      '', '| True class | Cases | Missed by all six | Shared miss rate |', '|---|---:|---:|---:|']
    for c,label in enumerate(['Rarely','Sometimes','High']):
        r=err['class_overlap'][str(c)];lines.append(f"| {label} | {r['n']:,} | {r['all_six_wrong']:,} | {pct(r['all_six_wrong_rate'])} |")
    lines += ['', 'Shared errors suggest missing predictive information or ambiguous classes. They do not prove an irreducible ceiling. An oracle choosing the correct model per respondent is not a deployable ensemble.',
      '', '## Additional fields and training coverage','',
      'All 39 candidates were declared before their associations were examined. Two fixed groups were tested: six context fields (25 total inputs), then those six plus 14 physical-health and three lifestyle fields (42 total inputs). Coverage below means a usable official response, not merely a nonblank value.',
      '', '| Variable | Meaning | Usable training responses | Group |', '|---|---|---:|---|']
    for n,s in spec.items():
        r=cov[n];label=' '.join(r['label'].split()).replace('|','/')
        lines.append(f"| {n} | {label} | {pct(float(r['valid_rate']))} | {s['group']} |")
    lines += ['', '### Interpretation and missingness','',
      '- Housing tenure, children, healthcare access, insurance and SNAP receipt supply additional household context. US insurance categories and SNAP are not directly transferable to an Indian questionnaire.',
      '- Physical history and disability questions add sensitive information and form burden; they are available at survey time, but this cross-sectional dataset does not establish causality or prediction of future stress.',
      '- All training respondents remain in the pilot. Categorical missing values receive an explicit category; the number of children uses a fit-fold median and a missing indicator. Complete-case filtering would unnecessarily discard respondents.',
      '- Missing codes are variable-specific: insurance codes 7, 9 and 88 are valid; checkup code 8 means never; children code 88 means zero. Refusal/unknown and blank responses are not relabeled as healthy answers.',
      '- Conditional asthma, smoking and alcohol follow-ups require skip-aware handling. Alcohol frequency uses weekly/monthly units and requires further plausibility checks. Caregiving has only 24.68% usable training responses and caregiving hours 4.66%; these are optional/conditional questions. These audit-only fields were not modeled.',
      '- MENTHLTH explicitly includes stress in the same reporting period. It and previously excluded life satisfaction/depression fields remain excluded, along with concurrent mental/emotional limitation fields. Not every excluded variable is literally derived from the target: some raise overlap, sensitivity or task-definition concerns instead.',
      '- The existing unhealthy_days input is PHYSHLTH (physical unhealthy days), not MENTHLTH. Strong univariate association is not proof of added predictive value.',
      '', '## Fixed training-only pilot','',
      'Three stratified group folds use the original 128,248 training respondents. Each outer training portion reserves a grouped inner calibration split. Preprocessing is fitted only on the fit portion. All three variants use identical rows and XGBoost settings. Calibration chooses decision multipliers; outer scoring labels never choose them.',
      '', 'The reference is a freshly fitted, fixed-setting 19-input XGBoost with the original decision multipliers, not the deployed model evaluated on these same training rows. An added-fields result cannot be compared directly with the website’s historical holdout accuracy.',
      '', '| Training OOF variant | Accuracy | Sometimes recall | High recall | Macro-F1 | QWK | Macro AUC |', '|---|---:|---:|---:|---:|---:|---:|']
    b=pilot['baseline_fixed']
    def row(n,m):return f"| {n} | {pct(m['accuracy'])} | {pct(m['recall'][1])} | {pct(m['recall'][2])} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['macro_auc']:.4f} |"
    lines.append(row('19 inputs, original decision rule (reference)',b))
    for n,v in pilot['variants'].items():lines.append(row(n+' — calibration-selected rule',v['constrained']))
    lines += [f"| Always rarely (training OOF) | {pct(pilot['majority_accuracy'])} | 0% | 0% | — | — | — |",'',
      '| Variant vs fixed reference | Accuracy change (points) | Extra high cases detected | Extra sometimes cases detected | Every fold passes safeguards? |', '|---|---:|---:|---:|---|']
    leads=[]
    for n,v in pilot['variants'].items():
        lines.append(f"| {n} | {100*v['accuracy_delta_vs_fixed_baseline']:+.3f} | {v['high_detected_delta']:+d} | {v['sometimes_detected_delta']:+d} | {'Yes' if v['all_folds_pass'] else 'No'} |")
        if n!='baseline' and v['accuracy_delta_vs_fixed_baseline']>0 and v['high_detected_delta']>=0 and v['sometimes_detected_delta']>=0 and v['all_folds_pass']:leads.append(n)
    lines += ['', 'A fold passes only when its inner calibration is feasible and outer high recall is at least the higher of 60% and the reference, with sometimes recall, macro-F1 and QWK also preserved. Infeasible fallback operating points are diagnostic failures, not eligible candidates.',
      '', '### Same-rule comparison to separate new features from decision changes','', '| Variant, original decision multipliers | Accuracy | Sometimes recall | High recall | Macro-F1 | QWK | Macro AUC |', '|---|---:|---:|---:|---:|---:|---:|']
    for n,v in pilot['variants'].items():lines.append(row(n,v['fixed_policy']))
    f=pilot['variants']['all_safe']['fixed_policy']
    lines += ['', f"At the unchanged decision rule, 23 extra fields gave a pooled {100*(f['accuracy']-b['accuracy']):+.3f} percentage-point accuracy change, {f['high_detected']-b['high_detected']:+d} high cases and {f['confusion_matrix'][1][1]-b['confusion_matrix'][1][1]:+d} sometimes cases detected. This is a small exploratory signal, not a qualifying replacement. Fold 0 missed five more high cases; fold 1 missed eight more sometimes cases. The calibrated operating points did not retain the accuracy gain.",
      '', 'Raw argmax results, per-class precision/F1, confusion matrices, detected/missed counts and per-fold timing are in pilot_summary.json and pilot_<fold>_<variant>.json. These descriptive diagnostics have no population weighting or confidence guarantee. One seed and three correlated folds cannot establish a reliable small improvement.',
      '', '## Decision and next step','']
    if leads:
        lines.append('Potential training-only leads meeting the declared screen: '+', '.join(leads)+'. Keep production unchanged. A further frozen comparison against the current model/ensemble on validation, followed by a separately governed final evaluation, is required before any replacement. Reduce question burden only through a new predeclared training-only experiment, not by selecting this pilot’s best scoring fold.')
    else:
        lines.append('No added-field group passed the declared improvement screen. Keep the current models. These results do not justify increasing the website’s advertised accuracy or adding 23 questions to its form. A further experiment would need a new hypothesis and frozen protocol; repeatedly adjusting against these scoring folds would overfit the analysis.')
    lines += ['', 'Mental-health/depression/satisfaction inputs would require an explicit change of task and questionnaire rationale. Their stronger associations alone do not justify adding them to inflate accuracy. Fewer respondents, deleting difficult cases, changing label balance or relabeling missing values would not be a fair improvement.',
      '', '## Reproduction and files','',
      '- protocol.json and candidate_spec.json: frozen experimental rules and variable-specific coding.',
      '- original_hashes.json and verification.json: protected-file hashes and checks.',
      '- codebook_catalog.json: locally supplied official BRFSS codebook metadata.',
      '- same_respondent_raw_candidates.csv and extraction.json: separate unchanged-code extract and join checks; the original dataset remains intact.',
      '- coverage.csv, training_associations.csv: availability and descriptive training-only associations.',
      '- validation_errors.json, validation_error_overlap.csv, validation_subgroups.csv: all-six error analysis on reused validation data. Small subgroup recalls are suppressed below 30 class cases.',
      '- pilot_roles.csv: fit/calibration/score membership; pilot_oof.npz: training-only out-of-fold probabilities and decisions.',
      '- pilot_summary.json and pilot_<fold>_<variant>.json: metrics and decision settings. pilot_feature_gain.csv is fitted gain importance, not causal evidence.',
      '- Reproduction scripts: audit_brfss_predictors.py (catalog/extract), analyze_predictor_audit.py (audit/pilot), verify_predictor_audit.py, report_predictor_audit.py. Use a fresh experiment directory and the existing ML environment; extraction runs in Windows Python, analysis in the existing WSL environment.',
      '', 'No rollback is necessary: no production artifact or active pointer was changed. The audit kept raw observations separate from derived diagnostics, following the data-analysis skill’s preservation guidance.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report saved:',out/'REPORT.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();build(Path(a.output))
