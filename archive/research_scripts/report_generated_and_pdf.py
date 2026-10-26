"""Readable deliverables for the isolated synthetic study and local PDF change."""
import argparse,csv,json,hashlib
from pathlib import Path

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main(workspace):
    out=workspace/'experiments/synthetic_simulation_20261026';data=load(out/'data_manifest.json');test=load(out/'SYNTHETIC_test_metrics.json');protocol=load(out/'protocol.json');rows=[]
    for name,obj in test['models'].items():
        for rule,m in obj.items():
            if not isinstance(m,dict):continue
            rows.append(dict(model=name,rule=rule,accuracy=m['accuracy'],macro_f1=m['macro_f1'],qwk=m['qwk'],macro_auc=m['macro_auc'],sometimes_recall=m['recall'][1],high_recall=m['recall'][2],high_detected=m['high_detected'],high_missed=m['high_missed']))
    with (out/'SYNTHETIC_comparison.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Synthetic stress simulation - educational experiment','',
      '**All 10,000 profiles and all labels are invented. No real BRFSS respondents were copied, resampled or modified. These are not patient records and these scores do not establish accuracy on real people.**','',
      '## What was built','',
      'The same 19 input names, valid ranges and category codes are retained. Age, sleep, physical unhealthy days, BMI and 15 categorical inputs are generated using a fixed random seed and hypothetical shared factors. The exact equation and its Gaussian uncertainty are in protocol.json and the generator. Coefficients are assumptions, not validated medical relationships. No clinical endorsement or calibrated probabilities are claimed.','',
      'There are 7,000 training-partition profiles, 1,500 validation profiles and 1,500 synthetic test profiles. Within training, 6,300 fit models and 700 select neural stopping epochs. Encoders are fitted only on the 6,300 fit profiles. All feature profiles and sample IDs are unique. No noise or label changes were made after looking at model results.','',
      '## Measured synthetic test results','',
      f"Majority-class baseline: **{100*test['majority_baseline']['accuracy']:.2f}%** accuracy, **0%** high and sometimes recall. Mean raw-argmax accuracy across six models: **{100*test['mean_six_raw_accuracy']:.2f}%**.",'',
      '| Model | Raw accuracy | Operating accuracy | Operating macro-F1 | Sometimes recall | High recall | High detected / missed |','|---|---:|---:|---:|---:|---:|---:|']
    for name,obj in test['models'].items():
        r=obj['raw'];m=obj['operating'];lines.append(f"| {name} | {100*r['accuracy']:.2f}% | {100*m['accuracy']:.2f}% | {m['macro_f1']:.4f} | {100*m['recall'][1]:.2f}% | {100*m['recall'][2]:.2f}% | {m['high_detected']} / {m['high_missed']} |")
    lines += ['', 'The operating rule was selected on validation: protect raw validation sometimes recall, macro-F1 and QWK, and require high recall at least max(60%, raw high recall). If no feasible policy exists, raw argmax remains and the failed policy is disclosed. Validation constraints do not guarantee matching recalls on test. Full raw/operating precision, recall, F1, AUC, confusion matrices and individual-seed validation variability are saved.','',
      'These numbers cannot be compared with approximately69% on BRFSS as a before/after improvement: populations, label mechanisms and class proportions differ. The fact that this noisy simulation did not produce80% reinforces that generating rows does not guarantee a high score. An easier deterministic simulator can score much higher by construction without becoming more accurate on real people.','',
      '## Generation options','',
      '| Option | Suitable purpose | Limitation |','|---|---|---|',
      '| Simple rule-based benchmark (toy_easy) | Classroom demonstration that all six algorithms can learn a known rule | High scores reflect how easy we made the invented rule |',
      '| Noisy probabilistic simulation (simulation; completed) | Test ambiguity, minority classes and the complete ML pipeline | Unvalidated assumptions; no real-world accuracy claim |',
      '| Harder overlapping simulation (stress_test) | Robustness and failure-case testing | Designed to be difficult, not to maximize accuracy |',
      '| Real-distribution-fitted generator (not implemented) | Investigate training-only augmentation/privacy research | Requires real training data; assess utility on real held-out respondents. Cannot add absent predictive signal by assumption |','',
      'Recommendation: present the real-data model as the empirical study and the synthetic version as a separately labelled simulation. For stronger real-person performance, collect better distinguishing information and an independent external evaluation rather than changing labels to suit an accuracy target.','',
      '## Reproduce','',
      'Use the existing project Python environment with numpy, pandas, scikit-learn, XGBoost, PyTorch and joblib. No new ML packages were installed for this experiment. Choose a fresh output folder; the script refuses overwrite.','',
      '```powershell',
      "python tools/synthetic_stress_experiment.py --project . --output experiments/synthetic_simulation_fresh --rows 10000 --scenario simulation",'```','',
      'The other predeclared modes are toy_easy and stress_test. Changing mode changes the hypothetical label-generating assumptions and is a different simulation study. Do not choose or advertise a mode as an improvement on real stress prediction.','',
      '## Website PDF export','',
      'Download analysis PDF now saves a two-page readable report directly in the browser: summary, combined probability bars, all six individual predictions, all19 answers with category labels, date, and clear limitations. No new backend dependency is required. The build allow-list includes the exporter for future Vercel builds. Existing model artifacts and source datasets remain unchanged. The live online deployment has not been updated.','',
      'A real browser download was exercised on localhost; the resulting PDF was opened, text-checked and rendered on both pages. The saved sample uses the default illustrative UI profile, not a real individual. PDF generation uses no external service.','',
      '## Preservation and rollback','',
      'The synthetic models are stored only in this experiment. Nothing changes models/active_model.json or models/current_models/. Discarding this experiment is not necessary to restore predictions because it was never activated.','',
      'The original frontend files are backed up under backups/analysis_pdf_export_20261026. To revert the download UI, restore its frontend/prediction.js and frontend/index.html and tools/build_vercel.cjs into the corresponding project paths. The unused new frontend/analysis-pdf.js may be left in place. No model rollback is needed.','',
      '## Evidence and interpretation sources','',
      '- [US Census: What are synthetic data?](https://www.census.gov/about/what/synthetic-data.html)',
      '- [UK Government: AI Insights - Synthetic Data](https://www.gov.uk/government/publications/ai-insights/ai-insights-synthetic-data-html)',
      '', 'The report and all numerical results are from this local experiment. External sources support the distinction between synthetic-data utility and real-world evaluation, not the invented coefficients.','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    cells=[]
    def md(t):cells.append(dict(cell_type='markdown',id=f'c{len(cells):03d}',metadata={},source=t.splitlines(True)))
    def code(t):cells.append(dict(cell_type='code',id=f'c{len(cells):03d}',metadata={},source=t.splitlines(True),execution_count=None,outputs=[]))
    md('# Transparent synthetic stress simulation\nAll profiles/labels are invented. Results measure learning an invented mechanism, not real-person accuracy. Production continues to use the protected real-data models.')
    code("from pathlib import Path\nimport json, pandas as pd\nE=Path.cwd()\nif not (E/'protocol.json').exists():\n    E=E/'experiments'/'synthetic_simulation_20261026'\nload=lambda name: json.loads((E/name).read_text())\nprotocol=load('protocol.json')\nprint(protocol['purpose'])")
    md('## 1. Inputs and label generation\nThe19core inputs use the project schema. Shared latent factors create correlated hypothetical profiles. The generator fixes both its label equation and uncertainty before fitting any model. The latent score is not a training input.')
    code("print(protocol['features'])\nprint(protocol['label_equation'])\ndf=pd.read_csv(E/load('data_manifest.json')['csv'])\nprint(df.groupby(['split','target']).size().unstack())\nassert df.sample_id.is_unique\nassert not df.duplicated(protocol['features']).any()")
    md('## 2. Preprocessing and split discipline\nNumeric columns use StandardScaler; categorical columns use a fixed one-hot schema. Each preprocessor is fit on the model-fitting rows only. Independent development rows select neural epochs; validation chooses decision rules; the synthetic test is evaluated after all choices are frozen.')
    code("print(protocol['split'])\nprint(pd.read_csv(E/'training_roles.csv').role.value_counts())\nprint(protocol['fitting'])")
    md('## 3. All six models\nThe existing families and architectures are retained. Neural models use two predetermined seeds and average probabilities. They operate across fixed tabular feature positions, not a time series.')
    code("v=load('validation_metrics.json')\nfor name,obj in v.items():\n    print(name, 'seed accuracies:', [round(s['accuracy'],4) for s in obj['per_seed_raw']], 'policy feasible:', obj['policy_feasible'])")
    md('## 4. Frozen evaluation\nReport raw argmax and the validation-selected operating rule separately. The test is synthetic; no BRFSS holdout was inspected for this experiment. A high score on an easy generator cannot establish real-world accuracy.')
    code("results=pd.read_csv(E/'SYNTHETIC_comparison.csv')\nprint(results.to_string(index=False))\ntest=load('SYNTHETIC_test_metrics.json')\nprint('Synthetic majority baseline:', test['majority_baseline']['accuracy'])\nprint('Real-world accuracy:',test['real_world_accuracy'])\nprint(load('verification.json'))")
    md('## 5. Reproduction and safe separation\nRun the generator/trainer with a fresh output folder. It refuses overwrite and never activates a synthetic model. README.md gives the exact command, alternative scenarios, results and frontend rollback instructions.')
    (out/'Synthetic_Simulation_Explained.ipynb').write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),indent=2),encoding='utf-8')
    # Preserve the prior bounded real-data result without another holdout evaluation.
    b=workspace/'experiments/bounded_evidence_tuning_20261026';summary=load(b/'summary.json');decision=load(b/'decision.json')
    lines=['# Bounded real-data follow-up','',f"Outcome: **{decision['status']}**. None of the predeclared variants passed every fold's accuracy and minority-recall/F1/QWK safeguards. Production was retained. No additional real validation or test evaluation was performed.",'','This two-fold bounded run is not exhaustive5x2CV. Neural fits used two fixed seeds averaged,12epochs maximum and60-second soft caps; these are limited-budget experiments.','',f"Always-rarely training-OOF baseline: {100*summary['majority_accuracy']:.2f}% accuracy,0% high and sometimes recall. Historical reused holdout baseline65.78% is a different partition.",'','| Model | Configuration | Accuracy | Accuracy delta vs matched baseline (pp) | Macro-F1 delta | Sometimes recall delta (pp) | High recall delta (pp) | Qualifies |','|---|---|---:|---:|---:|---:|---:|---|']
    for name,variants in summary['models'].items():
        for config,obj in variants.items():
            m=obj['metrics'];ref=obj['matched_baseline'];lines.append(f"| {name} | {config} | {100*m['accuracy']:.2f}% | {100*(m['accuracy']-ref['accuracy']):+.3f} | {m['macro_f1']-ref['macro_f1']:+.4f} | {100*(m['recall'][1]-ref['recall'][1]):+.3f} | {100*(m['recall'][2]-ref['recall'][2]):+.3f} | {'Yes' if obj['eligible'] else 'No'} |")
    lines+=['','Eligibility also compares the frozen original decision rule and requires passing each scoring fold; pooled gains alone cannot override a failed fold. Full precision,recall,F1,AUC,confusion matrices,seed reports,times and splitIDs are in the JSON/NPZ/roles artifacts. The registered stacking diagnostic was not executed after the user redirected work to synthetic data and PDF export.','']
    (b/'README.md').write_text('\n'.join(lines),encoding='utf-8');print('Synthetic report, comparison table, notebook and prior bounded-run summary saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);a=p.parse_args();main(Path(a.workspace))
