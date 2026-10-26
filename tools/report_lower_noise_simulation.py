"""Report fixed lower-noise benchmark without selecting new models or policies."""
import argparse,ast,csv,json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import ndtr
from tune_six_models import read,save,sha,NAMES

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [mid-half,mid+half]

def main(root,out):
    assert read(out/'verification.json')['passed']
    for n,h in read(out/'protected_hashes.json').items():assert sha(root/n)==h,n
    for n,h in read(out/'frozen_source_hashes.json').items():assert sha(out/n)==h,n
    frozen=read(out/'frozen_models.json')
    for n,h in frozen['model_hashes'].items():assert sha(out/n)==h,n
    previous=root/'archive/previous_experiments/initial_training';old=read(previous/'generated_test_metrics.json');new=read(out/'test_metrics.json');manifest=read(out/'data_manifest.json');table=[]
    for name in NAMES:
        a=old['models'][name]['raw'];b=new['models'][name]['raw'];cm=np.asarray(b['confusion_matrix']);n=int(cm.sum());fit=read(out/'validation_metrics.json')[name]['per_seed']
        table.append(dict(model=name,previous_noise_accuracy=a['accuracy'],lower_noise_accuracy=b['accuracy'],accuracy_difference_percentage_points=100*(b['accuracy']-a['accuracy']),accuracy_interval95=wilson(int(np.trace(cm)),n),macro_f1=b['macro_f1'],qwk=b['qwk'],macro_auc=b['macro_auc'],sometimes_recall=b['recall'][1],high_recall=b['recall'][2],high_detected=b['high_detected'],high_missed=b['high_missed'],training_seconds=sum(r['fit']['seconds'] for r in fit),seed_validation_accuracy=[r['raw_validation']['accuracy'] for r in fit]))
    save(out/'comparison.json',dict(rows=table,old_majority_baseline=old['majority_baseline'],new_majority_baseline=new['majority_baseline'],mean_accuracy=new['mean_six_accuracy'],population='paired inputprofiles; different targetlabels/noiseassumptions; not a fixedlabel modelcomparison',confidence='Wilson95% intervals for this separate generatedtest only; no multiplecomparison adjustment; neural2seed stability is limited.'))
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    # Known conditional distribution: descriptive explanation, not a fitted competitor.
    df=pd.read_csv(out/manifest['csv']);tr=df[df.split.eq('train')];module=ast.parse((out/'source/simulation_training.py').read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='generate');assign={n.targets[0].id:n.value for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    env={'f':tr,'np':np};env['material']=eval(compile(ast.Expression(assign['material']),'material','eval'),env);score=np.asarray(eval(compile(ast.Expression(assign['signal']),'signal','eval'),env));a=ndtr((-.65-score)/.25);b=ndtr((.85-score)/.25);expected=float(np.column_stack([a,b-a,1-b]).max(1).mean());save(out/'analytic_reference.json',dict(noise_sd=.25,training_profile_expected_optimal_accuracy=expected,note='Known hypothetical generator,not a trainedmodel or realpopulationceiling.'))
    lines=['# Separate lower-noise simulation experiment','',
      '**Generated-data educational benchmark. These scores do not establish accuracy on real people. Existing datasets, models and website selections remain unchanged.**','',
      '## Fixed experiment','',
      'Noise standard deviation was fixed at0.25 before training, compared with1.0 in the previous simulation. The19features,10,000profile IDs, all split assignments and fit/development row order are identical. The hidden-score equation and class boundaries did not change. Labels are different because the noise assumption changed; this is an easier simulation, not evidence that tuning improved prediction on the old ground truth. No noise value was adjusted after reading results.','',
      f"There are6,300model-fitting profiles,700development profiles,1,500validation profiles and1,500test profiles. Labels changed for {sum(manifest['changed_labels_by_split'].values()):,} of the10,000profiles relative to the old simulation. The test was evaluated once after all checkpoints were frozen. Validation selected only neural stopping epochs via a separate development set; the validation metrics themselves selected no settings. All family decisions use raw-probability argmax.",'',
      '## Measured test results','',
      '| Model | Previous noise1.0 | New noise0.25 | New macro-F1 | Sometimes recall | High recall | High detected / missed |','|---|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['model']} | {100*r['previous_noise_accuracy']:.2f}% | {100*r['lower_noise_accuracy']:.2f}% | {r['macro_f1']:.4f} | {100*r['sometimes_recall']:.2f}% | {100*r['high_recall']:.2f}% | {r['high_detected']} / {r['high_missed']} |")
    lines += ['',f"Six-family mean accuracy: **{100*new['mean_six_accuracy']:.2f}%**, versus {100*old['mean_six_raw_accuracy']:.2f}% in the previous higher-noise simulation. This difference reflects different generated labels, not an apples-to-apples training improvement.",'',
      f"Always predicting the training-majority class gives **{100*new['majority_baseline']['accuracy']:.2f}%** on the new test (majorityclass={new['majority_class']}). The prior simulation's always-rarely baseline was **{100*old['majority_baseline']['accuracy']:.2f}%**. Baselines change when class proportions change.",'',
      f"The simulator's known conditional distribution supports approximately **{100*expected:.2f}% expected optimal accuracy** over the training feature profiles at this noise level. This is a mathematical reference under invented assumptions, not a clinical accuracy estimate or strict finite-sample bound.",'',
      '## Settings and stability','',
      'Three classical models use the same configuration-cloning recipe as before. Every neural family uses its existing architecture, two predetermined seeds averaged, AdamW learningrate0.0007, weightdecay0.001, weighted crossentropy, batch256, maximum40epochs, patience6 and a120-second soft cap checked between epochs. All preprocessing fits only on fitting rows. There was no hyperparameter search or post-test adjustment. Neural inputs are ordered tabular feature positions, not temporal histories.','',
      '| Model | Validation accuracy per fixed seed | Total fitting seconds | New test accuracy95% interval |','|---|---|---:|---|']
    for r in table:lines.append(f"| {r['model']} | {', '.join(f'{100*v:.2f}%' for v in r['seed_validation_accuracy'])} | {r['training_seconds']:.1f} | {100*r['accuracy_interval95'][0]:.2f}% to {100*r['accuracy_interval95'][1]:.2f}% |")
    lines+=['','Two seeds give limited stability evidence. Intervals describe generated-test sampling variation and do not cover uncertainty about whether the generator resembles real people. Full precision,recall,F1,QWK,AUC and confusion matrices remain in test_metrics.json.','',
      '## Safety and reproduction','',
      'protected_hashes.json verifies every previous simulation artifact plus the current real-data model bundle, active-model selection and original expanded dataset. frozen_source_hashes.json protects this experiment\'s protocol/source snapshot; frozen_models.json protects checkpoints/preprocessors. All nine checkpoints representing six families were loaded successfully for evaluation. Neither existing dataset was overwritten, no existing checkpoint was replaced and neither website was switched.','',
      'New dataset: model_training_data.csv in this experiment directory. Source, checkpoints, split roles, validation histories, final probabilities and results are stored alongside it. Choose a fresh output directory when reproducing.','',
      '```powershell',
      'wsl -d Ubuntu -u nobody --exec /opt/stress-venv/bin/python -B /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection/tools/train_lower_noise_simulation.py --project /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection --output /mnt/c/Users/parth/OneDrive/Desktop/stressleveldetection/experiments/lower_noise_fresh',
      '```','',
      'These new models remain experimental. A future website integration would need explicit selection of this dataset/version, input and prediction-parity verification and accurate simulation labelling. No rollback is currently necessary because the active versions were not changed.','']
    report='\n'.join(lines)
    report=report.replace('Validation selected only neural stopping epochs via a separate development set; the validation metrics themselves selected no settings.','Development rows selected neural stopping epochs; validation metrics were diagnostic and selected no settings.')
    for before,after in {'at0.25':'at 0.25','with1.0':'with 1.0','The19features,10,000profile':'The 19 features, 10,000 profile','There are6,300model-fitting profiles,700development profiles,1,500validation profiles and1,500test profiles.':'There are 6,300 model-fitting profiles, 700 development profiles, 1,500 validation profiles and 1,500 test profiles.','the10,000profiles':'the 10,000 profiles','noise1.0':'noise 1.0','noise0.25':'noise 0.25','majorityclass=':'majority class=','learningrate0.0007':'learning rate 0.0007','weightdecay0.001':'weight decay 0.001','maximum40epochs':'maximum 40 epochs','patience6':'patience 6','a120-second':'a 120-second','precision,recall,F1,QWK,AUC':'precision, recall, F1, QWK, AUC','and0%':'and 0%'}.items():report=report.replace(before,after)
    (out/'README.md').write_text(report,encoding='utf-8')
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',id=f'c{len(cells)}',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',id=f'c{len(cells)}',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('# Lower-noise simulation\nThis separate generated-data benchmark reduces hidden label randomness. Its higher scores cannot establish real-person accuracy.')
    code("from pathlib import Path\nimport json,pandas as pd\nE=Path.cwd()\nif not (E/'protocol.json').exists(): E=E/'model_training'/'base_models'\nload=lambda n:json.loads((E/n).read_text())\nprint(load('protocol.json')['comparison'])\nprint(load('data_manifest.json'))")
    md('## Same profiles and training recipe\nBoth variants share19features and exact profile/split assignments. Their target labels differ, so changes in scores describe a change in the simulation problem. Scaling and one-hot encoding fit only on the6,300fitting rows.700development rows choose stopping epochs. The test remains unused until checkpoints are frozen.')
    code("print(pd.read_csv(E/'training_roles.csv').role.value_counts())\nprint(load('protocol.json')['fitting'])\nprint(pd.read_csv(E/'comparison.csv').to_string(index=False))")
    md('## Results and preservation\nEvery individual prediction uses raw argmax. The two neural seeds are averaged, not cherry-picked. Read test_metrics.json for full class-level metrics and confusion matrices; comparison.json contains Wilson intervals. Neither existing website nor any current dataset/model was replaced.')
    code("print('Mean test accuracy:',load('test_metrics.json')['mean_six_accuracy'])\nprint('Majority baseline:',load('test_metrics.json')['majority_baseline'])\nprint(load('verification.json'))")
    save(out/'Lower_Noise_Experiment.ipynb',dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5))
    print(pd.DataFrame([{k:r[k] for k in ['model','previous_noise_accuracy','lower_noise_accuracy','high_recall','sometimes_recall']} for r in table]).to_string(index=False));print('Mean:',new['mean_six_accuracy'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.project),Path(a.output))
