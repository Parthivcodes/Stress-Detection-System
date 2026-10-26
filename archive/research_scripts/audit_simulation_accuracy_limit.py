"""Read-only noise diagnosis using the simulator's known conditional distribution."""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import ndtr
from sklearn.metrics import accuracy_score,confusion_matrix

def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main(root,out):
    if out.exists():raise RuntimeError('Use a fresh output directory.')
    study=root/'experiments/synthetic_simulation_20261026';manifest=read(study/'data_manifest.json');protocol=read(study/'protocol.json')
    paths=list((root/'models/current_models').rglob('*'))+list((study/'models').rglob('*'))+[root/'models/active_model.json',root/'data/expanded_survey_dataset/respondents.csv',study/manifest['csv'],study/'SYNTHETIC_test_metrics.json',study/'frozen_decisions.json']
    protected={p.relative_to(root).as_posix():sha(p) for p in paths if p.is_file()}
    assert sha(study/manifest['csv'])==manifest['sha256']
    frozen=read(study/'frozen_hashes.json')
    for n,h in frozen.items():assert sha(study/n)==h
    out.mkdir(parents=True);save(out/'protected_hashes.json',protected)
    save(out/'protocol.json',dict(scope='Read-only analytic diagnosis; no fitting, target edits, threshold search, deployment or new test evaluation.',
      source='Extract exact material-count and signal expressions from frozen generator AST. Conditional Gaussian class probabilities computed analytically using its recordednoiseSD and fixedboundaries.',
      estimate='Mean of maxclass conditionalprobability over7000training featureprofiles. This is a MonteCarlo estimate of expected unconstrained Bayesaccuracy under the intended simulationdistribution, not a hard upperbound on one realizedfinite testscore.',
      observed='Evaluate fixed known-rule argmax on existingvalidation only as a diagnostic reference. It is not an additional trained model or eligible productioncandidate. No testlabels used.',
      interpretation='Unconstrained optimal expectedaccuracy is already an upperbound on expectedaccuracy under additional recallconstraints. Does not establish any ceiling for real BRFSS data. Pseudorandom seed/ID reconstruction is not a predictive feature.'))
    df=pd.read_csv(study/manifest['csv']);tr=df[df.split.eq('train')].copy();va=df[df.split.eq('validation')].copy();del df
    module=ast.parse((study/'synthetic_stress_experiment.py').read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    assign={n.targets[0].id:n.value for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    # Extract fixed boundaries rather than silently copying a different equation.
    digitize=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='digitize')
    bounds=np.asarray(ast.literal_eval(digitize.args[1]),dtype=float);sigma=float(protocol['label_noise_sd'])
    def conditional(frame):
        env={'f':frame,'np':np};env['material']=eval(compile(ast.Expression(assign['material']),'generator-material','eval'),env)
        s=np.asarray(eval(compile(ast.Expression(assign['signal']),'generator-signal','eval'),env),dtype=float)
        a=ndtr((bounds[0]-s)/sigma);b=ndtr((bounds[1]-s)/sigma)
        p=np.column_stack([a,b-a,1-b]);assert np.allclose(p.sum(1),1) and (p>=0).all();return s,p
    s,tp=conditional(tr);_,vp=conditional(va);best=tp.max(1);mean=float(best.mean());se=float(best.std(ddof=1)/np.sqrt(len(best)))
    entropy=-np.sum(tp*np.log(np.clip(tp,1e-15,1)),axis=1)
    result=dict(label_noise_sd=sigma,boundaries=bounds.tolist(),training_profiles=len(tr),
      expected_best_accuracy_given_training_profiles=mean,feature_sampling_standard_error=se,
      feature_sampling_interval95=[mean-1.96*se,mean+1.96*se],
      expected_errors_even_with_known_rule=float(np.sum(1-best)),conditional_entropy_nats=float(entropy.mean()),
      validation_known_rule_accuracy=float(accuracy_score(va.target,vp.argmax(1))),validation_rows=len(va),
      validation_known_rule_confusion_matrix=confusion_matrix(va.target,vp.argmax(1),labels=[0,1,2]).tolist(),
      existing_validation_raw_accuracy={n:v['raw']['accuracy'] for n,v in read(study/'validation_metrics.json').items()},
      target80_gap_percentage_points=(.8-mean)*100,production_changed=False,test_evaluated=False)
    save(out/'noise_diagnosis.json',result)
    lines=['# Why the generated-data models are near65%','',
      f"The simulator adds independent Gaussian noise with standard deviation **{sigma:.2f}** to its hidden score before assigning the three labels. That uncertainty is not an input feature. Identical input patterns can therefore produce different valid labels under repeated simulation.",'',
      f"Using the exact known generating equation, the expected best unconstrained classification accuracy over the7,000training feature profiles is **{100*mean:.2f}%**. The approximate interval due to sampling input profiles is **{100*(mean-1.96*se):.2f}% to {100*(mean+1.96*se):.2f}%**. This interval is not a confidence interval for a particular trained model's test score.",'',
      f"A predictor given the exact equation's conditional probabilities reaches **{100*result['validation_known_rule_accuracy']:.2f}%** on the existing1,500validation labels. This is a fixed analytic reference, not a trained candidate. No test labels or test performance were used in this calculation.",'',
      '## Meaning for80%','',
      'Tuning can reduce a model\'s gap to the best predictor supported by these inputs, but cannot recover independent random uncertainty.80% is not a defensible expected target for this fixed simulator. Requiring preserved high and sometimes recall can only constrain the achievable operating accuracy further. Finite observed scores can fluctuate; the calculated value is not an absolute bound on every finite sample.','',
      'The earlier63.52% mean on the generated test set therefore does not imply that another large hyperparameter sweep will reach80%. Compare all model families on the same validation partition before attributing small differences to algorithms.','',
      '| Model | Existing validation raw accuracy |','|---|---:|']
    for n,v in result['existing_validation_raw_accuracy'].items():lines.append(f'| {n} | {100*v:.2f}% |')
    lines+=['','## Recommended action','',
      'Retain the current data and models. A bounded same-data tuning exercise could still improve weaker models by small amounts; it must use training-only fitting and validation selection, with protected minority recalls and no promised80% target. The simulator\'s known equation can serve as an analytic diagnostic, but presenting it as independently learned clinical knowledge would be misleading.','',
      'An easier simulation would need different assumptions about randomness and label overlap. That is a separately labelled benchmark, not an accuracy improvement on this unchanged dataset or on real people. Do not alter current labels, cherry-pick easy rows or use sample IDs/random-seed reconstruction to create apparent gains.','',
      'This mathematical diagnosis relies on the known simulation mechanism. It cannot establish a maximum accuracy for real survey data. All original and training-preview model artifacts and both source datasets were verified unchanged.','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    for n,h in protected.items():assert sha(root/n)==h,n
    save(out/'verification.json',dict(passed=True,all_protected_files_unchanged=True,models_retrained=False,datasets_changed=False,test_evaluated=False))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.project),Path(a.output))
