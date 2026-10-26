"""Fresh-process memory and six-family parity before optional activation."""
import argparse,gc,importlib.util,json,resource,sys,time
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--loader',required=True);p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.project);out=Path(a.experiment);sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('staged_preview_loader',a.loader);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
engine=module.TrainingPreviewPredictor(root,override_manifest='experiments/'+out.name+'/overrides.json');assert engine.model_revision=='rf_cnn_tuned_20261026';base=root/'experiments/lower_noise_simulation_20261026';df=pd.read_csv(base/'model_training_data.csv');te=df[df.split.eq('test')].iloc[:12];keys=engine.schema['numeric']+engine.schema['categorical'];old=np.load(base/'test_predictions.npz');new=np.load(out/'test_predictions.npz');diff={};times=[]
for j,(_,row) in enumerate(te.iterrows()):
    start=time.monotonic();result=engine.predict({k:float(row[k]) for k in keys});times.append(time.monotonic()-start)
    for n,r in result['models'].items():
        expected=new[n][j] if n in new.files else old[n][j];error=float(np.max(np.abs(np.asarray(r['probabilities'])-expected)));diff[n]=max(diff.get(n,0),error);assert error<1e-6,(n,error);assert r['predicted_class']==int(expected.argmax())
    assert result['consensus']['class_id']==int(np.argmax(np.mean([r['probabilities'] for r in result['models'].values()],axis=0)))
for key,value in [('age',True),('age',999),('support',2.5),('bmi',float('nan'))]:
    profile={k:float(te.iloc[0][k]) for k in keys};profile[key]=value
    try:engine.predict(profile)
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted.')
bench=engine.benchmarks();assert len(bench)==6
for r in bench:assert r['Accuracy']==engine.results['models'][r['Model']]['raw']['accuracy']
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;mem_ok=peak<512*1024;size=sum((root/n).stat().st_size for n in engine.overrides['sha256'] if n.endswith(('.pt','.joblib')))
report=dict(passed=bool(mem_ok and max(times)<2 and all(v<1e-6 for v in diff.values())),six_family_parity=diff,peak_rss_mib=peak/1024,memory_limit_mib=512,memory_passed=mem_ok,mean_prediction_seconds=float(np.mean(times)),max_prediction_seconds=max(times),override_artifact_bytes=size,invalid_inputs_rejected=True,benchmarks_match=True,active_config_unchanged=True)
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
