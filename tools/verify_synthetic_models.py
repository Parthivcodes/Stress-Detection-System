"""Checkpoint loading parity, without fitting or selecting anything."""
import argparse,json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from tune_six_models import read,save,sha,prob,NAMES
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--experiment',required=True);a=p.parse_args();root=Path(a.project);out=Path(a.experiment);sys.path.insert(0,str(root));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
torch.set_num_threads(4);classes=[Conv1DNet,LSTMNet,CNNLSTMNet];schema=read(root/'models/current_models/schema.json');features=schema['numeric']+schema['categorical'];df=pd.read_csv(out/read(out/'data_manifest.json')['csv']);te=df[df.split.eq('test')].iloc[:8];stored=np.load(out/'SYNTHETIC_test_predictions.npz');assert np.array_equal(stored['sample_id'][:8],te.sample_id)
results={}
for i,n in enumerate(NAMES):
    pp=[];started=time.monotonic()
    for seed in [20261026,20261027] if i>=3 else [20261026]:
        stem=out/'models'/f'{i}_{seed}';pre=joblib.load(stem.with_suffix('.preprocessor.joblib'));x=pre.transform(te[features]).astype('float32')
        if i<3:m=joblib.load(stem.with_suffix('.joblib'))
        else:m=classes[i-3](input_dim=x.shape[1]);m.load_state_dict(torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True))
        pp.append(prob(m,x));del m
    mean=np.mean(pp,axis=0);error=float(np.max(np.abs(mean-stored[n][:8])));assert error<1e-6,(n,error);assert np.allclose(mean.sum(1),1,atol=1e-6);results[n]=dict(max_probability_error=error,load_and_batch_seconds=time.monotonic()-started)
for name,h in read(out/'protected_real_artifacts.json').items():assert sha(root/name)==h,name
save(out/'checkpoint_verification.json',dict(passed=True,all_six_loaded=True,stored_prediction_parity=results,real_artifacts_unchanged=True,production_changed=False,architecture_sha256=sha(root/'src/models.py')));print('All six synthetic checkpoint reloads match saved predictions. Real artifacts unchanged.')
