"""Inference for the separately versioned BRFSS models."""
from pathlib import Path
import json, math, os
import numpy as np
import pandas as pd
import joblib, torch
from src.models import Conv1DNet, LSTMNet, CNNLSTMNet
from src.model_paths import MODEL_ROOT, PROJECT_ROOT, model_dir

class BRFSSPredictor:
    def __init__(self, version=None):
        active = MODEL_ROOT/'active_model.json'
        version = version or os.environ.get('BRFSS_MODEL_VERSION') or (json.loads(active.read_text())['version'] if active.exists() else 'brfss2022_expanded')
        if version not in ('brfss2022', 'brfss2022_tuned', 'brfss2022_expanded'):
            raise ValueError('Unsupported BRFSS model version.')
        self.root = model_dir(version)
        ROOT_LOCAL = self.root
        self.version = version
        if version == 'brfss2022_expanded':
            ready = json.loads((ROOT_LOCAL/'READY.json').read_text())
            gate_path=PROJECT_ROOT/'reports/brfss2022_expanded/deployment_gate.json'
            gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
            if not ready.get('all_six_pass') or not gate.get('all_models_and_ensemble_pass'):
                raise RuntimeError('Expanded models have not all passed the high-stress recall and accuracy gates; keep the existing version active.')
        self.selection = json.loads((ROOT_LOCAL/'selection.json').read_text()) if version!='brfss2022' else {}
        self._load(ROOT_LOCAL)

    def _load(self, ROOT):
        if not (ROOT/'READY.json').exists():raise RuntimeError('BRFSS training has not completed.')
        self.schema=json.loads((ROOT/'schema.json').read_text())
        self.pre=joblib.load(ROOT/'preprocessor.joblib')
        self.models={name:joblib.load(ROOT/file) for name,file in [('Logistic Regression','logistic_regression.joblib'),('Random Forest','random_forest.joblib'),('XGBoost','xgboost.joblib')]}
        torch.set_num_threads(4)
        for name,file,cls in [('1D CNN','conv1d_net.pt',Conv1DNet),('LSTM','lstm_net.pt',LSTMNet),('CNN-LSTM','cnn_lstm_net.pt',CNNLSTMNet)]:
            model=cls(input_dim=self.schema['input_dim']);model.load_state_dict(torch.load(ROOT/file,map_location='cpu',weights_only=True));model.eval();self.models[name]=model
    def predict(self,data):
        keys=self.schema['numeric']+self.schema['categorical']
        if not isinstance(data,dict) or set(data)!=set(keys):raise ValueError(f'Supply exactly the {len(keys)} BRFSS survey inputs.')
        clean={}
        for key in keys:
            value=data[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError(key+' must be numeric.')
            lo,hi=self.schema['limits'][key]
            if not math.isfinite(value) or not lo<=value<=hi:raise ValueError(f'{key} must be between {lo} and {hi}.')
            if key!='bmi' and value!=int(value):raise ValueError(key+' must be a whole number.')
            clean[key]=value
        x=self.pre.transform(pd.DataFrame([clean])[keys]).astype('float32');details={};votes=[];probs=[]
        for name,model in self.models.items():
            if isinstance(model,torch.nn.Module):
                with torch.no_grad():p=model(torch.from_numpy(x)).softmax(1).numpy()[0]
            else:p=model.predict_proba(x)[0]
            if name in self.selection:
                p = np.asarray(p,dtype=np.float64)*np.asarray(self.selection[name]['multipliers'])
                p = p/p.sum()
            label=int(p.argmax());votes.append(label);probs.append(p)
            details[name]={'predicted_class':label,'label':self.schema['labels'][label],'confidence':float(p[label]),'probabilities':p.tolist()}
        winner=int(np.bincount(votes,minlength=3).argmax())
        return {'status':'success','dataset':'BRFSS 2022','model_version':self.version,'target':'Self-reported stress frequency','models':details,'consensus':{'class_id':winner,'label':self.schema['labels'][winner],'agreement_rate':round(100*votes.count(winner)/6,1),'probabilities':np.mean(probs,axis=0).tolist()},'tie_break_rule':'Lowest class ID among equally frequent votes','limitations':'Uncalibrated, cross-sectional survey estimates; tuned models may include validation-selected class-prior adjustment. Not a diagnosis or future-stress forecast.'}
