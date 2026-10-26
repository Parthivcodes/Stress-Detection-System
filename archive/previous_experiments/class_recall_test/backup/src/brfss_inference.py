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
        self.ensemble_policy = None
        self.ordinal_models = None
        # Original six checkpoints and per-model outputs remain unchanged.
        policy_dir = Path(os.environ.get('BRFSS_ENSEMBLE_DIR', str(self.root/'ensemble')))
        policy_path = policy_dir/'policy.json'
        if self.version == 'brfss2022_expanded' and policy_path.exists() and os.environ.get('BRFSS_DISABLE_SOFT_ENSEMBLE') != '1':
            policy = json.loads(policy_path.read_text())
            if policy.get('enabled'):
                from src.compact_forest import sha256
                from src.ensemble_policy import weighted_probabilities, classify
                for file, expected in policy['source_sha256'].items():
                    if Path(file).name != file or sha256(self.root/file) != expected:
                        raise RuntimeError('Ensemble source checkpoint mismatch.')
                allowed = {'Logistic Regression', 'XGBoost', 'LSTM', 'Ordinal XGBoost'}
                if not set(policy['weights']) <= allowed:
                    raise RuntimeError('Unsupported ensemble member.')
                dummy = {name:np.array([.3,.3,.4]) for name in policy['weights']}
                classify(weighted_probabilities(dummy,policy['weights']),policy['high_threshold'])
                if 'Ordinal XGBoost' in policy['weights']:
                    files=['ordinal_at_least_sometimes.joblib','ordinal_high.joblib']
                    for file in files:
                        if sha256(policy_dir/file) != policy['ordinal_sha256'][file]:
                            raise RuntimeError('Ordinal checkpoint mismatch.')
                    self.ordinal_models=[joblib.load(policy_dir/file) for file in files]
                self.ensemble_policy=policy

    def ensemble_info(self):
        if self.ensemble_policy is None:
            return {'method':'majority_vote', 'majority_baseline_accuracy':20207/30718}
        p=self.ensemble_policy
        return {key:p[key] for key in ('method','weights','high_threshold','test_metrics','majority_baseline_accuracy')}

    def _load(self, ROOT):
        if not (ROOT/'READY.json').exists():raise RuntimeError('BRFSS training has not completed.')
        self.schema=json.loads((ROOT/'schema.json').read_text())
        self.pre=joblib.load(ROOT/'preprocessor.joblib')
        # Opt-in deployment optimization. Original checkpoints remain the default
        # for training, local development, historical versions and parity tests.
        compact_dir = os.environ.get('BRFSS_COMPACT_FOREST_DIR')
        self.models = {}
        for name, file in [('Logistic Regression','logistic_regression.joblib'),('Random Forest','random_forest.joblib'),('XGBoost','xgboost.joblib')]:
            if name == 'Random Forest' and compact_dir:
                from src.compact_forest import CompactForest
                self.models[name] = CompactForest(compact_dir, ROOT/file)
            else:
                self.models[name] = joblib.load(ROOT/file)
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
        x=self.pre.transform(pd.DataFrame([clean])[keys]).astype('float32');details={};votes=[];probs=[];raw={}
        for name,model in self.models.items():
            if isinstance(model,torch.nn.Module):
                with torch.no_grad():p=model(torch.from_numpy(x)).softmax(1).numpy()[0]
            else:p=model.predict_proba(x)[0]
            raw[name]=np.array(p,dtype=np.float64,copy=True)
            if name in self.selection:
                p = np.asarray(p,dtype=np.float64)*np.asarray(self.selection[name]['multipliers'])
                p = p/p.sum()
            label=int(p.argmax());votes.append(label);probs.append(p)
            details[name]={'predicted_class':label,'label':self.schema['labels'][label],'confidence':float(p[label]),'probabilities':p.tolist()}
        winner=int(np.bincount(votes,minlength=3).argmax())
        consensus_p=np.mean(probs,axis=0)
        rule='Lowest class ID among equally frequent votes'
        if self.ensemble_policy:
            from src.ensemble_policy import ordinal_probabilities,weighted_probabilities,classify
            if self.ordinal_models:
                a,b=[m.predict_proba(x)[:,1] for m in self.ordinal_models]
                raw['Ordinal XGBoost']=ordinal_probabilities(a,b)[0]
            consensus_p=weighted_probabilities(raw,self.ensemble_policy['weights'])
            winner=int(classify(consensus_p,self.ensemble_policy['high_threshold']))
            rule='High class when weighted raw P(high) >= cutoff; otherwise argmax of rarely/sometimes; lower-class ties select rarely.'
        consensus={'class_id':winner,'label':self.schema['labels'][winner],
                   'agreement_rate':round(100*votes.count(winner)/6,1),
                   'agreement_meaning':'Fraction of six displayed individual classifications matching the consensus; not how the ensemble is selected.',
                   'probabilities':consensus_p.tolist(),'decision_rule':rule,**self.ensemble_info()}
        return {'status':'success','dataset':'BRFSS 2022','model_version':self.version,'target':'Self-reported stress frequency','models':details,'consensus':consensus,'tie_break_rule':rule,'limitations':'Uncalibrated, cross-sectional survey estimates; individual models retain validation-selected class adjustments. Soft ensemble uses raw probabilities and one high-stress cutoff. Not a diagnosis or future-stress forecast.'}
