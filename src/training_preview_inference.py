"""Educational inference over generated records; independent of production models."""
import hashlib,json,math,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from src.project_paths import resolve_path
from src.models import Conv1DNet,LSTMNet,CNNLSTMNet

NAMES=['Logistic Regression','Random Forest','XGBoost','1D CNN','LSTM','CNN-LSTM']
SEEDS=[20261026,20261027]

def read(path):return json.loads(resolve_path(path).read_text())
def sha(path):
    with resolve_path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class TrainingPreviewPredictor:
    version='model_training_simulation'
    def __init__(self,project_root=None,experiment=None,override_manifest=None):
        self.project=Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        config=self.project/'models/training_preview.json'
        settings=read(config) if config.exists() else {'experiment':'original'}
        self.experiment=experiment or settings['experiment']
        versions={'original':('synthetic_simulation_20261026','frozen_decisions.json','SYNTHETIC_test_metrics.json',1.0),'lower_noise':('lower_noise_simulation_20261026','frozen_models.json','test_metrics.json',.25)}
        if self.experiment not in versions:raise ValueError('Unsupported training preview experiment.')
        folder,frozen,metrics,self.noise_sd=versions[self.experiment]
        self.root=resolve_path(self.project/'experiments'/folder)
        self.schema=read(self.project/'models/current_models/schema.json')
        self.manifest=read(self.root/'data_manifest.json');self.frozen=read(self.root/frozen)
        self.results=read(self.root/metrics)
        if sha(self.root/self.manifest['csv'])!=self.manifest['sha256']:raise RuntimeError('Training dataset checksum mismatch.')
        for name,digest in self.frozen['model_hashes'].items():
            if sha(self.root/name)!=digest:raise RuntimeError('Training checkpoint checksum mismatch.')
        self.overrides=None;self.model_revision='base'
        # An explicit experiment override loads its historical bundle unchanged.
        selected_overrides=override_manifest or settings.get('overrides')
        if experiment is None and selected_overrides and self.experiment=='lower_noise':
            override_path=resolve_path(self.project/selected_overrides).resolve()
            if not any(override_path.is_relative_to((self.project/p).resolve()) for p in ['experiments','model_training']):raise ValueError('Override path must remain inside training directories.')
            self.overrides=read(override_path)
            if self.overrides['data_sha256']!=self.manifest['sha256']:raise RuntimeError('Tuned model dataset mismatch.')
            if not set(self.overrides['models'])<={'Random Forest','1D CNN'}:raise ValueError('Unsupported model override.')
            for name,digest in self.overrides['sha256'].items():
                path=resolve_path(self.project/name).resolve()
                if not path.is_relative_to(self.project.resolve()) or sha(path)!=digest:raise RuntimeError('Tuned checkpoint/source checksum mismatch.')
            self.model_revision=self.overrides['revision']
        self.models={};classes=[Conv1DNet,LSTMNet,CNNLSTMNet];torch.set_num_threads(4)
        for i,name in enumerate(NAMES):
            bundles=[]
            if self.overrides and name in self.overrides['models']:
                from src.training_variants import PositionConv1D
                entry=self.overrides['models'][name]
                if not entry.get('performance_passed'):raise RuntimeError('Tuned model did not pass the performance gate.')
                for bundle in entry['bundles']:
                    pre=joblib.load(resolve_path(self.project/bundle['encoder']))
                    if name=='Random Forest':m=joblib.load(resolve_path(self.project/bundle['checkpoint']))
                    else:
                        c=entry['config']
                        if c['architecture']=='position_cnn':m=PositionConv1D(bundle['input_dim'],channels=c['channels'],hidden=c['hidden'],dropout=c['dropout'])
                        elif c['architecture']=='original_cnn':m=Conv1DNet(bundle['input_dim'])
                        else:raise ValueError('Unknown CNN checkpoint architecture.')
                        m.load_state_dict(torch.load(resolve_path(self.project/bundle['checkpoint']),map_location='cpu',weights_only=True));m.eval()
                    bundles.append((pre,m))
                self.models[name]=bundles
                self.results['models'][name]['raw']=entry['metrics']
                continue
            for seed in SEEDS if i>=3 else SEEDS[:1]:
                stem=self.root/'models'/f'{i}_{seed}';pre=joblib.load(stem.with_suffix('.preprocessor.joblib'))
                if i<3:m=joblib.load(stem.with_suffix('.joblib'))
                else:
                    m=classes[i-3](input_dim=self.schema['input_dim']);m.load_state_dict(torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True));m.eval()
                bundles.append((pre,m))
            self.models[name]=bundles

    def public_schema(self):
        return dict(model_version=self.version,input_count=19,records=self.manifest['rows'],
          training_records=self.manifest['split_counts']['train'],test_records=self.manifest['split_counts']['test'],
          data_origin='synthetic_simulation',ensemble={'method':'equal_raw_probability_mean'},
          majority_accuracy=self.results['majority_baseline']['accuracy'],
          majority_class=self.results.get('majority_class',self.results.get('training_majority_class',0)),
          experiment=self.experiment,noise_sd=self.noise_sd,dataset_file=self.manifest['csv'],model_revision=self.model_revision)

    def benchmarks(self):
        return [dict(Model=n,Accuracy=self.results['models'][n]['raw']['accuracy'],
          **{'F1-Score (Macro)':self.results['models'][n]['raw']['macro_f1'],'ROC-AUC (Macro)':self.results['models'][n]['raw']['macro_auc'],
             'Recall by class':self.results['models'][n]['raw']['recall']},
          data_origin='synthetic_simulation',evaluation='1500 held-out generated profiles; not real-person validation') for n in NAMES]

    def validate(self,data):
        keys=self.schema['numeric']+self.schema['categorical']
        if not isinstance(data,dict) or set(data)!=set(keys):raise ValueError('Supply exactly the19 profile inputs.')
        for key in keys:
            value=data[key];lo,hi=self.schema['limits'][key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+key)
            if key!='bmi' and value!=int(value):raise ValueError(key+' must be a whole number.')
        return keys

    def raw_probabilities(self,data):
        keys=self.validate(data);frame=pd.DataFrame([data])[keys];raw={}
        for name,bundles in self.models.items():
            pp=[]
            for pre,m in bundles:
                x=pre.transform(frame).astype('float32')
                if isinstance(m,torch.nn.Module):
                    with torch.no_grad():p=m(torch.from_numpy(x)).softmax(1).numpy()[0]
                else:p=m.predict_proba(x)[0]
                pp.append(p)
            raw[name]=np.mean(pp,axis=0,dtype=np.float64)
        return raw

    def predict(self,data):
        raw=self.raw_probabilities(data);details={};votes=[]
        for name,p in raw.items():
            label=int(p.argmax());votes.append(label);details[name]=dict(predicted_class=label,label=self.schema['labels'][label],confidence=float(p[label]),probabilities=p.tolist())
        combined=np.mean(list(raw.values()),axis=0);winner=int(combined.argmax())
        return dict(status='success',dataset='Lower-noise generated training-data simulation' if self.experiment=='lower_noise' else 'Generated training-data simulation',model_version=self.version,data_origin='synthetic_simulation',experiment=self.experiment,noise_sd=self.noise_sd,model_revision=self.model_revision,
          target='Simulated stress-frequency class',models=details,
          consensus=dict(class_id=winner,label=self.schema['labels'][winner],agreement_rate=round(100*votes.count(winner)/6,1),
            probabilities=combined.tolist(),method='equal_raw_probability_mean',decision_rule='Average six raw probability vectors equally; choose largest class probability; lower-class tie break.'),
          limitations='Educational simulation trained from generated records. Accuracy is measured on generated test profiles and does not establish accuracy for real people. Each neural family averages two fixed seeds. Individual labels use raw-probability argmax.')
