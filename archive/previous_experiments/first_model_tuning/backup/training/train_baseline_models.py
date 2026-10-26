"""Reproducible, separate CDC BRFSS 2022 preparation/training; never writes legacy models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, hashlib, zipfile, collections, copy
import numpy as np
import pandas as pd
import joblib, torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.models import Conv1DNet, LSTMNet, CNNLSTMNet
from src.model_paths import model_dir, LEGACY_ROOT, preserved_path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/baseline_survey_dataset'; OUT=model_dir('brfss2022'); REPORT=ROOT/'reports/brfss2022'
NUM=['age','sleep_duration','unhealthy_days','bmi']
CAT=['sex','exercise','general_health','marital','education','employment','income','support']
FIELDS={'age':(1984,2),'sleep_duration':(114,2),'unhealthy_days':(102,2),'bmi':(1998,4),'sex':(91,1),'exercise':(113,1),'general_health':(101,1),'marital':(168,1),'education':(169,1),'employment':(183,1),'income':(186,2),'support':(362,1),'stress_raw':(370,1)}
LIMITS={'age':(18,80),'sleep_duration':(1,24),'unhealthy_days':(0,30),'bmi':(1,99.98),'sex':(1,2),'exercise':(1,2),'general_health':(1,5),'marital':(1,6),'education':(1,6),'employment':(1,8),'income':(1,11),'support':(1,5),'stress_raw':(1,5)}
LABELS=['Rarely / never','Sometimes','Usually / always']
NAMES=['Logistic Regression','Random Forest','XGBoost','1D CNN','LSTM','CNN-LSTM']
FILES=['logistic_regression.joblib','random_forest.joblib','xgboost.joblib','conv1d_net.pt','lstm_net.pt','cnn_lstm_net.pt']
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def prepare():
    if (OUT/'READY.json').exists():
        raise SystemExit('Archived baseline exists. Refusing to overwrite its data or models; use a new experiment version.')
    for p in [DATA,OUT,REPORT]: p.mkdir(parents=True,exist_ok=True)
    originals=list(LEGACY_ROOT.glob('*.*'))+list((ROOT/'data/original_sleep_dataset').glob('*.csv'))
    manifest={str(p.relative_to(ROOT)):digest(p) for p in originals}
    if not (REPORT/'legacy_hashes.json').exists():
        dump(REPORT/'legacy_hashes.json',manifest)
        with zipfile.ZipFile(REPORT/'legacy_backup.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in originals:z.write(p,str(p.relative_to(ROOT)))
    ids=set(); rows=[]; invalid=collections.Counter(); stress=collections.Counter(); total=0
    archive=DATA/'raw/LLCP2022ASC.zip'
    with zipfile.ZipFile(archive) as z, z.open(z.namelist()[0]) as f:
        for i,line in enumerate(f,1):
            total+=1
            assert len(line.rstrip(b'\r\n'))==2052,(i,len(line))
            key='2022_'+line[:2].decode()+'_'+line[35:45].decode().strip()
            assert key not in ids,('duplicate respondent',key)
            ids.add(key)
            values={k:int(line[s-1:s-1+n].strip() or b'-1') for k,(s,n) in FIELDS.items()}
            stress[str(values['stress_raw'])]+=1
            raw=values.copy()
            if values['unhealthy_days']==88:values['unhealthy_days']=0
            values['bmi']/=100
            bad=[k for k,(lo,hi) in LIMITS.items() if not lo<=values[k]<=hi]
            invalid.update(bad)
            if bad:continue
            values.update(respondent_id=key,source_row=i,target=0 if values['stress_raw']>=4 else 1 if values['stress_raw']==3 else 2)
            values.update({'raw_'+k:v for k,v in raw.items()})
            rows.append(values)
    assert total==445132 and len(rows)>=30000
    frame=pd.DataFrame(rows).sample(n=30000,random_state=42).reset_index(drop=True)
    groups=pd.util.hash_pandas_object(frame[NUM+CAT],index=False).astype(str)
    fold=np.zeros(len(frame),dtype=int)
    for n,(_,idx) in enumerate(StratifiedGroupKFold(10,shuffle=True,random_state=42).split(frame,frame.target,groups)):fold[idx]=n
    frame['split']=np.where(fold<2,'test',np.where(fold==2,'validation','train'))
    assert frame.respondent_id.nunique()==30000
    assert pd.DataFrame({'group':groups,'split':frame.split}).groupby('group')['split'].nunique().max()==1
    frame.to_csv(DATA/'respondents_30000.csv',index=False)
    dump(REPORT/'dataset_audit.json',{'source':'https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022ASC.zip','sha256':digest(archive),'source_records':total,'unique_source_keys':len(ids),'eligible_complete_cases':len(rows),'invalid_per_field_overlapping':dict(invalid),'raw_stress_counts':dict(stress),'sample_records':len(frame),'unique_sample_keys':frame.respondent_id.nunique(),'class_counts':frame.target.value_counts().sort_index().to_dict(),'splits':frame.split.value_counts().to_dict(),'seed':42,'identical_profiles_kept_in_same_split':True})
    print('DATA AUDIT',json.loads((REPORT/'dataset_audit.json').read_text()),flush=True)
    return frame
def train(frame):
    # Never advertise a partially overwritten rerun as ready.
    (OUT/'READY.json').unlink(missing_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    pre=ColumnTransformer([('numeric',StandardScaler(),NUM),('categorical',OneHotEncoder(sparse_output=False,handle_unknown='error',categories=[list(range(LIMITS[k][0],LIMITS[k][1]+1)) for k in CAT]),CAT)])
    tr=frame[frame.split=='train']; va=frame[frame.split=='validation']; te=frame[frame.split=='test']
    x=pre.fit_transform(tr[NUM+CAT]).astype('float32');v=pre.transform(va[NUM+CAT]).astype('float32');t=pre.transform(te[NUM+CAT]).astype('float32')
    y=tr.target.to_numpy(copy=True);vy=va.target.to_numpy(copy=True);ty=te.target.to_numpy(copy=True)
    joblib.dump(pre,OUT/'preprocessor.joblib');dump(OUT/'schema.json',{'numeric':NUM,'categorical':CAT,'limits':LIMITS,'labels':LABELS,'input_dim':x.shape[1],'dataset':'CDC BRFSS 2022, 30,000 unique respondent records'})
    results=[]
    def evaluate(name,prob):
        pred=prob.argmax(1)
        row={'Model':name,'Accuracy':accuracy_score(ty,pred),'Balanced accuracy':balanced_accuracy_score(ty,pred),'F1-Score (Macro)':f1_score(ty,pred,average='macro'),'ROC-AUC (Macro)':roc_auc_score(ty,prob,multi_class='ovr',average='macro'),'confusion_matrix':confusion_matrix(ty,pred).tolist()}
        results.append(row);dump(REPORT/'metrics.json',results);print(row,flush=True)
    counts=np.bincount(y,minlength=3)/len(y);evaluate('Training-prior baseline',np.tile(counts,(len(ty),1)))
    classical=[LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=8,class_weight='balanced',n_jobs=4,random_state=42),XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=42,eval_metric='mlogloss')]
    for i,model in enumerate(classical):
        print('TRAINING',NAMES[i],flush=True)
        model.fit(x,y,**({'sample_weight':compute_sample_weight('balanced',y)} if i==2 else {}));joblib.dump(model,OUT/FILES[i]);evaluate(NAMES[i],model.predict_proba(t))
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=256,shuffle=True)
    weights=torch.tensor(compute_class_weight('balanced',classes=np.arange(3),y=y),dtype=torch.float32)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    def probabilities(model,a):
        model.eval()
        with torch.no_grad():return torch.cat([model(torch.from_numpy(b)).softmax(1) for b in np.array_split(a,max(1,int(np.ceil(len(a)/256))))]).numpy()
    for i,cls in enumerate([Conv1DNet,LSTMNet,CNNLSTMNet],3):
        model=cls(input_dim=x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=-1;stale=0;history=[]
        for epoch in range(1,31):
            model.train()
            for bx,by in loader:
                optimizer.zero_grad();loss=criterion(model(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            score=f1_score(vy,probabilities(model,v).argmax(1),average='macro');history.append(score)
            print(NAMES[i],epoch,'validation macro F1',round(score,4),flush=True)
            if score>best+.0001:best=score;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if stale>=5:break
        model.load_state_dict(state);torch.save(state,OUT/FILES[i]);dump(REPORT/(FILES[i]+'.history.json'),history);evaluate(NAMES[i],probabilities(model,t))
    dump(OUT/'metrics.json',results[1:])
    preserved=json.loads((REPORT/'legacy_hashes.json').read_text());assert all(digest(preserved_path(p))==h for p,h in preserved.items())
    dump(OUT/'READY.json',{'models':NAMES,'dataset_sha256':digest(DATA/'respondents_30000.csv'),'legacy_verified_unchanged':True})
    print('ALL SIX MODELS COMPLETE; LEGACY HASHES VERIFIED',flush=True)
if __name__=='__main__': train(prepare())
