"""Bounded validation-only tuning. No data modification or test-driven selection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy, json, shutil, sys
import numpy as np
import pandas as pd
import torch, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
from training.train_baseline_models import ROOT, NUM, CAT, NAMES, FILES, digest, dump
from src.model_paths import model_dir

OLD=model_dir('brfss2022'); OUT=model_dir('brfss2022_tuned'); REPORT=ROOT/'reports/brfss2022_tuned'
DATA=ROOT/'data/baseline_survey_dataset/respondents_30000.csv'
EXPONENTS=[0,.15,.3,.45,.6,.8,1.0]
def summary(y,p):
    pred=p.argmax(1)
    return {'Accuracy':float(accuracy_score(y,pred)),'Balanced accuracy':float(balanced_accuracy_score(y,pred)),
            'F1-Score (Macro)':float(f1_score(y,pred,average='macro')),'Recall by class':recall_score(y,pred,labels=[0,1,2],average=None,zero_division=0).tolist()}
def probability(model,x):
    if not isinstance(model,torch.nn.Module):return model.predict_proba(x)
    model.eval()
    with torch.no_grad():return torch.cat([model(torch.from_numpy(b)).softmax(1) for b in np.array_split(x,max(1,int(np.ceil(len(x)/256))))]).numpy()
def adjusted(p,m):
    p=np.asarray(p,dtype=np.float64)*m
    return p/p.sum(axis=1,keepdims=True)
def main():
    if (OUT/'READY.json').exists():raise SystemExit('Completed tuned version exists; refusing to overwrite it or retune against test results.')
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    original={str(p.relative_to(ROOT)):digest(p) for p in [DATA,*OLD.glob('*.*')]}
    dump(REPORT/'preserved_hashes.json',original)
    dump(REPORT/'protocol.json',{'seed':2026,'same_30000_records':True,'same_splits':True,'selection':'Maximize mean of validation accuracy and macro F1, subject to macro F1 >= original validation F1, balanced accuracy >= original minus .03, usually/always recall >= .30. Original retained as fallback.','probability_prior_exponents':EXPONENTS,'neural_max_epochs':15,'neural_patience':5,'test_policy':'Freeze all six selections before evaluating test once. Test was evaluated in the earlier project iteration, so this is a reused internal holdout, not new external validation.','probabilities':'Class-prior-adjusted uncalibrated estimates, not clinically calibrated probabilities.'})
    frame=pd.read_csv(DATA);assert len(frame)==frame.respondent_id.nunique()==30000
    assert frame.groupby(NUM+CAT)['split'].nunique().max()==1
    tr=frame[frame.split=='train'];va=frame[frame.split=='validation']
    pre=joblib.load(OLD/'preprocessor.joblib');x=pre.transform(tr[NUM+CAT]).astype('float32');v=pre.transform(va[NUM+CAT]).astype('float32')
    y=tr.target.to_numpy(copy=True);vy=va.target.to_numpy(copy=True)
    prior=np.bincount(y,minlength=3)/len(y);weights=compute_class_weight('balanced',classes=np.arange(3),y=y)
    torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
    # WSL's unprivileged user can write contents but cannot set Windows timestamps.
    for f in ['schema.json','preprocessor.joblib']:shutil.copyfile(OLD/f,OUT/f)
    all_trials=[];selections={};selected_models=[];old_models=[]
    classical=[
        [(f'LR C={c}, sqrt weights',LogisticRegression(C=c,max_iter=1500,class_weight=dict(enumerate(weights**.5)),random_state=2026)) for c in [.1,1,10]],
        [(f'RF depth={depth}, weight exponent={a}',RandomForestClassifier(n_estimators=300,max_depth=depth,min_samples_leaf=leaf,max_features=.8,class_weight=dict(enumerate(weights**a)),n_jobs=4,random_state=2026)) for depth,leaf,a in [(12,6,.5),(20,10,.5),(16,8,0)]],
        [(f'XGB depth={depth}, weight exponent={a}',XGBClassifier(n_estimators=300,max_depth=depth,learning_rate=.035,min_child_weight=10,reg_lambda=5,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=2026,eval_metric='mlogloss'),a) for depth,a in [(3,.5),(5,.5),(3,0)]]]
    for i,name in enumerate(NAMES):
        if i<3:old=joblib.load(OLD/FILES[i])
        else:
            old=[Conv1DNet,LSTMNet,CNNLSTMNet][i-3](input_dim=x.shape[1]);old.load_state_dict(torch.load(OLD/FILES[i],map_location='cpu',weights_only=True))
        old_models.append(old);base=summary(vy,probability(old,v));best=(base['Accuracy']+base['F1-Score (Macro)'])/2
        chosen=copy.deepcopy(old);chosen_record={'candidate':'original','exponent':0,'multipliers':[1,1,1],'validation':base}
        def consider(model,description):
            nonlocal best,chosen,chosen_record
            p=probability(model,v);improved=False
            for exponent in EXPONENTS:
                mult=prior**exponent;s=summary(vy,adjusted(p,mult));score=(s['Accuracy']+s['F1-Score (Macro)'])/2
                eligible=s['F1-Score (Macro)']>=base['F1-Score (Macro)'] and s['Balanced accuracy']>=base['Balanced accuracy']-.03 and s['Recall by class'][2]>=.30
                record={'model':name,'candidate':description,'exponent':exponent,'multipliers':mult.tolist(),'validation':s,'eligible':eligible}
                all_trials.append(record)
                if eligible and score>best+1e-6:best=score;chosen=copy.deepcopy(model);chosen_record=record;improved=True
            return improved
        consider(old,'original checkpoint, prior adjustment')
        if i<3:
            for spec in classical[i]:
                description,model=spec[:2];print('TRAIN',name,description,flush=True)
                model.fit(x,y,**({'sample_weight':weights[y]**spec[2]} if i==2 else {}));consider(model,description)
        else:
            model=copy.deepcopy(old);optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.001)
            loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights**.5,dtype=torch.float32))
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=256,shuffle=True)
            stale=0
            for epoch in range(1,16):
                model.train()
                for bx,by in loader:
                    optimizer.zero_grad();loss=loss_fn(model(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                improvement=consider(model,f'fine-tuned sqrt weights epoch {epoch}')
                stale=0 if improvement else stale+1
                print(name,'epoch',epoch,'selected validation',chosen_record['validation'],flush=True)
                if stale>=5:break
        if i<3:joblib.dump(chosen,OUT/FILES[i])
        else:torch.save(chosen.state_dict(),OUT/FILES[i])
        selections[name]=chosen_record;selected_models.append(chosen)
        dump(REPORT/'validation_trials.json',all_trials);dump(OUT/'selection.json',selections)
        print('FROZEN',name,chosen_record,flush=True)
    # The selection file is complete before the test features/labels enter evaluation.
    selection_hash=digest(OUT/'selection.json')
    te=frame[frame.split=='test'];t=pre.transform(te[NUM+CAT]).astype('float32');ty=te.target.to_numpy(copy=True)
    metrics=[];comparison=[];predictions={'respondent_id':te.respondent_id.to_numpy(),'target':ty}
    for i,name in enumerate(NAMES):
        p=adjusted(probability(selected_models[i],t),selections[name]['multipliers']);before=probability(old_models[i],t)
        s={'Model':name,**summary(ty,p),'ROC-AUC (Macro)':float(roc_auc_score(ty,p,multi_class='ovr')),'confusion_matrix':confusion_matrix(ty,p.argmax(1)).tolist()}
        metrics.append(s);comparison.append({'Model':name,'before':summary(ty,before),'after':s,'selection':selections[name]['candidate'],'prior_exponent':selections[name]['exponent']})
        predictions[name]=p.argmax(1)
        for c in range(3):predictions[f'{name} probability {c}']=p[:,c]
        print('FINAL TEST',comparison[-1],flush=True)
    assert digest(OUT/'selection.json')==selection_hash
    assert all(digest(ROOT/p)==h for p,h in original.items())
    dump(OUT/'metrics.json',metrics);dump(REPORT/'comparison.json',comparison)
    pd.DataFrame(predictions).to_csv(REPORT/'test_predictions.csv',index=False)
    dump(OUT/'READY.json',{'models':NAMES,'dataset_sha256':digest(DATA),'selection_sha256':selection_hash,'originals_unchanged':True,'version':'brfss2022_tuned'})
    print('COMPLETE: all originals and CSV unchanged; selections frozen before test evaluation.',flush=True)
if __name__=='__main__':main()
