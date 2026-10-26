"""Expanded real-respondent experiment, with frozen recall-constrained selection."""
import copy, hashlib, json, zipfile, collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
from src.model_paths import model_dir, LEGACY_ROOT
from src.brfss_inference import BRFSSPredictor
from training.train_baseline_models import ROOT,NUM,CAT,FIELDS,LIMITS,LABELS,NAMES,FILES,digest,dump
from training.tune_baseline_models import probability,adjusted

DATA=ROOT/'data/expanded_survey_dataset';OUT=ROOT/'models/current_models';REPORT=ROOT/'reports/brfss2022_expanded'
EXTRA={'isolation':(363,1,5,'SDHISOLT'),'job_loss':(364,1,2,'SDHEMPLY'),'food_insecurity':(366,1,5,'SDHFOOD1'),'bill_difficulty':(367,1,2,'SDHBILLS'),'utility_threat':(368,1,2,'SDHUTILS'),'transport_barrier':(369,1,2,'SDHTRNSP'),'medical_cost':(111,1,2,'MEDCOST1')}
CATEGORIES=CAT+list(EXTRA);FEATURES=NUM+CATEGORIES
def score(y,p):
    cm=np.bincount(3*y+p.argmax(1),minlength=9).reshape(3,3)
    recall=np.diag(cm)/np.maximum(1,cm.sum(1));precision=np.diag(cm)/np.maximum(1,cm.sum(0))
    f1=2*precision*recall/np.maximum(1e-15,precision+recall)
    return {'Accuracy':float(np.trace(cm)/cm.sum()),'Balanced accuracy':float(recall.mean()),'F1-Score (Macro)':float(f1.mean()),'Recall by class':recall.tolist(),'Precision by class':precision.tolist(),'confusion_matrix':cm.tolist()}
def groupkey(frame):
    # Hash canonical strings so original integer/float dtype differences cannot change groups.
    return frame[NUM+CAT].astype(float).apply(lambda r: hashlib.sha256((','.join(format(v,'.8g') for v in r)+'|expanded-20261018').encode()).hexdigest(),axis=1)
def prepare():
    old=pd.read_csv(ROOT/'data/baseline_survey_dataset/respondents_30000.csv');oldgroups=groupkey(old)
    prior_train=set(oldgroups[old.split=='train']);quarantine=set(oldgroups[old.split!='train'])
    assert not prior_train&quarantine
    rows=[];seen=set();invalid=collections.Counter()
    with zipfile.ZipFile(ROOT/'data/baseline_survey_dataset/raw/LLCP2022ASC.zip') as z,z.open(z.namelist()[0]) as f:
        for line_no,line in enumerate(f,1):
            assert len(line.rstrip(b'\r\n'))==2052
            key='2022_'+line[:2].decode()+'_'+line[35:45].decode().strip();assert key not in seen;seen.add(key)
            v={k:int(line[s-1:s+n-1].strip() or b'-1') for k,(s,n) in FIELDS.items()}
            v.update({k:int(line[s-1:s].strip() or b'-1') for k,(s,_,_,_) in EXTRA.items()})
            raw=v.copy()
            if v['unhealthy_days']==88:v['unhealthy_days']=0
            v['bmi']/=100
            limits={**LIMITS,**{k:(1,hi) for k,(_,_,hi,_) in EXTRA.items()}}
            bad=[k for k,(lo,hi) in limits.items() if not lo<=v[k]<=hi];invalid.update(bad)
            if bad:continue
            v.update(respondent_id=key,source_row=line_no,target=0 if v['stress_raw']>=4 else 1 if v['stress_raw']==3 else 2)
            v.update({'raw_'+k:value for k,value in raw.items()});rows.append(v)
    frame=pd.DataFrame(rows);groups=groupkey(frame)
    buckets=groups.map(lambda h:int(h[:8],16)%10)
    frame['split']=np.where(buckets<2,'test',np.where(buckets==2,'validation','train'))
    frame.loc[groups.isin(prior_train),'split']='train'
    frame.loc[groups.isin(quarantine),'split']='quarantine'
    frame['profile_group']=groups
    assert len(seen)==445132 and frame.respondent_id.is_unique
    assert frame.groupby('profile_group')['split'].nunique().max()==1
    assert not set(frame.loc[frame.split.isin(['test','validation']),'respondent_id'])&set(old.respondent_id)
    assert not set(groups[frame.split.isin(['test','validation'])])&set(oldgroups)
    frame.to_csv(DATA/'respondents.csv',index=False)
    audit={'source_records':len(seen),'eligible_records':len(frame),'unique_respondents':frame.respondent_id.nunique(),'new_respondents_not_in_original':int((~frame.respondent_id.isin(old.respondent_id)).sum()),'split_counts':frame.split.value_counts().to_dict(),'class_counts_by_split':pd.crosstab(frame.split,frame.target).to_dict(),'invalid_counts_overlapping':dict(invalid),'new_predictor_dictionary':EXTRA,'source_sha256':digest(ROOT/'data/baseline_survey_dataset/raw/LLCP2022ASC.zip'),'no_previous_respondents_or_profiles_in_validation_or_test':True}
    dump(REPORT/'dataset_audit.json',audit);print('DATA',audit,flush=True)
    return frame
def main():
    if (OUT/'READY.json').exists():raise SystemExit('Completed experiment exists; no test-driven reruns permitted.')
    for p in [DATA,OUT,REPORT]:p.mkdir(parents=True,exist_ok=True)
    protected=list(model_dir('brfss2022').glob('*.*'))+list(model_dir('brfss2022_tuned').glob('*.*'))+list(LEGACY_ROOT.glob('*.*'))+list((ROOT/'models').glob('*.*'))+list((ROOT/'data/original_sleep_dataset').glob('*.csv'))+[ROOT/'data/baseline_survey_dataset/respondents_30000.csv']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in protected};dump(REPORT/'preserved_hashes.json',hashes)
    dump(REPORT/'protocol.json',{'seed':20261018,'predictors':FEATURES,'excluded':['SDHSTRE1 and target-derived fields','MENTHLTH','LSATISFY','ADDEPEV3','respondent IDs and state'],'selection':'Highest validation accuracy satisfying high-frequency recall >= max(original,tuned)+0.02 and macro F1 >= max(original,tuned). Otherwise report candidate as ineligible.','adjustment_grid':{'sometimes':[.7,1,1.3],'usually_always':[.8,1,1.25,1.5,2,3]},'epochs_max':14,'early_stop_patience':4,'evaluation':'Freeze all six choices before test. Compare all three versions on identical new respondents. No test-based retuning. No deployment if high-frequency recall or accuracy is worse than either prior model on final holdout.','limitations':'New internal same-year holdout, not external validation; complete cases, unweighted US survey; concurrent associations, not future prediction.'})
    frame=prepare();tr=frame[frame.split=='train'];va=frame[frame.split=='validation']
    categories=[list(range(LIMITS[k][0],LIMITS[k][1]+1)) if k in CAT else list(range(1,EXTRA[k][2]+1)) for k in CATEGORIES]
    pre=ColumnTransformer([('numeric',StandardScaler(),NUM),('categorical',OneHotEncoder(categories=categories,sparse_output=False,handle_unknown='error'),CATEGORIES)])
    x=pre.fit_transform(tr[FEATURES]).astype('float32');v=pre.transform(va[FEATURES]).astype('float32')
    y=tr.target.to_numpy(copy=True);vy=va.target.to_numpy(copy=True)
    joblib.dump(pre,OUT/'preprocessor.joblib');dump(OUT/'schema.json',{'numeric':NUM,'categorical':CATEGORIES,'limits':{**LIMITS,**{k:[1,hi] for k,(_,_,hi,_) in EXTRA.items()}},'labels':LABELS,'input_dim':x.shape[1],'dataset':'Expanded BRFSS 2022','feature_count':len(FEATURES)})
    torch.set_num_threads(4);torch.manual_seed(20261018);np.random.seed(20261018)
    old=BRFSSPredictor('brfss2022');tuned=BRFSSPredictor('brfss2022_tuned')
    oldv=old.pre.transform(va[NUM+CAT]).astype('float32')
    weights=len(y)/(3*np.bincount(y,minlength=3));trials=[];selections={};models=[]
    for i,name in enumerate(NAMES):
        b1=score(vy,probability(old.models[name],oldv));b2=score(vy,adjusted(probability(tuned.models[name],oldv),tuned.selection[name]['multipliers']))
        floor=min(1,max(b1['Recall by class'][2],b2['Recall by class'][2])+.02);f1floor=max(b1['F1-Score (Macro)'],b2['F1-Score (Macro)'])
        bestkey=(-1,-1,-1);chosen=None;record=None
        def consider(model,desc):
            nonlocal bestkey,chosen,record
            probs=probability(model,v);changed=False
            for a in [.7,1,1.3]:
                for b in [.8,1,1.25,1.5,2,3]:
                    s=score(vy,adjusted(probs,[1,a,b]));eligible=s['Recall by class'][2]>=floor and s['F1-Score (Macro)']>=f1floor
                    key=(int(eligible),s['Accuracy'] if eligible else s['Recall by class'][2],s['F1-Score (Macro)'])
                    r={'model':name,'candidate':desc,'multipliers':[1,a,b],'validation':s,'eligible':eligible,'high_recall_floor':floor,'macro_f1_floor':f1floor}
                    trials.append(r)
                    if key>bestkey:bestkey=key;chosen=copy.deepcopy(model);record=r;changed=True
            return changed
        if i==0:candidates=[(f'C={c}',LogisticRegression(C=c,class_weight=dict(enumerate(weights**.5)),max_iter=1200,random_state=20261018)) for c in [.1,1]]
        elif i==1:candidates=[(f'depth={d}',RandomForestClassifier(n_estimators=250,max_depth=d,min_samples_leaf=12,max_features=.8,class_weight='balanced',n_jobs=4,random_state=20261018)) for d in [12,20]]
        elif i==2:candidates=[(f'depth={d}',XGBClassifier(n_estimators=350,max_depth=d,learning_rate=.04,min_child_weight=12,reg_lambda=5,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=20261018,eval_metric='mlogloss')) for d in [3,5]]
        if i<3:
            for desc,model in candidates:
                print('TRAIN',name,desc,flush=True);model.fit(x,y,**({'sample_weight':weights[y]**.5} if i==2 else {}));consider(model,desc)
        else:
            model=[Conv1DNet,LSTMNet,CNNLSTMNet][i-3](input_dim=x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.001)
            loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights**.5,dtype=torch.float32));stale=0
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=512,shuffle=True)
            for epoch in range(1,15):
                model.train()
                for bx,by in loader:
                    optimizer.zero_grad();loss=loss_fn(model(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                improved=consider(model,f'epoch {epoch}');stale=0 if improved else stale+1
                print(name,epoch,'selected',record['validation'],'eligible',record['eligible'],flush=True)
                if stale>=4:break
        if i<3:joblib.dump(chosen,OUT/FILES[i])
        else:torch.save(chosen.state_dict(),OUT/FILES[i])
        selections[name]=record;models.append(chosen);dump(OUT/'selection.json',selections);dump(REPORT/'validation_trials.json',trials)
        print('FROZEN',name,record,flush=True)
    selectionhash=digest(OUT/'selection.json')
    te=frame[frame.split=='test'];t=pre.transform(te[FEATURES]).astype('float32');oldt=old.pre.transform(te[NUM+CAT]).astype('float32');ty=te.target.to_numpy(copy=True)
    comparisons=[];metrics=[];pred={'respondent_id':te.respondent_id.to_numpy(),'target':ty};approved=[]
    for i,name in enumerate(NAMES):
        versions={'original':probability(old.models[name],oldt),'tuned':adjusted(probability(tuned.models[name],oldt),tuned.selection[name]['multipliers']),'expanded':adjusted(probability(models[i],t),selections[name]['multipliers'])}
        scores={k:{**score(ty,p),'ROC-AUC (Macro)':float(roc_auc_score(ty,p,multi_class='ovr'))} for k,p in versions.items()}
        passed=selections[name]['eligible'] and all(scores['expanded']['Recall by class'][2]>=scores[k]['Recall by class'][2] and scores['expanded']['Accuracy']>=scores[k]['Accuracy'] for k in ['original','tuned'])
        comparisons.append({'Model':name,**scores,'passes_no_more_high_stress_misses_and_accuracy_gate':bool(passed)});metrics.append({'Model':name,**scores['expanded']})
        if passed:approved.append(name)
        for ver,p in versions.items():
            pred[f'{name} {ver}']=p.argmax(1)
            for c in range(3):pred[f'{name} {ver} probability {c}']=p[:,c]
        print('FINAL',comparisons[-1],flush=True)
    assert digest(OUT/'selection.json')==selectionhash
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    pd.DataFrame(pred).to_csv(REPORT/'test_predictions.csv',index=False)
    dump(REPORT/'comparison.json',comparisons);dump(OUT/'metrics.json',metrics)
    dump(OUT/'READY.json',{'version':'brfss2022_expanded','dataset_sha256':digest(DATA/'respondents.csv'),'selection_sha256':selectionhash,'models':NAMES,'approved_models':approved,'all_six_pass':len(approved)==6,'originals_unchanged':True,'active_version_not_changed':True})
    print('COMPLETE; approved:',approved,'Existing dashboard unchanged.',flush=True)
if __name__=='__main__':main()
