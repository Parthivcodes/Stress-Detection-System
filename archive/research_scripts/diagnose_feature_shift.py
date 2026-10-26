"""Read-only missingness/outlier audit and feature-only adversarial validation."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from tune_six_models import read,save,preprocessor

def run(root,out):
    if (out/'feature_shift.json').exists():raise RuntimeError('Do not overwrite diagnosis.')
    schema=read(root/'models/current_models/schema.json');features=schema['numeric']+schema['categorical'];df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')];va=df[df.split.eq('validation')]
    save(out/'feature_shift_protocol.json',dict(seed=20261026,source='Originaltrain and validation only; no stress outcome used for domain classifier.',domain='Balanced sample15197 train vs all15197 validation,3stratifiedgroup folds,150tree depth3 XGB,2threads. Preprocessing fit within domain-training folds. Distinct from stress model training; no stress labels/features derived from target.',outliers='Training-only numeric IQR*1.5 and1st/99th percentile thresholds. Counts only; no rows removed or values changed.',missing='Report original complete-case selection exclusions. Missingness mechanism cannot be proved random from retained records.'))
    numeric={}
    for n in schema['numeric']:
        q1,q3=tr[n].quantile([.25,.75]);iqr=q3-q1;low,high=q1-1.5*iqr,q3+1.5*iqr;p1,p99=tr[n].quantile([.01,.99])
        numeric[n]=dict(min=float(tr[n].min()),max=float(tr[n].max()),iqr_lower=float(low),iqr_upper=float(high),percentile1=float(p1),percentile99=float(p99),train_iqr_flags=int((~tr[n].between(low,high)).sum()),validation_iqr_flags=int((~va[n].between(low,high)).sum()),note='Outlier flags do not establish data errors; survey codes and meaningful extremes remain intact.')
    raw_audit=read(root/'reports/brfss2022_expanded/dataset_audit.json')
    domain=pd.concat([tr.sample(len(va),random_state=20261026).assign(domain=0),va.assign(domain=1)],ignore_index=True);y=domain.domain.to_numpy();pred=np.zeros(len(domain));scores=[]
    for fold,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=20261026).split(domain,y,domain.profile_group)):
        assert not set(domain.iloc[a].profile_group)&set(domain.iloc[b].profile_group)
        pre=preprocessor(schema).fit(domain.iloc[a][features]);m=XGBClassifier(n_estimators=150,max_depth=3,min_child_weight=20,learning_rate=.05,reg_lambda=5,n_jobs=2,random_state=20261026,eval_metric='logloss');m.fit(pre.transform(domain.iloc[a][features]),y[a]);pred[b]=m.predict_proba(pre.transform(domain.iloc[b][features]))[:,1];scores.append(float(roc_auc_score(y[b],pred[b])))
    result=dict(numeric=numeric,missing_predictor_values_in_retained_table=int(df[features].isna().sum().sum()),raw_source_records=raw_audit['source_records'],retained_records=raw_audit['eligible_records'],raw_invalid_or_missing_counts_overlapping=raw_audit['invalid_counts_overlapping'],adversarial_auc=float(roc_auc_score(y,pred)),fold_auc=scores,domain_rows=len(domain),note='AUC near0.5 indicates no readily detected train/validation feature shift under this classifier, not absence of all drift. No test drift evaluated. No imputation or outlier edits performed.')
    save(out/'feature_shift.json',result);print(json.dumps(dict(domain_auc=result['adversarial_auc'],fold_auc=scores,numeric=numeric)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
