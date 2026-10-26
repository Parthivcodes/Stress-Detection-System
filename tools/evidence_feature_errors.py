"""Descriptive feature and subgroup diagnostics. Never selects on test data."""
import argparse,json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from xgboost import DMatrix
from tune_six_models import read,save,sha,partition,preprocessor,metric,NAMES

def main(root,out):
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(seed=20261026,scope='Diagnostics only; no data/model changes or adaptive candidate selection.',
      mi='Mutual information on deterministic10000 originaltraining rows; categorical inputs discrete,numericcontinuous. Association not causation.',
      permutation='Refit originalXGB in first outer training fold with innerdevelopment/calibration excluded exactly as boundedrun. First5000 deterministic outer-score sample,3permutations per rawcolumn. Logloss increase; no stress validation/test used.',
      shap='Native XGBoost TreeSHAP rawmargin contributions on500 held-out training-fold respondents. Aggregate absolute encoded contributions by originalfeature. Correlatedfeatures complicate attribution; not causal.',
      errors='Reproduced current6 validation probabilities only. Raw and currentdecision error overlap,age/sex/support subgroups. Historically reused validation,descriptive only. Do not claim person-specific causes or edit labels.'))
    models=root/'models/current_models';s=read(models/'schema.json');features=s['numeric']+s['categorical'];df=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')].reset_index(drop=True);va=df[df.split.eq('validation')]
    sample=tr.sample(10000,random_state=20261026)
    mi=mutual_info_classif(sample[features],sample.target,discrete_features=[False]*len(s['numeric'])+[True]*len(s['categorical']),random_state=20261026,n_jobs=1)
    save(out/'mutual_information.json',dict(sorted(zip(features,mi.tolist()),key=lambda kv:-kv[1])))
    a,b=next(StratifiedGroupKFold(2,shuffle=True,random_state=20261026).split(tr,tr.target,tr.profile_group));pool,cal=partition(tr.iloc[a],5,20261026);ft,dev=partition(pool,8,20261026);sc=tr.iloc[b].sample(5000,random_state=20261026)
    pre=preprocessor(s).fit(ft[features]);m=clone(joblib.load(models/'xgboost.joblib')).set_params(n_jobs=1,random_state=20261026);y=ft.target.to_numpy();w=(len(y)/(3*np.bincount(y)))**.5;m.fit(pre.transform(ft[features]),y,sample_weight=w[y]);base=log_loss(sc.target,m.predict_proba(pre.transform(sc[features])),labels=[0,1,2]);rng=np.random.default_rng(20261026);permutation={}
    for k in features:
        values=[]
        for repeat in range(3):
            f=sc[features].copy();f[k]=rng.permutation(f[k].to_numpy());values.append(float(log_loss(sc.target,m.predict_proba(pre.transform(f)),labels=[0,1,2])-base))
        permutation[k]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),repeats=values)
    save(out/'permutation_importance.json',dict(base_logloss=float(base),rows=len(sc),features=permutation))
    contrib=m.get_booster().predict(DMatrix(pre.transform(sc[features].iloc[:500]).astype('float32')),pred_contribs=True)
    assert contrib.shape==(500,3,s['input_dim']+1)
    names=list(s['numeric'])+[k for k in s['categorical'] for _ in range(s['limits'][k][0],s['limits'][k][1]+1)]
    importance={k:float(np.mean(np.sum(np.abs(contrib[:,:,:-1][:,:,np.array(names)==k]),axis=2))) for k in features}
    save(out/'tree_shap.json',dict(rows=500,raw_margin_mean_absolute_group_contribution=importance,aggregation='Sumabsolute dummy-contributions then meanoverpeople/classes; redundantfeatures share attribution.'))
    raw=np.load(root/'experiments/hyperparameter_tuning_20261023/original_validation.npz');sel=read(models/'selection.json');yv=va.target.to_numpy();pp=np.column_stack([(raw[n]*sel[n]['multipliers']).argmax(1) for n in NAMES]);rp=np.column_stack([raw[n].argmax(1) for n in NAMES]);errors={}
    for kind,p in [('current_decision',pp),('raw_argmax',rp)]:
        wrong=p!=yv[:,None];errors[kind]=dict(all_six_wrong=int(wrong.all(1).sum()),all_six_wrong_by_class={str(c):dict(count=int((wrong.all(1)&(yv==c)).sum()),class_size=int((yv==c).sum()),fraction=float(wrong.all(1)[yv==c].mean())) for c in range(3)},prediction_correlation=np.corrcoef(p.T).tolist(),note='Correlation of ordinal predictedlabels; descriptive and not probabilitycalibration.')
    groups={'all':np.ones(len(va),dtype=bool),'age18_39':va.age.lt(40).to_numpy(),'age40_59':va.age.between(40,59).to_numpy(),'age60plus':va.age.ge(60).to_numpy(),'sex_code1':va.sex.eq(1).to_numpy(),'sex_code2':va.sex.eq(2).to_numpy(),'low_support':va.support.ge(4).to_numpy(),'other_support':va.support.lt(4).to_numpy()}
    sub={}
    for g,mask in groups.items():
        sub[g]=dict(rows=int(mask.sum()),class_counts=np.bincount(yv[mask],minlength=3).tolist(),majority_accuracy=float(np.mean(yv[mask]==0)),models={n:metric(yv[mask],raw[n][mask],pp[mask,i]) for i,n in enumerate(NAMES)})
    save(out/'error_overlap.json',errors);save(out/'subgroups.json',sub)
    save(out/'hashes.json',dict(data=sha(root/'data/expanded_survey_dataset/respondents.csv'),source=sha(__file__),validation_probabilities=sha(root/'experiments/hyperparameter_tuning_20261023/original_validation.npz')))
    print('Feature rankings, TreeSHAP, error overlap and subgroups saved.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.project),Path(a.output))
