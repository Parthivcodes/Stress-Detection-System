"""Phase-one diagnosis: immutable originals, no test evaluation or label editing."""
import argparse,copy,gc,hashlib,json,os,resource,shutil,sys,time
from collections import Counter
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.base import clone
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tune_six_models import read,save,sha,metric,preprocessor,partition,prob,NAMES,FILES

SEEDS=[20261026,20261027]
def protect():
    for n,h in read(OUT/'original_hashes.json').items():assert sha(ROOT/n)==h,n
def subsets(tr):
    # Nested group-hash subsets, independent of labels/correctness. Roles fixed first.
    ft,dev=partition(tr,10,SEEDS[0]);fraction=ft.profile_group.map(lambda s:int(s[:8],16)/2**32)
    return ft,dev,[(name,ft[fraction<cut].copy()) for name,cut in [('small',.087),('medium',.35),('full',1.)]]
def initialize():
    if OUT.exists():raise RuntimeError('Use a fresh experiment directory.')
    OUT.mkdir(parents=True)
    paths=[p for p in MODELS.rglob('*') if p.is_file()]+[ROOT/n for n in ['models/active_model.json','src/models.py','src/brfss_inference.py','src/ensemble_policy.py','data/expanded_survey_dataset/respondents.csv','training/train_expanded_models.py']]
    save(OUT/'original_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for p in paths:
        if p.suffix=='.csv':continue
        dest=OUT/'backup'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    save(OUT/'protocol.json',dict(phase=1,seeds=SEEDS,features=FEATURES,target={'0':'rarely/never','1':'sometimes','2':'usually/always'},
      scope='Diagnosis first. Preserve algorithms and production. No test evaluation, threshold optimization, label changes, row deletion or promotion in phase1.',
      baseline='Reload six active artifacts; raw and current-decision metrics on full existing validation. Compare saved probabilities. Deterministic5000-row training diagnostic sample for gap. Historical validation reused, not external.',
      curves='Original training divided into grouped90%fit/10%early-stop dev once. Nested group-hash cutoffs0.087/0.35/1.0 of fit(~10k/40k/115k). Actual sizes and class counts recorded. Same sets for every model/seed. Fit-only preprocessing. Classical active hyperparameters, seed20261026; neural current architecture2seeds,14epochmax,patience4,120secsoftcapcheckedbetweenepochs. Unweighted devlogloss selects checkpoint. No validation-based epoch/model selection.',
      curve_metrics='Training sample up to5000 per fit vs full validation. Rawargmax and frozen active multipliers, accuracy/F1/AUC/recall/confusion matrices; no threshold tuning. Neural mean/std over2seeds, classical one seed. Short/time-limited runs cannot prove a learning ceiling.',
      ambiguity='Count exact-ID duplicates, identical19feature profiles and conflicting labels. Near-duplicate proxy: samecategories, age5yearbins,BMI1unitbins,sleep1hour,physicaldays5daybins; not proof of sameperson or erroneouslabels. Use previously saved training-only group-OOF XGB predictions for confidence disagreement review, not a cleanlab-estimated label-error rate. No relabel/delete without source evidence.',
      drift='Optional train-vs-validation domain classifier uses no target label and grouped roles. Test excluded.',
      ceiling='Cannot identify Bayes/noise ceiling from these diagnostics. Empirical conflicting-profile errors constrain exact matching on observed records only, not future attainable accuracy. Majority baseline65.78% historicalholdout and per-partition baselines.',
      next='Use diagnosis to prioritize Phases2-6; missing-value methods inapplicable to complete-case table. Broader sweeps/5x2CV not yet performed; do not claim them. No untouched test exists in this project.',
      compute=dict(cpu_threads=4,logical_cpus=os.cpu_count(),cuda=torch.cuda.is_available(),memory=Path('/proc/meminfo').read_text().splitlines()[:3],budget='9classicalfits+18neuralfits; bounded pilot, no paidcompute/new dependencies.'))) 
    shutil.copyfile(__file__,OUT/'diagnose_six_models.py');save(OUT/'frozen_hashes.json',{n:sha(OUT/n) for n in ['protocol.json','diagnose_six_models.py']})
    print('PROTOCOL FROZEN',flush=True)

def load_active(i):
    if i<3:return joblib.load(MODELS/FILES[i])
    m=CLASSES[i-3](input_dim=SCHEMA['input_dim']);m.load_state_dict(torch.load(MODELS/FILES[i],map_location='cpu',weights_only=True));return m
def metrics(y,p,n):return dict(raw=metric(y,p),current_rule=metric(y,p,(p*SEL[n]['multipliers']).argmax(1)),logloss=float(log_loss(y,p,labels=[0,1,2])))
def audit_data(df,tr,va):
    assert df.respondent_id.is_unique and df.groupby('profile_group').split.nunique().max()==1
    assert not df[FEATURES].isna().any().any()
    for k in FEATURES:
        lo,hi=SCHEMA['limits'][k];assert df[k].between(lo,hi).all()
    assert np.array_equal(df.target,np.where(df.stress_raw>=4,0,np.where(df.stress_raw==3,1,2)))
    # No test outcomes used: conflict review is training only.
    cnt=tr.groupby(FEATURES,dropna=False).target.agg(['size','nunique'])
    tab=tr.groupby(FEATURES+['target'],dropna=False).size();best=tab.groupby(level=list(range(len(FEATURES)))).max()
    irreducible_observed=int(len(tr)-best.sum())
    near=tr[FEATURES].copy();near.age=near.age//5;near.bmi=np.floor(near.bmi);near.unhealthy_days=near.unhealthy_days//5
    near['target']=tr.target.to_numpy();ng=near.groupby(FEATURES,dropna=False).target.agg(['size','nunique'])
    result=dict(rows=len(df),unique_ids=int(df.respondent_id.nunique()),split_counts=df.split.value_counts().to_dict(),training_class_counts=tr.target.value_counts().to_dict(),validation_class_counts=va.target.value_counts().to_dict(),missing_predictors=0,groups_crossing_splits=0,training_exact_duplicate_feature_extra_rows=int(tr.duplicated(FEATURES).sum()),training_conflicting_feature_groups=int(cnt.nunique if False else (cnt['nunique']>1).sum()),training_rows_in_conflicting_groups=int(cnt.loc[cnt['nunique']>1,'size'].sum()),observed_exact_profile_min_errors=irreducible_observed,observed_profile_lookup_max_accuracy=1-irreducible_observed/len(tr),near_bin_groups_with_multiple_rows=int((ng['size']>1).sum()),near_bin_conflicting_groups=int((ng['nunique']>1).sum()),near_definition='Same19categories/numericbins; age5,BMI1,physicaldays5; NOT respondent deduplication rule.',note='Observed lookup maximum is an in-sample combinatorial fact, NOT a generalization or noise ceiling. All original rows retained.')
    save(OUT/'data_audit.json',result)
    conflicting=tr.merge(cnt[cnt['nunique']>1].reset_index()[FEATURES],on=FEATURES,how='inner');conflicting.to_csv(OUT/'training_conflicting_profiles.csv',index=False)
    source=ROOT/'experiments/predictor_audit_20261025';old=read(source/'original_hashes.json')
    for n,h in old.items():
        n=n.replace('\\','/')
        if n=='data/expanded_survey_dataset/respondents.csv':assert sha(ROOT/n)==h
    oof=np.load(source/'pilot_oof.npz');assert np.array_equal(oof['respondent_id'],tr.respondent_id)
    p=oof['baseline'];y=tr.target.to_numpy();pred=p.argmax(1);truep=p[np.arange(len(y)),y];maxp=p.max(1)
    flagged=(pred!=y)&(maxp>=.8)&(truep<=.1)
    sample=tr.loc[flagged,['respondent_id','profile_group']+FEATURES+['target','stress_raw']].copy();sample['oof_predicted']=pred[flagged];sample['oof_confidence']=maxp[flagged];sample['oof_observed_label_probability']=truep[flagged]
    sample.sort_values('oof_observed_label_probability').head(30).to_csv(OUT/'ambiguity_review_sample.csv',index=False)
    save(OUT/'ambiguity.json',dict(method='Predeclared high-confidence group-OOF disagreement; NOT cleanlab/confident-learning noise estimation.',rows=len(tr),flags=int(flagged.sum()),flag_fraction=float(flagged.mean()),thresholds=dict(max_predicted=.8,max_observed_label=.1),class_flag_counts={str(c):int(np.sum(flagged&(y==c))) for c in range(3)},source_probability_sha256=sha(source/'pilot_oof.npz'),interpretation='Can reflect missing predictors, model bias, noisy answers or legitimate differences. Ground-truth errors not established. No labels changed.'))
    print('DATA AUDIT',json.dumps(result),flush=True)

def baseline(tr,va):
    pre=joblib.load(MODELS/'preprocessor.joblib');fitpre=preprocessor(SCHEMA).fit(tr[FEATURES]);assert np.allclose(pre.named_transformers_['numeric'].mean_,fitpre.named_transformers_['numeric'].mean_)
    sample=tr.sample(min(5000,len(tr)),random_state=SEEDS[0]);tx=pre.transform(sample[FEATURES]).astype('float32');vx=pre.transform(va[FEATURES]).astype('float32');saved=dict(np.load(ROOT/'experiments/hyperparameter_tuning_20261023/original_validation.npz'));result={}
    for i,n in enumerate(NAMES):
        m=load_active(i);vp=prob(m,vx);tp=prob(m,tx);assert np.allclose(vp,saved[n],atol=1e-6),n
        result[n]=dict(train_diagnostic=metrics(sample.target.to_numpy(),tp,n),validation=metrics(va.target.to_numpy(),vp,n));del m;gc.collect();print('BASELINE VERIFIED',n,flush=True)
    save(OUT/'baseline.json',dict(models=result,training_sample_rows=len(sample),validation_rows=len(va),majority_validation_accuracy=float(va.target.eq(0).mean()),note='Reloaded active models. Validation historically reused; train sample is in-sample descriptive.'))

def train(i,frame,dev,seed):
    start=time.monotonic();pre=preprocessor(SCHEMA).fit(frame[FEATURES]);x=pre.transform(frame[FEATURES]).astype('float32');y=frame.target.to_numpy(copy=True);dx=pre.transform(dev[FEATURES]).astype('float32');dy=dev.target.to_numpy(copy=True);w=(len(y)/(3*np.bincount(y,minlength=3)))**.5;history=[];capped=False
    torch.manual_seed(seed);np.random.seed(seed)
    if i<3:
        m=clone(joblib.load(MODELS/FILES[i]));params=dict(random_state=seed)
        if 'n_jobs' in m.get_params():params['n_jobs']=4
        if i==0:params['class_weight']=dict(enumerate(w))
        m.set_params(**params);m.fit(x,y,**({'sample_weight':w[y]} if i==2 else {}));bestepoch=None
    else:
        m=CLASSES[i-3](input_dim=x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.0007,weight_decay=.001);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32));loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=512,shuffle=True)
        best=float('inf');state=None;stale=0;bestepoch=0
        for epoch in range(1,15):
            m.train();total=0.
            for bx,by in loader:
                opt.zero_grad();loss=lossfn(m(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=float(loss.detach())*len(by)
            ll=float(log_loss(dy,prob(m,dx),labels=[0,1,2]));history.append(dict(epoch=epoch,weighted_train_loss=total/len(y),dev_logloss=ll))
            if ll<best-1e-5:best=ll;bestepoch=epoch;state=copy.deepcopy(m.state_dict());stale=0
            else:stale+=1
            if epoch%2==0:print('EPOCH',NAMES[i],len(frame),seed,epoch,round(ll,4),flush=True)
            capped=time.monotonic()-start>=120
            if stale>=4 or capped:break
        m.load_state_dict(state)
    return pre,m,dict(seconds=time.monotonic()-start,history=history,best_epoch=bestepoch,time_cap_reached=capped,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def curves(tr,va):
    ft,dev,sets=subsets(tr);assert not set(ft.profile_group)&set(dev.profile_group) and not set(tr.profile_group)&set(va.profile_group)
    membership=ft[['respondent_id','profile_group']].copy();membership['size_membership']=membership.respondent_id.map(lambda i:','.join(n for n,f in sets if i in IDSETS[n])) if False else 'fit'
    pd.concat([f[['respondent_id','profile_group']].assign(role=n) for n,f in sets]+[dev[['respondent_id','profile_group']].assign(role='early_stop')]).to_csv(OUT/'learning_curve_roles.csv',index=False)
    records=[]
    for size,frame in sets:
        for i,n in enumerate(NAMES):
            for seed in SEEDS if i>=3 else SEEDS[:1]:
                path=OUT/f'curve_{size}_{i}_{seed}.json'
                if path.exists():records.append(read(path));continue
                print('CURVE FIT',n,size,len(frame),seed,flush=True);pre,m,info=train(i,frame,dev,seed)
                sample=frame.sample(min(5000,len(frame)),random_state=SEEDS[0]);tp=prob(m,pre.transform(sample[FEATURES]).astype('float32'));vp=prob(m,pre.transform(va[FEATURES]).astype('float32'))
                r=dict(model=n,size=size,rows=len(frame),class_counts=frame.target.value_counts().to_dict(),seed=seed,train=metrics(sample.target.to_numpy(),tp,n),validation=metrics(va.target.to_numpy(),vp,n),fit=info)
                save(path,r);records.append(r);print('CURVE RESULT',n,size,seed,'train',r['train']['raw']['accuracy'],'val',r['validation']['raw']['accuracy'],flush=True);del m,pre;gc.collect()
    save(OUT/'learning_curves.json',records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','diagnose','curves']);p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical'];SEL=read(MODELS/'selection.json')
    sys.path.insert(0,str(ROOT));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];torch.set_num_threads(4)
    if a.phase=='initialize':initialize()
    else:
        protect();df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')].copy();va=df[df.split.eq('validation')].copy()
        if a.phase=='diagnose':audit_data(df,tr,va);baseline(tr,va)
        else:curves(tr,va)
        protect();print('PHASE COMPLETE',a.phase,flush=True)
