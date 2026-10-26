"""Transparent from-scratch SIMULATION; never evidence of real-person accuracy.

Only schema/configuration files are read from the real project, never respondents.
No synthetic model is activated by this script.
"""
import argparse,copy,gc,hashlib,json,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tune_six_models import read,save,sha,metric,preprocessor,prob,NAMES,FILES
from tune_two_boundaries import boundaries,predict

SEED=20261026

def generate(n,noise):
    r=np.random.default_rng(SEED);strain=r.normal(size=n);health=r.normal(size=n);social=.4*strain+r.normal(size=n)
    ordinal=lambda z,size:np.clip(np.rint((size+1)/2+z),1,size).astype(int)
    yes=lambda z:np.where(r.random(n)<1/(1+np.exp(-z)),1,2)
    age=np.clip(np.rint(r.normal(45,16,n)),18,80).astype(int)
    f=pd.DataFrame(dict(age=age,sleep_duration=np.clip(np.rint(7.5-.65*strain-.25*health+r.normal(0,1,n)),3,12).astype(int),
      unhealthy_days=np.where(r.random(n)<.55,0,np.clip(np.rint(np.exp(1.7+.65*health+r.normal(0,.8,n))),1,30)).astype(int),
      bmi=np.round(np.clip(26+3*health+r.normal(0,4,n),16,48),2),sex=r.integers(1,3,n),exercise=yes(.6-.35*health),
      general_health=ordinal(.6*health+.2*strain+r.normal(0,.7,n),5),
      marital=r.choice(np.arange(1,7),n,p=[.43,.13,.07,.05,.25,.07]),education=ordinal(r.normal(.9,1.2,n),6),
      employment=r.choice(np.arange(1,9),n,p=[.48,.11,.05,.04,.07,.08,.12,.05]),
      income=ordinal(-1.2*strain+r.normal(0,1.8,n),11),support=ordinal(.6*social+r.normal(0,.8,n),5),
      isolation=ordinal(-.65*social+r.normal(.4,.8,n),5),job_loss=yes(-1.8+.6*strain),
      food_insecurity=ordinal(-.6*strain+r.normal(.7,.9,n),5),bill_difficulty=yes(-1.6+.8*strain),
      utility_threat=yes(-2+.8*strain),transport_barrier=yes(-1.9+.6*strain),medical_cost=yes(-1.3+.6*strain)))
    material=sum((f[k]==1).astype(float) for k in ['job_loss','bill_difficulty','utility_threat','transport_barrier','medical_cost'])
    # Hypothetical equation, NOT learned/validated medical relationships.
    signal=.85*(8-f.sleep_duration)/2+.7*(f.general_health-3)/2+.65*(f.support-3)/2+.85*(3-f.isolation)/2+.5*(3-f.food_insecurity)/2+.25*f.unhealthy_days/10+.5*(material-1)-.15*(f.age-45)/25+.15*(f.exercise==2)+.1*(f.bmi-26)/5+.1*(6-f.income)/3
    latent=signal.to_numpy()+r.normal(0,noise,n)
    f['target']=np.digitize(latent,[-.65,.85]);f['sample_id']=[f'SYN-{i:06d}' for i in range(1,n+1)];f['data_origin']='synthetic_simulation'
    f['split']='train';a,b=train_test_split(np.arange(n),test_size=.30,random_state=SEED,stratify=f.target);v,t=train_test_split(b,test_size=.5,random_state=SEED+1,stratify=f.target.iloc[b]);f.loc[v,'split']='validation';f.loc[t,'split']='test'
    return f

def fit(i,ft,dev,seed):
    torch.manual_seed(seed);np.random.seed(seed);start=time.monotonic();pre=preprocessor(SCHEMA).fit(ft[FEATURES]);x=pre.transform(ft[FEATURES]).astype('float32');dx=pre.transform(dev[FEATURES]).astype('float32');y=ft.target.to_numpy(copy=True);dy=dev.target.to_numpy(copy=True);w=(len(y)/(3*np.bincount(y,minlength=3)))**.5
    history=[];bestepoch=None
    if i<3:
        m=clone(joblib.load(MODELS/FILES[i]));params=dict(random_state=seed)
        if 'n_jobs' in m.get_params():params['n_jobs']=4
        if i==0:params['class_weight']=dict(enumerate(w))
        m.set_params(**params);m.fit(x,y,**({'sample_weight':w[y]} if i==2 else {}))
    else:
        m=CLASSES[i-3](input_dim=x.shape[1]);optimizer=torch.optim.AdamW(m.parameters(),lr=.0007,weight_decay=.001);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32));loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=256,shuffle=True)
        best=float('inf');stale=0;state=None
        for epoch in range(1,41):
            m.train()
            for bx,by in loader:
                optimizer.zero_grad();loss=lossfn(m(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);optimizer.step()
            ll=float(log_loss(dy,prob(m,dx),labels=[0,1,2]));history.append(dict(epoch=epoch,development_logloss=ll))
            if ll<best-1e-5:best=ll;state=copy.deepcopy(m.state_dict());bestepoch=epoch;stale=0
            else:stale+=1
            if epoch%10==0:print('SYNTHETIC EPOCH',NAMES[i],seed,epoch,round(ll,4),flush=True)
            if stale>=6 or time.monotonic()-start>=120:break
        m.load_state_dict(state)
    return pre,m,dict(seconds=time.monotonic()-start,seed=seed,history=history,best_epoch=bestepoch)

def main():
    if OUT.exists():raise RuntimeError('Use fresh experiment directory; never overwrite.')
    OUT.mkdir(parents=True);paths=list(MODELS.rglob('*'))+[ROOT/'models/active_model.json',ROOT/'data/expanded_survey_dataset/respondents.csv'];protected={p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.is_file()};save(OUT/'protected_real_artifacts.json',protected)
    save(OUT/'protocol.json',dict(data_origin='SYNTHETIC SIMULATION - NOT REAL RESPONDENTS',rows=ARGS.rows,seed=SEED,neural_seeds=[SEED,SEED+1],scenario=ARGS.scenario,label_noise_sd=NOISE,
      purpose='Independent educational simulation with19compatible inputs. Synthetic test accuracy measures recovery of inventedlabelrules,not realstressprediction. Never replace real-model metrics or activate these artifacts.',
      source='Fromscratch fixed distributions/sharedlatentvariables; no BRFSS respondent rows read during generation/training. Only feature coding,architectures and classical model hyperparameters reused.',
      label_equation='0.85*(8-sleep)/2 +0.7*(general_health-3)/2 +0.65*(support-3)/2 +0.85*(3-isolation)/2 +0.5*(3-food_insecurity)/2 +0.25*physical_days/10 +0.5*(material_count-1) -0.15*(age-45)/25 +0.15*no_exercise +0.1*(BMI-26)/5 +0.1*(6-income)/3 + Normal(0,noiseSD); classes at fixed boundaries -0.65,+0.85. These coefficients are invented; not clinically validated.',
      split='Seeded stratified70/15/15. Assert all19feature profiles unique and nonoverlapping across roles. Train90%fit/10%earlystopdev. Fitpreprocessingonfit only. No training on validation/test.',
      fitting='Three classical configs cloned from protected currentmodels; neural originalarchitecture,AdamW .0007 decay.001,weightedCE sqrt inverse prevalence,40epochmax,patience6,120secsoftcap. Two fixedseeds averaged; neitherbestseedchosen.',
      decisions='Rawargmax and optional two-boundary operatingrule selectedonvalidation. Require highrecall>=max(.60,rawvalidationhigh) and sometimes/F1/QWK at least rawvalidation. If infeasible keepraw and disclose. Freezeallcheckpoints/policies before one synthetictestpass. No results-driven regeneration/noise changes.',
      evaluation='Synthetic test ONLY. No real holdout evaluation or promotion. No claim of improvement versus real-data69%. Majority baseline specific to synthetic classmix.',
      scenarios={'simulation':1.,'toy_easy':.25,'stress_test':1.6},features=FEATURES))
    shutil.copyfile(__file__,OUT/'synthetic_stress_experiment.py');save(OUT/'frozen_hashes.json',{'protocol.json':sha(OUT/'protocol.json'),'synthetic_stress_experiment.py':sha(__file__)})
    df=generate(ARGS.rows,NOISE);assert not df[FEATURES].isna().any().any();assert not df.duplicated(FEATURES).any(),'Duplicate featureprofiles: stop rather than leak across splits.'
    for k in FEATURES:
        lo,hi=SCHEMA['limits'][k];assert df[k].between(lo,hi).all()
    csv=OUT/'model_training_data.csv';df.to_csv(csv,index=False)
    save(OUT/'data_manifest.json',dict(csv=csv.name,sha256=sha(csv),data_origin='synthetic_simulation',rows=len(df),split_counts=df.split.value_counts().to_dict(),class_counts={s:df[df.split.eq(s)].target.value_counts().sort_index().to_dict() for s in ['train','validation','test']},unique_ids=bool(df.sample_id.is_unique),duplicate_feature_profiles=0,labels=['Rarely / never','Sometimes','Usually / always'],raw_record_source='None;invented probabilisticgenerator'))
    train=df[df.split.eq('train')];va=df[df.split.eq('validation')];fa,da=train_test_split(np.arange(len(train)),test_size=.1,random_state=SEED,stratify=train.target);ft=train.iloc[fa];dev=train.iloc[da];policies={};valid={};test_models={}
    pd.concat([ft[['sample_id']].assign(role='fit'),dev[['sample_id']].assign(role='early_stop')]).to_csv(OUT/'training_roles.csv',index=False)
    for i,n in enumerate(NAMES):
        predictions=[];infos=[];seed_metrics=[];bundles=[]
        for seed in [SEED,SEED+1] if i>=3 else [SEED]:
            print('TRAIN SYNTHETIC',n,seed,flush=True);pre,m,info=fit(i,ft,dev,seed);p=prob(m,pre.transform(va[FEATURES]).astype('float32'));predictions.append(p);seed_metrics.append(metric(va.target.to_numpy(),p));infos.append(info)
            folder=OUT/'models';folder.mkdir(exist_ok=True);stem=folder/f'{i}_{seed}';joblib.dump(pre,stem.with_suffix('.preprocessor.joblib'))
            if i<3:joblib.dump(m,stem.with_suffix('.joblib'))
            else:torch.save(m.state_dict(),stem.with_suffix('.pt'))
            bundles.append((pre,m))
        p=np.mean(predictions,axis=0);y=va.target.to_numpy();raw=metric(y,p);policy=boundaries(y,p,raw);policies[n]=policy['thresholds'] if policy['feasible'] else None
        valid[n]=dict(raw=raw,operating=policy['metrics'] if policy['feasible'] else raw,policy_feasible=policy['feasible'],fit=infos,per_seed_raw=seed_metrics);test_models[n]=bundles
        save(OUT/'validation_metrics.json',valid);print('SYNTHETIC VALIDATION',n,raw['accuracy'],flush=True)
    save(OUT/'frozen_decisions.json',dict(policies=policies,model_hashes={p.relative_to(OUT).as_posix():sha(p) for p in (OUT/'models').iterdir()},test_unseen_so_far=True))
    save(OUT/'TEST_STARTED.json',dict(frozen_sha256=sha(OUT/'frozen_decisions.json'),evaluation_population='synthetic ONLY'))
    te=df[df.split.eq('test')];ty=te.target.to_numpy();results={};rawsaved={}
    for n,bundles in test_models.items():
        start=time.monotonic();p=np.mean([prob(m,pre.transform(te[FEATURES]).astype('float32')) for pre,m in bundles],axis=0);elapsed=time.monotonic()-start;pred=predict(p,policies[n]) if policies[n] else p.argmax(1)
        results[n]=dict(raw=metric(ty,p),operating=metric(ty,p,pred),inference_ms_per_row=elapsed/len(te)*1000);rawsaved[n]=p
    np.savez_compressed(OUT/'SYNTHETIC_test_predictions.npz',sample_id=te.sample_id.to_numpy(dtype=str),target=ty,**rawsaved)
    majority=int(train.target.mode().iloc[0]);baseline_p=np.zeros((len(te),3));baseline_p[:,majority]=1
    save(OUT/'SYNTHETIC_test_metrics.json',dict(models=results,rows=len(te),training_majority_class=majority,majority_baseline=metric(ty,baseline_p),mean_six_raw_accuracy=float(np.mean([v['raw']['accuracy'] for v in results.values()])),real_world_accuracy='Not established',production_changed=False))
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    save(OUT/'verification.json',dict(real_artifacts_unchanged=True,synthetic_only=True,production_changed=False,data_rows=len(df),all_six_tested=True))
    print('SYNTHETIC COMPLETE',json.dumps({n:round(v['raw']['accuracy'],4) for n,v in results.items()}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);ap.add_argument('--rows',type=int,default=10000);ap.add_argument('--scenario',choices=['simulation','toy_easy','stress_test'],default='simulation');ARGS=ap.parse_args();ROOT=Path(ARGS.project);OUT=Path(ARGS.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical'];NOISE={'simulation':1.,'toy_easy':.25,'stress_test':1.6}[ARGS.scenario]
    sys.path.insert(0,str(ROOT));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];torch.set_num_threads(4);main()
