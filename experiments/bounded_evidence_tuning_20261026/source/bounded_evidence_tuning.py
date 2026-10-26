"""Bounded follow-up to diagnosis; grouped train-only comparisons, immutable production."""
import argparse,copy,gc,json,resource,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import log_loss
from tune_six_models import read,save,sha,metric,partition,preprocessor,prob,NAMES,FILES
from tune_two_boundaries import boundaries,predict,violations

SEEDS=[20261026,20261027]
EXTRA_NAMES=['short_sleep','long_sleep','sleep_distance8','physical_days_log','physical_days14','bmi_low','bmi_high','bmi_clipped','age_squared','low_support','frequent_isolation','social_pair','material_risks','social_material_product','physical_sleep_product']

class Encoder:
    def __init__(self,schema,mode):self.schema=schema;self.mode=mode
    def extra(self,f):
        short=(f.sleep_duration<7).astype(float);support=(f.support>=4).astype(float);isolation=(f.isolation<=3).astype(float)
        material=sum((f[k]==1).astype(float) for k in ['job_loss','bill_difficulty','utility_threat','transport_barrier','medical_cost'])+(f.food_insecurity<=3).astype(float)
        return np.column_stack([short,f.sleep_duration>9,abs(f.sleep_duration-8),np.log1p(f.unhealthy_days),f.unhealthy_days>=14,f.bmi<18.5,f.bmi>=30,np.clip(f.bmi,*self.bmi_limits),(f.age/80)**2,support,isolation,support*isolation,material,material*(support+isolation),short*np.log1p(f.unhealthy_days)]).astype('float32')
    def fit(self,f):
        self.base=preprocessor(self.schema)
        if self.mode=='robust':self.base.transformers[0]=('numeric',RobustScaler(),self.schema['numeric'])
        self.base.fit(f);self.bmi_limits=np.quantile(f.bmi,[.01,.99]);self.scale=StandardScaler().fit(self.extra(f));return self
    def transform(self,f):
        x=self.base.transform(f)
        return np.concatenate([x,self.scale.transform(self.extra(f))],axis=1).astype('float32') if self.mode=='derived' else x.astype('float32')

def configs(i):
    if i<3:return ['baseline','derived','focus']
    return ['baseline','focus']

def protect():
    for n,h in read(DIAG/'original_hashes.json').items():assert sha(ROOT/n)==h,n

def initialize():
    if OUT.exists():raise RuntimeError('Fresh directory required.')
    protect();OUT.mkdir(parents=True)
    save(OUT/'protocol.json',dict(seeds=SEEDS,scope='User approved bounded strongest-findings run. Same six families; no deployment; no source rows or labels edited. Existing Phase1 backup retained.',
      diagnosis_sha256=sha(DIAG/'verification.json'),data_sha256=sha(ROOT/'data/expanded_survey_dataset/respondents.csv'),
      cv='Two stratified profile-group outer folds of ALL original training respondents. Outer fit pool split 20% calibration; remaining split 12.5% early-stop development. Independent fit/dev/calibration/score roles. Fit-only encoders. Original validation/test not used in CV.',
      features='19 core inputs preserved. Derived variant appends 15 explicit features (see source/EXTRA_NAMES); no learned label encoding. BMI quantiles and scaling fit only on fit. Original BMI retained. Neither bad-label flags nor outliers remove rows.',derived=EXTRA_NAMES,
      configurations={NAMES[i]:configs(i) for i in range(6)},
      classical='Baseline clone of active config; derived adds features only. Focus LR C=.1,max_iter=2000,tol=1e-5; RF240 trees,depth10,minleaf50,maxfeatures.8,maxsamples.8,classweights sqrt inverse prevalence; XGB600 rounds,lr.025,depth3,minchild30,lambda15,alpha1,gamma.2,earlystop25 on dev. Others unchanged.',
      neural='Two same predetermined seeds per config, probabilities averaged for both baseline and candidate; per-seed raw metrics saved. Original architecture/fixed feature ordering. Baseline standard scaling/AdamW .0007,decay.001,original dropout. Focus robust scaling,AdamW .001,decay.003,dropout.35,ReduceLROnPlateau factor.5 patience2. Both weighted CE sqrt inverse prevalence,batch512,gradientclip1,patience4,max12epochs,60second soft cap checked each epoch. Dev unweighted logloss picks checkpoint. No temporal-history claim.',
      decision='Fresh baseline frozen individual multipliers define calibration reference. Optimize two-boundary baseline policy on calibration under its own recall/F1/QWK constraints; use feasible policy else original multipliers. Candidate policy optimized on same calibration under BOTH original and matched-baseline componentwise strict safeguards. Never optimize thresholds on outer score. Outer score requires accuracy strictly higher and both recalls/F1/QWK nonlower versus both baseline rules, high recall>=.60. Require this on EACH fold and pooled OOF. No slack or rounded eligibility.',
      selection='Among eligible per-family configurations highest pooled accuracy then F1. No alternative if later validation fails. Final refit uses grouped90%train/10%dev; validation selects thresholds versus actual original and recall-matched original. Freeze before single reused-holdout evaluation. Require paired accuracyCI lower>0 and recalls/F1/QWK nonlower for promotion, then512MB/API/rollback verification. No candidate eligible => no holdout access.',
      budget='18 classical +24 neural fits; sequential4threads. Twofold bounded search, NOT5x2CV or exhaustiveOptuna. No unrequested paidcompute. Two seeds measure limited stability, not reliable uncertainty.',
      baseline='Always rarely reported per comparison; historical reused holdout65.78%,0%minorityrecalls. Existing holdout NOT untouched.',
      calibration='Decision threshold selection only; no probability-calibration claim.',
      ranking='Feature group permutation logloss on first outer score with baselineXGB,3repeats,diagnostic only/no adaptive feature selection. Training-only MI optional; no SHAP package required.'))
    for n in ['bounded_evidence_tuning.py','tune_six_models.py','tune_two_boundaries.py','tune_xgb_features.py','feature_pipeline.py']:
        d=OUT/'source'/n;d.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/n,d)
    save(OUT/'frozen_hashes.json',{p.relative_to(OUT).as_posix():sha(p) for p in [OUT/'protocol.json']+list((OUT/'source').glob('*.py'))})

def fit(i,c,ft,dev,seed):
    start=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
    enc=Encoder(SCHEMA,'derived' if c=='derived' else 'robust' if c=='focus' and i>=3 else 'original').fit(ft)
    x=enc.transform(ft);dx=enc.transform(dev);y=ft.target.to_numpy(copy=True);dy=dev.target.to_numpy(copy=True);w=(len(y)/(3*np.bincount(y,minlength=3)))**.5
    history=[];capped=False;bestepoch=None
    if i<3:
        m=clone(joblib.load(MODELS/FILES[i]));params={'random_state':seed}
        if 'n_jobs' in m.get_params():params['n_jobs']=4
        if i==0:params['class_weight']=dict(enumerate(w))
        if c=='focus':
            if i==0:params.update(C=.1,max_iter=2000,tol=1e-5)
            elif i==1:params.update(n_estimators=240,max_depth=10,min_samples_leaf=50,max_features=.8,max_samples=.8,bootstrap=True,class_weight=dict(enumerate(w)))
            else:params.update(n_estimators=600,learning_rate=.025,max_depth=3,min_child_weight=30,reg_lambda=15,reg_alpha=1,gamma=.2,early_stopping_rounds=25)
        m.set_params(**params)
        m.fit(x,y,**({'sample_weight':w[y],**({'eval_set':[(dx,dy)],'verbose':False} if c=='focus' else {})} if i==2 else {}))
    else:
        m=CLASSES[i-3](input_dim=x.shape[1]);focus=c=='focus'
        if focus:
            for layer in m.modules():
                if isinstance(layer,torch.nn.Dropout):layer.p=.35
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001 if focus else .0007,weight_decay=.003 if focus else .001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2) if focus else None
        lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32));loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=512,shuffle=True)
        best=float('inf');stale=0;state=None
        for epoch in range(1,13):
            m.train();total=0
            for bx,by in loader:
                optimizer.zero_grad();loss=lossfn(m(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);optimizer.step();total+=float(loss.detach())*len(by)
            ll=float(log_loss(dy,prob(m,dx),labels=[0,1,2]));history.append(dict(epoch=epoch,train_weighted_loss=total/len(y),dev_logloss=ll))
            if ll<best-1e-5:best=ll;state=copy.deepcopy(m.state_dict());bestepoch=epoch;stale=0
            else:stale+=1
            if scheduler:scheduler.step(ll)
            capped=time.monotonic()-start>=60
            if stale>=4 or capped:break
        m.load_state_dict(state)
    return enc,m,dict(seconds=time.monotonic()-start,seed=seed,history=history,best_epoch=bestepoch,time_cap_reached=capped,encoded_features=x.shape[1],process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def strictref(*refs):
    r=copy.deepcopy(refs[0])
    for k in ['accuracy','macro_f1','qwk']:r[k]=max(b[k] for b in refs)
    r['recall']=[max(b['recall'][j] for b in refs) for j in range(3)];return r

def better(m,b):return m['accuracy']>b['accuracy']+1e-12 and max(violations(m,b))<=1e-12

def run():
    if not OUT.exists():initialize()
    protect()
    for n,h in read(OUT/'frozen_hashes.json').items():assert sha(OUT/n)==h
    assert sha(__file__)==sha(OUT/'source/bounded_evidence_tuning.py'),'Use frozen script after starting.'
    df=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv');tr=df[df.split.eq('train')].reset_index(drop=True);del df
    y=tr.target.to_numpy();roles=[];records=[];oof={}
    for fold,(a,b) in enumerate(StratifiedGroupKFold(2,shuffle=True,random_state=SEEDS[0]).split(tr,y,tr.profile_group)):
        pool,cal=partition(tr.iloc[a],5,SEEDS[0]+fold);ft,dev=partition(pool,8,SEEDS[0]+fold);sc=tr.iloc[b]
        sets=[set(f.profile_group) for f in [ft,dev,cal,sc]]
        assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
        for role,f in [('fit',ft),('dev',dev),('calibration',cal),('score',sc)]:roles.extend(dict(respondent_id=i,profile_group=g,fold=fold,role=role) for i,g in zip(f.respondent_id,f.profile_group))
        for i,n in enumerate(NAMES):
            for c in configs(i):
                key=f'{i}_{c}';cp=[];sp=[];infos=[];seedmetrics=[]
                for seed in SEEDS if i>=3 else SEEDS[:1]:
                    stem=OUT/f'fit_{fold}_{key}_{seed}';npz=stem.with_suffix('.npz');js=stem.with_suffix('.json')
                    if npz.exists() and js.exists():
                        p=np.load(npz);assert np.array_equal(p['score_ids'],sc.respondent_id);cp.append(p['cal']);sp.append(p['score']);infos.append(read(js))
                    else:
                        print('FIT',fold,n,c,seed,flush=True);enc,m,info=fit(i,c,ft,dev,seed);pc=prob(m,enc.transform(cal));ps=prob(m,enc.transform(sc))
                        np.savez_compressed(npz,cal=pc,score=ps,score_ids=sc.respondent_id.to_numpy(dtype=str),cal_ids=cal.respondent_id.to_numpy(dtype=str));save(js,info);cp.append(pc);sp.append(ps);infos.append(info);del m,enc;gc.collect()
                    seedmetrics.append(metric(sc.target.to_numpy(),sp[-1]))
                pc=np.mean(cp,axis=0);ps=np.mean(sp,axis=0);cy=cal.target.to_numpy();sy=sc.target.to_numpy()
                if c=='baseline':
                    original_cal=metric(cy,pc,(pc*SEL[n]['multipliers']).argmax(1));original_score=metric(sy,ps,(ps*SEL[n]['multipliers']).argmax(1));policy=boundaries(cy,pc,original_cal)
                    bp=predict(ps,policy['thresholds']) if policy['feasible'] else (ps*SEL[n]['multipliers']).argmax(1)
                    matched_cal=policy['metrics'] if policy['feasible'] else original_cal;matched_score=metric(sy,ps,bp);calref=strictref(original_cal,matched_cal);scoreref=strictref(original_score,matched_score)
                    predicted=bp;passed=False
                else:
                    policy=boundaries(cy,pc,calref);predicted=predict(ps,policy['thresholds']);passed=bool(policy['feasible'] and better(metric(sy,ps,predicted),scoreref))
                mm=metric(sy,ps,predicted)
                r=dict(fold=fold,model=n,config=c,raw=metric(sy,ps),decision=mm,policy=policy,reference=scoreref,original_baseline=original_score,matched_baseline=matched_score,eligible=passed,per_seed_raw=seedmetrics,fit=infos,majority_accuracy=float(np.mean(sy==0)))
                save(OUT/f'result_{fold}_{key}.json',r);records.append(r)
                if key not in oof:oof[key]=dict(p=np.zeros((len(tr),3)),pred=np.zeros(len(tr),dtype=int),original=np.zeros(len(tr),dtype=int))
                oof[key]['p'][b]=ps;oof[key]['pred'][b]=predicted
                if c=='baseline':oof[key]['original'][b]=(ps*SEL[n]['multipliers']).argmax(1)
                print('RESULT',fold,n,c,'accuracy',round(mm['accuracy'],5),'high',round(mm['recall'][2],5),'sometimes',round(mm['recall'][1],5),'eligible',passed,flush=True)
    pd.DataFrame(roles).to_csv(OUT/'roles.csv',index=False)
    np.savez_compressed(OUT/'oof.npz',respondent_id=tr.respondent_id.to_numpy(dtype=str),target=y,**{k+'_'+sub:v for k,obj in oof.items() for sub,v in obj.items()})
    summary={};selected={}
    for i,n in enumerate(NAMES):
        base=oof[f'{i}_baseline'];bm=metric(y,base['p'],base['pred']);om=metric(y,base['p'],base['original']);ref=strictref(bm,om);summary[n]={}
        for c in configs(i):
            r=oof[f'{i}_{c}'];mm=metric(y,r['p'],r['pred']);ok=c!='baseline' and better(mm,ref) and all(v['eligible'] for v in records if v['model']==n and v['config']==c)
            summary[n][c]=dict(metrics=mm,raw=metric(y,r['p']),eligible=ok,original_baseline=om,matched_baseline=bm)
        elig=[c for c in configs(i) if summary[n][c]['eligible']]
        if elig:selected[n]=max(elig,key=lambda c:(summary[n][c]['metrics']['accuracy'],summary[n][c]['metrics']['macro_f1']))
    save(OUT/'summary.json',dict(models=summary,selected=selected,majority_accuracy=float(np.mean(y==0)),rows=len(y)))
    save(OUT/'decision.json',dict(status='training_cv_complete_candidates_pending_validation' if selected else 'rejected_in_training_cv',selected=selected,production_changed=False,test_evaluated=False,validation_evaluated=False))
    protect();print('COMPLETE',json.dumps(read(OUT/'decision.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--diagnosis',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);DIAG=Path(a.diagnosis);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');SEL=read(MODELS/'selection.json')
    sys.path.insert(0,str(ROOT));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];torch.set_num_threads(4);run()
