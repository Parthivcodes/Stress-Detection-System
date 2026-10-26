"""Train-only tuning for two families; fixed dataset, strict recall safeguards."""
import argparse,copy,gc,json,resource,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import log_loss
from tune_six_models import read,save,sha,metric,prob

SEEDS=[20261026,20261027];NAMES=['Random Forest','1D CNN']
CONFIGS={
 'Random Forest':[
  dict(id='baseline',encoder='original',params={}),
  dict(id='risk_leaf2',encoder='engineered',params=dict(n_estimators=350,max_depth=18,min_samples_leaf=2,max_features=1.,max_samples=.9,bootstrap=True,class_weight=None)),
  dict(id='risk_deep',encoder='engineered',params=dict(n_estimators=450,max_depth=None,min_samples_leaf=2,max_features=.85,max_samples=.9,bootstrap=True,class_weight='balanced')),
  dict(id='risk_leaf4',encoder='engineered',params=dict(n_estimators=400,max_depth=20,min_samples_leaf=4,max_features=1.,max_samples=.9,bootstrap=True,class_weight='balanced'))],
 '1D CNN':[
  dict(id='baseline',encoder='original',architecture='original_cnn',dropout=.3,lr=.0007,decay=.001,epochs=40,patience=6,scheduler=False),
  dict(id='original_longer',encoder='original',architecture='original_cnn',dropout=.15,lr=.001,decay=.0003,epochs=90,patience=10,scheduler=True),
  dict(id='position_skip',encoder='original',architecture='position_cnn',dropout=.15,lr=.001,decay=.0003,epochs=80,patience=10,scheduler=True,channels=16,hidden=64)]}

def protect():
    for n,h in read(OUT/'protected_hashes.json').items():assert sha(ROOT/n)==h,n
def passes(m,b,margin=0):
    return bool(m['accuracy']>max(.8,b['accuracy'])+1e-12 and m['recall'][2]+1e-12>=max(.6,b['recall'][2]+margin) and m['recall'][1]+1e-12>=b['recall'][1]+margin and m['macro_f1']+1e-12>=b['macro_f1'] and m['qwk']+1e-12>=b['qwk'])

def initialize():
    if OUT.exists():raise RuntimeError('Use a fresh experiment directory.')
    OUT.mkdir(parents=True);files=[p for p in BASE.rglob('*') if p.is_file()]+[p for p in (ROOT/'models/current_models').rglob('*') if p.is_file()]+[ROOT/n for n in ['models/active_model.json','models/training_preview.json','data/expanded_survey_dataset/respondents.csv','src/training_preview_inference.py','src/models.py','src/training_variants.py']]
    save(OUT/'protected_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    save(OUT/'protocol.json',dict(seeds=SEEDS,configs=CONFIGS,dataset_sha256=sha(BASE/'model_training_data.csv'),
      scope='Fixed existing lower-noise10000-row CSV; no generation,noisechanges,rowselection,targetchanges or model-family replacement. OnlyRF/CNN tune; other4families unchanged. No paidcompute/deployment.',
      validation='Current validation/test historically used; test is a reused internalholdout,not untouched. Choices use trainingCV then existingvalidation only; one finaltestpass of frozen qualifyingwinners; no alternate candidate after finalfailure.',
      cv='Two stratified outerfolds across7000training profiles. Within eachouterfit,10%stratifieddevelopment choosesCNNepochs. Same fit/dev/score IDs for allconfigs. Full19feature profiles unique. Fit encoders only on fit. Score rawargmax only; no thresholdsearch.',
      features='RFengineered mode uses19codedrawinputs plus14predeclared contextualaggregates from thoseinputs; target,IDs,randomseed and exactgeneratorscore are excluded. Motivated by featuremeaning and knowledge this is a simulation; unvalidatedindices,not clinicalknowledge. Datasetbytes unchanged.',
      neural='Baseline currentConv1DNet; longer originalarchitecture variant; or versioned1DCNN preserving featurepositions with linearinputskip.2fixedseeds averaged for EVERYbaseline/candidate. WeightedCE sqrtinverseprevalence,AdamW,batch256,clip1. Devunweightedlogloss pickscheckpoint. Baseline120secsoftcap,candidates180sec; checked betweenepochs. Cross-sectionaltabular positions,not timehistory.',
      selection='Candidate must have rawaccuracy>80% and greaterthan freshbaseline, highrecall>=max(.60,baselinehigh),sometimesrecall>=baseline,F1/QWK>=baseline in EACHCVfold and pooledOOF. Select eligible highestpooledaccuracy,thenmacroF1. Only1winnerperfamily. No winner => keepcurrent.',
      finalfit='Same6300fit/700development sampleIDs androworder as currentmodels. Same twoCNNseeds. Fit-only encoders. Validate winner against actual current rawvalidation metrics; demand accuracy>80% and higherthan current, high/sometimesrecalls>=current+1percentagepoint margin,macroF1/QWK nonlower. Margin not a futureguarantee.',
      finaltest='Freeze allcheckpoints,encoders,architecturemetadata andeligibility beforetest. Require rawaccuracy>80% and current, high/sometimesrecalls/F1/QWK nonlower than current, pairedstratifiedbootstrap95%accuracydelta lower>0.1000draws. No retuningaftertest. Existingtest alreadyseen so estimatesnotindependentconfirmation.',
      promotion='Performancepass alone does not activate. Need checkpointparity,all6prediction paths,PDF/benchmarks,rollback andmemory checks. Artifactfamilylimit64MiB; keepoldifanygatefails. Neveroverwriteoriginalcheckpoints.',
      budget='8RF+12CNN CVfits,up to1RF+2CNN finalfits,4CPUthreads sequential. Boundedfixedsearch,notOptuna/exhaustiveCV. No installs.',
      baseline='CurrentRF76.7333% andCNN78.0667% onreportedlower-noisetest. Eachfoldmajoritybaseline plus fullgeneratedtestmajority39.3333%; majorityclass sometimes. Existing empiricalBRFSS baseline65.78% is separate and not compared as thisbenchmark.'))
    for n in ['tune_lower_noise_rf_cnn.py','tune_six_models.py']:
        d=OUT/'source'/n;d.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/n,d)
    shutil.copyfile(ROOT/'src/training_variants.py',OUT/'source/training_variants.py')
    save(OUT/'frozen_source_hashes.json',{p.relative_to(OUT).as_posix():sha(p) for p in [OUT/'protocol.json']+list((OUT/'source').glob('*.py'))})
    save(OUT/'baseline_rf_parameters.json',joblib.load(ROOT/'models/current_models/random_forest.joblib').get_params())
    print('PROTOCOL FROZEN',flush=True)

def fit(name,c,ft,dev,seed):
    start=time.monotonic();torch.manual_seed(seed);np.random.seed(seed);enc=FeatureEncoder(SCHEMA,c['encoder']).fit(ft[FEATURES]);x=enc.transform(ft[FEATURES]);dx=enc.transform(dev[FEATURES]);y=ft.target.to_numpy(copy=True);dy=dev.target.to_numpy(copy=True);history=[];bestepoch=None;capped=False
    if name=='Random Forest':
        m=clone(joblib.load(ROOT/'models/current_models/random_forest.joblib'));m.set_params(**{**c['params'],'random_state':seed,'n_jobs':4});m.fit(x,y)
    else:
        if c['architecture']=='position_cnn':m=PositionConv1D(input_dim=x.shape[1],channels=c['channels'],hidden=c['hidden'],dropout=c['dropout'])
        else:
            m=Conv1DNet(input_dim=x.shape[1])
            for layer in m.modules():
                if isinstance(layer,torch.nn.Dropout):layer.p=c['dropout']
        optimizer=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['decay']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3) if c['scheduler'] else None
        w=(len(y)/(3*np.bincount(y,minlength=3)))**.5;lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32));loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=256,shuffle=True);best=float('inf');stale=0;state=None
        for epoch in range(1,c['epochs']+1):
            m.train();total=0
            for bx,by in loader:
                optimizer.zero_grad();loss=lossfn(m(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);optimizer.step();total+=float(loss.detach())*len(by)
            ll=float(log_loss(dy,prob(m,dx),labels=[0,1,2]));history.append(dict(epoch=epoch,train_weighted_loss=total/len(y),dev_logloss=ll))
            if ll<best-1e-5:best=ll;state=copy.deepcopy(m.state_dict());bestepoch=epoch;stale=0
            else:stale+=1
            if scheduler:scheduler.step(ll)
            if epoch%10==0:print('EPOCH',c['id'],seed,epoch,round(ll,4),flush=True)
            capped=time.monotonic()-start>=(120 if c['id']=='baseline' else 180)
            if stale>=c['patience'] or capped:break
        m.load_state_dict(state)
    return enc,m,dict(seconds=time.monotonic()-start,history=history,best_epoch=bestepoch,time_cap_reached=capped,seed=seed,encoded_dim=x.shape[1],peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def predict_saved(name,c,paths,frame):
    pp=[]
    for encpath,modelpath,dim in paths:
        enc=joblib.load(OUT/encpath);x=enc.transform(frame[FEATURES])
        if name=='Random Forest':m=joblib.load(OUT/modelpath)
        elif c['architecture']=='position_cnn':m=PositionConv1D(dim,channels=c['channels'],hidden=c['hidden'],dropout=c['dropout']);m.load_state_dict(torch.load(OUT/modelpath,map_location='cpu',weights_only=True))
        else:m=Conv1DNet(dim);m.load_state_dict(torch.load(OUT/modelpath,map_location='cpu',weights_only=True))
        pp.append(prob(m,x));del enc,m;gc.collect()
    return np.mean(pp,axis=0)

def main():
    initialize();protect();df=pd.read_csv(BASE/'model_training_data.csv');assert df.sample_id.is_unique and not df.duplicated(FEATURES).any();tr=df[df.split.eq('train')].reset_index(drop=True);va=df[df.split.eq('validation')];y=tr.target.to_numpy();oof={};records=[];roles=[]
    for fold,(a,b) in enumerate(StratifiedKFold(2,shuffle=True,random_state=SEEDS[0]).split(tr,y)):
        f,d=train_test_split(a,test_size=.1,random_state=SEEDS[0]+fold,stratify=y[a]);ft=tr.iloc[f];dev=tr.iloc[d];sc=tr.iloc[b]
        assert not set(ft.sample_id)&set(dev.sample_id) and not set(tr.iloc[a].sample_id)&set(sc.sample_id)
        for role,frame in [('fit',ft),('dev',dev),('score',sc)]:roles.extend(dict(sample_id=i,fold=fold,role=role) for i in frame.sample_id)
        for name in NAMES:
            for c in CONFIGS[name]:
                pp=[];infos=[];seedmetrics=[]
                for seed in SEEDS if name=='1D CNN' else SEEDS[:1]:
                    print('CV FIT',fold,name,c['id'],seed,flush=True);enc,m,info=fit(name,c,ft,dev,seed);p=prob(m,enc.transform(sc[FEATURES]));pp.append(p);infos.append(info);seedmetrics.append(metric(sc.target.to_numpy(),p));del m,enc;gc.collect()
                p=np.mean(pp,axis=0);mm=metric(sc.target.to_numpy(),p)
                if c['id']=='baseline':baseline=mm
                r=dict(fold=fold,model=name,config=c['id'],metrics=mm,baseline=baseline,eligible=c['id']!='baseline' and passes(mm,baseline),fit=infos,per_seed=seedmetrics,majority_accuracy=float(np.mean(sc.target==tr.iloc[a].target.mode().iloc[0])))
                save(OUT/f'cv_{fold}_{name.replace(" ","_")}_{c["id"]}.json',r);records.append(r);key=name+'__'+c['id']
                if key not in oof:oof[key]=np.zeros((len(tr),3))
                oof[key][b]=p;print('CV RESULT',fold,name,c['id'],round(mm['accuracy'],4),'high',round(mm['recall'][2],4),'sometimes',round(mm['recall'][1],4),'eligible',r['eligible'],flush=True)
    pd.DataFrame(roles).to_csv(OUT/'cv_roles.csv',index=False);np.savez_compressed(OUT/'cv_predictions.npz',sample_id=tr.sample_id.to_numpy(dtype=str),target=y,**oof);summary={};selected={}
    for name in NAMES:
        baseline=metric(y,oof[name+'__baseline']);summary[name]={}
        for c in CONFIGS[name]:
            mm=metric(y,oof[name+'__'+c['id']]);ok=c['id']!='baseline' and passes(mm,baseline) and all(r['eligible'] for r in records if r['model']==name and r['config']==c['id']);summary[name][c['id']]=dict(metrics=mm,baseline=baseline,eligible=ok)
        eligible=[c for c in CONFIGS[name] if summary[name][c['id']]['eligible']]
        if eligible:selected[name]=max(eligible,key=lambda c:(summary[name][c['id']]['metrics']['accuracy'],summary[name][c['id']]['metrics']['macro_f1']))
    save(OUT/'cv_summary.json',summary);save(OUT/'cv_selected.json',selected)
    if not selected:save(OUT/'decision.json',dict(status='rejected_in_cv',performance_passed=[],production_changed=False,test_evaluated=False));protect();return
    roles=pd.read_csv(BASE/'training_roles.csv');idx=df.set_index('sample_id',drop=False);ft=idx.loc[roles.loc[roles.role.eq('fit'),'sample_id']].reset_index(drop=True);dev=idx.loc[roles.loc[roles.role.eq('early_stop'),'sample_id']].reset_index(drop=True);oldval=read(BASE/'validation_metrics.json');valid={};frozen={};newval={}
    for name,c in selected.items():
        pp=[];paths=[];infos=[]
        for seed in SEEDS if name=='1D CNN' else SEEDS[:1]:
            print('FINAL FIT',name,c['id'],seed,flush=True);enc,m,info=fit(name,c,ft,dev,seed);p=prob(m,enc.transform(va[FEATURES]));pp.append(p);infos.append(info);stem=OUT/'models'/f'{name.replace(" ","_")}_{seed}';stem.parent.mkdir(exist_ok=True);ep=stem.with_suffix('.encoder.joblib');mp=stem.with_suffix('.joblib' if name=='Random Forest' else '.pt');joblib.dump(enc,ep)
            if name=='Random Forest':joblib.dump(m,mp)
            else:torch.save(m.state_dict(),mp)
            paths.append([ep.relative_to(OUT).as_posix(),mp.relative_to(OUT).as_posix(),info['encoded_dim']]);del enc,m;gc.collect()
        p=np.mean(pp,axis=0);mm=metric(va.target.to_numpy(),p);ref=oldval[name]['raw'];ok=passes(mm,ref,margin=.01);size=sum((OUT/p).stat().st_size for bundle in paths for p in bundle[:2]);ok=bool(ok and size<64*1024**2)
        valid[name]=dict(candidate=mm,baseline=ref,eligible=ok,fit=infos,artifact_bytes=size,seed_validation_accuracy=[]);frozen[name]=dict(config=c,paths=paths,eligible=ok);newval[name]=p
        print('VALIDATION',name,round(mm['accuracy'],4),'high',round(mm['recall'][2],4),'sometimes',round(mm['recall'][1],4),'eligible',ok,flush=True)
    save(OUT/'validation.json',valid);np.savez_compressed(OUT/'validation_predictions.npz',sample_id=va.sample_id.to_numpy(dtype=str),**newval);hashes={p.relative_to(OUT).as_posix():sha(p) for p in (OUT/'models').iterdir()};save(OUT/'frozen_candidates.json',dict(models=frozen,hashes=hashes,data_sha256=sha(BASE/'model_training_data.csv')))
    qualifying=[n for n,v in frozen.items() if v['eligible']]
    if not qualifying:save(OUT/'decision.json',dict(status='rejected_on_validation',performance_passed=[],production_changed=False,test_evaluated=False));protect();return
    save(OUT/'TEST_STARTED.json',dict(frozen_sha256=sha(OUT/'frozen_candidates.json'),note='Single frozen pass on reused lower-noise internalholdout.'))
    te=df[df.split.eq('test')];ty=te.target.to_numpy();old=np.load(BASE/'test_predictions.npz');assert np.array_equal(old['sample_id'],te.sample_id) and np.array_equal(old['target'],ty);results={};predictions={};passed=[]
    for name in qualifying:
        for n,h in hashes.items():assert sha(OUT/n)==h
        f=frozen[name];p=predict_saved(name,f['config'],f['paths'],te);mm=metric(ty,p);bm=metric(ty,old[name]);pred=p.argmax(1);bp=old[name].argmax(1);rng=np.random.default_rng(SEEDS[0]);groups=[np.flatnonzero(ty==j) for j in range(3)];deltas=[]
        for _ in range(1000):
            ids=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);deltas.append(float(np.mean(pred[ids]==ty[ids])-np.mean(bp[ids]==ty[ids])))
        ci=np.quantile(deltas,[.025,.975]).tolist();ok=bool(passes(mm,bm) and ci[0]>0);results[name]=dict(candidate=mm,baseline=bm,paired_accuracy_difference_ci95=ci,performance_passed=ok);predictions[name]=p
        if ok:passed.append(name)
        print('FINAL TEST',name,round(mm['accuracy'],4),'high',round(mm['recall'][2],4),'sometimes',round(mm['recall'][1],4),'pass',ok,flush=True)
    save(OUT/'test_results.json',dict(models=results,majority_accuracy=float(np.mean(ty==int(tr.target.mode().iloc[0])))),);np.savez_compressed(OUT/'test_predictions.npz',sample_id=te.sample_id.to_numpy(dtype=str),target=ty,**predictions)
    save(OUT/'decision.json',dict(status='performance_pass_pending_runtime' if passed else 'rejected_on_reused_holdout',performance_passed=passed,production_changed=False,test_evaluated=True));protect();print('DONE',json.dumps(read(OUT/'decision.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);BASE=ROOT/'model_training/base_models';SCHEMA=read(ROOT/'models/current_models/schema.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical'];sys.path.insert(0,str(ROOT));from src.models import Conv1DNet
    from src.training_variants import FeatureEncoder,PositionConv1D
    torch.set_num_threads(4);main()
