"""Train six isolated 10k-subset models; select thresholds separately from scoring."""
import argparse,copy,gc,json,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.base import clone
from sklearn.metrics import log_loss
from tune_six_models import NAMES,FILES,sha,save,read,preprocessor,partition,metric,prob
from tune_two_boundaries import boundaries,predict,violations

SEED=202610251
def protect():
    for n,h in read(OUT/'original_hashes.json').items():assert sha(ROOT/n)==h,n
def baseline(raw,y):
    r={n:metric(y,raw[n],(raw[n]*SEL[n]['multipliers']).argmax(1)) for n in NAMES}
    p=sum(w*raw[n].astype(float) for n,w in POLICY['weights'].items());p/=p.sum(1,keepdims=True)
    r['Ensemble']=metric(y,p,np.where(p[:,2]>=POLICY['high_threshold'],2,p[:,:2].argmax(1)))
    return r
def passes(m,b):return bool(m['accuracy']>b['accuracy']+1e-12 and max(violations(m,b))<=1e-12)

def neural(i,pre,frame,epochs,dev=None):
    torch.manual_seed(SEED);np.random.seed(SEED)
    x=pre.transform(frame[FEATURES]).astype('float32');y=frame.target.to_numpy()
    weights=(len(y)/(3*np.bincount(y,minlength=3)))**.5
    m=CLASSES[i-3](input_dim=x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.0007,weight_decay=.001)
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=512,shuffle=True)
    vx=pre.transform(dev[FEATURES]).astype('float32') if dev is not None else None
    best=float('inf');bestepoch=epochs;stale=0;history=[]
    for epoch in range(1,epochs+1):
        m.train();total=0.
        for bx,by in loader:
            opt.zero_grad();loss=lossfn(m(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=float(loss.detach())*len(by)
        r=dict(epoch=epoch,train_weighted_loss=total/len(y))
        if dev is not None:
            ll=float(log_loss(dev.target,prob(m,vx),labels=[0,1,2]));r['dev_logloss']=ll
            if ll<best-1e-5:best=ll;bestepoch=epoch;stale=0
            else:stale+=1
        history.append(r)
        print('EPOCH',NAMES[i],'select' if dev is not None else 'refit',epoch,flush=True)
        if dev is not None and stale>=4:break
    return m,bestepoch,history

def run():
    if OUT.exists():raise RuntimeError('Refusing to overwrite experiment.')
    OUT.mkdir(parents=True)
    paths=[p for p in MODELS.rglob('*') if p.is_file()]+[ROOT/n for n in ['models/active_model.json','data/expanded_survey_dataset/respondents.csv','data/experimental_brfss_10000/brfss_10000_respondents.csv','src/models.py','src/brfss_inference.py','src/ensemble_policy.py']]
    save(OUT/'original_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    for p in paths:
        if p.suffix=='.csv':continue
        dest=OUT/'backup'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    configs={}
    for i in range(3):
        m=joblib.load(MODELS/FILES[i]);params=m.get_params();params.pop('missing',None);params['random_state']=SEED
        if 'n_jobs' in params:params['n_jobs']=4
        if i==0:params['class_weight']='recomputed_inverse_frequency_power_0.5'
        configs[NAMES[i]]=params;del m
    save(OUT/'protocol.json',dict(seed=SEED,features=FEATURES,configs=configs,neural=dict(max_epochs=14,patience=4,lr=.0007,weight_decay=.001,batch_size=512,weight_power=.5,architecture='Current src.models classes,69 encoded inputs'),
      fitting='Use exactly subset train7364. Classical models retain active configurations; LR/XGB fit-only weights recomputed, RF retains existing weighting. Neural epoch selection on grouped10% inner training dev, then fresh seeded refit on all7364 using selected epoch count. Both preprocessors fit only their fitting rows.',
      calibration='Subset validation872 selects two-boundary thresholds, preserving current per-model high/sometimes recalls,F1,QWK and highfloor60%. No resampling of calibration or evaluation.',
      comparison='Full existing validation excluding872 calibration IDs (14325). Same rows for old/new. Cached originals verified by hashes. This is historically reused validation, not external evidence. No threshold selection on scoring rows.',
      ensemble='Current fixed LR0.4/XGB0.5/LSTM0.1 weights applied to new raw probabilities. One ensemble decision rule chosen on calibration; compare against active ensemble.',
      gates='Calibration must be feasible. Scoring accuracy strictly higher, highrecall>=max(.60,baseline),sometimes recall,F1,QWK no lower than same current model. No choosing a new threshold after score. A qualifying model still requires frozen one-pass reused-holdout and512MB runtime/API/rollback checks before activation.',
      holdout='Do not evaluate subset1764 test rows or full test unless a candidate passes scoring. If eligible freeze all choices and compare once on full original30718 reused internal holdout, not merely selected subset. No test-based retuning.',
      safety='Originals read-only, backups verified, no deployment. One seed per neural model means limited stability evidence; no automatic activation on a small gain.',baseline='Alwaysrarely shown on each scored split; historical holdout65.78% with zero minorityrecalls.'))
    shutil.copyfile(__file__,OUT/'train_subset_six.py');save(OUT/'frozen_hashes.json',{n:sha(OUT/n) for n in ['protocol.json','train_subset_six.py']})
    sub=pd.read_csv(ROOT/'data/experimental_brfss_10000/brfss_10000_respondents.csv');full=pd.read_csv(ROOT/'data/expanded_survey_dataset/respondents.csv')
    assert len(sub)==10000 and sub.respondent_id.is_unique
    assert sub.groupby('profile_group').split.nunique().max()==1
    source=full.set_index('respondent_id');assert sub.set_index('respondent_id').equals(source.loc[sub.respondent_id])
    tr=sub[sub.split.eq('train')].copy();cal=sub[sub.split.eq('validation')].copy()
    va=full[full.split.eq('validation')].reset_index(drop=True);is_cal=va.respondent_id.isin(cal.respondent_id).to_numpy();score=va[~is_cal].copy();cal=va[is_cal].copy()
    assert (len(tr),len(cal),len(score))==(7364,872,14325)
    assert not set(tr.profile_group)&set(va.profile_group)
    fit,dev=partition(tr,10,SEED);assert not set(fit.profile_group)&set(dev.profile_group)
    pd.concat([f[['respondent_id','profile_group']].assign(role=r) for r,f in [('neural_epoch_fit',fit),('neural_epoch_dev',dev),('threshold_calibration',cal),('comparison',score)]]).to_csv(OUT/'roles.csv',index=False)
    olddir=ROOT/'experiments/hyperparameter_tuning_20261023'
    for n,h in read(olddir/'original_hashes.json').items():
        if n.startswith('models/current_models/') or n=='data/expanded_survey_dataset/respondents.csv':assert sha(ROOT/n)==h,n
    old=dict(np.load(olddir/'original_validation.npz'));assert all(v.shape==(len(va),3) for v in old.values())
    oldcal={n:p[is_cal] for n,p in old.items()};oldscore={n:p[~is_cal] for n,p in old.items()}
    brcal=baseline(oldcal,cal.target.to_numpy());brscore=baseline(oldscore,score.target.to_numpy())
    pre=preprocessor(SCHEMA).fit(tr[FEATURES]);joblib.dump(pre,OUT/'preprocessor.joblib')
    devpre=preprocessor(SCHEMA).fit(fit[FEATURES]);x=pre.transform(tr[FEATURES]).astype('float32');cx=pre.transform(cal[FEATURES]).astype('float32');sx=pre.transform(score[FEATURES]).astype('float32');fy=tr.target.to_numpy()
    rawcal={};rawscore={};results={}
    for i,n in enumerate(NAMES):
        start=time.monotonic();print('TRAIN',n,flush=True)
        if i<3:
            m=clone(joblib.load(MODELS/FILES[i]));updates=dict(random_state=SEED)
            if 'n_jobs' in m.get_params():updates['n_jobs']=4
            weights=(len(fy)/(3*np.bincount(fy,minlength=3)))**.5
            if i==0:updates['class_weight']=dict(enumerate(weights))
            m.set_params(**updates);m.fit(x,fy,**({'sample_weight':weights[fy]} if i==2 else {}));history=[];epochs=None
            joblib.dump(m,OUT/FILES[i])
        else:
            trial,epochs,history=neural(i,devpre,fit,14,dev);del trial;gc.collect()
            m,_,refithistory=neural(i,pre,tr,epochs);torch.save(m.state_dict(),OUT/FILES[i]);history=dict(selection=history,refit=refithistory)
        rawcal[n]=prob(m,cx).astype(float);rawscore[n]=prob(m,sx).astype(float)
        t=boundaries(cal.target.to_numpy(),rawcal[n],brcal[n]);adjusted=metric(score.target.to_numpy(),rawscore[n],predict(rawscore[n],t['thresholds']))
        results[n]=dict(original=brscore[n],candidate=adjusted,original_raw=metric(score.target.to_numpy(),oldscore[n]),candidate_raw=metric(score.target.to_numpy(),rawscore[n]),policy=t,eligible=bool(t['feasible'] and passes(adjusted,brscore[n])),epochs=epochs,history=history,seconds=time.monotonic()-start,artifact_bytes=(OUT/FILES[i]).stat().st_size)
        save(OUT/f'model_{i}.json',results[n]);print('RESULT',n,'old',brscore[n]['accuracy'],'new',adjusted['accuracy'],'eligible',results[n]['eligible'],flush=True);del m;gc.collect()
    for d in [rawcal,rawscore]:
        d['Ensemble']=sum(w*d[n] for n,w in POLICY['weights'].items());d['Ensemble']/=d['Ensemble'].sum(1,keepdims=True)
    t=boundaries(cal.target.to_numpy(),rawcal['Ensemble'],brcal['Ensemble']);m=metric(score.target.to_numpy(),rawscore['Ensemble'],predict(rawscore['Ensemble'],t['thresholds']))
    results['Ensemble']=dict(original=brscore['Ensemble'],candidate=m,policy=t,eligible=bool(t['feasible'] and passes(m,brscore['Ensemble'])))
    np.savez_compressed(OUT/'comparison_predictions.npz',respondent_id=score.respondent_id.to_numpy(dtype=str),target=score.target.to_numpy(),**rawscore)
    np.savez_compressed(OUT/'calibration_predictions.npz',respondent_id=cal.respondent_id.to_numpy(dtype=str),target=cal.target.to_numpy(),**rawcal)
    eligible=[n for n,r in results.items() if r['eligible']]
    save(OUT/'summary.json',dict(results=results,train_rows=len(tr),calibration_rows=len(cal),comparison_rows=len(score),majority_accuracy=float(score.target.eq(0).mean()),original_six_mean_accuracy=float(np.mean([results[n]['original']['accuracy'] for n in NAMES])),candidate_six_mean_accuracy=float(np.mean([results[n]['candidate']['accuracy'] for n in NAMES]))))
    save(OUT/'decision.json',dict(status='frozen_candidates_need_final_checks' if eligible else 'all_rejected',eligible=eligible,production_changed=False,test_evaluated=False))
    protect();print('DONE',json.dumps(read(OUT/'decision.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models'
    sys.path.insert(0,str(ROOT));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];SCHEMA=read(MODELS/'schema.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical'];SEL=read(MODELS/'selection.json');POLICY=read(MODELS/'ensemble/policy.json');torch.set_num_threads(4);run()
