"""Separate paired-profile noise experiment; existing models/data are read-only."""
import argparse,gc,hashlib,json,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
import synthetic_stress_experiment as base
from tune_six_models import read,save,sha,metric,prob,NAMES

NOISE=.25
SEEDS=[20261026,20261027]

def protect():
    for name,digest in read(OUT/'protected_hashes.json').items():assert sha(ROOT/name)==digest,name

def main():
    if OUT.exists():raise RuntimeError('Use fresh experiment directory; refuse overwrite.')
    OUT.mkdir(parents=True)
    previous=ROOT/'experiments/synthetic_simulation_20261026';manifest=read(previous/'data_manifest.json')
    assert sha(previous/manifest['csv'])==manifest['sha256']
    files=[p for p in previous.rglob('*') if p.is_file()]+[p for p in MODELS.rglob('*') if p.is_file()]+[ROOT/'models/active_model.json',ROOT/'data/expanded_survey_dataset/respondents.csv']
    save(OUT/'protected_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    save(OUT/'protocol.json',dict(data_origin='synthetic_simulation',purpose='Controlled easier generated-data benchmark, not improved real-person prediction.',
      seed=SEEDS[0],neural_seeds=SEEDS,rows=manifest['rows'],previous_noise_sd=1.,noise_sd=NOISE,
      source_dataset_sha256=manifest['sha256'],features=FEATURES,
      comparison='Same10000featureprofiles/sampleIDs in sameorder; same train/validation/test and fit/dev roles as prior simulation. Only the independentlabelnoise scale changes. New target labels yield a different,easier simulation. Do not interpret accuracy delta as an improvement on unchanged labels or BRFSS.',
      generation='Reuse prior exact generator with fixedseed and fixedscoreboundaries; .25 is predeclared toy_easy noise. Restore original profile split assignments after generation; no re-stratification by new labels. Neither rows nor coefficients filtered for accuracy. Features checked identical before fitting.',
      fitting='Reuse prior model classes and settings exactly:3classical activeconfiguration clones; neural existingarchitecture,AdamW .0007 decay.001,weightedCE sqrtinversefrequency,batch256,max40epochs,patience6,120secsoftcap.2fixedneural seeds averaged. Fit-only standardnumeric/onehot preprocessing. Devlogloss selects checkpoint; seednotcherry-picked.',
      decisions='Rawprobability argmax for every family, matching website rawdecisionrule. No thresholdsearch or ensembleweightselection.',
      freeze='Freeze ALLcheckpoints/preprocessors and choices before one final1500profile generatedtestevaluation. Do not regenerate/change noise or tuneafterseeingtest.',
      safety='Writesonlyfresh experimentfolder. Hash every previous simulation artifact plusactive realmodels/data before/after. No modelactivation,websitechange or deployment.',
      limits='More separable inventedlabels can produce higher scores without adding evidence of clinical or realpopulation utility. Minorityrecalls reported onchangedlabels; these are not same-groundtruth recallpreservation claims. Sameprofiles enable descriptive pairedaccuracy differences acrossdifferent simulationoutcomes,not fixedlabel modelcomparison.'))
    for name in ['train_lower_noise_simulation.py','synthetic_stress_experiment.py','tune_six_models.py','tune_two_boundaries.py','tune_xgb_features.py','feature_pipeline.py']:
        dest=OUT/'source'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/name,dest)
    save(OUT/'frozen_source_hashes.json',{p.relative_to(OUT).as_posix():sha(p) for p in [OUT/'protocol.json']+list((OUT/'source').glob('*.py'))})
    old=pd.read_csv(previous/manifest['csv']);df=base.generate(len(old),NOISE)
    assert np.array_equal(df.sample_id,old.sample_id)
    np.testing.assert_allclose(df[FEATURES].to_numpy(),old[FEATURES].to_numpy(),rtol=1e-12,atol=1e-12)
    df['split']=old.split.to_numpy();assert not df.duplicated(FEATURES).any() and df.sample_id.is_unique
    path=OUT/'model_training_data.csv';df.to_csv(path,index=False)
    roles=pd.read_csv(previous/'training_roles.csv');assert roles.sample_id.is_unique
    indexed=df.set_index('sample_id',drop=False);ft=indexed.loc[roles.loc[roles.role.eq('fit'),'sample_id']].reset_index(drop=True);dev=indexed.loc[roles.loc[roles.role.eq('early_stop'),'sample_id']].reset_index(drop=True)
    va=df[df.split.eq('validation')];assert set(ft.sample_id)|set(dev.sample_id)==set(df[df.split.eq('train')].sample_id)
    groups=[set(f.sample_id) for f in [ft,dev,va,df[df.split.eq('test')]]];assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
    roles.to_csv(OUT/'training_roles.csv',index=False)
    save(OUT/'data_manifest.json',dict(csv=path.name,sha256=sha(path),data_origin='synthetic_simulation',rows=len(df),noise_sd=NOISE,identical_previous_features=True,identical_previous_splits=True,
      fit_rows=len(ft),dev_rows=len(dev),split_counts=df.split.value_counts().to_dict(),
      changed_labels_by_split={s:int((df.loc[df.split.eq(s),'target'].to_numpy()!=old.loc[old.split.eq(s),'target'].to_numpy()).sum()) for s in ['train','validation','test']},
      class_counts={s:df[df.split.eq(s)].target.value_counts().sort_index().to_dict() for s in ['train','validation','test']}))
    del old
    # Existing fit helper gets only the original architecture/schema/config inputs.
    base.MODELS=MODELS;base.SCHEMA=SCHEMA;base.FEATURES=FEATURES;base.CLASSES=CLASSES
    reports={};paths=[]
    for i,name in enumerate(NAMES):
        pp=[];seed_reports=[]
        for seed in SEEDS if i>=3 else SEEDS[:1]:
            print('LOWER-NOISE FIT',name,seed,flush=True);pre,m,info=base.fit(i,ft,dev,seed);p=prob(m,pre.transform(va[FEATURES]).astype('float32'));pp.append(p)
            folder=OUT/'models';folder.mkdir(exist_ok=True);stem=folder/f'{i}_{seed}';ep=stem.with_suffix('.preprocessor.joblib');joblib.dump(pre,ep);mp=stem.with_suffix('.joblib' if i<3 else '.pt')
            if i<3:joblib.dump(m,mp)
            else:torch.save(m.state_dict(),mp)
            paths.extend([ep,mp]);seed_reports.append(dict(fit=info,raw_validation=metric(va.target.to_numpy(),p)));del m,pre;gc.collect()
        mean=np.mean(pp,axis=0);reports[name]=dict(raw=metric(va.target.to_numpy(),mean),per_seed=seed_reports)
        save(OUT/'validation_metrics.json',reports);print('LOWER-NOISE VALIDATION',name,reports[name]['raw']['accuracy'],flush=True)
    save(OUT/'frozen_models.json',dict(decision='rawargmax',neural_seeds=SEEDS,model_hashes={p.relative_to(OUT).as_posix():sha(p) for p in paths},data_sha256=sha(path)))
    protect();save(OUT/'TEST_STARTED.json',dict(frozen_sha256=sha(OUT/'frozen_models.json'),population='separatelower-noisesimulation',new_test_pass=True))
    te=df[df.split.eq('test')];ty=te.target.to_numpy();results={};predictions={};reloads={}
    for i,name in enumerate(NAMES):
        pp=[];start=time.monotonic()
        for seed in SEEDS if i>=3 else SEEDS[:1]:
            stem=OUT/'models'/f'{i}_{seed}';pre=joblib.load(stem.with_suffix('.preprocessor.joblib'));x=pre.transform(te[FEATURES]).astype('float32')
            if i<3:m=joblib.load(stem.with_suffix('.joblib'))
            else:m=CLASSES[i-3](input_dim=x.shape[1]);m.load_state_dict(torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True));m.eval()
            p=prob(m,x);pp.append(p);del m,pre;gc.collect()
        mean=np.mean(pp,axis=0);predictions[name]=mean;results[name]=dict(raw=metric(ty,mean),load_and_inference_seconds=time.monotonic()-start)
        assert np.isfinite(mean).all() and np.allclose(mean.sum(1),1,atol=1e-6);reloads[name]=True
        print('LOWER-NOISE TEST',name,results[name]['raw']['accuracy'],flush=True)
    majority=int(df.loc[df.split.eq('train'),'target'].mode().iloc[0]);bp=np.zeros((len(te),3));bp[:,majority]=1
    np.savez_compressed(OUT/'test_predictions.npz',sample_id=te.sample_id.to_numpy(dtype=str),target=ty,**predictions)
    save(OUT/'test_metrics.json',dict(models=results,rows=len(te),noise_sd=NOISE,majority_class=majority,majority_baseline=metric(ty,bp),mean_six_accuracy=float(np.mean([v['raw']['accuracy'] for v in results.values()])),real_world_accuracy='Not established',production_changed=False))
    protect();save(OUT/'verification.json',dict(passed=True,all_prior_artifacts_unchanged=True,feature_profiles_identical=True,split_roles_identical=True,all_six_loaded=reloads,production_changed=False,one_final_generated_test_pass=True))
    print('LOWER-NOISE COMPLETE',json.dumps({n:r['raw']['accuracy'] for n,r in results.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.project);OUT=Path(a.output);MODELS=ROOT/'models/current_models';SCHEMA=read(MODELS/'schema.json');FEATURES=SCHEMA['numeric']+SCHEMA['categorical'];sys.path.insert(0,str(ROOT));from src.models import Conv1DNet,LSTMNet,CNNLSTMNet
    CLASSES=[Conv1DNet,LSTMNet,CNNLSTMNet];torch.set_num_threads(4);main()
