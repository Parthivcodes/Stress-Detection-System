"""Prepare frozen performance-qualified overrides, never activate them."""
import argparse,csv,json
from pathlib import Path
from tune_six_models import read,save,sha
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--experiment',required=True);a=p.parse_args();root=Path(a.project);out=Path(a.experiment);decision=read(out/'decision.json');frozen=read(out/'frozen_candidates.json');results=read(out/'test_results.json');base=root/'experiments/lower_noise_simulation_20261026'
for n,h in read(out/'protected_hashes.json').items():assert sha(root/n)==h,n
for n,h in read(out/'frozen_source_hashes.json').items():assert sha(out/n)==h,n
assert sha(root/'src/training_variants.py')==sha(out/'source/training_variants.py')
models={};hashes={'src/training_variants.py':sha(root/'src/training_variants.py')};rows=[];prefix='experiments/'+out.name+'/'
for name in decision['performance_passed']:
    r=results['models'][name];assert r['performance_passed'];f=frozen['models'][name];assert f['eligible'];bundles=[]
    for ep,mp,dim in f['paths']:
        assert sha(out/ep)==frozen['hashes'][ep] and sha(out/mp)==frozen['hashes'][mp]
        bundles.append(dict(encoder=prefix+ep,checkpoint=prefix+mp,input_dim=dim));hashes[prefix+ep]=sha(out/ep);hashes[prefix+mp]=sha(out/mp)
    models[name]=dict(config=f['config'],bundles=bundles,metrics=r['candidate'],performance_passed=True)
    b=r['baseline'];c=r['candidate'];rows.append(dict(model=name,old_accuracy=b['accuracy'],new_accuracy=c['accuracy'],old_high_recall=b['recall'][2],new_high_recall=c['recall'][2],old_sometimes_recall=b['recall'][1],new_sometimes_recall=c['recall'][1],old_high_missed=b['high_missed'],new_high_missed=c['high_missed'],new_macro_f1=c['macro_f1'],new_qwk=c['qwk'],paired_accuracy_difference_ci95=r['paired_accuracy_difference_ci95']))
assert models
save(out/'overrides.json',dict(revision='rf_cnn_tuned_20261026',data_sha256=frozen['data_sha256'],models=models,sha256=hashes,scope='Performance-qualified local generated-data simulation; activation requires runtime verification.'))
with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Fixed-data Random Forest and 1D CNN tuning','',
 'This experiment uses the unchanged10,000-row lower-noise generated dataset. It is an educational simulation; results do not establish accuracy on real people. Dataset bytes, labels and split roles were not altered. No exact simulator score, target or sample ID was supplied as a feature.','',
 '## Final results on the same reused1,500-profile internal holdout','',
 '| Model | Accuracy before / after | High recall before / after | Sometimes recall before / after | High cases missed before / after |','|---|---|---|---|---|']
for r in rows:lines.append(f"| {r['model']} | {100*r['old_accuracy']:.2f}% / {100*r['new_accuracy']:.2f}% | {100*r['old_high_recall']:.2f}% / {100*r['new_high_recall']:.2f}% | {100*r['old_sometimes_recall']:.2f}% / {100*r['new_sometimes_recall']:.2f}% | {r['old_high_missed']} / {r['new_high_missed']} |")
lines+=['',f"Always predicting sometimes: {100*results['majority_accuracy']:.2f}% accuracy,100% sometimes recall and0% high recall on this generated holdout.",'',
 'Each candidate improved accuracy above80%, high recall,sometimes recall,macro-F1 and QWK against its current counterpart. Paired1000-draw stratified bootstrap accuracy-difference intervals are recorded in test_results.json; their lower bounds exceed0. These descriptive intervals do not account for historical reuse of this test set. Validation constraints cannot guarantee unchanged recall for future observations.','',
 '## What changed','',
 '- Random Forest retains its algorithm family. It uses450trees, unlimited maximum depth,min_samples_leaf2,max_features0.85,max_samples0.9 and balanced class weights. Its encoder derives14fixed contextual summaries from the same19inputs, including social/material burden. These are hypothetical aggregates, not validated clinical scales. No CSV column or source response was changed.',
 '- The CNN remains a1D convolutional network. It preserves feature positions through its classification head and adds a direct input skip connection. The prior pooling architecture discarded some position-specific information. The original CNN trained longer did not qualify, and was rejected. The new version stores matching architecture metadata and uses two predetermined seed checkpoints averaged.',
 '- The other four model families and their checkpoints remain untouched. The combined display still equally averages all six raw probability vectors; no independent ensemble-validation claim is made.','',
 '## Selection protocol','',
 'Four RF settings and three CNN settings, including explicit baselines, were compared on two training-only stratified folds. Each outer fit partition had a separate10%development set for neural early stopping. Preprocessing fit only on fitting rows. Rawargmax decisions were used throughout; no threshold search. Both scoring folds and pooled predictions had to pass the accuracy and minority-recall/F1/QWK safeguards.',
 '', 'Only the selected winner per family was refitted on the original6,300fitting and700development rows. Validation required gains and a1percentage-point margin above each existing model\'s high and sometimes recall. Checkpoints, preprocessing and eligibility were frozen before one final pass on the already-reused internal holdout. No alternative was selected after test.','',
 '## Preservation, runtime and rollback','',
 'Original datasets, original model checkpoints and the four other families are protected by protected_hashes.json. New checkpoints are separate in this folder. frozen_source_hashes.json and frozen_candidates.json record the exact source/configurations and checkpoint digests. cv_predictions.npz and cv_roles.csv contain training-fold evidence; validation.json and test_results.json hold complete class metrics and confusion matrices.',
 '', 'overrides.json describes performance-qualified replacements but does not itself activate them. Runtime verification must check saved-prediction parity, all six paths, invalid inputs,512MiB memory, artifact sizes,PDF export and rollback before installation. See runtime_verification.json and activation.json for the final outcome; absence of activation.json means no activation was recorded.','',
 'Once activated, rollback is to remove the overrides field from models/training_preview.json, restart the simulation preview and refresh the page. Base checkpoints remain in their original experiment folder. Backups of the prior configuration/loader/frontend are retained under backups/before_rf_cnn_promotion. The separate real-data server and its active_model.json are not switched by this experiment.','']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8');print('Prepared performance-qualified override manifest and comparison; active selection unchanged.')
