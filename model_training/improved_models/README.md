# Fixed-data Random Forest and 1D CNN tuning

This experiment uses the unchanged10,000-row lower-noise generated dataset. It is an educational simulation; results do not establish accuracy on real people. Dataset bytes, labels and split roles were not altered. No exact simulator score, target or sample ID was supplied as a feature.

## Final results on the same reused1,500-profile internal holdout

| Model | Accuracy before / after | High recall before / after | Sometimes recall before / after | High cases missed before / after |
|---|---|---|---|---|
| Random Forest | 76.73% / 89.07% | 79.18% / 91.26% | 67.46% / 84.92% | 81 / 34 |
| 1D CNN | 78.07% / 90.53% | 78.66% / 91.77% | 75.59% / 86.95% | 83 / 32 |

Always predicting sometimes: 39.33% accuracy,100% sometimes recall and0% high recall on this generated holdout.

Each candidate improved accuracy above80%, high recall,sometimes recall,macro-F1 and QWK against its current counterpart. Paired1000-draw stratified bootstrap accuracy-difference intervals are recorded in test_results.json; their lower bounds exceed0. These descriptive intervals do not account for historical reuse of this test set. Validation constraints cannot guarantee unchanged recall for future observations.

## What changed

- Random Forest retains its algorithm family. It uses450trees, unlimited maximum depth,min_samples_leaf2,max_features0.85,max_samples0.9 and balanced class weights. Its encoder derives14fixed contextual summaries from the same19inputs, including social/material burden. These are hypothetical aggregates, not validated clinical scales. No CSV column or source response was changed.
- The CNN remains a1D convolutional network. It preserves feature positions through its classification head and adds a direct input skip connection. The prior pooling architecture discarded some position-specific information. The original CNN trained longer did not qualify, and was rejected. The new version stores matching architecture metadata and uses two predetermined seed checkpoints averaged.
- The other four model families and their checkpoints remain untouched. The combined display still equally averages all six raw probability vectors; no independent ensemble-validation claim is made.

## Selection protocol

Four RF settings and three CNN settings, including explicit baselines, were compared on two training-only stratified folds. Each outer fit partition had a separate10%development set for neural early stopping. Preprocessing fit only on fitting rows. Rawargmax decisions were used throughout; no threshold search. Both scoring folds and pooled predictions had to pass the accuracy and minority-recall/F1/QWK safeguards.

Only the selected winner per family was refitted on the original6,300fitting and700development rows. Validation required gains and a1percentage-point margin above each existing model's high and sometimes recall. Checkpoints, preprocessing and eligibility were frozen before one final pass on the already-reused internal holdout. No alternative was selected after test.

## Preservation, runtime and rollback

Original datasets, original model checkpoints and the four other families are protected by protected_hashes.json. New checkpoints are separate in this folder. frozen_source_hashes.json and frozen_candidates.json record the exact source/configurations and checkpoint digests. cv_predictions.npz and cv_roles.csv contain training-fold evidence; validation.json and test_results.json hold complete class metrics and confusion matrices.

overrides.json describes performance-qualified replacements but does not itself activate them. Runtime verification must check saved-prediction parity, all six paths, invalid inputs,512MiB memory, artifact sizes,PDF export and rollback before installation. See runtime_verification.json and activation.json for the final outcome; absence of activation.json means no activation was recorded.

Once activated, rollback is to remove the overrides field from models/training_preview.json, restart the simulation preview and refresh the page. Base checkpoints remain in their original experiment folder. Backups of the prior configuration/loader/frontend are retained under backups/before_rf_cnn_promotion. The separate real-data server and its active_model.json are not switched by this experiment.

## Activation completed

Both qualifying models are active on the local simulation preview at http://localhost:8001/#benchmarks. All six live paths, updated benchmarks and a two-page PDF download were verified. Fresh-process peak RSS was 429.14 MiB; no paid service or cloud deployment was used. The original lower-noise models remain available for rollback. Remove the overrides field from models/training_preview.json, restart port8001 and refresh to restore them. The original real-data server on port8000 remains unchanged.
