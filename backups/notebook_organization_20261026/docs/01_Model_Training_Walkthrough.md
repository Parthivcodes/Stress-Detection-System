# Model Training and Accuracy Improvement Walkthrough

NERVA stress-frequency project | Study and viva preparation

This guide explains the actual saved implementation and results. It covers dataset preparation, preprocessing, model training, safe tuning, evaluation, website integration and the important files. The source checklist at the end identifies the evidence behind the explanation.

## 1. What improved, and what this result means

Random Forest improved from 76.73% to 89.07%, and 1D CNN improved from 78.07% to 90.53%, on the same 1,500-profile lower-noise simulation holdout. The CSV, target labels, profile IDs and split assignments did not change during this RF/CNN experiment. Both high-class and sometimes recall increased.

| Model | Old accuracy | New accuracy | High-class recall | Sometimes recall |
|---|---:|---:|---|---|
| Random Forest | 76.73% | 89.07% | 79.18% -> 91.26% | 67.46% -> 84.92% |
| 1D CNN | 78.07% | 90.53% | 78.66% -> 91.77% | 75.59% -> 86.95% |

The majority-class baseline is always predicting sometimes: 39.33% accuracy, 100% sometimes recall and 0% high-class recall. Accuracy therefore must be considered with class recall, not alone.

Two different changes happened during the project and must be explained separately:

- Earlier, a separate generated dataset reduced label-noise standard deviation from 1.0 to 0.25. That made the simulated classification problem easier; its six-model average rose from 63.52% to 84.62%. The labels changed in that earlier comparison.
- Later, the RF/CNN tuning described here used the already-created lower-noise CSV without changing its contents. Gains came from derived features, forest settings, a CNN architecture change and a disciplined training/selection process.

These are generated-data simulation results. They do not establish accuracy for real people, and the dataset is not Kaggle-sourced. The separate real BRFSS implementation is retained. The generated holdout had already been reported before this tuning run, so it is a reused internal holdout rather than independent external confirmation.

## 2. Dataset and the classification target

The current demo uses 10,000 generated profiles in model_training_data.csv inside experiments/lower_noise_simulation_20261026/. Generation used a fixed seed, hypothetical shared factors, a fixed hidden-score equation and random label uncertainty. Its assumptions are documented; the coefficients are not validated clinical relationships.

The target has three ordered classes:

| Target code | Meaning in the demo |
|---|---|
| 0 | Rarely / never |
| 1 | Sometimes |
| 2 | Usually / always |

These labels describe simulated frequency, not a medical diagnosis or measured biological stress severity.

The 19 actual model inputs are:

| Group | Input columns | Representation |
|---|---|---|
| Numeric | age, sleep_duration, unhealthy_days, bmi | Age, hours, physically unhealthy days and BMI |
| Personal context | sex, exercise, general_health, marital, education, employment, income | Integer category codes defined in schema.json |
| Social context | support, isolation | Frequency categories |
| Financial/access context | job_loss, food_insecurity, bill_difficulty, utility_threat, transport_barrier, medical_cost | Yes/no or frequency categories |

The CSV also contains sample_id, split, data_origin and target. These are metadata/outcomes and are excluded from the feature matrix. The hidden score, exact simulator equation, sample ID and random-seed reconstruction were not supplied to the tuned models as predictive features.

Data integrity checks verified unique sample IDs, no duplicate complete 19-feature profiles, and separate split memberships. The generated table contains no missing input values. Therefore this experiment did not impute data, remove difficult rows, oversample validation/test, or correct labels based on predictions.

## 3. Split the data before fitting anything

| Partition | Profiles | Purpose |
|---|---:|---|
| Training partition | 7,000 | Training-only cross-validation and final fitting/development roles |
| Final fitting rows within training | 6,300 | Fit encoders, class weights and model parameters |
| Final development rows within training | 700 | Select neural stopping epochs |
| Validation | 1,500 | Final acceptance check for the selected candidate |
| Reused internal holdout | 1,500 | One final frozen comparison in this tuning run |

The 6,300/700 roles use exactly the same sample IDs and order as the prior lower-noise models. No model was trained on all 10,000 profiles. Validation and holdout outcomes are never training inputs.

During tuning, two stratified outer folds were created from the 7,000 training profiles. Each fold had about 3,500 outer scoring rows. The other approximately 3,500 rows were divided into 3,150 fitting and 350 development rows. Every configuration in a fold used the same roles. Stratification maintains class representation; no identical feature profiles crossed roles because the complete profiles were unique.

Development labels choose a neural stopping epoch; outer scoring labels assess a configuration. This separation prevents reporting a development score as if it were an unbiased scoring-fold result.

## 4. Preprocessing and fit-only learning

For the original-style and neural input pipeline, the four numeric inputs are standardized using fitting-row mean and standard deviation. The 15 categorical inputs use fixed-category one-hot encoding. This produces 69 numeric values: four standardized numbers plus 65 category indicators.

The scaler and encoder are fitted only on fitting rows. The same fitted objects transform development, validation, holdout and website inputs. Re-fitting a scaler separately on a website submission would change the meaning of the inputs and is not done.

The tuned forest uses a separate encoder: 19 coded inputs plus 14 deterministic derived values, producing 33 values. Its tree splits do not require standard scaling. Some category codes are passed as integers in this experimental forest; this imposes a representation choice on nominal categories, not a validated medical ordering. The unchanged categorical meanings remain documented in the schema and UI.

Both encoders consume the same 19 user inputs. Derived features are calculated inside preprocessing; no extra source columns or labels are written into the CSV.

## 5. Why the original Random Forest was weaker

The baseline forest used 250 trees, maximum depth 20, minimum 12 samples per leaf, max_features 0.8 and balanced class weights. Its large leaves and many independent input decisions limited how precisely it could recover the simulated additive relationships. This is an explanation supported by the tested bundled change; the study did not isolate every hyperparameter's causal contribution.

The RF search included the current-style baseline and three predeclared alternatives. All alternatives retained the Random Forest algorithm and used derived contextual features. Their settings varied leaf size, tree count, depth, feature fraction, bootstrap sample fraction and class weighting.

The selected configuration, risk_deep, was chosen using training-fold results:

| Setting | Selected value | Meaning |
|---|---|---|
| n_estimators | 450 | Number of decision trees |
| max_depth | None | No explicit depth cap; other controls still limit growth |
| min_samples_leaf | 2 | Smallest permitted leaf size |
| min_samples_split | 2 | Minimum rows required to split a node |
| max_features | 0.85 | Fraction of transformed features considered at a split |
| bootstrap | True | Fit trees from bootstrap samples |
| max_samples | 0.9 | Bootstrap sample fraction |
| class_weight | balanced | Weight classes inversely to their fitting frequency |
| criterion | gini | Split impurity criterion |
| random_state | 20261026 | Reproducible random seed |

### The 14 derived forest features

The implementation is FeatureEncoder in src/training_variants.py. It uses the following fixed calculations; these are hypothetical contextual summaries rather than validated clinical scales.

| Number | Feature | Calculation |
|---|---|---|
| 1 | Support-risk code | (support - 1) / 4 |
| 2 | Isolation-risk code | (5 - isolation) / 4 |
| 3 | Food-insecurity risk code | (5 - food_insecurity) / 4 |
| 4 | General-health risk code | (general_health - 1) / 4 |
| 5 | Material/access count | Count yes responses across job_loss, bill_difficulty, utility_threat, transport_barrier and medical_cost |
| 6 | Sleep shortfall | max(0, 8 - sleep_duration) |
| 7 | Long-sleep amount | max(0, sleep_duration - 9) |
| 8 | Unhealthy-days transform | log(1 + unhealthy_days) |
| 9 | Frequent physical unhealthy days | Indicator for unhealthy_days >= 14 |
| 10 | Social-context sum | Support-risk + isolation-risk + food-risk |
| 11 | Contextual burden | Health-risk + support-risk + isolation-risk + food-risk + material_count/3 + sleep_shortfall/2 |
| 12 | Coded burden | Sleep_shortfall + general_health + support + (6-isolation) + (6-food_insecurity) + 2*material_count + unhealthy_days/10 |
| 13 | Material/social interaction | material_count * (support-risk + isolation-risk) |
| 14 | Sleep/health interaction | sleep_shortfall * health-risk |

These transformations are motivated by feature meanings and knowledge that this is a simulation. They do not copy the exact simulator score or use target outcomes. The comparison measures the whole engineered-feature/settings package; it does not prove that every derived feature helped individually.

## 6. Why the 1D CNN changed from about 78% to 90%

The original CNN applied convolutions over the 69 input positions and then adaptive average pooling down to four positions. The network had 32 and 64 convolution channels, a 64-unit hidden layer, and dropout 0.3. Pooling compressed position-specific information, which can be useful for images but is restrictive when individual tabular columns have distinct meanings.

One tested alternative simply trained the original architecture longer with revised regularization and a scheduler. It failed the safeguards and was rejected. The successful variant changed how feature-position information reaches the classifier while keeping the 1D CNN family.

The selected PositionConv1D architecture is:

```text
69 preprocessed input values
  -> reshape to one channel with length 69
  -> Conv1D: 1 to 16 channels, kernel 3, padding 1
  -> Batch normalization -> ReLU
  -> Conv1D: 16 to 16 channels, kernel 3, padding 1
  -> Batch normalization -> ReLU
  -> flatten all positions: 16 * 69 = 1,104 values
  -> dense layer: 1,104 to 64 -> ReLU -> dropout 0.15
  -> dense layer: 64 to 3 logits

In parallel: original 69 values -> linear layer -> 3 logits
Final logits = convolutional-branch logits + skip-branch logits
Softmax turns the 3 final logits into probabilities
```

Flattening keeps feature positions available. The direct skip branch provides a short path for simple global input relationships. The convolutional branch can learn additional patterns. This is a CNN with a skip connection, not a replacement with a different algorithm family.

The 69 positions are an ordered representation of tabular inputs, not observations through time. Neither this CNN nor the LSTM should be described as learning a person's longitudinal history from this dataset.

### Training settings and actual stopping epochs

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0003 |
| Batch size | 256 |
| Training loss | Class-weighted cross-entropy |
| Loss weights | sqrt(n / (3 * class_count)), from fitting rows only |
| Maximum epochs | 80 |
| Early-stopping patience | 10 development epochs without sufficient improvement |
| Scheduler | ReduceLROnPlateau; factor 0.5, patience 3 |
| Gradient clipping | Norm 1 |
| Time cap | 180 seconds per candidate fit, checked between epochs |
| Seeds | 20261026 and 20261027 |

The maximum of 80 epochs was a budget, not the number actually used. For final fitting:

| Seed | Best checkpoint epoch | Epochs actually run | Time cap reached |
|---|---:|---:|---|
| 20261026 | 9 | 19 | False |
| 20261027 | 9 | 19 | False |

The best checkpoint is selected by unweighted development log loss, not holdout accuracy. Both predetermined seeds are retained and their probabilities averaged. No seed was chosen because its test result looked better.

## 7. Candidate selection and recall safeguards

The bounded search performed eight forest CV fits and twelve CNN CV fits, then one forest and two CNN final fits for the selected winners. This was not a large Optuna search or exhaustive 5x2 cross-validation.

For each scoring fold, a candidate had to exceed 80% accuracy and its freshly fitted baseline accuracy. High-class recall had to be at least the larger of 60% or the baseline's high recall. Sometimes recall, macro-F1 and quadratic weighted kappa also had to be no lower than baseline. The same conditions applied to pooled training-fold predictions, and every fold had to pass.

Eligible configurations were ranked by pooled accuracy and then macro-F1. Only one winner per family proceeded. The RF winner was risk_deep; the CNN winner was position_skip. The current production-trained checkpoints were not used to predict their own training-fold scoring rows as a baseline: baselines were freshly refitted inside each fold.

Final fitting used the original 6,300/700 fitting/development roles. The validation gate compared each winner with the actual previously active model and required:

- Accuracy above 80% and above that model's validation accuracy.
- High recall at least max(60%, original validation high recall + 1 percentage point).
- Sometimes recall at least original validation sometimes recall + 1 percentage point.
- Macro-F1 and QWK no lower than the original model.
- The candidate family's artifacts smaller than 64 MiB.

RF validation accuracy was 85.53%; CNN validation accuracy was 87.40%. Both passed. Checkpoint, encoder, architecture metadata and eligibility were frozen before the final holdout pass. There was no decision-threshold search; all individual labels use raw-probability argmax.

## 8. Final evaluation and what the metrics mean

Accuracy is the fraction of all classifications that are correct. Recall for a class is the fraction of actual members of that class detected. High recall therefore answers: of the profiles labelled usually/always, how many did the model identify? Sometimes recall asks the corresponding question for the middle class.

Precision measures how often a predicted class is correct. Macro-F1 gives each class equal importance when averaging F1. QWK accounts for the order of the classes. Macro OVR AUC measures ranking discrimination across one-versus-rest tasks. A high individual confidence percentage in the UI is not the model's test accuracy, and these probabilities are not clinically calibrated.

The reused holdout has 521 rarely/never, 590 sometimes and 389 usually/always profiles. On those same labels:

- RF high cases detected increased from 308 to 355; high cases missed fell from 81 to 34. Sometimes detections rose from 398 to 501.
- CNN high cases detected increased from 306 to 357; high cases missed fell from 83 to 32. Sometimes detections rose from 446 to 513.

### Confusion matrices: rows are actual, columns are predicted

| RF new | Rarely predicted | Sometimes predicted | Usually predicted |
|---|---:|---:|---:|
| Actual rarely | 480 | 41 | 0 |
| Actual sometimes | 52 | 501 | 37 |
| Actual usually | 0 | 34 | 355 |

| CNN new | Rarely predicted | Sometimes predicted | Usually predicted |
|---|---:|---:|---:|
| Actual rarely | 488 | 33 | 0 |
| Actual sometimes | 42 | 513 | 35 |
| Actual usually | 0 | 32 | 357 |

The paired 1,000-draw stratified bootstrap accuracy-difference interval was approximately +10.07 to +14.47 percentage points for RF and +10.40 to +14.67 points for CNN. Both lower bounds exceeded zero. These are descriptive intervals and do not correct for historical holdout reuse. Preserved recall on this holdout cannot guarantee preserved recall on future profiles.

## 9. Current six-model results and website workflow

| Model | Accuracy | Macro-F1 | High-class recall |
|---|---:|---:|---:|
| Logistic Regression | 90.00% | 0.9014 | 90.75% |
| Random Forest | 89.07% | 0.8934 | 91.26% |
| XGBoost | 86.93% | 0.8715 | 85.35% |
| 1D CNN | 90.53% | 0.9071 | 91.77% |
| LSTM | 91.40% | 0.9159 | 93.06% |
| CNN-LSTM | 84.60% | 0.8486 | 88.95% |

The mean of the six individual accuracies is 88.76%. This is not a separately measured accuracy for the combined website prediction. Logistic Regression, XGBoost, LSTM and CNN-LSTM were not retrained or replaced during this RF/CNN improvement.

```text
User submits the 19 profile values
  -> browser sends POST /api/predict
  -> server checks required keys, ranges, types and integer categories
  -> each model's own fitted encoder transforms the profile
  -> selected checkpoints calculate raw probabilities
  -> two fixed seeds are averaged within each neural family
  -> each individual label is the largest class probability
  -> the display averages six family probability vectors equally
  -> the browser displays six results and enables PDF download
```

The simulation's combined display is an equal raw-probability mean with argmax and lower-class tie break. It is not majority voting and does not use the original real-data ensemble's weights. The separate real-data server retains its own previously configured inference policy.

The browser-generated PDF uses the latest completed API result and submitted profile. It contains a readable summary, class probability bars, the six model results, all 19 answers, date and source/interpretation notes. No third-party PDF service receives the profile.

## 10. Loading the correct files safely

models/training_preview.json selects the lower_noise simulation and points to experiments/lower_noise_rf_cnn_tuning_20261026/overrides.json. The override manifest supplies only the two qualified replacements. The four other families load from the original lower-noise experiment.

src/training_preview_inference.py verifies the CSV hash, checkpoint/source hashes, supported model names, version and architecture metadata. It uses each checkpoint's matching encoder and rejects invalid inputs. The browser also checks the experiment and model revision so an old page does not silently combine mismatched model versions.

The real-data path is different: models/active_model.json and models/current_models/ are used by src/brfss_inference.py. Cleanup does not switch this selection.

Fresh-process six-family parity checks found maximum probability differences below 0.000001 versus saved results. Runtime peak RSS was 429.14 MiB against a 512 MiB process check. Average prediction time in that check was 96.2 ms. This is a local measurement, not a guarantee for cloud concurrency or future hardware.

All six website paths, updated benchmark rows, invalid-input rejection, a two-page browser PDF download and explicit loading of the original rollback bundle were checked. Both PDF pages were rendered and inspected. No cloud deployment was performed by this tuning task.

## 11. Where to find the important files

| Friendly entry point or file | What it denotes |
|---|---|
| Start-Demo-Website.ps1 | Start the current generated-data demo on port 8001 |
| Start-RealData-Website.ps1 | Start the separate real-data reference on port 8000 |
| training/create_and_train_demo.py | Friendly entry point for a fresh six-family simulation experiment |
| training/improve_forest_and_cnn.py | Friendly entry point for the fixed-data RF/CNN tuning workflow |
| training/create_demo_training_report.py | Build a readable report from a completed lower-noise experiment |
| tools/train_lower_noise_simulation.py | Verified implementation behind the six-model training entry point |
| tools/tune_lower_noise_rf_cnn.py | Detailed CV, training, gating and evaluation implementation |
| tools/synthetic_stress_experiment.py | Shared generated-data recipe and baseline fitting helper |
| src/training_variants.py | RF derived-feature encoder and PositionConv1D architecture |
| src/training_preview_inference.py | Load the selected demo models and predict |
| frontend/prediction.js | Build inputs, request predictions, display results and trigger PDF download |
| frontend/analysis-pdf.js | Create the readable PDF in the browser |
| models/ACTIVE_MODELS.md | Human-readable map from algorithm names to exact checkpoint paths |
| experiments/lower_noise_rf_cnn_tuning_20261026/ | Frozen protocol, trials, candidates, metrics and activation evidence |
| backups/before_rf_cnn_promotion/ | Recoverable previous models, config, data and frontend |

The friendly training files are entry points delegating to the verified implementations. They do not contain a second, different training algorithm. Historical experiment/model filenames retain their identifiers because manifests and rollback depend on them. A numeric checkpoint prefix is a family index, not evidence of duplicated models: 0=LR, 1=RF, 2=XGBoost, 3=CNN, 4=LSTM, 5=CNN-LSTM. A dated suffix denotes the recorded training seed.

## 12. Reproduce without overwriting a working version

Use the existing WSL Ubuntu environment at /opt/stress-venv/bin/python. The project needs Python, NumPy, pandas, scikit-learn, XGBoost, PyTorch and joblib. No paid compute or additional package was needed for this tuning experiment.

```powershell
# Run from the project directory. Choose a NEW output directory.
wsl -d Ubuntu -u nobody --cd $PWD.Path --exec `
  /opt/stress-venv/bin/python -B training/improve_forest_and_cnn.py `
  --project . --output experiments/rf_cnn_fresh_run
```

The training implementation refuses an existing output directory. It runs selection against the documented base lower-noise models, not automatically against whichever later overrides happen to be active. Do not repeatedly inspect and retune on the reused holdout to chase a target score. A successful new training run still needs the separate validation, performance, runtime and activation gates.

The older real-data training scripts in training/ remain available for their original purpose. They are not the source of the reported simulation gains.

## 13. Rollback and directory cleanup

The original checkpoints and datasets were retained and checksum-verified. To restore the original lower-noise RF/CNN, remove only the overrides field from models/training_preview.json, restart the demo server and refresh the browser. Setting experiment to original instead selects the older higher-noise simulation. The real-data server's separate active_model.json is unaffected.

Cleanup archives historical research/setup scripts and the old generated public/ build instead of deleting them. Only first-party compiled .pyc caches are disposable. Saved datasets, checkpoints, experiment results, source assets, Python environments and existing backups are preserved. Cleanup mappings and checksums are in reports/directory_cleanup/cleanup_manifest.json.

## 14. Short viva answers

### What exactly increased the CNN accuracy?
We kept the dataset fixed, preserved feature positions in the CNN head, added a direct input skip connection, adjusted training settings, averaged two predetermined seeds and accepted the result only after validation and recall safeguards passed.

### Did you change the labels to achieve 90.53%?
No labels changed during this RF/CNN tuning run. Earlier in the project, a separate lower-noise simulation had been created; that was a different experiment and is documented separately.

### Did the Random Forest become a different algorithm?
No. It is still Random Forest, with engineered input summaries and revised forest hyperparameters.

### Why do you need validation and a development split?
Development chooses neural stopping epochs inside training. Validation is a later acceptance check. The final holdout is used after model choices are frozen. Each has a different role.

### Why not just maximize accuracy?
An imbalanced model can get many majority cases correct while missing important minority cases. We required improved accuracy while protecting high-class and sometimes recall, macro-F1 and QWK.

### Did you use all 10,000 profiles for training?
No. Final fitting used 6,300 profiles, development used 700, validation used 1,500 and holdout used 1,500.

### Was this an untouched test set?
No. It is a reused internal simulation holdout. This run froze choices before one final pass, but historical reuse limits the strength of independent confirmation.

### Is 90.53% real-world stress-prediction accuracy?
No. It is accuracy on generated simulation labels. Validation on genuinely independent real people would be required for that claim.

### Is an LSTM learning a user's time history here?
No. Inputs are cross-sectional tabular feature positions. The project contains no longitudinal sequence for each person.

### What is the difference between accuracy and the confidence shown on a card?
Accuracy is measured across labelled evaluation rows. A card's percentage is the model's estimated class probability for one submitted profile; it is not a measured accuracy or calibrated medical probability.

### What does deployment currently mean in this project?
The current demo runs locally through a Python API and browser frontend. Vercel/Render-related code is retained, but this tuning task did not deploy the new simulation models to the cloud.

### How can you prove the old data were not lost?
Protected-file SHA-256 manifests and verified backups preserve the old files. New experiments write to new directories and do not overwrite the original checkpoints or CSVs.

## 15. Evidence checklist

The guide was written from actual saved artifacts, not reconstructed accuracy claims. Inspect these files for the exact values and code:

- experiments/lower_noise_rf_cnn_tuning_20261026/protocol.json
- experiments/lower_noise_rf_cnn_tuning_20261026/cv_selected.json
- experiments/lower_noise_rf_cnn_tuning_20261026/cv_summary.json
- experiments/lower_noise_rf_cnn_tuning_20261026/validation.json
- experiments/lower_noise_rf_cnn_tuning_20261026/test_results.json
- experiments/lower_noise_rf_cnn_tuning_20261026/runtime_verification.json
- experiments/lower_noise_rf_cnn_tuning_20261026/activation.json
- experiments/lower_noise_simulation_20261026/data_manifest.json
- src/training_variants.py
- src/training_preview_inference.py
- tools/tune_lower_noise_rf_cnn.py
