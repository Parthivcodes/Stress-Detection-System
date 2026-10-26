# Project File Guide - read this line by line

## The five places to start

1. README.md - start here to launch the main data website and find the study guide.
2. docs/01_Model_Training_Walkthrough.pdf - complete printable training and accuracy explanation.
3. training/improve_forest_and_cnn.py - friendly entry point for the relevant improvement workflow.
4. models/ACTIVE_MODELS.md - exact active checkpoint/encoder mapping by algorithm name.
5. model_training/improved_models/README.md - measured before/after results and rollback.

## Directory structure

```text
stressleveldetection/
  README.md                         Start here
  Start-Main-Website.ps1             Primary Main Website launcher (port 8001)
  Start-Website.ps1                  Main Website launcher (delegates to Start-Main-Website.ps1)
  Start-RealData-Website.ps1         Historical raw-data reference launcher (port 8000)
  Start-Demo-Website.ps1             Alternate launcher (delegates to Start-Main-Website.ps1)
  frontend/                         Main page, inputs, results, animation, PDF export
  src/                              Model definitions, encoders and inference loaders
  training/                         Clearly named current entry points and original training scripts
  models/                           Selection files, active models, index
  data/                             Preserved source datasets
  experiments/                      Versioned datasets, checkpoints and scientific evidence
  tests/                            Validation and API checks
  tools/                            Current implementations and shared compatibility helpers
  docs/                             Walkthrough, file guide and earlier documentation
  reports/                          Audits, evaluations and cleanup manifests
  deploy/                           Preserved deployment configuration
  api/ + serverless/                Preserved frontend-to-backend cloud adapters
  backups/                          Recoverable known-working datasets/models/configs
  archive/research_scripts/         Older exploratory tools, preserved verbatim
  archive/maintenance_scripts/      Previous setup/migration/alternate-launch helpers
  archive/generated_website/        Previous public/ build, recoverable and rebuildable
  figures/ + output/                Saved plots, reference assets and prior outputs
  .venv/ + .runtime/                Preserved local environments/dependencies
  .cleanup/                         Earlier recovery snapshots; not active training data
```

## Which dataset belongs to which result?

| Location | What it denotes |
|---|---|
| model_training/base_models/model_training_data.csv | Primary 10,000-profile dataset; powers the Main Website (~87-91% accuracy) |
| model_training/improved_models/ | High-accuracy RF, CNN and tuned model checkpoints for the Main Website |
| data/expanded_survey_dataset/respondents.csv | Historical raw BRFSS survey dataset |
| archive/previous_experiments/initial_training/model_training_data.csv | Initial training exploration dataset |

## Current runtime files

| File | Purpose |
|---|---|
| tools/start_training_preview.py | Main Website server (port 8001) |
| server.py | Server engine and HTTP request handler |
| models/training_preview.json | Selects active model configuration for Main Website |
| models/active_model.json | Historical raw-data selection |
| src/training_preview_inference.py | High-accuracy six-model inference engine |
| src/training_variants.py | Tuned forest encoder and position-preserving CNN |
| src/models.py | Original CNN, LSTM, CNN-LSTM architecture definitions |
| src/brfss_inference.py | Separate real survey-model inference |
| frontend/index.html | Main website structure |
| frontend/landing.js and landing.css | Particle experience, layout and navigation |
| frontend/prediction.js and prediction.css | Nineteen-input form, API calls, result cards and benchmark table |
| frontend/analysis-pdf.js | Browser-only readable PDF generation |
| frontend/dashboard/ | Additional React dashboard; its dist/ build is still needed by the server |

## Current training entry points

| Easy entry point | Verified implementation |
|---|---|
| training/create_and_train_demo.py | tools/train_lower_noise_simulation.py |
| training/improve_forest_and_cnn.py | tools/tune_lower_noise_rf_cnn.py |
| training/create_demo_training_report.py | tools/report_lower_noise_simulation.py |

These entry points delegate rather than duplicate training logic. Shared helper module names remain stable because old notebooks/imports depend on them: tune_six_models.py, tune_two_boundaries.py, tune_xgb_features.py and feature_pipeline.py. The historical full workflows inside these helper files are not the recommended current entry points.

tools/simulation_training.py is the shared generation/baseline-fitting recipe. prepare_rf_cnn_promotion.py prepares a qualified update manifest; it does not bypass performance gates. verify_rf_cnn_runtime.py checks predictions, input handling and memory. build_vercel.cjs and export_compact_forest.py are retained because the build/deployment configuration references them.

## Reading an experiment directory

| File or folder | Meaning |
|---|---|
| protocol.json | Frozen design, seeds, budgets, safeguards and selection rules |
| protected_hashes.json | Digests of existing files that must not change |
| source/ and frozen_source_hashes.json | Versioned source snapshot and checksums |
| cv_*.json, cv_predictions.npz, cv_roles.csv | Training-fold results, probabilities and sample roles |
| cv_selected.json | Winner selected without candidate holdout scores |
| validation.json | Candidate-versus-current validation acceptance |
| models/ | Newly trained checkpoints and matching encoders |
| frozen_candidates.json | Final frozen artifacts and architecture/config metadata |
| TEST_STARTED.json | Guard/evidence that the final pass was initiated after freezing |
| test_results.json | Before/after class metrics, confusion matrices and paired intervals |
| overrides.json | Qualified model paths, hashes and metrics used by the demo loader |
| runtime_verification.json | Six-path parity, input checks, memory and latency |
| activation.json | What was activated locally and what was verified |
| rollback_verification.json | Evidence the prior six-model bundle still loads |

## Naming rules used now

- Start-...Website.ps1 means a human-facing launcher.
- training/create_... means a fresh experiment workflow; improve_... means tuning on existing data.
- src/ files define reusable runtime behaviour; frontend/ files define user-visible behaviour.
- protocol, validation, test_results and activation are distinct stages, not interchangeable accuracy reports.
- A seed suffix identifies a training run. Multiple neural seeds are intentional and averaged.
- Historic experiment IDs and checkpoint names are retained where frozen hashes and rollback depend on them. ACTIVE_MODELS.md translates them into ordinary algorithm names.
- Archive files are historical references; current runtime and training imports must not depend on them.

## What cleanup does and does not remove

Historical research/setup scripts and the old public/ build are moved into archive/ with checksums. Byte-identical source/build files are reported, not indiscriminately deleted. Only verified first-party .pyc cache directories are deleted. No datasets, model checkpoints, immutable experiment evidence, source assets, dashboard/dist, existing backups or Python environments are deleted.

reports/directory_cleanup/cleanup_manifest.json records every move, checksum and removed cache. Restore an archived file to its listed original path to undo a move. Existing archive/ and backup/ contents are never overwritten.
