# Project map for your viva

```text
model_training/
  All_Six_Models_Training.ipynb   <- open this to study
  base_models/
    model_training_data.csv     <- current 10,000 profiles
    training_roles.csv          <- exact fitting/development IDs
    models/                     <- base checkpoints and encoders
  improved_models/
    models/                     <- improved RF and CNN
    overrides.json              <- selected replacements
models/
  current_models/               <- separate real-BRFSS models
  training_preview.json         <- demo model selector
  path_aliases.json             <- old-to-new experiment locations
src/                            <- preprocessing, architectures, inference
frontend/                       <- website
training/                       <- friendly training entry points
docs/                           <- walkthrough and file guide
archive/previous_experiments/   <- preserved older experiments
backups/                        <- rollback copies; do not delete
```

The current demo CSV contains generated simulation profiles. Neutral filenames do not change their provenance or establish real-person accuracy. This notebook includes training code and checkpoint-derived outputs; default execution does not retrain.

Historical manifests retain their original contents and hashes. `src/project_paths.py` resolves old locations using `models/path_aliases.json`. Old archived standalone scripts/PDF path references may need this map; current loading and notebook use the new structure.

Rollback of folder organization: first stop the demo server; reverse the prefix and filename moves in `models/path_aliases.json`, then restore changed source/config files from `backups/notebook_organization_20261026/` (and `final_updates/` for later edits). Do not restore the selector alone while folders remain relocated. No dataset or checkpoint contents were changed.
