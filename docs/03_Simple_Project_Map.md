# Project map for your viva

```text
model_training/
  All_Six_Models_Training.ipynb   <- open this to study all 6 models
  base_models/
    model_training_data.csv     <- primary 10,000 profile dataset (powers Main Website)
    training_roles.csv          <- exact fitting/development IDs
    models/                     <- base checkpoints and encoders
  improved_models/
    models/                     <- improved RF and CNN models
    overrides.json              <- selected high-accuracy models
models/
  training_preview.json         <- active model selector for Main Website
  current_models/               <- historical raw-BRFSS reference models
  path_aliases.json             <- old-to-new experiment locations
src/                            <- preprocessing, architectures, inference
frontend/                       <- main website interface
training/                       <- friendly training entry points
docs/                           <- walkthrough and file guide
archive/previous_experiments/   <- preserved older experiments
backups/                        <- rollback copies; do not delete
```

The primary application is the **Main Website** (launched via `Start-Main-Website.ps1` or `Start-Website.ps1` at `http://localhost:8001/#predictions`), powered by the primary 10,000-profile dataset across all six high-performing models (~87-91% accuracy). The study notebook (`All_Six_Models_Training.ipynb`) documents the complete six-model training pipeline and evaluation.

Historical manifests retain their original contents and hashes. `src/project_paths.py` resolves old locations using `models/path_aliases.json`. Old archived standalone scripts/PDF path references may need this map; current loading and notebook use the new structure.

Rollback of folder organization: first stop the demo server; reverse the prefix and filename moves in `models/path_aliases.json`, then restore changed source/config files from `backups/notebook_organization_20261026/` (and `final_updates/` for later edits). Do not restore the selector alone while folders remain relocated. No dataset or checkpoint contents were changed.
