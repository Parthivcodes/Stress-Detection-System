# Model versions — do not mix their preprocessing or inputs

## Current layout (updated after the cleanup audit)

The default is now `brfss2022_expanded`. Only this model set and the active
configuration remain here. All older models described below have moved to
`backups/baseline_models/` and `backups/tuned_models/`, while the
original top-level files are in `backups/original_models/`.
See `backups/README.md` for rollback instructions.

| Location | Purpose |
| --- | --- |
| `backups/original_models/` | Original sleep/lifestyle models |
| `backups/baseline_models/` | First 30,000-respondent BRFSS baseline, 12 inputs |
| `backups/tuned_models/` | Same survey cohort with tuned prediction settings |
| `models/current_models/` | Latest six models with 19 inputs |

Use `Start-Website.ps1` for the latest website on port 8004.
`active_model.json` selects the expanded version for launchers without an explicit
version override. Its previous contents are preserved in the backup folder.

The six algorithms are Logistic Regression, Random Forest, XGBoost, 1D CNN,
LSTM and CNN-LSTM. Similar filenames in different folders identify versions,
not six copies of the same algorithm.

SHA-256 auditing on 2026-10-20 found that baseline and tuned LR, XGBoost, CNN
and LSTM checkpoint files are identical. The tuned pipeline intentionally reuses
weights and applies its `selection.json` multipliers. CNN's chosen adjustment is
unchanged. Their preprocessor and schema are also identical. Keeping these files
preserves self-contained versions, rollback and existing hash-verification tests.
No latest expanded checkpoint duplicates another checkpoint in this project.

Do not delete the baseline versions merely because the expanded website exists:
the comparison/validation scripts and preservation manifests still reference them.
Their paths now resolve through `src/model_paths.py`.
See `reports/project_cleanup/README.md` for the dataset and cleanup audit.
