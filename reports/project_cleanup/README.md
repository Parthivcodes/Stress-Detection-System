# Project duplicate and cleanup audit — 20 October 2026

Follow-up: at the user's request, the expanded version is now the default.
Older models were subsequently moved to `backups/models/`; see that folder's
README. The findings below describe the earlier audit. `before.json` and
`after.json` remain historical snapshots, not the current directory layout.

## Dataset findings

| Dataset | Rows | Unique IDs | Exact repeated rows | Repeated answer-profile excess |
| --- | ---: | ---: | ---: | ---: |
| Original sleep/lifestyle | 374 | 374 | 0 | 242 |
| BRFSS baseline | 30,000 | 30,000 | 0 | 5 |
| BRFSS expanded | 183,040 | 183,040 | 0 | 118 |

BRFSS profiles use each version's model input columns. The legacy comparison
uses all columns except Person ID, including Stress Level. These are not
necessarily confirmed repeated people: a different respondent can give the same
answers. Unique IDs alone cannot independently establish the legacy dataset's
real-world provenance. No observations were removed or changed.

Neither BRFSS dataset contains an identical input profile spanning multiple
splits. The two BRFSS dataset versions share 29,582 respondent IDs by design;
they must not be concatenated and described as wholly independent respondents.
The expanded pipeline quarantines prior evaluation respondents/profiles.
The original and expanded CSVs are separate training versions, not duplicate files.

## Model findings

Four checkpoint pairs are byte-identical across baseline and tuned versions:
`conv1d_net.pt`, `logistic_regression.joblib`, `lstm_net.pt`, `xgboost.joblib`.
Their preprocessor and schema are identical too. Tuned inference also reads
`selection.json`, so identical weights need not imply identical predictions.
The CNN is an unchanged retained baseline. All six expanded checkpoints have
distinct hashes from each other and the older checkpoints.

Keep all versions: inference, legacy apps, baseline comparison tests and frozen
preservation manifests depend on their current paths and bytes. No retraining,
model substitution, active-version change or metric changes were made.

## Cleanup

Moved unused historical frontend drafts, eight obsolete top-level JS/CSS files,
project/source Python bytecode caches and the TypeScript incremental-build cache
into `.cleanup/2026-10-20/`, preserving their relative paths. The current
`index.html`/workspace redirect and tests do not reference those JS/CSS files.
The archive is ignored by Git and excluded from Vercel uploads.

Restore from the project directory with:

```powershell
.\tools\Cleanup-ObsoleteFiles.ps1 -Restore
```

Restore refuses to overwrite files. If caches have been regenerated, resolve
those conflicts first; source assets are not overwritten automatically.
This is recoverable organization, not disk-space reclamation.

Preserved `wellness-dashboard` and its built `dist`: they remain referenced by
the root npm workspace, Vercel `/dashboard` routes and the visible 3D Cockpit link.
Preserved reference frames, reports, raw CDC source and backup ZIPs as research
and design evidence. Preserved `.venv`, `.runtime` and `node_modules`: launchers
still reference them or they supply build dependencies. Dependency internals are
excluded from byte-for-byte duplicate analysis, not assumed disposable.

## Reproduce

Post-cleanup verification: all 47 existing files under `data/` and `models/`
retain their pre-cleanup SHA-256 hashes. All 38 archived files (1,553,869 bytes)
match their original hashes. Landing-page behavioral smoke tests passed,
prediction JavaScript syntax passed, and all eight local asset references in
the main/redirect HTML resolve. No live prediction request was run for this
cleanup; model binaries and inference code were not changed.

Run `tests/audit_project.py --output reports/project_cleanup/after.json` with
Python 3.11+. It hashes project files and independently counts CSV identities,
rows, profiles, split overlaps and cross-version IDs. It excludes dependency,
Git and cleanup-archive internals. `before.json` records the pre-cleanup hashes.

Existing issues outside this cleanup: the local Python server uses single-threaded
HTTPServer (previously observed request stalls); older README/defense sections
describe the legacy dataset, not the latest expanded research. Do not quote the
legacy accuracy as expanded-model accuracy. These are not duplicate-file issues.
