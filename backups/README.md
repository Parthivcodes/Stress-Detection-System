# Preserved model versions

Relocated on 20 October 2026, without retraining or changing checkpoint bytes.
These are local backups in the same project, not an independent off-device backup.
Datasets now have descriptive folder names under `data/`; historical reports retain their names.

| Backup | Original location |
| --- | --- |
| `original_models/*` | `models/*` (eight legacy model/preprocessing files) |
| `baseline_models/` | `models/brfss2022/` |
| `tuned_models/` | `models/brfss2022_tuned/` |
| `previous_active_model.json` | Copy of the previous `models/active_model.json` |

## Use an older version without moving files back

In PowerShell, launch a separate WSL process with an explicit version:

```powershell
wsl.exe -d Ubuntu -u nobody --cd $PWD.Path --exec env PORT=8005 BRFSS_MODEL_VERSION=brfss2022_tuned /opt/stress-venv/bin/python -u server.py
```

Use `BRFSS_MODEL_VERSION=brfss2022` for the first BRFSS version. For the original
sleep/lifestyle backend use `STRESS_DATASET=legacy` instead. Keep these separate
from the expanded version; their inputs and dataset differ.

To make the tuned version the default again, set `version` in
`models/active_model.json` to `brfss2022_tuned`, then restart the relevant server.
No checkpoint relocation is necessary.

Historical preservation manifests are intentionally unchanged. Their old paths
resolve through `src/model_paths.py`; the old active-config checksum verifies
the snapshot here, not the deliberately updated live default.

## Verification

`tests/model_layout_validation.py --inference` passed after relocation:
all 47 original data/model files match their prior SHA-256 hashes, all three
historical preservation manifests validate, each BRFSS version returns six
real predictions, and all six legacy models load from their backup location.
Legacy/baseline training entry points refuse to overwrite completed backups.
