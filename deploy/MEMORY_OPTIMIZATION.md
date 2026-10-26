# Lossless inference optimization — 2026-10-21

The original CSVs, six trained checkpoints, preprocessor, calibration multipliers,
metrics and backups are unchanged. No retraining, quantization, pruning, reduced
precision, new training data or replacement models are involved.

## What changed

The 250-tree random-forest joblib contains training metadata and allocates about
360 MiB when loaded. `tools/export_compact_forest.py` creates separate inference
arrays containing only split features, exact float64 thresholds, child indices
and exact float64 leaf values. Integer indices are narrowed only after bounds
checks. The derived arrays occupy 44,875,306 bytes, compared with the original
188,145,217-byte checkpoint, and are loaded read-only with memory mapping.

Both the original checkpoint and every derived array are SHA-256 checked before
use. A mismatch fails closed; it never substitutes fake predictions. The exporter
refuses to overwrite an existing output directory. Keep the original checkpoint.

The Docker build generates these arrays and sets `BRFSS_COMPACT_FOREST_DIR`.
Without that environment variable, local inference keeps its previous behavior.
Unset it to return to the original runtime (which needs more memory). The Docker
server uses one model-owning Gunicorn worker with four request threads and a
prediction lock. The master does not preload the models.

## Verification

- Forest comparison: all 183,040 real respondent records, zero class changes.
- Maximum forest probability difference: 1.11e-15 (floating-point summation).
- All six models: 512 spread-out records / 3,072 predictions, zero class changes.
  The five unchanged models returned identical probabilities in that comparison.
- Standalone engine peak RSS: 727.875 MiB before, 414.082 MiB after.
- Actual authenticated Gunicorn API: tested inside a 512 MiB Linux cgroup with
  swap disabled, including 512 prediction requests, invalid-input rejection,
  authentication checks and concurrent-request backpressure. See saved reports
  for whole-service memory, including master, worker, test client and file cache.

These tests do not constitute a cloud deployment or guarantee bit-for-bit
floating-point agreement on different CPUs. Model classes matched in all tested
cases; the tiny forest summation differences are not retraining or rounding of
stored model parameters. Docker/Render verification remains required.

## Reproduce in the existing Linux Python environment

From the project root, choose a new output directory for the derived arrays:

```sh
python tools/export_compact_forest.py models/current_models/random_forest.joblib /tmp/new-compact-forest
python tests/forest_full_parity.py --project "$PWD" --compact /tmp/new-compact-forest --output /tmp/forest-parity.json
python tests/compact_runtime_validation.py --project "$PWD" --output /tmp/baseline.json
python tests/compact_runtime_validation.py --project "$PWD" --compact /tmp/new-compact-forest --output /tmp/optimized.json
```

The API memory test uses `systemd-run` with `MemoryMax=512M`, `MemorySwapMax=0`,
`--uid=nobody`, `--wait --pipe --collect`, and invokes:

```sh
python tests/cloud_memory_validation.py --installed --project /absolute/project --compact /tmp/new-compact-forest --output /tmp/api-memory.json
```

Run only the test process inside that transient cgroup; do not change systemwide
memory settings. Gunicorn is the only newly added server dependency (plus its
packaging dependency and the environment's bundled pip installer).
