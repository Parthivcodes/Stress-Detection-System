# Readable project layout — 20 October 2026

The root README is the current line-by-line project guide.

## Preservation

- 116 files were inventoried before relocation and verified byte-for-byte
  immediately after moving. See `reorganization.json` for old/new paths.
- All 48 relocated data/model/config files still match those original hashes
  after path updates. No CSV rows, weights, preprocessing parameters or metrics
  were changed. Source/configuration/documentation edits are intentional.
- All 47 files covered by the earlier data/model audit and the three historical
  preservation manifests also validate through historical path aliases.
- Dependency folders moved with the React dashboard; their individual files were
  excluded from hashing. No dependencies were reinstalled.
- Existing user deletions of earlier defense/reference documents were preserved.
  The prior README was retained in `docs/PREVIOUS_README.md`.

## Verification

- `tests/model_layout_validation.py --inference`: passed. Six actual predictions
  for each BRFSS version; all six original sleep/lifestyle models load.
- `tests/website_structure_validation.py`: passed. Default expanded version,
  19-input schema, main page, local assets, workspace redirect, React build,
  six HTTP prediction results matching direct inference, and HTTP 400 for
  invalid inputs. Relocated training modules import without executing training.
- `tests/landing-smoke.cjs`: passed, including reduced motion, interrupted
  animations, pause/resume, mobile budget and no-WebGL fallback.

## Naming decisions

Human-readable directory names replace technical names in the working layout.
Internal version IDs, historical report names and model artifact filenames
retain their established names for traceability and checkpoint compatibility.
`src/model_paths.py` maps between version IDs and friendly directories.

Launch with `Start-Website.ps1`; alternate launchers are under `tools/`.
The local `/dashboard/` route now explicitly serves the relocated React build.
The React dashboard remains a separately documented fictional-demo frontend;
real six-model predictions are on the main page's predictions section.

## Not changed

Training results, feature definitions, target labels and split assignments were
not modified. The single-threaded development HTTP server and older Vercel API
behavior are outside this file-reorganization change. Use local Python inference
for the actual model-backed demonstration.
