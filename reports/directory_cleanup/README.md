# Directory cleanup verification

Historical scripts and the generated public/ snapshot were archived, preserving original bytes. Only checked first-party .pyc cache files were deleted. All protected dataset, checkpoint, experiment-evidence and retained-runtime hashes match the pre-cleanup inventory.

The current model/data layout validation passed. All three friendly training entry points passed CLI/import checks without fitting. Both friendly launchers passed PowerShell syntax checks. Both running local websites returned all six predictions; the PDF exporter remained available. The11-page walkthrough PDF was rendered and visually inspected.

See cleanup_manifest.json for every relocation and removed cache, cleanup_plan.json for protected hashes, inventory_before.csv for the first-party inventory, duplicate_files.json for identical-file groups and verification.json for the final checks.

To undo a move, return its archive destination to the listed source path. The original root README, ignore rules and layout test are saved under backups/directory_cleanup_20261026. New documentation and entry points can remain in place; no model rollback is necessary.
