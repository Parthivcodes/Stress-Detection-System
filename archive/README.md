# Archived project history

These files were moved rather than deleted. research_scripts/ contains past experiments and report builders; maintenance_scripts/ contains previous setup and migration helpers. generated_website/ preserves the old public/ build, if it existed. None is required by the current local prediction runtime.

Original script bytes and SHA-256 hashes are recorded in reports/directory_cleanup/cleanup_manifest.json. Historical commands inside these scripts may refer to their former locations. Use immutable source snapshots saved with each experiment when investigating an older result; these archived scripts are references, not the current training entry points. Current training starts in training/create_and_train_demo.py or training/improve_forest_and_cnn.py.

Do not remove datasets, saved experiment evidence or checkpoints merely because similar filenames exist in multiple runs: they can represent different data, seeds and hyperparameters.
