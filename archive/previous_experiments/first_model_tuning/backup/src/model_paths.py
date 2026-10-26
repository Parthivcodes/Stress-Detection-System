"""Canonical model locations and relocation of historical manifest paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
BACKUP_ROOT = PROJECT_ROOT / "backups"
LEGACY_ROOT = BACKUP_ROOT / "original_models"


def model_dir(version):
    if version == "brfss2022_expanded":
        return MODEL_ROOT / "current_models"
    if version == "brfss2022":
        return BACKUP_ROOT / "baseline_models"
    if version == "brfss2022_tuned":
        return BACKUP_ROOT / "tuned_models"
    raise ValueError(f"Unsupported model version: {version}")


def preserved_path(relative):
    """Resolve old evidence paths without changing historical hash manifests.

    The former active-version config is a snapshot, not the live default.
    """
    # Evidence files retain their original names and checksum keys.
    aliases = {
        "data/Sleep_health_and_lifestyle_dataset.csv": "data/original_sleep_dataset/sleep_lifestyle.csv",
        "data/brfss2022_expanded/": "data/expanded_survey_dataset/",
        "data/brfss2022/": "data/baseline_survey_dataset/",
        "models/brfss2022_expanded/": "models/current_models/",
        "backups/models/brfss2022/": "backups/baseline_models/",
        "backups/models/brfss2022_tuned/": "backups/tuned_models/",
        "backups/models/legacy/": "backups/original_models/",
    }
    normalized = str(relative).replace("\\", "/")
    for old, new in aliases.items():
        if normalized == old or (old.endswith("/") and normalized.startswith(old)):
            return PROJECT_ROOT / (new + normalized[len(old):])
    path = Path(normalized)
    parts = path.parts
    if parts[:1] == ("models",):
        if len(parts) > 2 and parts[1] in ("brfss2022", "brfss2022_tuned"):
            return model_dir(parts[1]).joinpath(*parts[2:])
        if len(parts) == 2 and parts[1] == "brfss_active.json":
            return BACKUP_ROOT / "previous_active_model.json"
        if len(parts) == 2 and parts[1] not in ("README.md",):
            return LEGACY_ROOT / parts[1]
    return PROJECT_ROOT / path
