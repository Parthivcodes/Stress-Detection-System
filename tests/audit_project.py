"""Read-only project audit; writes only its JSON report when --output is supplied."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_paths import model_dir
EXCLUDED = {".git", ".venv", ".runtime", "node_modules", ".npm-cache", ".cleanup"}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit():
    inventory, hashes, sizes = [], defaultdict(list), defaultdict(int)
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink()]
        for name in files:
            p = Path(directory) / name
            if p.is_symlink():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel.startswith("reports/project_cleanup/"):
                continue
            length = p.stat().st_size
            digest = sha(p)
            inventory.append({"path": rel, "bytes": length, "sha256": digest})
            hashes[digest].append(rel)
            sizes[rel.split("/")[0]] += length

    datasets, ids = {}, {}
    for p in sorted((ROOT / "data").rglob("*.csv")):
        with p.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        rel = p.relative_to(ROOT).as_posix()
        id_col = next((k for k in ["respondent_id", "Person ID"] if k in fields), None)
        numeric = categorical = []
        if "respondent_id" in fields:
            version = {"baseline_survey_dataset": "brfss2022",
                       "expanded_survey_dataset": "brfss2022_expanded"}[p.parent.name]
            schema = json.loads((model_dir(version) / "schema.json").read_text())
            numeric, categorical = schema["numeric"], schema["categorical"]
        features = numeric + categorical
        if not features:
            features = [k for k in fields if k != id_col]
        exact = Counter(tuple(r[k] for k in fields) for r in rows)
        profiles = Counter(tuple(r[k] for k in features) for r in rows)
        row_ids = Counter(r[id_col] for r in rows) if id_col else Counter()
        groups = defaultdict(set)
        if "split" in fields:
            for r in rows:
                groups[tuple(r[k] for k in features)].add(r["split"])
        ids[rel] = set(row_ids)
        datasets[rel] = {
            "rows": len(rows), "id_column": id_col,
            "unique_ids": len(row_ids), "duplicate_id_excess": sum(n-1 for n in row_ids.values()),
            "exact_duplicate_row_excess": sum(n-1 for n in exact.values()),
            "profile_columns": features, "unique_profiles": len(profiles),
            "repeated_profile_excess": sum(n-1 for n in profiles.values()),
            "profiles_across_splits": sum(len(s) > 1 for s in groups.values()),
            "split_counts": dict(Counter(r.get("split", "unsplit") for r in rows)),
        }
    overlaps = []
    paths = list(ids)
    for i, a in enumerate(paths):
        for b in paths[i+1:]:
            if "survey_dataset" in a and "survey_dataset" in b:
                overlaps.append({"a": a, "b": b, "shared_respondent_ids": len(ids[a] & ids[b])})
    return {"excluded_dependency_or_archive_directories": sorted(EXCLUDED),
            "inventory": inventory, "directory_bytes": dict(sizes),
            "exact_duplicate_files": [v for v in hashes.values() if len(v) > 1],
            "datasets": datasets, "dataset_overlap": overlaps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / "reports" / "project_cleanup"):
            raise ValueError("Audit outputs must stay in reports/project_cleanup")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "inventory"}, indent=2))
