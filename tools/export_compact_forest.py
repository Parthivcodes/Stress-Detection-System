"""Generate lossless derived inference arrays; never overwrite a checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.compact_forest import sha256


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    model = joblib.load(source)
    if model.n_outputs_ != 1 or model.n_features_in_ >= np.iinfo(np.int16).max:
        raise ValueError("Unsupported forest shape.")
    internal_count = sum(np.count_nonzero(t.tree_.children_left >= 0) for t in model.estimators_)
    leaf_count = sum(np.count_nonzero(t.tree_.children_left < 0) for t in model.estimators_)
    if max(internal_count, leaf_count) >= np.iinfo(np.int32).max:
        raise ValueError("Forest exceeds exact int32 index capacity.")
    arrays = {
        "roots": np.empty(len(model.estimators_), dtype=np.int32),
        "left": np.empty(internal_count, dtype=np.int32),
        "right": np.empty(internal_count, dtype=np.int32),
        "feature": np.empty(internal_count, dtype=np.int16),
        "threshold": np.empty(internal_count, dtype=np.float64),
        "values": np.empty((leaf_count, len(model.classes_)), dtype=np.float64),
    }
    ni = nl = 0
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        internal = np.flatnonzero(tree.children_left >= 0)
        leaves = np.flatnonzero(tree.children_left < 0)
        mapping = np.empty(tree.node_count, dtype=np.int32)
        mapping[internal] = np.arange(ni, ni + len(internal))
        mapping[leaves] = -np.arange(nl + 1, nl + len(leaves) + 1)
        segment = slice(ni, ni + len(internal))
        arrays["roots"][index] = mapping[0]
        arrays["left"][segment] = mapping[tree.children_left[internal]]
        arrays["right"][segment] = mapping[tree.children_right[internal]]
        arrays["feature"][segment] = tree.feature[internal]
        arrays["threshold"][segment] = tree.threshold[internal]
        arrays["values"][nl:nl + len(leaves)] = tree.value[leaves, 0, :]
        ni += len(internal)
        nl += len(leaves)
    destination.mkdir(parents=True)
    files = {}
    for name, array in arrays.items():
        path = destination / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        files[path.name] = sha256(path)
    manifest = {
        "format": 1, "source_sha256": sha256(source),
        "n_features": int(model.n_features_in_), "classes": model.classes_.tolist(),
        "tree_count": len(model.estimators_), "files": files,
        "note": "Exact float64 thresholds and leaf values; no retraining or quantization.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"trees": len(model.estimators_), "inference_bytes": sum(a.nbytes for a in arrays.values()), "source_sha256": manifest["source_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    export(args.source, args.destination)
