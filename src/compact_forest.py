"""Lossless, read-only random-forest inference without training-only tree state.

Thresholds and leaf values remain float64. Only exact integer node indices
are narrowed. This module intentionally does not import sklearn or torch.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CompactForest:
    def __init__(self, directory, original_checkpoint):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest["format"] != 1 or sha256(original_checkpoint) != manifest["source_sha256"]:
            raise RuntimeError("Compact forest does not match the original checkpoint.")
        self.arrays = {}
        for name, digest in manifest["files"].items():
            path = self.directory / name
            if sha256(path) != digest:
                raise RuntimeError(f"Compact forest checksum mismatch: {name}")
            self.arrays[path.stem] = np.load(path, mmap_mode="r", allow_pickle=False)
        self.n_features_in_ = manifest["n_features"]
        self.classes_ = np.asarray(manifest["classes"])
        self.tree_count = manifest["tree_count"]

    def predict_proba(self, x):
        # sklearn's trees also convert their input to float32 before traversal.
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("Expected finite, two-dimensional model features.")
        a = self.arrays
        result = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        for root in a["roots"]:
            nodes = np.full(len(x), root, dtype=np.int32)
            while True:
                rows = np.flatnonzero(nodes >= 0)
                if not len(rows):
                    break
                active = nodes[rows]
                left = x[rows, a["feature"][active]] <= a["threshold"][active]
                nodes[rows] = np.where(left, a["left"][active], a["right"][active])
            p = np.array(a["values"][-nodes - 1], copy=True)
            normalizer = p.sum(axis=1)
            normalizer[normalizer == 0.0] = 1.0
            p /= normalizer[:, None]
            result += p
        result /= self.tree_count
        return result
