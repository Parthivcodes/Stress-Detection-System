"""Shared validation/runtime rules for raw-probability soft ensembles."""
import numpy as np


def ordinal_probabilities(at_least_sometimes, high):
    a = np.clip(np.asarray(at_least_sometimes, dtype=np.float64), 0, 1)
    b = np.minimum(np.clip(np.asarray(high, dtype=np.float64), 0, 1), a)
    return np.column_stack((1-a, a-b, b))


def weighted_probabilities(raw, weights):
    if not weights or any(w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError('Ensemble weights must be nonnegative and sum to one.')
    p = sum(float(weight) * np.asarray(raw[name], dtype=np.float64) for name, weight in weights.items())
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid model probabilities.')
    return p / p.sum(axis=-1, keepdims=True)


def classify(probabilities, threshold):
    p = np.asarray(probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid high-stress threshold.')
    # A single cutoff applies AFTER averaging raw probabilities. Lower-class
    # ties use class 0. No per-model multipliers enter the ensemble.
    return np.where(p[..., 2] >= threshold, 2, p[..., :2].argmax(axis=-1))


def matched_threshold(y, p, recall=0.60):
    high = np.sort(np.asarray(p)[np.asarray(y) == 2, 2])
    if not len(high):
        raise ValueError('Validation must contain high-stress cases.')
    required = int(np.ceil(recall * len(high)))
    # Highest cutoff achieving the requested recall; ties may raise recall.
    return float(high[len(high)-required])
