"""Compatibility helpers from the archived exploratory question analysis.

The current response control selects a real Dynamic checkpoint on validation;
run_checkpoint_matching.py is its executable entry point.
"""

from __future__ import annotations

import itertools

import numpy as np

def _sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    permutations = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permutations.append(float(np.mean(values * np.asarray(signs))))
    return float(np.mean(np.asarray(permutations) >= observed - 1e-15))

def _balanced_repeat_split(conditions: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selection, test = [], []
    for condition in np.unique(conditions):
        indices = np.flatnonzero(conditions == condition)
        indices = indices[rng.permutation(len(indices))]
        half = len(indices) // 2
        selection.extend(indices[:half])
        test.extend(indices[half : 2 * half])
    return np.asarray(selection, dtype=np.int64), np.asarray(test, dtype=np.int64)

def run_question_analysis(*args, **kwargs):
    raise RuntimeError("The historical output-noise analysis is archived. Use run_checkpoint_matching.py for validation-selected Dynamic checkpoints.")
