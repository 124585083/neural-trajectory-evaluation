from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gpfa import GaussianProcessFactorAnalysis
from .metrics import METRIC_DIRECTIONS, trajectory_metrics


@dataclass(frozen=True)
class ReliabilityConfig:
    frame_rate_hz: float = 30.0
    observation_step: int = 4
    n_splits: int = 200
    seed: int = 0
    block_sizes: tuple[int, ...] = (4, 8, 16, 32)
    null_names: tuple[str, ...] | None = None


def _balanced_half(repeats: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(repeats)
    half = repeats // 2
    return np.sort(order[:half]), np.sort(order[half : 2 * half])


def _derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(size)
    for _ in range(100):
        candidate = rng.permutation(size)
        if np.all(candidate != base):
            return candidate
    return np.roll(base, 1)


def _block_shuffle(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [values[:, start : start + block_size] for start in range(0, values.shape[1], block_size)]
    order = rng.permutation(len(blocks))
    if np.array_equal(order, np.arange(len(blocks))):
        order = np.roll(order, 1)
    return np.concatenate([blocks[index] for index in order], axis=1)


def _null_response(
    values: np.ndarray,
    null_name: str,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> np.ndarray:
    conditions, time, neurons = values.shape
    if null_name == "condition_shuffle":
        return values[_derangement(conditions, rng)]
    if null_name == "circular_shift":
        result = values.copy()
        for condition in range(conditions):
            shift = int(rng.integers(max(1, time // 8), max(2, time - time // 8)))
            result[condition] = np.roll(result[condition], shift, axis=0)
        return result
    if null_name == "frame_shuffle":
        result = values.copy()
        for condition in range(conditions):
            result[condition] = result[condition, rng.permutation(time)]
        return result
    if null_name == "time_reversal":
        return values[:, ::-1].copy()
    if null_name == "block_shuffle":
        if block_size is None:
            raise ValueError("block_shuffle requires a block size")
        return _block_shuffle(values, block_size, rng)
    if null_name == "independent_neuron_shift":
        result = np.empty_like(values)
        for condition in range(conditions):
            shifts = rng.integers(1, time, size=neurons)
            for neuron, shift in enumerate(shifts):
                result[condition, :, neuron] = np.roll(values[condition, :, neuron], int(shift))
        return result
    raise KeyError(null_name)


def _latent(
    model: GaussianProcessFactorAnalysis,
    full_response: np.ndarray,
    observation_indices: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    observed = full_response[:, observation_indices]
    return model.transform_query(observed, query_times)


def run_split_half_reliability(
    oracle_conditions: list[np.ndarray],
    model: GaussianProcessFactorAnalysis,
    config: ReliabilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(oracle_conditions) < 2 or any(values.ndim != 3 or len(values) < 2 for values in oracle_conditions):
        raise ValueError("reliability requires multiple conditions with repeated trials")
    time = oracle_conditions[0].shape[1]
    if any(values.shape[1:] != oracle_conditions[0].shape[1:] for values in oracle_conditions):
        raise ValueError("all oracle trials must share time and neuron axes")
    observation_indices = np.arange(0, time, config.observation_step)
    if len(observation_indices) != len(model.times_seconds):
        raise AssertionError("GPFA observation grid and reliability grid differ")
    query_times = np.arange(time, dtype=np.float64) / config.frame_rate_hz
    rng = np.random.default_rng(config.seed)
    observed_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    all_null_specs = [
        ("condition_shuffle", None),
        ("circular_shift", None),
        ("frame_shuffle", None),
        ("time_reversal", None),
        ("independent_neuron_shift", None),
        *[("block_shuffle", size) for size in config.block_sizes],
    ]
    if config.null_names is None:
        null_specs = all_null_specs
    else:
        requested = set(config.null_names)
        null_specs = []
        for name, block_size in all_null_specs:
            label = name if block_size is None else f"block_shuffle_{block_size}"
            if label in requested:
                null_specs.append((name, block_size))
        missing = requested - {
            name if block_size is None else f"block_shuffle_{block_size}"
            for name, block_size in null_specs
        }
        if missing:
            raise ValueError(f"unknown requested nulls: {sorted(missing)}")
    for split in range(config.n_splits):
        left, right = [], []
        for repeats in oracle_conditions:
            left_indices, right_indices = _balanced_half(len(repeats), rng)
            left.append(repeats[left_indices].mean(axis=0))
            right.append(repeats[right_indices].mean(axis=0))
        left_response = np.stack(left)
        right_response = np.stack(right)
        left_latent = _latent(model, left_response, observation_indices, query_times)
        right_latent = _latent(model, right_response, observation_indices, query_times)
        for metric, value in trajectory_metrics(left_latent, right_latent, 1.0 / config.frame_rate_hz).items():
            observed_rows.append({"split": split, "metric": metric, "value": value})
        for null_name, block_size in null_specs:
            null_response = _null_response(right_response, null_name, rng, block_size)
            null_latent = _latent(model, null_response, observation_indices, query_times)
            label = null_name if block_size is None else f"block_shuffle_{block_size}"
            for metric, value in trajectory_metrics(left_latent, null_latent, 1.0 / config.frame_rate_hz).items():
                null_rows.append(
                    {"split": split, "null": label, "metric": metric, "value": value}
                )
    return pd.DataFrame(observed_rows), pd.DataFrame(null_rows)


def summarize_reliability(observed: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, observed_group in observed.groupby("metric"):
        direction = METRIC_DIRECTIONS[metric]
        observed_values = observed_group.sort_values("split")["value"].to_numpy()
        for null_name, null_group in nulls[nulls.metric == metric].groupby("null"):
            null_values = null_group.sort_values("split")["value"].to_numpy()
            if len(null_values) != len(observed_values):
                raise AssertionError("observed and null distributions are not paired")
            oriented = observed_values - null_values
            if direction == "lower":
                oriented = -oriented
            pooled = np.sqrt(0.5 * (np.var(observed_values, ddof=1) + np.var(null_values, ddof=1)))
            rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "null": null_name,
                    "observed_mean": float(np.mean(observed_values)),
                    "observed_ci_low": float(np.quantile(observed_values, 0.025)),
                    "observed_ci_high": float(np.quantile(observed_values, 0.975)),
                    "null_mean": float(np.mean(null_values)),
                    "null_ci_low": float(np.quantile(null_values, 0.025)),
                    "null_ci_high": float(np.quantile(null_values, 0.975)),
                    "paired_superiority": float(np.mean(oriented > 0)),
                    "paired_p_value": float((1 + np.sum(oriented <= 0)) / (len(oriented) + 1)),
                    "standardized_separation": float(np.mean(oriented) / max(pooled, 1e-12)),
                    "n_splits": int(len(oriented)),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "null"]).reset_index(drop=True)
