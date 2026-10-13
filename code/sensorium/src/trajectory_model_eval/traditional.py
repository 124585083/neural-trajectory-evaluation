from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr


def column_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("column correlation requires matching sample x feature arrays")
    x = left.astype(np.float64) - np.mean(left, axis=0, keepdims=True)
    y = right.astype(np.float64) - np.mean(right, axis=0, keepdims=True)
    numerator = np.sum(x * y, axis=0)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12
    )


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return column_correlation(left.T, right.T)


def scalar_correlation(left: np.ndarray, right: np.ndarray) -> float:
    value = column_correlation(left.reshape(-1, 1), right.reshape(-1, 1))[0]
    return float(value) if np.isfinite(value) else 0.0


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right):
        raise ValueError("CKA requires matching sample axes")
    x = left.astype(np.float64) - np.mean(left, axis=0, keepdims=True)
    y = right.astype(np.float64) - np.mean(right, axis=0, keepdims=True)
    cross = x.T @ y
    numerator = np.sum(cross * cross)
    denominator = np.sqrt(np.sum((x.T @ x) ** 2) * np.sum((y.T @ y) ** 2))
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    return squareform(pdist(patterns, metric="correlation"))


def compare_rdms(left: np.ndarray, right: np.ndarray, method: str) -> float:
    upper = np.triu_indices(len(left), 1)
    x, y = left[upper], right[upper]
    if method == "pearson":
        return scalar_correlation(x, y)
    value = spearmanr(x, y, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else 0.0


def condition_means(values: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    return np.stack([values[conditions == condition].mean(axis=0) for condition in np.unique(conditions)])


def _best_lag_correlation(left: np.ndarray, right: np.ndarray, max_lag: int = 15) -> tuple[float, int]:
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            score = scalar_correlation(left[:, -lag:], right[:, :lag])
        elif lag > 0:
            score = scalar_correlation(left[:, :-lag], right[:, lag:])
        else:
            score = scalar_correlation(left, right)
        scores.append(score)
    best = int(np.argmax(scores))
    return float(scores[best]), best - max_lag


def evaluate_traditional(
    neural: np.ndarray,
    prediction: np.ndarray,
    conditions: np.ndarray,
    *,
    cka_max_samples: int = 2000,
    temporal_rsa_stride: int = 10,
    seed: int = 42,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if neural.shape != prediction.shape or neural.ndim != 3:
        raise ValueError("traditional evaluation requires trial x time x neuron tensors")
    flat_neural = neural.reshape(-1, neural.shape[-1])
    flat_prediction = prediction.reshape(-1, prediction.shape[-1])
    per_neuron = column_correlation(flat_neural, flat_prediction)
    per_sample = row_correlation(flat_neural, flat_prediction)
    neural_mean = condition_means(neural, conditions)
    prediction_mean = condition_means(prediction, conditions)
    mean_flat_neural = neural_mean.reshape(-1, neural.shape[-1])
    mean_flat_prediction = prediction_mean.reshape(-1, prediction.shape[-1])
    per_neuron_mean = column_correlation(mean_flat_neural, mean_flat_prediction)
    per_sample_mean = row_correlation(mean_flat_neural, mean_flat_prediction)
    rng = np.random.default_rng(seed)
    if len(flat_neural) > cka_max_samples:
        chosen = np.sort(rng.choice(len(flat_neural), cka_max_samples, replace=False))
    else:
        chosen = np.arange(len(flat_neural))
    condition_patterns_neural = neural_mean.mean(axis=1)
    condition_patterns_prediction = prediction_mean.mean(axis=1)
    condition_rdm_neural = correlation_rdm(condition_patterns_neural)
    condition_rdm_prediction = correlation_rdm(condition_patterns_prediction)
    time_rsa_spearman, time_rsa_pearson = [], []
    for time_index in range(neural_mean.shape[1]):
        brain_rdm = correlation_rdm(neural_mean[:, time_index])
        model_rdm = correlation_rdm(prediction_mean[:, time_index])
        time_rsa_spearman.append(compare_rdms(brain_rdm, model_rdm, "spearman"))
        time_rsa_pearson.append(compare_rdms(brain_rdm, model_rdm, "pearson"))
    temporal_rsa_spearman, temporal_rsa_pearson = [], []
    for condition in range(len(neural_mean)):
        brain_rdm = correlation_rdm(neural_mean[condition, ::temporal_rsa_stride])
        model_rdm = correlation_rdm(prediction_mean[condition, ::temporal_rsa_stride])
        temporal_rsa_spearman.append(compare_rdms(brain_rdm, model_rdm, "spearman"))
        temporal_rsa_pearson.append(compare_rdms(brain_rdm, model_rdm, "pearson"))
    state_brain = neural_mean[:, ::temporal_rsa_stride].reshape(-1, neural.shape[-1])
    state_model = prediction_mean[:, ::temporal_rsa_stride].reshape(-1, neural.shape[-1])
    delta_brain = np.diff(neural_mean, axis=1)
    delta_model = np.diff(prediction_mean, axis=1)
    delta_per_neuron = column_correlation(
        delta_brain.reshape(-1, neural.shape[-1]), delta_model.reshape(-1, neural.shape[-1])
    )
    ranked_brain = rankdata(mean_flat_neural, axis=1)
    ranked_model = rankdata(mean_flat_prediction, axis=1)
    spearman_population = row_correlation(ranked_brain, ranked_model)
    best_lag, best_lag_frames = _best_lag_correlation(neural_mean, prediction_mean)
    brain_variance = float(np.mean((neural - neural.mean()) ** 2))
    brain_temporal = float(np.mean(np.std(neural_mean, axis=1)))
    prediction_temporal = float(np.mean(np.std(prediction_mean, axis=1)))
    metrics = {
        "response_single_trial_neuron_mean_r": float(np.nanmean(per_neuron)),
        "response_single_trial_neuron_median_r": float(np.nanmedian(per_neuron)),
        "response_condition_average_neuron_mean_r": float(np.nanmean(per_neuron_mean)),
        "response_condition_average_neuron_median_r": float(np.nanmedian(per_neuron_mean)),
        "population_vector_single_trial_mean_r": float(np.nanmean(per_sample)),
        "population_vector_single_trial_median_r": float(np.nanmedian(per_sample)),
        "population_vector_condition_average_mean_r": float(np.nanmean(per_sample_mean)),
        "population_vector_condition_average_spearman_mean_r": float(np.nanmean(spearman_population)),
        "temporal_difference_neuron_mean_r": float(np.nanmean(delta_per_neuron)),
        "pooled_zero_lag_r": scalar_correlation(neural_mean, prediction_mean),
        "pooled_best_lag_r": best_lag,
        "pooled_best_lag_frames": float(best_lag_frames),
        "normalized_mse": float(np.mean((neural - prediction) ** 2) / max(brain_variance, 1e-12)),
        "pooled_explained_variance": float(1.0 - np.mean((neural - prediction) ** 2) / max(brain_variance, 1e-12)),
        "temporal_std_ratio": float(prediction_temporal / max(brain_temporal, 1e-12)),
        "cka_single_trial_time_aligned": linear_cka(flat_neural[chosen], flat_prediction[chosen]),
        "cka_condition_average_time_aligned": linear_cka(mean_flat_neural, mean_flat_prediction),
        "cka_temporal_difference": linear_cka(
            delta_brain.reshape(-1, neural.shape[-1]), delta_model.reshape(-1, neural.shape[-1])
        ),
        "cka_condition_pattern": linear_cka(condition_patterns_neural, condition_patterns_prediction),
        "rsa_condition_spearman": compare_rdms(condition_rdm_neural, condition_rdm_prediction, "spearman"),
        "rsa_condition_pearson": compare_rdms(condition_rdm_neural, condition_rdm_prediction, "pearson"),
        "rsa_time_resolved_spearman_mean": float(np.mean(time_rsa_spearman)),
        "rsa_time_resolved_pearson_mean": float(np.mean(time_rsa_pearson)),
        "rsa_within_condition_temporal_spearman_mean": float(np.mean(temporal_rsa_spearman)),
        "rsa_within_condition_temporal_pearson_mean": float(np.mean(temporal_rsa_pearson)),
        "rsa_condition_time_state_spearman": compare_rdms(
            correlation_rdm(state_brain), correlation_rdm(state_model), "spearman"
        ),
        "rsa_condition_time_state_pearson": compare_rdms(
            correlation_rdm(state_brain), correlation_rdm(state_model), "pearson"
        ),
    }
    return metrics, per_neuron, per_sample


def metric_family(metric: str) -> str:
    if metric.startswith("response") or metric.startswith("normalized") or metric.startswith("pooled_explained"):
        return "response"
    if metric.startswith("population"):
        return "population_vector"
    if metric.startswith("cka"):
        return "CKA"
    if metric.startswith("rsa"):
        return "RSA"
    return "temporal"


def paired_bootstrap(
    static_neuron: np.ndarray,
    dynamic_neuron: np.ndarray,
    static_sample: np.ndarray,
    dynamic_sample: np.ndarray,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for metric, left, right, unit in (
        ("response_single_trial_neuron_mean_r", static_neuron, dynamic_neuron, "neuron"),
        ("population_vector_single_trial_mean_r", static_sample, dynamic_sample, "trial_time"),
    ):
        valid = np.isfinite(left) & np.isfinite(right)
        delta = right[valid] - left[valid]
        draws = np.empty(samples)
        for index in range(samples):
            draws[index] = np.mean(rng.choice(delta, size=len(delta), replace=True))
        rows.append(
            {
                "metric": metric,
                "comparison": "dynamic_minus_static",
                "resampling_unit": unit,
                "delta": float(np.mean(delta)),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "probability_dynamic_better": float(np.mean(draws > 0)),
                "bootstrap_samples": int(samples),
            }
        )
    return pd.DataFrame(rows)

