from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


METRIC_DIRECTIONS = {
    "position_correlation": "higher",
    "position_cosine": "higher",
    "normalized_position_rmse": "lower",
    "velocity_direction_cosine": "higher",
    "speed_profile_correlation": "higher",
    "path_length_similarity": "higher",
    "acceleration_direction_cosine": "higher",
    "zero_lag_correlation": "higher",
    "best_lag_correlation": "higher",
}


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x, y = left.ravel(), right.ravel()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(pearsonr(x, y).statistic)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def trajectory_metrics(left: np.ndarray, right: np.ndarray, dt_seconds: float) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("trajectories must be condition x time x latent with matching shape")
    pooled_center = np.mean(np.concatenate([left, right], axis=0), axis=(0, 1), keepdims=True)
    left_centered = left - pooled_center
    right_centered = right - pooled_center
    scale = np.sqrt(np.mean(left_centered**2))
    rmse = np.sqrt(np.mean((left - right) ** 2)) / max(float(scale), 1e-12)
    left_velocity = np.diff(left, axis=1) / dt_seconds
    right_velocity = np.diff(right, axis=1) / dt_seconds
    left_speed = np.linalg.norm(left_velocity, axis=-1)
    right_speed = np.linalg.norm(right_velocity, axis=-1)
    left_path = np.sum(np.linalg.norm(np.diff(left, axis=1), axis=-1), axis=1)
    right_path = np.sum(np.linalg.norm(np.diff(right, axis=1), axis=-1), axis=1)
    ratio = np.mean(right_path / np.clip(left_path, 1e-12, None))
    left_acceleration = np.diff(left_velocity, axis=1) / dt_seconds
    right_acceleration = np.diff(right_velocity, axis=1) / dt_seconds
    lag_scores = []
    max_lag = min(15, left.shape[1] // 4)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            lag_scores.append(_correlation(left[:, -lag:], right[:, :lag]))
        elif lag > 0:
            lag_scores.append(_correlation(left[:, :-lag], right[:, lag:]))
        else:
            lag_scores.append(_correlation(left, right))
    return {
        "position_correlation": _correlation(left_centered, right_centered),
        "position_cosine": float(np.mean(_cosine(left_centered, right_centered))),
        "normalized_position_rmse": float(rmse),
        "velocity_direction_cosine": float(np.mean(_cosine(left_velocity, right_velocity))),
        "speed_profile_correlation": _correlation(left_speed, right_speed),
        "path_length_similarity": float(np.exp(-abs(np.log(max(float(ratio), 1e-12))))),
        "acceleration_direction_cosine": float(np.mean(_cosine(left_acceleration, right_acceleration))),
        "zero_lag_correlation": lag_scores[max_lag],
        "best_lag_correlation": float(max(lag_scores)),
    }

