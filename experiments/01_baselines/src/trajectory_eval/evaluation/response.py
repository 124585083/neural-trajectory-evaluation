from __future__ import annotations

import numpy as np


def pearson_per_neuron(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Pearson correlation over observations, independently per neuron."""
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError(f"expected matching [observations, neurons], got {target.shape} and {prediction.shape}")
    target_centered = target - target.mean(axis=0, keepdims=True)
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    numerator = np.sum(target_centered * pred_centered, axis=0)
    denominator = np.sqrt(np.sum(target_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    result[~np.isfinite(result)] = 0
    return result


def _flatten_trials(trials: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(trials, axis=0)


def correlation_summary(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
    stimulus_ids: list[str] | None = None,
) -> dict[str, float | np.ndarray]:
    """Return shared single-trial and optional repeat-averaged correlations.

    Every trial is [T,N]. Repeat averaging groups trials by stimulus_ids before
    concatenation. Variable-length repeats are trimmed to their shared minimum.
    """
    single = pearson_per_neuron(_flatten_trials(targets), _flatten_trials(predictions))
    result: dict[str, float | np.ndarray] = {
        "single_trial_per_neuron": single,
        "single_trial_mean": float(single.mean()),
    }
    if stimulus_ids is None:
        return result
    groups: dict[str, list[int]] = {}
    for index, stimulus_id in enumerate(stimulus_ids):
        groups.setdefault(stimulus_id, []).append(index)
    averaged_targets: list[np.ndarray] = []
    averaged_predictions: list[np.ndarray] = []
    for indices in groups.values():
        common_t = min(targets[i].shape[0] for i in indices)
        averaged_targets.append(np.stack([targets[i][:common_t] for i in indices]).mean(axis=0))
        averaged_predictions.append(np.stack([predictions[i][:common_t] for i in indices]).mean(axis=0))
    trial_average = pearson_per_neuron(_flatten_trials(averaged_targets), _flatten_trials(averaged_predictions))
    result["trial_average_per_neuron"] = trial_average
    result["trial_average_mean"] = float(trial_average.mean())
    result["repeat_group_count"] = len(groups)
    return result

