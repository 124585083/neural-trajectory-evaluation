from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import (
    compute_train_neuron_precision,
    deterministic_neuron_order,
    load_behavior_features,
    load_response_trials,
    load_session_metadata,
    split_train_calibration_indices,
)
from .gpfa import GaussianProcessFactorAnalysis
from .pipeline import load_config, resolve_from_project


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def evaluate_behavior_conditioned_prior(config_path: str | Path) -> dict[str, Any]:
    """Compare one shared temporal prior with behavior-state-specific priors."""
    config, project_root = load_config(config_path)
    data, gpfa_config = config["data"], config["gpfa"]
    seed = int(config["project"]["seed"])
    session_path = resolve_from_project(project_root, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    neurons = int(gpfa_config["primary_neurons"])
    neuron_indices = deterministic_neuron_order(metadata, seed)[:neurons]
    fit_indices, calibration_indices = split_train_calibration_indices(
        metadata, float(data["calibration_fraction"]), seed
    )
    fit_features = load_behavior_features(metadata, fit_indices)
    calibration_features = load_behavior_features(metadata, calibration_indices)
    scaler = StandardScaler().fit(fit_features)
    projector = PCA(n_components=1, random_state=seed).fit(scaler.transform(fit_features))
    fit_score = projector.transform(scaler.transform(fit_features))[:, 0]
    calibration_score = projector.transform(scaler.transform(calibration_features))[:, 0]
    thresholds = np.quantile(fit_score, [1 / 3, 2 / 3])
    fit_labels = np.asarray([f"behavior_{value}" for value in np.digitize(fit_score, thresholds)])
    calibration_labels = np.asarray(
        [f"behavior_{value}" for value in np.digitize(calibration_score, thresholds)]
    )
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    step = int(data["effective_observation_step"])
    observation_indices = np.arange(0, frame_stop - frame_start, step)
    times = observation_indices / float(data["frame_rate_hz"])
    precision = compute_train_neuron_precision(
        metadata, fit_indices, neuron_indices, frame_start, frame_stop
    )
    fit_trials = load_response_trials(
        metadata, fit_indices, neuron_indices, frame_start, frame_stop, precision
    )[
        :, observation_indices
    ]
    calibration_trials = load_response_trials(
        metadata, calibration_indices, neuron_indices, frame_start, frame_stop, precision
    )[:, observation_indices]
    common = dict(
        latent_dim=4,
        times_seconds=times,
        max_em_iterations=int(gpfa_config["max_em_iterations"]),
        tolerance=float(gpfa_config["tolerance"]),
        initial_lengthscale_seconds=0.25,
        lengthscale_bounds_seconds=tuple(
            float(value) for value in gpfa_config["lengthscale_bounds_seconds"]
        ),
        random_seed=seed,
    )
    shared = GaussianProcessFactorAnalysis(**common).fit(fit_trials, split_name="train")
    conditional = GaussianProcessFactorAnalysis(**common).fit(
        fit_trials, fit_labels, split_name="train"
    )
    denominator = calibration_trials.shape[1] * calibration_trials.shape[2]
    shared_nll = -shared.score_samples(calibration_trials) / denominator
    conditional_nll = -conditional.score_samples(calibration_trials, calibration_labels) / denominator
    improvement = shared_nll - conditional_nll
    mean_improvement = float(improvement.mean())
    standard_error = float(improvement.std(ddof=1) / np.sqrt(len(improvement)))
    conditional_standard_error = float(conditional_nll.std(ddof=1) / np.sqrt(len(conditional_nll)))
    # One-standard-error rule for nested models: retain the simpler shared
    # prior unless it is worse than the conditional model by more than the
    # uncertainty of the best model's held-out score.
    supported = bool(shared_nll.mean() > conditional_nll.mean() + conditional_standard_error)
    result = {
        "session": metadata.session_id,
        "classification_source": "balanced tertiles of train-fit behavior-covariate PC1; neural responses excluded",
        "classes": 3,
        "fit_class_counts": {
            str(label): int(np.sum(fit_labels == label)) for label in np.unique(fit_labels)
        },
        "calibration_class_counts": {
            str(label): int(np.sum(calibration_labels == label))
            for label in np.unique(calibration_labels)
        },
        "shared_nll_per_observation": float(shared_nll.mean()),
        "conditional_nll_per_observation": float(conditional_nll.mean()),
        "paired_improvement": mean_improvement,
        "paired_standard_error": standard_error,
        "conditional_score_standard_error": conditional_standard_error,
        "one_standard_error_threshold": float(conditional_nll.mean() + conditional_standard_error),
        "paired_95_ci": [
            mean_improvement - 1.96 * standard_error,
            mean_improvement + 1.96 * standard_error,
        ],
        "conditioned_prior_supported": supported,
        "selected_prior": "behavior_conditioned" if supported else "shared_natural_video",
        "shared_lengthscales": {key: value.tolist() for key, value in (shared.lengthscales or {}).items()},
        "conditional_lengthscales": {
            key: value.tolist() for key, value in (conditional.lengthscales or {}).items()
        },
        "coordinate_constraint": "C, d, and R shared across behavior classes within the conditional model",
    }
    output = resolve_from_project(project_root, config["project"]["output_dir"])
    (output / "behavior_conditioned_prior.json").write_text(_json(result), encoding="utf-8")
    return result
