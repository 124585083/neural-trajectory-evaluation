from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import (
    compute_train_neuron_precision,
    deterministic_neuron_order,
    discover_oracle_conditions,
    load_oracle_conditions,
    load_response_trials,
    load_session_metadata,
    split_train_calibration_indices,
)
from .gpfa import GaussianProcessFactorAnalysis
from .reliability import ReliabilityConfig, run_split_half_reliability, summarize_reliability
from .selection import select_gpfa


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config, config_path.parent.parent


def resolve_from_project(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def inspect_data(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    data = config["data"]
    session_path = resolve_from_project(project_root, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    conditions = discover_oracle_conditions(metadata, float(data["oracle_stimulus_similarity"]))
    result = {
        "session": metadata.session_id,
        "neurons": int(metadata.unit_ids.size),
        "tier_counts": {
            str(name): int(count)
            for name, count in zip(*np.unique(metadata.tiers, return_counts=True))
        },
        "frame_start": int(data["frame_start"]),
        "frame_stop": int(data["frame_stop"]),
        "trajectory_frames": int(data["frame_stop"] - data["frame_start"]),
        "frame_rate_hz": float(data["frame_rate_hz"]),
        "effective_observation_step": int(data["effective_observation_step"]),
        "effective_observations": len(
            range(0, int(data["frame_stop"] - data["frame_start"]), int(data["effective_observation_step"]))
        ),
        "oracle_conditions": [
            {
                "condition_id": condition.condition_id,
                "repeat_count": len(condition.dataset_indices),
                "dataset_indices": list(condition.dataset_indices),
                "trial_ids": list(condition.trial_ids),
            }
            for condition in conditions
        ],
    }
    return result


def run_pipeline(config_path: str | Path, smoke: bool = False) -> dict[str, Any]:
    config, project_root = load_config(config_path)
    seed = int(config["project"]["seed"])
    data = config["data"]
    gpfa_config = config["gpfa"]
    output_dir = resolve_from_project(project_root, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = inspect_data(config, project_root)
    (output_dir / "data_audit.json").write_text(_json(audit), encoding="utf-8")
    session_path = resolve_from_project(project_root, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    neuron_count = 64 if smoke else int(gpfa_config["primary_neurons"])
    neuron_indices = deterministic_neuron_order(metadata, seed)[:neuron_count]
    fit_indices, calibration_indices = split_train_calibration_indices(
        metadata, float(data["calibration_fraction"]), seed
    )
    if smoke:
        fit_indices = fit_indices[:24]
        calibration_indices = calibration_indices[:8]
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    observation_step = int(data["effective_observation_step"])
    full_time = frame_stop - frame_start
    observation_indices = np.arange(0, full_time, observation_step)
    times_seconds = observation_indices / float(data["frame_rate_hz"])
    selection_precision = compute_train_neuron_precision(
        metadata, fit_indices, neuron_indices, frame_start, frame_stop
    )
    fit_trials = load_response_trials(
        metadata,
        fit_indices,
        neuron_indices,
        frame_start,
        frame_stop,
        selection_precision,
    )[
        :, observation_indices
    ]
    calibration_trials = load_response_trials(
        metadata,
        calibration_indices,
        neuron_indices,
        frame_start,
        frame_stop,
        selection_precision,
    )[:, observation_indices]
    dimensions = [2, 4] if smoke else [int(value) for value in gpfa_config["latent_dimensions"]]
    initializations = (
        [0.25, 0.5]
        if smoke
        else [float(value) for value in gpfa_config["initial_lengthscales_seconds"]]
    )
    max_iterations = 2 if smoke else int(gpfa_config["max_em_iterations"])
    selection = select_gpfa(
        fit_trials,
        calibration_trials,
        times_seconds,
        dimensions,
        initializations,
        max_iterations,
        float(gpfa_config["tolerance"]),
        tuple(float(value) for value in gpfa_config["lengthscale_bounds_seconds"]),
        seed,
    )
    selection.table.to_csv(output_dir / "model_selection.csv", index=False)

    # Hyperparameters are frozen after train/calibration selection. Refit on all
    # official train trials; oracle responses remain completely unseen.
    all_train_indices = np.sort(np.concatenate([fit_indices, calibration_indices]))
    final_precision = compute_train_neuron_precision(
        metadata, all_train_indices, neuron_indices, frame_start, frame_stop
    )
    all_train = load_response_trials(
        metadata,
        all_train_indices,
        neuron_indices,
        frame_start,
        frame_stop,
        final_precision,
    )[:, observation_indices]
    model = GaussianProcessFactorAnalysis(
        latent_dim=selection.latent_dim,
        times_seconds=times_seconds,
        max_em_iterations=max_iterations,
        tolerance=float(gpfa_config["tolerance"]),
        initial_lengthscale_seconds=selection.initial_lengthscale_seconds,
        lengthscale_bounds_seconds=tuple(
            float(value) for value in gpfa_config["lengthscale_bounds_seconds"]
        ),
        random_seed=seed,
    ).fit(all_train, split_name="train")
    model.save(output_dir / "gpfa.pkl")
    np.savez_compressed(
        output_dir / "preprocessing.npz",
        neuron_indices=neuron_indices,
        neuron_ids=metadata.unit_ids[neuron_indices],
        raw_to_gpfa_precision=final_precision,
        frame_start=np.asarray(frame_start),
        frame_stop=np.asarray(frame_stop),
        observation_indices=observation_indices,
        normalization_source=np.asarray("official train tier only; one scalar standard deviation per neuron"),
    )

    conditions = discover_oracle_conditions(metadata, float(data["oracle_stimulus_similarity"]))
    oracle = load_oracle_conditions(
        metadata,
        conditions,
        neuron_indices,
        frame_start,
        frame_stop,
        final_precision,
    )
    reliability_config = ReliabilityConfig(
        frame_rate_hz=float(data["frame_rate_hz"]),
        observation_step=observation_step,
        n_splits=5 if smoke else int(config["reliability"]["splits"]),
        seed=seed,
        block_sizes=tuple(int(value) for value in config["reliability"]["block_sizes_frames"]),
    )
    observed, nulls = run_split_half_reliability(oracle, model, reliability_config)
    summary = summarize_reliability(observed, nulls)
    observed.to_csv(output_dir / "split_half_observed.csv", index=False)
    nulls.to_csv(output_dir / "null_distributions.csv", index=False)
    summary.to_csv(output_dir / "reliability_summary.csv", index=False)
    run = {
        "status": "smoke_complete" if smoke else "primary_reliability_complete",
        "session": metadata.session_id,
        "neuron_count": neuron_count,
        "fit_trials": int(len(fit_indices)),
        "calibration_trials": int(len(calibration_indices)),
        "refit_trials": int(len(all_train)),
        "oracle_conditions": len(conditions),
        "oracle_repeats": [len(condition.dataset_indices) for condition in conditions],
        "source_frames": full_time,
        "effective_observations": len(observation_indices),
        "output_latent_frames": full_time,
        "selected_latent_dim": selection.latent_dim,
        "selected_initial_lengthscale_seconds": selection.initial_lengthscale_seconds,
        "learned_lengthscales_seconds": {
            key: values.tolist() for key, values in (model.lengthscales or {}).items()
        },
        "gpfa_parameter_digest": model.parameter_digest(),
        "response_normalization": "train-only global per-neuron standard deviation",
        "reliability_splits": reliability_config.n_splits,
    }
    (output_dir / "run_summary.json").write_text(_json(run), encoding="utf-8")
    return run
