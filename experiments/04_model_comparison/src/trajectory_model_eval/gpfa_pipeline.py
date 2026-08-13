from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump, output_dir, resolve
from .protocol import load_protocol


def _imports(config: dict[str, Any]) -> None:
    source = resolve(config, config["references"]["phase2_root"]) / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def fit_gpfa_and_reliability(config: dict[str, Any]) -> dict[str, Any]:
    _imports(config)
    from trajectory_reliability.data import (
        compute_train_neuron_precision,
        discover_oracle_conditions,
        load_oracle_conditions,
        load_response_trials,
        load_session_metadata,
    )
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    from trajectory_reliability.reliability import (
        ReliabilityConfig,
        run_split_half_reliability,
        summarize_reliability,
    )
    from trajectory_reliability.selection import select_gpfa

    out = output_dir(config)
    lock = load_protocol(config)
    data = config["data"]
    gpfa = config["gpfa"]
    session_path = resolve(config, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    neurons = lock["neuron_indices"].astype(np.int64)
    fit_indices = lock["fit_indices"].astype(np.int64)
    calibration_indices = lock["calibration_indices"].astype(np.int64)
    selected = lock["selected_train_indices"].astype(np.int64)
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    full_time = frame_stop - frame_start
    step = int(gpfa["effective_observation_step"])
    observation_indices = np.arange(0, full_time, step)
    times = observation_indices / float(data["frame_rate_hz"])

    selection_precision = compute_train_neuron_precision(
        metadata, fit_indices, neurons, frame_start, frame_stop
    )
    fit_trials = load_response_trials(
        metadata, fit_indices, neurons, frame_start, frame_stop, selection_precision
    )[:, observation_indices]
    calibration_trials = load_response_trials(
        metadata, calibration_indices, neurons, frame_start, frame_stop, selection_precision
    )[:, observation_indices]
    selection = select_gpfa(
        fit_trials,
        calibration_trials,
        times,
        [int(gpfa["latent_dim"])],
        [float(value) for value in gpfa["initial_lengthscales_seconds"]],
        int(gpfa["selection_em_iterations"]),
        float(gpfa["tolerance"]),
        tuple(float(value) for value in gpfa["lengthscale_bounds_seconds"]),
        int(config["project"]["seed"]),
    )
    selection.table.to_csv(out / "gpfa_model_selection.csv", index=False)

    final_precision = compute_train_neuron_precision(
        metadata, selected, neurons, frame_start, frame_stop
    )
    selected_trials = load_response_trials(
        metadata, selected, neurons, frame_start, frame_stop, final_precision
    )[:, observation_indices]
    model = GaussianProcessFactorAnalysis(
        latent_dim=int(gpfa["latent_dim"]),
        times_seconds=times,
        max_em_iterations=int(gpfa["final_em_iterations"]),
        tolerance=float(gpfa["tolerance"]),
        initial_lengthscale_seconds=float(selection.initial_lengthscale_seconds),
        lengthscale_bounds_seconds=tuple(
            float(value) for value in gpfa["lengthscale_bounds_seconds"]
        ),
        random_seed=int(config["project"]["seed"]),
    ).fit(selected_trials, split_name="train")
    model.save(out / "gpfa.pkl")
    np.savez_compressed(
        out / "gpfa_preprocessing.npz",
        neuron_indices=neurons,
        neuron_ids=metadata.unit_ids[neurons],
        raw_to_gpfa_precision=final_precision,
        official_all_precision=metadata.response_precision[neurons],
        frame_start=np.asarray(frame_start),
        frame_stop=np.asarray(frame_stop),
        observation_indices=observation_indices,
        selected_train_indices=selected,
    )

    conditions = discover_oracle_conditions(
        metadata, float(data["oracle_stimulus_similarity"])
    )
    oracle = load_oracle_conditions(
        metadata, conditions, neurons, frame_start, frame_stop, final_precision
    )
    reliability_config = ReliabilityConfig(
        frame_rate_hz=float(data["frame_rate_hz"]),
        observation_step=step,
        n_splits=int(config["reliability"]["splits"]),
        seed=int(config["project"]["seed"]),
        block_sizes=tuple(int(value) for value in config["reliability"]["block_sizes_frames"]),
    )
    observed, nulls = run_split_half_reliability(oracle, model, reliability_config)
    summary = summarize_reliability(observed, nulls)
    observed.to_csv(out / "gpfa_reliability_observed.csv", index=False)
    nulls.to_csv(out / "gpfa_reliability_nulls.csv", index=False)
    summary.to_csv(out / "gpfa_reliability_summary.csv", index=False)
    result = {
        "status": "gpfa_fit_and_full_reliability_complete",
        "session": metadata.session_id,
        "neurons": int(len(neurons)),
        "selected_train_trials": int(len(selected)),
        "fit_trials": int(len(fit_indices)),
        "calibration_trials": int(len(calibration_indices)),
        "latent_dim": int(model.latent_dim),
        "effective_observations": int(len(observation_indices)),
        "selected_initial_lengthscale_seconds": float(selection.initial_lengthscale_seconds),
        "learned_lengthscales_seconds": {
            name: values.tolist() for name, values in (model.lengthscales or {}).items()
        },
        "parameter_digest": model.parameter_digest(),
        "reliability_splits": int(reliability_config.n_splits),
        "oracle_repeats": [int(len(condition.dataset_indices)) for condition in conditions],
    }
    json_dump(out / "gpfa_run_summary.json", result)
    return result
