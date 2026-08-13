from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    compute_train_neuron_precision,
    deterministic_neuron_order,
    discover_oracle_conditions,
    load_oracle_conditions,
    load_response_trials,
    load_session_metadata,
)
from .gpfa import GaussianProcessFactorAnalysis
from .pipeline import load_config, resolve_from_project
from .reliability import ReliabilityConfig, run_split_half_reliability, summarize_reliability


PRIMARY_METRICS = (
    "position_correlation",
    "normalized_position_rmse",
    "velocity_direction_cosine",
    "speed_profile_correlation",
    "path_length_similarity",
    "acceleration_direction_cosine",
)

SATURATION_NULLS = (
    "condition_shuffle",
    "circular_shift",
    "time_reversal",
    "independent_neuron_shift",
    "block_shuffle_16",
)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def run_saturation(config_path: str | Path) -> dict[str, Any]:
    config, project_root = load_config(config_path)
    data = config["data"]
    gpfa_config = config["gpfa"]
    sat = config["saturation"]
    seed = int(config["project"]["seed"])
    output_dir = resolve_from_project(project_root, config["project"]["output_dir"]) / "saturation"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = resolve_from_project(project_root, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    neuron_counts = [int(value) for value in sat["neurons"]]
    max_neurons = max(neuron_counts)
    neuron_indices = deterministic_neuron_order(metadata, seed)[:max_neurons]
    train_indices = metadata.indices("train").copy()
    np.random.default_rng(seed).shuffle(train_indices)
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    full_time = frame_stop - frame_start
    unit_precision = np.ones(max_neurons, dtype=np.float64)
    full_train = load_response_trials(
        metadata,
        train_indices,
        neuron_indices,
        frame_start,
        frame_stop,
        unit_precision,
    )
    conditions = discover_oracle_conditions(metadata, float(data["oracle_stimulus_similarity"]))
    full_oracle = load_oracle_conditions(
        metadata,
        conditions,
        neuron_indices,
        frame_start,
        frame_stop,
        unit_precision,
    )
    primary_neurons = int(gpfa_config["primary_neurons"])
    primary_dim = 4
    primary_initialization = 0.25
    max_iterations = int(gpfa_config["max_em_iterations"])
    tolerance = float(gpfa_config["tolerance"])
    bounds = tuple(float(value) for value in gpfa_config["lengthscale_bounds_seconds"])
    profile_splits = 100
    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    primary_model_path = resolve_from_project(project_root, config["project"]["output_dir"]) / "gpfa.pkl"
    primary_model = GaussianProcessFactorAnalysis.load(primary_model_path)

    def fit_and_measure(
        axis: str,
        value: float | int | str,
        neurons: int = primary_neurons,
        latent_dim: int = primary_dim,
        train_fraction: float = 1.0,
        run_seed: int = seed,
        observation_step: int = 4,
        splits: int = profile_splits,
        reuse_primary: bool = False,
    ) -> None:
        observation_indices = np.arange(0, full_time, observation_step)
        times = observation_indices / float(data["frame_rate_hz"])
        trial_count = max(8, int(round(len(full_train) * train_fraction)))
        raw_train = full_train[:trial_count, :, :neurons]
        flattened = raw_train.reshape(-1, neurons)
        std = flattened.std(axis=0, ddof=1)
        threshold = 0.01 * float(np.mean(std))
        precision = 1.0 / np.where(std > threshold, std, threshold)
        train = raw_train[:, observation_indices] * precision
        if reuse_primary:
            model = primary_model
        else:
            model = GaussianProcessFactorAnalysis(
                latent_dim=latent_dim,
                times_seconds=times,
                max_em_iterations=max_iterations,
                tolerance=tolerance,
                initial_lengthscale_seconds=primary_initialization,
                lengthscale_bounds_seconds=bounds,
                random_seed=run_seed,
            ).fit(train, split_name="train")
        oracle = [trial[:, :, :neurons] * precision for trial in full_oracle]
        observed, nulls = run_split_half_reliability(
            oracle,
            model,
            ReliabilityConfig(
                frame_rate_hz=float(data["frame_rate_hz"]),
                observation_step=observation_step,
                n_splits=splits,
                seed=run_seed,
                block_sizes=(16,),
                null_names=SATURATION_NULLS,
            ),
        )
        summary = summarize_reliability(observed, nulls)
        observed_summary = observed.groupby("metric")["value"].agg(
            mean="mean",
            ci_low=lambda x: np.quantile(x, 0.025),
            ci_high=lambda x: np.quantile(x, 0.975),
        )
        for metric in PRIMARY_METRICS:
            item = observed_summary.loc[metric]
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "metric": metric,
                    "observed_mean": float(item["mean"]),
                    "observed_ci_low": float(item["ci_low"]),
                    "observed_ci_high": float(item["ci_high"]),
                    "neurons": neurons,
                    "latent_dim": latent_dim,
                    "train_fraction": train_fraction,
                    "seed": run_seed,
                    "observation_step": observation_step,
                    "splits": splits,
                }
            )
        summary.insert(0, "axis", axis)
        summary.insert(1, "axis_value", value)
        summary.to_csv(output_dir / f"nulls_{axis}_{value}.csv", index=False)
        run_records.append(
            {
                "axis": axis,
                "value": value,
                "neurons": neurons,
                "latent_dim": latent_dim,
                "train_trials": trial_count,
                "seed": run_seed,
                "observation_step": observation_step,
                "effective_observations": len(observation_indices),
                "splits": splits,
                "em_iterations": len(model.history),
                "lengthscales": {key: item.tolist() for key, item in (model.lengthscales or {}).items()},
                "parameter_digest": model.parameter_digest(),
            }
        )

    # Baseline estimate and split-count convergence use the already frozen primary model.
    fit_and_measure("split_count", 500, splits=500, reuse_primary=True)
    for count in neuron_counts:
        fit_and_measure("neuron_count", count, neurons=count, reuse_primary=count == primary_neurons)
    for dimension in [int(value) for value in sat["latent_dimensions"]]:
        fit_and_measure("latent_dim", dimension, latent_dim=dimension, reuse_primary=dimension == primary_dim)
    for fraction in [float(value) for value in sat["train_fractions"]]:
        fit_and_measure("train_fraction", fraction, train_fraction=fraction, reuse_primary=fraction == 1.0)
    for run_seed in [int(value) for value in sat["seeds"]]:
        fit_and_measure("random_seed", run_seed, run_seed=run_seed, reuse_primary=run_seed == seed)
    fit_and_measure("observation_step", 4, observation_step=4, reuse_primary=True)
    fit_and_measure("observation_step", 1, observation_step=1)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "saturation_metrics.csv", index=False)
    (output_dir / "saturation_runs.json").write_text(_json(run_records), encoding="utf-8")
    result = {
        "status": "saturation_complete",
        "session": metadata.session_id,
        "profiles": int(len(run_records)),
        "primary_metrics": list(PRIMARY_METRICS),
        "nulls": list(SATURATION_NULLS),
        "output": str(output_dir),
    }
    (output_dir / "saturation_summary.json").write_text(_json(result), encoding="utf-8")
    return result


def run_split_count_saturation(config_path: str | Path) -> dict[str, Any]:
    """Run one 500-split sample and quantify convergence of the estimated mean."""
    config, project_root = load_config(config_path)
    data = config["data"]
    seed = int(config["project"]["seed"])
    output_dir = resolve_from_project(project_root, config["project"]["output_dir"]) / "saturation"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = resolve_from_project(project_root, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    neurons = int(config["gpfa"]["primary_neurons"])
    neuron_indices = deterministic_neuron_order(metadata, seed)[:neurons]
    conditions = discover_oracle_conditions(metadata, float(data["oracle_stimulus_similarity"]))
    preprocessing = np.load(
        resolve_from_project(project_root, config["project"]["output_dir"]) / "preprocessing.npz"
    )
    precision = preprocessing["raw_to_gpfa_precision"]
    oracle = load_oracle_conditions(
        metadata,
        conditions,
        neuron_indices,
        int(data["frame_start"]),
        int(data["frame_stop"]),
        precision,
    )
    model = GaussianProcessFactorAnalysis.load(
        resolve_from_project(project_root, config["project"]["output_dir"]) / "gpfa.pkl"
    )
    observed, nulls = run_split_half_reliability(
        oracle,
        model,
        ReliabilityConfig(
            frame_rate_hz=float(data["frame_rate_hz"]),
            observation_step=int(data["effective_observation_step"]),
            n_splits=500,
            seed=seed,
            block_sizes=(16,),
            null_names=SATURATION_NULLS,
        ),
    )
    observed.to_csv(output_dir / "split_count_observed_500.csv", index=False)
    nulls.to_csv(output_dir / "split_count_nulls_500.csv", index=False)
    rows = []
    for count in [int(value) for value in config["saturation"]["split_counts"]]:
        for metric in PRIMARY_METRICS:
            values = observed[(observed.metric == metric) & (observed.split < count)].value.to_numpy()
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
            rows.append(
                {
                    "split_count": count,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "standard_error": standard_error,
                    "mean_ci_low": float(values.mean() - 1.96 * standard_error),
                    "mean_ci_high": float(values.mean() + 1.96 * standard_error),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "split_count_convergence.csv", index=False)
    result = {
        "status": "split_count_saturation_complete",
        "maximum_splits": 500,
        "counts": [int(value) for value in config["saturation"]["split_counts"]],
        "output": str(output_dir / "split_count_convergence.csv"),
    }
    (output_dir / "split_count_summary.json").write_text(_json(result), encoding="utf-8")
    return result
