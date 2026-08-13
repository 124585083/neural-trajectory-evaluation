from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump, output_dir, resolve
from .protocol import load_protocol


PRIMARY_METRICS = (
    "position_correlation",
    "normalized_position_rmse",
    "velocity_direction_cosine",
    "speed_profile_correlation",
    "acceleration_direction_cosine",
)
SENSITIVITY_NULLS = (
    "condition_shuffle",
    "circular_shift",
    "time_reversal",
    "independent_neuron_shift",
    "block_shuffle_16",
)


def run_sensitivity(config: dict[str, Any]) -> dict[str, Any]:
    phase2_source = resolve(config, config["references"]["phase2_root"]) / "src"
    if str(phase2_source) not in sys.path:
        sys.path.insert(0, str(phase2_source))
    from trajectory_reliability.data import (
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

    out = output_dir(config)
    sensitivity_out = out / "sensitivity"
    sensitivity_out.mkdir(parents=True, exist_ok=True)
    lock = load_protocol(config)
    data = config["data"]
    gpfa = config["gpfa"]
    sat = config["sensitivity"]
    seed = int(config["project"]["seed"])
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    full_time = frame_stop - frame_start
    frame_rate = float(data["frame_rate_hz"])
    neurons = lock["neuron_indices"].astype(np.int64)
    selected_train = lock["selected_train_indices"].astype(np.int64).copy()
    np.random.default_rng(seed).shuffle(selected_train)
    metadata = load_session_metadata(resolve(config, data["sensorium_root"]) / data["session"])
    unit_precision = np.ones(len(neurons), dtype=np.float64)
    raw_train = load_response_trials(
        metadata, selected_train, neurons, frame_start, frame_stop, unit_precision
    )
    conditions = discover_oracle_conditions(metadata, float(data["oracle_stimulus_similarity"]))
    raw_oracle = load_oracle_conditions(
        metadata, conditions, neurons, frame_start, frame_stop, unit_precision
    )
    primary_model = GaussianProcessFactorAnalysis.load(out / "gpfa.pkl")
    with np.load(out / "gpfa_preprocessing.npz") as preprocessing:
        primary_precision = preprocessing["raw_to_gpfa_precision"]

    profile_splits = int(sat["profile_splits"])
    rows: list[dict[str, Any]] = []
    null_rows: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []

    def fit_profile(
        axis: str,
        value: int | float,
        *,
        neuron_count: int = len(neurons),
        latent_dim: int = int(gpfa["latent_dim"]),
        train_fraction: float = 1.0,
        run_seed: int = seed,
        observation_step: int = int(gpfa["effective_observation_step"]),
        reuse_primary: bool = False,
    ) -> None:
        trial_count = max(8, int(round(len(raw_train) * train_fraction)))
        profile_train = raw_train[:trial_count, :, :neuron_count]
        flattened = profile_train.reshape(-1, neuron_count)
        std = flattened.std(axis=0, ddof=1)
        threshold = 0.01 * float(np.mean(std))
        precision = 1.0 / np.where(std > threshold, std, threshold)
        observation_indices = np.arange(0, full_time, observation_step)
        times = observation_indices / frame_rate
        if reuse_primary:
            model = primary_model
            precision = primary_precision
        else:
            model = GaussianProcessFactorAnalysis(
                latent_dim=latent_dim,
                times_seconds=times,
                max_em_iterations=int(gpfa["final_em_iterations"]),
                tolerance=float(gpfa["tolerance"]),
                initial_lengthscale_seconds=0.25,
                lengthscale_bounds_seconds=tuple(
                    float(item) for item in gpfa["lengthscale_bounds_seconds"]
                ),
                random_seed=run_seed,
            ).fit(profile_train[:, observation_indices] * precision, split_name="train")
        oracle = [trial[:, :, :neuron_count] * precision for trial in raw_oracle]
        observed, nulls = run_split_half_reliability(
            oracle,
            model,
            ReliabilityConfig(
                frame_rate_hz=frame_rate,
                observation_step=observation_step,
                n_splits=profile_splits,
                seed=run_seed,
                block_sizes=(16,),
                null_names=SENSITIVITY_NULLS,
            ),
        )
        observed_summary = observed.groupby("metric")["value"].agg(
            mean="mean",
            ci_low=lambda values: np.quantile(values, 0.025),
            ci_high=lambda values: np.quantile(values, 0.975),
        )
        for metric in PRIMARY_METRICS:
            item = observed_summary.loc[metric]
            rows.append(
                {
                    "axis": axis,
                    "axis_value": value,
                    "metric": metric,
                    "observed_mean": float(item["mean"]),
                    "observed_ci_low": float(item["ci_low"]),
                    "observed_ci_high": float(item["ci_high"]),
                    "neurons": neuron_count,
                    "latent_dim": latent_dim,
                    "selected_train_fraction": train_fraction,
                    "selected_train_trials": trial_count,
                    "seed": run_seed,
                    "observation_step": observation_step,
                    "splits": profile_splits,
                }
            )
        summary = summarize_reliability(observed, nulls)
        summary.insert(0, "axis", axis)
        summary.insert(1, "axis_value", value)
        null_rows.append(summary)
        run_records.append(
            {
                "axis": axis,
                "axis_value": value,
                "neurons": neuron_count,
                "latent_dim": latent_dim,
                "selected_train_trials": trial_count,
                "seed": run_seed,
                "observation_step": observation_step,
                "effective_observations": int(len(observation_indices)),
                "em_iterations": int(len(model.history)),
                "parameter_digest": model.parameter_digest(),
            }
        )

    primary_neurons = len(neurons)
    primary_dim = int(gpfa["latent_dim"])
    primary_step = int(gpfa["effective_observation_step"])
    for count in [int(item) for item in sat["neuron_counts"]]:
        fit_profile("neuron_count", count, neuron_count=count, reuse_primary=count == primary_neurons)
    for dimension in [int(item) for item in sat["latent_dimensions"]]:
        fit_profile("latent_dim", dimension, latent_dim=dimension, reuse_primary=dimension == primary_dim)
    for fraction in [float(item) for item in sat["selected_train_fractions"]]:
        fit_profile("selected_train_fraction", fraction, train_fraction=fraction, reuse_primary=fraction == 1.0)
    for run_seed in [int(item) for item in sat["seeds"]]:
        fit_profile("random_seed", run_seed, run_seed=run_seed, reuse_primary=run_seed == seed)
    for step in [int(item) for item in sat["observation_steps"]]:
        fit_profile("observation_step", step, observation_step=step, reuse_primary=step == primary_step)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(sensitivity_out / "sensitivity_metrics.csv", index=False)
    pd.concat(null_rows, ignore_index=True).to_csv(
        sensitivity_out / "sensitivity_null_summary.csv", index=False
    )
    json_dump(sensitivity_out / "sensitivity_runs.json", run_records)

    reliability = pd.read_csv(out / "gpfa_reliability_observed.csv")
    convergence_rows = []
    for count in [int(item) for item in sat["split_counts"]]:
        for metric in PRIMARY_METRICS:
            values = reliability[
                (reliability.metric == metric) & (reliability.split < count)
            ].value.to_numpy()
            convergence_rows.append(
                {
                    "split_count": count,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                }
            )
    pd.DataFrame(convergence_rows).to_csv(
        sensitivity_out / "split_count_convergence.csv", index=False
    )
    primary_nulls = pd.concat(null_rows, ignore_index=True)
    gate = primary_nulls[primary_nulls.metric.isin(PRIMARY_METRICS)]
    result = {
        "status": "reliability_sensitivity_complete",
        "profiles": int(len(run_records)),
        "profile_splits": profile_splits,
        "axes": sorted(metrics.axis.unique().tolist()),
        "primary_metrics": list(PRIMARY_METRICS),
        "nulls": list(SENSITIVITY_NULLS),
        "minimum_paired_superiority_across_profiles": float(gate.paired_superiority.min()),
        "maximum_paired_p_value_across_profiles": float(gate.paired_p_value.max()),
    }
    json_dump(sensitivity_out / "sensitivity_summary.json", result)
    return result

