from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump, output_dir, resolve
from .traditional import evaluate_traditional, metric_family, paired_bootstrap


def _load_predictions(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = output_dir(config) / "oracle_predictions.npz"
    if not path.exists():
        raise FileNotFoundError("oracle_predictions.npz is missing; run predictions first")
    with np.load(path, allow_pickle=True) as values:
        return {key: values[key] for key in values.files}


def evaluate_traditional_models(config: dict[str, Any]) -> dict[str, Any]:
    values = _load_predictions(config)
    setting = config["traditional"]
    rows, details = [], {}
    for model_name in ("static", "dynamic"):
        metrics, per_neuron, per_sample = evaluate_traditional(
            values["neural"],
            values[model_name],
            values["conditions"],
            cka_max_samples=int(setting["cka_max_single_trial_samples"]),
            temporal_rsa_stride=int(setting["temporal_rsa_stride"]),
            seed=int(config["project"]["seed"]),
        )
        details[model_name] = (per_neuron, per_sample)
        rows.extend(
            {
                "model": model_name,
                "family": metric_family(metric),
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir(config) / "traditional_metrics.csv", index=False)
    np.savez_compressed(
        output_dir(config) / "traditional_distributions.npz",
        static_per_neuron=details["static"][0],
        dynamic_per_neuron=details["dynamic"][0],
        static_per_trial_time=details["static"][1],
        dynamic_per_trial_time=details["dynamic"][1],
    )
    bootstrap = paired_bootstrap(
        details["static"][0],
        details["dynamic"][0],
        details["static"][1],
        details["dynamic"][1],
        int(setting["paired_bootstrap_samples"]),
        int(config["project"]["seed"]),
    )
    bootstrap.to_csv(output_dir(config) / "traditional_paired_bootstrap.csv", index=False)
    result = {
        "status": "traditional_evaluation_complete",
        "models": ["static", "dynamic"],
        "metric_count_per_model": int(len(table) // 2),
        "families": sorted(table.family.unique().tolist()),
        "primary_response": table[table.metric == "response_single_trial_neuron_mean_r"].set_index("model")["value"].to_dict(),
        "primary_cka": table[table.metric == "cka_condition_average_time_aligned"].set_index("model")["value"].to_dict(),
        "primary_rsa": table[table.metric == "rsa_condition_time_state_spearman"].set_index("model")["value"].to_dict(),
    }
    json_dump(output_dir(config) / "traditional_summary.json", result)
    return result


def _condition_means(values: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    return np.stack([values[conditions == c].mean(axis=0) for c in np.unique(conditions)])


def _derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(size)
    for _ in range(100):
        candidate = rng.permutation(size)
        if np.all(candidate != base):
            return candidate
    return np.roll(base, 1)


def _model_null(values: np.ndarray, name: str, rng: np.random.Generator, block_size: int) -> np.ndarray:
    result = values.copy()
    conditions, time, neurons = result.shape
    if name == "condition_shuffle":
        return result[_derangement(conditions, rng)]
    if name == "circular_shift":
        for condition in range(conditions):
            result[condition] = np.roll(result[condition], int(rng.integers(time // 8, time - time // 8)), axis=0)
        return result
    if name == "time_reversal":
        return result[:, ::-1].copy()
    if name == "block_shuffle":
        for condition in range(conditions):
            blocks = [result[condition, start : start + block_size] for start in range(0, time, block_size)]
            order = rng.permutation(len(blocks))
            if np.array_equal(order, np.arange(len(blocks))):
                order = np.roll(order, 1)
            result[condition] = np.concatenate([blocks[index] for index in order], axis=0)
        return result
    if name == "independent_neuron_shift":
        for condition in range(conditions):
            for neuron, shift in enumerate(rng.integers(1, time, size=neurons)):
                result[condition, :, neuron] = np.roll(values[condition, :, neuron], int(shift))
        return result
    raise KeyError(name)


def evaluate_gpfa_models(config: dict[str, Any]) -> dict[str, Any]:
    source = resolve(config, config["references"]["phase2_root"]) / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    from trajectory_reliability.metrics import METRIC_DIRECTIONS, trajectory_metrics

    out = output_dir(config)
    values = _load_predictions(config)
    model = GaussianProcessFactorAnalysis.load(out / "gpfa.pkl")
    with np.load(out / "gpfa_preprocessing.npz", allow_pickle=True) as prep:
        train_precision = prep["raw_to_gpfa_precision"]
        all_precision = prep["official_all_precision"]
        observation_indices = prep["observation_indices"].astype(np.int64)
        frame_start = int(prep["frame_start"])
        frame_stop = int(prep["frame_stop"])
    if all_precision.ndim == 1:
        factor = train_precision / all_precision
    elif all_precision.ndim == 2:
        official_interval = all_precision[:, frame_start:frame_stop]
        if official_interval.shape != (len(train_precision), values["neural"].shape[1]):
            raise AssertionError("official time-resolved response scale is not aligned to the evaluation interval")
        factor = train_precision[None, :] / official_interval.T
    else:
        raise ValueError(f"unexpected official response precision shape {all_precision.shape}")
    neural = values["neural"].astype(np.float64) * factor
    predictions = {
        name: values[name].astype(np.float64) * factor for name in ("static", "dynamic")
    }
    conditions = values["conditions"]
    neural_mean = _condition_means(neural, conditions)
    model_means = {name: _condition_means(prediction, conditions) for name, prediction in predictions.items()}
    query_times = np.arange(neural.shape[1], dtype=np.float64) / float(config["data"]["frame_rate_hz"])

    def latent(response: np.ndarray) -> np.ndarray:
        return model.transform_query(response[:, observation_indices], query_times)

    neural_latent = latent(neural_mean)
    model_latents = {name: latent(prediction) for name, prediction in model_means.items()}
    rows, observed_by_model = [], {}
    for name, prediction in model_means.items():
        prediction_latent = model_latents[name]
        metrics = trajectory_metrics(neural_latent, prediction_latent, 1.0 / float(config["data"]["frame_rate_hz"]))
        observed_by_model[name] = metrics
        rows.extend({"model": name, "metric": metric, "value": value} for metric, value in metrics.items())
        np.savez_compressed(out / f"gpfa_trajectories_{name}.npz", neural=neural_latent, prediction=prediction_latent)
    observed = pd.DataFrame(rows)
    observed.to_csv(out / "gpfa_model_metrics.csv", index=False)

    permutations = int(config["model_nulls"]["permutations"])
    block_size = int(config["model_nulls"]["block_size_frames"])
    rng = np.random.default_rng(int(config["project"]["seed"]))
    null_rows = []
    null_names = ("condition_shuffle", "circular_shift", "time_reversal", "block_shuffle", "independent_neuron_shift")
    for name, prediction in model_means.items():
        for null_name in null_names:
            for permutation in range(permutations):
                null_latent = latent(_model_null(prediction, null_name, rng, block_size))
                metrics = trajectory_metrics(neural_latent, null_latent, 1.0 / float(config["data"]["frame_rate_hz"]))
                null_rows.extend(
                    {"model": name, "null": null_name, "permutation": permutation, "metric": metric, "value": value}
                    for metric, value in metrics.items()
                )
    null_table = pd.DataFrame(null_rows)
    null_table.to_csv(out / "gpfa_model_nulls.csv", index=False)
    summary_rows = []
    for (name, metric, null_name), group in null_table.groupby(["model", "metric", "null"]):
        observed_value = observed_by_model[name][metric]
        null_values = group.value.to_numpy()
        if METRIC_DIRECTIONS[metric] == "higher":
            extreme = null_values >= observed_value
        else:
            extreme = null_values <= observed_value
        summary_rows.append(
            {
                "model": name,
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "null": null_name,
                "observed": observed_value,
                "null_mean": float(np.mean(null_values)),
                "null_ci_low": float(np.quantile(null_values, 0.025)),
                "null_ci_high": float(np.quantile(null_values, 0.975)),
                "null_exceedances": int(np.sum(extreme)),
                "null_exceedance_rate": float(np.mean(extreme)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / "gpfa_model_null_summary.csv", index=False)

    bootstrap_rows = []
    bootstrap_samples = int(config["traditional"]["paired_bootstrap_samples"])
    for bootstrap in range(bootstrap_samples):
        selected = rng.integers(0, len(neural_latent), size=len(neural_latent))
        static_metrics = trajectory_metrics(
            neural_latent[selected], model_latents["static"][selected], 1.0 / float(config["data"]["frame_rate_hz"])
        )
        dynamic_metrics = trajectory_metrics(
            neural_latent[selected], model_latents["dynamic"][selected], 1.0 / float(config["data"]["frame_rate_hz"])
        )
        for metric in static_metrics:
            raw_delta = dynamic_metrics[metric] - static_metrics[metric]
            oriented = raw_delta if METRIC_DIRECTIONS[metric] == "higher" else -raw_delta
            bootstrap_rows.append({"bootstrap": bootstrap, "metric": metric, "raw_dynamic_minus_static": raw_delta, "oriented_dynamic_advantage": oriented})
    boot = pd.DataFrame(bootstrap_rows)
    boot.to_csv(out / "gpfa_model_paired_bootstrap_raw.csv", index=False)
    boot.groupby("metric").agg(
        mean_oriented_advantage=("oriented_dynamic_advantage", "mean"),
        ci_low=("oriented_dynamic_advantage", lambda x: np.quantile(x, 0.025)),
        ci_high=("oriented_dynamic_advantage", lambda x: np.quantile(x, 0.975)),
        probability_dynamic_better=("oriented_dynamic_advantage", lambda x: np.mean(x > 0)),
    ).reset_index().to_csv(out / "gpfa_model_paired_bootstrap.csv", index=False)
    result = {
        "status": "frozen_gpfa_model_evaluation_complete",
        "latent_dim": int(model.latent_dim),
        "models": observed.pivot(index="metric", columns="model", values="value").to_dict(),
        "null_permutations": permutations,
        "paired_condition_bootstrap_samples": bootstrap_samples,
        "alignment": "none; both predictions use the frozen brain-defined GPFA directly",
    }
    json_dump(out / "gpfa_model_summary.json", result)
    return result
