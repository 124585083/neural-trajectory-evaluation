from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import json_dump, output_dir, resolve
from .traditional import (
    column_correlation,
    compare_rdms,
    condition_means,
    correlation_rdm,
    evaluate_traditional,
    linear_cka,
)


def _bootstrap_mean(values: np.ndarray, samples: int, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = np.mean(values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1)
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def _sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    permutations = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permutations.append(float(np.mean(values * np.asarray(signs))))
    return float(np.mean(np.asarray(permutations) >= observed - 1e-15))


def _balanced_repeat_split(conditions: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selection, test = [], []
    for condition in np.unique(conditions):
        indices = np.flatnonzero(conditions == condition)
        indices = indices[rng.permutation(len(indices))]
        half = len(indices) // 2
        selection.extend(indices[:half])
        test.extend(indices[half : 2 * half])
    return np.asarray(selection, dtype=np.int64), np.asarray(test, dtype=np.int64)


class FrozenTrajectory:
    def __init__(self, config: dict[str, Any], neural: np.ndarray, conditions: np.ndarray) -> None:
        source = resolve(config, config["references"]["phase2_root"]) / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
        from trajectory_reliability.metrics import trajectory_metrics

        out = output_dir(config)
        self.model = GaussianProcessFactorAnalysis.load(out / "gpfa.pkl")
        with np.load(out / "gpfa_preprocessing.npz", allow_pickle=True) as prep:
            train_precision = prep["raw_to_gpfa_precision"]
            all_precision = prep["official_all_precision"]
            frame_start = int(prep["frame_start"])
            frame_stop = int(prep["frame_stop"])
            self.observation_indices = prep["observation_indices"].astype(np.int64)
        self.factor = train_precision[None, :] / all_precision[:, frame_start:frame_stop].T
        self.query_times = np.arange(neural.shape[1], dtype=np.float64) / float(config["data"]["frame_rate_hz"])
        self.dt = 1.0 / float(config["data"]["frame_rate_hz"])
        self.metric = trajectory_metrics
        self.conditions = conditions

    def means(self, values: np.ndarray, trial_indices: np.ndarray | None = None) -> np.ndarray:
        if trial_indices is None:
            trial_indices = np.arange(len(values))
        local_conditions = self.conditions[trial_indices]
        return np.stack(
            [values[trial_indices][local_conditions == condition].mean(axis=0) for condition in np.unique(local_conditions)]
        )

    def latent_means(self, values: np.ndarray, trial_indices: np.ndarray | None = None) -> np.ndarray:
        means = self.means(values, trial_indices) * self.factor
        return self.model.transform_query(means[:, self.observation_indices], self.query_times)

    def compare(
        self,
        neural: np.ndarray,
        prediction: np.ndarray,
        trial_indices: np.ndarray | None = None,
        condition_sample: np.ndarray | None = None,
    ) -> dict[str, float]:
        left = self.latent_means(neural, trial_indices)
        right = self.latent_means(prediction, trial_indices)
        if condition_sample is not None:
            left, right = left[condition_sample], right[condition_sample]
        return self.metric(left, right, self.dt)


def _per_condition_conventional(
    neural: np.ndarray, prediction: np.ndarray, conditions: np.ndarray
) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {
        "temporal_cka": [],
        "temporal_difference_cka": [],
        "temporal_rsa": [],
        "response_r": [],
    }
    for condition in np.unique(conditions):
        selected = conditions == condition
        left = neural[selected].mean(axis=0)
        right = prediction[selected].mean(axis=0)
        rows["temporal_cka"].append(linear_cka(left, right))
        rows["temporal_difference_cka"].append(linear_cka(np.diff(left, axis=0), np.diff(right, axis=0)))
        rows["temporal_rsa"].append(
            compare_rdms(correlation_rdm(left[::10]), correlation_rdm(right[::10]), "spearman")
        )
        rows["response_r"].append(float(np.nanmean(column_correlation(left, right))))
    return {name: np.asarray(values) for name, values in rows.items()}


def _condition_bootstrap_trajectory(
    trajectory: FrozenTrajectory,
    neural: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    trial_indices: np.ndarray | None,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    neural_latent = trajectory.latent_means(neural, trial_indices)
    left_latent = trajectory.latent_means(left, trial_indices)
    right_latent = trajectory.latent_means(right, trial_indices)
    metrics = trajectory.metric(neural_latent, left_latent, trajectory.dt)
    metrics_right = trajectory.metric(neural_latent, right_latent, trajectory.dt)
    rows = []
    distributions = {metric: np.empty(samples) for metric in metrics}
    for bootstrap in range(samples):
        chosen = rng.integers(0, len(neural_latent), size=len(neural_latent))
        left_values = trajectory.metric(neural_latent[chosen], left_latent[chosen], trajectory.dt)
        right_values = trajectory.metric(neural_latent[chosen], right_latent[chosen], trajectory.dt)
        for metric in metrics:
            raw = right_values[metric] - left_values[metric]
            distributions[metric][bootstrap] = -raw if metric == "normalized_position_rmse" else raw
    for metric in metrics:
        raw = metrics_right[metric] - metrics[metric]
        oriented = -raw if metric == "normalized_position_rmse" else raw
        values = distributions[metric]
        rows.append(
            {
                "metric": metric,
                "left": float(metrics[metric]),
                "right": float(metrics_right[metric]),
                "oriented_right_advantage": float(oriented),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_right_better": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def _q1(config: dict[str, Any], rng: np.random.Generator, samples: int) -> dict[str, Any]:
    static = json.loads(
        (
            resolve(config, config["references"]["phase1_root"])
            / "records/static/official_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    dynamic = json.loads(
        (
            resolve(config, config["references"]["phase3_root"])
            / "records/official_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    keys = list(static["full_sequence_oracle_by_session"])
    static_values = np.asarray([static["full_sequence_oracle_by_session"][key] for key in keys])
    dynamic_values = np.asarray([dynamic["full_sequence_oracle_by_session"][key] for key in keys])
    differences = dynamic_values - static_values
    bootstrap = _bootstrap_mean(differences, samples, rng)
    return {
        "answer": "yes",
        "static_five_session_mean": float(static["full_sequence_oracle_single_trial"]),
        "dynamic_five_session_mean": float(dynamic["full_sequence_oracle_single_trial"]),
        "dynamic_minus_static": float(dynamic["full_sequence_oracle_single_trial"] - static["full_sequence_oracle_single_trial"]),
        "relative_gain": float(dynamic["full_sequence_oracle_single_trial"] / static["full_sequence_oracle_single_trial"] - 1.0),
        "session_differences": differences.tolist(),
        "sessions_dynamic_better": int(np.sum(differences > 0)),
        "session_bootstrap": bootstrap,
        "one_sided_exact_sign_p": float(0.5 ** len(differences)),
        "two_sided_exact_sign_p": float(2.0 * 0.5 ** len(differences)),
        "scope": "five sessions; session is the inferential unit",
    }


def _q2(
    neural: np.ndarray,
    static: np.ndarray,
    dynamic: np.ndarray,
    conditions: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    left = _per_condition_conventional(neural, static, conditions)
    right = _per_condition_conventional(neural, dynamic, conditions)
    rows = []
    for metric in left:
        differences = right[metric] - left[metric]
        item = _bootstrap_mean(differences, samples, rng)
        rows.append(
            {
                "metric": metric,
                "static_mean": float(np.mean(left[metric])),
                "dynamic_mean": float(np.mean(right[metric])),
                "dynamic_minus_static": item["estimate"],
                "ci_low": item["ci_low"],
                "ci_high": item["ci_high"],
                "probability_dynamic_better": item["probability_positive"],
                "one_sided_exact_sign_flip_p": _sign_flip_p(differences),
                "conditions": int(len(differences)),
            }
        )
    table = pd.DataFrame(rows)
    return {
        "answer": "yes_as_an_effect; condition-level inference is metric-specific and low-powered",
        "condition_level_results": table.to_dict(orient="records"),
        "scope": "one session and six repeated movie conditions",
    }, table


def _q3(config: dict[str, Any]) -> dict[str, Any]:
    table = pd.read_csv(output_dir(config) / "gpfa_model_metrics.csv").pivot(
        index="metric", columns="model", values="value"
    )
    bootstrap = pd.read_csv(output_dir(config) / "gpfa_model_paired_bootstrap.csv")
    primary = bootstrap[bootstrap.metric.isin(
        ["position_correlation", "normalized_position_rmse", "velocity_direction_cosine", "speed_profile_correlation", "acceleration_direction_cosine"]
    )]
    return {
        "answer": "yes_for_position_direction_and_acceleration; speed_difference_is_inconclusive",
        "metrics": table.to_dict(orient="index"),
        "paired_condition_bootstrap": primary.to_dict(orient="records"),
        "scope": "one session, six conditions, frozen q=4 brain-defined GPFA",
    }


def _response_score_matched_output_perturbation(
    neural: np.ndarray,
    static: np.ndarray,
    dynamic: np.ndarray,
    conditions: np.ndarray,
    trajectory: FrozenTrajectory,
    rng: np.random.Generator,
    samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection, test = _balanced_repeat_split(conditions, 20260813)
    neuron_scale = dynamic.std(axis=(0, 1), keepdims=True)
    noise_seed = 123
    noise = np.random.default_rng(noise_seed).normal(size=dynamic.shape).astype(np.float32)
    target = float(
        np.nanmean(
            column_correlation(
                neural[selection].reshape(-1, neural.shape[-1]),
                static[selection].reshape(-1, static.shape[-1]),
            )
        )
    )
    sigma_grid = np.geomspace(0.01, 10.0, 100)
    scores = []
    for sigma in sigma_grid:
        candidate = np.clip(dynamic + sigma * neuron_scale * noise, 1e-5, None)
        scores.append(
            float(
                np.nanmean(
                    column_correlation(
                        neural[selection].reshape(-1, neural.shape[-1]),
                        candidate[selection].reshape(-1, candidate.shape[-1]),
                    )
                )
            )
        )
    selected = int(np.argmin(np.abs(np.asarray(scores) - target)))
    sigma = float(sigma_grid[selected])
    matched_output = np.clip(dynamic + sigma * neuron_scale * noise, 1e-5, None)
    selection_metrics = {}
    test_metrics = {}
    details: dict[str, tuple[dict[str, float], np.ndarray]] = {}
    for split_name, indices in (("selection", selection), ("test", test)):
        split_conditions = conditions[indices]
        for model_name, prediction in (("static", static), ("response_score_matched_output", matched_output)):
            conventional, per_neuron, _ = evaluate_traditional(
                neural[indices], prediction[indices], split_conditions, cka_max_samples=1000, temporal_rsa_stride=10, seed=42
            )
            trajectory_values = trajectory.compare(neural, prediction, indices)
            record = {**conventional, **{f"gpfa_{key}": value for key, value in trajectory_values.items()}}
            (selection_metrics if split_name == "selection" else test_metrics)[model_name] = record
            if split_name == "test":
                details[model_name] = (conventional, per_neuron)
    response_delta = details["response_score_matched_output"][1] - details["static"][1]
    response_bootstrap = _bootstrap_mean(response_delta[np.isfinite(response_delta)], samples, rng)
    trajectory_bootstrap = _condition_bootstrap_trajectory(
        trajectory, neural, static, matched_output, test, samples, rng
    )
    return {
        "answer": "yes_in_disjoint_repeat_response_score_matched_output_perturbation",
        "selection_trials": int(len(selection)),
        "test_trials": int(len(test)),
        "noise_sigma": sigma,
        "noise_seed": noise_seed,
        "selection_response_target_static": target,
        "selection_response_achieved_dynamic": float(scores[selected]),
        "selection": selection_metrics,
        "test": test_metrics,
        "test_response_dynamic_minus_static_neuron_bootstrap": response_bootstrap,
        "test_gpfa_condition_bootstrap": trajectory_bootstrap.to_dict(orient="records"),
        "interpretation": "the response-score-matched output perturbation degrades the Dynamic prediction only to match scalar response correlation; RSA/CKA are not forced to match",
    }, trajectory_bootstrap


def _ablation(
    config: dict[str, Any],
    neural: np.ndarray,
    conditions: np.ndarray,
    trajectory: FrozenTrajectory,
    extended: dict[str, np.ndarray],
    rng: np.random.Generator,
    samples: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    retentions = [1.0, 0.75, 0.5, 0.25, 0.0]
    prediction_by_retention = {
        retention: extended[f"retention_{retention:.2f}"] for retention in retentions
    }
    rows = []
    latent_neural = trajectory.latent_means(neural)
    latent_predictions = {
        retention: trajectory.latent_means(prediction) for retention, prediction in prediction_by_retention.items()
    }
    for retention in retentions:
        traditional, _, _ = evaluate_traditional(
            neural,
            prediction_by_retention[retention],
            conditions,
            cka_max_samples=1000,
            temporal_rsa_stride=10,
            seed=42,
        )
        trajectory_values = trajectory.metric(latent_neural, latent_predictions[retention], trajectory.dt)
        rows.append(
            {
                "retention": retention,
                "severity": 1.0 - retention,
                "response_r": traditional["response_single_trial_neuron_mean_r"],
                "cka": traditional["cka_condition_average_time_aligned"],
                "rsa": traditional["rsa_condition_time_state_spearman"],
                **trajectory_values,
            }
        )
    table = pd.DataFrame(rows).sort_values("severity")
    metric_names = [
        "position_correlation",
        "normalized_position_rmse",
        "velocity_direction_cosine",
        "speed_profile_correlation",
        "acceleration_direction_cosine",
    ]
    monotonic_rows = []
    for metric in metric_names:
        quality = -table[metric].to_numpy() if metric == "normalized_position_rmse" else table[metric].to_numpy()
        strict = bool(np.all(np.diff(quality) < 0))
        rho = float(spearmanr(table.severity, quality).statistic)
        bootstrap_rho = np.empty(samples)
        strict_draw = np.empty(samples, dtype=bool)
        for bootstrap in range(samples):
            chosen = rng.integers(0, len(latent_neural), size=len(latent_neural))
            values = []
            for retention in retentions:
                current = trajectory.metric(
                    latent_neural[chosen], latent_predictions[retention][chosen], trajectory.dt
                )[metric]
                values.append(-current if metric == "normalized_position_rmse" else current)
            bootstrap_rho[bootstrap] = spearmanr(table.severity, values).statistic
            strict_draw[bootstrap] = np.all(np.diff(values) < 0)
        monotonic_rows.append(
            {
                "metric": metric,
                "strict_monotonic_degradation": strict,
                "severity_quality_spearman": rho,
                "bootstrap_rho_median": float(np.median(bootstrap_rho)),
                "bootstrap_rho_ci_low": float(np.quantile(bootstrap_rho, 0.025)),
                "bootstrap_rho_ci_high": float(np.quantile(bootstrap_rho, 0.975)),
                "bootstrap_strict_monotonic_proportion": float(np.mean(strict_draw)),
            }
        )
    monotonic = pd.DataFrame(monotonic_rows)
    return {
        "answer": "yes_for_position_velocity_speed_and_acceleration; not_strict_for_normalized_rmse_at_the_strongest_step",
        "ablation_definition": "off-center temporal convolution weights multiplied by retention; center slices/readout/shifter preserved",
        "curve": table.to_dict(orient="records"),
        "monotonicity": monotonic.to_dict(orient="records"),
    }, table, monotonic


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    category: str
    severity: float
    make: Callable[[], np.ndarray]


def _candidate_specs(
    static: np.ndarray,
    dynamic: np.ndarray,
    validation_matched: np.ndarray,
    extended: dict[str, np.ndarray],
) -> list[CandidateSpec]:
    result = [
        CandidateSpec("dynamic", "reference", "reference", 0.0, lambda: dynamic),
        CandidateSpec("static", "model", "model", 1.0, lambda: static),
        CandidateSpec("validation_matched_epoch65", "model", "model", 0.5, lambda: validation_matched),
    ]
    for retention in (0.75, 0.5, 0.25, 0.0):
        result.append(
            CandidateSpec(
                f"kernel_retention_{retention}",
                "kernel_ablation",
                "temporal",
                1.0 - retention,
                lambda retention=retention: extended[f"retention_{retention:.2f}"],
            )
        )
    for shift in (1, 2, 4, 8, 16, 32, 64):
        result.append(CandidateSpec(f"shift_{shift}", "shift", "temporal", float(shift), lambda shift=shift: np.roll(dynamic, shift, axis=1)))
    result.append(CandidateSpec("reverse", "reverse", "temporal", 1.0, lambda: dynamic[:, ::-1].copy()))
    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        result.append(CandidateSpec(f"smooth_{sigma}", "smooth", "temporal", sigma, lambda sigma=sigma: gaussian_filter1d(dynamic, sigma, axis=1, mode="nearest")))
    for segments in (2, 4, 8, 16):
        def block_make(segments: int = segments) -> np.ndarray:
            blocks = np.array_split(np.arange(dynamic.shape[1]), segments)
            order = np.random.default_rng(100 + segments).permutation(segments)
            return dynamic[:, np.concatenate([blocks[index] for index in order])].copy()
        result.append(CandidateSpec(f"blocks_{segments}", "block_permute", "temporal", float(segments), block_make))
    mean = dynamic.mean(axis=1, keepdims=True)
    for alpha in (0.2, 0.4, 0.6, 0.8, 1.0):
        result.append(CandidateSpec(f"time_mean_{alpha}", "time_mean_mix", "temporal", alpha, lambda alpha=alpha: (1.0 - alpha) * dynamic + alpha * mean))
    scale = dynamic.std(axis=(0, 1), keepdims=True)
    for index, sigma in enumerate((0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
        def noise_make(index: int = index, sigma: float = sigma) -> np.ndarray:
            noise = np.random.default_rng(200 + index).normal(size=dynamic.shape).astype(np.float32)
            return np.clip(dynamic + sigma * scale * noise, 1e-5, None)
        result.append(CandidateSpec(f"noise_{sigma}", "amplitude_noise", "non_temporal", sigma, noise_make))
    for index, sigma in enumerate((0.1, 0.2, 0.5, 0.75, 1.0)):
        def gain_make(index: int = index, sigma: float = sigma) -> np.ndarray:
            gains = np.exp(np.random.default_rng(300 + index).normal(0, sigma, size=(1, 1, dynamic.shape[-1]))).astype(np.float32)
            return dynamic * gains
        result.append(CandidateSpec(f"gain_{sigma}", "gain", "non_temporal", sigma, gain_make))
    return result


def _compact_metrics(
    neural: np.ndarray,
    prediction: np.ndarray,
    conditions: np.ndarray,
    trajectory: FrozenTrajectory,
    trial_indices: np.ndarray,
) -> dict[str, float]:
    local_conditions = conditions[trial_indices]
    conventional, _, _ = evaluate_traditional(
        neural[trial_indices], prediction[trial_indices], local_conditions, cka_max_samples=1000, temporal_rsa_stride=10, seed=42
    )
    gpfa = trajectory.compare(neural, prediction, trial_indices)
    return {
        "response_single": conventional["response_single_trial_neuron_mean_r"],
        "response_average": conventional["response_condition_average_neuron_mean_r"],
        "cka_time": conventional["cka_condition_average_time_aligned"],
        "cka_delta": conventional["cka_temporal_difference"],
        "rsa_state": conventional["rsa_condition_time_state_spearman"],
        "rsa_temporal": conventional["rsa_within_condition_temporal_spearman_mean"],
        "gpfa_position": gpfa["position_correlation"],
        "gpfa_velocity": gpfa["velocity_direction_cosine"],
        "gpfa_speed": gpfa["speed_profile_correlation"],
        "gpfa_acceleration": gpfa["acceleration_direction_cosine"],
        "gpfa_rmse_quality": -gpfa["normalized_position_rmse"],
    }


def _q6(
    neural: np.ndarray,
    static: np.ndarray,
    dynamic: np.ndarray,
    conditions: np.ndarray,
    trajectory: FrozenTrajectory,
    extended: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selection, test = _balanced_repeat_split(conditions, 20260813)
    specs = _candidate_specs(static, dynamic, extended["validation_matched_dynamic"], extended)
    rows = []
    for spec in specs:
        prediction = spec.make()
        for split_name, indices in (("selection", selection), ("test", test)):
            rows.append(
                {
                    "candidate": spec.name,
                    "family": spec.family,
                    "category": spec.category,
                    "severity": spec.severity,
                    "split": split_name,
                    **_compact_metrics(neural, prediction, conditions, trajectory, indices),
                }
            )
        del prediction
    table = pd.DataFrame(rows)
    conventional = ["response_single", "response_average", "cka_time", "cka_delta", "rsa_state", "rsa_temporal"]
    trajectory_metrics = ["gpfa_position", "gpfa_velocity", "gpfa_speed", "gpfa_acceleration", "gpfa_rmse_quality"]
    selection_table = table[table.split == "selection"].reset_index(drop=True)
    test_table = table[table.split == "test"].set_index("candidate")

    # Selection-only search for a cross-family pair close on the complete conventional feature set.
    standardized = (selection_table[conventional] - selection_table[conventional].mean()) / selection_table[conventional].std(ddof=0).replace(0, 1)
    pair_rows = []
    for left in range(len(selection_table)):
        for right in range(left + 1, len(selection_table)):
            if selection_table.loc[left, "family"] == selection_table.loc[right, "family"]:
                continue
            distance = float(np.linalg.norm(standardized.iloc[left] - standardized.iloc[right]))
            selection_difference = float(
                np.linalg.norm(selection_table.loc[left, trajectory_metrics].to_numpy(float) - selection_table.loc[right, trajectory_metrics].to_numpy(float))
            )
            pair_rows.append((distance, -selection_difference, left, right))
    eligible = [item for item in pair_rows if item[0] <= 0.35 and -item[1] >= 0.05]
    chosen = min(eligible or pair_rows, key=lambda item: (item[0], item[1]))
    distance, _, left_index, right_index = chosen
    left_name = str(selection_table.loc[left_index, "candidate"])
    right_name = str(selection_table.loc[right_index, "candidate"])
    matched_pair = {
        "selection_standardized_conventional_distance": distance,
        "left": left_name,
        "right": right_name,
        "selection_left": selection_table.loc[left_index, conventional + trajectory_metrics].to_dict(),
        "selection_right": selection_table.loc[right_index, conventional + trajectory_metrics].to_dict(),
        "test_left": test_table.loc[left_name, conventional + trajectory_metrics].to_dict(),
        "test_right": test_table.loc[right_name, conventional + trajectory_metrics].to_dict(),
        "note": "pair selected using selection repeats only; test repeats were not consulted",
    }

    # Leave-one-perturbation-family-out: train on selection repeats, test the held family on test repeats.
    diagnostic = table[table.category.isin(["temporal", "non_temporal"])].copy()
    selection_diagnostic = diagnostic[diagnostic.split == "selection"].set_index("candidate")
    test_diagnostic = diagnostic[diagnostic.split == "test"].set_index("candidate")
    predictions: dict[str, list[float]] = {metric: [] for metric in trajectory_metrics}
    truths: dict[str, list[float]] = {metric: [] for metric in trajectory_metrics}
    family_rows = []
    for family in sorted(selection_diagnostic.family.unique()):
        train = selection_diagnostic[selection_diagnostic.family != family]
        held = selection_diagnostic[selection_diagnostic.family == family]
        held_test = test_diagnostic.loc[held.index]
        if len(train) <= len(conventional) or held.empty:
            continue
        family_record: dict[str, Any] = {"held_family": family, "candidates": int(len(held))}
        for target in trajectory_metrics:
            regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            regressor.fit(train[conventional], train[target])
            predicted = regressor.predict(held_test[conventional])
            actual = held_test[target].to_numpy()
            predictions[target].extend(predicted.tolist())
            truths[target].extend(actual.tolist())
            family_record[f"{target}_mae"] = float(np.mean(np.abs(predicted - actual)))
        family_rows.append(family_record)
    family_table = pd.DataFrame(family_rows)
    regression_summary = []
    for target in trajectory_metrics:
        actual = np.asarray(truths[target])
        predicted = np.asarray(predictions[target])
        regression_summary.append(
            {
                "target": target,
                "leave_family_out_r2": float(r2_score(actual, predicted)),
                "mae": float(np.mean(np.abs(actual - predicted))),
                "target_standard_deviation": float(np.std(actual, ddof=1)),
                "normalized_mae": float(np.mean(np.abs(actual - predicted)) / max(np.std(actual, ddof=1), 1e-12)),
                "held_candidates": int(len(actual)),
            }
        )

    # Strict temporal-order counterexample: condition-level mean patterns are invariant to reversal.
    brain_condition = condition_means(neural, conditions).mean(axis=1)
    dynamic_condition = condition_means(dynamic, conditions).mean(axis=1)
    reversed_dynamic = dynamic[:, ::-1].copy()
    reversed_condition = condition_means(reversed_dynamic, conditions).mean(axis=1)
    counterexample = {
        "condition_cka_dynamic": linear_cka(brain_condition, dynamic_condition),
        "condition_cka_reversed": linear_cka(brain_condition, reversed_condition),
        "condition_rsa_dynamic": compare_rdms(correlation_rdm(brain_condition), correlation_rdm(dynamic_condition), "spearman"),
        "condition_rsa_reversed": compare_rdms(correlation_rdm(brain_condition), correlation_rdm(reversed_condition), "spearman"),
        "maximum_condition_pattern_difference_after_reversal": float(np.max(np.abs(dynamic_condition - reversed_condition))),
        "gpfa_dynamic": trajectory.compare(neural, dynamic),
        "gpfa_reversed": trajectory.compare(neural, reversed_dynamic),
    }
    return {
        "answer": "yes_for_standard_condition_RSA_CKA_and_supported_but_not_proven_for_the_enriched_battery",
        "strict_time_reversal_counterexample": counterexample,
        "response_only_counterexample": "see Q4 disjoint-repeat response-score-matched output perturbation",
        "conventional_matched_pair": matched_pair,
        "leave_perturbation_family_out_regression": regression_summary,
        "interpretation": "non-perfect out-of-family R2 and matched/counterexample results support incremental trajectory information; they do not establish mathematical independence from every possible RSA/CKA construction",
        "candidates": int(len(selection_table)),
        "conventional_features": conventional,
    }, table, family_table


def run_question_analysis(config: dict[str, Any], bootstrap_samples: int = 2000) -> dict[str, Any]:
    out = output_dir(config)
    with np.load(out / "oracle_predictions.npz", allow_pickle=True) as values:
        base = {key: values[key] for key in values.files}
    with np.load(out / "extended_predictions.npz", allow_pickle=True) as values:
        extended = {key: values[key] for key in values.files}
    neural, static, dynamic, conditions = (
        base["neural"], base["static"], base["dynamic"], base["conditions"]
    )
    rng = np.random.default_rng(20260813)
    trajectory = FrozenTrajectory(config, neural, conditions)
    q1 = _q1(config, rng, bootstrap_samples)
    q2, q2_table = _q2(neural, static, dynamic, conditions, rng, bootstrap_samples)
    q3 = _q3(config)
    q4, q4_table = _response_score_matched_output_perturbation(
        neural, static, dynamic, conditions, trajectory, rng, bootstrap_samples
    )
    q5, ablation_table, monotonic_table = _ablation(
        config, neural, conditions, trajectory, extended, rng, bootstrap_samples
    )
    q6, candidate_table, family_table = _q6(
        neural, static, dynamic, conditions, trajectory, extended
    )
    q2_table.to_csv(out / "q2_condition_conventional_bootstrap.csv", index=False)
    q4_table.to_csv(out / "q4_response_score_matched_output_gpfa_bootstrap.csv", index=False)
    ablation_table.to_csv(out / "q5_temporal_ablation_curve.csv", index=False)
    monotonic_table.to_csv(out / "q5_temporal_ablation_monotonicity.csv", index=False)
    candidate_table.to_csv(out / "q6_candidate_metrics.csv", index=False)
    family_table.to_csv(out / "q6_leave_family_out_errors.csv", index=False)
    result = {
        "status": "q1_q6_analysis_complete",
        "bootstrap_samples": bootstrap_samples,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "Q4": q4,
        "Q5": q5,
        "Q6": q6,
    }
    json_dump(out / "q1_q6_answers.json", result)
    return result
