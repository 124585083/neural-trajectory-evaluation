"""Part 3 evaluation on an explicit, independently qualified test split.

Historical entry points are deliberately unchanged. This module fits only the
new neural GPFA and a validation-selected output perturbation; it never trains
an encoder. All posterior inference is offline and uses the complete window.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .traditional import column_correlation, condition_means, correlation_rdm, linear_cka


PRIMARY = ("position_correlation", "velocity_direction_cosine")
SECONDARY = ("normalized_position_rmse", "speed_profile_correlation", "acceleration_direction_cosine")
TRAJECTORY = PRIMARY + SECONDARY
LOWER = {"normalized_position_rmse", "response_nmse"}
NULLS = ("condition_shuffle", "circular_shift", "time_reversal", "block_shuffle", "independent_neuron_shift")
EPS = 1e-12


def _imports() -> None:
    source = Path(__file__).resolve().parents[1]
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _digest(*values: Any) -> str:
    digest = hashlib.sha256()
    def update(value: Any) -> None:
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                update(str(key))
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        else:
            digest.update(json.dumps(_clean(value), sort_keys=True, separators=(",", ":")).encode())
    for value in values:
        update(value)
    return digest.hexdigest()


def _mean(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.mean(values[valid])) if np.any(valid) else float("nan")


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    return float(column_correlation(left.reshape(-1, 1), right.reshape(-1, 1))[0])


def _bundle_checks(bundle: dict[str, np.ndarray]) -> None:
    neural = np.asarray(bundle["neural"])
    if neural.ndim != 3 or min(neural.shape) < 2:
        raise ValueError("Expected trial x time x neuron arrays with nonempty axes")
    for name in ("neural", "static", "dynamic"):
        value = np.asarray(bundle[name])
        if value.shape != neural.shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must share complete finite trial/time/neuron support")
    for name in ("roles", "conditions", "dataset_indices", "trial_ids"):
        if np.asarray(bundle[name]).shape != (len(neural),):
            raise ValueError(f"{name} does not identify every trial")
    for role in ("train", "validation", "test"):
        role_mask = np.asarray(bundle["roles"]).astype(str) == role
        if not np.any(role_mask):
            raise ValueError(f"No {role} trials")
        if not np.any(neural[role_mask] != 0):
            raise ValueError(f"All-zero {role} responses cannot qualify as genuine labeled data")
    if len(np.unique(bundle["dataset_indices"])) != len(neural):
        raise ValueError("Dataset indices are duplicated")
    if len(np.unique(bundle["trial_ids"])) != len(neural):
        raise ValueError("Original trial IDs are duplicated")
    if np.asarray(bundle["neuron_ids"]).shape != (neural.shape[2],):
        raise ValueError("Neuron IDs do not match response columns")
    if np.asarray(bundle["frame_indices"]).shape != (neural.shape[1],):
        raise ValueError("Frame indices do not match response rows")
    if not np.all(np.diff(bundle["frame_indices"]) == 1):
        raise ValueError("This offline GPFA interface requires contiguous frame indices")
    precision = np.asarray(bundle["encoder_precision"])
    if precision.shape not in ((neural.shape[2],), neural.shape[1:]):
        raise ValueError("encoder_precision must be neuron or time x neuron")
    if not np.isfinite(precision).all() or np.any(precision <= 0):
        raise ValueError("Encoder precision must be finite and positive")
    if "movie_ids" in bundle and not np.array_equal(np.asarray(bundle["conditions"]).astype(str), np.asarray(bundle["movie_ids"]).astype(str)):
        raise ValueError("Condition groups must equal the audited movie identities")


def _noise(bundle: dict[str, np.ndarray], indices: np.ndarray, seed: int) -> np.ndarray:
    """Trial-identity noise is reproducible without consulting neural targets."""
    shape = np.asarray(bundle["dynamic"]).shape[1:]
    session = str(np.asarray(bundle["session"]).item())
    values = []
    for index in indices:
        identity = _digest(session, str(bundle["dataset_indices"][index]), int(seed))
        rng = np.random.default_rng(int(identity[:16], 16))
        values.append(rng.normal(size=shape))
    return np.stack(values)


def _response_r(neural: np.ndarray, prediction: np.ndarray) -> float:
    return _mean(column_correlation(neural.reshape(-1, neural.shape[-1]), prediction.reshape(-1, prediction.shape[-1])))


def _input_identity(bundle: dict) -> str:
    return _digest(*[np.asarray(bundle[k]) for k in ("session", "dataset_indices", "trial_ids", "conditions", "roles", "neuron_ids", "frame_indices")])


def _qualify(bundle: dict, manifest: dict, config: dict) -> None:
    from .unified_contract import canonical_digest, validate_bundle, validate_manifest
    validate_manifest(manifest, require_test=True)
    validate_bundle(bundle, manifest)
    _bundle_checks(bundle)
    if manifest["config_sha256"] != canonical_digest(config):
        raise ValueError("Configuration digest differs from the split manifest")
    for name in ("frame_start", "frame_stop", "frame_rate_hz"):
        if config["data"][name] != manifest["time"][name]:
            raise ValueError(f"Configured {name} differs from the split manifest")
    if int(config["gpfa"]["effective_observation_step"]) != manifest["time"]["observation_step"]:
        raise ValueError("GPFA observation grid differs from the split manifest")
    if config["data"].get("session", manifest["session"]) != manifest["session"]:
        raise ValueError("Configured session differs from the split manifest")
    if int(config["data"].get("neurons", len(bundle["neuron_ids"]))) != len(bundle["neuron_ids"]):
        raise ValueError("Configured neuron count differs from the split manifest")


@dataclass
class FrozenAnalysis:
    model: Any
    precision: np.ndarray
    encoder_precision: np.ndarray
    observations: np.ndarray
    query_times: np.ndarray
    neuron_ids: np.ndarray
    frame_indices: np.ndarray
    parameter_digest: str
    fit_digest: str
    sigma: float
    noise_scale: np.ndarray
    matching: dict[str, Any]
    config_digest: str
    input_identity: str
    transform_digest: str
    control_digest: str
    cache_identity: dict[str, Any]

    def latent(self, encoder_values: np.ndarray) -> np.ndarray:
        return self.latent_gpfa(self.gpfa_input(encoder_values))

    def gpfa_input(self, encoder_values: np.ndarray) -> np.ndarray:
        return np.asarray(encoder_values, dtype=np.float64) / self.encoder_precision * self.precision

    def latent_gpfa(self, gpfa_values: np.ndarray) -> np.ndarray:
        """Infer from already converted input, including response-space nulls."""
        before = self.model.parameter_digest()
        if before != self.parameter_digest:
            raise AssertionError("Frozen GPFA parameters changed before inference")
        if self.transform_digest != _digest(self.precision, self.encoder_precision, self.observations, self.query_times,
                self.neuron_ids, self.frame_indices, self.model.times_seconds, self.model.jitter, self.model.min_noise):
            raise AssertionError("Frozen GPFA preprocessing or time grid changed")
        value = self.model.transform_query(gpfa_values[:, self.observations], self.query_times)
        if self.model.parameter_digest() != before:
            raise AssertionError("GPFA inference changed fitted parameters")
        return value


def fit_frozen(bundle: dict[str, np.ndarray], manifest: dict, config: dict, output_dir: Path) -> FrozenAnalysis:
    """Fit with train neural data; select allowed initialization on validation.

    The final fit uses train again, never train plus validation. Test responses
    do not enter preprocessing, selection, output matching, or fit identity.
    """
    _imports()
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    from trajectory_reliability.selection import select_gpfa

    _qualify(bundle, manifest, config)
    from .unified_contract import cache_identity, file_sha256
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "frozen_gpfa.pkl").exists():
        raise FileExistsError("The frozen GPFA already exists; use a new run directory")
    role = np.asarray(bundle["roles"]).astype(str)
    train, validation = np.flatnonzero(role == "train"), np.flatnonzero(role == "validation")
    if len(validation) < 2:
        raise ValueError("GPFA initialization selection requires at least two validation trials")
    neural = np.asarray(bundle["neural"], dtype=np.float64)
    encoder_precision = np.asarray(bundle["encoder_precision"], dtype=np.float64).copy()
    raw_train = neural[train] / encoder_precision
    std = raw_train.reshape(-1, neural.shape[-1]).std(axis=0, ddof=1)
    floor = 0.01 * float(np.mean(std))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Training neural responses have no usable variance")
    precision = 1.0 / np.maximum(std, floor)
    gpfa = config["gpfa"]
    if int(gpfa["latent_dim"]) != 4:
        raise ValueError("Part 3 fixes latent_dim=4; no dimension search is allowed")
    frame_rate = float(config["data"]["frame_rate_hz"])
    observations = np.arange(0, neural.shape[1], int(gpfa["effective_observation_step"]))
    query_times = np.arange(neural.shape[1], dtype=float) / frame_rate
    times = query_times[observations]
    fit_values = raw_train[:, observations] * precision
    validation_values = (neural[validation] / encoder_precision)[:, observations] * precision
    seed = int(config["project"]["seed"])
    bounds = tuple(float(v) for v in gpfa["lengthscale_bounds_seconds"])
    selection = select_gpfa(fit_values, validation_values, times, [4],
                            list(map(float, gpfa["initial_lengthscales_seconds"])),
                            int(gpfa["selection_em_iterations"]), float(gpfa["tolerance"]), bounds, seed)
    selection.table.to_csv(output_dir / "gpfa_selection.csv", index=False)
    model = GaussianProcessFactorAnalysis(latent_dim=4, times_seconds=times,
        max_em_iterations=int(gpfa["final_em_iterations"]), tolerance=float(gpfa["tolerance"]),
        initial_lengthscale_seconds=selection.initial_lengthscale_seconds,
        lengthscale_bounds_seconds=bounds, random_seed=seed).fit(fit_values, split_name="train")
    model.save(output_dir / "frozen_gpfa.pkl")
    matching_cfg = config["response_matching"]
    tolerance = float(matching_cfg["tolerance"])
    if tolerance != 0.005:
        raise ValueError("The prespecified response-matching tolerance is 0.005")
    clip_minimum = float(matching_cfg.get("clip_minimum", 1e-5))
    if clip_minimum != 1e-5:
        raise ValueError("The inherited output-perturbation clipping floor is 1e-5")
    noise_scale = np.asarray(bundle["dynamic"])[train].std(axis=(0, 1), keepdims=True)
    noise = _noise(bundle, validation, int(matching_cfg["noise_seed"]))
    validation_neural_flat = neural[validation].reshape(-1, neural.shape[-1])
    static_correlations = column_correlation(validation_neural_flat,
        np.asarray(bundle["static"])[validation].reshape(validation_neural_flat.shape))
    candidates = list(map(float, matching_cfg.get("sigma_grid", [0., *np.geomspace(.01, 10., 100)])))
    if not candidates or min(candidates) < 0 or not np.isfinite(candidates).all():
        raise ValueError("sigma_grid must contain finite nonnegative fixed values")
    rows = []
    for sigma in candidates:
        candidate = np.clip(np.asarray(bundle["dynamic"])[validation] + sigma * noise_scale * noise, 1e-5, None)
        candidate_correlations = column_correlation(validation_neural_flat, candidate.reshape(validation_neural_flat.shape))
        shared_neurons = np.isfinite(static_correlations) & np.isfinite(candidate_correlations)
        target = _mean(static_correlations[shared_neurons])
        score = _mean(candidate_correlations[shared_neurons])
        rows.append({"sigma": sigma, "validation_static_response_r": target,
                     "validation_perturbed_dynamic_response_r": score, "absolute_gap": abs(score - target),
                     "n_shared_neurons": int(shared_neurons.sum()), "n_total_neurons": neural.shape[-1]})
    table = pd.DataFrame(rows)
    valid = np.isfinite(table.absolute_gap)
    selected = table.loc[table.loc[valid, "absolute_gap"].idxmin()] if valid.any() else None
    sigma = float(selected.sigma) if selected is not None else 0.0
    table["selected"] = False
    if selected is not None:
        table.loc[selected.name, "selected"] = True
    table.to_csv(output_dir / "response_matching_selection.csv", index=False)
    matching = {"sigma": sigma, "selection_role": "validation", "noise_scale_role": "train_predictions",
                "validation_absolute_gap": float(selected.absolute_gap) if selected is not None else None,
                "validation_matching_achieved": bool(selected is not None and selected.absolute_gap <= tolerance),
                "validation_shared_neurons": int(selected.n_shared_neurons) if selected is not None else 0,
                "total_gpfa_neurons_unchanged": neural.shape[-1],
                "status": "selected" if selected is not None else "undefined_validation_response_r",
                "tolerance": tolerance, "noise_seed": int(matching_cfg["noise_seed"]),
                "clip_lower": 1e-5, "kind": "dynamic_output_perturbation_not_a_separately_trained_model"}
    config_digest = _digest(config)
    fit_digest = _digest(neural[train], neural[validation], encoder_precision,
                        np.asarray(bundle["dynamic"])[train], np.asarray(bundle["static"])[validation],
                        np.asarray(bundle["dynamic"])[validation], np.asarray(bundle["dataset_indices"])[train],
                        np.asarray(bundle["dataset_indices"])[validation], np.asarray(bundle["neuron_ids"]),
                        np.asarray(bundle["frame_indices"]), config_digest)
    frozen = FrozenAnalysis(model, precision, encoder_precision, observations, query_times,
        np.asarray(bundle["neuron_ids"]).copy(), np.asarray(bundle["frame_indices"]).copy(),
        model.parameter_digest(), fit_digest, sigma, noise_scale, matching, config_digest,
        _input_identity(bundle), _digest(precision, encoder_precision, observations, query_times,
            np.asarray(bundle["neuron_ids"]), np.asarray(bundle["frame_indices"]), model.times_seconds, model.jitter, model.min_noise),
        _digest(sigma, noise_scale, matching), cache_identity(manifest, file_sha256(output_dir / "frozen_gpfa.pkl")))
    np.savez_compressed(output_dir / "frozen_gpfa_arrays.npz", C=model.C, d=model.d, R=model.R,
        precision=precision, encoder_precision=encoder_precision, neuron_ids=frozen.neuron_ids,
        frame_indices=frozen.frame_indices, observation_indices=observations,
        observation_times=times, query_times=query_times, noise_scale=noise_scale,
        train_dataset_indices=np.asarray(bundle["dataset_indices"])[train],
        validation_dataset_indices=np.asarray(bundle["dataset_indices"])[validation],
        lengthscales=np.stack([model.lengthscales[k] for k in sorted(model.lengthscales)]))
    _json(output_dir / "frozen_identity.json", {
        "fit_digest": fit_digest, "gpfa_parameter_digest": frozen.parameter_digest,
        "input_identity": frozen.input_identity, "transform_digest": frozen.transform_digest,
        "control_digest": frozen.control_digest, "cache_identity": frozen.cache_identity,
        "config_digest": config_digest, "selection": selection.table.to_dict("records"),
        "selected_initial_lengthscale_seconds": selection.initial_lengthscale_seconds,
        "latent_dim": 4, "fit_role": "train", "selection_role": "validation",
        "validation_in_final_fit": False, "inference": "full_window_offline_posterior",
        "inference_settings": {"jitter": model.jitter, "minimum_observation_noise": model.min_noise},
        "scaling_formula": "gpfa_input = encoder_scaled_response / encoder_precision * train_raw_precision",
        "training_precision_floor": floor, "response_matching": matching,
        "manifest_provenance": manifest, "manifest_is_not_part_of_fit_digest": True})
    return frozen


def _trajectory_scores(left: np.ndarray, rights: dict[str, np.ndarray], dt: float) -> tuple[dict, dict]:
    """Legacy metric definitions with explicit NA and shared derivative support."""
    _imports()
    from trajectory_reliability.metrics import trajectory_metrics
    result = {name: {k: v for k, v in trajectory_metrics(left, right, dt).items() if k in TRAJECTORY}
              for name, right in rights.items()}
    support = {"position_bins": int(np.prod(left.shape[:2]))}
    left_center = left - left.mean(axis=(0, 1), keepdims=True)
    for name, right in rights.items():
        if np.std(left_center) <= EPS or np.std(right - right.mean(axis=(0, 1), keepdims=True)) <= EPS:
            result[name]["position_correlation"] = float("nan")
        pooled = np.mean(np.concatenate([left, right]), axis=(0, 1), keepdims=True)
        if np.sqrt(np.mean((left - pooled) ** 2)) <= EPS:
            result[name]["normalized_position_rmse"] = float("nan")
    for order, metric in ((1, "velocity_direction_cosine"), (2, "acceleration_direction_cosine")):
        a = np.diff(left, n=order, axis=1) / dt**order
        others = {k: np.diff(v, n=order, axis=1) / dt**order for k, v in rights.items()}
        a_norm = np.linalg.norm(a, axis=-1)
        mask = a_norm > EPS
        for b in others.values():
            mask &= np.linalg.norm(b, axis=-1) > EPS
        support[metric + "_bins"] = int(mask.sum())
        support[metric + "_total_bins"] = int(mask.size)
        for name, b in others.items():
            denominator = a_norm * np.linalg.norm(b, axis=-1)
            result[name][metric] = float(np.mean(np.sum(a * b, axis=-1)[mask] / denominator[mask])) if mask.any() else float("nan")
    speed_left = np.linalg.norm(np.diff(left, axis=1), axis=-1)
    for name, right in rights.items():
        result[name]["speed_profile_correlation"] = _corr(speed_left, np.linalg.norm(np.diff(right, axis=1), axis=-1))
    return result, support


def _conventional(neural: np.ndarray, predictions: dict[str, np.ndarray], conditions: np.ndarray,
                  stride: int) -> tuple[dict, dict]:
    flat = neural.reshape(-1, neural.shape[-1])
    columns = {name: column_correlation(flat, value.reshape(flat.shape)) for name, value in predictions.items()}
    shared_neurons = np.isfinite(np.stack(list(columns.values()))).all(axis=0)
    means = condition_means(neural, conditions)
    variance = float(np.mean((neural - neural.mean()) ** 2))
    left_state = means[:, ::stride].reshape(-1, means.shape[-1])
    upper = np.triu_indices(len(left_state), 1)
    left_rdm = correlation_rdm(left_state)[upper]
    right_rdms = {}
    for name, value in predictions.items():
        state = condition_means(value, conditions)[:, ::stride].reshape(left_state.shape)
        right_rdms[name] = correlation_rdm(state)[upper]
    rdm_support = np.isfinite(left_rdm)
    for value in right_rdms.values():
        rdm_support &= np.isfinite(value)
    result = {}
    for name, prediction in predictions.items():
        right = condition_means(prediction, conditions).reshape(-1, neural.shape[-1])
        left = means.reshape(right.shape)
        dx, dy = np.diff(means, axis=1).reshape(-1, neural.shape[-1]), np.diff(condition_means(prediction, conditions), axis=1).reshape(-1, neural.shape[-1])
        cka = linear_cka(left, right) if np.linalg.norm(left - left.mean(0)) > EPS and np.linalg.norm(right - right.mean(0)) > EPS else float("nan")
        delta_cka = linear_cka(dx, dy) if np.linalg.norm(dx - dx.mean(0)) > EPS and np.linalg.norm(dy - dy.mean(0)) > EPS else float("nan")
        a, b = left_rdm[rdm_support], right_rdms[name][rdm_support]
        rsa = float(spearmanr(a, b).statistic) if len(a) >= 3 and np.std(a) > EPS and np.std(b) > EPS else float("nan")
        result[name] = {"response_neuron_mean_r": float(np.mean(columns[name][shared_neurons])) if shared_neurons.any() else float("nan"),
                        "response_nmse": float(np.mean((neural - prediction)**2) / variance) if variance > EPS else float("nan"),
                        "cka_condition_time_aligned": cka, "cka_temporal_difference": delta_cka,
                        "rsa_condition_time_state_spearman": rsa}
    return result, {"response_r_shared_neurons": int(shared_neurons.sum()), "response_r_total_neurons": neural.shape[-1],
                    "rsa_shared_edges": int(rdm_support.sum()), "rsa_total_edges": len(rdm_support)}


def _all_scores(neural: np.ndarray, predictions: dict[str, np.ndarray], conditions: np.ndarray,
                left: np.ndarray, rights: dict[str, np.ndarray], dt: float, stride: int) -> tuple[dict, dict]:
    trajectory, support = _trajectory_scores(left, rights, dt)
    conventional, other_support = _conventional(neural, predictions, conditions, stride)
    return {name: {**trajectory[name], **conventional[name]} for name in predictions}, {**support, **other_support}


def _reliability(bundle: dict[str, np.ndarray], frozen: FrozenAnalysis, config: dict, out: Path,
                 indices: np.ndarray) -> dict:
    from trajectory_reliability.reliability import _balanced_half, _null_response
    conditions = np.asarray(bundle["conditions"])[indices]
    unique = np.unique(conditions)
    groups = [np.asarray(bundle["neural"])[indices][conditions == c] for c in unique]
    counts = [len(g) for g in groups]
    if len(groups) < 2 or min(counts) < 2:
        status = {"status": "insufficient_test_repeats", "condition_ids": unique.tolist(),
                  "repeat_counts": counts, "conditions_removed": 0}
        _json(out / "reliability_status.json", status)
        pd.DataFrame([{**status, "metric": metric, "value": np.nan} for metric in TRAJECTORY]).to_csv(out / "reliability_summary.csv", index=False)
        return status
    rng = np.random.default_rng(int(config["project"]["seed"]))
    observed_rows, null_rows, split_identities = [], [], []
    dt = 1.0 / float(config["data"]["frame_rate_hz"])
    blocks = tuple(map(int, config["reliability"]["block_sizes_frames"]))
    if not blocks or min(blocks) <= 0 or len(set(blocks)) != len(blocks):
        raise ValueError("Reliability block sizes must be distinct positive integers")
    null_specs = [(name, name, None) for name in NULLS if name != "block_shuffle"]
    null_specs += [(f"block_shuffle_{size}", "block_shuffle", size) for size in blocks]
    for split in range(int(config["reliability"]["splits"])):
        halves = [_balanced_half(len(g), rng) for g in groups]
        for condition, half in zip(unique, halves):
            condition_indices = indices[conditions == condition]
            used = set(np.concatenate(half).tolist())
            for side, chosen in (("half1", half[0]), ("half2", half[1]),
                                 ("unused_odd_repeat", np.array([i for i in range(len(condition_indices)) if i not in used], dtype=int))):
                for local_index in chosen:
                    source = condition_indices[local_index]
                    split_identities.append({"split": split, "condition": condition, "half": side,
                        "session": str(np.asarray(bundle["session"]).item()), "role": "test",
                        "dataset_index": int(bundle["dataset_indices"][source]), "trial_id": str(bundle["trial_ids"][source])})
        a = np.stack([g[h[0]].mean(0) for g, h in zip(groups, halves)])
        b = np.stack([g[h[1]].mean(0) for g, h in zip(groups, halves)])
        left = frozen.latent(a)
        rights = {"observed": frozen.latent(b)}
        fixed_scale_b = frozen.gpfa_input(b)
        for label, name, block in null_specs:
            rights[label] = frozen.latent_gpfa(_null_response(fixed_scale_b, name, rng, block))
        metrics, support = _trajectory_scores(left, rights, dt)
        for metric, value in metrics["observed"].items():
            observed_rows.append({"split": split, "metric": metric, "value": value, **support})
        for label, _, _ in null_specs:
            for metric, value in metrics[label].items():
                null_rows.append({"split": split, "null": label, "metric": metric, "value": value,
                                  "observed": metrics["observed"][metric], **support})
    observed, nulls = pd.DataFrame(observed_rows), pd.DataFrame(null_rows)
    observed.to_csv(out / "reliability_observed.csv", index=False)
    nulls.to_csv(out / "reliability_nulls.csv", index=False)
    pd.DataFrame(split_identities).to_csv(out / "reliability_split_identities.csv", index=False)
    rows = []
    for (metric, null), group in nulls.groupby(["metric", "null"]):
        a, b = group.observed.to_numpy(), group.value.to_numpy()
        valid = np.isfinite(a) & np.isfinite(b)
        delta = (a - b) * (-1 if metric in LOWER else 1)
        rows.append({"metric": metric, "null": null, "observed_mean": _mean(a[valid]),
                     "null_mean": _mean(b[valid]), "paired_superiority": float(np.mean(delta[valid] > 0)) if valid.any() else np.nan,
                     "split_null_failures": int(np.sum(delta[valid] <= 0)), "undefined_splits": int((~valid).sum()),
                     "splits": len(group), "interval_unit": "repeat_half_randomization_not_biological_samples"})
    pd.DataFrame(rows).to_csv(out / "reliability_summary.csv", index=False)
    status = {"status": "computed", "condition_ids": unique.tolist(), "repeat_counts": counts,
              "conditions_removed": 0, "splits": int(config["reliability"]["splits"]),
              "nulls": [spec[0] for spec in null_specs], "normalization": "none"}
    _json(out / "reliability_status.json", status)
    return status


def score_frozen(bundle: dict[str, np.ndarray], frozen: FrozenAnalysis, config: dict, output_dir: Path) -> dict:
    """Evaluate current test arrays; never reuse historical prediction scores."""
    _imports()
    from trajectory_reliability.reliability import _null_response
    _bundle_checks(bundle)
    if frozen.config_digest != _digest(config):
        raise ValueError("Analysis configuration differs from the frozen configuration")
    if _input_identity(bundle) != frozen.input_identity:
        raise ValueError("Scoring trial, role, condition, neuron, or time identity changed")
    for name, saved in (("neuron_ids", frozen.neuron_ids), ("frame_indices", frozen.frame_indices), ("encoder_precision", frozen.encoder_precision)):
        if not np.array_equal(np.asarray(bundle[name]), saved):
            raise ValueError(f"Frozen {name} differs from the scoring input")
    if frozen.model.parameter_digest() != frozen.parameter_digest:
        raise AssertionError("Frozen GPFA changed")
    if frozen.control_digest != _digest(frozen.sigma, frozen.noise_scale, frozen.matching):
        raise AssertionError("Frozen output perturbation settings changed")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    indices = np.flatnonzero(np.asarray(bundle["roles"]).astype(str) == "test")
    conditions = np.asarray(bundle["conditions"])[indices]
    unique = np.unique(conditions)
    neural = np.asarray(bundle["neural"])[indices]
    predictions = {name: np.asarray(bundle[name])[indices] for name in ("static", "dynamic")}
    means = {name: condition_means(value, conditions) for name, value in predictions.items()}
    left = frozen.latent(condition_means(neural, conditions))
    rights = {name: frozen.latent(value) for name, value in means.items()}
    dt = 1.0 / float(config["data"]["frame_rate_hz"])
    stride = int(config["traditional"]["temporal_rsa_stride"])
    scores, support = _all_scores(neural, predictions, conditions, left, rights, dt, stride)
    session = str(np.asarray(bundle["session"]).item())
    identity = {"session": session, "role": "test", "seed": int(config["project"]["seed"])}
    rows = [{**identity, "model": name, "metric": metric, "value": value,
             "primary": metric in PRIMARY, "status": "defined" if np.isfinite(value) else "undefined",
             "n_conditions": len(unique), "n_trials": len(indices), "n_frames": neural.shape[1], **support}
            for name, metrics in scores.items() for metric, value in metrics.items()]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    neuron_rows = []
    for model, prediction in predictions.items():
        correlations = column_correlation(neural.reshape(-1, neural.shape[-1]), prediction.reshape(-1, prediction.shape[-1]))
        for neuron, value in zip(frozen.neuron_ids, correlations):
            neuron_rows.append({**identity, "model": model, "neuron_id": str(neuron), "response_r": value,
                                "status": "defined" if np.isfinite(value) else "undefined"})
    pd.DataFrame(neuron_rows).to_csv(out / "response_per_neuron.csv", index=False)
    condition_rows = []
    for i, condition in enumerate(unique):
        mask = conditions == condition
        local, local_support = _all_scores(neural[mask], {k: v[mask] for k, v in predictions.items()}, conditions[mask],
                                          left[i:i+1], {k: v[i:i+1] for k, v in rights.items()}, dt, stride)
        for name, metrics in local.items():
            for metric, value in metrics.items():
                condition_rows.append({**identity, "condition": condition, "model": name, "metric": metric, "value": value,
                    "dataset_indices": ";".join(map(str, np.asarray(bundle["dataset_indices"])[indices][mask])),
                    "trial_ids": ";".join(map(str, np.asarray(bundle["trial_ids"])[indices][mask])),
                    "n_repeats": int(mask.sum()), **local_support})
    condition_table = pd.DataFrame(condition_rows)
    condition_table.to_csv(out / "condition_metrics.csv", index=False)
    reliability = _reliability(bundle, frozen, config, out, indices)
    # A condition is the paired resampling unit. Encoders and GPFA remain fixed.
    rng = np.random.default_rng(int(config["project"]["seed"]))
    bootstrap_rows = []
    sample_count = int(config["traditional"]["paired_bootstrap_samples"])
    draws = {metric: [] for metric in scores["static"]}
    local_arrays = {name: condition_table[condition_table.model == name].pivot(index="condition", columns="metric", values="value").reindex(unique)
                    for name in predictions}
    for draw in range(sample_count):
        chosen = rng.integers(0, len(unique), size=len(unique))
        latent_metrics, _ = _trajectory_scores(left[chosen], {k: v[chosen] for k, v in rights.items()}, dt)
        for metric in draws:
            if metric in TRAJECTORY:
                delta = latent_metrics["dynamic"][metric] - latent_metrics["static"][metric]
            else:
                a = local_arrays["static"][metric].to_numpy()[chosen]
                b = local_arrays["dynamic"][metric].to_numpy()[chosen]
                paired = np.isfinite(a) & np.isfinite(b)
                delta = float(np.mean(b[paired] - a[paired])) if paired.any() else np.nan
            draws[metric].append(delta)
    for metric, values in draws.items():
        values = np.asarray(values)
        valid = np.isfinite(values)
        interval = np.quantile(values[valid], [.025, .975]) if valid.any() and len(unique) >= 2 else [np.nan, np.nan]
        if metric in TRAJECTORY:
            a_point, b_point = scores["static"][metric], scores["dynamic"][metric]
        else:
            a, b = local_arrays["static"][metric].to_numpy(), local_arrays["dynamic"][metric].to_numpy()
            paired = np.isfinite(a) & np.isfinite(b)
            a_point, b_point = _mean(a[paired]), _mean(b[paired])
        bootstrap_rows.append({**identity, "metric": metric, "static": a_point, "dynamic": b_point,
            "dynamic_minus_static": b_point - a_point,
            "pooled_static": scores["static"][metric], "pooled_dynamic": scores["dynamic"][metric],
            "pooled_dynamic_minus_static": scores["dynamic"][metric] - scores["static"][metric],
            "ci_low": interval[0], "ci_high": interval[1], "valid_bootstraps": int(valid.sum()),
            "undefined_bootstraps": int((~valid).sum()), "resampling_unit": "movie_condition",
            "interval_estimand": "pooled_condition_trajectory" if metric in TRAJECTORY else "mean_paired_condition_score",
            "n_conditions": len(unique), "biological_sample_count": "not_estimated"})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "paired_bootstrap.csv", index=False)
    null_rows = []
    for name in NULLS:
        if name == "condition_shuffle" and len(unique) < 2:
            null_rows.extend({**identity, "model": model, "null": name, "metric": metric, "value": np.nan,
                              "observed": scores[model][metric], "status": "insufficient_conditions"}
                             for model in predictions for metric in TRAJECTORY)
            continue
        repetitions = 1 if name == "time_reversal" else int(config["model_nulls"]["permutations"])
        for permutation in range(repetitions):
            paired_seed = int(rng.integers(0, 2**32))
            null_latents = {model: frozen.latent_gpfa(_null_response(frozen.gpfa_input(value), name, np.random.default_rng(paired_seed),
                int(config["model_nulls"]["block_size_frames"]) if name == "block_shuffle" else None)) for model, value in means.items()}
            values, null_support = _trajectory_scores(left, {**rights, **{k + "_null": v for k, v in null_latents.items()}}, dt)
            for model in predictions:
                for metric in TRAJECTORY:
                    null_rows.append({**identity, "model": model, "null": name, "permutation": permutation,
                                     "metric": metric, "observed": values[model][metric], "value": values[model + "_null"][metric], **null_support})
    pd.DataFrame(null_rows).to_csv(out / "model_nulls.csv", index=False)
    matching_prediction = np.clip(predictions["dynamic"] + frozen.sigma * frozen.noise_scale *
        _noise(bundle, indices, int(config["response_matching"]["noise_seed"])), 1e-5, None)
    control_predictions = {"static": predictions["static"], "perturbed_dynamic": matching_prediction}
    control_latents = {"static": rights["static"], "perturbed_dynamic": frozen.latent(condition_means(matching_prediction, conditions))}
    control, control_support = _all_scores(neural, control_predictions, conditions, left, control_latents, dt, stride)
    pd.DataFrame([{**identity, "model": name, "metric": metric, "value": value, **control_support}
                  for name, metrics in control.items() for metric, value in metrics.items()]).to_csv(out / "response_matching_metrics.csv", index=False)
    gap = abs(control["static"]["response_neuron_mean_r"] - control["perturbed_dynamic"]["response_neuron_mean_r"])
    matching = {**frozen.matching, "test_absolute_gap": gap,
                "test_matching_retained": bool(np.isfinite(gap) and gap <= 0.005),
                "test_retuned": False}
    _json(out / "response_matching.json", matching)
    control_bootstrap = []
    control_draws = {metric: [] for metric in TRAJECTORY}
    control_rng = np.random.default_rng(int(config["project"]["seed"]))
    for draw in range(sample_count):
        chosen = control_rng.integers(0, len(unique), size=len(unique))
        values, _ = _trajectory_scores(left[chosen], {k: v[chosen] for k, v in control_latents.items()}, dt)
        for metric in TRAJECTORY:
            control_draws[metric].append(values["perturbed_dynamic"][metric] - values["static"][metric])
    for metric, values in control_draws.items():
        values = np.asarray(values)
        finite = np.isfinite(values)
        interval = np.quantile(values[finite], [.025, .975]) if finite.any() and len(unique) >= 2 else [np.nan, np.nan]
        control_bootstrap.append({**identity, "metric": metric,
            "perturbed_dynamic_minus_static": control["perturbed_dynamic"][metric] - control["static"][metric],
            "ci_low": interval[0], "ci_high": interval[1], "valid_bootstraps": int(finite.sum()),
            "undefined_bootstraps": int((~finite).sum()), "resampling_unit": "movie_condition"})
    control_bootstrap_table = pd.DataFrame(control_bootstrap)
    control_bootstrap_table.to_csv(out / "response_matching_bootstrap.csv", index=False)
    from .unified_contract import save_cache
    save_cache(out / "test_latents.npz", {"neural": left, "static": rights["static"], "dynamic": rights["dynamic"],
        "conditions": unique, "frame_indices": frozen.frame_indices,
        "dataset_indices": np.asarray(bundle["dataset_indices"])[indices],
        "trial_ids": np.asarray(bundle["trial_ids"])[indices], "trial_conditions": conditions,
        "gpfa_parameter_digest": np.asarray(frozen.parameter_digest)}, frozen.cache_identity)
    primary = bootstrap[bootstrap.metric.isin(PRIMARY)]
    if not frozen.matching["validation_matching_achieved"] or not matching["test_matching_retained"]:
        q1 = "response_matching_not_achieved_or_not_retained"
    else:
        control_primary = control_bootstrap_table[control_bootstrap_table.metric.isin(PRIMARY)]
        separated = ((control_primary.ci_low > 0) | (control_primary.ci_high < 0)).any()
        q1 = "trajectory_difference_under_matched_response_summary" if separated else "no_resolved_primary_trajectory_difference_under_matched_response_summary"
    q3 = {row.metric: ("dynamic_positive_interval" if row.ci_low > 0 else
                       "static_positive_interval" if row.ci_high < 0 else
                       "undefined" if not np.isfinite(row.ci_low) else "interval_includes_zero")
          for row in primary.itertuples()}
    summary = {"status": "scored", "session": session, "n_test_conditions": len(unique), "n_test_trials": len(indices),
               "gpfa_parameter_digest": frozen.parameter_digest, "fit_digest": frozen.fit_digest,
               "test_content_digest": _digest(neural, predictions, np.asarray(bundle["dataset_indices"])[indices]),
               "primary_metrics": list(PRIMARY), "secondary_metrics": list(SECONDARY),
               "scores": scores, "support": support, "reliability": reliability, "Q1": q1,
               "Q2": "absolute_scores_and_paired_condition_intervals_reported", "Q3": q3,
               "response_matching": matching, "scope": "one_session_fixed_models_condition_resampling",
               "inference": "offline_full_window_not_causal", "undefined_metrics_are_not_zero": True}
    if frozen.model.parameter_digest() != frozen.parameter_digest:
        raise AssertionError("GPFA parameters changed during scoring")
    _json(out / "summary.json", summary)
    _figures(out, scores, bootstrap, condition_table, reliability)
    return summary


def _figures(out: Path, scores: dict, bootstrap: pd.DataFrame, condition_table: pd.DataFrame, reliability: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric in zip(axes, PRIMARY):
        for name, marker in (("static", "o"), ("dynamic", "s")):
            rows = condition_table[(condition_table.model == name) & (condition_table.metric == metric)]
            axis.plot(np.arange(len(rows)), rows.value, marker=marker, label=name.title())
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("Test movie condition (sorted ID)")
        axis.set_ylabel("Agreement")
        axis.legend()
    fig.savefig(out / "01_condition_trajectory.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric in zip(axes, PRIMARY):
        row = bootstrap[bootstrap.metric == metric].iloc[0]
        axis.axhline(0, color="gray", linewidth=1)
        axis.plot([0], [row.dynamic_minus_static], "o")
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            axis.vlines(0, row.ci_low, row.ci_high)
        axis.set_xticks([0], ["Dynamic minus Static"])
        axis.set_title(metric.replace("_", " "))
        axis.set_ylabel("Paired condition bootstrap interval")
    fig.savefig(out / "02_paired_differences.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(np.arange(len(reliability["repeat_counts"])), reliability["repeat_counts"])
    axes[0].set_xlabel("Test movie condition (sorted ID)")
    axes[0].set_ylabel("Available repeats")
    axes[0].set_title("All test conditions retained")
    if reliability["status"] == "computed":
        table = pd.read_csv(out / "reliability_summary.csv")
        for i, metric in enumerate(PRIMARY):
            rows = table[table.metric == metric]
            axes[1].plot(np.arange(len(rows)), rows.paired_superiority, "o-", label=metric.replace("_", " "))
        labels = sorted(table["null"].unique())
        axes[1].set_xticks(np.arange(len(labels)), [s.replace("_", " ") for s in labels], rotation=40, ha="right")
        axes[1].set_ylabel("Paired repeat splits above null")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(.5, .5, "Reliability unavailable:\ninsufficient test repeats", ha="center", va="center", transform=axes[1].transAxes)
    fig.savefig(out / "03_reliability_coverage.png", dpi=150)
    plt.close(fig)


def run_analysis(bundle: dict[str, np.ndarray], manifest: dict, config: dict,
                 output_dir: Path, *, synthetic: bool = False) -> dict:
    """Run only after provenance qualification; synthetic results stay labeled."""
    _qualify(bundle, manifest, config)
    if bool(manifest.get("synthetic", False)) != bool(synthetic):
        raise ValueError("Synthetic status must agree with the manifest")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory; frozen runs are not overwritten")
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "analysis_config.json", config)
    _json(out / "split_manifest.json", manifest)
    frozen = fit_frozen(bundle, manifest, config, out)
    result = score_frozen(bundle, frozen, config, out)
    result["status"] = "synthetic_verified" if synthetic else "real_analysis_complete"
    _json(out / "summary.json", result)
    generated = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    _json(out / "generated_manifest.json", {"status": result["status"], "files": generated,
        "gpfa_parameter_digest": frozen.parameter_digest, "fit_digest": frozen.fit_digest,
        "config_digest": frozen.config_digest, "input_manifest": manifest})
    return result
