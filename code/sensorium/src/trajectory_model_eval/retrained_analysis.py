"""Train-only GPFA and frozen evaluation for the retrained Part 3 protocol.

GPFA fitting and response-control selection accept development arrays only.
Test inference is a later operation, after all fitted artifacts are locked.
Historical reuse-checkpoint entry points and their stricter contracts are intact.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .traditional import column_correlation
from .unified_analysis import (FrozenAnalysis, _digest, _imports, _input_identity,
    _json, _mean, _noise, score_frozen)
from .unified_contract import canonical_digest, file_sha256, manifest_digest

PROTOCOL = "retrained_v1"
DEVELOPMENT = ("train", "validation")


def _row_identity(bundle: dict) -> str:
    """Hash identity values independently of NumPy string-width storage."""
    return canonical_digest({key: np.asarray(bundle[key]).astype(str).tolist() for key in
        ("session", "dataset_indices", "trial_ids", "conditions", "roles", "neuron_ids", "frame_indices")})


def validate_retrained_manifest(manifest: dict, config: dict) -> None:
    """Validate new roles without relabeling historical encoder identities."""
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("A retrained_v1 manifest is required")
    if manifest.get("split_sha256") != manifest_digest(manifest):
        raise ValueError("Retrained split manifest identity changed")
    if manifest.get("config_sha256") != canonical_digest(config):
        raise ValueError("Retrained manifest config_sha256 differs from the analysis configuration")
    if manifest["session"] != config["data"]["session"]:
        raise ValueError("Pilot session differs from the fixed configuration")
    for name in ("frame_start", "frame_stop", "frame_rate_hz"):
        if manifest["time"][name] != config["data"][name]:
            raise ValueError(f"Frozen time support changed: {name}")
    if manifest["time"]["observation_step"] != config["gpfa"]["effective_observation_step"]:
        raise ValueError("Observation grid differs from the fixed configuration")
    seen, groups, movies = set(), {}, {}
    for row in manifest["trials"]:
        key = (row["session"], row["dataset_index"])
        if key in seen:
            raise ValueError("A trial occurs twice in the split manifest")
        seen.add(key)
        role = row["role"]
        if role == "excluded":
            continue
        if role not in (*DEVELOPMENT, "test"):
            raise ValueError("Unknown Part 3 role")
        expected_tier = "oracle" if role == "test" else "train"
        if row["source_tier"] != expected_tier:
            raise ValueError("Part 3 roles do not follow the authorized source tiers")
        if not row.get("movie_id"):
            raise ValueError("Every included trial needs an audited stimulus group")
        if not row.get("group_id"):
            raise ValueError("Every included trial needs its identified overlap group")
        groups.setdefault(str(row["group_id"]), set()).add(role)
        movies.setdefault(str(row["movie_id"]), set()).add(role)
    if any(len(roles) != 1 for roles in (*groups.values(), *movies.values())):
        raise ValueError("An identified movie group crosses Part 3 roles")


def _identity_bundle(manifest: dict, roles: tuple[str, ...]) -> dict[str, np.ndarray]:
    rows = [r for r in manifest["trials"] if r["session"] == manifest["session"] and r["role"] in roles]
    return {"session": np.asarray(manifest["session"]),
        "dataset_indices": np.asarray([r["dataset_index"] for r in rows]),
        "trial_ids": np.asarray([str(r["trial_id"]) for r in rows]),
        "conditions": np.asarray([str(r["movie_id"]) for r in rows]),
        "movie_ids": np.asarray([str(r["movie_id"]) for r in rows]),
        "roles": np.asarray([r["role"] for r in rows]),
        "neuron_indices": np.asarray(manifest["neurons"]["indices"]),
        "neuron_ids": np.asarray(manifest["neurons"]["ids"]),
        "frame_indices": np.arange(manifest["time"]["frame_start"], manifest["time"]["frame_stop"])}


def validate_development_bundle(bundle: dict, manifest: dict, config: dict,
                                *, predictions: bool = False) -> None:
    validate_retrained_manifest(manifest, config)
    _validate_bundle(bundle, manifest, DEVELOPMENT, predictions)


def _validate_bundle(bundle: dict, manifest: dict, roles: tuple[str, ...], predictions: bool) -> None:
    expected = _identity_bundle(manifest, roles)
    for name, value in expected.items():
        if not np.array_equal(np.asarray(bundle.get(name, [])).astype(str), value.astype(str)):
            raise ValueError(f"Bundle {name} does not match the exact declared rows and order")
    shape = (len(expected["roles"]), len(expected["frame_indices"]), len(expected["neuron_ids"]))
    names = ("neural", "static", "dynamic") if predictions else ("neural",)
    for name in names:
        value = np.asarray(bundle.get(name, []))
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} is not finite on the complete fixed support {shape}")
    for role in roles:
        mask = expected["roles"] == role
        if not mask.any() or not np.any(bundle["neural"][mask] != 0):
            raise ValueError(f"Missing real response support for {role}")
    precision = np.asarray(bundle.get("encoder_precision", []))
    if precision.shape not in ((shape[-1],), shape[1:]) or not np.isfinite(precision).all() or np.any(precision <= 0):
        raise ValueError("Encoder precision must match the fixed finite positive response scale")


def load_neural_development(data_root: Path, manifest: dict, config: dict,
                            response_precision: np.ndarray) -> dict:
    """Load no oracle responses; return the GPFA development bundle only.

    response_precision is the saved encoder train statistic in [neuron, time]
    order, before pilot neuron/frame selection.
    """
    validate_retrained_manifest(manifest, config)
    bundle = _identity_bundle(manifest, DEVELOPMENT)
    neurons, frames = bundle["neuron_indices"], bundle["frame_indices"]
    scale = np.asarray(response_precision)[np.ix_(neurons, frames)].T
    rows = [r for r in manifest["trials"] if r["session"] == manifest["session"] and r["role"] in DEVELOPMENT]
    neural = []
    for row in rows:
        source = Path(data_root) / row["source_files"]["responses"]
        if file_sha256(source) != row["source_sha256"]["responses"]:
            raise ValueError(f"Audited response content changed: {row['session']}:{row['dataset_index']}")
        raw = np.load(source, mmap_mode="r")[np.ix_(neurons, frames)].T
        neural.append(np.asarray(raw, dtype=np.float32) * scale)
    bundle.update(neural=np.stack(neural), encoder_precision=scale)
    validate_development_bundle(bundle, manifest, config)
    return bundle


@dataclass
class FrozenGPFA:
    model: Any
    arrays: dict[str, np.ndarray]
    identity: dict


def _new_output(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Frozen stage outputs cannot be overwritten")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_gpfa(bundle: dict, manifest: dict, config: dict,
             preprocessing_sha256: str, output_dir: Path) -> FrozenGPFA:
    """Fit a new GPFA without encoders or test arrays; never train+validation refit."""
    validate_development_bundle(bundle, manifest, config)
    _imports()
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    from trajectory_reliability.selection import select_gpfa

    out = _new_output(output_dir)
    gpfa = config["gpfa"]
    if int(gpfa["latent_dim"]) != 4 or gpfa.get("refit_train_plus_validation", False):
        raise ValueError("The inherited Part 3 measurement fixes q=4 and train-only fitting")
    roles = np.asarray(bundle["roles"]).astype(str)
    train, validation = np.flatnonzero(roles == "train"), np.flatnonzero(roles == "validation")
    if len(validation) < 2:
        raise ValueError("Initialization selection requires at least two validation trials")
    encoder_precision = np.asarray(bundle["encoder_precision"], dtype=np.float64)
    raw = np.asarray(bundle["neural"], dtype=np.float64) / encoder_precision
    std = raw[train].reshape(-1, raw.shape[-1]).std(axis=0, ddof=1)
    floor = .01 * float(np.mean(std))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Training responses have no usable variance")
    precision = 1. / np.maximum(std, floor)
    observations = np.arange(0, raw.shape[1], int(gpfa["effective_observation_step"]))
    query_times = np.arange(raw.shape[1], dtype=float) / float(config["data"]["frame_rate_hz"])
    times = query_times[observations]
    fit_values = raw[train][:, observations] * precision
    validation_values = raw[validation][:, observations] * precision
    seed = int(config["project"]["seed"])
    bounds = tuple(map(float, gpfa["lengthscale_bounds_seconds"]))
    _json(out / "status.json", {"status": "fitting", "fit_role": "train", "test_loaded": False})
    try:
        selection = select_gpfa(fit_values, validation_values, times, [4],
            list(map(float, gpfa["initial_lengthscales_seconds"])), int(gpfa["selection_em_iterations"]),
            float(gpfa["tolerance"]), bounds, seed)
    except Exception as error:
        _json(out / "status.json", {"status": "failed", "stage": "gpfa_validation_selection",
            "test_loaded": False, "error": f"{type(error).__name__}: {error}"})
        raise
    selection.table.to_csv(out / "gpfa_selection.csv", index=False)
    try:
        model = GaussianProcessFactorAnalysis(latent_dim=4, times_seconds=times,
            max_em_iterations=int(gpfa["final_em_iterations"]), tolerance=float(gpfa["tolerance"]),
            initial_lengthscale_seconds=selection.initial_lengthscale_seconds,
            lengthscale_bounds_seconds=bounds, random_seed=seed).fit(fit_values, split_name="train")
    except Exception as error:
        _json(out / "status.json", {"status": "failed", "stage": "gpfa_final_train_fit",
            "test_loaded": False, "error": f"{type(error).__name__}: {error}"})
        raise
    model.save(out / "frozen_gpfa.pkl")
    pd.DataFrame(model.history).to_csv(out / "gpfa_fit_history.csv", index=False)
    arrays = {"C": model.C, "d": model.d, "R": model.R, "precision": precision,
        "encoder_precision": encoder_precision, "neuron_ids": np.asarray(bundle["neuron_ids"]),
        "frame_indices": np.asarray(bundle["frame_indices"]), "observations": observations,
        "observation_times": times, "query_times": query_times,
        "lengthscales": np.stack([model.lengthscales[k] for k in sorted(model.lengthscales)]),
        "train_dataset_indices": np.asarray(bundle["dataset_indices"])[train],
        "validation_dataset_indices": np.asarray(bundle["dataset_indices"])[validation]}
    np.savez_compressed(out / "frozen_gpfa_arrays.npz", **arrays)
    identity = {"protocol": PROTOCOL, "split_sha256": manifest["split_sha256"],
        "config_sha256": canonical_digest(config), "preprocessing_sha256": preprocessing_sha256,
        "gpfa_sha256": file_sha256(out / "frozen_gpfa.pkl"), "arrays_sha256": file_sha256(out / "frozen_gpfa_arrays.npz"),
        "gpfa_parameter_digest": model.parameter_digest(),
        "fit_digest": _digest(raw[train], raw[validation], encoder_precision, arrays["train_dataset_indices"],
            arrays["validation_dataset_indices"], arrays["neuron_ids"], arrays["frame_indices"], config["gpfa"]),
        "development_neural_digest": _digest(np.asarray(bundle["neural"], dtype=np.float32)),
        "development_identity": _row_identity(bundle), "fit_role": "train", "selection_role": "validation",
        "n_train_trials": len(train), "n_validation_trials": len(validation), "test_loaded": False,
        "selected_initial_lengthscale_seconds": selection.initial_lengthscale_seconds,
        "latent_dim": 4, "validation_in_final_fit": False, "training_precision_floor": floor,
        "inference": "offline_full_window_posterior", "selection_criterion": "validation_neural_likelihood",
        "scaling_formula": "gpfa_input = encoder_scaled_response / encoder_precision * train_raw_precision",
        "final_em_iterations": len(model.history), "jitter": model.jitter, "minimum_noise": model.min_noise}
    _json(out / "frozen_gpfa_identity.json", identity)
    _json(out / "status.json", {"status": "gpfa_frozen", "test_loaded": False,
        "gpfa_parameter_digest": identity["gpfa_parameter_digest"]})
    return FrozenGPFA(model, arrays, identity)


def load_gpfa(path: Path, manifest: dict, config: dict, preprocessing_sha256: str) -> FrozenGPFA:
    _imports()
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    validate_retrained_manifest(manifest, config)
    path = Path(path)
    identity = json.loads((path / "frozen_gpfa_identity.json").read_text(encoding="utf-8"))
    for key, value in {"protocol": PROTOCOL, "split_sha256": manifest["split_sha256"],
            "config_sha256": canonical_digest(config), "preprocessing_sha256": preprocessing_sha256,
            "gpfa_sha256": file_sha256(path / "frozen_gpfa.pkl"),
            "arrays_sha256": file_sha256(path / "frozen_gpfa_arrays.npz")}.items():
        if identity.get(key) != value:
            raise ValueError(f"Frozen GPFA {key} mismatch")
    model = GaussianProcessFactorAnalysis.load(path / "frozen_gpfa.pkl")
    if model.parameter_digest() != identity["gpfa_parameter_digest"]:
        raise ValueError("GPFA fitted parameters changed")
    with np.load(path / "frozen_gpfa_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name in ("C", "d", "R"):
        if not np.array_equal(getattr(model, name), arrays[name]):
            raise ValueError("GPFA array export differs from its inference artifact")
    return FrozenGPFA(model, arrays, identity)


def _check_checkpoints(checkpoints: dict, manifest: dict, preprocessing_sha256: str) -> None:
    if set(checkpoints) != {"static", "dynamic"}:
        raise ValueError("Both completed encoder identities are required before test")
    for name, identity in checkpoints.items():
        for key, value in {"status": "complete", "initialization": "from_scratch",
                "selection_role": "validation", "split_sha256": manifest["split_sha256"],
                "preprocessing_sha256": preprocessing_sha256}.items():
            if identity.get(key) != value:
                raise ValueError(f"Test remains pending: {name} {key} is not frozen for this run")
        digest = identity.get("sha256", "")
        if len(digest) != 64 or any(letter not in "0123456789abcdef" for letter in digest):
            raise ValueError("A SHA-256 is required for each completed encoder")
        if identity.get("checkpoint_path"):
            from sensorium_runtime.paths import repository_path
            if file_sha256(repository_path(identity["checkpoint_path"])) != digest:
                raise ValueError("Frozen encoder artifact changed")


def freeze_analysis(bundle: dict, manifest: dict, config: dict, preprocessing_sha256: str,
                    gpfa_dir: Path, checkpoint_identities: dict, output_dir: Path) -> dict:
    """Select the inherited output perturbation on validation before loading test."""
    validate_development_bundle(bundle, manifest, config, predictions=True)
    _check_checkpoints(checkpoint_identities, manifest, preprocessing_sha256)
    gpfa = load_gpfa(gpfa_dir, manifest, config, preprocessing_sha256)
    if gpfa.identity["development_identity"] != _row_identity(bundle):
        raise ValueError("GPFA and encoder development trial identities differ")
    if gpfa.identity["development_neural_digest"] != _digest(np.asarray(bundle["neural"], dtype=np.float32)):
        raise ValueError("Development neural responses changed after GPFA fitting")
    # Encoder exports may round the normalized targets to float32.
    if not np.array_equal(bundle["encoder_precision"], gpfa.arrays["encoder_precision"]):
        raise ValueError("GPFA and encoder response scales differ")
    out = _new_output(output_dir)
    cfg = config["response_matching"]
    if float(cfg["tolerance"]) != .005 or float(cfg.get("clip_minimum", 1e-5)) != 1e-5:
        raise ValueError("The inherited response-matching tolerance and clipping floor are fixed")
    roles = np.asarray(bundle["roles"]).astype(str)
    train, validation = np.flatnonzero(roles == "train"), np.flatnonzero(roles == "validation")
    noise_scale = np.asarray(bundle["dynamic"])[train].std(axis=(0, 1), keepdims=True)
    noise = _noise(bundle, validation, int(cfg["noise_seed"]))
    neural = np.asarray(bundle["neural"])[validation]
    flat = neural.reshape(-1, neural.shape[-1])
    static_r = column_correlation(flat, np.asarray(bundle["static"])[validation].reshape(flat.shape))
    candidates = list(map(float, cfg.get("sigma_grid", [0., *np.geomspace(.01, 10., 100)])))
    if not candidates or not np.isfinite(candidates).all() or min(candidates) < 0:
        raise ValueError("The fixed perturbation grid must be nonnegative and finite")
    rows = []
    for sigma in candidates:
        prediction = np.clip(np.asarray(bundle["dynamic"])[validation] + sigma * noise_scale * noise, 1e-5, None)
        candidate_r = column_correlation(flat, prediction.reshape(flat.shape))
        common = np.isfinite(static_r) & np.isfinite(candidate_r)
        a, b = _mean(static_r[common]), _mean(candidate_r[common])
        rows.append({"sigma": sigma, "validation_static_response_r": a,
            "validation_perturbed_dynamic_response_r": b, "absolute_gap": abs(a-b),
            "n_shared_neurons": int(common.sum()), "n_total_neurons": neural.shape[-1]})
    table = pd.DataFrame(rows)
    valid = np.isfinite(table.absolute_gap)
    selected = table.loc[table.loc[valid, "absolute_gap"].idxmin()] if valid.any() else None
    sigma = float(selected.sigma) if selected is not None else 0.
    table["selected"] = False
    if selected is not None:
        table.loc[selected.name, "selected"] = True
    table.to_csv(out / "response_matching_selection.csv", index=False)
    matching = {"sigma": sigma, "selection_role": "validation", "noise_scale_role": "train_predictions",
        "validation_absolute_gap": float(selected.absolute_gap) if selected is not None else None,
        "validation_matching_achieved": bool(selected is not None and selected.absolute_gap <= .005),
        "validation_shared_neurons": int(selected.n_shared_neurons) if selected is not None else 0,
        "total_gpfa_neurons_unchanged": neural.shape[-1], "tolerance": .005,
        "status": "selected" if selected is not None else "undefined_validation_response_r",
        "noise_seed": int(cfg["noise_seed"]), "clip_lower": 1e-5,
        "kind": "dynamic_output_perturbation_not_a_separately_trained_model",
        "summary": "mean per-neuron r across trial and time; repeated movies not required"}
    np.savez_compressed(out / "frozen_control.npz", noise_scale=noise_scale)
    identity = {"protocol": PROTOCOL, "status": "all_choices_frozen", "test_loaded": False,
        "split_sha256": manifest["split_sha256"], "config_sha256": canonical_digest(config),
        "preprocessing_sha256": preprocessing_sha256, "gpfa_identity": gpfa.identity,
        "checkpoint_identities": checkpoint_identities, "response_matching": matching,
        "control_sha256": file_sha256(out / "frozen_control.npz"),
        "control_digest": _digest(sigma, noise_scale, matching),
        "development_identity": _row_identity(bundle),
        "development_fit_digest": _digest(bundle["neural"], bundle["static"], bundle["dynamic"]),
        "fit_digest": _digest(gpfa.identity["fit_digest"], bundle["dynamic"][train],
            bundle["static"][validation], bundle["dynamic"][validation], bundle["neural"][validation]),
        "expected_scoring_identity": _row_identity(_identity_bundle(manifest, (*DEVELOPMENT, "test")))}
    identity["freeze_sha256"] = canonical_digest(identity)
    _json(out / "frozen_analysis.json", identity)
    _json(out / "analysis_config.json", config)
    return identity


def require_frozen(manifest: dict, config: dict, freeze_dir: Path) -> dict:
    """Check the persisted all-model lock before any oracle inference or scoring."""
    validate_retrained_manifest(manifest, config)
    path = Path(freeze_dir)
    if not (path / "frozen_analysis.json").exists():
        raise ValueError("Test remains pending: GPFA, encoders and validation choices are not all frozen")
    identity = json.loads((path / "frozen_analysis.json").read_text(encoding="utf-8"))
    unhashed = {k: v for k, v in identity.items() if k != "freeze_sha256"}
    if identity.get("freeze_sha256") != canonical_digest(unhashed):
        raise ValueError("The frozen analysis lock changed")
    if (identity.get("status") != "all_choices_frozen" or identity["config_sha256"] != canonical_digest(config)
            or identity["split_sha256"] != manifest["split_sha256"]):
        raise ValueError("The final analysis lock belongs to a different run")
    _check_checkpoints(identity["checkpoint_identities"], manifest, identity["preprocessing_sha256"])
    if file_sha256(path / "frozen_control.npz") != identity["control_sha256"]:
        raise ValueError("Frozen output perturbation changed")
    return identity


def evaluate_pilot(bundle: dict, manifest: dict, config: dict, gpfa_dir: Path,
                   freeze_dir: Path, output_dir: Path) -> dict:
    """Reuse existing metric/null mathematics on newly trained, frozen artifacts."""
    identity = require_frozen(manifest, config, freeze_dir)
    _validate_bundle(bundle, manifest, (*DEVELOPMENT, "test"), predictions=True)
    if _row_identity(bundle) != identity["expected_scoring_identity"]:
        raise ValueError("Scoring identities differ from the frozen manifest")
    gpfa = load_gpfa(gpfa_dir, manifest, config, identity["preprocessing_sha256"])
    if gpfa.identity != identity["gpfa_identity"]:
        raise ValueError("The GPFA differs from the final analysis lock")
    arrays, model = gpfa.arrays, gpfa.model
    with np.load(Path(freeze_dir) / "frozen_control.npz", allow_pickle=False) as source:
        noise_scale = source["noise_scale"]
    matching = identity["response_matching"]
    transform_digest = _digest(arrays["precision"], arrays["encoder_precision"], arrays["observations"],
        arrays["query_times"], arrays["neuron_ids"], arrays["frame_indices"], model.times_seconds, model.jitter, model.min_noise)
    cache_identity = {"schema_version": "sensorium_part3_retrained_cache_v1", "protocol": PROTOCOL,
        "split_sha256": manifest["split_sha256"], "config_sha256": canonical_digest(config),
        "preprocessing_sha256": identity["preprocessing_sha256"], "gpfa_sha256": gpfa.identity["gpfa_sha256"],
        "checkpoints": {k: v["sha256"] for k, v in identity["checkpoint_identities"].items()},
        "freeze_sha256": identity["freeze_sha256"]}
    frozen = FrozenAnalysis(model, arrays["precision"], arrays["encoder_precision"], arrays["observations"],
        arrays["query_times"], arrays["neuron_ids"], arrays["frame_indices"], model.parameter_digest(),
        identity["fit_digest"], matching["sigma"], noise_scale, matching, _digest(config),
        _input_identity(bundle), transform_digest, identity["control_digest"], cache_identity)
    out = _new_output(output_dir)
    result = score_frozen(bundle, frozen, config, out)
    result.update(protocol=PROTOCOL, status="held_out_evaluation_after_retraining_complete",
        evaluation_scope="one pilot session; historically explored oracle; retrained train/validation/test roles",
        freeze_sha256=identity["freeze_sha256"], checkpoints=cache_identity["checkpoints"])
    _json(out / "summary.json", result)
    _json(out / "generated_manifest.json", {"protocol": PROTOCOL, "status": result["status"],
        "cache_identity": cache_identity, "files": {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}})
    return result


def score_session_response(bundle: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All-neuron session response scores; condition is the paired bootstrap unit."""
    roles = np.asarray(bundle["roles"]).astype(str)
    if not np.all(roles == "test"):
        raise ValueError("All-session response scoring accepts test rows only")
    neural = np.asarray(bundle["neural"])
    predictions = {name: np.asarray(bundle[name]) for name in ("static", "dynamic")}
    if neural.ndim != 3 or any(v.shape != neural.shape for v in predictions.values()):
        raise ValueError("Response arrays must share trial/time/neuron axes")
    if not all(np.isfinite(v).all() for v in (neural, *predictions.values())):
        raise ValueError("Response evaluation requires common finite support")
    conditions = np.asarray(bundle["conditions"]).astype(str)
    session = str(np.asarray(bundle["session"]).item())
    def scores(mask):
        target = neural[mask].reshape(-1, neural.shape[-1])
        correlations = {name: column_correlation(target, value[mask].reshape(target.shape)) for name, value in predictions.items()}
        common = np.isfinite(np.stack(list(correlations.values()))).all(axis=0)
        variance = np.mean((target - target.mean()) ** 2)
        return {name: {"response_neuron_mean_r": _mean(value[common]),
            "response_nmse": float(np.mean((target - predictions[name][mask].reshape(target.shape)) ** 2) / variance) if variance > 1e-12 else np.nan}
            for name, value in correlations.items()}, int(common.sum())
    pooled, shared = scores(np.ones(len(neural), dtype=bool))
    rows, local_rows = [], []
    unique = np.unique(conditions)
    for name, metrics in pooled.items():
        for metric, value in metrics.items():
            rows.append({"session": session, "role": "test", "model": name, "metric": metric, "value": value,
                "n_trials": len(neural), "n_conditions": len(unique), "n_frames": neural.shape[1],
                "n_neurons": neural.shape[-1], "shared_r_neurons": shared, "aggregation": "pooled_trial_time_per_neuron"})
    for condition in unique:
        mask = conditions == condition
        local, count = scores(mask)
        for name, metrics in local.items():
            for metric, value in metrics.items():
                local_rows.append({"session": session, "condition": condition, "model": name, "metric": metric,
                    "value": value, "n_repeats": int(mask.sum()), "shared_r_neurons": count})
    return pd.DataFrame(rows), pd.DataFrame(local_rows)


def summarize_session_responses(pooled: pd.DataFrame, conditions: pd.DataFrame,
                                config: dict, output_dir: Path) -> dict:
    """Save paired movie intervals per session and the unweighted session mean.

    Conventional-score intervals concern equal-weight condition scores. The
    pooled trial/time point estimates are kept separately, as in pilot scoring.
    Five recording sessions are not presented as five independent animals.
    """
    out = _new_output(output_dir)
    sessions = sorted(pooled.session.unique())
    if not sessions or set(sessions) != set(conditions.session.unique()):
        raise ValueError("Pooled and condition response summaries have different sessions")
    rows = []
    rng = np.random.default_rng(int(config["project"]["seed"]))
    n_boot = int(config["traditional"]["paired_bootstrap_samples"])
    for session in sessions:
        local = conditions[conditions.session == session]
        for metric in sorted(local.metric.unique()):
            table = local[local.metric == metric].pivot(index="condition", columns="model", values="value")
            common = np.isfinite(table.static.to_numpy()) & np.isfinite(table.dynamic.to_numpy())
            deltas = table.dynamic.to_numpy() - table.static.to_numpy()
            draws = []
            for _ in range(n_boot):
                sample = rng.integers(0, len(table), size=len(table))
                valid = common[sample]
                draws.append(float(np.mean(deltas[sample][valid])) if valid.any() else np.nan)
            draws = np.asarray(draws)
            finite = np.isfinite(draws)
            interval = np.quantile(draws[finite], [.025, .975]) if finite.any() and len(table) >= 2 else [np.nan, np.nan]
            point = pooled[(pooled.session == session) & (pooled.metric == metric)].set_index("model").value
            rows.append({"session": session, "metric": metric, "static": _mean(table.static.to_numpy()[common]),
                "dynamic": _mean(table.dynamic.to_numpy()[common]), "dynamic_minus_static": _mean(deltas[common]),
                "pooled_static": point["static"], "pooled_dynamic": point["dynamic"],
                "pooled_dynamic_minus_static": point["dynamic"] - point["static"],
                "ci_low": interval[0], "ci_high": interval[1], "n_conditions": len(table),
                "defined_paired_conditions": int(common.sum()), "valid_bootstraps": int(finite.sum()),
                "undefined_bootstraps": int((~finite).sum()), "resampling_unit": "movie_condition_within_session",
                "interval_estimand": "mean_paired_condition_score", "biological_sample_count": "not_estimated"})
    pooled.to_csv(out / "response_session_metrics.csv", index=False)
    conditions.to_csv(out / "response_condition_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "response_session_paired_bootstrap.csv", index=False)
    summary = pooled.groupby(["model", "metric"], as_index=False).agg(
        value=("value", "mean"), n_sessions=("session", "nunique"), defined_sessions=("value", "count"))
    summary["aggregation"] = "unweighted_recording_session_mean"
    summary["biological_sample_count"] = "not_estimated"
    summary.to_csv(out / "response_session_summary.csv", index=False)
    result = {"status": "response_evaluation_complete", "sessions": sessions,
        "n_sessions": len(sessions), "summary": summary.to_dict("records"),
        "interval_unit": "movie_condition_within_session", "trajectory_sessions": 1,
        "scope": "recording sessions reported separately; no cross-session trajectory inference"}
    _json(out / "summary.json", result)
    return result
