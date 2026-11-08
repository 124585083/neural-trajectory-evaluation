"""Export full-sequence predictions from the new, completed Part 3 encoders."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from .unified_contract import canonical_digest, file_sha256
from .unified_predictions import predict_loader


def _load(value):
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def bundle_identity(config, manifest, preprocessing, checkpoint_identities, session, roles, neurons,
                    *, frozen_analysis=None):
    """Include the exact sources and roles, not just filenames or tensor shapes."""
    rows = [r for r in manifest["trials"] if r["session"] == session and r["role"] in roles]
    identity = {"schema_version": "sensorium_retrained_predictions_v1",
            "config_sha256": canonical_digest(config), "split_sha256": manifest["split_sha256"],
            "preprocessing_sha256": preprocessing["preprocessing_sha256"],
            "checkpoint_sha256": {key: value["sha256"] for key, value in checkpoint_identities.items()},
            "session": session, "roles": list(roles), "dataset_indices": [r["dataset_index"] for r in rows],
            "source_sha256": canonical_digest([r["source_sha256"] for r in rows]),
            "neuron_indices": list(map(int, neurons)),
            "frame_indices": list(range(config["data"]["frame_start"], config["data"]["frame_stop"]))}
    if "test" in roles:
        if frozen_analysis is None or frozen_analysis.get("status") != "all_choices_frozen":
            raise ValueError("Test cache identity requires the completed analysis freeze")
        if (frozen_analysis["split_sha256"] != manifest["split_sha256"]
                or frozen_analysis["config_sha256"] != canonical_digest(config)
                or frozen_analysis["preprocessing_sha256"] != preprocessing["preprocessing_sha256"]
                or frozen_analysis["checkpoint_identities"] != checkpoint_identities):
            raise ValueError("Test cache inputs differ from the frozen analysis")
        identity.update(freeze_sha256=frozen_analysis["freeze_sha256"],
                        gpfa_sha256=frozen_analysis["gpfa_identity"]["gpfa_sha256"])
    return identity


def verify_source_files(manifest, data_root, *, roles, sessions):
    """Reject stale source snapshots before exporting or scoring cached targets."""
    root = Path(data_root)
    for row in manifest["trials"]:
        if row["session"] not in sessions or row["role"] not in roles:
            continue
        for modality, relative in row["source_files"].items():
            if file_sha256(root / relative) != row["source_sha256"][modality]:
                raise ValueError(f"Source {row['session']}:{row['dataset_index']}:{modality} changed; cached targets are stale")


def verify_test_lock(config, manifest, preprocessing, encoder_root, freeze_dir, gpfa_dir=None):
    """Verify actual model files as well as their frozen metadata before test."""
    from .retrained_analysis import require_frozen

    if freeze_dir is None:
        raise ValueError("Test export requires a completed analysis freeze")
    frozen = require_frozen(manifest, config, freeze_dir)
    if frozen["preprocessing_sha256"] != preprocessing["preprocessing_sha256"]:
        raise ValueError("Test preprocessing differs from the frozen analysis")
    for kind, expected in frozen["checkpoint_identities"].items():
        directory = Path(encoder_root) / kind
        actual = _load(directory / "frozen_identity.json")
        if actual != expected:
            raise ValueError("Encoder identity changed after freezing")
        if file_sha256(directory / actual["checkpoint"]) != actual["sha256"]:
            raise ValueError("Actual encoder checkpoint changed after freezing")
    # Export needs the exact frozen bytes, but no GPFA inference. Keeping this
    # check independent of unpickling preserves the separate encoding runtime.
    directory = Path(gpfa_dir or Path(freeze_dir).parent / "gpfa")
    gpfa_identity = _load(directory / "frozen_gpfa_identity.json")
    if gpfa_identity != frozen["gpfa_identity"]:
        raise ValueError("Actual GPFA artifact changed after freezing")
    if (file_sha256(directory / "frozen_gpfa.pkl") != gpfa_identity["gpfa_sha256"]
            or file_sha256(directory / "frozen_gpfa_arrays.npz") != gpfa_identity["arrays_sha256"]):
        raise ValueError("Actual GPFA payload changed after freezing")
    return frozen


def save_bundle(path, bundle, identity):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Prediction artifacts are immutable; use a separate output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)
    record = {"identity": identity, "identity_sha256": canonical_digest(identity),
              "payload_sha256": file_sha256(path), "shape": list(bundle["neural"].shape),
              "parameters_and_batchnorm_unchanged": True, "full_sequence_frames": 300,
              "temporal_reduction_frames": 18, "inference": "eval with inference_mode"}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_bundle(path, expected_identity=None):
    path = Path(path)
    record = _load(path.with_suffix(".json"))
    if record["identity_sha256"] != canonical_digest(record["identity"]):
        raise ValueError("Prediction identity record was changed")
    if expected_identity is not None and record["identity"] != expected_identity:
        raise ValueError("Prediction cache belongs to different sources, roles, models or preprocessing")
    if record["payload_sha256"] != file_sha256(path):
        raise ValueError("Prediction payload changed; do not reuse cached scores")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def combine_pilot(development, test, manifest):
    """Order disjoint development/test exports by the frozen manifest."""
    per_trial = ("neural", "static", "dynamic", "roles", "conditions", "movie_ids", "dataset_indices", "trial_ids")
    if ("matched_dynamic" in development) != ("matched_dynamic" in test):
        raise ValueError("Matched checkpoint prediction is missing from one export")
    if "matched_dynamic" in development:
        per_trial += ("matched_dynamic",)
    fixed = ("session", "encoder_precision", "neuron_indices", "neuron_ids", "frame_indices")
    for key in fixed:
        if not np.array_equal(development[key], test[key]):
            raise ValueError(f"Development/test {key} differs")
    if set(np.asarray(development["roles"]).astype(str)) != {"train", "validation"}:
        raise ValueError("Development export must contain only train and validation")
    if set(np.asarray(test["roles"]).astype(str)) != {"test"}:
        raise ValueError("Test export contains a development row")
    result = {key: np.asarray(development[key]).copy() for key in fixed}
    for key in per_trial:
        result[key] = np.concatenate((development[key], test[key]))
    indices = result["dataset_indices"].tolist()
    expected = [r["dataset_index"] for r in manifest["trials"]
                if r["session"] == manifest["session"] and r["role"] != "excluded"]
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError("Combined pilot export has missing or duplicate parent trials")
    order = np.asarray([indices.index(index) for index in expected])
    for key in per_trial:
        result[key] = result[key][order]
    return result


def select_neurons(bundle, indices):
    """Use the fixed pilot subset without fitting or selecting on responses."""
    available = list(map(int, bundle["neuron_indices"]))
    positions = np.asarray([available.index(int(index)) for index in indices])
    result = dict(bundle)
    for key in ("neural", "static", "dynamic", "matched_dynamic"):
        if key in bundle:
            result[key] = bundle[key][..., positions]
    result["encoder_precision"] = bundle["encoder_precision"][..., positions]
    result["neuron_indices"] = bundle["neuron_indices"][positions]
    result["neuron_ids"] = bundle["neuron_ids"][positions]
    return result


def export_predictions(config, manifest, preprocessing_path, data_root, encoder_root,
                       output_root, *, roles, sessions, pilot_only=False, device="cuda",
                       freeze_dir=None, gpfa_dir=None):
    """Export identical target/input support; source files are checked before use."""
    import torch
    from .retrained_data import create_dataloaders, create_role_loader, load_preprocessing, validate_split
    from .retrained_training import load_frozen_encoder

    if tuple(roles) not in (("train", "validation"), ("test",)):
        raise ValueError("Export development and test in separate stages")
    manifest, preprocessing = _load(manifest), _load(preprocessing_path)
    validate_split(manifest)
    data_root, output_root, encoder_root = Path(data_root), Path(output_root), Path(encoder_root)
    frozen_analysis = (verify_test_lock(config, manifest, preprocessing, encoder_root, freeze_dir, gpfa_dir)
                       if "test" in roles else None)
    if pilot_only and list(sessions) != [manifest["session"]]:
        raise ValueError("The detailed subset belongs to the pilot session only")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Prediction destination is not empty")
    train_loaders = create_dataloaders(data_root, manifest, preprocessing_path)["train"]
    models, frozen = {}, {}
    for kind in ("static", "dynamic"):
        models[kind], frozen[kind] = load_frozen_encoder(
            kind, train_loaders, encoder_root / kind, config=config, manifest=manifest,
            preprocessing=preprocessing, device=device)
    summaries = {}
    for session in sessions:
        rows = [row for row in manifest["trials"] if row["session"] == session and row["role"] in roles]
        verify_source_files(manifest, data_root, roles=roles, sessions=[session])
        neurons = np.asarray(manifest["neurons"]["indices"] if pilot_only else
                             manifest["session_neurons"][session]["all_indices"], dtype=int)
        all_ids = np.asarray(manifest["session_neurons"][session]["all_ids"])
        start, stop = config["data"]["frame_start"], config["data"]["frame_stop"]
        statistics = load_preprocessing(preprocessing_path, session)
        precision = statistics["response_precision"][neurons, start:stop].T.astype(np.float64)
        chunks, seen_rows = {key: [] for key in ("neural", "static", "dynamic")}, []
        for role in roles:
            loader = create_role_loader(data_root, manifest, preprocessing_path, session, role, batch_size=1)
            seen_rows.extend(loader.dataset.rows)
            target = None
            for kind in ("static", "dynamic"):
                prediction, actual = predict_loader(models[kind], loader, session, neurons, start, stop, device)
                if target is not None and not np.array_equal(actual, target):
                    raise ValueError("Neural targets differ between frozen encoder passes")
                target = actual
                chunks[kind].append(prediction)
            chunks["neural"].append(target)
        seen = [row["dataset_index"] for row in seen_rows]
        order = np.asarray([seen.index(row["dataset_index"]) for row in rows])
        bundle = {key: np.concatenate(values)[order] for key, values in chunks.items()}
        bundle.update(session=np.asarray(session), encoder_precision=precision,
                      dataset_indices=np.asarray([row["dataset_index"] for row in rows]),
                      trial_ids=np.asarray([row["trial_id"] for row in rows]),
                      conditions=np.asarray([row["movie_id"] for row in rows]),
                      movie_ids=np.asarray([row["movie_id"] for row in rows]),
                      roles=np.asarray([row["role"] for row in rows]), neuron_indices=neurons,
                      neuron_ids=all_ids[neurons], frame_indices=np.arange(start, stop))
        for i, row in enumerate(rows):
            raw = np.load(data_root / row["source_files"]["responses"], mmap_mode="r")[neurons, start:stop].T
            if not np.allclose(bundle["neural"][i], raw * precision, rtol=3e-6, atol=3e-6):
                raise ValueError("Neural export does not match raw responses times the frozen train precision")
        identity = bundle_identity(config, manifest, preprocessing, frozen, session, roles, neurons,
                                   frozen_analysis=frozen_analysis)
        summaries[session] = save_bundle(output_root / session / "aligned_predictions.npz", bundle, identity)
        print(json.dumps({"exported_session": session, "roles": roles, "shape": list(bundle["neural"].shape)}), flush=True)
        del bundle, chunks
        gc.collect()
    del models, train_loaders
    if device == "cuda":
        torch.cuda.empty_cache()
    return summaries


def attach_matched_prediction(reference, matched):
    """Attach a sidecar only when all target, trial and measurement axes agree."""
    from .checkpoint_matching import IDENTITY_KEYS
    for key in (*IDENTITY_KEYS, "neural"):
        if key not in matched or not np.array_equal(reference[key], matched[key]):
            raise ValueError(f"Matched checkpoint {key} differs from the reference bundle")
    prediction = np.asarray(matched["matched_dynamic"])
    if prediction.shape != np.asarray(reference["neural"]).shape or not np.isfinite(prediction).all():
        raise ValueError("Matched checkpoint prediction lacks complete finite support")
    if "matched_dynamic" in reference and not np.array_equal(reference["matched_dynamic"], prediction):
        raise ValueError("An existing matched checkpoint prediction cannot be replaced")
    return {**reference, "matched_dynamic": prediction}


def matched_bundle_identity(selection, bundle):
    """Keep the checkpoint-control identity separate from original main caches."""
    from .checkpoint_matching import IDENTITY_KEYS, array_digest, validate_selection
    validate_selection(selection)
    roles = tuple(dict.fromkeys(np.asarray(bundle["roles"]).astype(str)))
    if set(roles) not in ({"train", "validation"}, {"validation"}, {"test"}):
        raise ValueError("Matched exports must separate development from test")
    return {"schema_version": "sensorium_matched_checkpoint_predictions_v1",
            "selection_sha256": selection["selection_sha256"],
            "selected_checkpoint": selection["selected_checkpoint"],
            "source_identity": selection["source_identity"],
            "roles": list(roles), "session": str(np.asarray(bundle["session"]).item()),
            "support_sha256": canonical_digest({key: array_digest(bundle[key]) for key in (*IDENTITY_KEYS, "neural")})}


def save_matched_bundle(path, bundle, selection):
    """Archive untouched model output together with the frozen control identity."""
    attach_matched_prediction(bundle, bundle)
    return save_bundle(path, bundle, matched_bundle_identity(selection, bundle))


def load_matched_bundle(path, selection, reference_bundle=None):
    bundle = load_bundle(path)
    record = _load(Path(path).with_suffix(".json"))
    if record["identity"] != matched_bundle_identity(selection, bundle):
        raise ValueError("Matched prediction cache belongs to a different selection or support")
    attach_matched_prediction(bundle if reference_bundle is None else reference_bundle, bundle)
    return bundle


def load_candidate_encoder(checkpoint_identity, train_loaders, *, config, manifest,
                           preprocessing, device="cuda", model=None, checkpoint_root=None):
    """Load a recovered complete-epoch state without changing the best encoder."""
    import torch
    from .checkpoint_matching import _checkpoint_identity, checkpoint_path
    from .retrained_training import build_encoder, load_architecture_config, state_digest
    identity = _checkpoint_identity(checkpoint_identity)
    path = checkpoint_path(identity, checkpoint_root)
    if file_sha256(path) != identity["sha256"]:
        raise ValueError("Candidate checkpoint payload changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training = payload.get("identity", {})
    expected = {"protocol_sha256": canonical_digest(config), "split_sha256": manifest["split_sha256"],
                "preprocessing_sha256": preprocessing["preprocessing_sha256"], "architecture": "dynamic",
                "seed": 42}
    if (payload.get("checkpoint_kind") != "part3_epoch_candidate" or payload.get("epoch") != identity["epoch"]
            or any(training.get(key) != value for key, value in expected.items())
            or training.get("scope") not in {"formal_training", "checkpoint_candidate_replay"}
            or payload.get("identity_digest") != canonical_digest(training)
            or payload.get("identity_digest") != identity["training_identity_digest"]):
        raise ValueError("Candidate is not the declared complete-epoch Dynamic training state")
    if identity.get("identity") is not None and identity["identity"] != training:
        raise ValueError("Candidate manifest training identity differs from the checkpoint payload")
    if training.get("scope") == "checkpoint_candidate_replay" and not training.get("source_training_identity_digest"):
        raise ValueError("Replay checkpoint must identify its original source training run")
    architecture = load_architecture_config("dynamic")
    if training.get("architecture_sha256") != architecture["_architecture_sha256"]:
        raise ValueError("Candidate architecture differs from the frozen Dynamic architecture")
    if state_digest(payload["state_dict"]) != identity["state_sha256"]:
        raise ValueError("Candidate state differs from the archived state identity")
    if model is None:
        model, _, _ = build_encoder("dynamic", train_loaders)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if state_digest(model.state_dict()) != identity["state_sha256"]:
        raise ValueError("Loaded candidate state differs from its archived identity")
    return model


def predict_checkpoint_bundle(config, manifest, preprocessing_path, data_root, checkpoint_identity,
                              *, roles, device="cuda", selection=None, train_loaders=None, model=None,
                              verify_sources=True, checkpoint_root=None):
    """Predict fixed pilot neurons; test is gated by the validation-only selection.

    Returns ``(bundle, model)`` so candidate scans can reuse one architecture.
    Validation-only calls require no selection; train/test exports require the
    completed selection and the exact selected checkpoint.
    """
    from .checkpoint_matching import validate_selection
    from .retrained_data import create_dataloaders, create_role_loader, load_preprocessing, validate_split
    roles = tuple(roles)
    if roles not in (("validation",), ("train", "validation"), ("test",)):
        raise ValueError("Export validation candidates, selected development and test separately")
    manifest, preprocessing = _load(manifest), _load(preprocessing_path)
    validate_split(manifest)
    if roles != ("validation",) and selection is None:
        raise ValueError("Selected checkpoint must be frozen before development/test export")
    if selection is not None:
        validate_selection(selection, verify_checkpoint=True, checkpoint_root=checkpoint_root)
        if checkpoint_identity != selection["selected_checkpoint"]:
            raise ValueError("Export checkpoint differs from the frozen selection")
        source = selection["source_identity"]
        if (source["source_config_sha256"] != canonical_digest(config)
                or source["split_sha256"] != manifest["split_sha256"]
                or source["preprocessing_sha256"] != preprocessing["preprocessing_sha256"]):
            raise ValueError("Selected export sources differ from the frozen selection")
    session = manifest["session"]
    if verify_sources:
        verify_source_files(manifest, data_root, roles=roles, sessions=[session])
    if train_loaders is None:
        train_loaders = create_dataloaders(data_root, manifest, preprocessing_path)["train"]
    model = load_candidate_encoder(checkpoint_identity, train_loaders, config=config, manifest=manifest,
                                   preprocessing=preprocessing, device=device, model=model, checkpoint_root=checkpoint_root)
    neurons = np.asarray(manifest["neurons"]["indices"], dtype=int)
    start, stop = config["data"]["frame_start"], config["data"]["frame_stop"]
    precision = load_preprocessing(preprocessing_path, session)["response_precision"][neurons, start:stop].T.astype(np.float64)
    rows = [row for row in manifest["trials"] if row["session"] == session and row["role"] in roles]
    seen_rows, predictions, targets = [], [], []
    for role in roles:
        loader = create_role_loader(data_root, manifest, preprocessing_path, session, role, batch_size=1)
        prediction, target = predict_loader(model, loader, session, neurons, start, stop, device)
        seen_rows.extend(loader.dataset.rows)
        predictions.append(prediction)
        targets.append(target)
    seen = [row["dataset_index"] for row in seen_rows]
    if len(seen) != len(set(seen)) or set(seen) != {row["dataset_index"] for row in rows}:
        raise ValueError("Candidate loader did not cover the exact manifest trials")
    order = np.asarray([seen.index(row["dataset_index"]) for row in rows])
    bundle = {"matched_dynamic": np.concatenate(predictions)[order], "neural": np.concatenate(targets)[order],
              "session": np.asarray(session), "encoder_precision": precision,
              "dataset_indices": np.asarray([row["dataset_index"] for row in rows]),
              "trial_ids": np.asarray([row["trial_id"] for row in rows]),
              "conditions": np.asarray([row["movie_id"] for row in rows]),
              "movie_ids": np.asarray([row["movie_id"] for row in rows]),
              "roles": np.asarray([row["role"] for row in rows]), "neuron_indices": neurons,
              "neuron_ids": np.asarray(manifest["session_neurons"][session]["all_ids"])[neurons],
              "frame_indices": np.arange(start, stop)}
    for index, row in enumerate(rows):
        raw = np.load(Path(data_root) / row["source_files"]["responses"], mmap_mode="r")[neurons, start:stop].T
        if not np.allclose(bundle["neural"][index], raw * precision, rtol=3e-6, atol=3e-6):
            raise ValueError("Candidate targets differ from source responses and training precision")
    return bundle, model
