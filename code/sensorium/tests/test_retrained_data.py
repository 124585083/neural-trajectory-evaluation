"""Identity and causal-dependence checks for the new Part 3 data adapter."""
import copy
import json

import numpy as np
import pytest

from trajectory_model_eval.retrained_data import (
    MODALITIES, RetrainedDataset, _overlap_edges, build_split, create_dataloaders,
    create_role_loader, fit_preprocessing, load_preprocessing, role_rows,
    validate_split,
)
from trajectory_model_eval.unified_contract import canonical_digest, file_sha256, manifest_digest


def fixture_inventory(tmp_path, two_sessions=False):
    rng = np.random.default_rng(100)
    sessions = ["s1", "s2"] if two_sessions else ["s1"]
    rows, metadata, neurons = [], [], {}
    for session in sessions:
        root = tmp_path / session
        ids = np.asarray(["a", "b", "c"])
        metadata_values = {"meta/neurons/unit_ids.npy": ids,
                           "meta/neurons/cell_motor_coordinates.npy": np.arange(9).reshape(3, 3),
                           "meta/trials/tiers.npy": np.asarray(["train"] * 10 + ["oracle"] * 4 + ["live_test_main"]),
                           "meta/trials/trial_idx.npy": np.arange(15)}
        for relative, value in metadata_values.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value)
            if "cell_motor" not in relative:
                metadata.append({"path": f"{session}/{relative}", "sha256": file_sha256(path)})
        neurons[session] = {"count_total": 3, "indices": [2, 0], "ids": ["c", "a"]}
        for i in range(15):
            files, hashes = {}, {}
            for modality in MODALITIES:
                shape = (8, 8, 300) if modality == "videos" else (3 if modality == "responses" else 2, 300)
                value = rng.uniform(.1, 2, size=shape).astype(np.float32)
                path = root / "data" / modality / f"{i}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, value)
                files[modality] = f"{session}/data/{modality}/{i}.npy"
                hashes[modality] = file_sha256(path)
            tier = "train" if i < 10 else "oracle" if i < 14 else "live_test_main"
            rows.append({"session": session, "dataset_index": i, "trial_id": str(i),
                         "movie_id": f"movie{i if i < 10 else 10 + i % 2}",
                         "source_tier": tier, "role": "historical", "eligible_labeled_trial": True,
                         "source_files": files, "source_sha256": hashes})
    return {"split_sha256": "a" * 64, "trials": rows, "sessions": sessions, "session": sessions[0],
            "metadata_files": metadata, "session_neurons": neurons, "neurons": neurons[sessions[0]],
            "movie_identity": {"similarity_threshold": .999},
            "time": {"frame_start": 50, "frame_stop": 300, "frame_rate_hz": 30., "observation_step": 4}}


def test_shared_movie_groups_and_original_tiers_are_preserved(tmp_path):
    prior = fixture_inventory(tmp_path, two_sessions=True)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    assert validate_split(manifest)
    for session in manifest["sessions"]:
        assert len(role_rows(manifest, session, "train")) == 8
        assert len(role_rows(manifest, session, "validation")) == 2
        assert len(role_rows(manifest, session, "test")) == 4
    assert [r["role"] for r in manifest["trials"] if r["session"] == "s1"] == [r["role"] for r in manifest["trials"] if r["session"] == "s2"]
    assert all(r["original_tier"] == r["source_tier"] for r in manifest["trials"])
    assert all(r["exclusion_reason"] == "hidden_test_zero_response_placeholder" for r in manifest["trials"] if r["role"] == "excluded")


def test_identified_train_test_movie_conflict_is_quarantined(tmp_path):
    prior = fixture_inventory(tmp_path)
    prior["trials"][0]["movie_id"] = prior["trials"][10]["movie_id"]
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    assert manifest["trials"][0]["role"] == "excluded"
    assert manifest["trials"][0]["exclusion_reason"] == "identified_stimulus_overlap_with_oracle_test"
    assert len(role_rows(manifest, "s1", "test")) == 4


def test_moments_preserve_axes_formula_and_nonnegative_response(tmp_path):
    prior = fixture_inventory(tmp_path)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    record = fit_preprocessing(tmp_path, manifest, tmp_path / "out")
    arrays = load_preprocessing(record, "s1", tmp_path / "out")
    train = role_rows(manifest, "s1", "train")
    for modality in MODALITIES:
        raw = np.stack([np.load(tmp_path / r["source_files"][modality]) for r in train]).astype(np.float64)
        np.testing.assert_allclose(arrays[f"{modality}_mean"], raw.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(arrays[f"{modality}_std"], raw.std(axis=0), rtol=1e-6)
    data = RetrainedDataset(tmp_path, manifest, record, "s1", "train", tmp_path / "out", offset=0)
    batch = data[0]
    assert batch.videos.shape == (3, 80, 8, 8)
    assert batch.responses.shape == (3, 80)
    assert batch.responses.min() >= 0
    assert len(data) == 8
    assert data.original_dataset_length == 15
    with pytest.raises(RuntimeError, match="forbidden"):
        _ = data.statistics


def test_changing_validation_and_test_cannot_change_training_statistics(tmp_path):
    prior = fixture_inventory(tmp_path)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    first = fit_preprocessing(tmp_path, manifest, tmp_path / "first")
    a = load_preprocessing(first, "s1", tmp_path / "first")
    for row in manifest["trials"]:
        if row["role"] in ("validation", "test"):
            for modality in MODALITIES:
                path = tmp_path / row["source_files"][modality]
                np.save(path, np.load(path) * 17 + 123)
    second = fit_preprocessing(tmp_path, manifest, tmp_path / "second")
    b = load_preprocessing(second, "s1", tmp_path / "second")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert first["sessions"]["s1"]["source_sha256"] == second["sessions"]["s1"]["source_sha256"]


def test_actual_training_closure_ids_exclude_oracle_test(tmp_path):
    prior = fixture_inventory(tmp_path)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    fit_preprocessing(tmp_path, manifest, tmp_path / "out")
    preprocessing = tmp_path / "out/preprocessing.json"
    loaders = create_dataloaders(tmp_path, manifest, preprocessing, batch_size=2)
    assert set(loaders) == {"train", "oracle"}
    assert loaders["oracle"]["s1"].dataset.role == "validation"
    test_ids = {r["dataset_index"] for r in role_rows(manifest, "s1", "test")}
    for key in loaders:
        assert not test_ids.intersection(loaders[key]["s1"].dataset.dataset_indices)
        assert next(iter(loaders[key]["s1"])).videos.shape == (2, 3, 80, 8, 8)
    full = create_role_loader(tmp_path, manifest, preprocessing, "s1", "test")
    assert next(iter(full)).responses.shape == (1, 3, 300)
    assert full.dataset.dataset_indices == [10, 11, 12, 13]


def test_modified_normalization_or_split_fails_closed(tmp_path):
    prior = fixture_inventory(tmp_path)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    record = fit_preprocessing(tmp_path, manifest, tmp_path / "out")
    changed = copy.deepcopy(manifest)
    changed["trials"][0]["role"] = "test"
    with pytest.raises(ValueError, match="digest"):
        validate_split(changed)
    path = tmp_path / "out" / record["sessions"]["s1"]["file"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        load_preprocessing(record, "s1", tmp_path / "out")


def test_shifted_stimulus_overlap_detected_without_neural_inputs(tmp_path):
    rng = np.random.default_rng(8)
    original = rng.normal(size=(8, 8, 360)).astype(np.float32)
    rows = []
    for index, offset in enumerate((0, 60)):
        path = tmp_path / f"{index}.npy"
        np.save(path, original[..., offset:offset + 300])
        rows.append({"source_files": {"videos": path.name}, "source_sha256": {"videos": file_sha256(path)},
                     "movie_id": f"movie{index}", "session": f"session{index}"})
    edges, audit = _overlap_edges(tmp_path, rows)
    assert len(edges) == 1
    assert edges[0]["lag_frames"] == -60
    assert edges[0]["common_frames"] == 240
    assert edges[0]["similarity"] > .999999


def test_adapter_matches_original_transform_math_with_replacement_statistics(tmp_path):
    pytest.importorskip("neuralpredictors")
    from neuralpredictors.data.transforms import (
        AddBehaviorAsChannels, ChangeChannelsOrder, ExpandChannels,
        NeuroNormalizer, ScaleInputs, Subsequence, ToTensor,
    )
    from types import SimpleNamespace
    from trajectory_model_eval.retrained_data import Batch
    prior = fixture_inventory(tmp_path)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    record = fit_preprocessing(tmp_path, manifest, tmp_path / "out")
    stats = load_preprocessing(record, "s1", tmp_path / "out")
    source = SimpleNamespace(statistics={key: {"train": {"mean": stats[f"{key}_mean"], "std": stats[f"{key}_std"]}} for key in MODALITIES},
                             data_keys=list(MODALITIES), _len=8)
    transforms = [NeuroNormalizer(source, stats_source="train", in_name="videos"),
                  ChangeChannelsOrder((2, 0, 1), in_name="videos"),
                  ChangeChannelsOrder((1, 0), in_name="responses"),
                  ChangeChannelsOrder((1, 0), in_name="behavior"),
                  ChangeChannelsOrder((1, 0), in_name="pupil_center"),
                  Subsequence(frames=80, channel_first=(), offset=17),
                  ChangeChannelsOrder((1, 0), in_name="responses"), ExpandChannels("videos"),
                  ChangeChannelsOrder((1, 0), in_name="behavior"),
                  ChangeChannelsOrder((1, 0), in_name="pupil_center"),
                  AddBehaviorAsChannels("videos"), ScaleInputs(scale=1, in_name="videos", channel_axis=-1),
                  ToTensor(False)]
    dataset = RetrainedDataset(tmp_path, manifest, record, "s1", "train", tmp_path / "out", offset=17)
    row = dataset.rows[0]
    value = Batch(*(np.load(tmp_path / row["source_files"][key]) for key in MODALITIES))
    for transform in transforms:
        value = transform(value)
    expected = dataset[0]
    for left, right in zip(value, expected):
        np.testing.assert_allclose(left.numpy(), right.numpy(), atol=2e-6, rtol=2e-6)


def test_small_negative_source_values_are_recorded_and_not_clipped(tmp_path):
    prior = fixture_inventory(tmp_path)
    manifest = build_split(tmp_path, prior, tmp_path / "out", check_overlaps=False)
    row = role_rows(manifest, "s1", "train")[0]
    path = tmp_path / row["source_files"]["responses"]
    values = np.load(path)
    values[0, 0] = -1e-7
    np.save(path, values)
    row["source_sha256"]["responses"] = file_sha256(path)
    manifest["split_sha256"] = manifest_digest(manifest)
    record = fit_preprocessing(tmp_path, manifest, tmp_path / "out")
    assert record["sessions"]["s1"]["source_response_negative_values"] == 1
    assert record["sessions"]["s1"]["source_response_minimum"] < 0
    dataset = RetrainedDataset(tmp_path, manifest, record, "s1", "train", tmp_path / "out", offset=0)
    np.testing.assert_allclose(dataset[0].responses[0, 0].item(),
                               values[0, 0] * dataset.statistics_arrays["response_precision"][0, 0])
    assert dataset[0].responses[0, 0].item() < 0
