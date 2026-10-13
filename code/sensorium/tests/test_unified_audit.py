"""A tiny file dataset checks inventory behavior without neural conclusions."""
import json

import numpy as np
import pytest

from trajectory_model_eval.unified_audit import audit_dataset
from trajectory_model_eval.unified_contract import manifest_digest, validate_manifest


@pytest.fixture
def data_files(tmp_path):
    session = tmp_path / "data/session"
    metadata = {"meta/trials/tiers.npy": np.array(["train", "oracle", "oracle", "live_test_main", "live_test_main"]),
                "meta/trials/trial_idx.npy": np.array([101, 103, 104, 106, 107]),
                "meta/neurons/unit_ids.npy": np.array([10, 20, 30]),
                "meta/statistics/responses/all/std.npy": np.ones((3, 300))}
    for relative, value in metadata.items():
        path = session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
    rng = np.random.default_rng(9)
    movies = [rng.normal(size=(8, 8, 300)) for _ in range(3)]
    for index in range(5):
        for modality in ("videos", "responses", "behavior", "pupil_center"):
            path = session / "data" / modality / f"{index}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            if modality == "videos":
                value = movies[0 if index == 0 else 1 if index < 3 else 2]
            elif modality == "responses":
                value = np.ones((3, 300)) if index < 3 else np.zeros((3, 300))
            else:
                value = np.ones((2, 300))
            np.save(path, value)
    config = {"project": {"seed": 42},
              "data": {"session": "session", "sessions": ["session"], "neurons": 2, "neuron_seed": 42,
                       "frame_start": 50, "frame_stop": 300, "stimulus_similarity": .999},
              "models": {name: {"trained_sessions": ["session"], "training_tiers": ["train"],
                                "selection_tiers": ["oracle"]} for name in ("static", "dynamic")}}
    checkpoints = {}
    for name in ("static", "dynamic"):
        checkpoints[name] = tmp_path / f"{name}.pt"
        checkpoints[name].write_bytes(f"synthetic {name} identity".encode())
    return tmp_path / "data", config, checkpoints


def test_zero_test_is_ineligible_and_roles_are_preserved(data_files, tmp_path):
    data, config, checkpoints = data_files
    manifest = audit_dataset(data, config, tmp_path / "out", checkpoints)
    assert manifest["eligibility"]["status"] == "blocked"
    assert "no_eligible_labeled_test_for_gpfa_session" in manifest["eligibility"]["reasons"]
    assert [row["role"] for row in manifest["trials"]] == ["train", "validation", "validation", "test", "test"]
    assert [row["trial_id"] for row in manifest["trials"]] == ["101", "103", "104", "106", "107"]
    assert all(row["response_status"] == "all_zero_placeholder" for row in manifest["trials"][3:])
    assert manifest["trials"][1]["movie_id"] == manifest["trials"][2]["movie_id"]
    assert manifest["trials"][0]["movie_id"] != manifest["trials"][1]["movie_id"]
    assert manifest["split_sha256"] == manifest_digest(manifest)
    text = (tmp_path / "out/split_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert json.loads(text)["preprocessing"]["status"] == "unverified"
    assert all(len(value) == 64 for value in manifest["trials"][0]["source_sha256"].values())
    assert not validate_manifest(manifest, require_test=False)["ready"]


def test_missing_modality_and_nonfinite_response_remain_invalid(data_files, tmp_path):
    data, config, checkpoints = data_files
    (data / "session/data/behavior/3.npy").unlink()
    values = np.ones((3, 300))
    values[2, 299] = np.nan
    np.save(data / "session/data/responses/4.npy", values)
    manifest = audit_dataset(data, config, tmp_path / "out", checkpoints)
    assert not manifest["trials"][3]["modalities_complete"]
    assert manifest["trials"][4]["response_status"] == "nonfinite"
    assert not manifest["trials"][4]["eligible_labeled_trial"]


def test_real_labels_do_not_certify_unknown_preprocessing(data_files, tmp_path):
    data, config, checkpoints = data_files
    np.save(data / "session/data/responses/3.npy", np.ones((3, 300)))
    manifest = audit_dataset(data, config, tmp_path / "out", checkpoints)
    assert manifest["eligibility"]["eligible_labeled_test_trials"] == 1
    assert "no_eligible_labeled_test_for_gpfa_session" not in manifest["eligibility"]["reasons"]
    assert "preprocessing_provenance_unverified" in manifest["eligibility"]["reasons"]
    assert manifest["eligibility"]["status"] == "blocked"


def test_configured_checkpoint_mismatch_is_rejected(data_files, tmp_path):
    data, config, checkpoints = data_files
    config["models"]["static"]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="static checkpoint SHA256"):
        audit_dataset(data, config, tmp_path / "out", checkpoints)


def test_movie_overlap_detected_across_original_roles(data_files, tmp_path):
    data, config, checkpoints = data_files
    np.save(data / "session/data/videos/3.npy", np.load(data / "session/data/videos/0.npy"))
    manifest = audit_dataset(data, config, tmp_path / "out", checkpoints)
    assert manifest["trials"][0]["movie_id"] == manifest["trials"][3]["movie_id"]
    assert "test_movies_overlap_development_movies" in manifest["eligibility"]["reasons"]


def test_padding_after_input_and_evaluation_support_is_allowed(data_files, tmp_path):
    data, config, checkpoints = data_files
    for modality in ("responses", "videos", "behavior", "pupil_center"):
        path = data / f"session/data/{modality}/0.npy"
        values = np.load(path)
        padding = np.full((*values.shape[:-1], 10), np.nan)
        np.save(path, np.concatenate([values, padding], axis=-1))
    manifest = audit_dataset(data, config, tmp_path / "out", checkpoints)
    row = manifest["trials"][0]
    assert row["response_status"] == "real_nonzero"
    assert row["time_support_valid"]
    assert row["response_finite_values"] < row["response_values"]
    assert row["evaluation_finite_values"] == row["evaluation_response_values"]


def test_nonfinite_input_prefix_is_invalid_and_movie_failure_is_traceable(data_files, tmp_path):
    data, config, checkpoints = data_files
    path = data / "session/data/videos/0.npy"
    values = np.load(path)
    values[0, 0, 4] = np.nan
    np.save(path, values)
    manifest = audit_dataset(data, config, tmp_path / "out", checkpoints)
    assert not manifest["trials"][0]["time_support_valid"]
    assert manifest["trials"][0]["movie_id"].startswith("unresolved:")
    assert "movie_identity_incomplete" in manifest["eligibility"]["reasons"]
    assert not validate_manifest(manifest, require_test=False)["ready"]


def test_replaced_response_changes_identity_even_with_same_value_counts(data_files, tmp_path):
    data, config, checkpoints = data_files
    before = audit_dataset(data, config, tmp_path / "before", checkpoints)
    np.save(data / "session/data/responses/0.npy", np.full((3, 300), 2.0))
    after = audit_dataset(data, config, tmp_path / "after", checkpoints)
    assert before["trials"][0]["response_nonzero_values"] == after["trials"][0]["response_nonzero_values"]
    assert before["trials"][0]["source_sha256"]["responses"] != after["trials"][0]["source_sha256"]["responses"]
    assert before["split_sha256"] != after["split_sha256"]
