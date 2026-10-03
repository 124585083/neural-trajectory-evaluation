"""Protocol identity checks use small synthetic files, never scientific results."""

import copy
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from trajectory_model_eval import protocol


@pytest.fixture
def locked_inputs(tmp_path, monkeypatch):
    session = tmp_path / "data/session"
    tiers = np.array(["train"] * 4 + ["oracle"] * 2)
    ids = np.array(["unit_a", "unit_b", "unit_c"])
    meta = types.SimpleNamespace(session_id="session", unit_ids=ids, indices=lambda tier: np.flatnonzero(tiers == tier))
    module = types.ModuleType("trajectory_reliability.data")
    module.load_session_metadata = lambda path: meta
    module.deterministic_neuron_order = lambda metadata, seed: np.arange(3)
    module.discover_oracle_conditions = lambda metadata, threshold: [types.SimpleNamespace(dataset_indices=(4, 5))]
    monkeypatch.setitem(sys.modules, "trajectory_reliability.data", module)
    for relative in ["meta/trials/tiers.npy", "meta/neurons/unit_ids.npy"] + [
        f"data/{group}/{i}.npy" for i in range(6)
        for group in ("responses", "videos", "behavior", "pupil_center")
    ]:
        path = session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.arange(6))
    phase2 = tmp_path / "phase2/src/trajectory_reliability"
    phase2.mkdir(parents=True)
    for name in ("gpfa.py", "selection.py"):
        (phase2 / name).write_text("# Synthetic source identity\n", encoding="utf-8")
    (tmp_path / "model.pt").write_bytes(b"synthetic checkpoint identity only")
    (tmp_path / "model.yaml").write_text("data:\n  root: old/path\nmodel:\n  width: 8\n", encoding="utf-8")
    config = {
        "project": {"seed": 42, "output_dir": str(tmp_path / "out")},
        "references": {"phase2_root": str(tmp_path / "phase2"), "static_checkpoint": str(tmp_path / "model.pt"), "static_config": str(tmp_path / "model.yaml")},
        "data": {"sensorium_root": str(tmp_path / "data"), "session": "session", "neuron_seed": 42, "neurons": 2,
                 "gpfa_train_fraction": 1.0, "gpfa_calibration_fraction": .25, "oracle_stimulus_similarity": .999, "frame_start": 50, "frame_stop": 300},
        "gpfa": {"latent_dim": 4, "tolerance": 1e-6},
    }
    protocol.prepare_protocol(config)
    return config, meta


def test_matching_lock_is_read_only(locked_inputs):
    config, _ = locked_inputs
    out = Path(config["project"]["output_dir"])
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    assert len(protocol.load_protocol(config)["oracle_indices"]) == 2
    protocol.prepare_protocol(config)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}


@pytest.mark.parametrize("section,key,value", [("project", "seed", 43), ("data", "frame_start", 51), ("gpfa", "latent_dim", 8), ("gpfa", "tolerance", 1e-5)])
def test_scientific_setting_mismatch_is_precise(locked_inputs, section, key, value):
    config, _ = locked_inputs
    changed = copy.deepcopy(config)
    changed[section][key] = value
    with pytest.raises(protocol.ProtocolMismatchError, match=f"scientific_config.{section}.{key}"):
        protocol.load_protocol(changed)


def test_changed_neuron_order_is_rejected(locked_inputs):
    config, meta = locked_inputs
    meta.unit_ids = meta.unit_ids[::-1]
    with pytest.raises(protocol.ProtocolMismatchError, match="array_digests.neuron_ids"):
        protocol.load_protocol(config)


def test_changed_response_bytes_are_rejected(locked_inputs):
    config, _ = locked_inputs
    path = Path(config["data"]["sensorium_root"]) / "session/data/responses/4.npy"
    np.save(path, np.ones(6))
    with pytest.raises(protocol.ProtocolMismatchError, match="data_files_sha256.data/responses/4.npy"):
        protocol.load_protocol(config)


def test_changed_checkpoint_is_rejected(locked_inputs):
    config, _ = locked_inputs
    Path(config["references"]["static_checkpoint"]).write_bytes(b"different checkpoint")
    with pytest.raises(protocol.ProtocolMismatchError, match="source_identity.static_checkpoint"):
        protocol.load_protocol(config)


def test_path_aliases_and_timestamps_do_not_change_identity(locked_inputs, tmp_path):
    config, _ = locked_inputs
    moved = tmp_path / "relocated with spaces"
    shutil.copytree(tmp_path / "data", moved / "data")
    shutil.copytree(tmp_path / "phase2", moved / "phase2")
    shutil.copytree(tmp_path / "out", moved / "out")
    shutil.copy(tmp_path / "model.pt", moved / "renamed.pt")
    (moved / "renamed.yaml").write_text("data:\n  root: migrated/path\nmodel:\n  width: 8\n", encoding="utf-8")
    changed = copy.deepcopy(config)
    changed["data"]["sensorium_root"] = str(moved / "data")
    changed["project"]["output_dir"] = str(moved / "out")
    changed["project"]["timestamp"] = "a new run time"
    changed["references"] = {"phase2_root": str(moved / "phase2"), "static_checkpoint": str(moved / "renamed.pt"), "static_config": str(moved / "renamed.yaml")}
    protocol.load_protocol(changed)


def test_changed_lock_arrays_are_rejected(locked_inputs):
    config, _ = locked_inputs
    arrays = protocol.load_protocol(config)
    arrays["oracle_indices"][0] = 3
    np.savez_compressed(Path(config["project"]["output_dir"]) / "protocol_lock.npz", **arrays)
    with pytest.raises(protocol.ProtocolMismatchError, match="lock_arrays.oracle_indices"):
        protocol.load_protocol(config)


def test_legacy_lock_requires_explicit_diagnostic_override(locked_inputs):
    config, _ = locked_inputs
    path = Path(config["project"]["output_dir"]) / "protocol_lock.json"
    saved = json.loads(path.read_text())
    saved.pop("validation")
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    with pytest.raises(protocol.ProtocolMismatchError, match="legacy-unverified"):
        protocol.load_protocol(config)
    config["protocol_guard"] = {"allow_legacy_unverified": True}
    with pytest.warns(RuntimeWarning, match="legacy-unverified"):
        protocol.load_protocol(config)
    assert path.read_bytes() == before


def test_missing_lock_is_not_created(locked_inputs, tmp_path):
    config, _ = locked_inputs
    config["project"]["output_dir"] = str(tmp_path / "new")
    with pytest.raises(FileNotFoundError, match="Run 'lock' explicitly"):
        protocol.load_protocol(config)
    assert not (tmp_path / "new/protocol_lock.npz").exists()


def test_new_lock_refuses_a_directory_with_existing_results(locked_inputs, tmp_path):
    config, _ = locked_inputs
    output = tmp_path / "existing_results"
    output.mkdir()
    (output / "scores.csv").write_text("saved historical results\n")
    config["project"]["output_dir"] = str(output)
    with pytest.raises(FileExistsError, match="empty output directory"):
        protocol.prepare_protocol(config)
    assert list(output.iterdir()) == [output / "scores.csv"]
