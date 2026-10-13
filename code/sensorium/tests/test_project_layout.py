"""Storage-only migration checks; no model fitting, inference, or data download."""
import hashlib
import importlib
import json
from pathlib import Path
import tomllib

import pytest

from sensorium_runtime.paths import PROJECT_ROOT, SOURCE_ROOT, repository_path, resolve_path
from trajectory_model_eval.common import load_config, resolve
from trajectory_model_eval.retrained_training import load_architecture_config
from trajectory_model_eval.unified_audit import _checkpoint_identities
from trajectory_model_eval.unified_contract import canonical_digest


@pytest.mark.parametrize("old,new", [
    ("experiments/01_baselines/records/static/training_config.yaml", "records/baselines/static/training_config.yaml"),
    ("../01_baselines/configs/static_dynamic_sensorium2023.yaml", "configs/static_dynamic_sensorium2023.yaml"),
    ("experiments/03_parameter_matching/records/training_summary.json", "records/parameter_matching/training_summary.json"),
    ("../03_parameter_matching/configs/dynamic_parameter_matched.yaml", "configs/dynamic_parameter_matched.yaml"),
    ("experiments/04_model_comparison/configs/retrained_v1.json", "configs/retrained_v1.json"),
    ("experiments/02_gpfa_reliability/src/trajectory_reliability/gpfa.py", "src/trajectory_reliability/gpfa.py"),
    ("D:/old-checkout/experiments/04_model_comparison/outputs/retrained_v1/encoders/static/best.pt", "outputs/retrained_v1/encoders/static/best.pt"),
])
def test_frozen_path_aliases(old, new):
    assert repository_path(old) == (PROJECT_ROOT / new).resolve()


def test_unknown_paths_are_not_redirected(tmp_path):
    assert resolve_path(tmp_path / "external/model.pt") == tmp_path / "external/model.pt"
    assert resolve_path("custom/results", base=tmp_path) == tmp_path / "custom/results"
    assert resolve_path("../01_baselines_extra/config.json", base=tmp_path) == (tmp_path / "../01_baselines_extra/config.json").resolve()


def test_historical_yaml_resolves_existing_configs_without_mutation():
    source = PROJECT_ROOT / "configs/pilot.yaml"
    before = source.read_bytes()
    config = load_config(source)
    for key in ("static_config", "dynamic_config"):
        assert resolve(config, config["references"][key]).is_file()
    assert source.read_bytes() == before


def test_main_architecture_and_protocol_identities_remain_frozen():
    source = PROJECT_ROOT / "configs/retrained_v1.json"
    before = source.read_bytes()
    config = json.loads(before)
    assert canonical_digest(config) == "882065b68a1a9ebef1cfa387db10a75cb3f7b334037ab5eefbd6131c96057293"
    for kind in ("static", "dynamic"):
        model = load_architecture_config(kind)
        assert model["_architecture_source"] == config["models"][kind]["configuration"]
        assert model["_architecture_sha256"] == config["models"][kind]["configuration_sha256"]
    assert source.read_bytes() == before


def test_relocated_checkpoint_still_requires_exact_bytes(tmp_path, monkeypatch):
    from sensorium_runtime import paths
    from trajectory_model_eval.retrained_analysis import _check_checkpoints

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    checkpoint = tmp_path / "outputs/run/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic immutable checkpoint")
    identity = {
        "status": "complete", "initialization": "from_scratch", "selection_role": "validation",
        "split_sha256": "split", "preprocessing_sha256": "preprocessing",
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_path": "D:/old/experiments/04_model_comparison/outputs/run/best.pt",
    }
    checkpoints = {name: dict(identity) for name in ("static", "dynamic")}
    _check_checkpoints(checkpoints, {"split_sha256": "split"}, "preprocessing")
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="Frozen encoder artifact changed"):
        _check_checkpoints(checkpoints, {"split_sha256": "split"}, "preprocessing")


def test_historical_source_records_resolve_with_original_names():
    source = PROJECT_ROOT / "configs/unified_v1.json"
    config = json.loads(source.read_bytes())
    identities = _checkpoint_identities(config, None)
    for name, identity in identities.items():
        assert identity["configuration"] == config["models"][name]["configuration"]
        assert identity["source_record"] == config["models"][name]["source_record"]
        actual = hashlib.sha256(repository_path(identity["source_record"]).read_bytes()).hexdigest()
        assert identity["source_record_sha256"] == actual


def test_current_cli_and_shared_packages_use_this_project():
    from trajectory_model_eval.retrained_cli import PROJECT
    assert PROJECT == PROJECT_ROOT
    for package in ("trajectory_model_eval", "trajectory_reliability", "trajectory_eval", "trajectory_param_match"):
        module = importlib.import_module(package)
        assert Path(module.__file__).resolve().is_relative_to(SOURCE_ROOT)


def test_merged_entry_points_and_separate_dependency_roles():
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    scripts = config["project"]["scripts"]
    assert {"trajectory-evaluation", "trajectory-official-dynamic", "trajectory-static-dynamic", "trajectory-parameter-matching"} <= set(scripts)
    extras = config["project"]["optional-dependencies"]
    assert "pandas==2.0.0" in extras["encoding"]
    assert "pandas>=2.1,<4" in extras["analysis"]
