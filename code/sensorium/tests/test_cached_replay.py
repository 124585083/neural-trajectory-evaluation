"""Orchestration contracts only; no scientific inference or fitting is executed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sensorium_runtime import replay
from trajectory_model_eval.unified_contract import canonical_digest, file_sha256


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_cache_preflight_rejects_identity_and_payload_changes(tmp_path):
    path = tmp_path / "cache.npz"
    path.write_bytes(b"synthetic payload: not scientific data")
    identity = {"source": "synthetic", "roles": ["test"]}
    record = {"identity": identity, "identity_sha256": canonical_digest(identity),
              "shape": [2, 3, 4], "payload_sha256": file_sha256(path)}
    dump(path.with_suffix(".json"), record)
    replay.verify_cache(path, identity, (2, 3, 4))
    with pytest.raises(ValueError, match="identity or shape"):
        replay.verify_cache(path, {**identity, "roles": ["train"]}, (2, 3, 4))
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum differs"):
        replay.verify_cache(path, identity, (2, 3, 4))


@pytest.mark.parametrize("relative", ["artifacts", "artifacts/new", ".", "evidence/new"])
def test_output_cannot_overlap_inputs(tmp_path, relative):
    with pytest.raises(ValueError, match="overlaps"):
        replay.validate_destination(tmp_path / relative, tmp_path / "artifacts", tmp_path / "evidence")


def test_nonempty_output_is_preserved(tmp_path):
    out = tmp_path / "out"
    out.mkdir(); (out / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        replay.validate_destination(out, tmp_path / "artifacts", tmp_path / "evidence")
    assert (out / "user.txt").read_text() == "keep"


def plan(tmp_path):
    return {"root": tmp_path / "artifacts", "status": "verified_frozen_inputs", "scope": "synthetic contract",
            "lock": {"freeze_sha256": "synthetic"}, "historical_evidence": {}, "cache_checksums": {}}


def test_check_does_not_create_output_or_call_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "preflight", lambda *args: plan(tmp_path))
    for name in ("replay_primary", "replay_measurement", "replay_reversal", "replay_encoder"):
        monkeypatch.setattr(replay, name, lambda *args: pytest.fail("Check must never score"))
    out = tmp_path / "out"
    result = replay.execute(tmp_path / "artifacts", out, evidence_root=tmp_path / "evidence")
    assert result["mode"] == "check" and not out.exists()


def test_cli_defaults_to_project_local_bundle_without_external_fallback(monkeypatch):
    seen = []
    monkeypatch.setattr(replay, "execute", lambda root, output, **kwargs: seen.append((root, output, kwargs)))
    replay.main(["--check"])
    assert seen == [(replay.REPOSITORY_ROOT / "models/sensorium/retrained_v1", None,
                     {"evidence_root": replay.EVIDENCE, "run": False})]


def test_failed_preflight_never_mutates_output(tmp_path, monkeypatch):
    def fail(*args):
        raise FileNotFoundError("Missing frozen source")
    monkeypatch.setattr(replay, "preflight", fail)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        replay.execute(tmp_path / "artifacts", out, evidence_root=tmp_path / "evidence", run=True)
    assert not out.exists()


@pytest.mark.parametrize("relative", ["evaluation/pilot/metrics.csv",
    "diagnostics/encoder/temporal_support.json", "diagnostics/encoder/weight_changes.json"])
def test_missing_scoring_source_prevents_output_creation(tmp_path, monkeypatch, relative):
    root = tmp_path / "artifacts"
    def check(*args):
        replay.verify_files(root, {relative: "required-checksum"})
        return plan(tmp_path)
    monkeypatch.setattr(replay, "preflight", check)
    with pytest.raises(FileNotFoundError, match="Required frozen input"):
        replay.execute(root, tmp_path / "new", evidence_root=tmp_path / "evidence", run=True)
    assert not (tmp_path / "new").exists()


def test_run_dispatches_only_retained_cached_stages(tmp_path, monkeypatch):
    calls = []
    def check(*args):
        calls.append("check")
        return plan(tmp_path)
    monkeypatch.setattr(replay, "preflight", check)
    for name in ("replay_primary", "replay_measurement", "replay_reversal", "replay_encoder"):
        def stage(context, path, name=name):
            calls.append(name)
            path.mkdir(parents=True)
            (path / "synthetic.txt").write_text("orchestration only")
        monkeypatch.setattr(replay, name, stage)
    result = replay.execute(tmp_path / "artifacts", tmp_path / "out", evidence_root=tmp_path / "evidence", run=True)
    assert calls == ["check", "replay_primary", "replay_measurement", "replay_reversal", "replay_encoder", "check"]
    assert result["recomputed_goals"] == [2, 3, 4, 5, 6]
    assert result["historical_checks"] == "verified_existing_records_only"


def test_encoder_scoring_uses_separate_writable_work(tmp_path, monkeypatch):
    from trajectory_model_eval import supplementary_encoder as module
    out, public, work = (tmp_path / name for name in ("inputs", "scores", "work"))
    config = tmp_path / "config.json"; dump(config, {})
    args = SimpleNamespace(output_dir=out, public_dir=public, config=config, work_dir=work)
    snapshot = work / "source_snapshots/scoring_implementation_response_v2.py"
    snapshot.parent.mkdir(parents=True); snapshot.write_text("synthetic scoring source")
    endpoint = work / "source_snapshots/endpoint.py"; endpoint.write_text("synthetic endpoint source")
    dump(public / "manifest.json", {"status": "complete", "config_sha256": canonical_digest({}),
        "response_correlation_numerical_fix": {"version": 2, "implementation_sha256": file_sha256(snapshot)},
        "files": {}, "supplementary_source_records": {}, "prediction_input_records": {},
        "zero_spatial_endpoint_correction": {"source_snapshot": "source_snapshots/endpoint.py",
                                               "implementation_sha256": file_sha256(endpoint)}})
    monkeypatch.setattr(module, "verify_completed_predictions", lambda _: True)
    assert module.verify_completed_scores(args)
    assert not out.exists()
    assert module.score_work_dir(SimpleNamespace(output_dir=out)) == out


def test_reversal_wrapper_redirects_cache_to_new_output(tmp_path, monkeypatch):
    from trajectory_model_eval import supplementary_reversal
    received = {}
    monkeypatch.setattr(supplementary_reversal, "run", lambda *args, **kwargs: received.update(kwargs))
    output = tmp_path / "new/reversal"
    replay.replay_reversal({"root": tmp_path / "inputs"}, output)
    assert received["cache_output_path"] == output / "reversal_predictions.npz"


def test_missing_inputs_fail_without_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay.execute(tmp_path / "absent", tmp_path / "new", evidence_root=tmp_path / "evidence", run=True)
    assert not (tmp_path / "new").exists()
