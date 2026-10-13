"""Small numerical contracts for the bounded frozen-encoder diagnostic."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_model_eval.supplementary_encoder import (
    intervention_state, mean_response_r, select_spatial_matches, temporal_support,
    verify_completed_predictions, write_json, _save_prediction, _cache_path,
    response_columns, _check_test_cache,
    verify_completed_scores, response_endpoint_status,
)
from trajectory_model_eval.unified_contract import canonical_digest, file_sha256


def example_state():
    return {
        "core.conv_temporal.weight": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3, 1, 1),
        "core.conv_spatial.weight": torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 2, 2),
        "core.bn.running_mean": torch.tensor([1., 2.]),
        "core.bias": torch.ones(2),
    }


def test_temporal_center_bias_and_batchnorm_are_preserved():
    state = example_state()
    before = copy.deepcopy(state)
    changed, records = intervention_state(state, "temporal", 0, ["core.conv_spatial.weight"])
    assert torch.equal(changed["core.conv_temporal.weight"][:, :, 1], state["core.conv_temporal.weight"][:, :, 1])
    assert torch.count_nonzero(changed["core.conv_temporal.weight"][:, :, [0, 2]]) == 0
    assert records[0]["eligible_parameters"] == 12
    for name in state:
        assert torch.equal(state[name], before[name])
        if name != "core.conv_temporal.weight":
            assert torch.equal(changed[name], before[name])


def test_spatial_control_does_not_change_temporal_weights():
    state = example_state()
    changed, records = intervention_state(state, "spatial", .5, ["core.conv_spatial.weight"])
    assert torch.equal(changed["core.conv_spatial.weight"], state["core.conv_spatial.weight"] * .5)
    assert records[0]["eligible_parameters"] == 24
    for name in state:
        if name != "core.conv_spatial.weight":
            assert torch.equal(changed[name], state[name])


@pytest.mark.parametrize("kind", ["temporal", "spatial"])
def test_retention_one_is_exact_identity(kind):
    state = example_state()
    changed, records = intervention_state(state, kind, 1, ["core.conv_spatial.weight"])
    assert all(torch.equal(changed[k], state[k]) for k in state)
    assert sum(r["actually_changed_parameters"] for r in records) == 0


def test_matching_uses_validation_only_retains_failure_and_breaks_ties():
    temporal = [{"role": "validation", "retention": .5, "response_r": .5}]
    spatial = [{"role": "validation", "retention": r, "response_r": s} for r, s in [(1, .6), (.5, .4)]]
    result = select_spatial_matches(temporal, spatial, .005)[0]
    assert result["spatial_retention"] == 1
    assert result["validation_match"] is False
    assert result["status"] == "outside_validation_tolerance"
    with pytest.raises(ValueError, match="only be selected using validation"):
        select_spatial_matches([{**temporal[0], "role": "test"}], spatial, .005)


def test_zero_retention_is_delayed_visual_not_static():
    support = temporal_support(0, [11, 5, 5])
    assert support["visual_and_appended_behavior_frame_offsets"] == [-9]
    assert support["pupil_shifter_frame_offsets"] == [0]
    assert support["is_static_model"] is False
    assert temporal_support(.25, [11, 5, 5])["visual_and_appended_behavior_frame_offsets"] == list(range(-18, 1))


def test_response_summary_uses_pooled_samples_and_preserves_undefined():
    target = np.arange(60, dtype=float).reshape(2, 10, 3)
    score, valid = mean_response_r(target, target * 2 + 5)
    assert score == pytest.approx(1)
    assert valid == 3
    score, valid = mean_response_r(target, np.ones_like(target))
    assert score is None and valid == 0


def test_completed_cache_reuse_checks_snapshot_payload_and_primary_sources(tmp_path):
    from types import SimpleNamespace
    root, out = tmp_path / "run", tmp_path / "output"
    root.mkdir()
    (root / "fixed.bin").write_bytes(b"frozen primary artifact")
    snapshot = out / "source_snapshots/prediction_implementation.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("# actual prediction source\n", encoding="utf-8")
    config = {"temporal": {"retentions": [1.]}, "spatial": {"retentions": [1.]}}
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    write_json(out / "fixed_config.json", config)
    identity = {"prediction_implementation_sha256": file_sha256(snapshot),
                "primary_sources": {"fixed.bin": file_sha256(root / "fixed.bin")}}
    write_json(out / "prediction_identity.json", identity)
    selection = {"matches": [{"temporal_retention": 1., "spatial_retention": 1.}]}
    selection["selection_sha256"] = canonical_digest(selection)
    write_json(out / "spatial_selection.json", selection)
    write_json(out / "prediction_status.json", {"status": "complete"})
    for role in ("validation", "test"):
        for kind in ("temporal", "spatial"):
            record = {**identity, "role": role, "kind": kind, "retention": 1.,
                "spatial_selection_sha256": selection["selection_sha256"] if role == "test" else None}
            _save_prediction(_cache_path(out, role, kind, 1.), {"prediction": np.ones((1, 2, 3))}, record, [])
    args = SimpleNamespace(output_dir=out, run_root=root, config=config_path)
    assert verify_completed_predictions(args)
    damaged = _cache_path(out, "test", "spatial", 1.)
    damaged.write_bytes(damaged.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="payload or support was changed"):
        verify_completed_predictions(args)


def test_float32_constant_is_na_but_real_nearconstant_variation_is_retained():
    # 1.3 is not exactly representable; float32 mean accumulation can differ
    # from the stored constant, so merely testing centered residuals is unsafe.
    prediction = np.full((2, 31, 2), np.float32(1.3), dtype=np.float32)
    target = np.arange(prediction.size, dtype=np.float32).reshape(prediction.shape)
    assert np.isnan(response_columns(target, prediction)).all()
    increments = np.arange(62, dtype=np.float32).reshape(2, 31) % 2
    prediction[:, :, 1] += increments * np.spacing(np.float32(1.3))
    target[:, :, 1] = increments
    values = response_columns(target, prediction)
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(1, abs=1e-12)
    score, count = mean_response_r(target, prediction)
    assert count == 1 and score == pytest.approx(1)


def test_endpoint_status_keeps_constant_invalid_and_small_real_variation_valid():
    target = np.arange(62, dtype=np.float32).reshape(2, 31, 1)
    prediction = np.full_like(target, np.float32(1.3))
    constant = response_endpoint_status(target, prediction, "ordinary_model")
    assert constant["response_r"] is None
    assert constant["response_status"] == "undefined_constant"
    alternating = np.arange(62, dtype=np.float32).reshape(2, 31, 1) % 2
    prediction += alternating * np.spacing(np.float32(1.3))
    ordinary = response_endpoint_status(alternating, prediction, "ordinary_model")
    assert ordinary["response_status"] == "defined"
    assert ordinary["response_r"] == pytest.approx(1)


def test_verified_spatial_endpoint_is_na_without_a_global_amplitude_cutoff():
    target = np.arange(62, dtype=np.float32).reshape(2, 31, 1) % 2
    prediction = np.full_like(target, np.float32(1.3))
    prediction += target * np.spacing(np.float32(1.3))
    ranges = np.ptp(prediction.reshape(-1, 1).astype(np.float64), axis=0)
    diagnostic = {"exact_constant_prediction_neurons": 0,
        "defined_response_neurons": 1, "total_neurons": 1,
        "prediction_range_min": float(ranges.min()), "prediction_range_median": float(np.median(ranges)),
        "prediction_range_max": float(ranges.max()), "maximum_absolute_prediction": float(np.abs(prediction).max())}
    weights = [{"tensor": "core.features.layer0.conv_spatial.weight", "changed_l2": 0.0}]
    result = response_endpoint_status(target, prediction, "spatial_0.00", diagnostic, weights)
    assert result["response_r"] is None and result["valid_neurons"] == 0
    assert result["response_status"] == "numerically_degenerate"
    assert result["raw_response_r"] == pytest.approx(1)
    assert result["raw_valid_neurons"] == 1
    with pytest.raises(ValueError, match="requires its recorded"):
        response_endpoint_status(target, prediction, "spatial_0.00")
    with pytest.raises(ValueError, match="differs from the recorded"):
        response_endpoint_status(target, prediction * 2, "spatial_0.00", diagnostic, weights)
    with pytest.raises(ValueError, match="verified zero"):
        response_endpoint_status(target, prediction, "spatial_0.00", diagnostic,
            [{**weights[0], "changed_l2": 1.0}])


@pytest.mark.parametrize("field,value", [("kind", "spatial"), ("retention", .25),
    ("neuron_indices", [1, 0]), ("frame_indices", [51, 52])])
def test_score_cache_rejects_intervention_or_support_swap(tmp_path, field, value):
    from types import SimpleNamespace
    test = {"neural": np.ones((1, 2, 2)), "conditions": np.asarray(["c"]),
        "dataset_indices": np.asarray([4]), "neuron_indices": np.asarray([0, 1]),
        "frame_indices": np.asarray([50, 51])}
    context = SimpleNamespace(manifest={"session": "s"}, test=test)
    identity = {"role": "test", "kind": "temporal", "retention": .5, "session": "s",
        "spatial_selection_sha256": "selection", "dataset_indices": [4],
        "neuron_indices": [0, 1], "frame_indices": [50, 51]}
    identity[field] = value
    path = _cache_path(tmp_path, "test", "temporal", .5)
    _save_prediction(path, test, identity, [])
    with pytest.raises(ValueError, match="name or exact trial/neuron/frame support differs"):
        _check_test_cache(tmp_path, "test", "temporal", .5, context, {}, "selection")


def test_completed_score_reuse_rejects_changed_public_result(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import trajectory_model_eval.supplementary_encoder as module
    out, public = tmp_path / "output", tmp_path / "public"
    snapshot = out / "source_snapshots/scoring_implementation_response_v2.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("# recorded numerical correction\n", encoding="utf-8")
    public.mkdir()
    result = public / "scores.csv"
    result.write_text("metric,value\nr,0.3\n", encoding="utf-8")
    config = tmp_path / "config.json"
    write_json(config, {"seed": 42})
    write_json(public / "manifest.json", {"status": "complete",
        "config_sha256": canonical_digest({"seed": 42}),
        "response_correlation_numerical_fix": {"version": 2, "implementation_sha256": file_sha256(snapshot)},
        "files": {"scores.csv": file_sha256(result)}, "supplementary_source_records": {}, "prediction_input_records": {}})
    monkeypatch.setattr(module, "verify_completed_predictions", lambda args: True)
    args = SimpleNamespace(public_dir=public, output_dir=out, config=config)
    assert verify_completed_scores(args)
    result.write_text("metric,value\nr,0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result changed"):
        verify_completed_scores(args)
