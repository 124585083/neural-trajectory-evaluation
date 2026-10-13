import numpy as np
import json
import pytest
from trajectory_model_eval.supplementary_reversal import condition_pattern_scores
from trajectory_model_eval.supplementary_common import paired_trajectory_intervals
from trajectory_model_eval.unified_analysis import _trajectory_scores


def test_pattern_invariance_does_not_imply_time_aligned_equality():
    rng=np.random.default_rng(8)
    neural=rng.normal(size=(12,31,7))
    prediction=neural*.8+rng.normal(size=neural.shape)*.1
    conditions=np.repeat(np.arange(6),2)
    scores,patterns=condition_pattern_scores(neural,{"original":prediction,"reversed":prediction[:,::-1]},conditions)
    np.testing.assert_allclose(patterns["original"],patterns["reversed"],atol=1e-14)
    for metric in scores["original"]:
        np.testing.assert_allclose(scores["original"][metric],scores["reversed"][metric],atol=1e-12)
    assert np.mean((prediction-neural)**2) < np.mean((prediction[:,::-1]-neural)**2)


def test_memoized_bootstrap_preserves_draws_and_estimand():
    rng=np.random.default_rng(5)
    left=rng.normal(size=(4,21,3))
    a=left+rng.normal(size=left.shape)
    b=left+rng.normal(size=left.shape)*.1
    table=paired_trajectory_intervals(left,a,b,seed=42,n_boot=35).set_index("metric")
    draws=np.random.default_rng(42).integers(0,4,size=(35,4))
    expected=[]
    for chosen in draws:
        values,_=_trajectory_scores(left[chosen],{"a":a[chosen],"b":b[chosen]},1/30)
        expected.append(values["b"]["position_correlation"]-values["a"]["position_correlation"])
    np.testing.assert_allclose(table.loc["position_correlation",["ci_low","ci_high"]].to_numpy(float),np.quantile(expected,[.025,.975]),atol=1e-12)
    assert (table.valid_bootstraps==35).all()


@pytest.mark.parametrize("changed", ["source", "cache", "output"])
def test_completed_reversal_rejects_changed_provenance(tmp_path, changed):
    from trajectory_model_eval.supplementary_reversal import run
    from trajectory_model_eval.unified_contract import canonical_digest, file_sha256
    root, out = tmp_path / "main", tmp_path / "diagnostic"
    (root / "diagnostics").mkdir(parents=True)
    out.mkdir()
    source = root / "weight"
    cache = root / "diagnostics/reversal_predictions.npz"
    output = out / "metrics.csv"
    for path in (source, cache, output):
        path.write_bytes(b"original")
    config = tmp_path / "config.json"
    config.write_text('{}', encoding="utf-8")
    saved = {"config_sha256": canonical_digest({}), "files": {"metrics.csv": file_sha256(output)},
             "source_artifacts": {"weight": file_sha256(source)}, "reversal_cache_sha256": file_sha256(cache)}
    (out / "manifest.json").write_text(json.dumps(saved), encoding="utf-8")
    assert run(root, out, config) == saved
    {"source": source, "cache": cache, "output": output}[changed].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run(root, out, config)
