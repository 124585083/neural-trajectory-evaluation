import numpy as np

from trajectory_reliability.metrics import trajectory_metrics
from trajectory_reliability.reliability import _block_shuffle, _null_response


def test_identical_trajectory_has_expected_metrics() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 30, 3)).cumsum(axis=1)
    metrics = trajectory_metrics(values, values.copy(), 1 / 30)
    assert metrics["position_correlation"] > 0.999
    assert metrics["normalized_position_rmse"] < 1e-12
    assert metrics["velocity_direction_cosine"] > 0.999


def test_nulls_preserve_shape_and_block_content() -> None:
    values = np.arange(3 * 20 * 4).reshape(3, 20, 4)
    rng = np.random.default_rng(1)
    for name in ("condition_shuffle", "circular_shift", "frame_shuffle", "time_reversal", "independent_neuron_shift"):
        assert _null_response(values, name, rng).shape == values.shape
    shuffled = _block_shuffle(values, 5, rng)
    assert shuffled.shape == values.shape
    assert np.array_equal(np.sort(shuffled.ravel()), np.sort(values.ravel()))

