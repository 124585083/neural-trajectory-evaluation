from pathlib import Path

import numpy as np

from trajectory_reliability.data import SessionMetadata, deterministic_neuron_order


def test_neuron_order_is_nested_and_deterministic(tmp_path: Path) -> None:
    metadata = SessionMetadata(
        "session",
        tmp_path,
        np.asarray(["train"]),
        np.asarray(["0"]),
        np.asarray([10, 20, 30, 40]),
        np.ones(4),
    )
    first = deterministic_neuron_order(metadata, 42)
    second = deterministic_neuron_order(metadata, 42)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == [0, 1, 2, 3]
    assert np.array_equal(first[:2], second[:2])

