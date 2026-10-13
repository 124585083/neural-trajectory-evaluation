import numpy as np

from trajectory_reliability.selection import select_gpfa


def test_selection_uses_calibration_without_refitting_it() -> None:
    rng = np.random.default_rng(4)
    times = np.arange(8) * 0.1
    C = rng.normal(size=(6, 2))
    latent = rng.normal(size=(12, 8, 2)).cumsum(axis=1)
    values = latent @ C.T + 0.2 * rng.normal(size=(12, 8, 6))
    result = select_gpfa(
        values[:8],
        values[8:],
        times,
        [1, 2],
        [0.2],
        max_em_iterations=1,
        tolerance=1e-4,
        bounds=(0.1, 1.0),
        seed=0,
    )
    assert result.fitted_model.fit_split == "train"
    assert result.fitted_model.fit_fingerprint is not None
    assert set(result.table.stage) == {"dimension", "initialization"}

