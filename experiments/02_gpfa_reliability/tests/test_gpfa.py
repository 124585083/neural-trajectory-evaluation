import numpy as np

from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis


def _synthetic(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    trials, time, latent, neurons = 12, 10, 2, 7
    times = np.arange(time) * 0.1
    C = rng.normal(size=(neurons, latent))
    values = []
    for _ in range(trials):
        x = np.stack(
            [np.sin(times * 4 + rng.normal(scale=0.2)), np.cos(times * 2 + rng.normal(scale=0.2))],
            axis=1,
        )
        values.append(x @ C.T + 0.15 * rng.normal(size=(time, neurons)))
    return np.asarray(values), times


def test_gpfa_fit_transform_and_no_refit() -> None:
    values, times = _synthetic()
    model = GaussianProcessFactorAnalysis(
        latent_dim=2,
        times_seconds=times,
        max_em_iterations=3,
        random_seed=1,
    ).fit(values[:8], split_name="train")
    digest = model.parameter_digest()
    latent = model.transform(values[8:])
    assert latent.shape == (4, 10, 2)
    assert np.isfinite(latent).all()
    assert digest == model.parameter_digest()
    assert np.isfinite(model.score_samples(values[8:])).all()


def test_query_grid_preserves_requested_length() -> None:
    values, times = _synthetic()
    model = GaussianProcessFactorAnalysis(2, times, max_em_iterations=2).fit(values[:8])
    query = np.linspace(times[0], times[-1], 25)
    latent = model.transform_query(values[8:], query)
    assert latent.shape == (4, 25, 2)


def test_conditional_kernels_share_observation_coordinates() -> None:
    values, times = _synthetic()
    labels = np.asarray(["slow", "fast"] * 4)
    model = GaussianProcessFactorAnalysis(2, times, max_em_iterations=2).fit(values[:8], labels)
    assert set(model.lengthscales) == {"slow", "fast"}
    assert model.C.shape == (values.shape[-1], 2)

