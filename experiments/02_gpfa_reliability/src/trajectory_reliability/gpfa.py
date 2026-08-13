from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from sklearn.decomposition import FactorAnalysis


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass
class PosteriorSystem:
    factor: tuple[np.ndarray, bool]
    covariance: np.ndarray
    logdet_prior: float
    logdet_precision: float


@dataclass
class GaussianProcessFactorAnalysis:
    """Exact linear-Gaussian GPFA with diagonal observation noise.

    The observation manifold is shared across trials and optional condition
    classes. Each latent dimension has an independent squared-exponential GP.
    Classes may have different temporal lengthscales while retaining the same
    C, d, and R, so their latent coordinates remain directly comparable.
    """

    latent_dim: int
    times_seconds: np.ndarray
    max_em_iterations: int = 50
    tolerance: float = 1e-4
    initial_lengthscale_seconds: float = 0.5
    lengthscale_bounds_seconds: tuple[float, float] = (0.10, 3.0)
    random_seed: int = 0
    jitter: float = 1e-5
    min_noise: float = 1e-5
    C: np.ndarray | None = None
    d: np.ndarray | None = None
    R: np.ndarray | None = None
    lengthscales: dict[str, np.ndarray] | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    fit_fingerprint: str | None = None
    fit_split: str | None = None
    is_fitted: bool = False
    _posterior_cache: dict[str, PosteriorSystem] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.times_seconds = np.asarray(self.times_seconds, dtype=np.float64)
        if self.times_seconds.ndim != 1 or self.times_seconds.size < 3:
            raise ValueError("GPFA needs at least three ordered time points")
        if np.any(np.diff(self.times_seconds) <= 0):
            raise ValueError("times_seconds must be strictly increasing")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")

    @staticmethod
    def _labels(labels: np.ndarray | None, trials: int) -> np.ndarray:
        if labels is None:
            return np.repeat("all", trials)
        result = np.asarray(labels).astype(str)
        if result.shape != (trials,):
            raise ValueError("condition labels must have one value per trial")
        return result

    def _kernel(self, lengthscale: float, query_times: np.ndarray | None = None) -> np.ndarray:
        times = self.times_seconds if query_times is None else np.asarray(query_times, dtype=np.float64)
        delta = times[:, None] - times[None, :]
        kernel = np.exp(-0.5 * (delta / lengthscale) ** 2)
        kernel.flat[:: kernel.shape[0] + 1] += self.jitter
        return kernel

    def _prior(self, class_name: str) -> tuple[np.ndarray, np.ndarray, float]:
        if self.lengthscales is None:
            raise RuntimeError("lengthscales are unavailable")
        values = self.lengthscales[class_name]
        time = self.times_seconds.size
        size = time * self.latent_dim
        covariance = np.zeros((size, size), dtype=np.float64)
        precision = np.zeros_like(covariance)
        logdet = 0.0
        for component, lengthscale in enumerate(values):
            kernel = self._kernel(float(lengthscale))
            factor = cho_factor(kernel, lower=True, check_finite=False)
            inverse = cho_solve(factor, np.eye(time), check_finite=False)
            indices = np.arange(time) * self.latent_dim + component
            covariance[np.ix_(indices, indices)] = kernel
            precision[np.ix_(indices, indices)] = inverse
            logdet += 2.0 * np.log(np.diag(factor[0])).sum()
        return covariance, precision, float(logdet)

    def _system(self, class_name: str) -> PosteriorSystem:
        if class_name in self._posterior_cache:
            return self._posterior_cache[class_name]
        C, _, R = self._require_fitted(allow_during_fit=True)
        _, prior_precision, logdet_prior = self._prior(class_name)
        observation_precision = C.T @ (C / R[:, None])
        precision = prior_precision + np.kron(np.eye(self.times_seconds.size), observation_precision)
        factor = cho_factor(precision, lower=True, check_finite=False)
        covariance = cho_solve(factor, np.eye(precision.shape[0]), check_finite=False)
        logdet_precision = 2.0 * np.log(np.diag(factor[0])).sum()
        result = PosteriorSystem(factor, covariance, logdet_prior, float(logdet_precision))
        self._posterior_cache[class_name] = result
        return result

    def _require_fitted(self, allow_during_fit: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.C is None or self.d is None or self.R is None:
            raise RuntimeError("GPFA observation parameters are unavailable")
        if not allow_during_fit and not self.is_fitted:
            raise RuntimeError("GPFA must be fit before inference")
        return self.C, self.d, self.R

    def _initialize(self, trials: np.ndarray, labels: np.ndarray) -> None:
        trial_count, time, neurons = trials.shape
        if time != self.times_seconds.size:
            raise ValueError("training data time axis does not match times_seconds")
        if self.latent_dim >= neurons:
            raise ValueError("latent dimension must be smaller than neuron count")
        flattened = trials.reshape(trial_count * time, neurons).astype(np.float64)
        fa = FactorAnalysis(
            n_components=self.latent_dim,
            random_state=self.random_seed,
            max_iter=300,
            svd_method="randomized",
            iterated_power=3,
        ).fit(flattened)
        self.C = fa.components_.T.copy()
        self.d = fa.mean_.copy()
        self.R = np.clip(fa.noise_variance_.copy(), self.min_noise, None)
        self.lengthscales = {
            class_name: np.full(self.latent_dim, self.initial_lengthscale_seconds, dtype=np.float64)
            for class_name in np.unique(labels)
        }
        self._posterior_cache.clear()

    def _posterior(self, trials: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[str, PosteriorSystem]]:
        C, d, R = self._require_fitted(allow_during_fit=True)
        posterior = np.empty((len(trials), self.times_seconds.size, self.latent_dim), dtype=np.float64)
        systems: dict[str, PosteriorSystem] = {}
        for class_name in np.unique(labels):
            selected = np.flatnonzero(labels == class_name)
            system = self._system(class_name)
            systems[class_name] = system
            eta = ((trials[selected] - d) / R) @ C
            solved = cho_solve(
                system.factor,
                eta.reshape(len(selected), -1).T,
                check_finite=False,
            ).T
            posterior[selected] = solved.reshape(len(selected), self.times_seconds.size, self.latent_dim)
        return posterior, systems

    def _update_observation(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        posterior: np.ndarray,
        systems: dict[str, PosteriorSystem],
    ) -> None:
        trial_count, time, neurons = trials.shape
        n_observations = trial_count * time
        sum_x = posterior.sum(axis=(0, 1))
        sum_y = trials.sum(axis=(0, 1))
        sum_yx = np.einsum("mtn,mtq->nq", trials, posterior, optimize=True)
        sum_xx = np.einsum("mtq,mtr->qr", posterior, posterior, optimize=True)
        for class_name, system in systems.items():
            count = int(np.sum(labels == class_name))
            for time_index in range(time):
                section = slice(time_index * self.latent_dim, (time_index + 1) * self.latent_dim)
                sum_xx += count * system.covariance[section, section]
        mean_x = sum_x / n_observations
        mean_y = sum_y / n_observations
        centered_xx = sum_xx - n_observations * np.outer(mean_x, mean_x)
        centered_yx = sum_yx - n_observations * np.outer(mean_y, mean_x)
        ridge = self.jitter * np.eye(self.latent_dim)
        C = np.linalg.solve(centered_xx + ridge, centered_yx.T).T
        d = mean_y - C @ mean_x
        mean_residual = trials - (posterior @ C.T + d)
        residual_ss = np.einsum("mtn,mtn->n", mean_residual, mean_residual, optimize=True)
        uncertainty = np.zeros(neurons, dtype=np.float64)
        for class_name, system in systems.items():
            count = int(np.sum(labels == class_name))
            for time_index in range(time):
                section = slice(time_index * self.latent_dim, (time_index + 1) * self.latent_dim)
                block = system.covariance[section, section]
                uncertainty += count * np.einsum("nq,qr,nr->n", C, block, C, optimize=True)
        self.C = C
        self.d = d
        self.R = np.clip((residual_ss + uncertainty) / n_observations, self.min_noise, None)
        self._posterior_cache.clear()

    def _update_lengthscales(
        self,
        labels: np.ndarray,
        posterior: np.ndarray,
        systems: dict[str, PosteriorSystem],
    ) -> None:
        if self.lengthscales is None:
            raise RuntimeError("lengthscales are unavailable")
        lower, upper = np.log(self.lengthscale_bounds_seconds)
        time = self.times_seconds.size
        for class_name in np.unique(labels):
            selected = np.flatnonzero(labels == class_name)
            count = len(selected)
            system = systems[class_name]
            for component in range(self.latent_dim):
                indices = np.arange(time) * self.latent_dim + component
                means = posterior[selected, :, component]
                expected_xx = means.T @ means + count * system.covariance[np.ix_(indices, indices)]

                def objective(log_lengthscale: float) -> float:
                    kernel = self._kernel(float(np.exp(log_lengthscale)))
                    factor = cho_factor(kernel, lower=True, check_finite=False)
                    inverse_expected = cho_solve(factor, expected_xx, check_finite=False)
                    logdet = 2.0 * np.log(np.diag(factor[0])).sum()
                    return float(0.5 * (count * logdet + np.trace(inverse_expected)))

                result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
                self.lengthscales[class_name][component] = float(np.exp(result.x))
        self._posterior_cache.clear()

    def score_samples(self, trials: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        C, d, R = self._require_fitted()
        trials = np.asarray(trials, dtype=np.float64)
        class_labels = self._labels(labels, len(trials))
        if trials.shape[1:] != (self.times_seconds.size, C.shape[0]):
            raise ValueError("score input shape does not match fitted GPFA")
        result = np.empty(len(trials), dtype=np.float64)
        constant = trials.shape[1] * trials.shape[2] * np.log(2.0 * np.pi)
        observation_logdet = trials.shape[1] * np.log(R).sum()
        for class_name in np.unique(class_labels):
            selected = np.flatnonzero(class_labels == class_name)
            system = self._system(class_name)
            residual = trials[selected] - d
            eta = (residual / R) @ C
            solved = cho_solve(system.factor, eta.reshape(len(selected), -1).T, check_finite=False).T
            quadratic_y = np.einsum("mtn,mtn->m", residual, residual / R, optimize=True)
            quadratic_correction = np.einsum("mi,mi->m", eta.reshape(len(selected), -1), solved)
            logdet = observation_logdet + system.logdet_prior + system.logdet_precision
            result[selected] = -0.5 * (quadratic_y - quadratic_correction + logdet + constant)
        return result

    def fit(
        self,
        trials: np.ndarray,
        labels: np.ndarray | None = None,
        split_name: str = "train",
    ) -> "GaussianProcessFactorAnalysis":
        if split_name != "train":
            raise AssertionError("GPFA may only be fit on the real neural training split")
        values = np.asarray(trials, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("training data must be trial x time x neuron")
        class_labels = self._labels(labels, len(values))
        self._initialize(values, class_labels)
        self.is_fitted = True
        previous = -np.inf
        self.history.clear()
        for iteration in range(self.max_em_iterations):
            posterior, systems = self._posterior(values, class_labels)
            self._update_observation(values, class_labels, posterior, systems)
            posterior, systems = self._posterior(values, class_labels)
            self._update_lengthscales(class_labels, posterior, systems)
            log_likelihood = float(self.score_samples(values, class_labels).sum())
            improvement = log_likelihood - previous
            self.history.append(
                {
                    "iteration": float(iteration + 1),
                    "log_likelihood": log_likelihood,
                    "improvement": improvement,
                }
            )
            if np.isfinite(previous) and abs(improvement) <= self.tolerance * (1.0 + abs(previous)):
                break
            previous = log_likelihood
        self.fit_fingerprint = _array_digest(values)
        self.fit_split = split_name
        self.is_fitted = True
        return self

    def transform(self, trials: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        C, _, _ = self._require_fitted()
        values = np.asarray(trials, dtype=np.float64)
        if values.shape[1:] != (self.times_seconds.size, C.shape[0]):
            raise ValueError("inference input shape does not match fitted GPFA")
        class_labels = self._labels(labels, len(values))
        posterior, _ = self._posterior(values, class_labels)
        return posterior

    def transform_query(
        self,
        trials: np.ndarray,
        query_times_seconds: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        observed = self.transform(trials, labels)
        class_labels = self._labels(labels, len(observed))
        query_times = np.asarray(query_times_seconds, dtype=np.float64)
        output = np.empty((len(observed), len(query_times), self.latent_dim), dtype=np.float64)
        if self.lengthscales is None:
            raise RuntimeError("lengthscales are unavailable")
        for class_name in np.unique(class_labels):
            selected = np.flatnonzero(class_labels == class_name)
            for component, lengthscale in enumerate(self.lengthscales[class_name]):
                kernel_oo = self._kernel(float(lengthscale))
                delta = query_times[:, None] - self.times_seconds[None, :]
                kernel_qo = np.exp(-0.5 * (delta / float(lengthscale)) ** 2)
                weights = cho_solve(
                    cho_factor(kernel_oo, lower=True, check_finite=False),
                    observed[selected, :, component].T,
                    check_finite=False,
                )
                output[selected, :, component] = (kernel_qo @ weights).T
        return output

    def parameter_digest(self) -> str:
        C, d, R = self._require_fitted()
        digest = hashlib.sha256()
        for value in (C, d, R):
            digest.update(np.ascontiguousarray(value).tobytes())
        if self.lengthscales is None:
            raise RuntimeError("lengthscales are unavailable")
        for key in sorted(self.lengthscales):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(self.lengthscales[key]).tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._posterior_cache.clear()
        with destination.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "GaussianProcessFactorAnalysis":
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError("checkpoint does not contain GaussianProcessFactorAnalysis")
        model._require_fitted()
        return model

