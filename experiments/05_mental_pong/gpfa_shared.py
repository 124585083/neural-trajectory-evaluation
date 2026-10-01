"""Exact Gaussian inference/EM for a GPFA with ONE shared RBF timescale.

This is a constrained GPFA, not the per-latent-timescale model. All latent
processes have the same unit-variance kernel. Observation noise is diagonal.
Training arrays must contain only finite active bins from training conditions.
There is no fitting of scaling, split assignment, or position labels here.

The observation-precision eigensystem decouples the latent processes exactly.
No (time * latent)-square matrix is used except in the explicit tiny debug API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from sklearn.decomposition import FactorAnalysis


class SharedTimescaleGPFA:
    def __init__(self, latent_dim, initial_tau=.15, tau_bounds=(.025, 2.),
                 nugget=1e-3, min_noise=1e-4, max_iterations=400,
                 tolerance=1e-6, seed=42, bin_size_seconds=.05):
        self.latent_dim = int(latent_dim)
        self.initial_tau = float(initial_tau)
        self.tau_bounds = tuple(float(v) for v in tau_bounds)
        self.nugget = float(nugget)
        self.min_noise = float(min_noise)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.bin_size_seconds = float(bin_size_seconds)
        self.bin_width_seconds = self.bin_size_seconds
        if self.latent_dim < 1 or self.max_iterations < 1:
            raise ValueError("latent_dim and max_iterations must be positive")
        if not 0 <= self.nugget < 1 or self.min_noise <= 0 or self.bin_size_seconds <= 0:
            raise ValueError("Invalid kernel, noise floor, or bin width")
        if not (0 < self.tau_bounds[0] <= self.initial_tau <= self.tau_bounds[1]):
            raise ValueError("initial_tau must lie in positive ordered tau bounds")
        if self.tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        self.C = self.d = self.R = None
        self.tau = self.initial_tau
        self.n_features_in_ = None
        self.is_fitted = False
        self.history = []
        self.initialization_info = {}
        self.stop_reason = "not_fitted"
        self.converged = False
        self._invalidate()

    @property
    def parameter_count(self):
        if self.n_features_in_ is None:
            raise RuntimeError("No fitted observation dimension")
        return self.n_features_in_ * self.latent_dim + 2 * self.n_features_in_ + 1

    def _invalidate(self):
        self._precision_eigen = None
        self._systems = {}
        self._prefix_weights = {}

    def _set_parameters(self, C, d, R, tau):
        C, d, R = (np.asarray(a, dtype=np.float64).copy() for a in (C, d, R))
        if C.ndim != 2 or C.shape[1] != self.latent_dim:
            raise ValueError("C must have shape [neuron,latent]")
        if d.shape != (C.shape[0],) or R.shape != d.shape:
            raise ValueError("d/R must have one entry per neuron")
        if not all(np.isfinite(a).all() for a in (C, d, R)) or np.any(R <= 0):
            raise ValueError("Parameters must be finite and noise positive")
        if not self.tau_bounds[0] <= float(tau) <= self.tau_bounds[1]:
            raise ValueError("tau outside configured bounds")
        self.C, self.d, self.R, self.tau = C, d, R, float(tau)
        self.n_features_in_ = C.shape[0]
        self.is_fitted = True
        self._invalidate()

    def _require_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Fit or load the model before inference")

    def _validate_sequences(self, sequences):
        values = [np.asarray(x, dtype=np.float64) for x in sequences]
        if not values:
            raise ValueError("At least one sequence is required")
        n = values[0].shape[1] if values[0].ndim == 2 else -1
        if any(x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != n or not np.isfinite(x).all()
               for x in values):
            raise ValueError("Sequences must be finite active [time,neuron] arrays with a common neuron axis")
        if self.is_fitted and n != self.n_features_in_:
            raise ValueError("Neuron count differs from fitted model")
        return values

    @staticmethod
    def _group_sequences(sequences):
        groups = {}
        for index, sequence in enumerate(sequences):
            groups.setdefault(len(sequence), []).append((index, sequence))
        return {length: (np.asarray([index for index, _ in items], dtype=int),
                         np.stack([sequence for _, sequence in items]))
                for length, items in groups.items()}

    def kernel(self, length, tau=None):
        tau = self.tau if tau is None else float(tau)
        times = np.arange(int(length), dtype=np.float64) * self.bin_size_seconds
        K = (1 - self.nugget) * np.exp(-.5 * ((times[:, None] - times[None, :]) / tau) ** 2)
        K.flat[::len(K) + 1] += self.nugget
        return K

    def _eigen(self):
        if self._precision_eigen is None:
            G = self.C.T @ (self.C / self.R[:, None])
            eigenvalues, V = np.linalg.eigh((G + G.T) / 2)
            if eigenvalues.min() < -1e-9 * max(1., eigenvalues.max()):
                raise np.linalg.LinAlgError("Observation precision is not positive semidefinite")
            self._precision_eigen = (np.maximum(eigenvalues, 0.), V)
        return self._precision_eigen

    def _system(self, length):
        if length not in self._systems:
            eigenvalues, V = self._eigen()
            kappa, U = np.linalg.eigh(self.kernel(length))
            if kappa.min() <= 0:
                raise np.linalg.LinAlgError("RBF covariance is not positive definite; use a positive fixed nugget")
            F = kappa[:, None] / (1 + kappa[:, None] * eigenvalues[None, :])
            self._systems[length] = {"U": U, "kappa": kappa, "F": F, "V": V,
                "latent_covariance_sum": (V * F.sum(axis=0)) @ V.T,
                "time_covariance_sum": (U * F.sum(axis=1)) @ U.T,
                "logdet": float(length * np.log(self.R).sum() +
                                np.log1p(kappa[:, None] * eigenvalues[None, :]).sum())}
        return self._systems[length]

    def _posterior_batch(self, values):
        """Shared length [batch,time,neuron] -> mean, LL, and common covariance factors."""
        B, L, N = values.shape
        system = self._system(L)
        residual = values - self.d
        eta = (residual / self.R) @ self.C
        rotated_eta = eta @ system["V"]
        eig_eta = np.einsum("st,bti->bsi", system["U"].T, rotated_eta, optimize=True)
        rotated_mean = np.einsum("ts,bsi->bti", system["U"], eig_eta * system["F"], optimize=True)
        mean = rotated_mean @ system["V"].T
        quadratic = np.sum(residual * (residual / self.R), axis=(1, 2)) - np.sum(rotated_eta * rotated_mean, axis=(1, 2))
        ll = -.5 * (L * N * np.log(2 * np.pi) + system["logdet"] + quadratic)
        return mean, ll, system

    def get_posterior_statistics(self, Y):
        """Tiny-debug smooth posterior: mean[T,q], covariance[T,q,T,q], LL.

        Raises above 2000 latent-time coordinates to prevent accidental dense
        allocations in real high-dimensional runs. Production EM uses factors.
        """
        self._require_fitted()
        y = self._validate_sequences([Y])[0]
        if len(y) * self.latent_dim > 2000:
            raise ValueError("Dense posterior statistics are restricted to tiny debug inputs")
        mean, ll, system = self._posterior_batch(y[None])
        U, V, F = (system[key] for key in ("U", "V", "F"))
        temporal_by_latent = np.einsum("tk,sk,ki->tsi", U, U, F, optimize=True)
        covariance = np.einsum("tsi,ai,bi->tasb", temporal_by_latent, V, V, optimize=True)
        return {"mean": mean[0], "covariance": covariance, "log_likelihood": float(ll[0])}

    def _expectation(self, groups):
        N, q = self.n_features_in_, self.latent_dim
        stats = {"n": 0, "sum_y": np.zeros(N), "sum_x": np.zeros(q),
                 "sum_yy": np.zeros(N), "sum_yx": np.zeros((N, q)),
                 "sum_xx": np.zeros((q, q)), "tau_stats": {}, "log_likelihood": 0.}
        for L, (_, Y) in groups.items():
            mean, ll, system = self._posterior_batch(Y)
            B = len(Y)
            flat_y, flat_x = Y.reshape(-1, N), mean.reshape(-1, q)
            stats["n"] += B * L
            stats["sum_y"] += flat_y.sum(axis=0)
            stats["sum_x"] += flat_x.sum(axis=0)
            stats["sum_yy"] += np.sum(flat_y ** 2, axis=0)
            stats["sum_yx"] += flat_y.T @ flat_x
            stats["sum_xx"] += flat_x.T @ flat_x + B * system["latent_covariance_sum"]
            stats["tau_stats"][L] = (B, np.einsum("bti,bsi->ts", mean, mean, optimize=True) + B * system["time_covariance_sum"])
            stats["log_likelihood"] += float(ll.sum())
        return stats

    def _observation_m_step(self, stats):
        n = stats["n"]
        mean_x, mean_y = stats["sum_x"] / n, stats["sum_y"] / n
        xx = stats["sum_xx"] - n * np.outer(mean_x, mean_x)
        yx = stats["sum_yx"] - n * np.outer(mean_y, mean_x)
        factor = cho_factor((xx + xx.T) / 2, lower=True, check_finite=False)
        C = cho_solve(factor, yx.T, check_finite=False).T
        d = mean_y - C @ mean_x
        # Centered sufficient statistics include posterior uncertainty. With C
        # at its optimum, SSE = centered E[y^2] - diag(C * centered E[yx]^T).
        residual_ss = stats["sum_yy"] - n * mean_y ** 2 - np.sum(C * yx, axis=1)
        R = np.maximum(residual_ss / n, self.min_noise)
        return C, d, R

    def _tau_m_step(self, tau_stats):
        def objective(log_tau):
            tau = float(np.exp(log_tau))
            total = 0.
            for L, (count, expected_sum) in tau_stats.items():
                factor = cho_factor(self.kernel(L, tau), lower=True, check_finite=False)
                logdet = 2 * np.log(np.diag(factor[0])).sum()
                total += .5 * (count * self.latent_dim * logdet +
                                np.trace(cho_solve(factor, expected_sum, check_finite=False)))
            return float(total)
        before = objective(np.log(self.tau))
        result = minimize_scalar(objective, bounds=np.log(self.tau_bounds), method="bounded",
                                 options={"xatol": 1e-6, "maxiter": 100})
        # A bounded optimizer need not return a better endpoint than the old
        # value; retaining the old tau preserves the EM ascent guarantee.
        candidates = [(before, self.tau)]
        if result.success and np.isfinite(result.fun):
            candidates.append((float(result.fun), float(np.exp(result.x))))
        for boundary in self.tau_bounds:
            candidates.append((objective(np.log(boundary)), boundary))
        best_value, tau = min(candidates, key=lambda item: item[0])
        return tau, {"tau_objective_before": before, "tau_objective_after": best_value,
                     "tau_optimizer_success": bool(result.success), "tau_optimizer_evaluations": int(result.nfev)}

    def fit(self, sequences, init_parameters=None, callback=None):
        """Fit only the provided finite training sequences; callback(model,row).

        C is perturbed by independent Gaussian noise with SD=1% of each column's
        across-neuron SD, using seed. This changes the local starting point;
        it is not a latent rotation. d/R are initialized directly from FA.
        """
        values = self._validate_sequences(sequences)
        if self.latent_dim >= values[0].shape[1]:
            raise ValueError("latent_dim must be smaller than neuron count")
        if init_parameters is None:
            fa = FactorAnalysis(n_components=self.latent_dim, svd_method="lapack", max_iter=500,
                                tol=1e-4, random_state=self.seed).fit(np.concatenate(values))
            initial = {"C": fa.components_.T, "d": fa.mean_, "R": fa.noise_variance_}
            origin = "training-sequence FactorAnalysis LAPACK"
        else:
            initial = init_parameters
            origin = "caller-supplied training-only C/d/R"
        C = np.asarray(initial["C"], dtype=np.float64).copy()
        if C.shape != (values[0].shape[1], self.latent_dim):
            raise ValueError("Initialization C has incompatible dimensions")
        scale = .01 * C.std(axis=0)
        C += np.random.default_rng(self.seed).normal(size=C.shape) * scale
        self._set_parameters(C, initial["d"], np.maximum(initial["R"], self.min_noise), self.initial_tau)
        self.initialization_info = {"source": origin, "seed": self.seed,
                                    "loading_jitter_per_column_sd": scale.tolist(),
                                    "loading_jitter_rule": ".01 times initial across-neuron column standard deviation",
                                    "initial_tau": self.initial_tau}
        groups = self._group_sequences(values)
        self.history = []
        self.converged = False
        self.stop_reason = "maximum_iterations"
        streak = 0
        start = time.perf_counter()
        for iteration in range(self.max_iterations):
            step_start = time.perf_counter()
            stats = self._expectation(groups)
            old_ll = stats["log_likelihood"]
            old = (self.C.copy(), self.d.copy(), self.R.copy(), self.tau)
            C, d, R = self._observation_m_step(stats)
            tau, tau_info = self._tau_m_step(stats["tau_stats"])
            self._set_parameters(C, d, R, tau)
            new_ll = float(self._score_groups(groups, len(values)).sum())
            delta = new_ll - old_ll
            numeric_tolerance = 1e-9 * (1 + abs(old_ll))
            rollback = not np.isfinite(new_ll) or delta < -numeric_tolerance
            if rollback:
                self._set_parameters(*old)
                self.stop_reason = "likelihood_decrease_rollback"
                streak = 0
            else:
                # Negative improvements never count as convergence, even when
                # within roundoff tolerance. There is no abs(delta) criterion.
                streak = streak + 1 if 0 <= delta <= self.tolerance * (1 + abs(old_ll)) else 0
            row = {"iteration": iteration + 1, "log_likelihood": old_ll if rollback else new_ll,
                   "previous_log_likelihood": old_ll, "proposed_log_likelihood": new_ll,
                   "delta": delta, "relative_delta": delta / (1 + abs(old_ll)),
                   "tau": self.tau, "noise_min": float(self.R.min()),
                   "seconds": time.perf_counter() - step_start, "elapsed_seconds": time.perf_counter() - start,
                   "accepted": not rollback, "rolled_back": rollback,
                   "likelihood_decrease_tolerance": numeric_tolerance,
                   "consecutive_small_nonnegative_improvements": streak, **tau_info}
            self.history.append(row)
            if callback is not None:
                callback(self, row.copy())
            if rollback:
                break
            if streak >= 3:
                self.converged = True
                self.stop_reason = "three_small_nonnegative_likelihood_improvements"
                break
        return self

    def _score_groups(self, groups, count):
        result = np.empty(count)
        for _, (indices, Y) in groups.items():
            _, ll, _ = self._posterior_batch(Y)
            result[indices] = ll
        return result

    def score_sequences(self, sequences):
        self._require_fitted()
        values = self._validate_sequences(sequences)
        return self._score_groups(self._group_sequences(values), len(values))

    def _last_weights(self, length):
        if length not in self._prefix_weights:
            system = self._system(length)
            # [time,latent], each column is the final row of that latent's P.
            self._prefix_weights[length] = system["U"] @ (system["U"][-1, :, None] * system["F"])
        return self._prefix_weights[length]

    def infer_sequence(self, Y, mode="causal"):
        self._require_fitted()
        y = np.asarray(Y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.n_features_in_ or len(y) < 1:
            raise ValueError("Input must be nonempty [time,fitted-neuron]")
        if mode not in {"causal", "smooth", "current"}:
            raise ValueError("mode must be causal, smooth, or current")
        if mode == "smooth":
            self._validate_sequences([y])
            z = self._posterior_batch(y[None])[0][0]
        else:
            eigenvalues, V = self._eigen()
            z = np.full((len(y), self.latent_dim), np.nan)
            finite_rows = np.isfinite(y).all(axis=1)
            if mode == "current":
                eta = ((y[finite_rows] - self.d) / self.R) @ self.C @ V
                z[finite_rows] = (eta / (1 + eigenvalues)) @ V.T
            else:
                # An unknown future row cannot change a known prefix. A missing
                # row and all subsequent causal states are undefined; no fake
                # zero observations or retrospective missing-data imputation.
                end = int(np.flatnonzero(~finite_rows)[0]) if not finite_rows.all() else len(y)
                eta = ((y[:end] - self.d) / self.R) @ self.C @ V
                for t in range(end):
                    z[t] = np.sum(self._last_weights(t + 1) * eta[:t + 1], axis=0) @ V.T
        return z @ self.C.T + self.d, z

    def infer_sequences(self, sequences, mode="causal"):
        return [self.infer_sequence(sequence, mode=mode) for sequence in sequences]

    def parameter_digest(self):
        self._require_fitted()
        h = hashlib.sha256()
        for value in (self.C, self.d, self.R, np.asarray([self.tau, self.nugget, self.bin_size_seconds])):
            h.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
        return h.hexdigest()

    def _configuration(self):
        return {key: getattr(self, key) for key in ("latent_dim", "initial_tau", "tau_bounds", "nugget", "min_noise",
                "max_iterations", "tolerance", "seed", "bin_size_seconds")}

    def save(self, path):
        self._require_fitted()
        metadata = {"configuration": self._configuration(), "history": self.history,
                    "initialization_info": self.initialization_info, "stop_reason": self.stop_reason,
                    "converged": self.converged, "parameter_digest": self.parameter_digest(),
                    "model_family": "exact tied-timescale unit-amplitude RBF GPFA"}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, C=self.C, d=self.d, R=self.R, tau=np.asarray(self.tau),
                                metadata=np.asarray(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"]))
            model = cls(**metadata["configuration"])
            model._set_parameters(stored["C"], stored["d"], stored["R"], float(stored["tau"]))
        model.history = metadata["history"]
        model.initialization_info = metadata["initialization_info"]
        model.stop_reason, model.converged = metadata["stop_reason"], metadata["converged"]
        if model.parameter_digest() != metadata["parameter_digest"]:
            raise ValueError("Saved GPFA parameter digest mismatch")
        return model
