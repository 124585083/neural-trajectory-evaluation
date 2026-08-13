from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gpfa import GaussianProcessFactorAnalysis


@dataclass(frozen=True)
class SelectionResult:
    latent_dim: int
    initial_lengthscale_seconds: float
    table: pd.DataFrame
    fitted_model: GaussianProcessFactorAnalysis


def _fit_and_score(
    fit_trials: np.ndarray,
    calibration_trials: np.ndarray,
    times_seconds: np.ndarray,
    latent_dim: int,
    initialization: float,
    max_em_iterations: int,
    tolerance: float,
    bounds: tuple[float, float],
    seed: int,
    fit_labels: np.ndarray | None = None,
    calibration_labels: np.ndarray | None = None,
) -> tuple[GaussianProcessFactorAnalysis, np.ndarray]:
    model = GaussianProcessFactorAnalysis(
        latent_dim=latent_dim,
        times_seconds=times_seconds,
        max_em_iterations=max_em_iterations,
        tolerance=tolerance,
        initial_lengthscale_seconds=initialization,
        lengthscale_bounds_seconds=bounds,
        random_seed=seed,
    ).fit(fit_trials, fit_labels, split_name="train")
    # Normalize by scalar observations so dimensions and neuron-count saturation
    # runs remain on a comparable scale.
    nll = -model.score_samples(calibration_trials, calibration_labels)
    nll /= calibration_trials.shape[1] * calibration_trials.shape[2]
    return model, nll


def select_gpfa(
    fit_trials: np.ndarray,
    calibration_trials: np.ndarray,
    times_seconds: np.ndarray,
    dimensions: list[int],
    initializations: list[float],
    max_em_iterations: int,
    tolerance: float,
    bounds: tuple[float, float],
    seed: int,
    fit_labels: np.ndarray | None = None,
    calibration_labels: np.ndarray | None = None,
) -> SelectionResult:
    """Select dimension by one-SE CV, then select timescale initialization."""
    if not initializations:
        raise ValueError("at least one initialization is required")
    rows: list[dict[str, float | int | bool | str]] = []
    dimension_models: dict[int, GaussianProcessFactorAnalysis] = {}
    dimension_scores: dict[int, np.ndarray] = {}
    reference_initialization = float(initializations[len(initializations) // 2])
    for dimension in dimensions:
        model, scores = _fit_and_score(
            fit_trials,
            calibration_trials,
            times_seconds,
            int(dimension),
            reference_initialization,
            max_em_iterations,
            tolerance,
            bounds,
            seed,
            fit_labels,
            calibration_labels,
        )
        dimension_models[int(dimension)] = model
        dimension_scores[int(dimension)] = scores
        rows.append(
            {
                "stage": "dimension",
                "latent_dim": int(dimension),
                "initial_lengthscale_seconds": reference_initialization,
                "calibration_nll_per_observation": float(scores.mean()),
                "calibration_standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                "selected": False,
            }
        )
    dimension_table = pd.DataFrame(rows)
    best_row = dimension_table.loc[dimension_table.calibration_nll_per_observation.idxmin()]
    threshold = float(best_row.calibration_nll_per_observation + best_row.calibration_standard_error)
    eligible = dimension_table[dimension_table.calibration_nll_per_observation <= threshold]
    selected_dim = int(eligible.latent_dim.min())
    dimension_table.loc[
        (dimension_table.stage == "dimension") & (dimension_table.latent_dim == selected_dim), "selected"
    ] = True

    initialization_models: dict[float, GaussianProcessFactorAnalysis] = {}
    initialization_scores: dict[float, np.ndarray] = {}
    initialization_rows = []
    for initialization in initializations:
        value = float(initialization)
        if value == reference_initialization:
            model = dimension_models[selected_dim]
            scores = dimension_scores[selected_dim]
        else:
            model, scores = _fit_and_score(
                fit_trials,
                calibration_trials,
                times_seconds,
                selected_dim,
                value,
                max_em_iterations,
                tolerance,
                bounds,
                seed,
                fit_labels,
                calibration_labels,
            )
        initialization_models[value] = model
        initialization_scores[value] = scores
        initialization_rows.append(
            {
                "stage": "initialization",
                "latent_dim": selected_dim,
                "initial_lengthscale_seconds": value,
                "calibration_nll_per_observation": float(scores.mean()),
                "calibration_standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                "selected": False,
            }
        )
    initialization_table = pd.DataFrame(initialization_rows)
    selected_initialization = float(
        initialization_table.loc[
            initialization_table.calibration_nll_per_observation.idxmin(),
            "initial_lengthscale_seconds",
        ]
    )
    initialization_table.loc[
        initialization_table.initial_lengthscale_seconds == selected_initialization, "selected"
    ] = True
    table = pd.concat([dimension_table, initialization_table], ignore_index=True)
    return SelectionResult(
        latent_dim=selected_dim,
        initial_lengthscale_seconds=selected_initialization,
        table=table,
        fitted_model=initialization_models[selected_initialization],
    )

