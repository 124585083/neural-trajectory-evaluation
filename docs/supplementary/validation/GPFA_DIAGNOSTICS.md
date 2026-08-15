# GPFA Supplementary Diagnostics

This document preserves detailed diagnostics that support the interpretation summarized in [GPFA Validation](../../GPFA_VALIDATION.md). These diagnostics are supplementary and are not required for the main reading path.

Unless otherwise noted, the diagnostics below refer to the Phase 2 full-train method-development assay on the pilot session: the deterministic 512-neuron subset, frames 50–299, and the primary seed-42 `q = 4` GPFA refitted on all 348 training trials using the approximately 7.5-Hz observation grid. The `q = 8` and `q = 16` variance-coverage entries are short-EM diagnostic fits rather than frozen primary models.

## 1. Reliability as a function of repeat averaging

The Phase 2 full-train method-development GPFA was evaluated with 200 random disjoint splits while varying the number of repeats used in each half:

| Repeats / half | Position | Norm. RMSE | Velocity | Speed | Path | Acceleration |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5545 | 0.9611 | 0.3417 | 0.3694 | 0.8940 | 0.3051 |
| 2 | 0.7117 | 0.7718 | 0.4859 | 0.5434 | 0.9272 | 0.4410 |
| 3 | 0.7864 | 0.6612 | 0.5736 | 0.6400 | 0.9363 | 0.5269 |
| 4 | 0.8328 | 0.5871 | 0.6352 | 0.7077 | 0.9384 | 0.5869 |

The 95% split interval for single-repeat position is approximately `0.4035–0.6613`. Reliability increases with repeat averaging, and single-trial trajectories are materially less reliable than the condition-average primary assay. The main scientific interpretation is summarized in [GPFA Validation](../../GPFA_VALIDATION.md).

## 2. Strict-convergence diagnostic

A supplementary `q = 4` fit used a maximum of 20 EM iterations with tolerance `1e-8`. The detailed fit comparison is:

| Diagnostic | Value |
|---|---|
| Base fitted timescales | `0.25048 / 0.25087 / 0.25046 / 0.25062 s` |
| Strict fitted timescales | `0.25403 / 0.25792 / 0.25438 / 0.25593 s` |
| NLL/observation improvement | `2.489e-5` |
| `C`-subspace principal angles | `6.21°, 3.99°, 0.90°, 0.37°` |

On the same 50 splits:

| Metric | Base | Strict |
|---|---:|---:|
| Position | 0.85641 | 0.85748 |
| Normalized RMSE | 0.54262 | 0.54084 |
| Velocity | 0.66363 | 0.66526 |
| Speed | 0.74265 | 0.74606 |
| Acceleration | 0.61327 | 0.61691 |

Stricter optimization modestly changes the fitted timescales and loading subspace, while all listed trajectory reliability metrics change by less than approximately `0.004`. The exact fitted timescales are therefore less strongly identified than the trajectory reliability conclusion.

## 3. Low-dimensional response-variance coverage

Oracle responses were reconstructed from the GPFA posterior through `C x + d`:

| q | Population R² | Mean per-neuron R² | Median per-neuron R² | 90th percentile |
|---:|---:|---:|---:|---:|
| 4 | 0.0200 | 0.00369 | 0.01296 | 0.05275 |
| 8 | 0.04231 | 0.02214 | 0.03099 | 0.08899 |
| 16 | 0.07148 | 0.04123 | 0.04916 | 0.12258 |

At `q = 4`, approximately 69.5% of neurons have positive per-neuron R², and the population R² for trial-averaged responses is approximately `0.02824`. The `q = 8` and `q = 16` models are short-EM diagnostic fits rather than the frozen primary model.

The GPFA captures a repeatable low-dimensional shared subspace but explains only a limited fraction of total neural-response variance. Trajectory evaluation therefore complements rather than replaces response prediction.
