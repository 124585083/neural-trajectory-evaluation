# Phase 2 — Validating a brain-defined GPFA trajectory assay

**Research question:** Are GPFA trajectory metrics reliable and sensitive to genuine temporal organization in repeated neural population responses before they are used to compare encoding models?

## Why this phase exists

A low-dimensional trajectory can look smooth and interpretable even when it is unstable, stimulus-insensitive, or unable to distinguish temporal order. Phase 2 therefore validates the measurement assay using recorded neural responses alone. Static and Dynamic model predictions are deliberately excluded so that neither model can influence the latent space, its dimensionality, or the choice of trajectory metrics.

## Experimental design

The proof-of-concept assay uses one Dynamic Sensorium 2023 session, a deterministic 512-neuron subset, and the six repeated oracle movies. GPFA selection uses only the official neural training tier: 278 trials for fitting and 70 for calibration, followed by a final refit on all 348 training trials. Oracle responses are used only after the model is frozen.

Latent dimensions 4, 8, 12, and 16 are compared by held-out calibration marginal negative log likelihood. A one-standard-error rule selects the smallest eligible model, `q = 4`; oracle reliability and null performance do not enter this choice. The primary GPFA observes 63 samples per trial at 7.5 Hz-equivalent resolution and queries the continuous posterior at all 250 official 30 Hz timestamps.

Reliability is evaluated with 200 balanced, disjoint split halves of repeated neural responses. Matched nulls disrupt movie identity, absolute timing, temporal direction, local order, or coordinated population timing while preserving other aspects of the data where possible.

## Current result

The assay passes as a **one-session method-development validation**.

| Metric | Neural split-half mean | Interpretation |
|---|---:|---|
| Position correlation | 0.8566 | Reliable time-aligned population state |
| Normalized position RMSE | 0.5428 | Reliable position error after scale normalization |
| Velocity-direction cosine | 0.6627 | Reliable local direction |
| Speed-profile correlation | 0.7434 | Reliable timing of fast and slow trajectory segments |
| Acceleration-direction cosine | 0.6163 | Informative but higher-variance local curvature diagnostic |

Position, normalized error, velocity direction, speed profile, and acceleration direction outperform condition-shuffle, circular-shift, time-reversal, block-shuffle, and independent-neuron-shift nulls in every one of the 200 paired splits. Path length is reproducible but fails important circular-shift and time-reversal controls, so it is retained only as a descriptive quantity rather than a primary temporal-alignment metric.

## Interpretation and limits

This phase establishes that the selected metrics can recover repeatable, stimulus-locked temporal structure from real neural activity. It does not rank Static and Dynamic models and does not establish a five-session biological conclusion.

The neural signal was acquired at approximately 8 Hz and represented on the official 30 Hz grid. Velocity and especially acceleration at 30 Hz describe derivatives of the continuous GPFA posterior, not independent 30 Hz neural measurements. These metrics are therefore interpreted conservatively and checked against observation-grid, neuron-count, latent-dimension, train-fraction, seed, and split-count sensitivity analyses.

## Reproducible assets

- Locked configuration: [`configs/pilot.yaml`](configs/pilot.yaml)
- Frozen GPFA and preprocessing: [`../../models/gpfa_reliability/`](../../models/gpfa_reliability/)
- Compact results and sensitivity tables: [`../../results/tables/02_gpfa_reliability/`](../../results/tables/02_gpfa_reliability/)
- Full result report: [GPFA Reliability Results](../../docs/results/GPFA_RELIABILITY_RESULTS.md)
- Method specification: [Methods](../../docs/METHODS.md) and [locked GPFA protocol](../../docs/methods/GPFA_PROTOCOL_LOCKED.md)

## Minimal reproduction entry points

Run from this phase directory:

```text
python -m trajectory_reliability.cli inspect --config configs/pilot.yaml
python -m trajectory_reliability.cli smoke --config configs/pilot.yaml
python -m trajectory_reliability.cli run --config configs/pilot.yaml
python -m trajectory_reliability.cli saturation --config configs/pilot.yaml
python -m trajectory_reliability.cli split-saturation --config configs/pilot.yaml
python -m trajectory_reliability.cli condition-prior --config configs/pilot.yaml
```

The phase reads the authorized Sensorium dataset in place and does not modify the data or Phase 1 checkpoints.

**Previous phase:** [Phase 1](../01_baselines/README.md) establishes the encoding baselines.  
**Next phase:** [Phase 3](../03_parameter_matching/README.md) constructs a Dynamic model whose parameter count is matched to the Static model.
