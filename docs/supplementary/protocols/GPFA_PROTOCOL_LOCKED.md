# Sensorium 2023 GPFA Trajectory-Reliability Protocol

Status: internally locked before trajectory-reliability results were inspected. This document records a result-blind analysis specification; it was not registered on an external preregistration platform.

## Scope

This phase tests whether trajectory metrics recover reproducible neural dynamics and reject
matched nulls. It does not yet rank the Static and Dynamic encoding models.

## Locked temporal alignment

- Each source trial contains 300 aligned stimulus/response frames at the official 30 Hz grid.
- Only original frames 50--299 enter trajectory analysis: 250 frames, 8.33 seconds.
- Sensorium neural activity was acquired at approximately 8 Hz and resampled to 30 Hz.
- Primary GPFA parameter fitting therefore observes every fourth official frame (7.5 Hz,
  63 observations) and evaluates the continuous GP posterior on all 250 official timestamps.
- A constrained direct-30-Hz fit is a sensitivity analysis, not a source of extra independent
  neural observations.

## Leakage controls

- GPFA is fit only to real neural responses from the official `train` tier.
- Response scaling is computed only from the applicable GPFA-fit training trials: one global
  standard deviation per neuron across trials and time. No `oracle` statistic and no per-timepoint
  variance normalization is used; GPFA's `d` parameter handles the mean.
- Train trials are divided into parameter-fit and calibration partitions before model fitting.
- Latent dimension, initialization, and conditional-prior decisions use calibration trials only.
- Official `oracle` repeated movies are used only after the GPFA is frozen.
- Model predictions never fit, rotate, scale, or otherwise modify the neural GPFA coordinates.

## Data classification

- A separate observation manifold is required for every mouse/session because neuron identities
  differ.
- The initial experiment contains natural-video trials only.
- Oracle movie conditions are recovered from grayscale stimulus similarity; behavior channels are
  explicitly excluded from condition identity.
- Individual movie identities do not receive separately fit observation manifolds.
- Future stimulus/behavior-conditioned fits must share `C`, `d`, and `R`; only temporal-kernel
  hyperparameters may vary unless a held-out analysis explicitly justifies a stronger model.
- Behavior-state candidates use balanced low/middle/high strata derived from training covariates
  alone. The simpler shared prior is retained whenever it lies within one standard error of the
  best held-out conditional score.

## GPFA model selection

- Gaussian observation model with diagonal neuron-specific noise.
- Shared linear observation parameters `C`, `d`, and `R`.
- Independent squared-exponential GP prior for each latent dimension.
- FA initialization and multiple result-blind, prespecified timescale initializations.
- Candidate dimensions: 4, 8, 12, and 16.
- Primary selection: held-out trial marginal likelihood, using the smallest dimension within one
  standard error of the best candidate.
- Reconstruction error and residual temporal autocorrelation are diagnostics, not selection targets.

## Reliability and nulls

For each oracle movie, balanced repeat halves are averaged independently. The frozen GPFA maps both
halves into one shared coordinate system; no Procrustes alignment is allowed.

Matched nulls:

1. condition derangement;
2. nonzero circular time shift;
3. full-frame permutation;
4. time reversal;
5. independent per-neuron circular shifts;
6. block permutations at 4, 8, 16, and 32 frames.

Metrics are reported separately: position correlation/cosine/error, velocity direction, speed
profile, path length, acceleration direction, and lagged correlation. No arbitrary composite score
is used. Reliability requires confidence-interval stability and separation from the relevant null,
not merely a high raw value.

## Saturation analysis

Reliability will be repeated across neuron count, latent dimension, GPFA fit-trial fraction, random
seed, temporal grid, and split count. A result is called saturated only when increasing the resource
changes the estimate by less than its bootstrap uncertainty and does not change its null-separation
decision.
