# Phase 2 GPFA trajectory-reliability result

## Decision

**PASS as a one-session method-development validation, with metric-specific restrictions.**

The frozen neural GPFA produces highly repeatable position, velocity-direction, speed-profile, and
acceleration-direction trajectories for repeated natural movies, and these metrics reject matched
stimulus, time-order, timing, and population-synchrony nulls in every tested configuration. Path
length is reproducible but is not a reliable temporal-alignment metric.

This is not yet an all-five-session biological conclusion and does not rank the Dynamic and Static
encoding models.

## Data and leakage controls

- Session: `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20`.
- 7,863 recorded neurons; a deterministic nested 512-neuron subset is primary.
- GPFA fit/calibration: 278/70 trials from the official `train` tier.
- Final frozen GPFA refit: all 348 official train trials.
- Reliability: 58 official `oracle` trials grouped into six grayscale natural movies with repeat
  counts 10/10/9/10/9/10.
- No oracle response, encoding-model prediction, or trajectory score enters GPFA fitting or model
  selection.
- Response scaling is estimated from the applicable training trials only: one standard deviation
  per neuron across all training trials and time. The official `all` per-timepoint response
  statistics are not used for GPFA fitting.
- Analysis interval: original frames 50--299, 250 timestamps at 30 Hz (8.33 seconds).
- Primary GPFA observations: every fourth official frame, 63 points at 7.5 Hz; the continuous GP
  posterior is evaluated on all 250 official timestamps.

## Selected GPFA

- Exact linear-Gaussian GPFA fitted by EM; diagonal neuron-specific observation noise.
- Squared-exponential temporal prior, one kernel per latent dimension.
- Candidate dimensions: 4, 8, 12, 16.
- Selection criterion: calibration marginal NLL with the smallest dimension within one standard
  error of the best model.
- Selected latent dimension: 4.
- Selected initialization: 0.25 seconds.
- Learned timescales: 0.2505, 0.2509, 0.2505, and 0.2506 seconds.
- Parameter digest: `789771cab85943288c022a19a2f571a1704fe0e0af547465e9a06b6ef9051180`.

Although the raw mean calibration NLL decreases through 16 dimensions, all candidates lie within
one standard error; the conservative 4-dimensional model is therefore selected. Latent-dimension
sensitivity remains part of the saturation result below.

## Primary split-half reliability

The table reports the distribution across 200 balanced repeat splits. The interval is the 2.5th to
97.5th percentile of split values, not a confidence interval for the mean.

| Metric | Split-half mean | Split distribution 95% interval |
|---|---:|---:|
| Position correlation | 0.8566 | 0.8184--0.8885 |
| Normalized position RMSE | 0.5428 | 0.4789--0.6527 |
| Velocity-direction cosine | 0.6627 | 0.6250--0.6949 |
| Speed-profile correlation | 0.7434 | 0.6791--0.7951 |
| Path-length similarity | 0.9449 | 0.8584--0.9978 |
| Acceleration-direction cosine | 0.6163 | 0.5824--0.6548 |

For position correlation, velocity direction, speed profile, acceleration direction, and normalized
position error, the observed value beat every condition-shuffle, circular-shift, time-reversal,
16-frame block-shuffle, and independent-neuron-shift null in all 200 paired splits. The conservative
finite-sample paired p-value is 1/201 = 0.00498 for each comparison.

Representative position-correlation null means:

| Null | Mean |
|---|---:|
| Condition derangement | -0.0538 |
| Circular time shift | 0.1385 |
| Time reversal | 0.1949 |
| 16-frame block permutation | 0.2433 |
| Independent per-neuron time shift | 0.3733 |

## What the metrics capture

### Position correlation and normalized position error

They detect stimulus identity and time-aligned population state. They reject movie-condition
derangement, global timing changes, local window reordering, and destruction of coordinated neural
population structure. They do not, by themselves, distinguish the direction or derivative structure
of motion through the latent space.

### Velocity and acceleration direction

They detect local order and direction. Their matched-null values are near zero for circular shifts,
frame/block shuffles, and independent-neuron shifts. They are therefore appropriate tests of whether
an encoding model reproduces local latent dynamics rather than only visiting similar states.

### Speed-profile correlation

It detects when the neural trajectory changes rapidly or slowly. It rejects condition and temporal
nulls, while discarding direction information. It complements velocity-direction cosine.

### Path length

Path-length similarity is high for the real split halves, but it does not reject circular shift
(`paired superiority = 0.63`, `p = 0.373`) or time reversal (`0.445`, `p = 0.557`). This is expected:
both operations preserve nearly the same total distance traveled. Path length may be reported as a
descriptive magnitude diagnostic, but it must not be used as a primary trajectory-alignment score.

### Independent-neuron shift

All primary geometry/dynamics metrics drop strongly when each neuron is shifted independently. The
metric battery therefore captures coordinated population activity, not merely each neuron's marginal
autocorrelation.

## Condition-specific temporal priors

The natural-video training data were divided into balanced low/middle/high behavior-state strata
using the first principal component of behavior covariates; neural responses were excluded from the
classification. Counts were 93/92/93 for fit and 22/30/18 for calibration.

The shared-prior calibration NLL per observation was 1.4031800295; the behavior-conditioned value was
1.4031800335. The conditional model did not improve held-out fit and the shared model is comfortably
within one standard error. The selected model therefore keeps one natural-video temporal prior.

The implementation supports condition-specific timescales with shared `C`, `d`, and `R`, but the
extra parameters are not enabled without held-out evidence. Individual movie identities never receive
separate GPFA coordinate systems.

## Saturation validation

### Split count

Position-correlation estimates were 0.8542, 0.8541, 0.8532, 0.8542, and 0.8559 at 25, 50, 100, 200,
and 500 splits in the dedicated convergence run; the standard error of the mean fell from 0.0037 to
0.0009. Velocity direction changed from 0.6655 to 0.6614 between 25 and 500 splits, with final standard
error 0.0009. Correlation-like estimates are effectively stable by 100--200 splits; 500 splits are
retained as the saturated reference.

### Neuron count

| Neurons | Position corr. | Velocity cosine | Speed corr. |
|---:|---:|---:|---:|
| 128 | 0.8162 | 0.5593 | 0.6536 |
| 256 | 0.8437 | 0.5920 | 0.7407 |
| 512 | 0.8532 | 0.6634 | 0.7403 |
| 1024 | 0.8594 | 0.6710 | 0.7696 |

Position and velocity reliability are close to a plateau at 512--1024 neurons (changes 0.0062 and
0.0076). Speed-profile magnitude still changes by about 0.029, so its exact numerical ceiling is not
fully neuron-saturated even though its null-separation decision is stable.

### Latent dimension

Across 4/8/12/16 dimensions, position correlation spans 0.8465--0.8538 and velocity cosine spans
0.6374--0.6634, with no monotonic improvement. Speed-profile correlation increases from 0.7403 at
four dimensions to 0.7935 at 16 dimensions. Thus position/velocity conclusions are dimension-stable;
speed magnitude must always be reported with the chosen GPFA dimension.

### Training fraction and random seed

Across 25/50/75/100% train fractions, position correlation spans 0.8410--0.8551. Across three seeds it
spans 0.8532--0.8622. No reliability/null decision changes. Derivative metrics vary more than position
metrics, so their uncertainty must include GPFA-fit seeds in later model comparisons.

### Temporal observation grid

The 7.5-Hz-equivalent and direct-30-Hz fits give position correlations 0.8532 and 0.8475. Velocity
cosines are 0.6634 and 0.6190. Null separation is unchanged, but derivative magnitude is not sampling-
grid invariant. The 7.5-Hz-equivalent analysis remains primary because the neural signal was acquired
at approximately 8 Hz before Sensorium upsampling; the 30-Hz result is a required sensitivity check.

## Reliability gate for subsequent model evaluation

Use as primary, reported separately:

1. position correlation or normalized position error;
2. velocity-direction cosine;
3. speed-profile correlation.

Use acceleration direction as a higher-variance diagnostic. Use path length only as a descriptive
control. Do not combine these metrics into an arbitrary scalar.

For each encoding model, apply the frozen GPFA and frozen train-only preprocessing; do not refit or
align latent axes on model predictions. Report results at the selected 4-dimensional GPFA and repeat
the key comparison at 8/12/16 dimensions and across GPFA seeds.

## Scope limitation and next decision

The metric battery is sufficiently reliable to proceed to a **held-out Dynamic-versus-Static model
prediction test for this session**. Before making a dataset-wide scientific claim, repeat the neural
split-half gate across the other four official sessions and meta-analyze session-level effects.
