# Methods

The study evaluates Static and Dynamic neural encoding models at the response, output-space population-similarity, and neural-trajectory levels. This document defines the data, models, training, transformations, metrics, controls, and statistical summaries used in those evaluations.

## 1. Dataset and analysis scope

The project uses Dynamic Sensorium 2023 natural-movie stimuli, mouse V1 population responses, two behavioral covariates, and pupil-center measurements. Five official competition sessions are used for encoding-model training and full-sequence oracle evaluation:

| Session | Train trials | Oracle trials | Neurons |
|---|---:|---:|---:|
| `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20` | 348 | 58 | 7,863 |
| `dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 329 | 56 | 7,908 |
| `dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 60 | 8,202 |
| `dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 359 | 60 | 7,939 |
| `dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 59 | 8,122 |
| **Total** | **1,744** | **293** | **40,034** |

The official `train` tier is used for encoding-model and GPFA fitting. The official `oracle` tier supplies repeated movies for checkpoint evaluation, trajectory reliability, and model comparison. Hidden `final_test_main` responses are unavailable locally and are not used in the analyses reported here.

Detailed RSA, CKA, GPFA, and stress-test analyses use the first session in the table. A deterministic 512-neuron subset is selected by sorting stable unit IDs according to the SHA-256 digest of `"42:<unit_id>"` and retaining the first 512. This selection does not use response amplitude, neural reliability, or model performance.

The pilot session contains 58 oracle trials grouped into six repeated natural-movie conditions with repeat counts `10/10/9/10/9/10`. Conditions are recovered from grayscale stimulus similarity: fixed spatial/temporal stimulus subsamples are centered, normalized to unit length, connected at cosine similarity `>= 0.999`, and grouped by connected component. Behavior and neural responses are excluded from condition identification.

All detailed analyses use original frames 50–299, giving 250 timestamps per trial. The shared oracle tensor therefore has shape:

```text
[trial, time, neuron] = [58, 250, 512]
```

## 2. Common preprocessing and temporal alignment

Raw trials contain video, response, behavior, and pupil-center arrays with a longer padded/synchronization tail. The official `CutVideos(max_frame=None)` transform crops every modality to their common finite interval, yielding 300 aligned frames on the official 30 Hz grid.

For encoding-model input, one grayscale movie channel is combined with two behavioral traces broadcast over the `36 × 64` image grid:

```text
model input       [batch, 3, time, 36, 64]
responses         [batch, session_neurons, time]
behavior          [batch, 2, time]
pupil_center      [batch, 2, time]
```

Pupil center is not merged into the three core channels. It is passed separately to the session-specific shifter. The official `NeuroNormalizer(stats_source="all")` transformation is applied consistently to video, responses, behavior, and pupil center, with identical neuron ordering for all models.

During training, the official video loader extracts random valid 80-frame snippets. Full-sequence evaluation disables snippet cutting and passes each complete 300-frame trial through the model once; separate 80-frame predictions are not concatenated.

The Dynamic core uses valid temporal kernels of 11, 5, and 5 frames and therefore produces 18 fewer output frames than its input. For an 80-frame training snippet, its 62 outputs align to original response frames 18–79. The Static model initially produces one output for every input frame, then uses a zero-parameter adapter that retains frames 18–79 so that the training target support is identical.

For complete-trial inference:

```text
source input                            frames 0–299
Dynamic / adapted Static output         282 frames aligned to 18–299
official response burn-in               discard response frames 0–49
prediction end-crop                     retain final 250 outputs
final common evaluation support         original frames 50–299
```

The official 50-frame burn-in is distinct from the 18-frame structural reduction. Both models are evaluated on exactly the same trial identities, neurons, and original timestamps.

## 3. Encoding models

### 3.1 Static encoding model

Static processes each frame independently. The input is reshaped from `[B, 3, T, 36, 64]` to `[B×T, 3, 36, 64]`, passed through one shared 2D core, read out frame by frame, and reshaped back to `[B, T, N_session]`. It contains no temporal convolution, recurrence, frame mixing, or history buffer.

The four-layer core comprises:

| Layer | Operation | Channels | Kernel / padding |
|---|---|---:|---|
| 0 | 2D convolution | 64 | `9 × 9`, valid |
| 1–3 | pointwise → depthwise → pointwise blocks | 64 | depthwise `7 × 7`, padding 3 |

Layers use batch normalization and AdaptiveELU. The final core feature map for one frame is `[B, 64, 28, 56]`.

Each session has a `FullGaussian2d` readout with a cortical-coordinate grid predictor (`2 → 30 → 2 → tanh`), plus an MLP pupil shifter (`2 → 5 → 5 → 2`). The output nonlinearity is `ELU(x) + 1`. The parameter-free temporal adapter crops the first 18 framewise outputs solely to match Dynamic's valid-convolution support.

A frame-permutation implementation check permutes input frames and their pupil-center values, applies the model, reverses the output permutation, and verifies exact recovery of the original predictions. This checks that the reshape, normalization, readout, and shifter introduce no cross-frame dependence.

### 3.2 Dynamic encoding model

Dynamic uses a three-stage Factorized3D core. Each block applies a spatial convolution followed by an explicit temporal convolution:

| Block | Spatial kernel | Temporal kernel | Full-baseline channels | Reduced-control channels |
|---|---|---|---:|---:|
| 0 | `(1, 11, 11)` | `(11, 1, 1)` | 32 | 16 |
| 1 | `(1, 5, 5)` | `(5, 1, 1)` | 64 | 32 |
| 2 | `(1, 5, 5)` | `(5, 1, 1)` | 128 | 64 |

All convolutions use stride 1 and zero padding. The effective temporal receptive field is 19 frames, and an 80-frame input produces 62 feature timestamps. Blocks use BatchNorm3d, ELU, a first-spatial-layer Laplace penalty, and a first-temporal-layer regularizer.

Dynamic uses the same readout family, cortical-coordinate grid predictor, pupil-shifter architecture, behavioral inputs, and positive output nonlinearity as the Static pipeline, with Dynamic-specific readout initialization and regularization settings. It contains no GRU; temporal access is supplied by the three learned temporal convolutions.

The complete full-width Dynamic benchmark uses channels `[32, 64, 128]`. The model-comparison control uses `[16, 32, 64]`.

### 3.3 Total-parameter-matched Dynamic control

The Total-parameter-matched Dynamic model applies one predeclared `0.5` width multiplier to every Dynamic core stage while preserving the 1:2:4 widening rule, depth, spatial/temporal kernels, activations, normalization, regularizers, readout family, shifter, and output nonlinearity.

| Component | Static | Total-parameter-matched Dynamic | Difference |
|---|---:|---:|---:|
| Core | 50,624 | 98,672 | +48,048 |
| Readout | 2,763,106 | 2,763,106 | 0 |
| Shifter | 285 | 285 | 0 |
| Temporal adapter | 0 | — | 0 |
| **Total** | **2,814,015** | **2,862,063** | **+48,048 (+1.707%)** |

The control approximately matches total trainable parameter count while retaining a distinct Factorized3D core. Static and Total-parameter-matched Dynamic remain different complete architectures with different core parameter counts and convolutional operations.

Complete architecture audits remain available in the current [Static](supplementary/model_reports/STATIC_MODEL.md), [Dynamic](supplementary/model_reports/DYNAMIC_MODEL.md), and [total-parameter-matching](supplementary/model_reports/PARAMETER_MATCHED_DYNAMIC.md) reports.

## 4. Encoding-model training and checkpoint selection

Static, the full Dynamic benchmark, and Total-parameter-matched Dynamic use the official Dynamic Sensorium training framework. Static and the reduced control use the same five sessions, trial identities, neuron order, input resolution, normalization, training tiers, and temporal targets.

| Training item | Setting |
|---|---|
| Encoding-model seed | 42 |
| Trainer | `sensorium.training.video_training_loop.standard_trainer` |
| Maximum epochs | 200 |
| Training sample | Random valid 80-frame snippet |
| Physical batch | 8 snippets per session |
| Session accumulation | Five sessions |
| Effective batch | 40 snippets per optimizer step |
| Epoch structure | 225 session microbatches / 45 optimizer steps |
| Objective | Summed Poisson loss; `average_loss=false`, `scale_loss=true` |
| Optimizer | AdamW, learning rate `0.005` |
| AdamW settings | betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, AMSGrad off |
| Scheduler | ReduceLROnPlateau, factor `0.3`, up to four decay stages |
| Stopping settings | patience 5, absolute tolerance `1e-6`, minimum learning rate `1e-4` |
| Checkpoint rule | Restore the best state according to the official correlation closure |
| Numerical precision | FP32; mixed precision disabled |

`LongCycler` cycles the shorter session loaders to the length of the longest session. Gradients are accumulated sequentially over the five session microbatches before each optimizer update. Model-specific regularization remains part of the respective architecture configuration: Static uses its 2D input/readout penalties, whereas Dynamic uses the Factorized3D spatial/temporal penalties.

The published checkpoints are the best complete states restored by the official trainer. The full-width Dynamic benchmark is the procedural exception: its local run was stopped by project decision after validation at epoch 103, before natural official early-stop termination; the published best complete state is from epoch 97, and the partial epoch-104 state is not used. This status does not alter the Static versus Total-parameter-matched Dynamic training protocol used in the primary control.

Full-sequence oracle evaluation reconstructs each architecture, loads the released state dictionary strictly, processes each complete trial with batch size 1, retains original frames 50–299, and computes correlations with `sensorium.utility.scores.get_correlations`.

## 5. Prediction export and shared response tensor

Phase 4 reconstructs the Static and Total-parameter-matched Dynamic models using their locked configurations and published checkpoints. The official oracle loader is instantiated with `batch_size=1`, snippet cutting disabled, and zero offset. For each batch, the complete model output is reduced to its final 250 predictions, while recorded responses are explicitly indexed at frames 50–299.

The same locked 512 neuron indices are applied to predictions and responses. Export requires:

- identical dataloader sampler indices for Static and Dynamic inference;
- exact equality of the two recorded target tensors;
- exact correspondence with the locked oracle trial/condition map;
- identical neuron IDs/order and frame indices.

Any mismatch stops the pipeline. The exported archive contains:

```text
neural       [58, 250, 512]
static       [58, 250, 512]
dynamic      [58, 250, 512]
conditions   [58]
dataset indices, neuron indices/IDs, and frame indices
```

All conventional and trajectory comparisons use these aligned output-space tensors.

## 6. Response-level metrics

### Single-trial neuron response correlation

For each neuron, trial and time are flattened into one sample axis. Pearson correlation is computed between predicted and recorded responses, then summarized by the mean and median across neurons.

### Condition-average neuron response correlation

Repeats are averaged within each movie condition. Condition and time are flattened, per-neuron Pearson correlations are computed, and their mean and median are reported.

### Population-vector correlation

At every trial/time sample, Pearson correlation is computed across neurons between the predicted and recorded population vectors. The samplewise values are summarized by their mean and median. A condition-average version first averages repeats and then performs the same calculation at each condition/time sample; its Spearman analogue rank-transforms each population vector before correlation.

### Temporal-difference and lag diagnostics

First differences are taken along time after condition averaging. Trial and time are flattened and per-neuron correlations are computed on the difference tensors. Additional diagnostics include pooled zero-lag correlation, best pooled correlation over lags from −15 to +15 frames, normalized mean-squared error, pooled explained variance, and the ratio of predicted to recorded temporal standard deviation.

The five-session benchmark separately applies the official Sensorium correlation implementation to full-sequence predictions and reports the neuron-weighted mean together with session-specific values.

## 7. Output-space RSA and CKA

RSA and CKA operate on predicted and recorded neural population responses, not on hidden model activations. This keeps response, population-geometry, and trajectory analyses defined on the same output tensor.

### Representational similarity analysis

For population patterns `p_i` and `p_j`, the RDM uses correlation distance:

```text
d(i, j) = 1 − corr(p_i, p_j)
```

The upper triangles of recorded and predicted RDMs are compared by Spearman correlation (primary) and Pearson correlation (secondary). Implemented variants are:

1. **Condition RSA:** average repeats and time to obtain one population pattern per movie.
2. **Time-resolved RSA:** at each timestamp, construct an RDM across the six movie conditions, then average brain–model RDM similarity over time.
3. **Within-condition temporal RSA:** within each movie, sample states every 10 official frames, form a temporal RDM, compare recorded and predicted RDMs, and average across movies.
4. **Condition × time state RSA:** average repeats, sample every 10 frames, flatten condition and sampled-time into one state axis, and compare a single recorded/predicted RDM pair.

### Centered kernel alignment

Linear CKA is computed on mean-centered sample-by-neuron matrices `X` and `Y`:

```text
CKA(X, Y) = ||XᵀY||²_F / sqrt(||XᵀX||²_F ||YᵀY||²_F)
```

Implemented variants are:

1. **Single-trial time-aligned CKA:** flatten trial and time; if necessary, select at most 2,000 samples deterministically with seed 42.
2. **Condition-average time-aligned CKA:** average repeats and flatten condition/time.
3. **Within-condition temporal CKA:** compute CKA separately on each movie's time-by-neuron condition-average matrix.
4. **Temporal-difference CKA:** take first temporal differences of condition-average tensors and flatten condition/time.
5. **Condition-pattern CKA:** average repeats and time to obtain one pattern per movie.

## 8. Neural-data-defined GPFA

For trial `m`, time `t`, scaled neural response `y_m(t)`, and latent state `x_m(t)`, the observation model is:

```text
y_m(t) = C x_m(t) + d + ε_m(t)
ε_m(t) ~ Normal(0, R)
```

`C` is a shared neural loading matrix, `d` is the shared neuronal offset, and `R` is diagonal neuron-specific observation noise. Each latent dimension has an independent squared-exponential GP prior:

```text
k_j(t, t′) = exp[-0.5 ((t − t′) / τ_j)²]
```

The implementation performs exact linear-Gaussian posterior inference. Factor analysis initializes `C`, `d`, and `R`; expectation-maximization updates the observation parameters, and bounded scalar optimization updates each latent timescale. Lengthscales are constrained to `0.10–3.0 s`; observation-noise flooring and covariance jitter stabilize factorization.

The Phase 2 method-development fit uses at most 30 EM iterations with relative tolerance `1e-4`. The comparison-subset selection fits use at most 12 iterations, and its final refit uses at most 20 with tolerance `1e-6`.

GPFA parameters, train-derived scaling, neuron order, observation grid, training indices, and temporal support are frozen after fitting. Measurement reliability, null separation, sensitivity, and interpretation limits are reported in [GPFA Validation](GPFA_VALIDATION.md).

## 9. GPFA selection and the two frozen fits

### 9.1 Method-development GPFA

The method-development fit uses all 348 training-tier trials from the pilot session. A seed-42 shuffle assigns 278 trials to parameter fitting and 70 to calibration. Candidate dimensions `q = 4, 8, 12, 16` are scored by calibration marginal negative log likelihood per scalar observation. The smallest dimension within one standard error of the minimum is selected, yielding `q = 4` without using oracle reliability or encoding-model comparison performance.

At `q = 4`, initial timescales `0.125`, `0.25`, `0.5`, and `1.0 s` are compared on calibration likelihood. After dimensionality and initialization selection, the GPFA and train-derived preprocessing are refitted on all 348 training trials and frozen.

### 9.2 Comparison-subset GPFA

The model-comparison protocol deterministically selects 174 of the same session's 348 training trials using seed 42. A second shuffle divides this locked subset into 139 fit trials and 35 calibration trials. The primary dimension `q = 4` is retained from assay development rather than reselected using oracle or model-comparison results.

The four initial timescales are compared on the 35 calibration trials. After selecting the initialization, the exact GPFA used for Static–Dynamic trajectory evaluation is refitted on all 174 selected training trials and frozen.

These are two distinct fitted objects. Their roles and reliability are documented together in [GPFA Validation](GPFA_VALIDATION.md).

## 10. GPFA temporal sampling and prediction conversion

### Observation grid

Sensorium arrays use the official 30 Hz time grid, whereas the calcium signal was acquired at approximately 8 Hz before resampling. GPFA observes every fourth retained official frame:

```text
observation indices          0, 4, 8, ..., 248
observations per trial       63
effective observation rate   approximately 7.5 Hz
posterior query grid          all 250 timestamps at 30 Hz
```

The continuous GP posterior is queried at every evaluation timestamp. Velocity and acceleration computed on the 30 Hz query grid are derivatives of the inferred posterior trajectory, not independent 30 Hz neural measurements.

### Train-derived scaling and prediction conversion

For each selected neuron `n`, one standard deviation is computed across the applicable neural training trials and frames 50–299:

```text
std_n       = std(training response for neuron n)
floor       = 0.01 × mean_neuron_std
precision_n = 1 / max(std_n, floor)
```

The mean is handled by GPFA parameter `d`. No oracle response or model prediction contributes to this precision.

Encoding-model exports are already represented in the official `NeuroNormalizer` response scale. They are converted into the frozen train-only GPFA scale by:

```text
GPFA-scaled value
    = official-normalized value
      × train-only GPFA precision
      / aligned official normalization precision
```

For time-resolved official normalization, the denominator is indexed to original frames 50–299. This conversion is equivalent to restoring the applicable raw response scale and applying the train-derived scalar precision. Recorded oracle responses and both prediction tensors receive the same conversion and frozen neuron ordering before posterior inference.

## 11. GPFA reliability procedure

For each of 200 reliability splits and each of the six movies, oracle repeats are randomly divided into balanced, disjoint halves. Ten-repeat conditions contribute five trials per half; nine-repeat conditions contribute four per half and leave one unused. Each half is averaged independently, transformed by the same frozen GPFA, and compared without latent alignment.

For a matched-null draw, the right-half neural response is perturbed before GPFA inference while the left half remains unchanged:

| Null | Operation |
|---|---|
| Condition derangement | Permute six movie identities with no fixed points |
| Circular shift | Apply a nonzero time shift independently within each condition |
| Time reversal | Reverse the complete time axis |
| Independent-neuron shift | Circularly shift every neuron independently |
| Block shuffle | Permute 4-, 8-, 16-, or 32-frame blocks |
| Frame shuffle | Randomly permute all frames within each condition |

Metric direction is oriented so a positive paired difference means that the observed split is better than its null. Paired superiority is the fraction of splits with a positive oriented difference; split/null failure frequency is the fraction for which the observed value is not better.

The repeated splits reuse the same oracle trials and are not independent biological samples. Superiority and failure frequencies are descriptive robustness summaries, not formal independent-sample biological p-values. Model-prediction null checks similarly use 200 sampled transformations and are reported descriptively.

## 12. Trajectory metrics

Condition-average recorded and predicted responses are inferred through the frozen GPFA to produce:

```text
Z, Z_hat: [condition, time, latent] = [6, 250, 4]
dt = 1 / 30 seconds
```

For position metrics, a pooled latent mean is computed across both trajectories and subtracted from each. The metric battery is:

| Metric | Operational definition | Direction / content |
|---|---|---|
| Position correlation | Pearson correlation of the flattened pooled-centered trajectories | Higher; time-aligned latent state |
| Position cosine | Mean cosine of pooled-centered position vectors | Higher; local position orientation |
| Normalized position RMSE | `RMS(Z − Z_hat)` divided by the RMS scale of centered recorded `Z` | Lower; scale-normalized position error |
| Velocity-direction cosine | Mean cosine between `diff(Z)/dt` and `diff(Z_hat)/dt` | Higher; local direction |
| Speed-profile correlation | Pearson correlation between velocity norms | Higher; timing of fast/slow motion |
| Path-length similarity | `exp(−abs(log(mean predicted/recorded path ratio)))` | Higher; total traveled distance; descriptive only |
| Acceleration-direction cosine | Mean cosine between second temporal derivatives | Higher; local directional change |
| Zero-lag correlation | Flattened latent correlation at lag zero | Higher; pooled synchronous agreement |
| Best-lag correlation | Maximum flattened correlation over lags −15…+15 frames | Higher; lag-tolerant agreement |

Metrics are reported separately rather than combined into an arbitrary scalar. Path length is not used as primary evidence of temporal alignment; the validation basis for that restriction is given in [GPFA Validation](GPFA_VALIDATION.md).

## 13. Static–Dynamic trajectory evaluation

The comparison uses the frozen 174-trial comparison-subset GPFA:

```text
recorded response / Static prediction / Total-parameter-matched Dynamic prediction
    → identical selected neurons and order
    → identical frozen train-derived scaling
    → identical GPFA posterior inference
    → identical trajectory metrics
```

Static and Total-parameter-matched Dynamic predictions are treated as new observations `y(t)`. Their trajectories are inferred with fixed `C`, `d`, `R`, latent timescales, observation grid, and query times. The transform is the full Gaussian conditional-posterior inference, not multiplication by `Cᵀ` or a pseudoinverse.

No model-specific GPFA refit, Procrustes alignment, rotation, scale adjustment, or latent-axis selection is performed. Model-comparison findings are reported in [Results](RESULTS.md).

## 14. Stress-test methods

### 14.1 Response-matching stress test

Within each movie condition, repeats are split with seed `20260813` into disjoint selection and test halves. Each half contains 28 trials; the extra repeat from each nine-repeat condition is unused. Static's mean per-neuron response correlation on the selection half defines the matching target.

A fixed Gaussian noise tensor with seed `123` is generated at the shape of the intact Total-parameter-matched Dynamic prediction. Noise is scaled separately for each neuron by that neuron's prediction standard deviation over the locked trial/time tensor. One hundred amplitudes logarithmically spaced from `0.01` to `10` are evaluated:

```text
candidate = clip(dynamic + σ × neuron_scale × fixed_noise, 1e-5, infinity)
```

The amplitude whose selection-half mean per-neuron response correlation is closest to the Static target is retained. **Test-half neural responses are not used to select the response-matching perturbation strength.** Static and the response-score-matched Dynamic output are then evaluated on the test half using the response, RSA/CKA, and frozen-GPFA batteries.

This is a metric-sensitivity stress test, not a separately trained fair model comparison. It matches one scalar response summary and does not force individual-neuron scores, response variance, RSA, or CKA to match.

A secondary trained-checkpoint control selects Dynamic epoch 65 from the recorded five-session validation history by proximity to the Static validation correlation. Selection uses validation history rather than oracle trajectory metrics; this control remains distinct from the output-perturbation procedure.

### 14.2 Graded temporal-weight attenuation

The best Total-parameter-matched Dynamic state dictionary contains three learned temporal-convolution weight tensors. For each retention value:

```text
retention = 1.00, 0.75, 0.50, 0.25, 0.00
```

every off-center temporal slice is multiplied by the retention value, while each center temporal slice is multiplied by 1. Biases, spatial convolutions, normalization, nonlinearities, readout, and pupil shifter remain unchanged. Retention `0.00` removes the off-center temporal-kernel weights while preserving the central slices and the rest of the network.

Each altered state is loaded into the same architecture and evaluated on the locked oracle trials, neurons, and timestamps. Response correlation, condition-average CKA, condition × time RSA, and all trajectory metrics are recomputed. For monotonicity summaries, normalized RMSE is sign-reversed so all metrics are oriented as quality; severity–quality Spearman correlation and strict stepwise degradation are recomputed within paired condition-resampling draws.

The procedure perturbs learned temporal weights; it is not defined as a causal isolation of temporal computation.

### 14.3 Time reversal

The intact Total-parameter-matched Dynamic prediction is reversed along its 250-frame time axis independently within every trial:

```text
reversed_dynamic = dynamic[:, ::-1, :]
```

Condition-average population patterns, condition CKA/RSA, the conventional temporal battery, and frozen-GPFA trajectory metrics are recomputed against the same recorded response tensor. No model weights or neural responses are altered.

### 14.4 Conventional-metric sufficiency and leave-family-out regression

A fixed candidate set is constructed from Static, intact and validation-matched Dynamic predictions plus controlled prediction transformations. The 40 temporal/non-temporal diagnostic candidates span temporal-kernel attenuation, circular shifts, reversal, Gaussian temporal smoothing, block permutation, mixing with the time mean, per-neuron-scaled additive noise, and multiplicative per-neuron gain. All stochastic transformations use fixed family-specific seeds.

Each candidate is evaluated separately on the selection and test repeat halves. The conventional feature battery is:

```text
single-trial response correlation
condition-average response correlation
condition-average time-aligned CKA
temporal-difference CKA
condition × time state RSA
within-condition temporal RSA
```

The GPFA targets are position, velocity direction, speed profile, acceleration direction, and negative normalized RMSE (quality orientation). For each held perturbation family, a pipeline of `StandardScaler` followed by `Ridge(alpha=1.0)` is fitted on selection-half candidates from all other families. It predicts the held family's GPFA targets using that family's test-half conventional features. Predictions and observed test-half targets are pooled across held families to compute leave-family-out `R²` and error summaries.

A separate matched-pair diagnostic standardizes the six conventional features on selection repeats, searches across candidates from different perturbation families for the closest conventional-feature pair while preferring nontrivial GPFA separation, and evaluates the selected pair on untouched test repeats.

## 15. Statistical summaries and resampling

Resampling units are chosen separately for each analysis:

| Analysis | Resampling / summary unit |
|---|---|
| Five-session response comparison | Resample the five paired session differences; also report exact signs across sessions |
| Pilot neuron response | Resample paired neuron-level correlations |
| Pilot population-vector response | Resample paired trial × time values |
| Condition-level temporal CKA/RSA | Resample the six paired movie-condition differences; enumerate all condition-level sign flips where reported |
| Static–Dynamic trajectory comparison | Resample six movie conditions and recompute both models on each paired draw |
| Response-matched held-out response | Resample paired neuron-level response differences on the test repeat half |
| Response-matched trajectory comparison | Resample held-out movie conditions |
| Temporal-weight attenuation | Resample movie conditions and recompute every retention level on the same draw |

Unless otherwise specified, 2,000 percentile-bootstrap draws are used. For normalized position RMSE, signs are reversed when an oriented higher-is-better advantage is reported. The Q2 sign-flip summary enumerates all `2^6` sign assignments for the six condition differences.

Neuron-level and trial × time bootstraps are descriptive summaries of those axes; they are not reclassified as independent animals or sessions. Likewise, one-session condition bootstraps describe uncertainty across the six movies and do not establish dataset-wide inference across mice. No formal equivalence test or family-wise multiplicity correction is applied.

## 16. Frozen choices, seeds, and leakage controls

| Item | Frozen choice |
|---|---|
| Encoding-model training seed | 42 |
| Neuron-selection and primary protocol seed | 42 |
| GPFA fit/split seed | 42; seeds 314 and 2718 used only in sensitivity analysis |
| Single-trial CKA subsampling seed | 42 |
| Response-matching/Q6 repeat split | 20260813 |
| Response-matching noise seed | 123 |
| Detailed temporal support | Original frames 50–299 |
| Primary neuron set | Deterministic 512-unit order |
| GPFA dimensionality | `q = 4`, selected/retained without oracle model-comparison selection |
| GPFA observation/query grids | Every fourth frame / all 250 official timestamps |

Training/calibration indices, the 174-trial comparison subset, oracle trial/condition identities, neuron IDs/order, frame indices, checkpoint paths, and artifact hashes are stored in the phase locks and records. Oracle responses are excluded from GPFA fitting, scaling, dimensionality selection, and timescale-initialization selection. Model predictions never define or modify the GPFA coordinates.

The Phase 2 trajectory-reliability protocol was internally locked before its results were inspected; it was not externally preregistered. Metric definitions and model-comparison preprocessing are frozen before the final comparisons, and the pipeline fails closed when stored identities or tensor contracts do not match.

## 17. Reproducibility pointers

- [Data and Reproducibility Guide](DATA_AND_REPRODUCIBILITY.md): official data source, directory structure, environments, tests, and weight loading.
- Phase workflows: [Phase 1](../experiments/01_baselines/README.md), [Phase 2](../experiments/02_gpfa_reliability/README.md), [Phase 3](../experiments/03_parameter_matching/README.md), and [Phase 4](../experiments/04_model_comparison/README.md).
- Locked configurations: [Static](../experiments/01_baselines/configs/static_dynamic_sensorium2023.yaml), [full Dynamic](../experiments/01_baselines/configs/phase1A_dynamic_official.yaml), [Total-parameter-matched Dynamic](../experiments/03_parameter_matching/configs/dynamic_parameter_matched.yaml), [method-development GPFA](../experiments/02_gpfa_reliability/configs/pilot.yaml), and [model comparison](../experiments/04_model_comparison/configs/pilot.yaml).
- Released artifacts: [encoding-model and frozen-GPFA objects](../models/).
- Scientific findings: [Results](RESULTS.md) and [detailed Q1–Q6 evidence](results/Q1_Q6_ANSWERS.md).
- Measurement validation: [GPFA Validation](GPFA_VALIDATION.md).
- Design logic: [Design Rationale](DESIGN_RATIONALE.md).
