# Methods

## Overview

This study compares a frame-wise Static encoding model with a parameter-matched Dynamic encoding model on Dynamic Sensorium 2023. The evaluation is organized into three levels:

1. **Response prediction:** agreement between predicted and recorded neural responses.
2. **Output-space population-response similarity:** RSA and linear CKA applied to predicted and recorded neural population-response patterns.
3. **Neural trajectories:** agreement after recorded and predicted responses are projected through a frozen GPFA fitted only to neural training data.

Model capacity, response quality, temporal history, data alignment, and latent-space definition are controlled separately. The full response benchmark uses all five official sessions. The detailed RSA, CKA, GPFA, response-matching, and temporal-ablation analyses are a method-development study on one session with 512 neurons and six repeated natural movies.

Unless otherwise stated, random seed 42 is used for data locking and primary analysis, and all model comparisons use frozen neural-network checkpoints.

## Data preprocessing and temporal alignment

### Dataset and primary analysis session

The data are from Dynamic Sensorium 2023 and contain natural movies, mouse V1 population responses, pupil position, and behavioral covariates. The primary trajectory-analysis session is:

```text
dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20
```

This session contains 7,863 recorded neurons and 348 official training trials. Its oracle tier contains 58 repeated trials grouped into six natural-movie conditions with repeat counts:

```text
10 / 10 / 9 / 10 / 9 / 10
```

The five-session response comparison additionally uses the other four official Dynamic Sensorium sessions and their complete neuron sets.

### Locked neurons

The detailed analysis uses a deterministic 512-neuron subset. Neurons are ordered by the SHA-256 digest of the string formed from seed 42 and the stable unit identifier, and the first 512 are retained. Selection is therefore independent of response amplitude, response reliability, model prediction quality, and trajectory results. The same locked neuron identities and order are used for recorded responses, Static predictions, Dynamic predictions, GPFA fitting, and all control analyses.

### Trial and condition locking

Oracle movie conditions are recovered from the stimulus rather than from behavior or neural responses. Each grayscale video is spatially and temporally subsampled, centered, normalized to unit length, and compared by cosine similarity. Trials are connected into the same condition when stimulus similarity is at least 0.999. Behavior channels are excluded from this operation.

The resulting trial indices, condition labels, neuron identities, and selected training indices are stored with cryptographic fingerprints in the Phase 4 protocol lock. Static and Dynamic prediction generation must reproduce the locked trial order and target tensor exactly; the pipeline stops if either differs.

### Temporal interval

Each source trial contains 300 aligned stimulus and response frames on the official 30 Hz grid. Only original frames 50 through 299 are evaluated:

```text
source trial                     300 frames
discarded onset interval         frames 0-49
evaluation interval              frames 50-299
retained timestamps              250
nominal duration                 250 / 30 = 8.33 s
```

The Dynamic core uses valid temporal convolutions and has an 18-frame structural reduction. Both models are nevertheless evaluated on the same original interval. For every trial, the final 250 temporally aligned predictions are retained. The Static model generates frame-wise predictions and applies the same output crop; the crop introduces no temporal parameters. The exported analysis tensor has shape:

```text
[trial, time, neuron] = [58, 250, 512]
```

### Prediction scale and neural targets

Neural-network prediction and conventional evaluation use the official Sensorium `NeuroNormalizer` scale. Static and Dynamic checkpoints are reconstructed with the same official loader, and the recorded target tensors produced during the two inference passes must be bitwise identical.

GPFA uses a separate train-only scaling rule to prevent oracle leakage. For each selected neuron, one standard deviation is computed across the applicable neural training trials and frames 50-299. Standard deviations below 1% of the mean neuronal standard deviation are floored at that threshold. The reciprocal standard deviation is used as the GPFA precision. No oracle response and no model prediction contributes to this scale.

Before an officially normalized response or prediction is passed to GPFA, it is converted to the frozen train-only GPFA scale:

```text
GPFA-scaled value
    = official-normalized value
      x train-only precision
      / official normalization precision
```

For time-resolved official normalization, the conversion factor is aligned to the same frames 50-299. Recorded responses and both model predictions receive exactly the same conversion.

## Response and output-space RSA/CKA

All conventional metrics compare model outputs with recorded population responses. They do not compare hidden neural-network layers.

The terms **output-space RSA/CKA** and **population-response representation similarity** are used throughout this project. Both analyses operate on predicted neural responses `Y_hat(t)` and recorded neural responses `Y(t)`, not on hidden model activations `H(t)`. This is intentional: response correlation, population-geometry metrics, and GPFA trajectories all evaluate the same model output tensor. Hidden-layer RSA/CKA would require layer selection and an additional correspondence decision and is outside the primary comparison.

### Response correlation

The primary local response score is the mean per-neuron Pearson correlation. For each neuron, trials and timestamps are flattened into one sample axis, Pearson correlation is computed between predicted and recorded responses, and correlations are averaged across neurons. The median per-neuron correlation is also reported.

For condition-average response scores, repeated trials are first averaged within each movie condition. Condition and time are then flattened before computing per-neuron correlations. This reduces repeat-specific noise and matches the primary unit used in the trajectory analysis.

Additional response diagnostics include:

- population-vector correlation across neurons at each trial-time sample;
- condition-average population-vector Pearson and Spearman correlations;
- correlation of first temporal differences for each neuron;
- pooled zero-lag and best-lag correlations, with lags from -15 to +15 frames;
- normalized mean-squared error and pooled explained variance;
- the ratio of predicted to recorded temporal standard deviation.

The five-session oracle benchmark uses the official `sensorium.utility.scores.get_correlations` implementation on full sequences and reports the neuron-weighted all-session single-trial mean. Session-specific scores are retained so session can be used as the inferential unit for Q1.

### Representational similarity analysis

RSA uses correlation distance between population patterns:

```text
d(i, j) = 1 - corr(pattern_i, pattern_j)
```

Distances form a representational dissimilarity matrix (RDM). Two RDMs are compared using their upper triangles, with Spearman correlation as the primary comparison and Pearson correlation as a secondary value.

Four RSA variants are computed:

1. **Condition RSA.** Repeats are averaged within condition, responses are then averaged across time, and one population pattern is obtained for each movie.
2. **Time-resolved RSA.** At each timestamp, an RDM is formed across the six movie conditions. Brain-model RDM correlations are averaged over time.
3. **Within-condition temporal RSA.** Within each movie, time states sampled every 10 official frames are treated as patterns. Brain-model temporal RDM similarity is computed per movie and averaged across conditions.
4. **Condition-by-time state RSA.** Condition-average population states are sampled every 10 frames, the condition and time axes are flattened into one state axis, and a single RDM is compared between brain and model.

The 10-frame stride reduces redundant state pairs and computational cost; at the official grid it corresponds to approximately 3 Hz sampling of the RDM states. The primary summary in the README is the Spearman condition-by-time state RSA, while the Q2 condition-level analysis emphasizes within-condition temporal RSA.

### Centered kernel alignment

Linear CKA is computed on mean-centered sample-by-neuron matrices. For brain matrix `X` and model matrix `Y`:

```text
CKA(X, Y) = ||X'Y||_F^2
            / sqrt(||X'X||_F^2 ||Y'Y||_F^2)
```

The analysis includes:

1. **Single-trial time-aligned CKA:** trial and time are flattened; at most 2,000 samples are selected deterministically with seed 42.
2. **Condition-average time-aligned CKA:** repeats are averaged, then condition and time are flattened.
3. **Temporal-difference CKA:** first temporal differences of the condition-average response tensors are flattened and compared.
4. **Condition-pattern CKA:** each movie is represented by its population response averaged across repeats and time.

For condition-level inference in Q2, temporal CKA is also computed independently for each of the six repeated movies using its time-by-neuron condition-average matrix. Temporal-difference CKA is computed on the corresponding first differences.

## GPFA and trajectory metrics

### Training data and leakage control

The model-comparison GPFA is fitted on a deterministic 50% subset of the primary session's official training tier: 174 of 348 trials. The selected subset is split into 139 parameter-fit trials and 35 calibration trials. Oracle responses are excluded from preprocessing, fitting, initialization selection, and model selection. Encoding-model predictions are never used to fit or align GPFA.

During assay development, latent dimensions 4, 8, 12, and 16 were evaluated in a neural training/calibration model-selection phase. Each candidate was fitted on neural training-fit trials and scored by marginal negative log likelihood per scalar observation on held-out neural calibration trials. The lowest mean calibration NLL occurred at `q = 16` (1.39161), with standard error 0.03626. The one-standard-error threshold was therefore 1.42786; all candidates fell within it, and the smallest eligible dimension, `q = 4`, was selected. Oracle reliability, null separation, and encoding-model performance did not enter this choice. The model-comparison protocol locks that preselected `q = 4` rather than reselecting dimension on the comparison data.

At `q = 4`, GPFA initial lengthscales of 0.125, 0.25, 0.5, and 1.0 seconds were compared by calibration marginal negative log likelihood; 0.25 seconds had the lowest value. After initialization selection, the final model-comparison GPFA is refitted on all 174 selected training trials. Oracle responses are used only after freezing, for reliability and model evaluation.

### Temporal observation grid

Although the files are represented at 30 Hz, the neural signal was acquired at approximately 8 Hz before upsampling. GPFA therefore observes every fourth official frame:

```text
observation indices              0, 4, 8, ..., 248
observations per trial           63
effective observation rate       7.5 Hz
posterior query grid              all 250 timestamps at 30 Hz
```

The continuous GP posterior is queried at every official timestamp after fitting. A direct 30 Hz fit is retained as a sensitivity analysis rather than treated as additional independent neural information. Velocity and especially acceleration at the 30 Hz query grid are derivatives of the inferred GP posterior trajectory, not independently observed 30 Hz neural dynamics. Higher-order derivative conclusions are therefore interpreted conservatively and checked against observation-grid sensitivity.

### GPFA model

For trial `m` and time `t`, the observation model is:

```text
y_m(t) = C x_m(t) + d + epsilon_m(t)
epsilon_m(t) ~ Normal(0, R)
```

`y_m(t)` is the scaled neural response, `x_m(t)` is the four-dimensional latent state, `C` is a shared loading matrix, `d` is the neuronal offset, and `R` is diagonal neuron-specific observation noise.

Each latent dimension has an independent squared-exponential Gaussian-process prior:

```text
k_j(t, t') = exp[-0.5 ((t - t') / tau_j)^2]
```

The implementation performs exact linear-Gaussian posterior inference and expectation-maximization. Factor analysis initializes `C`, `d`, and `R`; EM updates the observation parameters and bounded scalar optimization updates each latent timescale. Selection fits use at most 12 EM iterations, the final fit uses at most 20, relative tolerance is `1e-6`, lengthscales are bounded to 0.10-3.0 seconds, and covariance jitter is `1e-5`.

After fitting, GPFA parameters, train-only scaling, neuron order, observation grid, training indices, and time interval are frozen. Static and Dynamic predictions are treated as new observation sequences `y(t)`. Their latent trajectories are obtained by applying the same Gaussian conditional-posterior inference used for neural responses, with fixed `C`, `d`, `R`, latent timescales, and observation grid. This is not a simple multiplication by `C` transpose or a pseudoinverse. No model-specific GPFA, rotation, scaling, Procrustes transform, or latent-axis selection is permitted.

### Condition-average trajectories

Repeated oracle trials are averaged within movie condition before the primary model comparison. Recorded responses and each model prediction then produce arrays with shape:

```text
[condition, time, latent] = [6, 250, 4]
```

The following metrics compare the recorded trajectory `Z` with predicted trajectory `Z_hat`. The time step for derivatives is `dt = 1/30` seconds.

| Metric | Definition | Direction |
|---|---|---|
| Position correlation | Pearson correlation after both trajectories are centered by their pooled latent mean | Higher is better |
| Position cosine | Mean cosine similarity of pooled-centered latent position vectors | Higher is better |
| Normalized position RMSE | Position RMSE divided by the RMS scale of the centered recorded trajectory | Lower is better |
| Velocity-direction cosine | Mean cosine similarity of first-difference vectors divided by `dt` | Higher is better |
| Speed-profile correlation | Pearson correlation between velocity magnitudes | Higher is better |
| Path-length similarity | `exp(-abs(log(predicted path / recorded path)))` | Higher is better |
| Acceleration-direction cosine | Mean cosine similarity of second temporal derivatives | Higher is better |
| Zero-lag correlation | Pooled latent correlation at zero lag | Higher is better |
| Best-lag correlation | Maximum pooled correlation from -15 to +15 frames | Higher is better |

Position correlation or normalized position RMSE, velocity direction, and speed profile are reported separately as the primary trajectory battery. Acceleration direction is a higher-variance diagnostic. Path length is descriptive only because circular shifts and time reversal can preserve total traveled distance. Metrics are not combined into an arbitrary composite score.

## Reliability, null tests, and bootstrap inference

### Split-half neural reliability

Trajectory reliability is established using recorded oracle responses before model predictions are evaluated. For each of 200 splits and each movie condition, repeats are randomly divided into two balanced, disjoint halves. Conditions with 10 repeats contribute five trials per half; conditions with nine repeats contribute four per half and leave one repeat unused for that split. Responses are averaged within each half, passed independently through the same frozen GPFA, and compared without latent alignment.

The reported 2.5th and 97.5th percentiles describe the empirical distribution of split values. They are not confidence intervals for the split-distribution mean.

### Reliability nulls

For every split, the right-half condition-average response is perturbed before GPFA projection while the left half remains unchanged. The nulls are:

| Null | Operation | Information disrupted |
|---|---|---|
| Condition derangement | Permute movie identities with no fixed points | Stimulus identity |
| Circular shift | Apply a nonzero time shift independently per condition | Absolute timing and phase |
| Frame shuffle | Randomly permute all frames within each condition | Local and global temporal order |
| Time reversal | Reverse the full sequence | Temporal direction while preserving visited states and approximately preserving path length |
| Independent-neuron shift | Circularly shift every neuron independently | Coordinated population timing |
| Block shuffle | Permute blocks of 4, 8, 16, or 32 frames | Temporal organization at graded scales |

Observed and null values are paired by split. Metric direction is oriented so positive values always mean that the observed trajectory is better than the null. Paired superiority is the fraction of splits with positive oriented difference. The finite-sample paired p-value is:

```text
p = (1 + number of non-superior observed splits) / (number of splits + 1)
```

With 200 splits, the minimum value is `1/201`. Standardized separation divides the mean oriented difference by the pooled observed/null standard deviation.

The same reliability logic is repeated across neuron count, latent dimension, GPFA training fraction, GPFA seed, observation grid, and split count. Model-specific null tests use 200 perturbations per model for condition shuffle, circular shift, time reversal, 16-frame block shuffle, and independent-neuron shift.

### Bootstrap units

Bootstrap samples preserve the dependence structure appropriate to each question:

- **Five-session response gain:** the five session-level Dynamic-Static differences are resampled; exact sign probabilities are also reported.
- **Primary response comparison:** paired neurons are resampled for per-neuron response correlation; trial-time samples are resampled for population-vector correlation.
- **Condition-level RSA and CKA:** the six paired movie-condition differences are resampled.
- **Trajectory model comparison:** the six movie conditions are resampled, and Static and Dynamic are recomputed on the same bootstrap draw.
- **Response-matched response score:** paired neuron-level differences on the held-out repeat split are resampled.
- **Response-matched trajectory scores:** held-out movie conditions are resampled.
- **Temporal-ablation monotonicity:** movie conditions are resampled and every retention level is recomputed on the same draw.

Unless noted otherwise, 2,000 bootstrap samples are used. Intervals are percentile 95% intervals. For lower-is-better normalized RMSE, the sign is reversed when reporting an oriented Dynamic advantage. Q2 additionally enumerates all sign flips of the six condition differences for a one-sided exact sign-flip p-value.

These analyses are metric-specific and exploratory at the one-session stage. No family-wise multiplicity correction or formal equivalence test is applied. The intervals should not be interpreted as dataset-wide inference across mice.

## Response matching and temporal ablation

### Response-matched stress test

Response matching tests whether trajectory metrics still distinguish temporal predictions when the primary scalar response score is nearly the same.

Within each of the six movie conditions, repeats are randomly divided with seed 20260813 into disjoint selection and test halves. Each half contains 28 trials in total; the unused repeat from each nine-repeat condition is excluded. The Static response score on the selection half is the target.

A fixed Gaussian noise tensor is generated with seed 123. Noise is scaled separately for each neuron by the standard deviation of the intact Dynamic prediction across trials and time. This makes the perturbation approximately proportional to each neuron's predicted dynamic range instead of imposing a common absolute noise scale across heterogeneous neurons. One hundred noise amplitudes logarithmically spaced from 0.01 to 10 are evaluated on the selection half:

```text
candidate = clip(dynamic + sigma x neuron_scale x fixed_noise, 1e-5, infinity)
```

The amplitude whose mean per-neuron response correlation is closest to the Static target is selected. The test repeats do not participate in this selection. Static and response-matched Dynamic predictions are then compared on the held-out test half with the complete conventional battery and frozen GPFA.

This procedure matches only the scalar mean per-neuron response correlation. It does not force RSA, CKA, response variance, or every individual-neuron score to match, and it is not a formal equivalence test. Its purpose is to test whether scalar response correlation is sufficient to explain the observed trajectory difference.

### Training-history response match

A complementary control selects a naturally trained Dynamic checkpoint from the recorded validation history. Dynamic epoch 65 has validation correlation 0.163885, compared with 0.163956 for the Static checkpoint. Selection uses the official five-session validation score rather than oracle trajectory results. Because its oracle response score is not as closely matched as the disjoint-repeat stress test, it is treated as supporting rather than primary evidence for Q4.

### Temporal-history ablation

Temporal ablation directly manipulates the best parameter-matched Dynamic checkpoint. For every learned temporal-convolution weight tensor, all off-center temporal slices are multiplied by a retention factor. The center slice is left unchanged. Temporal biases, spatial kernels, normalization, nonlinearities, readout, shifter, and all other parameters remain fixed.

Five retention levels are evaluated:

```text
retention                1.00   0.75   0.50   0.25   0.00
ablation severity        0.00   0.25   0.50   0.75   1.00
```

Retention 1.00 is the intact model. Retention 0.00 removes all off-center temporal-kernel weights but preserves the center slices and the rest of the network. All three temporal-convolution layers are manipulated simultaneously.

Each ablated model is run on the same locked oracle trials, neurons, and timestamps. Response correlation, condition-average CKA, condition-by-time RSA, and every GPFA trajectory metric are recomputed. For monotonicity analysis, lower-is-better RMSE is multiplied by -1 so that all quantities are expressed as quality. Strict monotonic degradation requires quality to decrease at every increase in severity. Spearman correlation between severity and quality is also reported. Condition-resampling bootstrap draws provide the distribution of Spearman correlation and the proportion of draws with strict monotonic degradation.

This intervention establishes a graded dose-response test of sensitivity to distributed learned temporal history. It does not identify which temporal layer, kernel lag, or biological timescale is responsible; those questions require layer-specific and lag-specific ablations.

The present experiment does not include a magnitude-matched non-temporal damage control. Progressive degradation could therefore partly reflect generic disruption of trained weights rather than selective loss of temporal history. A planned control will perturb spatial weights or temporal center slices by a matched parameter-space magnitude while preserving off-center temporal history, then compare the resulting response, output-space RSA/CKA, and trajectory degradation with the temporal-ablation curve. This limitation does not invalidate the current proof-of-concept dose-response result, but it constrains claims of temporal specificity.

## Analysis scope and reproducibility

The principal inferential units are sessions for the five-session response result and movie conditions for the one-session output-space geometry and trajectory analyses. Condition-average trajectories are primary because their neural split-half reliability is substantially higher than single-trial reliability. Single-trial response analyses remain important and are reported separately.

All paths, fixed parameters, selected trial and neuron identities, seeds, checkpoint hashes, compact result tables, and frozen GPFA objects are retained in the repository. The detailed implementation is organized under:

- [`experiments/01_baselines/`](../experiments/01_baselines/)
- [`experiments/02_gpfa_reliability/`](../experiments/02_gpfa_reliability/)
- [`experiments/03_parameter_matching/`](../experiments/03_parameter_matching/)
- [`experiments/04_model_comparison/`](../experiments/04_model_comparison/)

Scientific motivation and result interpretation are separated from this implementation description in:

- [Design Rationale](DESIGN_RATIONALE.md)
- [GPFA Reliability Results](results/GPFA_RELIABILITY_RESULTS.md)
- [Model Comparison Results](results/MODEL_COMPARISON_RESULTS.md)
- [Q1-Q6 Answers](results/Q1_Q6_ANSWERS.md)
