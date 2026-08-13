# Brain-Based GPFA Trajectory Evaluation: Design, Reliability, and Validation Report

Updated: 2026-08-11

## Executive Summary

In this project, I trained a frozen Gaussian-process factor analysis (GPFA) coordinate system using real neural responses from Dynamic Sensorium 2023 to evaluate whether an encoding model can not only predict individual-neuron responses, but also reproduce the low-dimensional trajectories along which large neural populations evolve over time.

The current validation uses session `dynamic29515-10-12` as the method-development session. The primary analysis deterministically selects 512 neurons from 7,863 neurons. GPFA is fit only on 348 official train trials; the 58 oracle trials are used only for reliability testing after the model has been frozen. For each trial, only original frames 50–299 are analyzed, corresponding to 250 frames or 8.33 s. Because the calcium signal was acquired at approximately 8 Hz while the official data were upsampled to 30 Hz, the primary GPFA uses one observation every four official frames, yielding 63 observations at an effective rate of 7.5 Hz, and then queries the continuous GP posterior at all 250 timestamps.

The frozen primary model is a 4-dimensional GPFA. Across 200 balanced split-half repetitions over six repeated natural movies, the results are:

| Metric | Mean | 95% split interval |
|---|---:|---:|
| Position correlation | 0.8566 | 0.8184–0.8885 |
| Normalized position RMSE | 0.5428 | 0.4789–0.6527 |
| Velocity-direction cosine | 0.6627 | 0.6250–0.6949 |
| Speed-profile correlation | 0.7434 | 0.6791–0.7951 |
| Path-length similarity | 0.9449 | 0.8584–0.9978 |
| Acceleration-direction cosine | 0.6163 | 0.5824–0.6548 |

Position, normalized error, velocity, speed, and acceleration outperform condition shuffle, circular shift, time reversal, 16-frame block shuffle, and independent-neuron shift in all 200 paired splits. The finite-sample paired p-value is `1/201 = 0.004975`. Although path length has high split-half reliability, it cannot reject circular shift or time reversal, so it should be treated only as a descriptive quantity rather than a primary temporal-alignment metric.

My final assessment is:

> Brain-based GPFA trajectory evaluation has passed the single-session methodological reliability gate. It can detect stimulus identity, time alignment, local direction, speed modulation, and coordinated population structure. However, it captures only a relatively small but repeatable shared subspace of the neural response. It is therefore appropriate as a complementary metric to response correlation, not as a replacement for response prediction, and neither the current 4-dimensional solution nor the approximately 0.25-s timescale should be interpreted as the complete or unique biological dimensionality of the system.

Before making dataset-wide biological claims, the same reliability gate should be extended to the other four official sessions, together with leave-neuron-out dimension selection and multiple neuron-subset draws.

## 1. Why Use Brain-Based GPFA

### 1.1 Evaluation Target

The standard Sensorium metric computes the correlation between predicted and measured responses for each neuron and then averages across neurons. This answers whether individual-neuron responses are predicted accurately, but it does not directly answer:

- whether the predicted population state visits the correct location at the correct time;
- whether the trajectory moves in the correct direction;
- whether dynamic changes occur too quickly or too slowly;
- whether the model preserves coordinated activity across neurons rather than only the smoothed marginal statistics of individual neurons.

I therefore use a shared low-dimensional space defined by real neural data, and map the real responses, Dynamic predictions, and Static predictions into the same frozen coordinate system for comparison.

### 1.2 Strict Meaning of “Brain-Based”

This coordinate system satisfies the following constraints:

1. `C`, `d`, `R`, and the temporal priors are fit only from real neural responses in the official train tier.
2. Oracle responses do not participate in GPFA parameter fitting or model selection.
3. Encoding-model predictions do not participate in GPFA fitting, rotation, scaling, Procrustes alignment, or axis selection.
4. Static and Dynamic predictions must use the same neuron IDs, the same neuron order, the same train-only scaling, and the same frozen GPFA.

This makes the trajectory score represent how well model predictions match the real neural population manifold, rather than how well two models match after each is allowed to define its own most favorable latent space.

The methodological basis for GPFA is the unified probabilistic smoothing and dimensionality-reduction framework introduced by Yu et al.: [Yu et al., 2009, Journal of Neurophysiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712272/). That work also proposed evaluating goodness-of-fit by predicting held-out neurons from the remaining neurons, which motivates the leave-neuron-out selection procedure that should be added later.

## 2. Current Experimental Data

### 2.1 Session and Tiers

Primary session:

```text
dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20
neurons: 7,863
```

Complete tier counts:

| Tier | Trials | Use in this stage |
|---|---:|---|
| train | 348 | GPFA fit, calibration, final refit |
| oracle | 58 | Split-half reliability after freezing |
| live_test_main | 56 | Not used |
| live_test_bonus | 58 | Not used |
| final_test_main | 57 | Not used; responses withheld |
| final_test_bonus | 134 | Not used; responses withheld |

### 2.2 Temporal Interval

I used exactly the same final interval as the official Dynamic Sensorium evaluation:

```text
source trial                    300 aligned frames
discarded onset/burn-in         frames 0–49
trajectory interval            frames 50–299
trajectory length              250 frames
official time grid             30 Hz
duration                        250 / 30 = 8.33 s
```

GPFA does not analyze only the 30 retained frames from the 80-frame training snippets; it analyzes the full 250-frame interval used by the complete-trial evaluation. This ensures that the subsequent trajectory metrics and the final encoding-model evaluation cover the same temporal support.

### 2.3 Oracle Condition Recovery

The 58 oracle trials do not directly provide movie-condition labels suitable for the present analysis, so I clustered them using only the grayscale stimulus itself:

- fixed spatial and temporal subsampling of the grayscale video over original frames 0–299;
- mean removal followed by unit-norm normalization;
- graph construction using cosine similarity `>= 0.999`, followed by connected-component extraction;
- explicit exclusion of behavior channels so that repetitions of the same movie are not split because of behavioral differences.

This yields six repeated natural-movie conditions with repeat counts:

```text
10 / 10 / 9 / 10 / 9 / 10
```

The complete dataset indices and trial IDs are published in [`results/tables/02_gpfa_reliability/data_audit.json`](../../results/tables/02_gpfa_reliability/data_audit.json).

### 2.4 Neuron Selection

The primary analysis uses 512 neurons. Selection does not depend on response amplitude, reliability, or trajectory results:

```text
key = SHA256(f"{seed}:{unit_id}")
sort by hash and take the first N neurons
seed = 42
```

This ordering is deterministic, train-independent, and nested. Therefore the 128-, 256-, 512-, and 1024-neuron subsets used for neuron-count saturation are progressively nested.

## 3. Leakage-Free Preprocessing

### 3.1 Train/Calibration Split

The 348 train trials are deterministically shuffled with seed 42 and divided using a 20% calibration fraction:

```text
parameter-fit trials     278
calibration trials        70
```

Latent dimension, timescale initialization, and the behavior-conditioned prior decision use only these 70 calibration trials. After model selection, the primary GPFA is refit on all 348 train trials using the selected settings and then frozen.

### 3.2 Response Scaling

For each selected neuron, I computed one global standard deviation using only the applicable train-fit trials and frames 50–299:

```text
std_n = std(response[fit_trials, frames 50:300, neuron n])
precision_n = 1 / max(std_n, 0.01 * mean_neuron_std)
```

No oracle statistics are used, no separate normalization is applied at each time point, and no scale is estimated from model predictions. The GPFA parameter `d` handles the mean. A single scalar scale per neuron prevents high-variance neurons from dominating the fit.

The final refit recomputes the same train-only precision definition using all 348 train trials and stores it, together with neuron indices, unit IDs, frame interval, and observation grid, in `preprocessing.npz`.

### 3.3 Transformation of Encoding-Model Predictions

Sensorium prediction exports use the official response normalization, whereas GPFA uses train-only scalar precision. Officially normalized predictions therefore cannot be passed directly into GPFA. The implemented pipeline is:

```text
official normalized prediction
    -> invert official response normalization to recover raw response scale
    -> multiply by frozen train-only GPFA precision
    -> select the exact frozen neuron order
    -> frozen GPFA posterior inference
```

Real responses and predictions from both encoding models share this exact pipeline.

## 4. GPFA Mathematical Model and Implementation

### 4.1 Observation Model

For trial `m` and time `t`:

```text
y_m(t) = C x_m(t) + d + ε_m(t)
ε_m(t) ~ Normal(0, R)
R = diagonal neuron-specific noise
```

where:

```text
y_m(t)  N-dimensional scaled neural response
x_m(t)  q-dimensional latent state
C       shared linear observation manifold
d       shared neuron mean
R       shared diagonal observation noise
```

### 4.2 Temporal Prior

Each latent dimension independently uses a squared-exponential Gaussian process:

```text
x_j(t) ~ GP(0, k_j)
k_j(t,t') = exp[-0.5 * ((t-t') / τ_j)^2]
```

Each latent dimension has its own timescale `τ_j`. By default, all natural-video trials share the same temporal prior. The implementation allows different behavior classes to use different `τ`, but `C`, `d`, and `R` must remain shared, so all classes still occupy the same comparable coordinate system.

### 4.3 Exact Inference and EM

The current implementation is exact linear-Gaussian GPFA, not a two-stage approximation that first runs Factor Analysis and then smooths the factors separately:

- `FactorAnalysis` is used only to initialize `C,d,R`;
- the E-step computes the full latent posterior mean and covariance;
- the M-step updates `C,d,R`;
- each latent timescale is updated by bounded scalar optimization;
- Cholesky factorization is used to compute the posterior and marginal likelihood;
- the observation-noise floor and kernel jitter are both `1e-5`.

Primary configuration:

```text
candidate q                         4, 8, 12, 16
initial τ                           0.125, 0.25, 0.5, 1.0 s
τ bounds                            0.10–3.0 s
maximum EM iterations               30
relative tolerance                  1e-4
random seed                         42
```

### 4.4 Why the Primary Observation Grid Is 7.5 Hz

The Sensorium files provide a 30-Hz grid, but the neural signal was originally acquired at approximately 8 Hz and then resampled to 30 Hz. Treating all 30-Hz samples as independent neural observations would repeatedly reuse highly correlated upsampled samples and could lead GPFA to infer a higher temporal bandwidth than the data actually provide.

The primary analysis therefore uses:

```text
observation_indices = 0,4,8,...,248
observations         63
effective rate       7.5 Hz
```

The GP posterior is then continuously queried at all 250 timestamps on the 30-Hz grid to produce a `[trial,250,q]` trajectory. A direct 30-Hz fit is retained as a sensitivity analysis and is not interpreted as providing additional independent neural information.

## 5. Hyperparameter Selection

### 5.1 Held-Out Trial Marginal Likelihood

Each candidate is trained on 278 fit trials and evaluated on 70 calibration trials using marginal negative log likelihood, divided by `time × neuron` to obtain NLL per scalar observation.

Dimension selection uses the one-standard-error rule: first identify the model with the lowest mean NLL, then select the smallest dimensionality whose mean falls within one standard error of that minimum.

| q | Calibration NLL / observation | SE | Selected |
|---:|---:|---:|---|
| 4 | 1.4034596041 | 0.0357813818 | yes by one-SE rule |
| 8 | 1.3970475998 | 0.0359737706 | no |
| 12 | 1.3938458240 | 0.0361219048 | no |
| 16 | 1.3916052517 | 0.0362566103 | raw minimum |

Mean NLL continues to improve through q=16, but all four candidates fall within the one-SE range, so the more conservative q=4 solution is selected.

### 5.2 Timescale Initialization

At q=4, the following initializations were compared:

| Initial τ | Calibration NLL / observation | Selected |
|---:|---:|---|
| 0.125 s | 1.4039462423 | no |
| 0.25 s | 1.4031800295 | yes |
| 0.5 s | 1.4034596041 | no |
| 1.0 s | 1.4043573179 | no |

The final 348-trial refit learns:

```text
τ = 0.25048368, 0.25086529, 0.25045968, 0.25062281 s
parameter digest = 789771cab85943288c022a19a2f571a1704fe0e0af547465e9a06b6ef9051180
```

These timescales remain very close to the initialization. The stricter convergence diagnostic below shows that trajectory metrics are stable even though `τ` continues to drift slowly. I therefore interpret approximately 0.25 s as an effective smoothing regularizer rather than claiming it as a precise biological time constant.

## 6. Data Classification and Conditional-Prior Test

### 6.1 Classification Principle

Neuron identities differ across mice and sessions, so each session must have its own observation manifold. The neuron axes from the five mice cannot simply be concatenated into a single GPFA.

Within a session, `C,d,R` should not be fit separately for each movie, because the resulting latent axes would no longer be directly comparable. Condition dependence may, at most, first be introduced through the temporal prior while keeping the observation coordinates shared.

### 6.2 Behavior-State Classification

I used only train-tier behavior covariates for classification. For each trial, I computed the median and interquartile range of each of the two behavior channels, standardized these features, took PC1 of the resulting behavior-feature space, and divided the fit trials into low/middle/high groups using tertiles. Neural responses did not participate in label construction.

```text
fit class counts          93 / 92 / 93
calibration class counts  22 / 30 / 18
```

I then compared a shared-prior model with a conditional model that uses three separate timescales while sharing `C,d,R`:

```text
shared NLL / observation       1.4031800295
conditional NLL / observation  1.4031800335
paired improvement            -3.92e-9
paired 95% CI                 [-5.65e-7, 5.58e-7]
```

The conditional prior provides no held-out improvement, so the one-SE rule selects the simpler `shared_natural_video` prior. The current data therefore do not support different GP timescales for different behavior strata.

## 7. Reliability Protocol

### 7.1 Balanced Split-Half

For each split and each movie condition:

1. randomly shuffle repeats;
2. take `floor(repeats/2)` trials as the left half;
3. take the same number of trials as the right half;
4. average the real responses within each half;
5. independently pass the two halves through the same frozen GPFA;
6. perform no latent alignment;
7. compute metrics over six conditions × 250 timestamps × q dimensions.

For conditions with 10 repeats, each half uses 5 repeats. For conditions with 9 repeats, each half uses 4 repeats and one repeat is unused in that split. The primary results use 200 splits with seed 42.

### 7.2 Trajectory Metrics

| Metric | Definition and primary information | Direction |
|---|---|---|
| Position correlation | Pearson correlation across all condition/time/latent elements after pooled centering of the two trajectories | Higher is better |
| Position cosine | Mean cosine similarity between position vectors | Higher is better |
| Normalized position RMSE | RMSE divided by the RMS scale of the centered trajectory | Lower is better |
| Velocity-direction cosine | Mean cosine similarity of first-difference vectors at 30 Hz | Higher is better |
| Speed-profile correlation | Pearson correlation between the time series of velocity norms | Higher is better |
| Path-length similarity | `exp(-abs(log(path ratio)))` | Higher is better |
| Acceleration-direction cosine | Mean cosine similarity of second-difference vectors | Higher is better |
| Zero/best-lag correlation | Correlation at zero lag and the best correlation within ±15 frames | Higher is better |

I do not combine these metrics arbitrarily into a single scalar score, because position, direction, speed, and total distance have different sensitivities to different null manipulations.

### 7.3 Matched Nulls

For each split, a null manipulation is applied to the right-half response before GPFA inference while the left half remains unchanged:

| Null | Operation | Information disrupted |
|---|---|---|
| Condition shuffle | Derangement of the six movie conditions | stimulus identity |
| Circular shift | Large nonzero temporal shift within each condition | absolute timing / phase |
| Frame shuffle | Full random permutation of frames within each condition | all local temporal order |
| Time reversal | Reverse the temporal sequence | direction, while preserving the state set and approximately preserving path length |
| Independent-neuron shift | Independently circular-shift each neuron | population synchrony |
| Block shuffle | Randomly permute 4/8/16/32-frame blocks | local/global order at different scales |

Observed and null values are paired within each split. For higher-is-better metrics, the comparison is `observed-null`; for RMSE, the comparison direction is reversed. The finite-sample p-value is:

```text
p = (1 + count[observed not better than null]) / (splits + 1)
```

## 8. Primary Reliability Results

### 8.1 Split-Half Distribution

| Metric | Mean | 2.5% | 97.5% |
|---|---:|---:|---:|
| Position correlation | 0.8565877 | 0.81843 | 0.88853 |
| Normalized position RMSE | 0.5427706 | 0.47893 | 0.65269 |
| Velocity-direction cosine | 0.6627189 | 0.62502 | 0.69489 |
| Speed-profile correlation | 0.7434044 | 0.67914 | 0.79506 |
| Path-length similarity | 0.9449109 | 0.85842 | 0.99781 |
| Acceleration-direction cosine | 0.6163235 | 0.58242 | 0.65476 |

These intervals are the empirical 2.5th–97.5th percentiles of the 200 split values, not confidence intervals for the mean.

### 8.2 Representative Position Nulls

| Null | Position-correlation mean |
|---|---:|
| Condition shuffle | -0.0538 |
| Circular shift | 0.1385 |
| Time reversal | 0.1949 |
| 16-frame block shuffle | 0.2433 |
| Independent-neuron shift | 0.3733 |
| Observed | 0.8566 |

For position correlation, normalized position RMSE, velocity direction, speed profile, and acceleration direction, paired superiority over the representative nulls above is `1.0`: the observed value is better in all 200/200 splits, with paired p=`0.004975`.

### 8.3 What Each Metric Actually Captures

Position correlation/error jointly depend on movie identity and the time-aligned population state. They detect condition swaps, timing shifts, block reordering, and population desynchronization, but by themselves cannot determine whether the direction of local trajectory motion is correct.

Velocity cosine directly measures the direction of movement between adjacent latent states. Time reversal, frame/block shuffling, and timing mismatches strongly reduce it, making it particularly suitable for testing whether an encoding model reproduces local latent dynamics.

Speed-profile correlation ignores direction and tests only whether the real trajectory moves quickly or slowly at the correct times. It complements velocity cosine but is more sensitive to latent dimensionality and the temporal sampling grid.

Acceleration cosine is a higher-order, more noise-sensitive diagnostic of local curvature/direction. It is reliable in the present analysis but should not serve as the sole primary metric.

Independent-neuron shifts strongly reduce all major geometry/dynamics metrics, indicating that these metrics depend on coordinated activity across neurons rather than only on the autocorrelation of each individual neuron.

Path length reaches 0.9449 in the real split-halves, but paired superiority is only `0.63, p=0.373` against circular shift and `0.445, p=0.557` against time reversal. Because these nulls theoretically preserve total distance almost exactly, path length cannot validate temporal alignment and should be used only as a magnitude descriptor.

## 9. Saturation and Sensitivity Validation

### 9.1 Split Count

Dedicated 500-split run:

| Splits | Position mean | Position SE | Velocity mean |
|---:|---:|---:|---:|
| 25 | 0.85424 | 0.00367 | 0.66550 |
| 50 | 0.85413 | — | 0.66280 |
| 100 | 0.85316 | — | 0.66337 |
| 200 | 0.85418 | — | 0.66089 |
| 500 | 0.85590 | 0.00086 | 0.66136 |

Correlation-like estimates are already largely stable by 100–200 splits; 500 splits provide a saturated reference. The main report retains 200 splits because its conclusions and null decisions are consistent with the 500-split run.

### 9.2 Neuron Count

| Neurons | Position | Velocity | Speed | Acceleration |
|---:|---:|---:|---:|---:|
| 128 | 0.8162 | 0.5593 | 0.6536 | 0.4702 |
| 256 | 0.8437 | 0.5920 | 0.7407 | — |
| 512 | 0.8532 | 0.6634 | 0.7403 | — |
| 1024 | 0.8594 | 0.6710 | 0.7696 | — |

Position and velocity change by only 0.0062 and 0.0076, respectively, from 512 to 1024 neurons and are therefore close to a plateau. Speed still changes by approximately 0.029, so its exact value is not yet neuron-saturated. All null-separation decisions remain unchanged.

### 9.3 Latent Dimension

| q | Position | Velocity | Speed |
|---:|---:|---:|---:|
| 4 | 0.8532 | 0.6634 | 0.7403 |
| 8 | 0.8503 | 0.6374 | 0.7464 |
| 12 | 0.8465 | 0.6437 | 0.7851 |
| 16 | 0.8538 | 0.6385 | 0.7935 |

The conclusions for position and velocity are robust to q and show no monotonic improvement, whereas speed changes noticeably with q. Therefore q=4 remains the result-blind primary model for the model comparison, while q=8/12/16 are retained as sensitivity analyses. In particular, speed values should never be reported without specifying q.

### 9.4 Train Fraction

| Train fraction | Position | Velocity |
|---:|---:|---:|
| 25% | 0.8410 | 0.6330 |
| 50% | 0.8551 | 0.6254 |
| 75% | 0.8477 | 0.6585 |
| 100% | 0.8532 | 0.6634 |

No null decision changes. More train trials improve the average stability of velocity, but they are not the sole reason for the strong split-half separation.

### 9.5 GPFA Random Seed

| Seed | Position | Velocity | Speed |
|---:|---:|---:|---:|
| 42 | 0.8532 | 0.6634 | 0.7403 |
| 314 | 0.8552 | 0.6317 | 0.7302 |
| 2718 | 0.8622 | 0.6667 | 0.7557 |

Position is highly stable, while derivative metrics show more seed variability. Static–Dynamic uncertainty should therefore be checked across multiple GPFA seeds.

### 9.6 Observation Grid

| Grid | Position | Velocity | Speed |
|---|---:|---:|---:|
| every 4th frame, 7.5 Hz | 0.8532 | 0.6634 | 0.7403 |
| every frame, 30 Hz | 0.8475 | 0.6190 | 0.7226 |

Position geometry and null separation are stable, whereas derivative metrics are sensitive to the sampling grid. The 7.5-Hz-equivalent grid remains primary, with the direct 30-Hz fit retained as a required sensitivity analysis.

### 9.7 Overall Saturation Assessment

A total of 18 profiles were run across neuron count, q, train fraction, seed, observation grid, and split count. The minimum paired superiority of position, velocity, and speed over the representative nulls remains `1.0`. Thus the conclusion that real repeated trajectories can be distinguished from matched nulls is saturated, although some exact metric values—especially speed and acceleration—cannot yet be considered fully invariant.

## 10. Additional Feasibility Diagnostics

### 10.1 Single-Trial vs. Condition-Average Reliability

The primary split-half analysis averages 4–5 repeats within each half. To determine whether the method is suitable for single trials, I held the frozen GPFA fixed, varied the number of repeats used per half, and ran 200 random disjoint splits:

| Repeats / half | Position | Norm. RMSE | Velocity | Speed | Path | Acceleration |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5545 | 0.9611 | 0.3417 | 0.3694 | 0.8940 | 0.3051 |
| 2 | 0.7117 | 0.7718 | 0.4859 | 0.5434 | 0.9272 | 0.4410 |
| 3 | 0.7864 | 0.6612 | 0.5736 | 0.6400 | 0.9363 | 0.5269 |
| 4 | 0.8328 | 0.5871 | 0.6352 | 0.7077 | 0.9384 | 0.5869 |

The 95% split interval for single-repeat position is approximately `0.4035–0.6613`. Therefore:

- condition-averaged natural-movie trajectories have high reliability;
- single-trial trajectories have only moderate reliability;
- the primary result of `0.8566` must not be described as single-trial reliability;
- the model comparison uses condition averages as the primary analysis and single trials as a secondary analysis.

### 10.2 Stricter EM Convergence

The relative tolerance of the primary model is relatively loose, allowing optimization to stop after approximately two EM iterations. A supplementary q=4 fit uses a maximum of 20 iterations with tolerance `1e-8`; total likelihood still shows a small improvement at iteration 20.

```text
base τ    0.25048 / 0.25087 / 0.25046 / 0.25062 s
strict τ  0.25403 / 0.25792 / 0.25438 / 0.25593 s
NLL/observation improvement  2.489e-5
C-subspace principal angles  6.21°, 3.99°, 0.90°, 0.37°
```

On the same 50 splits:

| Metric | Base | Strict |
|---|---:|---:|
| Position | 0.85641 | 0.85748 |
| Normalized RMSE | 0.54262 | 0.54084 |
| Velocity | 0.66363 | 0.66526 |
| Speed | 0.74265 | 0.74606 |
| Acceleration | 0.61327 | 0.61691 |

All changes are smaller than approximately 0.004, indicating that the trajectory conclusions are robust to further optimization. The exact timescale values are less strongly identified than the trajectory metrics themselves.

### 10.3 Response-Variance Coverage of the Low-Dimensional Subspace

Oracle responses are reconstructed from the GPFA posterior through `C x + d`:

| q | Population R² | Mean per-neuron R² | Median per-neuron R² | 90th percentile |
|---:|---:|---:|---:|---:|
| 4 | 0.0200 | 0.00369 | 0.01296 | 0.05275 |
| 8 | 0.04231 | 0.02214 | 0.03099 | 0.08899 |
| 16 | 0.07148 | 0.04123 | 0.04916 | 0.12258 |

At q=4, approximately 69.5% of neurons have positive per-neuron R², and the population R² for trial-averaged responses is approximately 0.02824. The q=8 and q=16 models are short-EM diagnostic fits rather than the currently frozen primary models.

This is one of the most important boundaries of the method: GPFA captures a stimulus-locked shared subspace that explains only a small fraction of total variance but is highly repeatable across repeats. It does not capture most of the neural response. Therefore:

- high trajectory reliability does not imply high response reconstruction;
- q=4 is likely a conservative, and potentially overcompressed, summary;
- response correlation must remain the primary prediction-performance gate;
- the trajectory metric can answer only whether the model reproduces this shared dynamic subspace.

## 11. Frozen Rules Used for Static–Dynamic Evaluation

### 11.1 Primary Analysis

```text
session                       current pilot: dynamic29515-10-12
condition                     six repeated natural movies
time interval                 original frames 50–299
primary response unit         condition average
neurons                       frozen 512-unit order
GPFA                          frozen q=4, seed42, 7.5-Hz observations
metrics                       position correlation/error, velocity cosine, speed correlation
diagnostic                    acceleration cosine
descriptive only              path-length similarity
```

### 11.2 Prespecified Sensitivity Analyses

The following were specified for reporting alongside the primary analysis:

- q=8, 12, and 16;
- GPFA seeds 42, 314, and 2718;
- 7.5-Hz-equivalent and direct 30-Hz grids;
- condition-average primary analysis and single-trial secondary analysis;
- matched nulls and paired finite-sample p-values;
- manifold reconstruction/residual diagnostics;
- the original Sensorium response correlation.

### 11.3 Prohibited Operations

- Do not refit GPFA separately to Static or Dynamic predictions.
- Do not separately rotate the two model trajectories.
- Do not recompute neuron scaling using oracle responses or model predictions.
- Do not select neurons, q, timescale, or movie conditions to favor either model.
- Do not use path length as primary evidence of temporal alignment.
- Do not claim q=4 or `τ≈0.25 s` as the unique dimensionality or time constant of the biological system.

## 12. Passing Criteria, Current Status, and Remaining Extensions

### 12.1 Criteria Currently Passed

```text
Temporal alignment to official 250-frame evaluation       PASS
Train/oracle leakage controls                              PASS
Frozen shared neural coordinate system                     PASS
Condition-average split-half reliability                   PASS
Matched-null separation for primary metrics                PASS
Neuron/q/train/seed/grid/split saturation of decisions     PASS
Behavior-conditioned prior justification                   PASS: shared prior retained
Single-session method-development gate                      PASS
```

### 12.2 Criteria Not Yet Passed or Not Appropriate to Overinterpret

```text
All-five-session reliability                               NOT YET RUN
Dataset-wide biological conclusion                         NOT YET SUPPORTED
High-fidelity reconstruction of total neural variance      NO
Single-trial reliability at condition-average level        NO
Path length as temporal metric                              FAIL
Unique biological latent dimension/timescale               NOT IDENTIFIED
```

### 12.3 Priority Extensions Beyond the Current Proof of Concept

1. Repeat the neural split-half reliability gate in the other four official sessions and summarize the effect using session as the statistical unit.
2. Add leave-neuron-out dimension selection: predict held-out neurons from the remaining neurons so that the current held-out-trial NLL is not dominated by large amounts of independent observation noise. The current one-SE rule selects q=4, but the response-coverage analysis suggests that q=4 may be overly conservative.

In addition, multiple deterministic neuron-subset seeds should be used within each session to verify that model ranking does not depend on a single 512-neuron draw.

## 13. Reproducibility Files and Validation Artifacts

```text
Result-blind locked protocol
docs/methods/GPFA_PROTOCOL_LOCKED.md

Primary configuration
experiments/02_gpfa_reliability/configs/pilot.yaml

Implementation
experiments/02_gpfa_reliability/src/trajectory_reliability/
  data.py
  gpfa.py
  selection.py
  conditions.py
  metrics.py
  reliability.py
  saturation.py

Frozen model and preprocessing
models/gpfa_reliability/gpfa.pkl
models/gpfa_reliability/preprocessing.npz

Model selection
results/tables/02_gpfa_reliability/model_selection.csv

Primary reliability raw results
results/tables/02_gpfa_reliability/split_half_observed.csv
results/tables/02_gpfa_reliability/null_distributions.csv
results/tables/02_gpfa_reliability/reliability_summary.csv

Condition-prior test
results/tables/02_gpfa_reliability/behavior_conditioned_prior.json

Saturation
results/tables/02_gpfa_reliability/saturation/saturation_metrics.csv
results/tables/02_gpfa_reliability/saturation/split_count_convergence.csv
results/tables/02_gpfa_reliability/saturation/split_count_observed_500.csv
results/tables/02_gpfa_reliability/saturation/split_count_nulls_500.csv

Automated tests
experiments/02_gpfa_reliability/tests/
```

All 7 automated tests and Python compile checks in the current implementation pass. They cover condition grouping, train/calibration separation, the GPFA fitting/inference contract, null generation, metric direction, and selection behavior.

## 14. Final Conclusion

I consider the practical feasibility of the GPFA trajectory method to be **conditionally supported**:

- for condition-averaged population responses to repeated natural movies, it produces low-dimensional trajectories with high reliability and strong null separation;
- it captures stimulus identity, absolute timing, local direction, speed modulation, and cross-neuron synchrony;
- reliability is substantially lower at the single-trial level;
- it captures only a small fraction of total neural variance;
- derivative metrics are more sensitive than position metrics to the sampling grid, q, and fit seed;
- path length contains insufficient information about temporal direction.

The most appropriate scientific use is therefore to first establish the basic predictive quality of both encoding models using the original response correlation, and then use the frozen brain-based GPFA to separately report position, velocity, and speed matching, together with matched nulls and cross-session inference. In this role, trajectory evaluation can provide dynamic information that is not directly available from response-level metrics without treating a low-dimensional repeatable subspace as if it were the complete neural response.
