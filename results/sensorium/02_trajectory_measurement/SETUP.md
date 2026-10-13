# Shared Trajectory Measurement: Setup

## 1. Research question

**Can one frozen Gaussian-process factor analysis (GPFA) model provide a useful common measurement for the current neural-trajectory comparison?** This goal examines how repeatable the inferred trajectories are, whether their agreement depends on the correct movie and temporal organisation, and how much response variance the observation model reconstructs.

Repeatability, null sensitivity and reconstructed response variance are distinct properties. These checks establish where this measurement is useful; optimality across representations remains untested.

| Check | Question addressed | What remains fixed |
|---|---|---|
| Disjoint repeat-half agreement | Do separate presentations of the same movies yield corresponding trajectories? | Neural-data-defined GPFA, preprocessing and metric definitions |
| Matched nulls | Does agreement depend on movie identity, timing or population coordination? | Left-half reference and posterior-inference procedure |
| One-, two- and balanced-repeat comparisons | How much does reliability depend on averaging? | GPFA and the underlying repeat-assignment plan |
| Frozen observation-model reconstruction | How much response variance is reconstructed from the inferred states? | Loadings, offsets, observation noise and temporal priors |

Training neural responses fit the four-dimensional GPFA shared by neural and model arrays. Here the checks concern neural responses; [Goal 3](../03_model_comparison/RESULTS.md) compares encoders. [Results](RESULTS.md) reports the measurement findings.

Auxiliary GPFA sensitivity was withdrawn from the current conclusions. No auxiliary fit or cross-profile robustness claim forms part of this goal.

![Shared current measurement and data roles](../figures/documentation/01_shared_pathway.png)

*Shared current measurement and data roles. This goal checks neural-response repeatability and reconstruction in the pilot; it does not compare encoder accuracy. The same fixed scales and offline GPFA later measure neural and predicted responses.*

## 2. Data and evaluation support

The measurement uses one pilot session:

```text
dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20
```

| Data role | Trials | Use in this measurement |
|---|---:|---|
| Train | 282 | GPFA scaling, parameter fitting and final refit |
| Validation | 66 | Selection of the initial timescale by neural likelihood |
| Test | 58 | Frozen repeat reliability, nulls and response reconstruction |

The test data contain six repeated movie conditions. Their repeat counts, sorted by count, are `9, 9, 10, 10, 10, 10`; this ordering does not identify individual movies. Condition identities and trial membership are retained in the [split manifest](../evidence/04_model_comparison/retrained_v1/split_manifest.json).

Of the session's 7,863 recorded neurons, 512 are selected by sorting stable unit IDs by the SHA-256 digest of `42:<unit_id>` and retaining the first 512. Neither neural reliability nor model accuracy determines this selection. No test condition is excluded on the basis of its reliability.

| Array | Axes and shape |
|---|---|
| Released neural response | Neuron × time |
| Aligned response or prediction | Trial × time × neuron: `[trial, 250, 512]` |
| Test condition means | Condition × time × neuron: `[6, 250, 512]` |
| Condition-mean latent trajectories | Condition × time × latent: `[6, 250, 4]` |

The 250 timestamps correspond to original frames 50–299. All compared arrays must have the same finite trial, time and neuron support. The responses are processed calcium signals on the released 30 Hz grid, not independent 30 Hz neural acquisitions.

The upstream tier named **`oracle`** supplies labelled repeated-movie responses for the current test. Hidden tiers containing zero-response placeholders are excluded. These test movies were explored in earlier work; the present fit uses separate train and validation roles, but the evaluation is not untouched external validation. Upstream source-film identity remains unresolved.

## 3. GPFA model and fixed identity

### Observation model and temporal prior

For neuron $n$ at time $t$, the observation model is

$$
y_n(t)=C_{n,:}z(t)+d_n+\epsilon_n(t).
$$

Here, $y(t)$ is the response in the GPFA input scale, $z(t)$ is a four-dimensional state, $C$ maps that state to the 512 neurons, and $d$ is the observation offset. Observation-noise covariance $R$ is diagonal.

Each latent dimension has an independent, unit-amplitude squared-exponential temporal prior:

$$
k_q(t,t')=\exp\left[-\frac{1}{2}\left(\frac{t-t'}{\tau_q}\right)^2\right],
\qquad q=1,\ldots,4.
$$

**This Sensorium model learns one timescale per latent dimension.** The 0.25-second value selected during calibration is an initialisation, not a constraint that all final timescales must equal 0.25 seconds. The fitted values recorded in `frozen_gpfa_arrays.npz` are:

| Latent dimension | Learned timescale (seconds) |
|---|---:|
| 1 | 0.2540135777751983 |
| 2 | 0.25688538456865606 |
| 3 | 0.2543113927285101 |
| 4 | 0.25858327387654434 |

| Stored quantity | Shape or count |
|---|---|
| Loadings `C` | `[512, 4]`: 2,048 entries |
| Offsets `d` | `[512]`: 512 entries |
| Diagonal observation noise `R` | `[512]`: 512 entries |
| Timescales | Four entries |
| GPFA scaling factors | `[512]` |
| Encoder response scaling factors over the evaluation interval | `[250, 512]` |
| Observation indices | `[63]` |
| Query times | `[250]` |

The loadings, offsets, diagonal noise and timescales total 3,076 learned scalar entries. Preprocessing arrays are recorded separately; numerical jitter is fixed.

### Fitting and initialisation selection

The latent dimension is fixed at four before calibration. Factor analysis initialises the observation parameters. Candidate initial timescales are compared using validation neural likelihood, followed by a final fit on the training trials only.

| Setting | Value |
|---|---|
| Candidate initial timescales | 0.125, 0.25, 0.5 and 1.0 seconds |
| Candidate fitting budget | Up to 12 expectation–maximisation (EM) iterations |
| Selection criterion | Validation neural likelihood |
| Selected initial timescale | 0.25 seconds |
| Final fitting data | 282 training trials; no train-plus-validation refit |
| Final fitting budget | Up to 20 EM iterations |
| Relative convergence tolerance | `1e-6` |
| Timescale bounds | 0.1–3 seconds |
| Numerical jitter | `1e-5` |
| Minimum observation noise | `1e-5` |

Encoder predictions and test response scores do not enter this selection. The [selection table](../evidence/04_model_comparison/retrained_v1/gpfa_selection.csv) and [fit history](../evidence/04_model_comparison/retrained_v1/gpfa_fit_history.csv) identify the candidate and final fits. The additional `stage=dimension` row is a four-dimensional bookkeeping fit at a 0.5-second initialisation; it is not evidence of a new dimension search.

### Artifact identity

The [frozen GPFA identity](../evidence/04_model_comparison/retrained_v1/frozen_gpfa_identity.json) connects the fitted object to its split, preprocessing and selection records. Its recorded identifiers are:

```text
GPFA pickle SHA-256
98de0cef99973c8a57b5b44d330763a5cd0fc5ad0cab36654c058591ae6eb52c

Array-export SHA-256
e1d5f87d29eab8f9406fc2a2ee1b9c72a509707c6583df839a888199e47a09f5

Parameter digest
e9b3e792da5d0cb8d64573d6ddc13de2607db9e775527363b09e9f2170d8a562
```

The fitted parameters and scale arrays are external or Git-ignored payloads. Access to the fitted payloads is separate from access to their public identity records. The [model manifest](../../../models/manifest.json) and [artifact guide](../../../models/README.md) describe their locations and access status.

## 4. Shared scaling and posterior inference

### Converting responses to the GPFA scale

Encoder response scaling and GPFA scaling are different fixed transforms. The encoder scale depends on time and neuron. GPFA uses one scale per neuron, estimated across raw training trials and original frames 50–299.

Let $s_n$ be the training response standard deviation for neuron $n$, computed with `ddof=1`. The lower bound is

$$
f=0.01\,\operatorname{mean}_n(s_n)=0.17263539191444882.
$$

The GPFA multiplicative scaling factor is $1/\max(s_n,f)$. No additional centring or clipping is applied. The stored variables named `precision` denote these multiplicative scaling factors.

For recorded responses and both encoder predictions, the conversion is identical:

```text
GPFA_input[t, n] = encoder_scaled[t, n]
                   / encoder_train_precision[t, n]
                   * GPFA_train_precision[n]
```

Dividing by the encoder factor recovers the raw response scale; multiplying by the GPFA factor applies the fixed neural-training scale. Both factors are fixed from training data. Every input uses the same scales and latent axes, with model-specific fitting, Procrustes alignment and rotation excluded from the protocol.

### Observation and query grids

The diagram distinguishes 63 observations (`0, 4, …, 248`) from 250 posterior queries (`0, 1, …, 249`), all relative to the scoring window. Query times use `index / 30` seconds.

**The frozen posterior conditions on the full window**, so later observations can affect an earlier state. The resulting inference is offline.

Query points provide inferred states on the full evaluation grid without adding independent observations. Their derivatives describe posterior changes; exact neural latency remains unresolved.

## 5. Repeat reliability and matched nulls

### Disjoint repeat assignments

Each of 200 seed-42 partitions permutes repeats within movie, then assigns disjoint halves of `floor(repeats / 2)`: four per side for nine-repeat movies, five for ten-repeat movies. Each balanced side has 28 trials; two are unused.

Responses are averaged within each side and condition before frozen inference. Corresponding left and right trajectories are then compared without fitting or aligning a new representation.

One- and two-repeat diagnostics use nested prefixes of the same unsorted permutations, sorted only for averaging. They replay the saved membership/null random stream and reuse balanced scores; GPFA is unchanged.

The [sampling implementation](../../../code/sensorium/src/trajectory_model_eval/supplementary_gpfa.py) and [saved main memberships](../evidence/04_model_comparison/retrained_v1/pilot/reliability_split_identities.csv) record these assignments.

### Where the nulls act

Each null transforms the right-half condition means **after conversion to the GPFA scale and before the same frozen posterior**. The left-half reference remains unchanged. These are response-level perturbations, not changes to the input videos.

| Null | Operation on the right-half response array | Preserved structure and changed correspondence |
|---|---|---|
| Condition shuffle | Fully derange the six condition rows; try up to 100 random proposals, then use a cyclic fallback | Preserves each complete response trajectory; disrupts movie matching |
| Circular shift | Apply one joint population shift per condition, with integer offset in `[31, 219)` | Preserves the neurons' relative timing and marginal sequence values; changes absolute phase |
| Time reversal | Apply `values[:, ::-1]` | Preserves sequence values and total response-space path length; reverses order and direction |
| Independent-neuron shift | Give each neuron in each condition an independent integer shift in `[1, 250)` | Preserves each neuron's circular sequence; changes coordinated population timing |
| Block shuffle, lengths 4, 8, 16 or 32 | Divide the 250-frame interval into blocks, retaining the short final block; apply one block permutation across conditions; replace the identity permutation with a cyclic roll | Preserves order within blocks; changes order between blocks |

The four block lengths make this **eight null configurations** in total. The optional legacy full-frame-shuffle null is not part of the current reliability run. Integer ranges above include the lower bound and exclude the upper bound.

Reversal is deterministic for a particular right-half mean. Across 200 repeat partitions, the same deterministic operation acts on 200 different repeat means. Posterior inference can change path length even when the response-space operation preserves it.

### Partition summaries

Observed and null derivative scores use the same effective support. A partition is superior to its matched null when the observed score is higher for correlation or cosine metrics, or lower for normalised RMSE. Ties and non-superior comparisons count as failures; undefined scores are counted separately.

Means, standard deviations and 2.5th/97.5th percentile summaries describe variation across repeat assignments within the same six movies. They are not animal-population confidence intervals or p-values. The resulting reliability scores are not themselves reliability-normalised, and repeat-average agreement is not a universal noise ceiling.

## 6. Metrics and response reconstruction

### Trajectory agreement

Trajectory metrics take arrays with axes `[condition, time, latent]`. Differences are computed within each condition, not across movie boundaries.

| Metric | Role and calculation | Preferred direction |
|---|---|---|
| Position correlation | Primary. Subtract one per-axis centre pooled across the compared pair, flatten, then compute Pearson correlation | Higher |
| Velocity-direction cosine | Primary. Average cosines between first-difference vectors on shared nonzero support | Higher |
| Normalised position RMSE | Secondary. Divide pooled error RMS by the neural-reference RMS about the same pair-pooled per-axis centre | Lower |
| Speed-profile correlation | Secondary. Correlate the norms of the first-difference vectors | Higher |
| Acceleration-direction cosine | Secondary. Average cosines between second-difference vectors on shared nonzero support | Higher |

First and second differences use `dt = 1/30` second. The implementation uses `EPS = 1e-12` and common nonzero support across the compared arrays, so a perturbation cannot silently change which derivative samples enter the comparison. See [`trajectory_metrics`](../../../code/sensorium/src/trajectory_reliability/metrics.py) for the recorded implementation.

Path-length similarity is outside the current five-metric battery. The historical descriptive formula is:

```text
exp(-abs(log(mean_c(length_right_c / max(length_left_c, 1e-12)))))
```

Total distance can survive timing or direction changes. Its descriptive role therefore remains separate from validation of temporal fidelity by the current five-metric battery.

### Frozen observation-model reconstruction

Reconstruction uses the same fixed posterior and observation model:

$$
\widehat y(t)=Cz(t)+d.
$$

In array form, this is `y_hat = z @ C.T + d`. It is applied to individual test trials and, separately, to the six condition means. Both the full 250-point query grid and the 63-point observation grid are scored. No decoder, scale or GPFA parameter is refitted for this check.

Let $u$ index the evaluated trial-or-condition/time samples for one response level and one grid. With a separate evaluation mean $\bar y_n$ for each neuron,

$$
\mathrm{SSE}_n=\sum_u(y_{u,n}-\widehat y_{u,n})^2,
\qquad
\mathrm{SST}_n=\sum_u(y_{u,n}-\bar y_n)^2.
$$

The reported centred population score is

$$
R^2_{\mathrm{population}}
=1-\frac{\sum_n\mathrm{SSE}_n}{\sum_n\mathrm{SST}_n}.
$$

A second score uses the frozen training offset as its reference:

$$
R^2_{\mathrm{train\text{-}intercept}}
=1-\frac{\sum_{u,n}(y_{u,n}-\widehat y_{u,n})^2}
{\sum_{u,n}(y_{u,n}-d_n)^2}.
$$

Per-neuron centred scores are $R_n^2=1-\mathrm{SSE}_n/\mathrm{SST}_n$. The population score pools sums of squares, giving a different weighting from the unweighted mean of per-neuron scores. Consequently, positive pooled reconstruction and a negative mean per-neuron score can coexist.

Response levels and grids have separate denominators. Evaluation means define scoring baselines without refitting GPFA. Denominators ≤ `1e-12` stay undefined; negative R² is retained. RMSE uses GPFA training-standard-deviation units.

This is reconstruction of responses used for frozen posterior inference. It is separate from the encoders' stimulus-to-response prediction and from agreement between independent repeat halves.

## 7. Evidence, attribution and scope

The main configuration is [retrained_v1.json](../../../code/sensorium/configs/retrained_v1.json), particularly its `gpfa`, `reliability` and `reporting` entries. [Supplementary GPFA settings](../../../code/sensorium/configs/supplementary_gpfa_v1.json) supply the retained `main`, `reliability` and `reconstruction` diagnostics only; auxiliary sensitivity settings remain outside this goal.

| Component | Source and relevant functions |
|---|---|
| GPFA fitting and posterior | [GPFA implementation](../../../code/sensorium/src/trajectory_reliability/gpfa.py): `GaussianProcessFactorAnalysis.fit`, `_posterior`, `transform_query`, `_update_lengthscales` |
| Current fitting wrapper | [Retrained analysis](../../../code/sensorium/src/trajectory_model_eval/retrained_analysis.py): `fit_gpfa`, `load_gpfa` |
| Shared inference and scoring | [Unified analysis](../../../code/sensorium/src/trajectory_model_eval/unified_analysis.py): `FrozenAnalysis.gpfa_input`, `.latent`, `_reliability`, `_trajectory_scores` |
| Repeat partitions and nulls | [Reliability operators](../../../code/sensorium/src/trajectory_reliability/reliability.py): `_balanced_half`, `_derangement`, `_block_shuffle`, `_null_response` |
| Repeat and reconstruction diagnostics | [Supplementary implementation](../../../code/sensorium/src/trajectory_model_eval/supplementary_gpfa.py): `sampling_plan`, `verify_plan`, `reconstruction_scores`, `_reconstruction`, `_summarize_reliability` |

GPFA is attributed to Yu et al. (2009), *Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity*, **Journal of Neurophysiology**, 102, 614–635 ([DOI](https://doi.org/10.1152/jn.90941.2008)). The source method provides context; this project's numerical choices are defined by its recorded configuration and implementation.

Earlier 348- and 174-trial GPFA fits are separate historical objects. An executable definition sufficient to verify exact equivalence with their reconstruction denominator is not available in the checked sources. Their reconstruction values are therefore not substituted for, or treated as numerically interchangeable with, the current 282-trial fit. The present results concern one session and six repeated movies, with offline inference and no retained claim of robustness across latent dimensions or observation grids.
