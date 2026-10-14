# Temporal-Weight and Spatial-Damage Controls: Setup

## 1. Research question

**Which response and trajectory scores change when the reduced Dynamic model's off-centre temporal weights are attenuated? When a fixed spatial-weight intervention produces a similar response score, do the trajectory changes agree?**

The [main model comparison](../03_model_comparison/RESULTS.md) evaluates frozen Static and reduced Dynamic models. This diagnostic probes copies of the current reduced Dynamic model through specified weight changes. A five-level temporal intervention tests dependence on the selected temporal weights. A separate spatial-weight intervention tests whether similar losses in one response summary accompany similar trajectory changes.

The first comparison is **temporal attenuation versus unchanged Dynamic**. The second is **spatial damage versus temporal attenuation at a similar mean per-neuron response correlation**. The matched comparison uses numerically evaluable pairs whose test gap remains within the fixed tolerance. A decline after temporal attenuation is compatible with both a temporal-specific effect and general disruption of the trained network.

This is a completed exploratory diagnostic. Its settings were fixed after the main outcomes were known and before its own scoring. It supplements the primary comparison by measuring sensitivity to the selected weights. The [output-noise response-matching control](../05_response_matching/SETUP.md) is a separate objective: its perturbation and selected amplitude remain unchanged. The analysis uses the retained main GPFA; auxiliary sensitivity fits were withdrawn.

## 2. Data roles and evaluation support

The pilot is `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20`. Its current split has 282 training trials, 66 validation trials and 58 repeated test trials from six movies. Test repeat counts, sorted by count, are 9, 9, 10, 10, 10 and 10. Seed-42 hashing of stable neuron IDs fixed the 512-neuron subset independently of response accuracy and reliability.

| Data role | Trials | Role in this diagnostic |
|---|---:|---|
| Train | 282 | Supplied the pilot’s preprocessing and neural GPFA fit, both held fixed for the interventions. |
| Validation | 66 | Score a fixed spatial-retention grid and select one spatial level for each temporal target. |
| Test | 58 | Score the fixed temporal curve, check transfer of spatial matching, and compare the valid pairs. |

Validation matching pools trial and time samples within each neuron, allowing validation movies without repeats. Test repeat means are used for representation and trajectory scores. The input identities, normalisation, neuron mapping and time mask stay unchanged between intervention strengths.

| Array or stage | Axes | Shape or support |
|---|---|---|
| Validation targets and predictions | Trial × time × neuron | `[66, 250, 512]` |
| Test targets and predictions | Trial × time × neuron | `[58, 250, 512]` |
| Test repeat means | Movie × time × neuron | `[6, 250, 512]` |
| GPFA posterior trajectories | Movie × time × latent dimension | `[6, 250, 4]` |

Inputs are released preprocessed mouse visual-cortex responses with aligned video, behavioural and pupil fields. Full encoder inference consumes 300 frames. The 18-frame temporal reduction is aligned to original timestamps, and frames 50–299 are retained for scoring on the 30 Hz array grid. Earlier project notes describe approximately 8 Hz calcium acquisition before resampling; that rate is documented in narrative sources rather than independently checked acquisition metadata here. The array grid contains resampled responses, and the posterior-query grid contains inferred states.

Existing bundles contain `neural`, `static`, `dynamic`, `conditions`, `dataset_indices`, `neuron_ids`, `frame_indices` and `encoder_precision`. The diagnostic uses the available trial arrays directly, preserving their original trial support. Hidden challenge-tier zero-response placeholders are excluded. The upstream labelled repeated-movie tier, called `oracle`, is test data for the retrained run, although these movies were explored historically. The wider data dependencies are in [Models and Data](../MODELS_AND_DATA.md#3-training-validation-and-test-data).

## 3. Fixed model, measurement and intervention copies

Here, **Dynamic** means the seed-42 reduced Dynamic encoder from `retrained_v1`, selected at epoch 109, with 2,862,063 trainable parameters. It has three factorised spatiotemporal stages of widths 16/32/64. The historical full Dynamic model is a different artifact. The current Static model, selected at epoch 59 with 2,814,015 parameters, is the reference architecture from the main comparison; the interventions here modify Dynamic rather than reconstructing Static.

Each intervention starts from a copy of the reduced Dynamic weights. Biases, BatchNorm buffers and every unspecified parameter remain fixed during evaluation-mode inference. Each intervention changes only the temporal or spatial-convolution weight tensor specified below; readout parameters remain fixed. The original checkpoint remains unchanged; each intervention operates on a copy of the selected encoder. [Main comparison setup](../03_model_comparison/SETUP.md#3-models-and-fitting) documents the original architecture and training choices.

The shared four-dimensional Gaussian-process factor analysis (GPFA) uses a linear observation model, diagonal neuron noise and one learned radial basis function (RBF) timescale per latent dimension. The 282 training neural trials determine its fitted parameters. Validation neural likelihood selected the 0.25-second initial timescale; the final fit used 19 expectation-maximisation (EM) iterations on training data alone.

The posterior observes 63 indices, `0, 4, …, 248`, within the scoring window and is queried at all 250 timestamps. Full-window inference is offline: later observations can affect an earlier state estimate. This same main GPFA, its input scales and its neuron mapping remain fixed for every intervention. Its reliability and response coverage are described in [the measurement results](../02_trajectory_measurement/RESULTS.md).

<details>
<summary>Frozen artifact identities</summary>

| Artifact | SHA-256 |
|---|---|
| Current Static checkpoint — main-comparison context only | `3ded37a0fac494f47a8553a432e04e8be7e95a0a99ed9444b07ae61f31ff7052` |
| Reduced Dynamic checkpoint — source of intervention copies | `c6eff392ec412494ef69d6e599193711f4b5428862fe57633899280144531954` |
| Main GPFA pickle | `98de0cef99973c8a57b5b44d330763a5cd0fc5ad0cab36654c058591ae6eb52c` |

These identities refer to model payloads, which store the weights separately from the identity JSON files. The [model access guide](../../../models/README.md) describes artifact availability.

</details>

## 4. Temporal-weight attenuation

### The retained fraction applies only to off-centre slices

For each temporal-convolution tensor $W^{(\ell)}$, let $k_\ell$ be its temporal length and $j_\ell=(k_\ell-1)/2$ its centre index. At retention $\rho$, the operation is

$$
\widetilde W^{(\ell,\rho)}_{o,i,j,0,0}
=
\begin{cases}
W^{(\ell)}_{o,i,j,0,0}, & j=j_\ell,\\
\rho W^{(\ell)}_{o,i,j,0,0}, & j\ne j_\ell.
\end{cases}
$$

The five fixed temporal retentions are $\rho\in\{1,0.75,0.50,0.25,0\}$. Lower retention means stronger attenuation. The implementation locates parameter names containing `conv_temporal` and ending in `weight`.

| Temporal weight tensor | Shape: output × input × time × height × width | Eligible off-centre entries |
| --- | --- | --- |
| `core.features.layer0.conv_temporal.weight` | [16,16,11,1,1] | 2,560 |
| `core.features.layer1.conv_temporal_1.weight` | [32,32,5,1,1] | 4,096 |
| `core.features.layer2.conv_temporal_2.weight` | [64,64,5,1,1] | 16,384 |

The three tensors contain **23,040 eligible off-centre entries**. All 23,040 entries are eligible for scaling and remain unchanged at retention one. Exact shapes, changed-entry counts and norms are recorded in [the weight-change record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/weight_changes.json).

At $\rho=1$, the weights retain their original values. Agreement with the existing Dynamic prediction cache must satisfy `rtol=atol=3e-6`, identifying the unmodified baseline.

<a id="zero-retention-is-not-the-static-model"></a>
### Input history retained at zero temporal retention

The unmodified temporal receptive field spans 19 input frames, with offsets $-18$ through $0$ relative to the aligned output time $t$. At $\rho=0$, only the centre slice of each temporal kernel remains. The 11/5/5 kernels then retain visual and appended-behaviour input at **$t-9$**, while the pupil shifter still uses the current time **$t$**.

Spatial processing, central temporal slices, readout and shifting remain. **Zero temporal retention produces a centre-only Dynamic model** with delayed visual input, distinct from the separately trained current-frame Static model. The [time-support record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/temporal_support.json) retains offsets and `is_static_model=false`.

## 5. Spatial damage and validation matching

### One fixed spatial intervention

Spatial retention $\eta$ multiplies the entire first spatial-convolution weight tensor:

$$
\widetilde W^{\mathrm{spatial},\eta}=\eta W^{\mathrm{spatial}}.
$$

The only target is `core.features.layer0.conv_spatial.weight`, with shape `[16, 3, 1, 11, 11]`, or **5,808 weights**. The three input channels are video and two spatially appended behavioural traces. The operation jointly attenuates weights acting on visual and behavioural input across all three channels.

The spatial grid contains 21 fixed values: $1,0.95,0.90,\ldots,0.05,0$. The intervention scales the full specified tensor deterministically. Temporal kernels, later spatial layers, biases, readout, pupil shifter and BatchNorm state remain unchanged. The comparison characterizes this spatial-weight intervention; alternative non-temporal interventions would test the breadth of the result.

### Match the validation response summary, then keep the choice fixed

Let $P^{T,\rho}$ be the temporally attenuated prediction and $P^{S,\eta}$ the spatially attenuated prediction. Superscripts $T$ and $S$ identify **temporal and spatial interventions** within Dynamic; the independently trained Static model is a separate reference.

For neural responses $Y$ and predictions $P$, the matching score is the mean per-neuron response correlation:

$$
r_{\mathrm{resp}}(Y,P)
=
\frac{1}{|\mathcal N|}\sum_{n\in\mathcal N}
\operatorname{corr}\left(
\operatorname{vec}_{i,u}Y_{i,u,n},
\operatorname{vec}_{i,u}P_{i,u,n}
\right).
$$

Trial $i$ and scoring time $u$ are flattened within each neuron. Pair comparisons use shared finite-neuron support. For each $\rho$, the original calibration selects the spatial grid value with the smallest absolute validation gap,

$$
\eta^*(\rho)
=
\underset{\eta\in\{1,0.95,\ldots,0\}}{\operatorname{argmin}}
\left|
 r_{\mathrm{resp}}(Y_{\mathrm{val}},P^{S,\eta}_{\mathrm{val}})
-r_{\mathrm{resp}}(Y_{\mathrm{val}},P^{T,\rho}_{\mathrm{val}})
\right|.
$$

Ties choose the highest spatial retention. The response-matching tolerance is **0.005**. Grid selection minimises the gap; a separate tolerance check determines whether the selected pair qualifies as matched.

All selected spatial levels are frozen before test scoring. On test, record the signed gap

$$
g_{\mathrm{test}}(\rho)
=
 r_{\mathrm{resp}}(Y_{\mathrm{test}},P^{S,\eta^*(\rho)}_{\mathrm{test}})
-r_{\mathrm{resp}}(Y_{\mathrm{test}},P^{T,\rho}_{\mathrm{test}}).
$$

A numerically evaluable pair retains matching when $|g_{\mathrm{test}}|\le0.005$. This criterion constrains one pooled scalar summary. Neuron-wise accuracy and other response statistics may still differ; statistical equivalence is not tested. Pairs outside tolerance stay in the result record with failed-match status.

The [validation grid](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/validation_grid.csv) and [matching records](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/spatial_response_matches.csv) retain the original choices. Current display columns incorporate the numerical endpoint correction below. The correction retains the original spatial strengths and the separate output-noise calibration.

### The zero-spatial endpoint is unevaluable for correlation matching

At $\eta=0$, the first spatial tensor has zero norm. Saved output diagnostics identify 76 exactly constant neurons and 436 neurons varying only at the float32 output floor. The endpoint is therefore marked `numerically_degenerate` for scientific response correlation. Its response-r value and correlation-matching gaps are **`NA`**, and the pair status is **`not_evaluable`** in validation and test.

This correction was made after the original calibration. The record retains the originally selected zero endpoint and its strength. Raw correlations remain in diagnostic fields; the scientific score is marked undefined because the residual fluctuations are at the numerical floor. The validity assessment covers both the 76 exact constants and the 436 numerical-floor outputs.

The endpoint rule follows its exact weight intervention and saved numerical evidence, without imposing a general cutoff on small-amplitude predictions. Other strengths and temporal zero keep their original support. Finite error or trajectory values at spatial zero remain descriptive endpoint values, with response-correlation matching marked unevaluable. [Numerical support](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/response_numeric_support.csv) and [the correction manifest](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/manifest.json) identify the affected endpoint and retained records.

## 6. Fixed inference and metric definitions

### One common measurement for every intervention

Each modified model produces a complete-trial prediction in evaluation mode. After retaining the aligned scoring interval, trial/time response scores are evaluated directly. For representation and trajectory scores, test repeats are averaged within each movie. Neural references are unchanged throughout.

For encoder-target-scaled prediction $P$ at scoring-window index $u$ and neuron $n$, the fixed conversion is

$$
G(P)_{u,n}
=
\frac{P_{u,n}}{e_{u,n}}\,g_n,
$$

where $e_{u,n}$ is the training-derived encoder precision and $g_n$ is the training-derived GPFA precision. Encoder precision depends on time and neuron; GPFA precision depends on neuron. Recorded responses and all intervention predictions use the same conversion, neuron order and full-window GPFA posterior. Scales, latent axes, GPFA parameters and checkpoint selections remain fixed. [Shared scaling and inference](../02_trajectory_measurement/SETUP.md#4-shared-scaling-and-posterior-inference) provides the common measurement details.

Each intervention follows the same path: complete-trial inference → fixed scored frames → trial/time response matching or repeat means → fixed GPFA conversion and trajectory scores. The neural reference, normalisation and posterior remain unchanged.

### Response and time-aware representation scores

Supplementary response calculations centre and accumulate in float64, leaving exact constants undefined. Individual curves use their own defined response neurons; pairs use common finite support. The zero-spatial validity rule identifies numerical-floor residuals separately from evaluable response variation.

| Measure | Input and calculation | Direction |
|---|---|---|
| Mean per-neuron response r | Pearson correlation over trial × time within each neuron, followed by a mean over defined neurons; paired comparisons share finite support. | Higher is better |
| Response normalised mean squared error (NMSE) | Pooled mean squared prediction error divided by the observed response tensor's pooled scalar variance. | Lower is better |
| Condition/time centred kernel alignment (CKA) | Average repeats, flatten movie × time and centre each neuron column; compute linear CKA. | Higher is better |
| Temporal-difference CKA | Apply the same CKA to within-movie first differences of repeat means. | Higher is better |
| Condition/time-state representational similarity analysis (RSA) | Sample every tenth frame, form population-pattern correlation distances and compute Spearman correlation over shared finite upper-triangle entries. | Higher is better |

For column-centred response matrices $X$ and $Y$,

$$
\operatorname{CKA}(X,Y)=
\frac{\|X^{\mathsf T}Y\|_F^2}
{\sqrt{\|X^{\mathsf T}X\|_F^2\,\|Y^{\mathsf T}Y\|_F^2}}.
$$

These representation measures retain temporal correspondence in output-space neural responses. They provide additional comparisons alongside the response-r matching target.

### Trajectory measures

Let $Z$ and $\widehat Z$ denote the neural and intervention trajectories, each with shape `[6, 250, 4]`. Define the same per-axis centre for the pair,

$$
m_q=\frac{1}{2CT}\sum_{c=1}^{C}\sum_{u=1}^{T}
       (Z_{c,u,q}+\widehat Z_{c,u,q}),
\qquad C=6,\quad T=250.
$$

| Measure | Calculation | Role and direction |
|---|---|---|
| Position correlation | Subtract $m$ from both arrays, flatten and compute Pearson r. | Primary; higher is better |
| Velocity-direction cosine | Take within-movie first differences divided by $\Delta t=1/30$ s; average corresponding direction cosines on jointly nonzero-norm support. | Primary; higher is better |
| Normalised position root mean squared error (RMSE) | $\operatorname{RMS}(Z-\widehat Z)/\operatorname{RMS}(Z-m)$. | Secondary; lower is better |
| Speed-profile correlation | Pearson r between flattened norms of the within-movie velocity vectors. | Secondary; higher is better |
| Acceleration-direction cosine | Use within-movie second differences divided by $\Delta t^2$ and average corresponding direction cosines. | Secondary; higher is better |

The normalised RMSE centre and denominator depend on the compared pair. Position denotes a GPFA latent state. Derivatives describe the smoothed offline posterior. All reported correlations, cosines, CKA scores and normalised errors are dimensionless.

Full support is 1,500 condition/time states, 1,494 velocity comparisons and 1,488 acceleration comparisons. Normal response support is 512 neurons; time-state RSA has 11,175 edges. Undefined values stay `NA`. The [main metric definitions](../03_model_comparison/SETUP.md#5-metrics) use these same state and time conventions.

## 7. Paired contrasts and uncertainty

The temporal-versus-baseline contrast for a metric $M$ is

$$
\Delta_T(\rho)=M(Y,P^{T,\rho})-M(Y,P^{T,1}).
$$

For a valid matched pair, the spatial-versus-temporal contrast is

$$
\Delta_{S-T}(\rho)
=M(Y,P^{S,\eta^*(\rho)})-M(Y,P^{T,\rho}).
$$

Here $M$ includes repeat averaging, scale conversion and GPFA inference where required. Positive spatial-minus-temporal differences favour spatial damage for correlations and cosines. Negative differences favour spatial damage for NMSE and normalised RMSE. Each intervention is scored against the same recorded neural reference, yielding a difference in agreement with that reference.

The bootstrap resamples six whole movie conditions with replacement, paired across compared predictions, for 2,000 draws using seed 42. It retains the same selected spatial level and frozen model state during resampling. The two statistical targets remain separate:

| Metric family | Point and interval target |
|---|---|
| Trajectory metrics | Recompute the pooled score from the sampled condition-average trajectories for each intervention, then take the paired difference. |
| Conventional response and representation metrics | Compute each movie's score, then resample the equal-weight mean paired movie difference. These points differ from the pooled conventional curves and matching scores. |

The 95% intervals describe paired whole-movie resampling conditional on the recorded spatial choices. Validation strength selection remains fixed during resampling, and the damage uses no random mask. Animal-, training-seed- and intervention-replicate variability would require additional runs or recordings.

The results retain the full temporal curve, failed matching and the unevaluable endpoint. The failed-tolerance comparisons retain their unmatched status, even where individual scores are finite. The intervals quantify contrasts against baseline; adjacent-level contrasts would require separate estimates.

## 8. Source records and interpretation scope

| Record or implementation | Relevant fields or functions |
|---|---|
| [Fixed diagnostic configuration](../../../code/sensorium/configs/supplementary_encoder_v1.json) | `temporal`, `spatial`, `support`, `statistics`, `zero_retention_interpretation` |
| [Temporal intervention implementation](../../../code/sensorium/src/trajectory_model_eval/extended_predictions.py) | `ablate_temporal_state`; off-centre temporal-weight scaling |
| [Diagnostic implementation](../../../code/sensorium/src/trajectory_model_eval/supplementary_encoder.py) | `intervention_state`, `temporal_support`, `select_spatial_matches`, `run_scores`, `response_columns`, `response_endpoint_status`, `mark_zero_spatial_endpoint`, `_plot_scores` |
| [Weight changes](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/weight_changes.json) | Exact tensor names and shapes, eligible and actually changed counts, tensor norms |
| [Temporal support](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/temporal_support.json) | Input offsets, centre-only support and `is_static_model=false` |
| [Validation grid](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/validation_grid.csv) | Original calibration values for the fixed spatial grid |
| [Matching records](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/spatial_response_matches.csv) | Frozen retentions, corrected validation display columns, test scores and gap, matching status; `raw_*` and `original_*` retain calculation history |
| [Numerical support](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/response_numeric_support.csv) | Constant-output counts, residual ranges and scientific response validity |
| [Diagnostic manifest](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/manifest.json) | `baseline_test_max_absolute_error`, `primary_sources`, `supplementary_selection_sha256`, `zero_spatial_endpoint_correction` |

[Results](RESULTS.md#8-result-records) links the score and interval tables. The experiment tests dependence on specified weight tensors in one trained network, alongside one particular spatial-damage comparison. The interventions retain different weight counts, pathways, perturbation geometry and response statistics. These differences define a within-model sensitivity test; isolating temporal computation or a causal neural mechanism would require further controls.

Replication across checkpoints, neuron selections and alternative damage families remains a future extension. The main model and GPFA remain unchanged; broader robustness remains unassessed after withdrawal of the auxiliary sensitivity fits. The supported interpretation is restricted to the retained interventions, valid matching levels and named metrics.
