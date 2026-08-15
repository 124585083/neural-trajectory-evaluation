# GPFA Validation

Trajectory metrics are useful only if their coordinate system is reproducible and independent of the encoding models being evaluated. The GPFA measurement was therefore defined and validated using recorded neural activity before it was used for the Static–Dynamic trajectory comparison.

## 1. What “neural-data-defined GPFA” means

The shared latent coordinate system is fitted exclusively from recorded training-tier neural responses. For neural population response `y(t)` and latent state `x(t)`, the observation model is:

```text
y(t) = C x(t) + d + ε(t),    ε(t) ~ Normal(0, R)
```

The observation manifold `C`, neural mean `d`, diagonal observation noise `R`, and latent temporal priors are shared. After fitting, they remain fixed for recorded oracle responses and for every encoding-model prediction.

The comparison obeys the following invariants:

- GPFA parameters are learned from training-tier neural responses only.
- Oracle responses and Static/Dynamic predictions do not fit or select the GPFA.
- Recorded responses and both models use the same neuron identities and order.
- All three inputs use the same frozen train-derived scaling, `C`, `d`, `R`, temporal priors, and posterior inference.
- No model-specific refitting, latent rotation, Procrustes alignment, scaling, or axis selection is permitted.

The trajectory comparison is consequently anchored to a coordinate system learned from the recorded neural population rather than to separately optimized model-specific spaces. Earlier repository documents call this design “brain-defined GPFA”; **neural-data-defined GPFA** is the clearer reader-facing term used here.

## 2. Leakage control and temporal support

### Train/oracle separation

GPFA fitting, neural scaling, dimensionality selection, and timescale-initialization selection use neural responses from the official training tier. The 58 oracle trials are introduced only after the relevant GPFA has been frozen, first to evaluate neural split-half reliability and then to evaluate encoding-model predictions.

### Evaluation interval and response scaling

The assay uses original frames 50–299: 250 timestamps covering the same temporal support as the encoding-model evaluation. A deterministic, seed-42 selection fixes 512 neuron identities and their order independently of response magnitude, reliability, or model performance.

Scaling is also train-only. One scalar precision is estimated for each selected neuron from the applicable training responses; no oracle-derived or model-derived scale is used. Recorded responses and model predictions are converted into this same frozen GPFA input scale. Detailed normalization and conversion formulas are provided in [Methods](METHODS.md).

### Observation and posterior-query grids

Sensorium arrays are represented on a 30 Hz grid, but the underlying calcium signal was acquired at approximately 8 Hz before resampling. The primary GPFA therefore observes every fourth official frame, producing 63 observations at an effective rate of approximately 7.5 Hz. Its continuous GP posterior is then queried at all 250 timestamps on the 30 Hz evaluation grid.

Velocity and especially acceleration on this query grid are derivatives of the inferred GPFA posterior trajectory. They are **not independently observed 30 Hz neural dynamics**. Direct 30 Hz fitting is treated as a sensitivity analysis, not as additional independent neural information.

## 3. GPFA selection and freezing

The Phase 2 method-development analysis compares latent dimensions `q = 4, 8, 12, 16` using marginal negative log likelihood on 70 held-out training-tier calibration trials. The raw mean calibration NLL is lowest at `q = 16`, but all four candidates fall within one standard error of that minimum. The one-standard-error rule therefore selects the smallest eligible model, `q = 4`, as the conservative primary dimensionality.

At `q = 4`, initial timescales of `0.125`, `0.25`, `0.5`, and `1.0` seconds are compared on the same calibration data; `0.25 s` is selected. Oracle responses, oracle reliability, encoding-model predictions, and Static–Dynamic comparison scores do not participate in either selection.

After selection, the GPFA and its preprocessing metadata are refitted on the designated neural training set and frozen. The four-dimensional solution and its approximately 0.25-second timescales are analysis parameters: they are not interpreted as the unique biological dimensionality or a uniquely identified cortical time constant.

## 4. Why two GPFA fits appear in the repository

The repository contains two related but non-identical frozen GPFA objects:

| Role | Available train trials | Fit / calibration split | Final refit | Main use |
|---|---:|---:|---:|---|
| Method-development GPFA | 348 | 278 / 70 | 348 | Establish measurement reliability and sensitivity |
| Comparison-subset GPFA | 174 locked trials | 139 / 35 | 174 | Final Static–Dynamic pilot comparison |

Both use the same pilot session, deterministic 512-neuron subset, frames 50–299, approximately 7.5 Hz observation logic, 30 Hz posterior-query grid, and six repeated oracle movies. Both exclude oracle responses and encoding-model predictions from fitting, scaling, and selection.

Phase 2 establishes the measurement design on the full training tier. Phase 4 then repeats the reliability gate for the separately fitted comparison-subset GPFA that is actually used to evaluate Static and Dynamic predictions. The comparison fit retains the locked primary dimension `q = 4`; calibration among the tested initial timescales selects `0.25 s`, and the final learned timescales are approximately `0.2550`, `0.2551`, `0.2524`, and `0.2580 s`.

The reliability values reported for these two stages therefore describe two frozen fits serving different roles; they should not be read as repeated estimates from one identical fitted object.

## 5. Reliability protocol

Reliability is tested on 58 oracle trials grouped by stimulus into six repeated natural-movie conditions with repeat counts `10/10/9/10/9/10`. For every balanced split:

1. repeats are divided into two disjoint, equal-sized halves within each movie;
2. conditions with ten repeats contribute five trials per half, while conditions with nine contribute four per half and leave one unused;
3. neural responses are averaged independently within each half;
4. both condition-average tensors are passed through the same frozen GPFA posterior inference;
5. the resulting trajectories are compared without latent alignment.

The primary battery reports complementary aspects of agreement:

- **Position correlation** and **normalized position RMSE** measure time-aligned latent state.
- **Velocity-direction cosine** measures local direction of motion.
- **Speed-profile correlation** measures whether fast and slow trajectory segments occur at corresponding times while discarding direction.
- **Acceleration-direction cosine** is a higher-order, higher-variance diagnostic of local directional change.
- **Path-length similarity** describes total traveled distance but is retained only as a descriptive quantity.

Condition-average trajectories are primary because their reliability is substantially higher than single-repeat trajectories.

## 6. Matched nulls

Smoothness and autocorrelation can produce apparently reliable trajectories without meaningful stimulus-locked population dynamics. Matched nulls therefore disrupt specific information while retaining other structure:

| Null | Information disrupted |
|---|---|
| Condition derangement | Stimulus identity |
| Nonzero circular shift | Absolute timing and phase |
| Time reversal | Temporal direction while preserving the visited state set |
| Independent-neuron shift | Coordinated population timing |
| Block shuffle | Temporal order at controlled local/global scales |
| Full frame shuffle | Local and global temporal order |

Observed and null values are paired within the same repeat split. For the principal metrics, the reader-facing summary reports how often the observed split is better than its matched null and the corresponding split/null failure count.

These repeated splits reuse the same 58 oracle trials and are not independent biological samples. Statements such as `200/200` paired superiority and `0/200` split/null failures are therefore descriptive robustness summaries, not formal independent-sample biological p-values.

## 7. Main reliability results

Both frozen fits recover highly reproducible condition-average trajectories:

| Metric | Full-train method-development GPFA | Comparison-subset GPFA |
|---|---:|---:|
| Position correlation | 0.8566 | 0.8583 |
| Normalized position RMSE | 0.5428 | 0.5424 |
| Velocity-direction cosine | 0.6627 | 0.6368 |
| Speed-profile correlation | 0.7434 | 0.7777 |
| Acceleration-direction cosine | 0.6163 | 0.5579 |

For the full-train fit, the empirical 2.5th–97.5th percentile split intervals are `0.8184–0.8885` for position, `0.4789–0.6527` for normalized RMSE, `0.6250–0.6949` for velocity, `0.6791–0.7951` for speed, and `0.5824–0.6548` for acceleration. The corresponding comparison-subset intervals are `0.8177–0.8906`, `0.4682–0.6453`, `0.5969–0.6723`, `0.7064–0.8354`, and `0.5181–0.6002`. These are distributions across repeat splits, not confidence intervals for independent biological samples.

For both primary fits, position, normalized RMSE, velocity, speed, and acceleration outperform condition, timing, direction, block-order, and population-synchrony nulls in all `200/200` paired splits, with `0/200` split/null failures. The agreement of the two reliability gates supports using the locked comparison-subset GPFA as the common coordinate system for model evaluation.

Path-length similarity is `0.9449` for the full-train fit, but its high raw reliability has a different interpretation and does not qualify it as a primary temporal metric.

## 8. Sensitivity and saturation

The method-development analysis varies split count, neuron count, latent dimension, neural training fraction, GPFA seed, and observation grid. Position is particularly stable: it changes from `0.8532` at 512 neurons to `0.8594` at 1,024 neurons, spans `0.8465–0.8538` across `q = 4/8/12/16`, and spans `0.8532–0.8622` across three GPFA seeds. Correlation-like estimates are largely stable by 100–200 repeat splits.

The comparison-subset analysis independently evaluates 16 profiles spanning:

- 128, 256, and 512 neurons;
- `q = 4, 8, 12, 16`;
- 25%, 50%, 75%, and 100% of the locked selected-train trials;
- GPFA seeds 42, 314, and 2718;
- observation steps 4 and 1.

Across these profiles, the reliability ranges are:

| Metric | Comparison-profile range |
|---|---:|
| Position correlation | 0.8173–0.8698 |
| Normalized position RMSE | 0.5209–0.6151 |
| Velocity-direction cosine | 0.5365–0.6407 |
| Speed-profile correlation | 0.6406–0.7958 |
| Acceleration-direction cosine | 0.4460–0.5778 |

Each profile uses 100 splits and five matched null classes. Minimum paired superiority remains `100/100`, and the maximum split/null failure count remains `0/100`.

The qualitative null-separation decision is therefore robust. Exact metric values are less invariant: position is the most stable, whereas speed and higher derivatives vary more with latent dimension, neuron count, observation grid, and fit seed. For example, the full-train 7.5 Hz and direct 30 Hz fits yield velocity cosines of `0.6634` and `0.6190`, respectively, even though the null-separation conclusion is unchanged.

## 9. Important negative findings and measurement limits

### 9.1 Path length is not a primary temporal metric

Path length is highly reproducible across real split halves, but circular shifting and time reversal preserve nearly the same total distance traveled. Against circular shift, path-length superiority is only `126/200`, with `74/200` failures; against time reversal it is `89/200`, with `111/200` failures.

Path length can describe trajectory magnitude, but it cannot establish correct temporal alignment or direction and is therefore descriptive only.

### 9.2 Single-trial trajectories are substantially less reliable

The main reliability values average four or five repeats per condition within each split half. With one repeat per half, position correlation falls to `0.5545` (split interval approximately `0.4035–0.6613`), compared with approximately `0.8566` for the primary condition-average analysis. Velocity, speed, and acceleration also fall to `0.3417`, `0.3694`, and `0.3051`.

Single-repeat trajectories are moderately, not highly, reliable. The primary model comparison therefore uses condition-average trajectories; the condition-average reliability value must not be generalized to single trials.

### 9.3 GPFA captures a limited fraction of total neural variance

Reconstructing oracle responses through the GPFA observation model gives population `R²` values of approximately `0.0200` at `q = 4`, `0.04231` at `q = 8`, and `0.07148` at `q = 16`. Higher dimensionality increases response coverage, but even the larger diagnostic fits capture a limited fraction of total response variance.

The GPFA isolates a low-dimensional, repeatable, stimulus-locked shared subspace. High split-half trajectory reliability does not imply high-fidelity reconstruction of the complete neural response. Response prediction must therefore remain a separate performance criterion, with trajectory evaluation used as a complement.

### 9.4 Speed and higher derivatives require cautious interpretation

Speed changes more with latent dimension, neuron count, and sampling grid than position does. Acceleration is a noisier second-derivative diagnostic, and both velocity and acceleration on the 30 Hz query grid describe the inferred posterior rather than independent neural observations. These metrics remain informative because they reliably separate the relevant nulls, but their exact magnitudes are more analysis-sensitive than position.

### 9.5 Latent dimension and timescale are analysis parameters

The one-standard-error rule justifies `q = 4` as a conservative analysis choice, not as the unique true neural dimensionality. Likewise, the selected approximately `0.25 s` timescale behaves as a smoothing/analysis parameter and is not identified as a unique cortical timescale.

## 10. Additional validation findings

### Shared rather than behavior-conditioned temporal priors

The training trials were divided into low/middle/high behavior-state strata using training-tier behavioral covariates without neural responses. A shared-prior model obtains calibration NLL per observation `1.4031800295`, compared with `1.4031800335` for behavior-conditioned timescales with shared `C`, `d`, and `R`. The conditioned prior provides no held-out improvement, so one shared natural-video temporal prior is retained.

### Stricter-convergence diagnostic

A stricter q=4 EM fit shifts the learned timescales slightly and changes the fitted subspace modestly, but changes the reliability metrics by less than approximately `0.004` on the same 50 splits. The trajectory conclusions are therefore more stable than the exact fitted timescale values.

Detailed repeat-count, strict-convergence, and response-variance-coverage diagnostics are preserved in [GPFA Supplementary Diagnostics](supplementary/validation/GPFA_DIAGNOSTICS.md).

## 11. What this validation establishes

The validation supports the following conclusions:

- Repeated natural movies produce reproducible condition-average trajectories in a frozen neural-data-defined GPFA space.
- The metric battery is sensitive to stimulus identity, temporal alignment, local direction, speed modulation, and coordinated population structure.
- The coordinate system and preprocessing remain independent of Static- or Dynamic-specific fitting and alignment.
- Null-separation decisions survive changes in neuron count, latent dimension, training fraction, fit seed, observation grid, and split count.
- The separately validated comparison-subset GPFA can serve as a common coordinate system for the Static–Dynamic pilot comparison.

This validation does **not** establish high-fidelity reconstruction of total neural activity, equally strong single-trial reliability, a unique biological latent dimension, a unique biological GP timescale, or that trajectory metrics can replace response correlation.

## 12. Use in model comparison

The model-comparison flow is fixed:

```text
recorded response / Static prediction / Dynamic prediction
    → same 512 neuron identities and order
    → same frozen train-derived scaling
    → same frozen GPFA posterior inference
    → same trajectory metrics
```

This document answers whether that measurement can be trusted. The resulting Static–Dynamic findings are reported separately in [Results](RESULTS.md); no model-comparison scores are used to establish the validation described here.

## 13. Documentation and reproducibility pointers

- [Methods](METHODS.md): full preprocessing, GPFA inference, trajectory-metric, null, and bootstrap definitions.
- [Results](RESULTS.md): Static–Dynamic scientific findings obtained with the validated comparison GPFA.
- [Internally locked GPFA protocol](supplementary/protocols/GPFA_PROTOCOL_LOCKED.md): result-blind Phase 2 analysis specification. It was internally locked before reliability results were inspected and was not registered on an external preregistration platform.
- [Phase 2 implementation and configuration](../experiments/02_gpfa_reliability/): assay fitting, reliability, null, and sensitivity workflow.
- [Full-train frozen GPFA](../models/gpfa_reliability/) and [comparison-subset frozen GPFA](../models/gpfa_model_comparison/): released fitted objects and preprocessing metadata.
- [Phase 2 reliability outputs](../results/tables/02_gpfa_reliability/) and [comparison-GPFA reliability outputs](../results/tables/04_model_comparison/): compact public validation artifacts.
