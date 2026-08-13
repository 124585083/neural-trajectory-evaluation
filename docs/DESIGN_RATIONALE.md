# Design Rationale

## Central question

This project asks whether trajectory-based neural evaluation reveals aspects of temporal encoding that are not fully summarized by point-wise response correlation or output-space population-response similarity.

The central hypothesis is:

> If a Dynamic encoding model improves not only the accuracy of individual neural responses but also the temporal organization of population activity, then a reliable trajectory metric should distinguish Dynamic from Static models after controlling model capacity and response quality, and it should degrade systematically when learned temporal history is removed.

The purpose of the design is not to make GPFA replace response correlation, RSA, or CKA. It is to test whether a brain-defined trajectory analysis contributes additional, interpretable evidence about temporal computation. In this project, RSA and CKA operate on predicted neural population responses, not hidden network activations.

## Why Dynamic Sensorium?

### Continuous stimuli make temporal computation testable

Dynamic Sensorium 2023 contains natural movies rather than isolated static images. Each prediction therefore belongs to a continuous sequence with meaningful temporal order. This is necessary for asking whether a model reproduces the evolution of a neural population state, including its position, direction, and speed in a shared latent space.

A static-image benchmark can test spatial feature selectivity and response prediction, but it cannot cleanly distinguish a model that uses temporal context from one that evaluates each frame independently. Dynamic Sensorium makes that distinction explicit: the Static model sees each frame independently, whereas the Dynamic model learns spatiotemporal filters across neighboring frames.

### It supports a controlled Static-Dynamic comparison

Both models can be trained and evaluated on the same:

- five recording sessions;
- movie frames and behavioral covariates;
- train and oracle trial identities;
- neurons and neuron order;
- response normalization;
- Gaussian readout family and pupil shifter;
- loss, optimizer, batch construction, and evaluation interval.

This common data contract is more important than comparing scores reported on unrelated static and dynamic benchmarks. A native Sensorium 2022 Static score is not an appropriate numerical control because Sensorium 2022 uses static images and a different task. The project therefore retrains the official Static Sensorium architecture on Dynamic Sensorium 2023 and labels it explicitly as **Static-on-Dynamic**.

### Repeated movies allow the trajectory assay to be validated first

The oracle tier contains repeated presentations of six natural movies in the primary method-development session. Independent repeat halves can be averaged and compared without fitting on the evaluation responses. This enables split-half reliability, matched temporal nulls, and sensitivity analyses before the trajectory metric is used to judge an encoding model.

That order matters. A metric should not be accepted merely because it favors the preferred model. It must first show that it recovers reproducible neural structure and rejects perturbations that destroy condition identity, temporal order, timing, or population synchrony.

### Why this dataset is not sufficient by itself

Dynamic Sensorium is a suitable proof-of-concept test bed, not universal evidence about cortical dynamics. The deeper GPFA comparison currently uses one session, a deterministic 512-neuron subset, and six repeated movies. Calcium activity was acquired at approximately 8 Hz and represented on a 30 Hz grid, which limits the temporal bandwidth that can be interpreted. The conclusions must therefore be extended across sessions, neuron subsets, model seeds, and potentially other dynamic datasets before becoming a broad biological claim.

## Why parameter matching?

### The confound

The full official Dynamic model has more trainable parameters than the Static-on-Dynamic model. If it performs better, the difference could reflect temporal computation, greater capacity, a wider feature representation, or some combination of these factors. A comparison between the full models is ecologically useful as a benchmark, but it is not a clean mechanistic control.

### The controlled comparison

The reduced Dynamic model applies one predeclared width multiplier to every Dynamic core layer while preserving the original architecture's depth, temporal kernels, spatial kernels, temporal receptive field, nonlinearities, normalization, regularization, readout, shifter, and output nonlinearity.

| Model | Total parameters | Temporal computation |
|---|---:|---|
| Static-on-Dynamic | 2,814,015 | None in the core; frames are evaluated independently |
| Parameter-matched Dynamic | 2,862,063 | Learned Factorized3D temporal convolutions |

The total difference is 48,048 parameters, or 1.707%. The two models have exactly the same five-session readout parameter count. Parameter matching therefore reduces the plausibility of the simple explanation that Dynamic wins only because it has more parameters.

### What parameter matching does and does not establish

Parameter matching controls total parameter count; it does not make the architectures identical in every other measure of complexity. A 2D frame-wise core and a 3D spatiotemporal core can differ in optimization geometry, inductive bias, effective computation, and feature reuse even when their parameter counts are close.

The appropriate interpretation is consequently:

> A Dynamic advantage after parameter matching is evidence that the effect is not explained by total parameter count alone.

It is not proof that every remaining difference is caused exclusively by one temporal kernel, one lag, or a uniquely biological mechanism. Graded temporal ablation is added to test the temporal component more directly.

## Why output-space RSA and CKA?

The present RSA and CKA analyses compare the model's predicted neural responses with recorded neural responses:

```text
predicted neural population response Y_hat(t)
    versus
recorded neural population response Y(t)
```

They do not compare a hidden model representation `H(t)` with the brain. The more precise description is therefore **output-space RSA/CKA** or **population-response representation similarity**, rather than hidden-layer representational analysis.

This choice keeps the comparison target fixed. Response correlation, RSA/CKA, and GPFA trajectory metrics all receive the same predicted neural-response tensor and ask different questions about it:

- response correlation tests neuron-wise predictive accuracy;
- output-space RSA/CKA tests the geometry of predicted population-response patterns;
- frozen-GPFA metrics test the temporal evolution of those population responses in a brain-defined latent space.

This shared output space makes the incremental-information question cleaner because differences between metrics cannot be attributed to choosing different hidden layers or fitting a mapping from hidden units to neurons. Hidden-layer RSA/CKA remains a valid extension, but it would answer a different question about internal network computation and would introduce layer-selection and correspondence choices outside the primary controlled comparison.

## Why is GPFA fitted only to brain data?

### The comparison space must be independent of the models being judged

If GPFA were fitted separately to Static predictions and Dynamic predictions, each model would receive its own latent axes, noise model, scale, and temporal prior. A subsequent alignment could make both trajectories look favorable while obscuring whether either model actually occupies the low-dimensional population structure found in the recorded neural data.

This project instead uses a **brain-defined** coordinate system:

1. GPFA preprocessing and parameters are estimated only from real neural responses in the official training tier.
2. Latent dimension is selected from neural training/calibration trials by marginal negative log likelihood and a one-standard-error rule; initialization is selected by calibration marginal negative log likelihood. Oracle reliability and encoding-model performance do not enter either selection.
3. The selected GPFA is frozen before model evaluation.
4. Brain responses, Static predictions, and Dynamic predictions use the same neuron order, train-only scaling, observation model, temporal prior, and latent axes.
5. No model-specific rotation, scaling, Procrustes transform, or latent alignment is allowed.

The resulting score asks whether a prediction follows a trajectory in coordinates defined independently by recorded population activity. It does not ask how well the prediction can define and fit its own latent space.

### Why GPFA rather than frame-wise dimensionality reduction?

GPFA combines a low-dimensional observation model with an explicit smooth temporal prior. This makes it possible to estimate continuous latent trajectories and compare position, local direction, speed, and acceleration. Static and Dynamic predictions are treated as new observations `y(t)` and their latent trajectories are obtained using the same frozen GPFA posterior inference with fixed `C`, `d`, `R`, and temporal priors. This is not a simple `C`-transpose or pseudoinverse projection. A frame-wise method such as PCA could summarize population variance, but it would not encode the same probabilistic temporal structure or distinguish measurement noise from a smooth latent process in the same way.

Velocity and especially acceleration on the 30 Hz query grid are derivatives of a continuous GPFA posterior inferred from 7.5 Hz-equivalent neural observations. They are not derivatives of independent 30 Hz neural measurements. High-order derivative results are therefore interpreted conservatively and checked against observation-grid sensitivity.

The choice is nevertheless pragmatic rather than ontological. The selected GPFA is a conservative summary of a reproducible shared neural subspace. It is not assumed to recover the complete neural state, the true biological dimensionality, or the unique dynamics of V1. This is why GPFA reliability, null separation, dimensionality sensitivity, neuron-count sensitivity, fit-seed sensitivity, and temporal-grid sensitivity are evaluated before model comparison.

## Why response matching?

### Response accuracy is an obvious alternative explanation

If Dynamic has higher response correlation and higher trajectory agreement, trajectory performance may simply be a consequence of generally better predictions. A trajectory advantage under those conditions is informative, but it does not establish that the trajectory metric adds information beyond response quality.

Response matching creates a more demanding diagnostic:

> When Static and Dynamic have nearly identical scalar response scores, can a trajectory metric still detect a difference in temporal organization?

### Separation of selection and test repeats

In the primary response-matched stress test, repeated oracle trials are divided into disjoint selection and test halves. The selection half is used to choose a train-independent amplitude-noise level that brings the Dynamic scalar response correlation close to the Static score. Noise is scaled by each neuron's predicted Dynamic standard deviation so the perturbation is proportional to its dynamic range rather than imposing one absolute scale across heterogeneous neurons. The held-out half is not used to select that noise level. Both predictions are then evaluated on the held-out repeats with the frozen GPFA.

This test is designed to avoid circularly choosing a perturbation that maximizes a held-out trajectory difference. It also avoids claiming formal statistical equivalence: the response scores are **nearly identical after predefined matching**, not proven equivalent under every response statistic.

### What response matching does and does not show

If trajectory differences remain after response matching, scalar response correlation is not a sufficient summary of the detected temporal difference. This is evidence for incremental sensitivity.

The stress test is not a new fair-training leaderboard. Adding amplitude noise deliberately changes one model's predictions, and it does not force output-space RSA, CKA, response variance, or every neuron-wise property to match. Its role is to test metric sufficiency, not to declare a universally superior model under artificial degradation. The validation-history response-matched checkpoint provides a complementary, naturally trained comparison, but the disjoint-repeat stress test is the cleaner control of the selected scalar response score.

## Why temporal ablation?

Parameter matching controls capacity and response matching controls one performance summary, but neither directly manipulates temporal computation. The temporal-history ablation scales the off-center weights of the learned temporal kernels while keeping the center slices, spatial core, readout, shifter, and biases fixed.

The five retention levels form a dose-response test:

```text
1.00 -> 0.75 -> 0.50 -> 0.25 -> 0.00 temporal-history retention
```

If trajectory evaluation is sensitive to learned temporal structure, direction- and order-sensitive scores should degrade as temporal history is removed. A graded curve is stronger evidence than a single intact-versus-ablated comparison because it tests whether the metric follows the severity of the temporal intervention.

This ablation still does not localize the effect to a particular layer or lag. All three temporal-convolution layers are scaled together. It also lacks a magnitude-matched non-temporal damage control. A planned control will perturb spatial or center-slice weights by a matched parameter-space magnitude without selectively removing temporal history, allowing temporal-ablation degradation to be compared with generic network damage. Until that control is run, the dose-response result supports temporal sensitivity but cannot completely exclude the possibility that progressive weight damage contributes to the decline. Layer-specific and lag-specific ablations would be required for finer causal attribution.

## What would falsify the central hypothesis?

The hypothesis is deliberately falsifiable. The following outcomes would count against its main claims in the tested setting.

### 1. No Dynamic advantage after controlling capacity

If the Dynamic trajectory advantage observed with the full model disappeared consistently after parameter matching, while response and output-space geometry metrics showed no remaining temporal-model effect, the original difference would be better explained by model capacity than by temporal computation.

### 2. No residual trajectory difference after response matching

If response-matched Static and Dynamic predictions also produced matching GPFA position and direction-sensitive trajectory scores, then trajectory evaluation would not provide evidence beyond the matched response statistic for those models and data.

### 3. No dose-response relationship under temporal ablation

If progressively removing learned temporal history left trajectory scores unchanged, improved them, or produced changes no more systematic than non-temporal control perturbations, the claim that these metrics are specifically sensitive to temporal computation would be undermined.

### 4. Conventional metrics fully account for the trajectory results

If response correlation, time-aware RSA, and CKA predicted trajectory scores essentially perfectly across held-out perturbation families, and no matched pair or temporal-order counterexample retained a trajectory difference, then trajectory evaluation would be redundant rather than incremental.

### 5. The effect fails to generalize

If the trajectory advantage repeatedly reversed or vanished across additional sessions, neuron subsets, GPFA seeds, encoding-model seeds, latent dimensions, or temporal grids, the current one-session result would not support a general claim about Dynamic models. A stable pilot effect can motivate expansion, but it cannot substitute for cross-session replication.

### Assay failure is different from hypothesis falsification

Some outcomes would invalidate the current test without directly showing that the central scientific hypothesis is false. Examples include poor neural split-half reliability, failure to reject time-order nulls, strong dependence on arbitrary latent rotation, leakage from oracle responses into GPFA fitting, or model rankings that change unpredictably with reasonable GPFA settings.

In those cases, the correct conclusion would be that the GPFA assay is not capable of adjudicating the hypothesis. The project therefore places the reliability and null-test gate before Static-Dynamic model comparison.

## Decision logic

| Observation | Interpretation |
|---|---|
| Dynamic improves response, output-space RSA/CKA, and trajectory | Dynamic has a broad modeling advantage; trajectory is descriptive but not yet shown to be incremental. |
| Dynamic retains a trajectory advantage after capacity and response controls | Supports incremental sensitivity to temporal population structure. |
| Trajectory similarity degrades with temporal-ablation severity | Supports sensitivity to the model's learned temporal history. |
| Trajectory differences vanish after response matching | Does not support added value beyond the matched response score. |
| Conventional metrics fully predict held-out trajectory effects | Suggests trajectory evaluation is redundant for the tested perturbations. |
| Neural trajectory metrics fail reliability or null tests | The assay fails; model-ranking claims should not be made. |

## Related project documents

- [Static model protocol](models/STATIC_MODEL.md)
- [Dynamic model protocol](models/DYNAMIC_MODEL.md)
- [Parameter-matched Dynamic design](models/PARAMETER_MATCHED_DYNAMIC.md)
- [Brain-defined GPFA method](methods/BRAIN_DEFINED_GPFA.md)
- [Result-blind locked GPFA reliability protocol](methods/GPFA_PROTOCOL_LOCKED.md)
- [GPFA reliability results](results/GPFA_RELIABILITY_RESULTS.md)
- [Static-Dynamic model-comparison results](results/MODEL_COMPARISON_RESULTS.md)
- [Direct answers to Q1-Q6](results/Q1_Q6_ANSWERS.md)
