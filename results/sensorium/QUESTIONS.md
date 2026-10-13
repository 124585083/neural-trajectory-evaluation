# Sensorium: Research Questions and Decisions

This document explains methodological choices in the completed `retrained_v1` study through observed results, measurement assumptions and possible interpretations. It makes no claim about personal motivations or preregistration. The question is whether time-resolved population trajectories help evaluate neural encoders alongside response prediction and representational similarity. The analysis evaluates **predicted neural responses and trajectories inferred from them**. Biological computations and object-specific internal states remain outside the identified quantities.

## I. What are the models actually comparing?

### 1. What does “dynamic” mean in our Dynamic model?

Dynamic's temporal kernels of 11, 5 and 5 frames give a **19-frame receptive field**, spanning `t−18` through `t` at the aligned output. Static processes frame `t` independently; cropping its first 18 outputs aligns timestamps without mixing frames.

![Current data roles, model histories and common evaluation pathway](figures/documentation/01_shared_pathway.png)

*The current models share prediction targets and a fixed neural-defined measurement. Dynamic integrates video and appended behaviour history; pupil shifting is a separate current-time pathway. Static's leading crop supplies the aligned output window. The test movies were explored historically.*

The architecture integrates a finite input history. Once an input leaves the window, its direct contribution ends; persistent recurrent or object-indexed memory is absent. Long-term tracking and object-update rules would therefore require evidence beyond an output-trajectory score.

See [models and data](MODELS_AND_DATA.md#4-models-and-training) and [comparison setup](03_model_comparison/SETUP.md).

### 2. Are we comparing the models' hidden states or their predicted neural responses?

Both encoders predict the **same recorded neurons**. Their outputs and recorded responses enter the same scale conversion and frozen GPFA, with no model-specific rotation or refitting. This supplies common coordinates for comparing temporal population-response patterns.

Hidden features would require a separate readout or alignment. The output interface measures predicted activity, leaving intermediate variables unobserved. Different algorithms can produce similar predictions, and a neuron readout may omit useful internal states. Mechanism evidence would require examining or manipulating those computations.

See [frozen transforms](MODELS_AND_DATA.md#5-frozen-preprocessing-and-the-evaluation-pathway) and [trajectory measurement](02_trajectory_measurement/SETUP.md).

### 3. Is the Static–Dynamic comparison fair?

The models share data roles, neuron identities, scored frames and input variable types. Total parameters differ by **1.707%** (2,814,015 versus 2,862,063). The current Static checkpoint passed 15 single-frame perturbation checks; a deliberately coupled negative control failed.

Similar parameter totals leave substantial computational differences. Readouts dominate the count, while cores contain **50,624 versus 98,672** parameters. Spatial processing, regularisation, initialisation and history access differ. Dynamic integrates appended behaviour as well as video; spatial damage also affects both input types. The comparison therefore evaluates complete architectures; attributing the result specifically to visual temporal processing requires further controls. The frame check establishes only the tested inference-mode property.

See [implementation checks](01_model_validation/RESULTS.md) and [weight-control setup](06_temporal_weight_controls/SETUP.md). Separating visual history, behaviour history and other architectural differences would require further controls, which were not completed here.

### 4. Which data trained the models, and which data tested the final results?

The diagram separates **1,395 training**, **349 validation** and **293 test trials**. Train determines encoder parameters/scales and fits pilot GPFA; validation selects checkpoints, GPFA initialisation and control strengths. The labelled repeated-movie `oracle` tier supplies the current test; hidden-tier placeholders are excluded. These current settings and GPFA parameters remain fixed during test scoring.

This is **held-out evaluation after retraining** using previously explored test movies. That history limits external-validation claims. Approximate overlap screening found no verified overlap among 7,725 candidate pairs, but source-film identities remain unavailable.

Response scoring covers five sessions and 40,034 neurons. Trajectory and intervention scoring covers **one session, 512 neurons, six movies and 58 test trials**, with 282 train and 66 validation trials. Five response sessions are not five trajectory or necessarily five animal replications. Only one encoder seed is available.

See [data roles](MODELS_AND_DATA.md#3-training-validation-and-test-data) and [comparison scope](03_model_comparison/SETUP.md).

## II. Why use trajectories, and how reliable is the measurement?

### 5. Why measure both trajectory position and direction?

Position asks whether corresponding times occupy similar **GPFA states**, not similar physical object locations. Direction asks whether neighbouring inferred states change in similar directions. Paths can occupy similar regions in different orders, or change similarly while displaced.

The primary measures are position correlation and velocity-direction cosine. Position nRMSE adds error magnitude; speed correlation and acceleration-direction cosine describe first- and second-difference structure. Their interpretation concerns the smoothed offline posterior; visual object speed and exact neural latency require different measurements. The full five-metric battery is retained, including unresolved speed differences. The records label the primary metrics but do **not** establish independent preregistration before earlier oracle inspection.

See [metric definitions](03_model_comparison/SETUP.md#5-metrics) and [complete comparisons](03_model_comparison/RESULTS.md).

### 6. Why use one shared GPFA space, and what might it miss?

GPFA models `y(t) = C z(t) + d + ε(t)` with linear loadings, neuron-specific observation noise and a smooth temporal prior. Training neural responses fit four latent dimensions; validation selects the **0.25-second initialisation**, followed by a train-only fit converging in 19 EM iterations. Both models use that same frozen measurement.

Freezing places both encoders in common neural-defined coordinates. Four dimensions retain only part of 512-neuron responses, and full-window inference uses observations on both sides of a time point. **63 observations queried at 250 times do not become 250 independent neural samples.** The inferred state is a descriptive, offline estimate; its biological identity remains unresolved.

### 7. What do repeat reliability and null tests each tell us?

![Repeat agreement and limited response reconstruction](figures/documentation/02_measurement.png)

*Reliability ranges describe 200 repeat assignments of the same six movies. Reconstruction uses the full query grid with separate response-level denominators. High repeatability and low variance coverage can coexist; the panels do not share a single measurement scale.*

**Reliability** compares disjoint repeats of the same movie. Position and direction agreement improve markedly from one repeat to balanced four/five-repeat halves. **Nulls** instead test specified broken correspondences: movie identity, phase, order or population coordination. All five metrics exceed prescribed nulls in every balanced partition; both primary metrics also do so at one and two repeats. Some one-repeat secondary comparisons fail and remain reported.

The current GPFA reconstructs only **1.89% of centred single-trial variance and 2.56% of condition-average variance**. A small shared component can be reliable without reconstructing most neural activity. Repeat partitions are not biological replications or population p-values; balanced reliability is neither single-trial reliability nor a universal hard noise ceiling. Scores are not reliability-normalised. Withdrawn auxiliary fits provide no cross-dimension or observation-grid robustness claim.

See [measurement setup](02_trajectory_measurement/SETUP.md) and [complete reliability, null and reconstruction results](02_trajectory_measurement/RESULTS.md).

## III. What do the model comparisons actually establish?

### 8. Where does Dynamic outperform Static, and where is the evidence weaker?

![Five-session responses and complete pilot trajectory comparisons](figures/documentation/03_model_comparison.png)

*Response points use all neurons in five sessions; trajectories use one 512-neuron pilot. The trajectory intervals concern pooled scores recomputed on six whole resampled movies. Per-movie panels show actual condition IDs and preserve the small movie_00005 direction exception.*

Dynamic improves response correlation in every session (unweighted means **0.1596 versus 0.1911**). In the pilot, position and direction differences favour Dynamic, as do position error and acceleration. Speed remains unresolved. Position improves in six movies, direction in five; the exception is not excluded.

Time-aware RSA/CKA also favour Dynamic. Prediction-null comparisons show that both models, including Static, retain measured correspondence. These results support better tested response prediction and selected trajectory summaries. **Broader “brain-likeness,” complete population fidelity and a unique cause of the gain remain unestablished**.

See [model comparison results](03_model_comparison/RESULTS.md).

### 9. What does the response-matching control add?

![Validation-selected scalar response matching and trajectory contrasts](figures/documentation/05_response_matching.png)

*The ±0.005 band is a fixed tolerance, not an equivalence interval. Trajectory intervals use fixed calibration and one trial-keyed noise realisation; speed remains unresolved.*

Training predictions determine a per-neuron noise scale; validation selects additive-noise amplitude **σ ≈ 0.869749**. Test applies the same noise rule and clipping without retuning. The scalar response-r gap stays within tolerance, yet both primary trajectory contrasts remain positive.

The control shows that similar **mean per-neuron correlation** can coexist with different trajectory scores. It matches that scalar alone: neuron-level accuracy, NMSE, time-aware RSA/CKA and error structure can still differ. The comparison uses perturbed frozen outputs, with independently retrained accuracy-matched models left untested. Noise may affect trial-level correlation more than repeat-averaged, smoothed trajectories. This explanation remains plausible, with its contribution unquantified.

See [response-matching setup](05_response_matching/SETUP.md) and [results](05_response_matching/RESULTS.md).

### 10. What happens when we reverse the predicted time sequence?

![Output reversal, pattern invariance and time-sensitive comparisons](figures/documentation/04_reversal.png)

*Reversal occurs in encoder-target scale before the time-dependent GPFA conversion. Its operation differs from the after-conversion primary null. Pattern invariance is deterministic; trajectory and conventional intervals retain their separate pooled and movie-mean targets.*

Changing **A → B → C** to **C → B → A** preserves the time average. The tested movie-average RSA/CKA therefore remain unchanged, while both primary trajectory measures decline. Response correlation and several time-aware RSA/CKA summaries also decline; Static temporal-difference CKA is unresolved.

The result identifies **what a particular aggregation discards**. Time-aware RSA/CKA retain temporal sensitivity alongside GPFA. Response accuracy was not matched. Because scale conversion depends on time, `G(RP)` and `R(GP)` differ; neither operation reverses latent vectors or input videos.

See [temporal test setup](04_temporal_metric_tests/SETUP.md) and [results](04_temporal_metric_tests/RESULTS.md).

### 11. Do trajectory metrics reveal new information, or describe the same responses differently?

Trajectories are transformations of the same neuron-by-time responses, not new recordings or information independent of the complete response data. Their value is **relative to particular summaries**: mean correlation compresses errors to one number, time averaging discards order, and GPFA measures shared-state geometry after projection and smoothing.

Response matching and reversal show that different summaries can conceal or expose different properties. Independence from other conventional metrics remains unresolved; time-aware RSA/CKA also detect differences. The usefulness of a trajectory metric rests on its reliability, null sensitivity and scientific interpretation, beyond its ability to change a model ranking.

See [comparison](03_model_comparison/RESULTS.md), [reversal](04_temporal_metric_tests/RESULTS.md) and [response matching](05_response_matching/RESULTS.md).

## IV. What can the interventions tell us about computation?

### 12. What do temporal-weight and spatial-weight changes tell us?

Independent encoder training compares complete learned systems. Weight attenuation asks a local question about **one frozen model under particular damage**, which can also disrupt feature scales and its learned operating regime.

![Current temporal attenuation and validation-selected spatial controls](figures/04_model_comparison/retrained_v1_diagnostics/encoder/weight_attenuation.png)

*The x axis is temporal retention; spatial controls appear at their associated temporal target, not their own retention. Black crosses mark test mismatch; red outlines identify the correlation-degenerate spatial-zero endpoint. Finite geometry at that endpoint does not make it a valid response match.*

Both primary trajectory scores fall as off-centre temporal weights are removed. Only two nontrivial spatial/temporal pairs retain test response matching. At temporal **0.75**, spatial damage preserves higher position and direction. At **0.25**, spatial damage has lower position but higher direction. The **0.50** pair fails test tolerance, so the valid results do not support uniform temporal-specific damage.

Two endpoint distinctions remain essential. Temporal zero retains video/behaviour at **t−9** and pupil shifting at **t**; it is not Static. Spatial zero produces 76 exact constants and 436 float32-floor outputs, so its correlation is **NA / numerically_degenerate** and matching is **not_evaluable**. These interventions establish metric-dependent sensitivity to specified weights. A biological temporal mechanism and an isolated effect of visual history remain unresolved.

See [weight-control setup](06_temporal_weight_controls/SETUP.md) and [results](06_temporal_weight_controls/RESULTS.md).

### 13. Can the current results tell us how the brain computes over time?

The evidence has three levels:

| Level | Supported result | Boundary |
|---|---|---|
| Prediction | Reduced Dynamic predicts the tested responses better. | This does not explain the gain. |
| Population representation | Its outputs better match selected properties in one fixed four-dimensional neural measurement. | Limited reconstruction prevents a complete population-fidelity claim. |
| Within-model dependence | Specified temporal-filter interventions change the trained model's predictions. | Architectures, behaviour history and generic damage remain incompletely isolated. |

A **biological computation** claim needs an additional discriminating test. For example, object-specific selective updating would require manipulating object assignment while controlling relevant motion or expectations, then comparing algorithms that predict different neural or behavioural outcomes. The present data lack that manipulation and a measurement of object-specific state updates. A theory of those updates would need to specify more than a finite convolutional history.

The current outcome is methodological and comparative: reliable time-resolved measurements expose differences obscured by selected scalar or averaged summaries, and the tested Dynamic model better matches some properties. They can help define predictions for a future test that distinguishes competing mechanisms.

See [comparison scope](03_model_comparison/RESULTS.md), [measurement limits](02_trajectory_measurement/RESULTS.md) and [weight controls](06_temporal_weight_controls/RESULTS.md).
