# Sensorium: Research Questions and Decisions

These questions explain the model choices, trajectory measurements and interpretation of the completed `retrained_v1` study. Data-use history and the timing of analysis choices are discussed where they affect interpretation. The question is whether time-resolved population trajectories help evaluate neural encoders alongside response prediction and representational similarity. The analysis evaluates **predicted neural responses and trajectories inferred from them**. These measurements provide a basis for designing later tests of biological computation and object-specific states.

## I. What are the models actually comparing?

### 1. What does “dynamic” mean in our Dynamic model?

Dynamic's temporal kernels of 11, 5 and 5 frames give a **19-frame receptive field**, spanning `t−18` through `t` at the aligned output. Static processes frame `t` independently; a crop of its first 18 outputs aligns the prediction timestamps.

![Current data roles, model histories and common evaluation pathway](figures/documentation/01_shared_pathway.png)

*The current models share prediction targets and a fixed neural-defined measurement. Dynamic integrates video and appended behaviour history; pupil shifting is a separate current-time pathway. Static's leading crop supplies the aligned output window. The test movies were explored historically.*

The architecture integrates a finite input history. Each output depends directly on the finite input window; the architecture uses temporal convolution rather than a persistent recurrent or object-indexed state. Testing long-term tracking or object-update rules requires evidence about how information persists and how an estimated state changes.

See [models and data](MODELS_AND_DATA.md#4-models-and-training) and [comparison setup](03_model_comparison/SETUP.md).

### 2. Are we comparing the models' hidden states or their predicted neural responses?

Both encoders predict responses for the **same recorded neurons**. Their outputs and recorded responses enter the same fixed scale conversion and GPFA coordinate system. This supplies common coordinates for comparing temporal population-response patterns.

Hidden features would require a separate readout or alignment. The output comparison measures predicted population activity; intermediate network variables belong to a separate analysis. Similar response predictions may arise from different algorithms, so intermediate states and their role in computation need direct examination. Examining or manipulating those computations would connect the output comparison to mechanism.

See [frozen transforms](MODELS_AND_DATA.md#5-frozen-preprocessing-and-the-evaluation-pathway) and [trajectory measurement](02_trajectory_measurement/SETUP.md).

### 3. Is the Static–Dynamic comparison fair?

The models share data roles, neuron identities, scored frames and input variable types. Total parameters differ by **1.707%** (2,814,015 versus 2,862,063). The current Static checkpoint passed 15 single-frame perturbation checks; a deliberately coupled negative control failed.

Parameter matching controls total model size while retaining the computational differences between the two architectures. Readouts dominate the count, while cores contain **50,624 versus 98,672** parameters. Spatial processing, regularisation, initialisation and history access differ. Dynamic integrates appended behaviour as well as video; spatial damage also affects both input types. The comparison evaluates complete architectures, including differences in visual and behavioural history processing. The frame check verifies frame independence under the tested inference-mode perturbations.

See [implementation checks](01_model_validation/RESULTS.md) and [weight-control setup](06_temporal_weight_controls/SETUP.md). A follow-up comparison could isolate visual history, behaviour history and other architectural differences with separate controls.

### 4. Which data trained the models, and which data tested the final results?

The diagram separates **1,395 training**, **349 validation** and **293 test trials**. Train determines encoder parameters/scales and fits pilot GPFA; validation selects checkpoints, GPFA initialisation and control strengths. The labelled repeated-movie `oracle` tier supplies the current test; scoring uses observed responses rather than hidden-tier placeholders. These current settings and GPFA parameters remain fixed during test scoring.

This is **held-out evaluation after retraining** using previously explored test movies. The evaluation therefore describes transfer to this run’s held-out data within a history of earlier test-movie exploration. Approximate overlap screening found no verified overlap among 7,725 candidate pairs, but source-film identities remain unavailable.

Response scoring covers five sessions and 40,034 neurons. Trajectory and intervention scoring covers **one session, 512 neurons, six movies and 58 test trials**, with 282 train and 66 validation trials. The five sessions broaden response evaluation; trajectory evidence comes from the pilot, and session counts should be distinguished from animal counts. The current comparison uses one encoder seed.

See [data roles](MODELS_AND_DATA.md#3-training-validation-and-test-data) and [comparison scope](03_model_comparison/SETUP.md).

## II. Why use trajectories, and how reliable is the measurement?

### 5. Why measure both trajectory position and direction?

Position asks whether corresponding times occupy similar **GPFA states**; its coordinates describe neural activity rather than physical object location. Direction asks whether neighbouring inferred states change in similar directions. Paths can occupy similar regions in different orders, or change similarly while displaced.

The primary measures are position correlation and velocity-direction cosine. Position nRMSE adds error magnitude; speed correlation and acceleration-direction cosine describe first- and second-difference structure. These metrics describe the smoothed offline posterior. Visual object speed and neural latency would require corresponding task-coordinate and timing measurements. The full five-metric battery is retained, including unresolved speed differences. The records identify the primary metrics; preregistration before earlier oracle inspection remains unverified.

See [metric definitions](03_model_comparison/SETUP.md#5-metrics) and [complete comparisons](03_model_comparison/RESULTS.md).

### 6. Why use one shared GPFA space, and what might it miss?

GPFA models `y(t) = C z(t) + d + ε(t)` with linear loadings, neuron-specific observation noise and a smooth temporal prior. Training neural responses fit four latent dimensions; validation selects the **0.25-second initialisation**, followed by a train-only fit converging in 19 EM iterations. Both models use that same frozen measurement.

Freezing places both encoders in common neural-defined coordinates. The four dimensions summarize a component of the 512-neuron response, using observations throughout the window for posterior inference. **The posterior uses 63 observations to estimate states at 250 query times.** The inferred state is an offline description of shared population activity; relating its axes to biological variables requires additional evidence.

### 7. What do repeat reliability and null tests each tell us?

![Repeat agreement and limited response reconstruction](figures/documentation/02_measurement.png)

*Reliability ranges describe 200 repeat assignments of the same six movies. Reconstruction uses the full query grid with separate response-level denominators. The panels report complementary quantities on separate scales: trajectory repeatability and response-variance coverage.*

**Reliability** compares disjoint repeats of the same movie. Position and direction agreement improve markedly from one repeat to balanced four/five-repeat halves. **Nulls** test the dependence of agreement on movie identity, phase, order or population coordination. All five metrics outperform their prescribed nulls in every balanced partition: correlations and cosines are higher, and nRMSE is lower. Both primary metrics also outperform the nulls at one and two repeats. The reported one-repeat secondary comparisons include cases that fall below the required null-superiority criterion.

The frozen GPFA reconstruction accounts for **1.89% of centred single-trial variance and 2.56% of condition-average variance**. A small shared component can be repeatable while accounting for a limited fraction of population variation. Repeat partitions describe within-dataset measurement variability; balanced-repeat reliability characterizes the averaged responses and provides a reference for interpreting model scores. Scores are not reliability-normalised. Cross-dimension and observation-grid robustness remain unassessed after the auxiliary fits were withdrawn.

See [measurement setup](02_trajectory_measurement/SETUP.md) and [complete reliability, null and reconstruction results](02_trajectory_measurement/RESULTS.md).

## III. What do the model comparisons actually establish?

### 8. Where does Dynamic outperform Static, and where is the evidence weaker?

![Five-session responses and complete pilot trajectory comparisons](figures/documentation/03_model_comparison.png)

*Response points use all neurons in five sessions; trajectories use one 512-neuron pilot. The trajectory intervals concern pooled scores recomputed on six whole resampled movies. Per-movie panels show actual condition IDs and preserve the small movie_00005 direction exception.*

Dynamic improves response correlation in every session (unweighted means **0.1596 versus 0.1911**). In the pilot, position and direction differences favour Dynamic, as do position error and acceleration. Speed remains unresolved. Position improves in six movies and direction in five; the remaining movie is retained in the comparison.

Time-aware RSA/CKA also favour Dynamic. Prediction-null comparisons show structured correspondence in both models’ outputs, including Static. These results support better tested response prediction and selected trajectory summaries. **The supported gains concern the tested responses and selected trajectory properties; identifying their cause is a separate question**.

See [model comparison results](03_model_comparison/RESULTS.md).

### 9. What does the response-matching control add?

![Validation-selected scalar response matching and trajectory contrasts](figures/documentation/05_response_matching.png)

*The ±0.005 band marks the fixed matching tolerance; statistical equivalence was not tested. Trajectory intervals use fixed calibration and one trial-keyed noise realisation; speed remains unresolved.*

Training predictions determine a per-neuron noise scale; validation selects additive-noise amplitude **σ ≈ 0.869749**. Test applies the same noise rule and clipping without retuning. The scalar response-r gap stays within tolerance, yet both primary trajectory contrasts remain positive.

The control shows that similar **mean per-neuron correlation** can coexist with different trajectory scores. It matches that scalar alone: neuron-level accuracy, NMSE, time-aware RSA/CKA and error structure can still differ. The comparison tests a fixed output perturbation; a comparison of independently retrained accuracy-matched models would address a different question. Noise may affect trial-level correlation more than repeat-averaged, smoothed trajectories. Quantifying that contribution would require controls that separately examine averaging and smoothing.

See [response-matching setup](05_response_matching/SETUP.md) and [results](05_response_matching/RESULTS.md).

### 10. What happens when we reverse the predicted time sequence?

![Output reversal, pattern invariance and time-sensitive comparisons](figures/documentation/04_reversal.png)

*Reversal occurs in encoder-target scale before the time-dependent GPFA conversion. Its operation differs from the after-conversion primary null. Pattern invariance is checked deterministically; trajectory intervals concern pooled scores, and conventional intervals concern equal-weight movie means.*

Changing **A → B → C** to **C → B → A** preserves the time average. The tested movie-average RSA/CKA therefore remain unchanged, while both primary trajectory measures decline. Response correlation and several time-aware RSA/CKA summaries also decline; Static temporal-difference CKA is unresolved.

The result identifies **which temporal relationships each aggregation preserves**. Time-aware RSA/CKA retain temporal sensitivity alongside GPFA. Response accuracy also changes under reversal. Because scale conversion depends on time, `G(RP)` and `R(GP)` are distinct response-level operations. Both retain the input videos and infer new trajectories on the original query-time grid.

See [temporal test setup](04_temporal_metric_tests/SETUP.md) and [results](04_temporal_metric_tests/RESULTS.md).

### 11. Do trajectory metrics reveal new information, or describe the same responses differently?

Trajectories transform the same neuron-by-time responses into a description of population states and their temporal organization. Their value is **relative to particular summaries**: mean correlation compresses errors to one number, time averaging discards order, and GPFA measures shared-state geometry after projection and smoothing.

Response matching and reversal show that different summaries can conceal or expose different properties. Time-aware RSA/CKA also detect differences, so the value established here is complementary description rather than independence from those metrics. A useful trajectory metric combines repeatability, sensitivity to specified perturbations and an interpretation tied to the research question.

See [comparison](03_model_comparison/RESULTS.md), [reversal](04_temporal_metric_tests/RESULTS.md) and [response matching](05_response_matching/RESULTS.md).

## IV. What can the interventions tell us about computation?

### 12. What do temporal-weight and spatial-weight changes tell us?

Independent encoder training compares complete learned systems. Weight attenuation measures the sensitivity of **one frozen model to specified weight changes**, including changes to feature scales and its learned operating regime.

![Current temporal attenuation and validation-selected spatial controls](figures/04_model_comparison/retrained_v1_diagnostics/encoder/weight_attenuation.png)

*The x axis is temporal retention; spatial controls appear at their associated temporal target, not their own retention. Black crosses mark test mismatch; red outlines identify the correlation-degenerate spatial-zero endpoint. The endpoint’s finite trajectory scores are displayed separately from its unevaluable response match.*

Both primary trajectory scores fall as off-centre temporal weights are removed. Only two nontrivial spatial/temporal pairs retain test response matching. At temporal **0.75**, spatial damage preserves higher position and direction. At **0.25**, spatial damage has lower position but higher direction. The **0.50** pair exceeds the test tolerance; the valid pairs show metric-dependent effects of the specified temporal and spatial interventions.

Two endpoint distinctions remain essential. Temporal zero retains video/behaviour at **t−9** and pupil shifting at **t**, giving a centre-only Dynamic intervention distinct from Static. Spatial zero produces 76 exact constants and 436 float32-floor outputs, so its correlation is **NA / numerically_degenerate** and matching is **not_evaluable**. These interventions establish metric-dependent sensitivity to specified weights. Isolating visual history or linking this dependence to a biological mechanism would require targeted follow-up comparisons.

See [weight-control setup](06_temporal_weight_controls/SETUP.md) and [results](06_temporal_weight_controls/RESULTS.md).

### 13. Can the current results tell us how the brain computes over time?

The evidence has three levels:

| Level | Supported result | Boundary |
|---|---|---|
| Prediction | Reduced Dynamic predicts the tested responses better. | The source of the gain requires a separate comparison. |
| Population representation | Its outputs better match selected properties in one fixed four-dimensional neural measurement. | Reconstruction defines the fraction of population variation covered by this measurement. |
| Within-model dependence | Specified temporal-filter interventions change the trained model's predictions. | The interpretation includes architecture, behaviour-history and general damage effects. |

A **biological computation** claim needs an additional discriminating test. For example, object-specific selective updating would require manipulating object assignment while controlling relevant motion or expectations, then comparing algorithms that predict different neural or behavioural outcomes. That experiment would add an object-assignment manipulation and a measurement of object-specific state updates. A theory of those updates would specify the state being maintained and the rule that changes it.

The study establishes repeatable time-resolved measurements and comparative results: selected scalar or averaged summaries omit differences captured by trajectory measures, and the tested Dynamic model better matches several of those properties. They can help define predictions for a future test that distinguishes competing mechanisms.

See [comparison scope](03_model_comparison/RESULTS.md), [measurement limits](02_trajectory_measurement/RESULTS.md) and [weight controls](06_temporal_weight_controls/RESULTS.md).
