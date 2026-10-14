# Next Steps: What the Mental-Pong Result Changes

**Status:** Research interpretation and proposed next study. The current Mental-Pong analysis is complete (`CLOSED_EXPLORATORY_WITH_LIMITATIONS`). **The neural–behavior grouping and temporal tests below form a proposed follow-up study.**

## 1. The question, expectation, and actual outcome

The completed study asked whether linear readouts of condition-mean dorsomedial frontal cortex (DMFC) activity reconstruct a **behavior-constrained candidate ball path** more accurately than the **objective physical path**. The motivation was that an animal's eventual paddle position might reflect an estimate that differs from the physical path. If that estimate were represented in neural activity, a suitable behavioral candidate could be a more informative reconstruction target.

The candidate used a fixed interpolation rule. For each physical condition, it used the *mean terminal paddle position* as its endpoint and connected that endpoint to the known physical geometry with fixed interpolation. The objective and candidate targets were reconstructed from the **same 50-dimensional FA or GPFA features** using separately trained linear readouts. The analysis included own-target and 2×2 cross-scoring (A), fixed-head condition mismatch (B), randomized endpoint assignments with refitted readouts (C), and a task-geometry baseline. See the [behavior-relation setup](02_behavior_relation/SETUP.md) for the exact definitions.

The principal observations are:

| Completed observation | Result | Interpretation |
|---|---|---|
| Objective position reconstruction | Full-interval vertical-position `r = 0.4928–0.7204` across the two animals and FA50/GPFA50 | Task-related position information is linearly recoverable in the tested representations. |
| Objective versus candidate, each using its own head and target | Full `Δr = −0.0248` to `−0.0108`; `ΔRMSE = −0.0322` to `−0.0220`, with positive differences defined to favor the candidate | **The objective path has better full-path own-target reconstruction.** The hidden-period comparisons also favor the objective target. |
| Fixed-head condition mismatch (B) | Correct condition correspondence performs better on matched support | Neural predictions retain condition-related structure, including the physical geometry and timing relationships changed by the mismatch. |
| Randomized endpoints with readout refitting (C) | Actual candidates outperform the specified random assignments on full, hidden, and endpoint-influence support | The real endpoint assignment contains structure relative to that null. The direct objective–candidate comparison is supplied by A. |
| C after collision | Random-at-least-as-good RMSE fractions are `0.643`, `0.406`, `0.534`, `0.489` across the four animal/representation cases | The candidate's advantage over randomized endpoints **depends on the phase and metric**. |
| Task-informed geometry baseline | All four full-interval candidate neural skills are negative (`−2.2594` to `−1.1445`) | This **unmatched-input**, geometry-rich baseline has lower whole-path squared error than the tested neural readouts. |

These comparisons evaluate target reconstruction and the structure retained under each control. The 100 condition splits provide 100 resamplings of the same observations and quantify partition sensitivity. The complete values and split dispersions are in [task-motion results](01_task_motion/RESULTS.md) and [behavior-relation results](02_behavior_relation/RESULTS.md).

**Current conclusion:** The released condition-mean DMFC representation supports physical task-position reconstruction and condition-related endpoint structure. Objective trajectories are reconstructed more accurately than the *specified mean-endpoint candidate* over the full path. This weakens the candidate-advantage hypothesis and prioritizes paired observations of behavior-related neural states, followed by tests of their computational origin.

## 2. What could explain the weaker candidate reconstruction?

The next study needs to distinguish the explanations below. Their evidential status ranges from measured target properties to untested neural hypotheses.

| Possible explanation | Evidence from the completed study | Evidence status and next test |
|---|---|---|
| **Similarity between the mean candidate and objective path may reduce sensitivity** | Opposite signed behavioral endpoint deviations often cancel; the current candidate uses the mean endpoint and inherits the physical path's geometry and timing | Target similarity and behavioral cancellation were measured; their effect on the sensitivity of this comparison remains to be tested. |
| **Averaging may attenuate behavior-related neural differences** | The neural release contains means per physical condition, with no recoverable assignment of neural samples to the corresponding behavioral trials | A plausible neural hypothesis to test with paired neural and behavioral observations; the current analysis measures no neural information loss. |
| **The imposed candidate path may differ from the animal's internal estimate** | The vertical path is constructed from the terminal paddle endpoint and fixed physical anchors | A construct-validity limitation; requires time-resolved behavioral or neural constraints capable of discriminating alternative internal paths. |
| **The task structure explains much of the decodable position** | Objective reconstruction and condition correspondence are present, yet the geometry-only comparator has lower full-path squared error | Task structure is an alternative explanation for high position `r`; the geometry baseline receives more physical information than the neural readout and has unmatched capacity. |
| **The selected representation or linear readout may omit useful information** | FA50/GPFA50 and fixed OLS recover a task-coordinate projection, with predictions compressed relative to the targets | A possible analysis limitation whose effect on recoverable information in the full neural response remains to be measured. |
| **The neural–behavior correspondence is unverified at the trial level** | Behavioral membership is recorded, while the trials contributing to each released neural pseudopopulation mean are unknown | Verified trial membership is needed to quantify any mismatch and its effect. |
| **Upstream filling or terminal-feedback timing may affect the comparison** | Released missing neural responses were filled with global condition/time information; strictly pre-feedback timing of the terminal paddle scalar is unresolved | `raw_preprocessing=fail` records the provenance issue; estimating its impact and any feedback contribution requires additional source observations. |
| **Behavioral dependence may be weak or absent in this population and task** | The current observation level supports condition-mean reconstruction; within-condition specificity remains unmeasured | This remains a viable explanation. A reliable paired-trial test would distinguish it from measurement and data limitations. |

The own-target result makes paired neural–behavior data the next priority, ahead of changing the decoder or candidate. We need evidence that separates *absence of a behavior-related neural effect* from *inability of the current data to measure that effect*. The next analysis should distinguish these explanations regardless of which outcome it favors.

## 3. How condition averaging could weaken this comparison

### 3.1 What is observed in behavior

Let `c` index a fixed physical ball trajectory and let `b_{c,i}` be the terminal paddle position on behavioral trial `i`. Define the signed endpoint deviation relative to the physical interception point as

\[
e_{c,i}=b_{c,i}-y_{\mathrm{obj},c}(t_{\mathrm{arrival}}).
\]

The current candidate uses `\bar b_c`, the mean endpoint of its retained behavioral records. With the fixed physical anchor and interpolation coefficient, its post-anchor vertical coordinate is

\[
y_{\mathrm{cand},c}(t)=[1-\alpha_c(t)]\,y_{\mathrm{anchor},c}
                         +\alpha_c(t)\,\bar b_c.
\]

Because this mapping is affine in `b`, averaging trial-constructed candidates is equivalent to constructing the candidate from the mean endpoint. It retains the condition-level endpoint while discarding trial membership in the above-target and below-target groups. As the signed deviations cancel, their mean can approach the objective endpoint; where the physical branches coincide, the candidate and objective trajectories can become harder to distinguish.

![Signed endpoint errors and their cancellation in behavioral condition means](figures/closeout_v1/descriptive/endpoint_mean_cancellation.png)

*Each point in the right panels is one valid physical condition. Below the identity line, the absolute mean error is smaller than the mean absolute individual error. This is a behavioral observation; the corresponding neural trial membership is unavailable.*

The quantities below distinguish behavioral cancellation from separation of the constructed targets:

| Quantity | Mahler | Perle |
|---|---:|---:|
| Valid physical conditions containing both signs of endpoint error | 77/78 | 78/78 |
| Condition-equal mean absolute trial error | 1.7272 | 1.3388 |
| Mean absolute condition-mean error | 0.9784 | 0.7908 |
| Median within-condition cancellation fraction | 0.5398 | 0.4804 |
| Full-path candidate–objective RMS separation | 0.6760 | 0.5612 |

The first four rows quantify averaging of **behavioral** deviations. The last measures separation of the **constructed, averaged target paths** on their retained support. These are behavioral and target-level measurements; neural effects require paired observations.

### 3.2 Why these facts could matter for neural reconstruction

The figure establishes opposing **behavioral** errors and their reduction by averaging. Fixed geometry then determines the separation of the two averaged target paths. A single neural mean per condition/time pools the opposite behavioral groups, leaving their group-specific responses unidentified. Paired observations are needed to test neural cancellation and whether target similarity reduces sensitivity enough to explain the result.

Target-path separation and decoder RMSE use different pooling rules; they describe signal separation and reconstruction error, while statistical power requires a separate analysis. The earlier repeated-neural-mean analysis changed both labels and condition weights. It compared target and weighting choices using the same neural means, leaving trial pairing and neural information loss unmeasured.

A direct test of the averaging hypothesis requires **neural activity from known trials partitioned by the behavior of those same trials**, allowing a comparison such as

\[
\mathbb{E}[R(t)\mid c,\,e>\delta]\quad\text{versus}\quad
\mathbb{E}[R(t)\mid c,\,e<-\delta].
\]

Here `\delta` denotes a meaningful signed-error tolerance to be selected before the proposed analysis; its value is still unspecified. The coordinate convention must first be checked so that positive/negative error corresponds correctly to above/below target in the display. Estimating these group-conditioned responses requires the corresponding trial-level neural records.

## 4. Data required for the next test

The immediate follow-up requires linked **neural and behavioral records from the same trials**. Each observation should include animal, session and stable trial identity; physical condition and ball trajectory; neural units and time bins; and terminal paddle movement. Event records should give the times of visibility, occlusion, collision, interception, motor response and feedback. The endpoint used to define a prospective estimate must be verified as available **before feedback**. Movement or joystick measurements, where available, should be retained as potential explanatory covariates.

The recordings must include enough trials of **both error signs within the same physical condition and recording context** to form reliable group estimates. Report conditions with inadequate paired observations as unsupported and retain their coverage records. Group membership must refer to linked neural and behavioral trials. Units assembled across sessions can support group-conditioned pseudopopulation means, which should be reported separately from simultaneously recorded single-trial population trajectories.

Data processing must preserve trial identity and distinguish observed data from imputed values. Training observations should determine neuron eligibility, normalization, latent fitting and behavioral readouts; held-out observations supply evaluation. To examine *when* a signal emerges, infer each Gaussian-process state from the response prefix available at that time and verify the temporal provenance of the input. Temporal availability depends on both the observations entering inference and the upstream preprocessing, alongside the frozen parameters.

**Decision gate:** Paired, time-resolved inputs are the prerequisite for the behavior-specific neural test. Until those inputs are available, retain the completed exploratory result and document the outstanding data requirement.

## 5. Proposed study: compare above-target and below-target behavior within the same physical condition

The **proposed protocol** below translates those data requirements into a follow-up analysis.

![Proposed paired-trial comparison and schematic persistent versus event-related differences](figures/documentation/methods_paired_trial_proposal.png)

*The diagram illustrates the proposed neural-group comparison. Dashed curves show two compatible qualitative patterns; numerical scales and uncertainty estimates are intentionally unspecified. Positive/negative endpoint error must be mapped to above/below only after checking the display-coordinate convention.*

### Step 1 — Define the behavioral groups before examining neural results

Predefine **above-target**, **below-target** and, where supported, **near-target** groups using measurement precision and task criteria. Keep continuous signed error as a complementary target. Set thresholds from measurement precision and task criteria before examining neural results. Match or account for error magnitude, trial counts, session, neuron availability and timing; report unsupported conditions. These groups describe **action outcomes** that may reflect an estimate, a goal, motor control or feedback; the underlying perceptual belief is an interpretation to test.

### Step 2 — Establish a measurable neural contrast

Partition **paired trials** within condition and session, then estimate preprocessing and fit a common representation and readouts on training observations. Held-out-trial and leave-whole-condition-out tests address different generalization questions and must be reported separately. Estimate group means on disjoint trial halves, test their reproducibility, and compare separation with within-condition label shuffles; report coverage and failures.

The proposed primary readout would use **pre-feedback neural activity to predict the signed behavioral residual relative to the objective endpoint** on held-out paired trials. Compare with condition/time-only and movement-informed references, and retain physical-position decoding as a task reference. Held-out endpoints supply evaluation targets only. **Reliable group separation precedes interpretation of trajectory shape.**

### Step 3 — Ask whether the neural difference is persistent or changes at a task event

Within the same condition, construct group trajectories in one **training-defined, fixed neural coordinate system**. Let `z_up(c,t)` and `z_down(c,t)` denote group-averaged latent states. Their separation can be described by

\[
\Delta z_c(t)=z_{\mathrm{up},c}(t)-z_{\mathrm{down},c}(t).
\]

A nonzero `||Δz||` quantifies latent separation; a signed ball-position interpretation requires a task-coordinate readout. A fixed readout trained on independent trials must test the expected direction of the above-minus-below projection. The schematic distinguishes **persistent offsets** from **event-related changes**, which may coexist. Persistent differences can reflect maintained state or motor preparation; event-related changes can reflect updating, attention, cues, timing or motor adjustment.

Test change **relative to a pre-event baseline**, with aligned support and null controls. Interpret post-event decoding together with the pre-event comparison. GPFA smoothing, measurement reliability and collision/feedback timing affect the estimated transition. Temporal ordering should be interpreted where these factors are sufficiently resolved.

### Step 4 — Test competing explanations with targeted controls

Shuffle **behavioral-group labels within the same physical condition**, preserving session and group sizes where possible. This preserves the objective path within each condition; the current B control instead remaps labels across conditions. Repeat with count-balanced groups and report residual movement/session imbalance. Use geometry/time and movement references, reliability estimates, and a predeclared temporal disruption to examine the corresponding alternative explanations.

Select any representation dimension or temporal settings inside training/validation data. Reserve the final group results for evaluating the selected analysis. After establishing reliable behavioral correspondence, assess alternative candidate curves or more flexible readouts on fresh held-out data. These comparisons still require verified trial pairing and input provenance.

## 6. Predictions, possible results, and decision rules

The following **predictions for future paired data** define how each possible outcome would guide the research decision. They are planned tests for a future dataset.

| Future observation | Interpretation and research decision |
|---|---|
| **Reliable group separation and a correctly signed pre-feedback neural readout**, beyond within-condition shuffle and geometry-only references | Supports a behavior-related population representation at the tested observation level. Internal physical simulation and prospective belief would require additional tests of represented content and timing. |
| **Group differences are persistent through occlusion without a clear event-linked change** | Supports a maintained behavior-associated state. Compare the persistent signal with motor preparation and other sustained covariates; selective updating requires an event-related contrast. |
| **A reproducible event-linked change occurs before feedback**, with the expected signed behavioral correspondence and suitable timing/movement controls | Consistent with state updating and motivates a more discriminating test of competing computations. Identifying the update algorithm and its neural causal role requires a discriminating follow-up experiment. |
| **Reliable neural group estimates but no held-out behavioral direction relationship**, under a sensitive predefined test | Weakens the specific behavior-linked representation hypothesis for this population, task and metric. Retain averaging and decoder choice as alternative explanations only where separate evidence supports them. |
| **Apparent effects arise only after feedback or disappear after movement/session adjustment** | Favors feedback-related or motor/contextual explanations over a pre-feedback estimate, within the measured controls. |
| **Groups are too small, trial identity is uncertain, or group neural reliability is inadequate** | The data are insufficient for the proposed group comparison. Retain the existing objective-position result and classify the behavioral comparison as unresolved because of inadequate measurement. |

Reliable signed-error prediction or group separation is a **representation-level observation**: neural activity covaries with behavior beyond the fixed physical condition. A computation-level claim would then require mechanisms that predict **different responses to a controlled manipulation**. Changing an expectation while holding sensory history as constant as possible is one possible later design. Such a manipulation could distinguish computations that produce similar reconstructions of the same mean path. This manipulation is a later design possibility.

## 7. Research decision and priority

The final FA50/GPFA50–OLS analysis recovers task-position structure, with **objective reconstruction more accurate than the specified behavior-constrained candidate over the full path**. Fixed-head mismatches test condition correspondence, and endpoint randomization tests the structure of the condition-to-endpoint assignment. The geometry comparator gives lower full-path error; identifying an internal behavioral estimate requires evidence beyond this task-informed reference. The next priority is therefore a paired observation that holds physical condition fixed.

The next work should proceed in this order:

1. **Data feasibility:** acquire or identify genuinely paired neural–behavior trials, raw-input provenance, precise event times and adequate within-condition behavioral variability.
2. **Measurement validity:** establish reliable above/below group neural differences on independent trial partitions, with within-condition nulls and support checks.
3. **Behavioral correspondence:** test held-out signed-error decoding relative to physical geometry and motor covariates.
4. **Temporal signatures:** distinguish sustained offsets from event-related changes, while respecting pre-feedback timing and latent-inference limits.
5. **Mechanistic discrimination, only if warranted:** design an independent manipulation under which persistent-state, updating and motor-related explanations make different predictions.

If step 1 remains infeasible, retain the completed negative result and specify the paired observations needed to proceed. If steps 1–3 succeed and step 4 is inconclusive, report behavior-associated neural representation and retain state updating as an open question.

## 8. Summary for the project overview

> The completed Mental-Pong analysis used condition-mean DMFC activity to reconstruct objective ball position more accurately than the mean-endpoint behavior-constrained path over the full interval. Observed endpoint assignments supported better reconstruction than specified randomized assignments in several phases, while a task-informed geometry baseline gave lower full-interval squared error. Opposing endpoint errors partly canceled in condition averages; measuring any corresponding neural information loss requires trial-level neural–behavior membership that is absent from the current release. The proposed next step compares reliable neural states from above-target and below-target trials *within the same physical condition*, first testing behavioral correspondence and then examining whether the difference persists or changes around task events.

## 9. Supporting records and boundaries

- [Goal 1 — Task-motion setup](01_task_motion/SETUP.md) and [results](01_task_motion/RESULTS.md): objective-position recovery, condition mismatch and physical-geometry reference.
- [Goal 2 — Behavior-relation setup](02_behavior_relation/SETUP.md) and [results](02_behavior_relation/RESULTS.md): own-target/cross-scoring, randomized endpoint controls, phase exceptions, geometry skill and behavioral cancellation.
- [Models and Data](MODELS_AND_DATA.md): original condition means, behavioral membership, representation/OLS identities and unresolved upstream input filling.
- [Research Questions](QUESTIONS.md): reconstruction versus prediction, trial generalization, causal prefix inference and the limit of internal-simulation claims.
- Original evidence records: `evidence/closeout_v1/A/self_reconstruction_main_table.csv`, `evidence/closeout_v1/A/cross_2x2_summary.csv`, `evidence/closeout_v1/C/random_endpoint_summary.csv`, `evidence/closeout_v1/descriptive/mean_cancellation_summary.json`, and `evidence/closeout_v1/A/previous_trial_version_comparison.csv`. Their exact filters and interpretations are listed in the goal Results documents.

**Data scope:** The current observations are published condition means. Their global condition/time filling remains unresolved (`raw_preprocessing=fail`), and original neural–behavior trial pairing is unavailable. Grouping, signed-error decoding, event-locked tests and mechanistic manipulation remain proposed analyses. Their expected outcomes are qualitative decision rules, with numerical effect sizes left unspecified.
