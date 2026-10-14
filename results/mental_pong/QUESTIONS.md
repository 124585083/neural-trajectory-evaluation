# Mental-Pong: Research Questions and Decisions

This document explains the reasoning behind the Mental-Pong comparisons and how their outcomes guide interpretation. We compare neural readouts of the **objective ball trajectory** with a **behavior-constrained candidate trajectory** constructed from condition-mean paddle endpoints. The candidate turns a behavioral hypothesis into a defined reconstruction target; the animal's internal estimate is a quantity to be inferred.

The neural input consists of processed **condition means** from macaque dorsomedial frontal cortex (DMFC). The current analysis uses FA50 and GPFA50 representations, ordinary least-squares (OLS) position readouts, 100 fixed condition splits, and two kinds of randomized controls. The findings describe condition-level reconstruction under this representation and readout protocol. The [original study](https://doi.org/10.1038/s41467-024-54688-y) provides the broader mental-simulation context; the present analysis tests a specific condition-mean reconstruction hypothesis.

The answers connect **what was calculated**, **what it shows**, and **which alternative explanations remain**. They explain documented research decisions and distinguish completed analyses from proposed tests.

## I. What did we set out to test?

### 1. What does “predicting the ball” mean in this analysis?

The term *prediction* covers several questions with different targets and data requirements.

- **Position reconstruction:** Can a readout recover the ball's position at time `t` from neural population activity indexed to time `t`? This is the main completed task-coordinate analysis.
- **Hidden-period tracking:** Can position still be reconstructed at time `t` when the ball is no longer visible? The completed analysis tests this on its hidden-period samples. The computation producing that position information is a separate question.
- **Future-state prediction:** Can neural activity available at `t` predict the ball's position at `t + Δ`, over and above what is already determined by the physical trajectory? This would require a dedicated forward-horizon test beyond the completed reconstruction analysis.
- **Future behavioral prediction:** Can neural activity before the relevant decision or feedback predict an individual trial's later paddle endpoint? A direct test requires individual neural–behavior trial pairs, whereas the current release supplies condition means.

The behavior-constrained target is built **retrospectively from the recorded mean endpoint**. Its reconstruction measures a relationship between neural activity and a label containing endpoint information. An online forecasting claim would additionally require evidence about when that information became available to the animal. An earlier response prefix specifies the neural inputs available to the estimator; interpreting its output as an internal forecast also requires evidence about forecast content and timing.

For this project, the precise terms are *position reconstruction*, *hidden-period reconstruction*, and *behavior-constrained trajectory comparison*. *Future prediction* and *internal simulation* refer to additional hypotheses that require their own tests.

See [task-motion setup](01_task_motion/SETUP.md), [behavior-relation setup](02_behavior_relation/SETUP.md), and [models and data](MODELS_AND_DATA.md).

### 2. Why compare the real ball path with a path based on the paddle endpoint?

The starting question was whether population activity corresponds more closely to the ball's **objective physical path** or to a path consistent with where the animal eventually positioned its paddle. If the animal's behavior reflects an erroneous estimate of the ball, a behavioral candidate might depart from the objective path in an informative way. The comparison supplies a concrete alternative target for evaluating hidden-period position readouts.

The candidate has a fixed construction. It retains the **objective horizontal (`x`) motion and arrival time**. Its vertical coordinate (`y`) is determined by the known condition geometry and the condition-mean endpoint:

```text
candidate_y(condition, time)
    = offset(condition, time)
    + alpha(condition, time) * mean_paddle_endpoint(condition)
```

Fixed task geometry maps the mean endpoint to a single time-aligned path. Because this map is affine, averaging the individual endpoint-generated curves agrees with generating a curve from the mean endpoint, up to floating-point precision. The saved geometry checks cover 6,084 endpoint/condition assignments per animal.

The construction yields **two clearly specified, time-aligned reconstruction targets**. Their shared `x` concentrates the behavioral comparison on vertical position and endpoint error. Geometry partly determines the candidate's temporal profile, and its terminal endpoint may reflect motor control or later task information. Interpreting that path as an internal estimate requires a verified pre-feedback endpoint time, which the current records do not provide.

The comparison favors the objective target: across the four animal-by-representation cases, the candidate's full-interval own-target RMSE is **0.0220–0.0322 higher**. This weakens the predicted reconstruction advantage of this candidate.

See [behavior setup](02_behavior_relation/SETUP.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 3. What kind of generalization did the 100 condition splits test?

In each saved split, the representation and OLS readout are fitted on training conditions and evaluated on held-out conditions. The OLS design assigns 39 nominal training and 40 nominal test conditions from the same 79-condition set; saved masks determine the geometry-valid scoring rows. The same fixed splits are used when comparing FA50 with GPFA50 and the objective with behavioral readouts. This tests transfer between released condition-mean patterns **within the same task and condition family**.

Condition holdout evaluates whether a learned position mapping transfers beyond its fitted response patterns and labels. The test conditions still share the task, physical rules, behavioral summary level and recorded populations. Testing transfer to a new task, animal, independently recorded neural trial or preprocessing procedure would require an additional dataset or analysis.

The **100 condition splits provide 100 resamplings of largely the same 79 conditions**, measuring partition sensitivity within the existing data. They show how the comparison changes with the assignment of conditions to training and testing. Scores across splits share the underlying condition means and can be statistically dependent. Mahler and Perle provide two animal datasets, each examined with FA50 and GPFA50.

Saved fitting and scaling identities document train/test separation in the current analysis. Upstream processing of the released arrays, described below, may already have coupled input information across conditions and times. That inherited dependence remains after condition-held-out OLS fitting.

The analysis tests **held-out-condition task-position readout within the released condition-mean dataset**. Paired individual trials and additional animals would extend the generalization test to those levels.

See [models and data](MODELS_AND_DATA.md), [task-motion setup](01_task_motion/SETUP.md), and [code and reproduction scope](../../code/mental_pong/README.md).

### 4. Why use FA or GPFA before fitting a linear position readout?

FA50 and GPFA50 summarize high-dimensional condition-mean responses in a lower-dimensional population space. **Factor analysis (FA)** expresses shared response covariance through latent factors. **Gaussian-process factor analysis (GPFA)** additionally imposes a smooth temporal prior on the latent sequence. The current comparison uses 50-dimensional versions of each. Neural responses alone fit both transformations; the subsequent OLS readout learns the task-coordinate mapping.

Representation fitting and task readout serve different purposes. The **latent coordinates** describe population activity; mapping their axes to horizontal position, vertical position or behavior requires an explicit task readout. The **OLS readout** asks whether a *single linear function of those coordinates*, fitted on training conditions, can recover the designated task target on held-out conditions. A fixed readout across visible and hidden samples is also more constrained than training a separate decoder for every time point or phase.

Separating the two stages lets neural responses define the representation and task labels define the readout. It also lets the objective and behavioral hypotheses be evaluated using the same dimensionality and OLS fitting protocol. Representation and readout choices shape the resulting measurement. FA and GPFA retain different components, and GPFA smooths the time course. Recovering other information may require nonlinear decoding or a different representation. Low scores indicate *limited recoverability through the tested representation and readout*; information in the full neural population remains an empirical question.

Mental-Pong evaluates **decoded ball positions in task coordinates** against objective or behavior-derived labels. Sensorium instead compares trajectories in a shared neural-defined latent space. The task-coordinate decoder measures recoverable position information; identifying latent geometry or a state-update algorithm requires additional analyses.

See [task-motion setup](01_task_motion/SETUP.md), [behavior setup](02_behavior_relation/SETUP.md), and [models and data](MODELS_AND_DATA.md).

## II. What can the neural measurements tell us?

### 5. What do whole-path and phase-by-phase scores tell us that one correlation misses?

Full-interval position correlation measures covariation between a decoded coordinate and its reference across the evaluated samples. It combines observations from different task periods. A decoder might follow the visible ball well and fail under occlusion. It might perform differently before and after a wall collision. High correlation may also coexist with a systematic vertical offset or a large endpoint error. **Correlation measures covariation, while RMSE measures coordinate error**, so the two scores provide complementary information.

The saved analysis therefore examines the reconstructed time course on **common valid condition/time support**, alongside task stages such as visible, hidden, collision-related and post-collision periods. It also retains splitwise, conditionwise and time-resolved results. The phase-wise results reveal changes in ordering across periods and metrics. Objective own-target reconstruction is better in all four full-interval animal-by-representation comparisons, with different orderings in some task phases. In **no-collision windows**, the candidate has lower own-target RMSE alongside lower correlation; **post-collision** own-target RMSE favors the objective path. The randomized-endpoint analysis also has a post-collision metric disagreement: its four recorded **RMSE** empirical tails are **0.643, 0.406, 0.534 and 0.489**, whereas the corresponding **correlation** tails are `0/1000`. The post-collision comparison therefore has a metric-dependent outcome.

The reconstructed path locates differences over task time. Phase-specific correlation, coordinate RMSE and endpoint error measure **task-position reconstruction** through a fitted linear mapping to spatial labels. State-transition analyses, perturbation-driven updating tests and comparisons of recurrent computations would address additional questions about **neural dynamics**. GPFA's smoothness prior can also influence apparent path continuity.

Phase and condition analyses reveal **time-, condition- and metric-dependent readout behavior** that a single summary obscures. These analyses locate reconstruction differences that can guide later tests of how a hidden object state is maintained.

See [task-motion results](01_task_motion/RESULTS.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 6. Does the GPFA analysis use future neural responses?

The observations used during **inference**, as well as the fitted parameters, determine temporal availability. A posterior state at 200 ms that incorporates responses recorded at 300 ms describes the trajectory offline. Interpreting it as available online at 200 ms would require restricting the observation window even with frozen GPFA weights.

The Mental-Pong implementation infers each GPFA state from the **released-response prefix** up to that time index. Saved tests compare prefix execution with explicit-prefix execution and perturb later released samples to check that earlier inferred states are unaffected. Within the supplied processed arrays, those checks pass. The FA50/GPFA50 fits use training neural responses and are tied to their training-side scales and split identities. Together, these checks verify the estimator's prefix-input rule.

The released arrays were processed before this inference step, so the audit separates implementation checks from upstream input provenance:

```text
provided_input_filtering = pass
fit_provenance         = pass
raw_preprocessing      = fail
```

The upstream missing-condition filling used information from all conditions and times. The values supplied to prefix inference already contain that upstream processing. In addition, the precise alignment of the retained behavioral endpoint to pre-feedback events remains unresolved. Interpreting a bin as biologically available online therefore depends on both input provenance and endpoint timing.

The checks support **prefix-only GPFA inference on the provided input**. Verifying temporal availability from raw spikes would require the missing upstream input records. This computational property characterizes the estimator: whether neurons themselves perform filtering or maintain and update the estimated state is a separate biological hypothesis.

See [models and data](MODELS_AND_DATA.md), [task-motion setup](01_task_motion/SETUP.md), and [reproduction limits](../../code/mental_pong/README.md).

### 7. What does hidden-ball decoding tell us about internal simulation?

Hidden-period position decoding is consistent with an internally maintained estimate during occlusion. Decoding establishes a **statistical relationship** between activity and position. Its computational origin requires comparison with alternative sources of the same task structure.

Several signals may carry similar task structure. Initial trajectory, physical rules and elapsed time may jointly predict objective position through a code other than an explicit sequence of simulated positions. Neurons may reflect time since motion onset, trial condition, intended interception location, paddle preparation or a mixture of these variables. If these quantities covary across the available conditions, a linear decoder can recover position from any sufficiently correlated code. These candidate explanations covary in the current condition-mean analysis.

The measured result illustrates this distinction. Full-interval objective `y` correlations range from **0.4928 to 0.7204** across the saved animal/representation cases, showing substantial linear decodability. The objective readouts' full-interval **geometry-baseline skills are all negative** (from **−1.7543 to −0.8923**), favoring the designated task-geometry reference under the specified skill definition. Position decodability and accuracy beyond task geometry answer different questions.

The comparison evaluates a limited-capacity neural readout against a strong geometry baseline; condition averaging may also affect the distinctions available to either analysis. To separate internal simulation from correlated time, stimulus and action variables, the next test needs competing explanations that predict **different neural or behavioral outcomes**. That test should distinguish the competing explanations, rather than add another correlated target.

See [task-motion setup](01_task_motion/SETUP.md) and [task-motion results](01_task_motion/RESULTS.md).

## III. Does the decoded trajectory follow behavior?

### 8. Why train two decoders and evaluate all four decoder–target combinations?

A lower objective-path error can reflect target difficulty, the fitted mapping or both. Two heads and four scores separate these possibilities:

![The candidate construction, two-by-two scoring and distinct randomized controls](figures/documentation/methods_targets_controls.png)

*The sketches show how the reconstruction targets are constructed. OO/BB are own-target scores; OO/BO and OB/BB compare heads against the same reference. B fixes the head; C refits it for each endpoint assignment.*

**OO versus BB** tests candidate reconstruction while changing both the fitted head and the target's variance and difficulty. The two **shared-target** comparisons hold the reference fixed, separating a mapping change from a scoring-label change. These scores describe target recoverability under a specified readout protocol.

In the retained full-interval results, **objective own-target reconstruction is better in all four cases**. Against the *same behavioral candidate*, the objective-trained head has higher full-interval correlation and lower RMSE than the behavior-trained head. This weakens the hypothesis that the candidate-trained mapping has a reconstruction advantage in these population means. The inference concerns the candidate construction, condition means and linear mapping; other behavioral information remains a separate question.

See [behavior setup](02_behavior_relation/SETUP.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 9. What is the difference between decoding position, beating random candidates and beating a geometry baseline?

Each reference defines a different scientific comparison.

**Decodable position:** Positive held-out correlation identifies target information recoverable by the trained linear map. Behavioral specificity and improvement over known condition variables are additional comparisons.

**Better than randomized behavioral candidates:** An advantage for the actual endpoint-based target identifies structure in the real condition-to-endpoint pairing. That structure includes correspondence with objective geometry and shared timing. Randomization preserves each destination's geometry and timing while changing the correspondence between physical condition and endpoint.

**Better than a geometry-only baseline:** This tests **accuracy relative to an explicit task-informed predictor using no neural features**, under the stated unequal input sets. The full-interval geometry skills are negative in all four objective cases (**−1.7543 to −0.8923**) and all four candidate cases (**−2.2594 to −1.1445**). The geometric predictor has lower full-interval squared error under the specified metric.

These results are compatible: a readout can correlate with the actual target and distinguish it from arbitrary labels while still being worse than an informed no-neural predictor. Explaining **trial-specific behavioral error** requires an additional observation level. That stronger claim requires information about the particular behavior associated with the neural response after matching or controlling the physical condition.

See [task-motion results](01_task_motion/RESULTS.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 10. Why are there two different randomized controls?

**B changes test correspondence while fixing the head.** It measures condition correspondence, including **physical correspondence**, because both change under the remapping.

**C changes the endpoint target and refits the head.** Geometry and time remain fixed. Refitting evaluates how well the same learning procedure recovers each assignment. Each control uses 1,000 allocations of the same dataset under fixed splits to form an algorithmic randomization reference.

The actual candidate outperforms the randomized-endpoint reference over the **full, hidden and endpoint-influence** ranges. Post-collision correlation and RMSE give different results, as shown by the tails in Question 5. These outcomes show that actual endpoint assignments matter relative to the selected randomized alternatives. The result concerns condition-to-endpoint structure in a geometry-constrained candidate; geometry-baseline performance and within-condition behavioral specificity are separate comparisons.

A useful question for every null is: **what relationship did we destroy, and what did we preserve?** A discriminating null should preserve the competing explanation while disrupting the hypothesized relationship.

See [behavior setup](02_behavior_relation/SETUP.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 11. How does averaging across trials limit what we can say about behavior?

The release supplies one population time course per condition; its membership cannot be linked to individual paddle endpoints. Each behavioral label would therefore be attached to the same condition-mean neural observation.

![Behavioral endpoint errors shrink when opposite signs are averaged within conditions](figures/closeout_v1/descriptive/endpoint_mean_cancellation.png)

*The scatter compares mean absolute individual error with absolute condition-mean error. Points below the identity line quantify cancellation in behavioral errors.*

Both error signs occur in **77/78 Mahler** and **78/78 Perle** valid conditions; median cancellation is **54.0% and 48.0%**. Current means use **7,407 and 84,873 behavioral records**, with different membership and weighting from older summaries.

The same neural condition mean could arise from identical responses on opposite-error trials or from opposite neural deviations that cancel. These possibilities are both compatible with the behavioral cancellation: `E[R | condition]` does not determine `E[R | condition, behavior group]`. The identity `E[candidate_y | condition] = offset + alpha * E[endpoint | condition]` verifies the affine averaging rule; interpreting neural computation requires additional evidence.

This is an **identifiability limit** requiring paired trials or correctly paired group means under the same physical condition. Resolving the pairing requires new trial-linked observations, rather than further transformations of the existing means.

See [models and data](MODELS_AND_DATA.md), [behavior results](02_behavior_relation/RESULTS.md), and [next steps](NEXT_STEPS.md).

## IV. How should the results change our interpretation?

### 12. Which hypothesis does the result weaken?

The most direct hypothesis was **operational**: given the available condition-mean DMFC representations, the specified FA50/GPFA50 plus OLS procedure would reconstruct a path constrained by the mean behavioral endpoint better than the objective physical path. The full-interval own-target comparisons favor the objective path. The objective target performs better across all four animal-by-representation cases on the full and hidden own-target comparisons, and the fixed-target cross-scoring also limits the candidate advantage.

The result lowers confidence in candidate advantage under **this particular target, measurement and readout combination**. Control C still identifies nonrandom structure in the candidate's endpoint assignments. Assessing within-condition behavioral specificity requires paired neural/behavior trials beyond these processed means, constrained targets and linear readouts. The **mental-simulation hypothesis as a general theory** also extends beyond this test: the original paper examined additional population analyses and recurrent-network comparisons.

Possible explanations concern the neural signal, the observations, the candidate definition and the readout. The behavior-dependent variable may be weak or absent in this population, averaging may reduce its expression, the endpoint or imposed time course may poorly represent the relevant belief, or the linear readout may have limited sensitivity to it. These are **competing explanations for future tests**.

The tested neural means support task-position reconstruction, stronger objective-path reconstruction in the direct comparison, and better candidate reconstruction than specified randomized labels. Whether the decoded hidden-ball path follows individual behavioral estimates requires a trial-linked test.

See [task-motion results](01_task_motion/RESULTS.md), [behavior results](02_behavior_relation/RESULTS.md), and [next steps](NEXT_STEPS.md).

### 13. What did this project establish, and what evidence would be needed next?

The completed project has three concrete outcomes.

The **task-coordinate analysis** combines FA50/GPFA50 representations, held-out-condition OLS readouts, the full `2 × 2` decoder–target matrix, phase-aware scores and two randomized controls. Saved source identities, masks and fit checks specify each comparison. Prefix GPFA checks support *provided-input* inference; the retained `raw_preprocessing=fail` status records the unresolved upstream dependence. Changing the target tests how reconstruction depends on the task reference; identifying a neural computation requires a separate comparison.

The **specific negative result** is that objective paths have stronger main own-target reconstruction. The candidate retains nonrandom endpoint-assignment structure, and the designated geometry baseline has lower full-path error than the neural readouts. The results support hidden-period decoding within this protocol; behavioral specificity and online internal simulation require further tests. Phase exceptions and undefined-support cases specify where each comparison applies.

The next research decision follows from **the missing observation**: whether neural activity differs between behavioral choices *for the same physical condition*, before the relevant feedback. Answering it would require known neural–behavior trial pairing and appropriate temporal alignment. Paired data would allow behavioral grouping within a fixed objective trajectory and decoder evaluation on disjoint trials or conditions. Geometry and motor/time covariates would provide competing references. These requirements define the proposed follow-up study. A positive result would support behavior-related representation; distinguishing its computational origin would require competing update mechanisms with different predictions.

The research decision depends on distinguishing **representation, behavioral correspondence and computation as levels of evidence**. The current variables, controls and metrics establish recoverable task structure and define the paired observation needed next. More discriminating experiments would then address the broader internal-simulation theory.

See [task-motion results](01_task_motion/RESULTS.md), [behavior results](02_behavior_relation/RESULTS.md), [next steps](NEXT_STEPS.md), and [code reproduction scope](../../code/mental_pong/README.md).

## Study source and evidence scope

The task and the broader mental-simulation argument originate from Rajalingham, Sohn and Jazayeri (2025), [*Dynamic tracking of objects in the macaque dorsomedial frontal cortex*](https://doi.org/10.1038/s41467-024-54688-y), *Nature Communications* 16:346. Its data are released at [Zenodo, record 13952210](https://doi.org/10.5281/zenodo.13952210). The numerical findings quoted here come from **this project's saved analysis records**. Exact scoring protocols and detailed tables belong in [task-motion setup/results](01_task_motion/SETUP.md) and [behavior-relation setup/results](02_behavior_relation/SETUP.md). Source-identity and reproducibility limits are documented in [models and data](MODELS_AND_DATA.md) and the [Mental-Pong code README](../../code/mental_pong/README.md).
