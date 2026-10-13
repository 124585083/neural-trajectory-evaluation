# Mental-Pong: Research Questions and Decisions

This document explains the reasoning behind the Mental-Pong comparisons and how their outcomes guide interpretation. We compare neural readouts of the **objective ball trajectory** with a **behavior-constrained candidate trajectory** constructed from condition-mean paddle endpoints. The candidate makes a behavioral hypothesis testable; the animal's internal estimate remains unobserved.

The available neural data are processed **condition means** from macaque dorsomedial frontal cortex (DMFC), rather than neural responses paired with each behavioral trial. The current analysis uses FA50 and GPFA50 representations, ordinary least-squares (OLS) position readouts, 100 fixed condition splits, and two kinds of randomized controls. Its conclusions apply to this observation level and these definitions. The analysis does not establish a trial-specific mental simulation or refute the broader evidence discussed in the [original study](https://doi.org/10.1038/s41467-024-54688-y).

The answers connect **what was calculated**, **what it shows**, and **which alternative explanations remain**. They explain documented research decisions and distinguish completed analyses from proposed tests.

## I. What did we set out to test?

### 1. What does “predicting the ball” mean in this analysis?

Several different questions are often described as *prediction*, but they require different targets and data.

- **Position reconstruction:** Can a readout recover the ball's position at time `t` from neural population activity indexed to time `t`? This is the main completed task-coordinate analysis.
- **Hidden-period tracking:** Can position still be reconstructed at time `t` when the ball is no longer visible? This is tested through the hidden-period portion of the same reconstruction analysis. It does not, by itself, reveal how that information is computed.
- **Future-state prediction:** Can neural activity available at `t` predict the ball's position at `t + Δ`, over and above what is already determined by the physical trajectory? The current study does not provide a dedicated forward-horizon test of this claim.
- **Future behavioral prediction:** Can neural activity before the relevant decision or feedback predict an individual trial's later paddle endpoint? The available condition-mean neural data are not paired with those individual behavioral outcomes, so this question is not tested directly.

The behavior-constrained target is built **retrospectively from the recorded mean endpoint**. Its reconstruction measures a relationship between neural activity and a label containing endpoint information. An online forecasting claim would additionally require evidence about when that information became available to the animal. Restricting neural input to an earlier response prefix addresses inference timing, but a later-outcome label alone leaves the content and timing of an internal forecast unresolved.

For this project, the precise terms are *position reconstruction*, *hidden-period reconstruction*, and *behavior-constrained trajectory comparison*. We should reserve *future prediction* or *internal simulation* for evidence that tests those stronger claims.

See [task-motion setup](01_task_motion/SETUP.md), [behavior-relation setup](02_behavior_relation/SETUP.md), and [models and data](MODELS_AND_DATA.md).

### 2. Why compare the real ball path with a path based on the paddle endpoint?

The starting question was whether population activity corresponds more closely to the ball's **objective physical path** or to a path consistent with where the animal eventually positioned its paddle. If the animal's behavior reflects an erroneous estimate of the ball, a behavioral candidate might depart from the objective path in an informative way. Comparing the two gives the analysis a concrete alternative target instead of calling every decodable hidden-period position an internal simulation.

The current candidate is deliberately constrained. It retains the **objective horizontal (`x`) motion and arrival time**. Its vertical coordinate (`y`) is determined by the known condition geometry and the condition-mean endpoint:

```text
candidate_y(condition, time)
    = offset(condition, time)
    + alpha(condition, time) * mean_paddle_endpoint(condition)
```

The recorded mean endpoint is mapped through the same geometry rather than selecting a different curve freely at every time point. Because this map is affine, averaging the individual endpoint-generated curves agrees with generating a curve from the mean endpoint, up to floating-point precision. The saved geometry checks cover 6,084 endpoint/condition assignments per animal.

The construction yields **two clearly specified, time-aligned reconstruction targets**. Their shared `x` concentrates the behavioral comparison on vertical position and endpoint error. Geometry partly determines the candidate's temporal profile, and its terminal endpoint may reflect motor control or later task information. The endpoint's exact pre-feedback timing is unavailable, leaving the proposed internal path an interpretation to test.

The comparison favors the objective target: across the four animal-by-representation cases, the candidate's full-interval own-target RMSE is **0.0220–0.0322 higher**. This result weakens the proposed advantage of the specified candidate.

See [behavior setup](02_behavior_relation/SETUP.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 3. What kind of generalization did the 100 condition splits test?

Each saved split trains the representation and OLS readout on designated **training conditions** and evaluates the readout on held-out **conditions**. The recorded OLS design uses 39 training and 40 test conditions drawn from the same set of 79 task conditions, with geometry-valid support determined by the saved masks. The same fixed splits are used when comparing FA50 with GPFA50 and the objective with behavioral readouts. This asks whether a relationship learned from some of the released condition-mean response patterns transfers to other patterns **within the same task and condition family**.

Condition holdout evaluates whether a learned position mapping transfers beyond its fitted response patterns and labels. The test conditions still share the task, physical rules, behavioral summary level and recorded populations. Transfer to a new task, unseen animals, independently acquired neural trials or a different preprocessing process remains untested.

The **100 splits are 100 resamplings of largely the same 79 conditions**, not 100 independent biological experiments. They show how strongly the measured advantage depends on which conditions are assigned to training or testing. Scores across splits share the underlying condition means and can be statistically dependent. Likewise, Mahler and Perle provide two animal datasets; FA50 and GPFA50 applied to the same animal are two analyses of that dataset, not two additional animals.

Saved fitting and scaling identities document train/test separation in the current analysis. Upstream processing of the released arrays, described below, may already have coupled input information across conditions and times. That inherited dependence remains after condition-held-out OLS fitting.

The analysis tests **held-out-condition task-position readout within the released condition-mean dataset**. Single-trial behavioral generalization and animal-population replication require different observations.

See [models and data](MODELS_AND_DATA.md), [task-motion setup](01_task_motion/SETUP.md), and [code and reproduction scope](../../code/mental_pong/README.md).

### 4. Why use FA or GPFA before fitting a linear position readout?

FA50 and GPFA50 give the high-dimensional condition-mean neural response a lower-dimensional population description. **Factor analysis (FA)** expresses shared response covariance through latent factors. **Gaussian-process factor analysis (GPFA)** additionally imposes a smooth temporal prior on the latent sequence. The current comparison uses 50-dimensional versions of each. Both transformations are fitted without behavioral trajectory labels; the subsequent OLS readout supplies the task-coordinate mapping.

These are two distinct operations. The **latent coordinates** summarize population activity, but their axes do not inherently mean horizontal position, vertical position, a behavioral prediction or any other task variable. The **OLS readout** asks whether a *single linear function of those coordinates*, fitted on training conditions, can recover the designated task target on held-out conditions. A fixed readout across visible and hidden samples is also more constrained than training a separate decoder for every time point or phase.

Separating representation and readout helps keep a task label from defining the neural representation itself. It also lets the objective and behavioral hypotheses be evaluated using the same dimensionality and OLS fitting protocol. This does **not** make the outcome independent of analytic choices. FA and GPFA retain different components, GPFA changes the time course through smoothing, and OLS may miss information that needs nonlinear decoding or a different representation. Low performance therefore means *limited recoverability through the tested representation and readout*, not proven absence of the variable from the original neural population.

Mental-Pong evaluates **decoded ball positions in task coordinates** against objective or behavior-derived labels. Sensorium instead compares trajectories in a shared neural-defined latent space. The task-coordinate decoder measures recoverable position information; identifying latent geometry or a state-update algorithm requires additional analyses.

See [task-motion setup](01_task_motion/SETUP.md), [behavior setup](02_behavior_relation/SETUP.md), and [models and data](MODELS_AND_DATA.md).

## II. What can the neural measurements tell us?

### 5. What do whole-path and phase-by-phase scores tell us that one correlation misses?

A full-interval position correlation says whether a decoded coordinate tends to rise and fall with the reference coordinate across the evaluated samples. It is useful, but it combines many questions. A decoder might follow the visible ball well and fail under occlusion. It might perform differently before and after a wall collision. Or a high correlation could coexist with a systematic vertical offset or a large endpoint error. **Correlation and RMSE do not penalize the same failures**, so they must be read together.

The saved analysis therefore examines the reconstructed time course on **common valid condition/time support**, alongside task stages such as visible, hidden, collision-related and post-collision periods. It also retains splitwise, conditionwise and time-resolved results. This is consequential because the observed ordering is not uniform across every phase and metric. The full-interval objective target wins the own-target reconstruction comparisons in all four animal-by-representation cases, but particular stages have different rankings. In **no-collision windows**, the candidate has lower own-target RMSE alongside lower correlation; **post-collision** own-target RMSE favors the objective path. The randomized-endpoint analysis also has a post-collision metric disagreement: its four recorded **RMSE** empirical tails are **0.643, 0.406, 0.534 and 0.489**, whereas the corresponding **correlation** tails are `0/1000`. These values cannot be collapsed into a uniform post-collision advantage.

The path shows *where* reconstruction succeeds or fails. Phase-specific correlation, coordinate RMSE and endpoint error measure **task-position reconstruction** through a fitted linear mapping to spatial labels. A test of **neural dynamics** would instead need to assess state transitions, perturbation-driven updating or competing recurrent computations. GPFA's smoothness prior can also influence apparent path continuity.

Phase and condition analyses reveal **time-, condition- and metric-dependent readout behavior** that a single summary obscures. They locate reconstruction differences for later computational tests; the mechanism sustaining a hidden object state remains unresolved.

See [task-motion results](01_task_motion/RESULTS.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 6. Does the GPFA analysis use future neural responses?

The observations used during **inference**, as well as the fitted parameters, determine temporal availability. A posterior state at 200 ms that incorporates responses recorded at 300 ms describes the trajectory offline. Interpreting it as available online at 200 ms would require restricting the observation window even with frozen GPFA weights.

The Mental-Pong implementation addresses this particular risk by inferring each GPFA state from the available **released-response prefix** up to its time index. Saved tests compare prefix execution with explicit-prefix execution and perturb later released samples to check that earlier inferred states are unaffected. Within the supplied processed arrays, those checks pass. The saved fits are tied to their training-side scaling and per-split identities; behavioral target labels are not used to train FA50/GPFA50. These are meaningful controls against a future-reading *inference implementation*.

The released arrays were processed before this inference step, so the audit separates implementation checks from upstream input provenance:

```text
provided_input_filtering = pass
fit_provenance         = pass
raw_preprocessing      = fail
```

The upstream missing-condition filling used information from all conditions and times. Prefix inference cannot reverse an operation already embedded in each released response value. In addition, the precise alignment of the retained behavioral endpoint to pre-feedback events remains unresolved. Both limits matter for claims that the representation at a given bin reflects only information biologically available at that moment.

The checks support **prefix-only GPFA inference on the provided input**. A future-free pipeline from raw spikes remains unverified because input provenance is unresolved. This computational property characterizes the estimator: whether neurons themselves perform filtering or maintain and update the estimated state is a separate biological hypothesis.

See [models and data](MODELS_AND_DATA.md), [task-motion setup](01_task_motion/SETUP.md), and [reproduction limits](../../code/mental_pong/README.md).

### 7. If we can decode the hidden ball, why can't we conclude that the brain is simulating it?

Hidden-period position decoding is consistent with an internally maintained estimate, because contemporaneous visual tracking alone is insufficient during occlusion. Decoding establishes a **statistical relationship** between activity and position. Its computational origin requires comparison with alternative sources of the same task structure.

Several signals may carry similar task structure. Given the initial trajectory, the physical rules and elapsed time, the expected objective position may be largely predictable even without representing each intermediate simulated position explicitly. Neurons may reflect time since motion onset, trial condition, intended interception location, paddle preparation or a mixture of these variables. If these quantities covary across the available conditions, a linear decoder can recover position from any sufficiently correlated code. The present condition-mean analysis does not independently vary all of those alternatives.

The measured result illustrates this distinction. Full-interval objective `y` correlations range from **0.4928 to 0.7204** across the saved animal/representation cases, showing substantial linear decodability. Yet the objective readouts' full-interval **geometry-baseline skills are all negative** (from **−1.7543 to −0.8923**): the designated reference based on known task geometry performs better under that benchmark's skill definition. A high position correlation therefore does not automatically mean the neural readout adds predictive value beyond what task structure already supplies.

The geometry comparison leaves ball-state representation in DMFC possible: the baseline is strong, the decoder has limited capacity, and condition averaging can obscure distinctions available to a richer analysis. To separate internal simulation from correlated time, stimulus and action variables, the next test needs competing explanations that predict **different neural or behavioral outcomes**. Reconstructing another correlated coordinate would leave that mechanism question open.

See [task-motion setup](01_task_motion/SETUP.md) and [task-motion results](01_task_motion/RESULTS.md).

## III. Does the decoded trajectory follow behavior?

### 8. Why train two decoders and evaluate all four decoder–target combinations?

A lower objective-path error can reflect target difficulty, the fitted mapping or both. Two heads and four scores separate these possibilities:

![The candidate construction, two-by-two scoring and distinct randomized controls](figures/documentation/methods_targets_controls.png)

*The trajectory sketches illustrate target construction, not measured internal estimates. OO/BB are own-target scores; OO/BO and OB/BB compare heads against the same reference. B fixes the head; C refits it for each endpoint assignment.*

**OO versus BB** tests the candidate's own-target hypothesis, but changes target variance and difficulty as well as the fitted head. The two **shared-target** comparisons hold the reference fixed, separating a mapping change from a scoring-label change. Neither comparison is a direct measure of neural preference.

In the retained full-interval results, **objective own-target reconstruction is better in all four cases**. Moreover, when both heads are judged against the *same behavioral candidate*, the objective-trained head has better full-interval correlation and RMSE than the behavior-trained head. Those findings constrain the claim that the available population means preferentially support the candidate. They do not imply that neural activity categorically lacks behavioral information; the candidate construction, condition means and linear mapping are part of the tested hypothesis.

See [behavior setup](02_behavior_relation/SETUP.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 9. What is the difference between decoding position, beating random candidates and beating a geometry baseline?

These comparisons answer different questions, and the strongest conclusion depends on **which reference was beaten**.

**Decodable position:** Positive held-out correlation identifies target information recoverable by the trained linear map. Behavioral specificity and improvement over known condition variables are additional comparisons.

**Better than randomized behavioral candidates:** An advantage for the actual endpoint-based target identifies structure in the real condition-to-endpoint pairing. That structure includes correspondence with objective geometry and shared timing. The randomized assignments therefore test arbitrary endpoint pairing while leaving physical covariates incompletely matched.

**Better than a geometry-only baseline:** This is a stricter test of **incremental predictive value relative to an explicit task reference without the neural features**. The full-interval geometry skills are negative in all four objective cases (**−1.7543 to −0.8923**) and all four candidate cases (**−2.2594 to −1.1445**). Under the specified metric and baseline, the neural readout has not shown an advantage over that geometric predictor.

These results are compatible: a readout can correlate with the actual target and distinguish it from arbitrary labels while still being worse than an informed no-neural predictor. None of the three outcomes alone demonstrates a neural explanation of **trial-specific behavioral error**. That stronger claim requires information about the particular behavior associated with the neural response after matching or controlling the physical condition.

See [task-motion results](01_task_motion/RESULTS.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 10. Why are there two different randomized controls?

**B changes test correspondence while fixing the head.** It tests alignment with the intended condition, but also breaks **physical correspondence**, so a positive effect is not behavior-specific evidence.

**C changes the endpoint target and refits the head.** Geometry and time remain fixed. Refitting evaluates how well the same learning procedure recovers each assignment. Each control uses 1,000 allocations of the same dataset under fixed splits, describing an algorithmic randomization reference rather than independent neural experiments.

The actual candidate outperforms the randomized-endpoint reference over the **full, hidden and endpoint-influence** ranges. The post-collision evidence is metric dependent; in particular, the RMSE tails listed in Question 5 do not provide uniform support there. These outcomes show that actual endpoint assignments matter relative to the selected randomized alternatives. They do not isolate the behavioral component from objective task geometry, demonstrate a uniquely represented internal trajectory, or override the negative geometry-baseline skills.

A useful question for every null is: **what relationship did we destroy, and what did we preserve?** Only a null that preserves the competing explanation while disrupting the hypothesized relationship can distinguish those explanations cleanly.

See [behavior setup](02_behavior_relation/SETUP.md) and [behavior results](02_behavior_relation/RESULTS.md).

### 11. How does averaging across trials limit what we can say about behavior?

The neural release supplies one population time course per condition, with no recoverable pairing to individual paddle endpoints. A condition-mean neural curve therefore cannot be treated as a new neural trial for each behavioral outcome.

![Behavioral endpoint errors shrink when opposite signs are averaged within conditions](figures/closeout_v1/descriptive/endpoint_mean_cancellation.png)

*The scatter compares mean absolute individual error with absolute condition-mean error. Points below the identity line show behavioral cancellation, not measured loss of neural information.*

Both error signs occur in **77/78 Mahler** and **78/78 Perle** valid conditions; median cancellation is **54.0% and 48.0%**. Current means use **7,407 and 84,873 behavioral records**, with different membership and weighting from older summaries.

The same neural condition mean could arise from identical responses on opposite-error trials or from opposite neural deviations that cancel. Behavioral cancellation cannot distinguish those possibilities: `E[R | condition]` does not determine `E[R | condition, behavior group]`. Likewise, `E[candidate_y | condition] = offset + alpha * E[endpoint | condition]` validates the affine averaging rule, not a neural mechanism.

This is an **identifiability limit** requiring paired trials or correctly paired group means under the same physical condition. Relabeling the released means, changing decoders or adding random splits cannot supply the missing pairing.

See [models and data](MODELS_AND_DATA.md), [behavior results](02_behavior_relation/RESULTS.md), and [next steps](NEXT_STEPS.md).

## IV. How should the results change our interpretation?

### 12. Which hypothesis did the negative result actually weaken?

The most direct hypothesis was **operational**: given the available condition-mean DMFC representations, the specified FA50/GPFA50 plus OLS procedure would reconstruct a path constrained by the mean behavioral endpoint better than the objective physical path. The full-interval own-target comparisons do **not** support that prediction. The objective target performs better across all four animal-by-representation cases on the full and hidden own-target comparisons, and the fixed-target cross-scoring also limits the candidate advantage.

The result lowers confidence in candidate advantage under **this particular target, measurement and readout combination**. Control C still identifies nonrandom structure in the candidate's endpoint assignments. Behavior-related information in DMFC remains unresolved because the analysis uses processed means, a constrained path and linear readouts without paired neural/behavior trials. The **mental-simulation hypothesis as a general theory** also extends beyond this test: the original paper examined additional population analyses and recurrent-network comparisons.

A negative result has several possible sources, and they should remain separate. The underlying representation might not carry the proposed behavior-dependent variable; the available condition means might not preserve it; the terminal endpoint might be a poor proxy for the relevant belief; the imposed candidate time course might be inaccurate; or the tested linear readout might not recover the relevant structure. These are **competing explanations**, not findings that the study has distinguished.

We can still say that the tested neural means carry recoverable task-position structure, that objective reconstruction is stronger under the reported direct comparison, and that the candidate is more informative than some randomized labels. The narrower evidence does not warrant the stronger claim that the decoded hidden-ball path tracks each animal's behavioral estimate.

See [task-motion results](01_task_motion/RESULTS.md), [behavior results](02_behavior_relation/RESULTS.md), and [next steps](NEXT_STEPS.md).

### 13. What did this project establish, and what evidence would be needed next?

The completed project has three concrete outcomes.

The **auditable task-coordinate comparison** combines current FA50/GPFA50 representations, held-out-condition OLS readouts, the full `2 × 2` decoder–target matrix, phase-aware scores and two distinct randomized controls. Saved source identities, masks and fit checks specify each comparison. Prefix GPFA checks support *provided-input* inference; the retained `raw_preprocessing=fail` status records the unresolved upstream dependence. Changing a target in this comparison changes the measurement, not the demonstrated neural computation.

The **specific negative result** is that objective paths have stronger main own-target reconstruction. The candidate retains nonrandom endpoint-assignment structure, and the designated geometry baseline has lower full-path error than the neural readouts. These findings preserve hidden-period position decoding within the protocol while leaving behavioral specificity and online internal simulation unresolved. Phase exceptions and numerical failures define the reach of that conclusion.

The next research decision follows from **the missing observation**: whether neural activity differs between behavioral choices *for the same physical condition*, before the relevant feedback. Answering it would require known neural–behavior trial pairing and appropriate temporal alignment. One could then compare outcomes grouped by behavior while holding the objective trajectory fixed, fit and test decoders on disjoint trials or conditions, and ask whether the behavior-related signal improves on physical geometry and motor/time covariates. These are proposed requirements for an identifiable test, **not completed analyses**. Even positive evidence at that stage would show behavior-related representation before proving the algorithm that generated it; a computation claim would still need contrasts between competing update mechanisms.

The research decision depends on distinguishing **representation, behavioral correspondence and computation as levels of evidence**. The current variables, controls and metrics establish recoverable task structure and define the paired observation needed next. They leave the broader internal-simulation theory open for more discriminating tests.

See [task-motion results](01_task_motion/RESULTS.md), [behavior results](02_behavior_relation/RESULTS.md), [next steps](NEXT_STEPS.md), and [code reproduction scope](../../code/mental_pong/README.md).

## Study source and evidence scope

The task and the broader mental-simulation argument originate from Rajalingham, Sohn and Jazayeri (2025), [*Dynamic tracking of objects in the macaque dorsomedial frontal cortex*](https://doi.org/10.1038/s41467-024-54688-y), *Nature Communications* 16:346. Its data are released at [Zenodo, record 13952210](https://doi.org/10.5281/zenodo.13952210). The numerical findings quoted here are from **this project's saved analysis briefs**, not from the original paper. Exact scoring protocols and detailed tables belong in [task-motion setup/results](01_task_motion/SETUP.md) and [behavior-relation setup/results](02_behavior_relation/SETUP.md). Source-identity and reproducibility limits are documented in [models and data](MODELS_AND_DATA.md) and the [Mental-Pong code README](../../code/mental_pong/README.md).
