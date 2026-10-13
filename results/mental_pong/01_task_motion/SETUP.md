# Recovering Task-Motion Information: Setup

## 1. Question and purpose

**Can condition-averaged neural activity from macaque dorsomedial frontal cortex (DMFC) recover the ball's physical position during both visible motion and occlusion?** This is the first question in the Mental-Pong study. Before comparing a readout with the animal's behavioral endpoint, we need to establish what task-related position information can be recovered from the available neural representation.

The primary measurement fits a linear readout, **`D_obj`**, from a 50-dimensional neural representation to the **objective, physically defined ball position**. Its predictions are scored on held-out physical conditions. The same fitted head is examined separately during visible motion, occlusion, collision and post-collision periods; it is not refitted for each phase.

Two controls assess the source of reconstruction accuracy. A **fixed-head condition mismatch** tests the intended condition-to-position correspondence. A **task-geometry baseline** tests whether the neural readout improves on position estimates from specified physical anchors and timing information. Together they distinguish condition alignment from added predictive value over geometry; identifying the underlying neural computation requires further evidence. The later [behavior-relation analysis](../02_behavior_relation/SETUP.md) uses the same representations to compare objective and endpoint-constrained targets.

## 2. Data and evaluation support

### Neural observations

The neural input comprises the released *condition means*, not individual neural trials. The source fields `neural_responses_sh1.occ` and `neural_responses_sh2.occ` have axes **neuron × 79 physical conditions × 100 time bins**. Each bin spans 50 ms. Global neuron identifiers align the two released halves, yielding **518 units for Mahler** and **2,058 for Perle**. The fitted representations and final OLS analysis use **half1**; the saved half2 latent arrays do not constitute an independent behavioral result.

The `occ` suffix names the *occlusion task category*. The terms *visible* and *hidden* refer to time ranges within that task. Reported neural values are processed means of spike counts per original 1 ms sample, averaged over 50 ms. Train-derived standardization sets the scale for representation fitting. The full source and preprocessing contracts are in [Models and Data](../MODELS_AND_DATA.md).

### Ball coordinates, timing and usable rows

The ball travels toward a paddle plane at **x = 10** in centered MWorks display coordinates. It is visible initially, then occluded; some physical conditions include a boundary collision before the ball reaches the paddle. The objective XY path comes from the recorded physical-trajectory geometry and event metadata, not from the animal's terminal paddle response.

The current analysis evaluates each neural bin at its completed-bin availability time, `(k + 1) × 50 ms`, for zero-based bin `k`. The objective-geometry reconstruction uses original bin centers, `50k + 24.5 ms`, for fitting, then evaluates the physical branches at completed-bin times. These choices specify label–response alignment; exact neural processing latency remains unresolved.

| Scoring range | Valid conditions | Unique condition × time rows |
|---|---:|---:|
| Full active interval | 78 | 3,369 |
| Visible | 78 | 1,931 |
| Hidden | 78 | 1,360 |
| Collision conditions (`bounce`) | 26 | 1,207 |
| No-collision conditions (`no_bounce`) | 52 | 2,162 |
| Post-collision | 26 | 631 |

The **78 full-interval rows outside the visible and hidden masks** are boundary/mixed bins, so the latter two categories do not partition all full support. The collision/no-collision labels group physical conditions independently of the visibility masks. Masks remove bins that overlap feedback or extend beyond estimated arrival.

Condition **`59920`** has unresolved terminal/collision geometry. It remains in the nominal 79-condition split identities but contributes no scored label rows. Unsupported times are left missing; no synthetic position labels are inserted. The two animals and both representations use the same physical-condition labels and scoring masks. The saved result names are `full`, `visible`, `hidden`, `bounce`, `no_bounce` and `post_bounce`.

## 3. Models, fits and train/test separation

![Condition-held-out neural fitting and test-only position scoring](../figures/documentation/methods_condition_workflow.png)

*Goal 1 evaluates the objective head within this shared protocol. Goal 2 additionally compares the candidate head and all four head/reference combinations. GPFA prefix inference is restricted to the released input, whose upstream global filling remains unresolved.*

### Representation and readout

For each animal and each of **100 fixed condition splits**, factor analysis (**FA50**) and Gaussian-process factor analysis (**GPFA50**) are fitted from half1 **training-condition neural responses only**. Both produce a 50-dimensional input for a separate ordinary least-squares (OLS) position readout.

| Component | Saved method and settings | Interpretation |
|---|---|---|
| FA50 | `sklearn.decomposition.FactorAnalysis`, 50 components; seed 42; `max_iter=300`, `tol=0.01`, LAPACK, no rotation | Framewise latent features from the fitted observation model |
| GPFA50 | `SharedTimescaleGPFA`; `y(t) = C z(t) + d + ε(t)`; diagonal observation noise and one shared RBF timescale | Smooth neural latent estimates; each emitted state uses the completed **released-input prefix** only |
| GPFA fitting | FA50 initialization with 1% loading jitter; initial timescale 0.15 s, bounds 0.025–2 s; nugget 0.001; minimum noise 0.0001; at most 400 iterations, tolerance `1e-6` | Fixed per-split fit, without behavioral labels |
| Objective OLS head | `LinearRegression(fit_intercept=True, positive=False)`; 50 features, one intercept per coordinate; float64 | Fixed mapping from neural features to task XY position |

OLS has **51 parameters per output coordinate**, or 102 parameters for an XY head. There is no OLS regularization, extra feature history, behavior input, explicit time covariate, sample weighting or readout hyperparameter search. A latent input can nevertheless reflect condition and elapsed-time structure indirectly. GPFA's prefix restriction rules out deliberate use of future *provided bins* during posterior inference; it cannot undo time- or condition-wide filling already applied by the dataset provider.

The current fitted objects are different from the released authors' FA factors, which were fitted across all conditions before readout cross-validation. Historical GPFA64 and Ridge tests also follow a different protocol and are not substituted for this study.

### Fixed condition splits and preprocessing

The retained split protocol is:

```python
GroupShuffleSplit(n_splits=100, train_size=0.5, random_state=0)
# Nominally 39 train conditions and 40 test conditions per split
```

The exact same physical-condition assignments are reused for Mahler, Perle, FA50, GPFA50 and both downstream position targets. Because condition `59920` has no valid rows, the contributing conditions are either **38 train / 40 test** or **39 train / 39 test**, depending on which side contains it. The missing condition is never replaced.

Each split uses **half1 training-condition active rows** to select neurons (finite fraction at least 0.99, training SD above `1e-5`) and compute each neuron's mean and SD. The transformation `(response − training_mean) / training_SD` is frozen for test conditions. Remaining nonfinite inputs are set to zero after that conversion. Recorded checks also require at least 0.95 observed retained-neuron coverage per measurement bin in each stored half. All 518/2,058 aligned neurons passed the recorded fit-eligibility checks.

Training-condition rows fit the FA/GPFA representation and the objective OLS head; held-out condition rows are used for scoring. The OLS fit pools **all valid training condition × time rows** across the active period, so longer conditions contribute more observations. There is no separate validation split and no phase-specific head adjustment. The 100 test partitions overlap and therefore are not 100 independent neural experiments.

## 4. How the objective position is reconstructed

After neuron alignment, the frozen half1 latent array has shape `[79,100,50]` for each split. The objective head is fitted on valid full-interval training rows and applied only to test rows; its predictions are reused across phases. Scores pool eligible condition × time rows **within each split**, then summarize the **100 separately scored splits**.

For the physical labels, `fit_condition_geometry` fits horizontal position against original bin centers and piecewise straight `y(x)` branches from audited uncontaminated samples, then evaluates completed-bin times. Arrival is solved at **x = 10**. Two post-collision observations permit separate branch fits; with one, the recorded elastic-reflection constraint is used. Unsupported event histories remain unscored.

These **task-coordinate readouts** map latent activity to physical position. Phase-wise r and RMSE quantify reconstruction within specified time ranges; local direction, future prediction and neural state updating require separate tests. Both targets share x and arrival time; the behavioral comparison belongs to [Goal 2](../02_behavior_relation/SETUP.md).

## 5. Metrics, correspondence control and geometric reference

### Readout scores and uncertainty

For each held-out split and epoch, Pearson `r` compares the pooled predicted and objective samples **within one coordinate**. RMSE is `sqrt(mean((prediction − target)^2))` in centered display-coordinate units. Higher `r` and lower RMSE indicate better reconstruction of the specified reference. Correlations with constant values or fewer than three valid rows remain `NA`.

Reported `mean ± SD` values use the 100 split-level scores and **population SD (`ddof=0`)**. These SDs describe sensitivity to overlapping train/test condition assignments, not confidence intervals over animals or independent trials. A condition's time bins are pooled within each split, so more persistent conditions receive more weight in the point score.

### B: fixed-head condition mismatch

For the objective readout, the fixed-head B control **keeps learned OLS weights and predictions unchanged** while uniformly permuting valid test-condition references at the same original timestamps. It uses 1,000 random allocations, with PCG64 seeds **314159** (Mahler) and **314160** (Perle). Matched and mismatched scores are recalculated on their shared valid condition/time support, including the same epoch masks.

The oriented effects are:

```text
paired_r    = r_matched − r_mismatched
paired_RMSE = RMSE_mismatched − RMSE_matched
```

Positive effects favor matched condition correspondence. Each random statistic averages the original 100 splits; the 2.5th–97.5th percentile range describes the sampled mismatch distribution. Because valid support varies by allocation, these ranges characterize the randomization reference rather than fixed-support permutation or animal-population confidence intervals. The mismatch disrupts physical condition information together with any behavioral correspondence, so its interpretation is condition-level alignment.

### Physical-geometry baseline

The second reference estimates the objective vertical position from each **test condition's known physical anchors, horizontal position and elapsed time**, combined with the **mean objective endpoint of valid training conditions**. This predictor uses task geometry without neural features. Its richer physical information makes it a descriptive benchmark for the neural-only OLS readout; the inputs are unmatched.

For each split, define:

```text
skill = 1 − SSE_neural / SSE_geometry
```

Both squared-error totals use the same evaluated objective target. A positive skill favors the neural readout, zero ties the baseline and a negative skill favors the geometry baseline. Undefined denominators remain `NA`. Split-level skills are averaged **after** calculation; they are not computed as a ratio of averaged SSEs.

## 6. Provenance and interpretation limits

The representation, OLS and null records trace the analysis to its protocol and source files. The evidence concerns **held-out physical conditions from the released, condition-averaged task**. Generalization to simultaneous individual neural trials, paired neural–behavior outcomes, unseen animals or another task remains untested. Elapsed time, physical condition, action preparation and an internally maintained estimate are alternative contributors to hidden-period correlation that the current variables leave unresolved.

Three data-provenance statuses must remain separate: `provided_input_filtering=pass`, `fit_provenance=pass`, and **`raw_preprocessing=fail`**. The released missing-condition neural responses were filled upstream using means across conditions and times; the unavailable unfilled arrays prevent a complete independent leakage audit. Prefix GPFA inference cannot reverse this earlier preprocessing. The final tables also do not contain event-window repeated-half reliability estimates for the per-split FA50/GPFA50 models, and the older GPFA64 results are not substitutes.

**Implementation and evidence:** [Models and Data](../MODELS_AND_DATA.md) provides source identities, model dimensions and the local artifact inventory. [Analysis protocol](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/configs/analysis_protocol.json) and [saved splits](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/configs/condition_splits_100.json) fix the OLS and condition partitions. [`decoding_condition.py`](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/decoding_condition.py) implements `fit_condition_ols`, `score_rows` and `run_condition_round`; [`trial_data.py`](../../../code/mental_pong/trajectory_project/trial_endpoint_fa_gpfa_v1/trial_data.py) implements `fit_condition_geometry` and `build_trial_data`. The [B control protocol](../../../code/mental_pong/trajectory_project/closeout_v1/configs/b_fixed_head_protocol.json), [`null_condition.py`](../../../code/mental_pong/trajectory_project/closeout_v1/null_condition.py) and [`null_endpoint.py`](../../../code/mental_pong/trajectory_project/closeout_v1/null_endpoint.py) define mismatch, support and geometric scoring. The [official protocol audit](../../../code/mental_pong/integration/official_protocol_audit.json) records correspondence with Rajalingham, Sohn and Jazayeri (2025), *Dynamic tracking of objects in the macaque dorsomedial frontal cortex* ([DOI](https://doi.org/10.1038/s41467-024-54688-y)), Fig. 3 and decoding methods. Source-access and replay limits are explained in the [code README](../../../code/mental_pong/README.md).
