# Recovering Task-Motion Information: Setup

## 1. Question and purpose

**Can a readout of condition-averaged dorsomedial frontal cortex (DMFC) activity recover physical ball position during visible motion and occlusion?** This is the first question in the Mental-Pong study. Establishing recoverable task-position information provides the reference for the later comparison with the animal's behavioral endpoint.

The primary measurement fits a linear readout, **`D_obj`**, from a 50-dimensional neural representation to the **objective, physically defined ball position**. Its predictions are scored on held-out physical conditions. The same fixed head is evaluated separately during visible motion, occlusion, collision and post-collision periods.

Two controls assess the source of reconstruction accuracy. A **fixed-head condition mismatch** tests the intended condition-to-position correspondence. A **task-geometry baseline** tests whether the neural readout improves on position estimates from specified physical anchors and timing information. Together, these controls assess condition correspondence and accuracy relative to geometry. The computational source of that correspondence requires a further discriminating test. The later [behavior-relation analysis](../02_behavior_relation/SETUP.md) uses the same representations to compare objective and endpoint-constrained targets.

## 2. Data and evaluation support

### Neural observations

The neural input consists of the released *condition-mean responses*. The source fields `neural_responses_sh1.occ` and `neural_responses_sh2.occ` have axes **neuron × 79 physical conditions × 100 time bins**. Each bin spans 50 ms. Global neuron identifiers align the two released halves, yielding **518 units for Mahler** and **2,058 for Perle**. The fitted representations and final OLS analysis use **half1**; half2 latent arrays are retained separately as representation records.

The `occ` suffix names the *occlusion task category*. The terms *visible* and *hidden* refer to time ranges within that task. Reported neural values are processed means of spike counts per original 1 ms sample, averaged over 50 ms. Train-derived standardization sets the scale for representation fitting. [Models and Data](../MODELS_AND_DATA.md) describes the source arrays and preprocessing.

### Ball coordinates, timing and usable rows

The ball travels toward a paddle plane at **x = 10** in centered MWorks display coordinates. It is visible initially, then occluded; some physical conditions include a boundary collision before the ball reaches the paddle. The objective XY path is defined by the recorded physical-trajectory geometry and event metadata.

The current analysis evaluates each neural bin at its completed-bin availability time, `(k + 1) × 50 ms`, for zero-based bin `k`. The objective-geometry reconstruction uses original bin centers, `50k + 24.5 ms`, for fitting, then evaluates the physical branches at completed-bin times. These choices define label–response alignment; estimating exact neural processing latency requires additional timing evidence.

| Scoring range | Valid conditions | Unique condition × time rows |
|---|---:|---:|
| Full active interval | 78 | 3,369 |
| Visible | 78 | 1,931 |
| Hidden | 78 | 1,360 |
| Collision conditions (`bounce`) | 26 | 1,207 |
| No-collision conditions (`no_bounce`) | 52 | 2,162 |
| Post-collision | 26 | 631 |

The full interval includes **78 boundary/mixed rows outside the visible and hidden masks**, alongside the rows assigned to those two phases. The collision/no-collision labels group physical conditions independently of the visibility masks. Masks remove bins that overlap feedback or extend beyond estimated arrival.

Condition **`59920`** has unresolved terminal/collision geometry. It is retained in the nominal 79-condition split identities, with its position labels left unscored. Unsupported times remain missing and unscored. The two animals and both representations use the same physical-condition labels and scoring masks. The saved result names are `full`, `visible`, `hidden`, `bounce`, `no_bounce` and `post_bounce`.

## 3. Models, fits and train/test separation

![Condition-held-out neural fitting and test-only position scoring](../figures/documentation/methods_condition_workflow.png)

*Goal 1 evaluates the objective head within this shared protocol. Goal 2 additionally compares the candidate head and all four head/reference combinations. GPFA prefix inference is restricted to the released input, whose upstream global filling remains unresolved.*

### Representation and readout

For each animal and each of **100 fixed condition splits**, factor analysis (**FA50**) and Gaussian-process factor analysis (**GPFA50**) are fitted from half1 **training-condition neural responses only**. Both produce a 50-dimensional input for a separate ordinary least-squares (OLS) position readout.

| Component | Saved method and settings | Interpretation |
|---|---|---|
| FA50 | `sklearn.decomposition.FactorAnalysis`, 50 components; seed 42; `max_iter=300`, `tol=0.01`, LAPACK, no rotation | Framewise latent features from the fitted observation model |
| GPFA50 | `SharedTimescaleGPFA`; `y(t) = C z(t) + d + ε(t)`; diagonal observation noise and one shared RBF timescale | Smooth neural latent estimates; each emitted state uses the completed **released-input prefix** only |
| GPFA fitting | FA50 initialization with 1% loading jitter; initial timescale 0.15 s, bounds 0.025–2 s; nugget 0.001; minimum noise 0.0001; at most 400 iterations, tolerance `1e-6` | Fixed per-split fit from training neural responses |
| Objective OLS head | `LinearRegression(fit_intercept=True, positive=False)`; 50 features, one intercept per coordinate; float64 | Fixed mapping from neural features to task XY position |

OLS has **51 parameters per output coordinate**, or 102 parameters for an XY head. OLS uses the current latent vector as its sole input, with an unregularized, unweighted fit and fixed settings. Additional feature history, behavior and explicit time covariates are absent from this readout design. The latent input may still reflect condition and elapsed-time structure indirectly. GPFA inference uses the available prefix of *provided bins*. These bins retain the dataset provider's earlier time- and condition-wide filling.

The current fitted objects are different from the released authors' FA factors, which were fitted across all conditions before readout cross-validation. Historical GPFA64 and Ridge tests are documented separately under their own protocols.

### Fixed condition splits and preprocessing

The retained split protocol is:

```python
GroupShuffleSplit(n_splits=100, train_size=0.5, random_state=0)
# Nominally 39 train conditions and 40 test conditions per split
```

The exact same physical-condition assignments are reused for Mahler, Perle, FA50, GPFA50 and both downstream position targets. Because condition `59920` has no valid rows, the contributing conditions are either **38 train / 40 test** or **39 train / 39 test**, depending on which side contains it. The original split membership is retained.

Each split uses **half1 training-condition active rows** to select neurons (finite fraction at least 0.99, training SD above `1e-5`) and compute each neuron's mean and SD. The transformation `(response − training_mean) / training_SD` is frozen for test conditions. Remaining nonfinite inputs are set to zero after that conversion. Recorded checks also require at least 0.95 observed retained-neuron coverage per measurement bin in each stored half. All 518/2,058 aligned neurons passed the recorded fit-eligibility checks.

Training-condition rows are used to fit the FA/GPFA representation and objective OLS head; held-out condition rows provide the scores. The OLS fit pools **all valid training condition × time rows** across the active period, so longer conditions contribute more observations. This train/test design uses one fixed head across all phases, with no separate validation partition. The 100 test partitions are overlapping resamplings of the same neural dataset and quantify partition sensitivity.

## 4. How the objective position is reconstructed

After neuron alignment, the frozen half1 latent array has shape `[79,100,50]` for each split. The objective head is fitted on valid full-interval training rows and applied only to test rows; its predictions are reused across phases. Scores pool eligible condition × time rows **within each split**, then summarize the **100 separately scored splits**.

For the physical labels, `fit_condition_geometry` fits horizontal position at the original bin centers and piecewise straight `y(x)` branches from audited uncontaminated samples. It then evaluates those branches at completed-bin times. Arrival is solved at **x = 10**. Two post-collision observations permit separate branch fits; with one, the recorded elastic-reflection constraint is used. Unsupported event histories remain unscored.

These **task-coordinate readouts** map latent activity to physical position. Phase-wise r and RMSE describe position reconstruction within each time range. Tests of local direction, future prediction and state updating would extend this task-coordinate analysis. Both targets share x and arrival time; the behavioral comparison belongs to [Goal 2](../02_behavior_relation/SETUP.md).

## 5. Metrics, correspondence control and geometric reference

### Readout scores and uncertainty

For each held-out split and task phase, Pearson `r` compares the pooled predicted and objective samples **within one coordinate**. RMSE is `sqrt(mean((prediction − target)^2))` in centered display-coordinate units. Higher `r` and lower RMSE indicate better reconstruction of the specified reference. Correlations with constant values or fewer than three valid rows remain `NA`.

Reported `mean ± SD` values use the 100 split-level scores and **population SD (`ddof=0`)**. These SDs quantify variation across overlapping train/test condition assignments within the two recorded animals. Time bins are pooled within each split, so conditions with longer valid intervals contribute more rows to the point score.

### B: fixed-head condition mismatch

For the objective readout, the fixed-head B control **keeps learned OLS weights and predictions unchanged** while uniformly permuting valid test-condition references at the same original timestamps. It uses 1,000 random allocations, with PCG64 seeds **314159** (Mahler) and **314160** (Perle). Matched and mismatched scores are recalculated on their shared valid condition/time support, including the same epoch masks.

Define the paired effects as:

```text
paired_r    = r_matched − r_mismatched
paired_RMSE = RMSE_mismatched − RMSE_matched
```

Positive effects favor matched condition correspondence. Each random statistic averages the original 100 splits; the 2.5th–97.5th percentile range describes the sampled mismatch distribution. Because valid support varies across allocations, these ranges are quantiles of the sampled mismatch reference, conditional on its changing support. The mismatch disrupts physical condition information together with any behavioral correspondence, so its interpretation is condition-level alignment.

### Physical-geometry baseline

The second reference estimates the objective vertical position from each **test condition's known physical anchors, horizontal position and elapsed time**, combined with the **mean objective endpoint of valid training conditions**. This predictor derives position from task geometry alone. Its richer physical information makes it a descriptive benchmark for the neural-only OLS readout; the inputs are unmatched.

For each split, define:

```text
skill = 1 − SSE_neural / SSE_geometry
```

Both squared-error totals use the same evaluated objective target. A positive skill favors the neural readout, zero ties the baseline and a negative skill favors the geometry baseline. Undefined denominators remain `NA`. Calculate skill within each split, then average the split-level values.

## 6. Evidence and interpretation

The representation, OLS and null records trace the analysis to its protocol and source files. The evidence concerns **held-out physical conditions from the released, condition-averaged task**. Transfer to paired individual trials, unseen animals or another task would require evaluation on those observations. Hidden-period correlation may reflect elapsed time, physical condition, action preparation or an internally maintained estimate; distinguishing their contributions requires additional comparisons.

The audit records three distinct checks: `provided_input_filtering=pass`, `fit_provenance=pass`, and **`raw_preprocessing=fail`**. The released missing-condition responses were filled using means across conditions and times. Assessing the independence of the raw inputs requires the unavailable unfilled arrays. Prefix GPFA inference operates on these already processed arrays. Event-window repeat-half reliability remains unmeasured for the final per-split FA50/GPFA50 models; historical GPFA64 scores concern a separate fit.

**Implementation and evidence:** [Models and Data](../MODELS_AND_DATA.md) provides source identities, model dimensions and the local artifact inventory. [Analysis protocol](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/configs/analysis_protocol.json) and [saved splits](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/configs/condition_splits_100.json) fix the OLS and condition partitions. [`decoding_condition.py`](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/decoding_condition.py) implements `fit_condition_ols`, `score_rows` and `run_condition_round`; [`trial_data.py`](../../../code/mental_pong/trajectory_project/trial_endpoint_fa_gpfa_v1/trial_data.py) implements `fit_condition_geometry` and `build_trial_data`. The [B control protocol](../../../code/mental_pong/trajectory_project/closeout_v1/configs/b_fixed_head_protocol.json), [`null_condition.py`](../../../code/mental_pong/trajectory_project/closeout_v1/null_condition.py) and [`null_endpoint.py`](../../../code/mental_pong/trajectory_project/closeout_v1/null_endpoint.py) define mismatch, support and geometric scoring. The [official protocol audit](../../../code/mental_pong/integration/official_protocol_audit.json) records correspondence with Rajalingham, Sohn and Jazayeri (2025), *Dynamic tracking of objects in the macaque dorsomedial frontal cortex* ([DOI](https://doi.org/10.1038/s41467-024-54688-y)), Fig. 3 and decoding methods. The [code README](../../../code/mental_pong/README.md) describes source access and the supported replay routes.
