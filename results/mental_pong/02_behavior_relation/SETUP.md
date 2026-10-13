# Relating Neural Reconstructions to Behavior-Constrained Paths: Setup

## 1. Question and study design

**Does the available condition-averaged DMFC representation reconstruct a ball path constrained by the animal's behavioral endpoint better than the objective ball path?** A second question is whether any correspondence to that candidate survives controls for condition identity, randomized endpoint assignments and known task geometry.

The analysis uses the same neural features to train two linear readouts: **`D_obj`**, whose target is the physical ball trajectory, and **`D_beh`**, whose target is a trajectory constructed from the *mean terminal paddle position* of the corresponding physical condition. The second path makes the behavioral hypothesis explicit as a reconstruction target. It is constructed retrospectively, with the analysis assessing correspondence to its time course. Testing an internal estimate or predicting a future endpoint from an earlier neural state would require additional observations.

The tests answer distinct questions:

| Test | Comparison | What it can establish |
|---|---|---|
| **A — Own-target and cross-scoring** | Objective and candidate heads fitted from identical neural features, then evaluated against both targets | Which target/head combination is easier to reconstruct, while separating changes in the trained head from changes in the scoring target |
| **B — Fixed-head condition mismatch** | Preserve fitted predictions but assign other test conditions' labels | Whether neural predictions retain the correct condition-to-position correspondence on paired valid support |
| **C — Random endpoints with refitting** | Reassign recorded condition-mean endpoints, rebuild physically constrained candidates, and fit a new readout for each assignment | Whether the observed candidate is better related to the neural representation than the specified endpoint-randomized alternatives |
| **Geometry baseline** | Predict position using physical anchors, timing and a training-condition mean endpoint, without neural features | Whether this neural readout improves on a strong, explicitly task-informed reference |

The comparisons assess candidate reconstruction, condition correspondence, endpoint-assignment structure and added accuracy over geometry. Convergent evidence beyond shared physical structure would support a candidate-specific neural interpretation. The resulting conclusion concerns this candidate and readout; internal computation and other behavioral representations require distinct tests. [Goal 1](../01_task_motion/RESULTS.md) establishes physical position readout feasibility first; [Goal 2 results](RESULTS.md) report the four comparisons here.

## 2. Observations and the two trajectory targets

### Available neural and behavioral data

Both animals are analyzed separately: **Mahler** (518 aligned neurons) and **Perle** (2,058). The neural data are published *condition-mean DMFC pseudopopulations*: **79 physical conditions × 100 bins of 50 ms**, assembled across recording sessions. For every one of the 100 fixed splits, FA50 or prefix-inferred GPFA50 provides 50 neural features per condition and time. The final OLS comparison uses **released half1**. Half2 is saved but is not an independent behavioral result.

Behavioral terminal-paddle records are retained after the saved occlusion-category, completed-record, finite-endpoint, geometry-support and `ignore=0` rules. These yield **7,407 Mahler records from six source sessions** and **84,873 Perle records from 58 source sessions**. Records are pooled by physical condition, with each retained record weighted equally. The membership of each mean is fixed across time: no time-specific re-selection, correct/incorrect-trial filter or endpoint-sign selection is applied. These are behavioral means, not neural trials paired to individual behavior.

Both targets share the same valid support: **78 conditions, 3,369 condition × time rows per animal**. Physical condition `59920` remains in the nominal 79-condition splitting file but has unresolved collision/terminal geometry and supplies no valid scoring rows.

| Evaluation range | Conditions | Unique condition × time rows | Meaning |
|---|---:|---:|---|
| `full` | 78 | 3,369 | Conservative active interval, including 78 mixed occlusion-boundary bins |
| `visible` | 78 | 1,931 | Eligible visible periods |
| `hidden` | 78 | 1,360 | Eligible occluded periods |
| `bounce` | 26 | 1,207 | Conditions containing a physical collision |
| `no_bounce` | 52 | 2,162 | Conditions without a physical collision |
| `post_bounce` | 26 | 631 | Post-collision periods |
| `endpoint_influence` | As retained by the saved mask | 2,793 | Additional C-control period in which the endpoint assignment can affect the candidate |

The visible and hidden masks omit the 78 mixed boundary bins; therefore their row counts do not sum to `full`. Physical collision/no-collision labels are condition groups, not alternatives to the visible/hidden distinction. The same finite, feedback-conservative masks are applied to both targets. See [Models and Data](../MODELS_AND_DATA.md) for source array definitions, released neural preprocessing and physical-event provenance.

### How the candidate path is constructed

![Collision and no-collision candidate construction, the two-by-two comparison, and distinct B/C controls](../figures/documentation/methods_targets_controls.png)

*The sketches illustrate the current affine construction, not measured internal trajectories. B keeps the fitted head fixed; C refits it after changing endpoint assignments. Neither control substitutes for the separate geometry baseline.*

Let \(b_c\) denote the retained **mean terminal paddle y-coordinate** for condition \(c\). The objective trajectory \((x_{\mathrm{obj},c}(t),y_{\mathrm{obj},c}(t))\) is determined from the released physical motion and conservative event geometry. The candidate keeps **objective x, arrival time, and valid masks** and substitutes a linearly endpoint-constrained y-path.

The collision anchor starts the candidate's deviation from the objective path; no-collision conditions use the metadata initial anchor. The candidate can therefore differ during **visible motion**, which distinguishes this construction from the historical hidden-only and successive-update proxies.

At or after the relevant anchor, with \((x_{a,c},y_{a,c})\) fixed by the task geometry,

\[
\alpha_c(t)=\frac{x_{\mathrm{obj},c}(t)-x_{a,c}}{10-x_{a,c}},
\qquad
 y_{\mathrm{beh},c}(t)=[1-\alpha_c(t)]y_{a,c}+\alpha_c(t)b_c,
 \qquad x_{\mathrm{beh},c}(t)=x_{\mathrm{obj},c}(t).
\]

The construction uses no fitted neural scores, new bounce inference, time warping, speed renormalization or dynamic endpoint update. It is affine in the endpoint: averaging the valid per-record constructed paths yields the same path as applying the construction to their mean endpoint, up to floating-point precision. **All 78 × 78 = 6,084 physical-condition/endpoint combinations per animal** were checked for finite geometry. Neither the exact physical collision time nor the strict pre-feedback timestamp of the terminal paddle scalar is available at single-trial resolution. A terminal label constructed retrospectively should therefore not be described as a directly observed prospective belief.

## 3. Fixed representations, readouts and data separation

The final analysis is **`condition_endpoint_fa_gpfa_v1`**, followed by **`closeout_v1`**; the completion status is `CLOSED_EXPLORATORY_WITH_LIMITATIONS`. FA50 and GPFA50 were fitted separately for each animal and each fixed condition split, using only **half1 training-condition neural responses**. GPFA50 emits each saved posterior coordinate from the prefix of completed *released* neural bins; it is not an unrestricted full-window posterior in this study. The same source latents and preprocessing were reused in the earlier trial-label version, which differs downstream in targets and row weighting.

The 100 saved whole-condition partitions use:

```python
GroupShuffleSplit(n_splits=100, train_size=0.5, random_state=0)
# 39 nominal training and 40 nominal test physical identities
```

The missing-geometry condition is never replaced, so a particular split may contain 38/40 or 39/39 *contributing* conditions. The same partition identities are used for both animals, both representations and both readout targets. Each training-side transform fixes neuron eligibility, mean, scale and 50 latent dimensions before test scoring. There is **no trial-specific neural–behavior pairing**, no separate validation-based OLS head search, and no test-conditioned change of the fitted representation.

Within a split, construct an identical design matrix \(X\) of valid training condition × time rows, shape \([n_{\mathrm{train\ rows}},50]\), and fit two separate unregularized ordinary least-squares maps, each with an intercept:

\[
\widehat{\mathbf y}_{\mathrm{obj}}=D_{\mathrm{obj}}(\mathbf z),\qquad
\widehat{\mathbf y}_{\mathrm{beh}}=D_{\mathrm{beh}}(\mathbf z).
\]

Each coordinate has **50 slopes + one intercept** (102 fitted coefficients for a complete xy head). The actual implementation is float64 `sklearn.linear_model.LinearRegression(fit_intercept=True, positive=False)` with **no added feature normalization, Ridge penalty, behavior covariate, time covariate, temporal history or trial-count weighting** in the OLS fit. All valid training times share the same linear head; longer conditions contribute more rows. Store predictions **only for held-out conditions** and reuse those predictions for every phase analysis. Since the two targets have identical x coordinates, their x labels and predicted x values should agree numerically; this is an implementation check, not behavior-specific evidence.

A saved GPFA prefix is future-free relative to the **provided, already processed array**. It cannot undo upstream filling of missing whole-condition neural responses using global condition/time information. Consequently `provided_input_filtering=pass` and `fit_provenance=pass` coexist with the retained **`raw_preprocessing=fail`**. Model choices and data provenance are recorded in [Models and Data](../MODELS_AND_DATA.md).

## 4. Evaluation procedure: A, B and C

### A. Own-target reconstruction and two-by-two comparison

The illustrated 2×2 matrix scores each frozen prediction against both references on the **same valid test rows**. **OO versus BB** changes both the fitted head and target, so target variance, time structure and difficulty can affect the result. **OO versus BO** and **OB versus BB** hold the respective reference fixed. None is a forecast of two individual trials' behavioral choices.

The primary outcome is the **vertical coordinate y**, because both targets have identical x. Six phase scores use fixed full-interval heads; no new phase-specific OLS fits are performed. The full-interval `bias`, target/prediction SD and amplitude ratio add calibration information that correlation alone omits.

### B. Fixed-head condition mismatch

At the original timestamps, **permute candidate labels across eligible test conditions** while preserving the fitted predictions. Matched and mismatched scores use the **same intersection** of valid masks within an allocation; intersection sizes can differ across allocations, especially after collisions.

Use **1,000 allocations** per animal across the same **100 saved splits**, `numpy.random.Generator(PCG64)` with seeds **314159 (Mahler)** and **314160 (Perle)**; the saved execution traverses allocation `q` outside the split loop, with no retries. Fixed points in a permutation are allowed. The readout weights, neural inputs and predictions remain unchanged. The resulting effect is **matched minus mismatched correlation** or **mismatched minus matched RMSE**: positive means correct correspondence is better. This is a test of *condition-related correspondence*, not a selective shuffle of behavioral information while preserving physical condition.

### C. Randomized endpoints with a newly fitted readout

**Permute the 78 observed mean endpoints among physical conditions**, preserving each destination's anchors, x, arrival time and masks. Rebuild its affine candidate, **refit OLS on the 39 nominal training identities**, and score the 40 nominal held-out identities. Refitting evaluates how well the same readout procedure learns each changed target.

Use **1,000 global allocations** with Mahler seed **271828** and Perle seed **271829** (`PCG64`). A given animal's allocation is reused for all 100 splits and both FA/GPFA representations. The original candidate is compared with these **fitted random-target alternatives**, not with untrained random outputs. The batched unweighted float64 OLS implementation was checked against the original scikit-learn fits. Saved allocations `q=0,1,2` support limited replay, while the completed statistical control includes all 1,000. Random targets can still have positive correlation because they preserve physical geometry and time.

### Task-geometry baseline

For each test condition, construct a trajectory from **its known physical anchors, x/time and the mean endpoint among valid training conditions**. This geometry-only prediction provides the denominator for skill against the corresponding neural head's target; it is recalculated under randomized endpoint assignments when required. The baseline receives additional physical variables, making it a strong descriptive reference with **unmatched architecture and inputs**.

## 5. Metrics and meaning of uncertainty

For each animal, representation, held-out split and phase, pool the eligible **condition × time** rows, then compute Pearson correlation and root mean square error (RMSE) for y:

\[
 r=\operatorname{corr}(\widehat y,y),\qquad
 \operatorname{RMSE}=\sqrt{\operatorname{mean}[(\widehat y-y)^2]}.
\]

For A, report the paired differences using saved full-precision split scores:

\[
 \Delta r=r_{\mathrm{BB}}-r_{\mathrm{OO}},\qquad
 \Delta\operatorname{RMSE}=\operatorname{RMSE}_{\mathrm{OO}}-\operatorname{RMSE}_{\mathrm{BB}}.
\]

**Positive A differences favor the candidate.** The reported mean and population SD (`ddof=0`) are across **100 overlapping condition splits**, not 100 independent animal or neural experiments. For A, the difference must be averaged at full precision before rounding; it need not equal the difference between the displayed four-decimal means.

For B, each of 1,000 allocations produces an effect statistic **already averaged across the 100 fixed splits**. Report its 2.5th–97.5th percentile range, the mean effect, coverage and the empirical fraction of nonpositive effects. The pair-specific support varies with permutation; these are *randomization-reference quantiles*, **not confidence intervals across animals** or exact fixed-support permutation p-values.

For C, each of 1,000 assignments similarly gives a split-averaged score. The correlation tail is the fraction with random `r` **greater than or equal to** the actual candidate's `r`; the error tail is the fraction with random RMSE **less than or equal to** the actual candidate's RMSE. A saved `0/1000` outcome is finite Monte Carlo resolution, not zero underlying probability.

The geometry comparator uses

\[
 \operatorname{skill}=1-\frac{\operatorname{SSE}_{\mathrm{neural}}}{\operatorname{SSE}_{\mathrm{geometry}}},
\]

computed **within each split** before averaging; a zero baseline denominator remains undefined (`NA`). A negative value means the neural readout has larger squared error on the matched target. The diagnostic **bias** is `mean(prediction − target)`, and **amplitude ratio** is `SD(prediction) / SD(target)`; they measure offset and compression that r can miss. Constant or fewer-than-three-point correlations remain undefined, never silently zero-filled.

These metrics quantify contemporaneous task-position reconstruction from condition means, with phase masks locating performance within the task. Local neural dynamics and internal simulation remain questions for tests of state transitions and competing computations.

## 6. Source code, evidence and limits

The analysis depends on the recorded source functions and immutable split/control configurations:

- [Condition averaging](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/condition_data.py) (`mean_payload`) and [physical/endpoint geometry](../../../code/mental_pong/trajectory_project/trial_endpoint_fa_gpfa_v1/trial_data.py) (`fit_condition_geometry`, `endpoint_candidates`): fixed mean labels and task branch construction.
- [Condition OLS](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/decoding_condition.py) (`fit_condition_ols`, `score_rows`) and [analysis protocol](../../../code/mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/configs/analysis_protocol.json): design rows, readout and holdout definition.
- [Closeout protocol](../../../code/mental_pong/trajectory_project/closeout_v1/configs/closeout_protocol.json), [B protocol](../../../code/mental_pong/trajectory_project/closeout_v1/configs/b_fixed_head_protocol.json), [C protocol](../../../code/mental_pong/trajectory_project/closeout_v1/configs/c_endpoint_protocol.json), [fixed-head null code](../../../code/mental_pong/trajectory_project/closeout_v1/null_condition.py) and [random-refit code](../../../code/mental_pong/trajectory_project/closeout_v1/null_endpoint.py): seeds, mapping and scoring implementation.
- [Model/data provenance](../MODELS_AND_DATA.md), [software and actual replay scope](../../../code/mental_pong/README.md), [Goal 1 physical reference](../01_task_motion/RESULTS.md), [Goal 2 numerical evidence](RESULTS.md) and [next observations](../NEXT_STEPS.md).

The local endpoint candidate and B/C controls are **project adaptations**, not experiments claimed as part of the original authors' paper (Rajalingham, Sohn and Jazayeri, 2025, [DOI](https://doi.org/10.1038/s41467-024-54688-y)). The behavioral endpoint's exact pre-feedback timing, the original neural–behavior trial memberships, neural means before upstream global-condition/time filling, and single-trial collision/feedback records are unavailable. `raw_preprocessing=fail` is retained. These gaps prevent within-condition behavioral-choice tests and strict raw-data causal reconstruction; the conclusion must remain at the condition-mean readout level.
