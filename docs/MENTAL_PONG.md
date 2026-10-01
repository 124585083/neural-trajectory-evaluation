# Mental-Pong: neural position reconstruction and behavioral references

## Motivation

A reliable neural trajectory can describe how a representation changes, but its geometry alone does not identify a task variable or a computation. Mental-Pong adds external task references: an objective ball path and a behavior-constrained candidate path. We asked which path is easier to reconstruct from the same neural representation and whether the correspondence survives two distinct randomization controls.

This differs from the Dynamic Sensorium study, which compares neural and model-predicted trajectories in a shared representation space. Mental-Pong uses task-coordinate readouts and external paths. Their scores, implementations and validation scopes remain separate. The common research direction concerns reliable temporal structure, its relationship to observable variables and the data needed to distinguish competing explanations.

## Data level

The released DMFC responses form a condition-mean pseudopopulation: units recorded across sessions are assembled by physical task condition, rather than observed together on a single trial. Mahler and Perle are analyzed separately using half1 and 50 ms completed bins. The 100 fixed nominal 39/40 splits retain all 79 IDs. Unresolved collision geometry for 59920 leaves 78 usable conditions and 3369 condition-by-time rows per animal. Behavioral means use 7407/84873 records, without an equivalent count of independent neural trials.

The original unit-by-trial membership of the neural means is unknown. Behavioral mean membership is traceable, but exact neural-behavioral membership matching has not been established. The [coverage and methods](../experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/REPORT.md#coverage-and-label-checks) preserve the available support and missing identities.

## Objective and candidate paths

The objective ball path is the physical reference. The candidate uses the condition-mean final paddle position. With a collision it follows the objective path up to the estimated collision anchor, then connects that anchor to the endpoint. Without a collision it connects the task starting point to the endpoint. x, arrival time and masks are shared. The final version contains no paddle-stop detector or successive target updates.

This construction gives a behavioral reference for decoding. It is not a measurement of the animal's internal path. The endpoint scalar's strictly pre-feedback timing remains unverified, and collision anchors come from released trajectories. [Protocol differences](../experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/protocol_diff.md) separate this version from earlier proxies.

## FA, GPFA and a fixed linear readout

Factor analysis (FA) describes shared neural variation using latent factors. Gaussian-process factor analysis (GPFA) adds a time prior; the version here has one shared learnable RBF time scale. Both have 50 latent dimensions. The study reuses each split's training-side FA50/GPFA50 and applies ordinary least squares (OLS), a linear regression with an intercept. Each coordinate has 51 coefficients and each xy head has 102. No added regularization, scaling, behavioral features or record-count weights are used. A single mapping spans valid visible and hidden rows.

Each target has its own trained head but shares the same inputs, split and support. The [locked protocol](../experiments/05_mental_pong/trajectory_project/closeout_v1/configs/closeout_protocol.json) retains the original timing and fit boundaries. This GPFA implementation and its tests are separate from Sensorium's brain-defined GPFA.

## Own-target reconstruction

Each split is scored before the 100 split scores are summarized. Repeated splits describe readout stability and are not independent animal experiments.

| animal | representation | r_obj | r_beh | RMSE_obj | RMSE_beh | Delta_r | Delta_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mahler | FA50 | 0.4928 | 0.4696 | 4.5909 | 4.6129 | -0.0232 | -0.0220 |
| mahler | GPFA50 | 0.5374 | 0.5126 | 4.4458 | 4.4781 | -0.0248 | -0.0322 |
| perle | FA50 | 0.7019 | 0.6910 | 3.8586 | 3.8829 | -0.0108 | -0.0242 |
| perle | GPFA50 | 0.7204 | 0.7094 | 3.8005 | 3.8265 | -0.0110 | -0.0261 |

All four full-epoch comparisons favor the objective target. No-collision candidates have lower RMSE but lower correlation; post-collision own-target RMSE favors the objective path. Different target variances and difficulties limit a neural-preference interpretation. The [complete report](../experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_REPORT.md#a-own-target-reconstruction) keeps phase-specific and per-condition results, including the post hoc examples 55062 and 241919.

## Cross-scoring

A 2x2 evaluation scores each head against both targets. OO versus BB changes both target and trained head. OB versus BB holds the candidate target fixed, and OO versus BO holds the objective target fixed. In all four full-epoch groups the objective-trained head performs better against either shared target. These comparisons do not reduce to decoding the target difference. [All four cells](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/A/cross_2x2_summary.csv) retain r and RMSE.

## Two randomization controls

**Fixed-readout condition mismatch (B)** exchanges valid test-condition labels within each split without fitting a new readout. Matched and mismatched scores use identical overlapping timestamps and phase membership. Correct correspondence performs better. The control disrupts both physical and behavioral associations, supporting condition-related task structure. Post-collision overlap averages only about 17.6%, with missing-support cases preserved. Changing support makes the reported tails empirical comparison proportions rather than exact fixed-support p-values.

**Random-endpoint fitting (C)** reallocates the observed endpoint set while preserving each condition's geometry. A fixed allocation applies across time, both representations and all original splits; new OLS heads fit its training labels. There are 1000 allocations per animal, each evaluated over 100 splits. Actual candidates outperform this reference over full, hidden and endpoint-influence ranges, with post-collision RMSE exceptions. Random candidates retain starting, collision and temporal structure, so positive random correlations are expected to remain possible. [B results](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/B/B_summary.csv) and [C results](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/C/random_endpoint_summary.csv) answer different questions.

## Mean-endpoint geometry baseline

The baseline combines a training-set mean endpoint with each test condition's known anchor and horizontal motion. It uses task geometry beyond the neural input. All four full-epoch neural results have negative `skill = 1 - SSE_neural/SSE_baseline`. Perle has small positive candidate skills for GPFA50 hidden samples and both representations' no-collision samples. Exceeding randomized endpoints therefore does not imply a general gain over shared geometry. The [phase-specific baseline account](../experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_REPORT.md#mean-endpoint-geometry-baseline) retains these exceptions.

## What averaging changes

Opposite endpoint errors occur in 77/78 Mahler and 78/78 Perle conditions. The condition-equal mean absolute trial errors are 1.7272/1.3388, whereas the absolute errors of condition means average 0.9784/0.7908. Median cancellation fractions are 54.0%/48.0%. Full-path label separation is 0.6760/0.5612 RMS, below the reconstruction-error scale. The comparison uses different aggregation from readout scoring and is descriptive rather than a power calculation.

These results establish behavioral cancellation. They do not measure neural information lost through averaging. The original neural groups cannot be recovered from their combined mean. The earlier trial-label version also changed condition weights, so its comparison with the final means is not a one-factor averaging test. See the [descriptive audit](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/descriptive/mean_cancellation_interpretation.md).

## Evidence limits and next design

The final scientific status is `CLOSED_EXPLORATORY_WITH_LIMITATIONS`. **`raw_preprocessing=fail` remains unresolved.** Filtering and fit provenance pass within the released-input scope, while filling across conditions and times persists upstream. Unknown neural members, terminal timing uncertainty and estimated collision anchors further limit interpretation. The results support readable task-position structure, do not support a full-epoch candidate-path advantage, and cannot test opposite behavioral groups within one physical condition. They do not establish a unique computation or neural causal mechanism.

The [single current future plan](FUTURE_DIRECTIONS.md) specifies the paired trial records needed for above-target/below-target/near-correct group comparisons, then asks what additional temporal information could distinguish explanations. It remains a design. Start with the [module reading and execution guide](../experiments/05_mental_pong/README.md) or the [final summary](../experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_SUMMARY.md). The [history](../experiments/05_mental_pong/archive/HISTORY.md) retains earlier experiments without blending their results into the final analysis.
