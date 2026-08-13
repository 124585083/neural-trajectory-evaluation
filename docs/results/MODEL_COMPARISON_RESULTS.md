# Static vs. Parameter-Matched Dynamic: Conventional Metrics, GPFA, Oracle Performance, and Reliability

Updated: 2026-08-12

## Conclusion

With approximately matched parameter counts, the reduced Dynamic model outperforms the Static model. This conclusion is supported at five levels: official five-session oracle response correlation, single-session/512-neuron response and population-vector correlations, CKA, RSA, and trajectory metrics in a frozen brain-defined GPFA space.

The clearest Dynamic advantage appears in temporal-difference metrics and local-direction metrics in GPFA space. The model difference in speed-profile correlation is small and its bootstrap interval crosses zero, so it would be inappropriate to claim that Dynamic is significantly better on every dynamic metric. This phase constitutes a completed single-session GPFA pilot plus a five-session response check, not a five-session GPFA biological conclusion.

## Model and Data Locks

- Static: 2,814,015 parameters; official Static-on-Dynamic best checkpoint.
- Reduced Dynamic: 2,862,063 parameters; 1.707% more parameters than Static; half-width `[16, 32, 64]` Factorized3D core.
- Both models use the same five Dynamic Sensorium 2023 sessions, train/oracle tiers, neuron ordering, inputs, and training protocol.
- The reduced Dynamic model completed formal training, with checkpoint validation correlation of 0.187893.
- GPFA pilot session: `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20`.
- 512 deterministic neurons; original frames 50--299, for 250 frames total.
- GPFA uses only a pre-locked subset of 174 (50%) of the 348 official train trials in this session. Of these, 139 are used for initialization/selection/fitting and 35 for calibration; after selection, the model is refit on all 174 selected-train trials.
- The 58 oracle trials form 6 repeated-movie conditions with repeat counts of 10/10/9/10/9/10. Neither oracle responses nor model predictions are used for GPFA fitting, scaling, or selection.

## Five-Session Official Oracle Response Correlation

| Model | Mean single-trial correlation |
|---|---:|
| Static | 0.164408 |
| Parameter-matched Dynamic | 0.187525 |
| Dynamic - Static | +0.023117 |

Dynamic improves over Static by 14.1% and scores higher in all five sessions; the session-wise differences are +0.0270, +0.0265, +0.0209, +0.0086, and +0.0325. The reduced Dynamic model retains 95.35% of the full Dynamic oracle score (0.196673).

## Conventional Multi-Metric Evaluation

All metrics use the same `[58 trials, 250 time, 512 neurons]` oracle tensor. The `response` metrics flatten trial and time and compute prediction--response correlation separately for each neuron; the `population-vector` metrics compute correlation across the neuron dimension for each trial/time sample.

| Metric | Static | Dynamic | Direction |
|---|---:|---:|---|
| Single-trial neuron response r | 0.1535 | 0.1812 | Dynamic |
| Condition-average neuron response r | 0.2778 | 0.3348 | Dynamic |
| Single-trial population-vector r | 0.1716 | 0.1895 | Dynamic |
| Condition-average population-vector r | 0.3008 | 0.3380 | Dynamic |
| Temporal-difference neuron r | 0.0215 | 0.0897 | Dynamic |
| Single-trial time-aligned linear CKA | 0.2390 | 0.2643 | Dynamic |
| Condition-average time-aligned linear CKA | 0.4438 | 0.4952 | Dynamic |
| Temporal-difference CKA | 0.0300 | 0.0937 | Dynamic |
| Condition RSA Spearman | 0.2750 | 0.3429 | Dynamic |
| Time-resolved RSA Spearman | 0.3871 | 0.4393 | Dynamic |
| Within-condition temporal RSA Spearman | 0.5535 | 0.6427 | Dynamic |
| Condition-time state RSA Spearman | 0.4713 | 0.5204 | Dynamic |

In the paired bootstrap (2,000 resamples), the Dynamic--Static difference in single-trial neuron response is +0.02776, with a 95% interval of `[+0.02279, +0.03292]`; the single-trial population-vector difference is +0.01793, with interval `[+0.01671, +0.01910]`. The bootstrap Dynamic-better proportion is 1.0 for both metrics.

## Subset GPFA and Full Reliability Reanalysis

The primary model fixes q=4. Among four initial lengthscales, calibration selects 0.25 s. The four final learned timescales are 0.2550, 0.2551, 0.2524, and 0.2580 s. The primary GPFA observes every 4 official frames, yielding 63 approximately 7.5-Hz-equivalent observations, and then queries the posterior at all 250 timestamps.

Balanced split-half oracle reliability over 200 splits:

| Metric | Mean | Approx. 95% split interval |
|---|---:|---:|
| Position correlation | 0.8583 | 0.8177--0.8906 |
| Normalized position RMSE | 0.5424 | 0.4682--0.6453 |
| Velocity-direction cosine | 0.6368 | 0.5969--0.6723 |
| Speed-profile correlation | 0.7777 | 0.7064--0.8354 |
| Acceleration-direction cosine | 0.5579 | 0.5181--0.6002 |

Position, normalized RMSE, velocity, speed, and acceleration all outperform condition shuffle, circular shift, time reversal, 16-frame block shuffle, and independent-neuron shift in 200/200 paired splits, with finite-sample paired p=`1/201=0.004975`.

The rerun sensitivity analysis contains 16 profiles spanning 128/256/512 neurons, q=4/8/12/16, selected-train fractions of 25/50/75/100%, seeds 42/314/2718, and observation step 4/1. Each profile uses 100 splits and five matched null classes. The minimum paired superiority across profiles remains 1.0, with maximum p=`1/101=0.009901`. Profile ranges for the primary reliability metrics are: position 0.8173--0.8698, normalized RMSE 0.5209--0.6151, velocity 0.5365--0.6407, speed 0.6406--0.7958, and acceleration 0.4460--0.5778.

## Frozen GPFA Model Evaluation

The same brain-defined GPFA and train-only preprocessing are applied directly to condition-average brain responses, Static predictions, and Dynamic predictions. GPFA is not refit on model predictions, and no rotations, Procrustes alignment, or model-specific scaling are performed.

| GPFA trajectory metric | Static | Dynamic | Bootstrap conclusion |
|---|---:|---:|---|
| Position correlation | 0.5203 | 0.7262 | Dynamic; advantage CI +0.0883--+0.4513 |
| Normalized position RMSE | 0.8705 | 0.7048 | Dynamic; oriented advantage CI +0.0717--+0.3457 |
| Velocity-direction cosine | 0.3045 | 0.4985 | Dynamic; CI +0.1336--+0.2486 |
| Speed-profile correlation | 0.5271 | 0.5375 | Inconclusive; CI -0.0513--+0.0859 |
| Acceleration-direction cosine | 0.1850 | 0.4525 | Dynamic; CI +0.1822--+0.3395 |
| Zero-lag correlation | 0.5478 | 0.7342 | Dynamic; CI +0.0826--+0.3724 |

The condition-level paired bootstrap uses 2,000 resamples. Except for speed-profile correlation, all metrics in the table have a Dynamic-better proportion of 1.0. For both models, position, normalized RMSE, velocity, speed, and acceleration reject all five model-prediction null classes, with 200-permutation finite-sample p=`1/201=0.004975`. This indicates that Static also captures genuine dynamic structure, but Dynamic better matches trajectory position, direction, and curvature.

## Interpretation Boundaries

1. The current conventional multi-metric and GPFA model comparison covers a deterministic 512-neuron subset from one session; current five-session support comes from official response correlation, not from a five-session GPFA meta-analysis.
2. q=4 is the pre-fixed conservative primary analysis. Sensitivity results support the stability of the reliability gate, but Static--Dynamic ranking has not yet been recomputed for every q/seed combination.
3. GPFA trajectories describe only a repeatable shared low-dimensional subspace of the training responses. They cannot replace response prediction and should not be interpreted as capturing total neural variance or a unique biological dimensionality.
4. The Static--Dynamic difference in speed-profile correlation is uncertain; “Dynamic is better overall” should not be rewritten as “Dynamic is significantly better on every dynamic metric.”

## Artifacts

- Primary protocol and execution entry point: [`experiments/04_model_comparison/`](../../experiments/04_model_comparison/)
- Conventional results: [`results/tables/04_model_comparison/traditional_metrics.csv`](../../results/tables/04_model_comparison/traditional_metrics.csv)
- Frozen GPFA model and train-only preprocessing: [`models/gpfa_model_comparison/`](../../models/gpfa_model_comparison/)
- Reliability results: [`gpfa_reliability_observed.csv`](../../results/tables/04_model_comparison/gpfa_reliability_observed.csv), [`gpfa_reliability_nulls.csv`](../../results/tables/04_model_comparison/gpfa_reliability_nulls.csv), and [`gpfa_reliability_summary.csv`](../../results/tables/04_model_comparison/gpfa_reliability_summary.csv)
- Sensitivity results: [`results/tables/04_model_comparison/sensitivity/`](../../results/tables/04_model_comparison/sensitivity/)
- GPFA model results: [`gpfa_model_metrics.csv`](../../results/tables/04_model_comparison/gpfa_model_metrics.csv), [`gpfa_model_null_summary.csv`](../../results/tables/04_model_comparison/gpfa_model_null_summary.csv), and [`gpfa_model_paired_bootstrap.csv`](../../results/tables/04_model_comparison/gpfa_model_paired_bootstrap.csv)
- Automated tests: [`experiments/04_model_comparison/tests/`](../../experiments/04_model_comparison/tests/) together with reused Phase 2 tests; all nine tests pass.
