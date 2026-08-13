# Q1–Q6: Hierarchical Conclusions for Static vs. Parameter-Matched Dynamic

Updated: 2026-08-13

## Summary

| Question | Answer | Evidence strength |
|---|---|---|
| Q1 Does Dynamic show a response-level gain? | Yes | Consistent across five sessions; clear effect, but session-level sample size remains small |
| Q2 Do RSA / CKA detect a Dynamic–Static difference? | Yes; most clearly in time-aware variants | Single session, 6 conditions |
| Q3 Does trajectory evaluation detect a difference? | Yes: position, velocity, acceleration; speed remains uncertain | Frozen GPFA + condition bootstrap |
| Q4 Can temporal differences still be detected when response scores are similar? | Yes | Response-matched stress test with non-overlapping repeat halves |
| Q5 Does temporal ablation produce monotonic degradation? | Strictly monotonic for four similarity metrics; RMSE is not fully strict | Five-level learned temporal-history ablation |
| Q6 Does trajectory information contain something not fully explained by response/RSA/CKA? | Evidence supports “not fully explained by the current conventional battery”; no claim of mathematical independence | Strict time-reversal counterexample + matched pair + leave-family-out regression |

## Q1: Does Dynamic Show a Response-Level Gain?

**Answer: Yes.**

Official oracle single-trial correlations across five sessions:

| Model | Score |
|---|---:|
| Static | 0.164408 |
| Parameter-matched Dynamic | 0.187525 |
| Dynamic - Static | +0.023117 |

The relative improvement is 14.06%. All five session-wise differences are positive: `+0.0270, +0.0265, +0.0209, +0.0086, +0.0325`. Using session as the resampling unit, the bootstrap 95% interval is `[+0.0147, +0.0291]`. With all five sessions positive, the one-sided exact sign p=`0.03125`; the two-sided p=`0.0625`.

It is therefore appropriate to state that “Dynamic shows a stable response-level gain,” but with only five independent sessions, the strength of the two-sided session-level significance should not be overstated.

## Q2: Do RSA / CKA Detect a Dynamic–Static Difference?

**Answer: Yes, especially in RSA/CKA variants that preserve temporal structure.**

Treating the six repeated movies as independent conditions:

| Metric | Static | Dynamic | Difference | Condition bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Within-condition temporal CKA | 0.5131 | 0.5985 | +0.0854 | +0.0304--+0.1655 |
| Temporal-difference CKA | 0.0991 | 0.2027 | +0.1036 | +0.0921--+0.1154 |
| Within-condition temporal RSA | 0.5535 | 0.6427 | +0.0892 | +0.0308--+0.1930 |

Earlier pooled metrics show the same direction: condition-average time-aligned CKA increases from `0.4438 → 0.4952`, and condition×time state RSA increases from `0.4713 → 0.5204`.

The conclusion should be stated as: “RSA/CKA detect the difference, but the difference is concentrated in time-aware variants; purely condition-averaged CKA shows only a small difference.” Because there are only six movie conditions, condition-level uncertainty still requires validation in additional sessions.

## Q3: Does Trajectory Evaluation Detect a Dynamic–Static Difference?

**Answer: Yes, with particularly strong differences in trajectory-direction metrics.**

Using the frozen q=4 brain-defined GPFA, with no latent alignment applied to either model:

| Metric | Static | Dynamic | Oriented Dynamic advantage 95% interval |
|---|---:|---:|---:|
| Position correlation | 0.5203 | 0.7262 | +0.0883--+0.4513 |
| Normalized position RMSE | 0.8705 | 0.7048 | +0.0717--+0.3457 |
| Velocity-direction cosine | 0.3045 | 0.4985 | +0.1336--+0.2486 |
| Speed-profile correlation | 0.5271 | 0.5375 | -0.0513--+0.0859 |
| Acceleration-direction cosine | 0.1850 | 0.4525 | +0.1822--+0.3395 |

Trajectory evaluation therefore supports the conclusion that Dynamic more closely matches the brain trajectory in position, local direction, and curvature/acceleration. The speed-profile difference is small and its interval crosses zero, so no reliable model difference should be claimed for that metric.

## Q4: When Response Scores Are Similar, Can Trajectory Evaluation Still Detect a Temporal Computation Difference?

**Answer: Yes, in a prespecified response-matched stress test.**

For each movie, repeats are divided into non-overlapping selection and test halves, with 28 trials in each. The selection half is used only to add train-independent amplitude noise to the best Dynamic predictions so that scalar response correlation matches Static; the test half is completely excluded from selection.

Selection response: Static `0.15553`, matched Dynamic `0.15520`. Held-out test response: Static `0.15651`, matched Dynamic `0.15687`; the per-neuron paired-bootstrap difference is `+0.00036`, with 95% interval `[-0.00510, +0.00537]`. The held-out response scores are therefore nearly identical under the predefined matching procedure. Because no formal equivalence test was performed, this result is not described as statistical indistinguishability.

However, the test-half frozen-GPFA results are:

| Metric | Static | Response-matched Dynamic | Advantage 95% interval |
|---|---:|---:|---:|
| Position correlation | 0.5130 | 0.6914 | +0.0522--+0.3872 |
| Normalized position RMSE | 0.8742 | 0.7292 | +0.0524--+0.2815 |
| Velocity cosine | 0.2892 | 0.4737 | +0.1514--+0.2151 |
| Speed correlation | 0.5171 | 0.5378 | -0.0342--+0.0658 |
| Acceleration cosine | 0.1450 | 0.3946 | +0.2125--+0.2807 |

Thus, when scalar response correlation is matched, trajectory evaluation still detects differences in position, velocity, and acceleration. This is a controlled response-matching result; it does not mean the two models were also forced to match on RSA/CKA. Test CKA/RSA still favor Dynamic.

In addition, the training history contains a validation-matched pair: Static validation=`0.1639558`, Dynamic epoch 65=`0.1638853`, differing by only `0.000071`. This pair also retains a trajectory advantage on oracle data, but its oracle responses are not as tightly matched as in the stress test above, so it is not treated as the strongest evidence for Q4.

## Q5: Does Temporal Ablation Produce Monotonic Degradation?

**Answer: Yes for the main trajectory-similarity metrics; normalized RMSE is not perfectly strict.**

The ablation simultaneously multiplies the off-center weights of all three learned temporal-convolution kernels by a retention factor, while leaving the center temporal slices, biases, spatial core, readout, and shifter unchanged.

| History retention | Position r | Velocity cosine | Speed r | Acceleration cosine | Norm. RMSE |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.7262 | 0.4985 | 0.5375 | 0.4525 | 0.7048 |
| 0.75 | 0.5528 | 0.4527 | 0.5126 | 0.4108 | 0.8527 |
| 0.50 | 0.1869 | 0.3048 | 0.3397 | 0.2862 | 1.0156 |
| 0.25 | -0.1295 | 0.0659 | 0.0323 | 0.0915 | 1.0836 |
| 0.00 | -0.1949 | -0.0675 | -0.0343 | -0.0291 | 1.0738 |

Position, velocity, speed, and acceleration decrease strictly monotonically as ablation severity increases, with severity–quality Spearman equal to `-1.0` for all four. Normalized RMSE improves slightly at the final step, from `1.0836` to `1.0738`, so it is not strictly monotonic; its overall Spearman is still `-0.9`.

It is therefore appropriate to state that “trajectory similarity exhibits monotonic degradation under graded temporal-history ablation,” while also noting the slight non-monotonicity of normalized error under the strongest ablation.

## Q6: Is Trajectory Information Not Fully Explained by Response Correlation, RSA, and CKA?

**Answer: Current evidence supports “not fully explained,” but does not establish mathematical independence from every possible definition of RSA/CKA.**

### Strict Counterexample: Time Reversal

After fully reversing the Dynamic prediction in time, the condition-average pattern remains unchanged. As a result, standard condition CKA is effectively identical: `0.84860060` vs `0.84860060`; condition RSA is exactly identical: `0.342857` vs `0.342857`.

However, the frozen-GPFA metrics change substantially:

| Metric | Original Dynamic | Time reversed |
|---|---:|---:|
| Position correlation | 0.7262 | 0.2033 |
| Velocity cosine | 0.4985 | 0.0720 |
| Speed correlation | 0.5375 | 0.1068 |
| Acceleration cosine | 0.4525 | 0.0440 |

This strictly demonstrates that standard condition-averaged RSA/CKA are insufficient to represent temporal order/direction. Q4 further shows that trajectory differences remain after scalar response correlation is matched. Thus, each of these three conventional summaries is individually insufficient as a sufficient statistic.

### Empirical Explanatory Power of an Expanded Conventional Battery

Across 40 temporal/non-temporal stress candidates, six conventional features—single-trial/average response, time-aware CKA, delta CKA, state RSA, and temporal RSA—are used to predict GPFA metrics in a held-out perturbation family:

| GPFA target | Leave-family-out R² |
|---|---:|
| Position | 0.819 |
| Velocity | 0.661 |
| Speed | 0.683 |
| Acceleration | 0.420 |
| RMSE quality | 0.733 |

These values show that an enriched RSA/CKA/response battery explains a substantial fraction of trajectory variation, but clearly does not predict it completely, especially for acceleration/direction. A cross-family conventional-matched pair selected using only selection repeats still shows GPFA position and RMSE differences on held-out repeats, further supporting the existence of residual trajectory information.

The most accurate statement is:

> Trajectory metrics contain temporal-order and local-direction information that is not fully captured by scalar response correlation or standard RSA/CKA, and remains only partially predictable from an enriched time-aware conventional metric battery.

This is not proof that trajectory metrics are statistically independent of all conventional metrics; rather, it is evidence of sufficiency failure and incremental information.

## Scope Limitations

- Q1 uses five sessions; the deeper Q2–Q6 analyses still use one session, a 512-neuron subset, and six repeated movies.
- Q4 is an amplitude-noise response-matching stress test; it addresses metric sensitivity rather than providing a new fair ranking of separately trained models.
- Q5 scales temporal-history weights in all three layers simultaneously, revealing a dose response but not identifying which layer or temporal lag is most important.
- The Q6 regression candidate set consists of controlled perturbations and does not represent the distribution of all possible models.

Complete machine-readable results: [`results/tables/04_model_comparison/q1_q6_answers.json`](../../results/tables/04_model_comparison/q1_q6_answers.json).
