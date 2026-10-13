# Temporal Metric Tests: Results

**Time-averaged condition-pattern scores from representational similarity analysis (RSA) and centred kernel alignment (CKA) remain unchanged when predictions are reversed. Both primary trajectory scores decrease for Static and reduced Dynamic.** Response correlation and several time-aware representation scores also fall. The contrast identifies a limitation of the chosen time-averaged summaries and shows that several measures detect the alteration.

## 1. Completed diagnostic and comparison scope

The supplementary diagnostic uses the frozen `retrained_v1` Static and reduced Dynamic encoders, the same main Gaussian-process factor analysis (GPFA), one pilot session, six movies, 58 test trials, 512 neurons and original frames 50–299. Each encoder's original output is compared with one complete reversal of its 250-frame prediction window. Recorded responses, movie labels, neuron order and reference timestamps are unchanged. No model is retrained or selected by this outcome.

The operation reverses **encoder-target-scaled predictions before the fixed response conversion and GPFA inference**. Video input and already inferred latent vectors remain in their original order. Because the conversion depends on time, it differs from the primary model-null reversal applied after conversion. The exact operations and their distinct scaling factors are given in [Setup](SETUP.md#4-reversal-and-the-order-of-scale-conversion).

This diagnostic was configured after the main results were known. It therefore supplies exploratory supplementary evidence. Withdrawn auxiliary GPFA sensitivity findings are not used. In the tables below, **Dynamic** denotes reduced Dynamic; the historical full-width baseline is a separate model.

![The same output-order operation leaves encoder-scale time-averaged patterns unchanged but changes trajectory and several time-aware response scores](../figures/documentation/04_reversal.png)

*The same output-order operation leaves encoder-scale time-averaged patterns unchanged but changes trajectory and several time-aware response scores. A has no sampling interval; B uses pooled trajectories; C–D use equal-weight movie means. All paired intervals come from 2,000 saved draws of the same six whole movies. Static temporal-difference CKA crosses zero; unchanged response accuracy is not a condition of this diagnostic.*

## 2. Time-averaged condition patterns remain unchanged

The six movie-average patterns retain their sums under reversal in encoder-target scale. Their RSA/CKA scores are therefore invariant up to the checked numerical tolerance, as shown in panel A.

| Model | Metric | Original | Reversed |
| --- | --- | ---: | ---: |
| Static | Time-averaged condition-pattern CKA | 0.837269 | 0.837269 |
| Static | Time-averaged condition-pattern RSA | 0.096429 | 0.096429 |
| Reduced Dynamic | Time-averaged condition-pattern CKA | 0.837879 | 0.837879 |
| Reduced Dynamic | Time-averaged condition-pattern RSA | 0.485714 | 0.485714 |

Direct numerical checks establish invariance beyond the displayed decimal precision. All **26 numerical checks pass**. The largest recorded errors for the relevant check types are:

| Check | Maximum absolute difference | Acceptance tolerance |
|---|---:|---:|
| Time-averaged condition patterns, original versus reversed | `9.99200722163e-16` | `1e-10` |
| Time-averaged condition-pattern RSA/CKA, original versus reversed | `2.22044604925e-16` | `1e-10` |
| Original-score replay versus the main table | `1.32405081543e-06` | `3e-6` |

The largest baseline replay difference occurs in response normalised mean squared error (NMSE). It satisfies the configured tolerance, with small differences from the main score table. The original scores below are the diagnostic's replayed values.

Pattern RSA has six movie patterns and **15 unique movie-pair edges**. The 11,175-edge support belongs to the separate time-state RSA. Pattern invariance is a deterministic numerical identity check; it carries no population confidence interval or randomisation p-value. Its evidence is recorded in [numerical checks](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/numerical_checks.csv).

## 3. Both primary trajectory scores decrease after reversal

All trajectory points and intervals below concern pooled condition-average trajectories in the same frozen four-dimensional GPFA space. Each difference is **reversed minus original**. Negative values indicate a loss of correlation or direction agreement; positive normalised position root mean squared error (RMSE) differences indicate increased error.

| Model | Metric | Original | Reversed | Reversed − original | 95% paired interval |
| --- | --- | ---: | ---: | ---: | --- |
| Static | Position correlation (primary) | 0.416545 | -0.038918 | -0.455464 | [-0.647840, -0.309468] |
| Static | Velocity-direction cosine (primary) | 0.259418 | 0.011964 | -0.247453 | [-0.341951, -0.138485] |
| Static | Normalised position RMSE | 0.922172 | 1.149317 | +0.227145 | [+0.140870, +0.344684] |
| Static | Speed-profile correlation | 0.534564 | 0.050146 | -0.484419 | [-0.795748, -0.180057] |
| Static | Acceleration-direction cosine | 0.133099 | -0.009921 | -0.143020 | [-0.251170, -0.033740] |
| Reduced Dynamic | Position correlation (primary) | 0.781569 | 0.220325 | -0.561244 | [-0.721720, -0.463420] |
| Reduced Dynamic | Velocity-direction cosine (primary) | 0.516995 | 0.067771 | -0.449224 | [-0.582176, -0.319223] |
| Reduced Dynamic | Normalised position RMSE | 0.647897 | 1.053897 | +0.406000 | [+0.294478, +0.562213] |
| Reduced Dynamic | Speed-profile correlation | 0.555709 | 0.126430 | -0.429278 | [-0.669782, -0.178683] |
| Reduced Dynamic | Acceleration-direction cosine | 0.428879 | 0.037301 | -0.391578 | [-0.540701, -0.229937] |

Both primary trajectory intervals are below zero for each model (panel B). The reported differences use the stored contrasts at their original precision.

All secondary intervals also indicate poorer agreement. This reversal contrast differs from the unresolved unperturbed-model speed comparison in [Goal 3](../03_model_comparison/RESULTS.md#3-pilot-trajectory-comparison).

Dynamic's reversed position score remains positive at 0.220325. Some trajectory correspondence therefore remains after reversal. These scores quantify latent states and their derivatives under the fixed offline posterior; physical object positions and neural velocities were not directly measured.

The intervals resample six whole movie conditions, paired between original and reversed outputs, 2,000 times with seed 42. For these trajectory metrics, pooled scores are recomputed on each sampled set. Points and intervals consequently share the same statistical target. The [paired record](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/paired_intervals.csv) provides the stored `b_minus_a` differences; one-unit discrepancies in the last displayed decimal can arise when subtracting the rounded points.

## 4. Response and time-aware measures also detect changes

### Pooled point estimates

These scores compare the original and reversed outputs with the same recorded response arrays, without passing through GPFA. Response r and NMSE pool trial/time data. The representation measures use their prescribed within-movie repeat means and temporal aggregation. No condition-mean bootstrap interval is attached to these pooled points.

| Model | Metric | Pooled original | Pooled reversed |
| --- | --- | ---: | ---: |
| Static | Response r | 0.145129 | 0.029812 |
| Static | Response NMSE | 0.978172 | 1.010840 |
| Static | Condition/time CKA | 0.456054 | 0.300225 |
| Static | Temporal-difference CKA | 0.028074 | 0.028030 |
| Static | Condition/time-state RSA | 0.424808 | 0.215712 |
| Reduced Dynamic | Response r | 0.185953 | 0.040124 |
| Reduced Dynamic | Response NMSE | 0.962457 | 1.006065 |
| Reduced Dynamic | Condition/time CKA | 0.510811 | 0.313278 |
| Reduced Dynamic | Temporal-difference CKA | 0.093155 | 0.064301 |
| Reduced Dynamic | Condition/time-state RSA | 0.544941 | 0.242231 |

Both models lose response correlation and increase response NMSE; condition/time CKA and time-state RSA also fall. Response accuracy is therefore not held constant.

Static pooled temporal-difference CKA is nearly unchanged. The within-movie interval below evaluates the direction of its corresponding movie-mean contrast.

### Equal-weight movie means and paired intervals

For conventional metrics, the saved paired analysis first scores each movie and then averages those scores equally. Its points differ from the pooled values above. Both points and intervals in the following table concern this **equal-weight mean movie score**, with the contrast again defined as reversed minus original.

| Model | Metric | Mean-movie original | Mean-movie reversed | Mean paired difference | 95% paired interval |
| --- | --- | ---: | ---: | ---: | --- |
| Static | Response r | 0.104139 | 0.002568 | -0.101571 | [-0.122533, -0.073325] |
| Static | Response NMSE | 0.984082 | 1.014458 | +0.030376 | [+0.021762, +0.038368] |
| Static | Condition/time CKA | 0.527109 | 0.365798 | -0.161311 | [-0.216368, -0.106960] |
| Static | Temporal-difference CKA | 0.094810 | 0.090682 | -0.004128 | [-0.030433, +0.030278] |
| Static | Condition/time-state RSA | 0.520570 | 0.236503 | -0.284068 | [-0.399761, -0.169532] |
| Reduced Dynamic | Response r | 0.131987 | 0.003037 | -0.128951 | [-0.152582, -0.103913] |
| Reduced Dynamic | Response NMSE | 0.964256 | 1.006002 | +0.041746 | [+0.034739, +0.048139] |
| Reduced Dynamic | Condition/time CKA | 0.593809 | 0.388964 | -0.204846 | [-0.263560, -0.137157] |
| Reduced Dynamic | Temporal-difference CKA | 0.200528 | 0.155323 | -0.045205 | [-0.071116, -0.014001] |
| Reduced Dynamic | Condition/time-state RSA | 0.629593 | 0.224189 | -0.405404 | [-0.504034, -0.319569] |

Panels C–D show resolved losses in response r, condition/time CKA and time-state RSA for both models. Response-NMSE intervals indicate increased error.

**Static temporal-difference CKA remains unresolved:** its interval [−0.030433, +0.030278] spans zero. This does not prove invariance. Dynamic’s corresponding interval is below zero; retaining time information does not guarantee resolution of every alteration.

Across the ten trajectory rows and ten conventional rows in the paired table, every row has **2,000 valid and zero undefined bootstrap draws**. These are repeated draws of the same six movie conditions, not new sessions or animals. The pooled scores come from [operation metrics](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/metrics.csv); the movie-mean estimates and intervals come from [paired intervals](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/paired_intervals.csv). The [per-condition conventional records](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/condition_metrics.csv) retain all six movie identities.

## 5. Evaluation support and relation to the primary nulls

The recorded common support contains:

| Quantity | Retained support |
|---|---:|
| Condition/time states | 1,500 |
| Velocity comparisons | 1,494 |
| Acceleration comparisons | 1,488 |
| Response neurons | 512 |
| Time-state RSA sampled states | 150 |
| Time-state RSA edges | 11,175 |
| Time-averaged condition-pattern RSA edges | 15 |

The time-state and time-averaged RSA supports are kept separate because they use different summaries. Undefined values remain `NA`, rather than being replaced by zero. The numerical checks and baseline replay support interpretation of the reported operation; they are not additional biological replications.

The main comparison also has [model-prediction null records](../evidence/04_model_comparison/retrained_v1/pilot/model_nulls.csv). Its four random null families use 200 draws per model and metric, while `time_reversal` uses one. Those nulls act in converted response space before posterior inference. This supplementary counterexample reverses predictions before the time-dependent conversion. Their results are related but **cannot be pooled as repeated executions of the same reversal experiment**.

## 6. Interpretation and scope

The evidence supports a specific answer to the research question. The tested time-averaged condition patterns preserve their values under prediction reversal, so their RSA and CKA summaries cannot distinguish the two prediction orders. Under the same defined output operation, the primary trajectory measures register lower agreement in latent state and local direction for both encoders. They describe aspects of temporal correspondence that these particular averaged patterns discard.

That conclusion is narrower than saying that response metrics or RSA/CKA are time blind. Response correlation falls, and the time-aware CKA and RSA variants detect several changes. Static temporal-difference CKA has an unresolved interval, distinct from the exact pattern invariance. **The choice of inputs and temporal aggregation, not the method name alone, determines which comparison is being made.**

The concurrent loss of response alignment can explain several score changes, leaving independence from the original response information unresolved. The separate [response-matching goal](../05_response_matching/RESULTS.md) addresses one specified scalar response summary under a different control; its conclusion should not be imported into this reversal table.

The order of processing is also part of the result. Reversal and the time-dependent scale conversion do not commute, so the changed trajectory scores concern the complete stated measurement pathway. The exact invariance concerns encoder-scale time-averaged patterns. Both claims concern output reversal at the specified processing stage; latent-vector order and the input stimulus remain fixed.

Finally, the GPFA is an offline measurement with no identified biological meaning for individual axes. This supplementary analysis uses one session, six historically explored movies and the frozen models from one training seed. It does not establish independent animal-level effects, a neural algorithm, an optimal latent representation or cross-setting robustness. Within that scope, the result supports retaining both response/time-aware summaries and explicit trajectory measures, with each operation stated clearly.

## 7. Result records

| Record | Content used here |
|---|---|
| [Complete operation scores](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/metrics.csv) | `model`, `operation`, all 12 metrics, `value` and support columns |
| [Paired intervals](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/paired_intervals.csv) | `contrast=reversed_minus_original`, `a`, `b`, `b_minus_a`, `ci_low`, `ci_high`, `estimand` and valid/undefined draw counts |
| [Per-condition conventional scores](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/condition_metrics.csv) | Movie-specific original and reversed scores for all six movie IDs |
| [Numerical checks](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/numerical_checks.csv) | `check`, `absolute_error`, `tolerance` and `passed` |
| [Diagnostic provenance](../evidence/04_model_comparison/retrained_v1_diagnostics/reversal/summary.json) | Deterministic operation, freeze/source/cache identities and `main_artifacts_unchanged` |
| [Primary model-null records](../evidence/04_model_comparison/retrained_v1/pilot/model_nulls.csv) | The separate after-conversion controls, their observed/null values and transformation counts |

[Setup](SETUP.md#7-source-records-and-interpretation-scope) links the configuration and source functions. The inline figure renders these saved operation scores and intervals; it adds no independently sampled animal/session effect.