# Results

Sections 1–8 report Study 1, Dynamic Sensorium. They proceed from five-session response prediction to population-response similarity and trajectory agreement in a frozen neural-data-defined GPFA space. Stress tests examine which temporal properties the selected metrics capture. Section 9 summarizes the separate Mental-Pong study; scores and validation evidence are not pooled across datasets.

## 1. Evaluation scope and model comparison

The primary comparison approximately matches total trainable parameter count while retaining two different model architectures:

| Model | Core architecture | Core parameters | Total trainable parameters |
|---|---|---:|---:|
| Static | Four-layer framewise 2D core | 50,624 | 2,814,015 |
| Total-parameter-matched Dynamic | Three-stage Factorized3D core with explicit temporal convolutions | 98,672 | 2,862,063 |

The total-count difference is 48,048 parameters, or 1.707%. The readout dominates both totals. Core counts and architectures remain different, so this approximately matched complete-model comparison cannot isolate temporal history as the cause of a performance difference.

Two evidence scales are kept distinct:

- **Five-session response comparison:** all five official Dynamic Sensorium sessions and 40,034 recorded neurons contribute to the response-level result.
- **Detailed pilot analysis:** one session, a deterministic 512-neuron subset, 58 oracle trials grouped into six repeated natural-movie conditions, and original frames 50–299 (250 timestamps). Condition repeat counts are 10/10/9/10/9/10.

Thus, the response result is supported across five sessions, whereas the RSA/CKA, trajectory, and stress-test evidence is currently a one-session proof of concept.

## 2. Response-level comparison

The first result is a five-session gain in locally evaluable oracle response correlation:

| Model | Mean single-trial oracle correlation |
|---|---:|
| Static | 0.164408 |
| Total-parameter-matched Dynamic | 0.187525 |
| Dynamic − Static | **+0.023117** |

This corresponds to an approximately 14.1% relative improvement. Dynamic is higher in every session, with sessionwise differences of `+0.0270`, `+0.0265`, `+0.0209`, `+0.0086`, and `+0.0325`. Resampling the five session-level differences gives a 95% interval of `[+0.0147, +0.0291]`. Because the inferential unit is the session and only five sessions are available, the consistency of direction is clearer than the strength of a broad population-level inference.

These values are local oracle correlations. Static and reduced Dynamic checkpoints were selected using correlation on this same oracle tier; complete-sequence evaluation reuses those observations. The scores therefore describe the selected models on reused oracle data. See the [model-specific data-use table](METHODS.md#data-use-and-evaluation-independence). The official Dynamic reference uses hidden `final_test_main` labels that cannot be re-evaluated locally, so the hidden server score and the local oracle result are treated as different evaluation settings rather than as directly comparable numerical reproductions.

## 3. Conventional population-response metrics

RSA and CKA here compare predicted neural population responses with recorded population responses; they do not compare hidden neural-network representations. The most informative differences occur in variants that preserve within-movie temporal organization:

| Output-space metric | Static | Dynamic | Dynamic − Static | Condition-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Within-condition temporal CKA | 0.5131 | 0.5985 | +0.0854 | +0.0304–+0.1655 |
| Temporal-difference CKA | 0.0991 | 0.2027 | +0.1036 | +0.0921–+0.1154 |
| Within-condition temporal RSA | 0.5535 | 0.6427 | +0.0892 | +0.0308–+0.1930 |

Other time-resolved and condition-by-time state measures show the same direction. RSA and CKA detect the model difference most clearly when temporal structure is retained. These tested variants provide the conventional baseline for assessing complementary trajectory sensitivity.

## 4. Frozen neural-data-defined GPFA trajectory comparison

The GPFA coordinate system is fitted only to neural training data and its reliability is validated before model comparison. Train-only preprocessing, latent coordinates, and temporal priors are frozen; recorded responses, Static predictions, and Dynamic predictions then enter the same posterior-inference procedure. No model-specific GPFA refit, rotation, scaling, Procrustes alignment, or latent-axis selection is applied.

| Trajectory metric | Static | Dynamic | Oriented Dynamic-advantage 95% interval |
|---|---:|---:|---:|
| Position correlation | 0.5203 | 0.7262 | +0.0883–+0.4513 |
| Normalized position RMSE | 0.8705 | 0.7048 | +0.0717–+0.3457 |
| Velocity-direction cosine | 0.3045 | 0.4985 | +0.1336–+0.2486 |
| Speed-profile correlation | 0.5271 | 0.5375 | −0.0513–+0.0859 |
| Acceleration-direction cosine | 0.1850 | 0.4525 | +0.1822–+0.3395 |

Dynamic shows substantially stronger agreement in trajectory position, normalized position error, local velocity direction, and acceleration direction. Acceleration-direction cosine measures agreement in how velocity changes, including changes in speed; it is not a direct curvature measurement. The speed-profile difference is inconclusive because its interval crosses zero; the evidence does not support a Dynamic advantage on every trajectory metric.

Static predictions also exceed the sampled model-prediction nulls for the primary trajectory metrics. Both models therefore capture recorded trajectory structure, with stronger Dynamic agreement on several measured properties.

The reliability evidence and metric restrictions are documented in [GPFA Validation](GPFA_VALIDATION.md).

## 5. Stress tests: does trajectory evaluation add information?

### 5.1 Response-matching stress test

The response-score-matched output perturbation asks whether trajectory metrics still distinguish the predictions when their scalar response correlations are nearly matched. The selection half chooses the noise amplitude. Test-half neural responses are excluded from amplitude selection, but both halves belong to the oracle tier already used for encoding-model checkpoint selection.

On the held-out repeat half:

| Output | Mean response correlation |
|---|---:|
| Static | 0.15651 |
| Response-score-matched Dynamic output | 0.15687 |
| Dynamic − Static | +0.00036 |

The paired-bootstrap interval for the response difference is `[-0.00510, +0.00537]`. The test-half point estimates are close under the selected perturbation. No equivalence margin or formal equivalence test was used.

Despite the close response scores, frozen-GPFA position remains higher for the perturbed Dynamic output (`0.5130 → 0.6914`), as do velocity direction (`0.2892 → 0.4737`) and acceleration direction (`0.1450 → 0.3946`). The speed comparison remains inconclusive. Scalar response correlation can therefore be nearly matched while substantial trajectory-position and local-direction differences remain.

This is a metric-sensitivity stress test applied to model outputs, not a fair ranking of a newly trained response-matched model.

### 5.2 Graded temporal-weight attenuation

The attenuation intervention progressively scales learned off-center temporal-convolution weights at retention levels `1.00`, `0.75`, `0.50`, `0.25`, and `0.00`, while leaving the center temporal slices and remaining model components fixed.

Position, velocity-direction, speed-profile, and acceleration-direction similarity decrease strictly monotonically as history retention decreases. Normalized RMSE follows the overall degradation but is not perfectly monotonic at the strongest attenuation: it changes from `1.0836` to `1.0738` at the final step. The complete curve remains in the detailed evidence ledger.

The appropriate interpretation is: **consistent with a graded temporal-history effect, but temporal specificity remains to be controlled.** The intervention does not cleanly isolate temporal computation or establish that temporal history alone causes the full model difference.

### 5.3 Time reversal and incremental information

Full time reversal provides a strict counterexample for the tested time-averaged condition-pattern RSA/CKA summaries. It preserves the time-averaged condition patterns, leaving condition CKA and RSA unchanged, while strongly disrupting trajectory agreement:

| Metric | Original Dynamic | Time-reversed Dynamic |
|---|---:|---:|
| Condition CKA | 0.84860060 | 0.84860060 |
| Condition RSA | 0.342857 | 0.342857 |
| GPFA position correlation | 0.7262 | 0.2033 |
| GPFA velocity cosine | 0.4985 | 0.0720 |
| GPFA speed correlation | 0.5375 | 0.1068 |
| GPFA acceleration cosine | 0.4525 | 0.0440 |

This demonstrates that the tested time-averaged condition-pattern RSA/CKA summaries omit temporal order and direction. It does not imply that trajectory metrics are mathematically independent of RSA/CKA.

An enriched conventional battery was also used to predict GPFA metrics across held-out perturbation families:

| GPFA target | Leave-family-out R² |
|---|---:|
| Position | 0.819 |
| Velocity | 0.661 |
| Speed | 0.683 |
| Acceleration | 0.420 |
| RMSE quality | 0.733 |

The fitted ridge model predicts a substantial fraction of trajectory variation in perturbation families excluded from regression fitting, with lower predictive performance for acceleration than position. Its remaining errors establish the limits of this regression on the tested perturbations. They do not by themselves establish that the feature battery lacks the information: finite samples, model capacity, and distribution changes can also produce residuals. The time-reversal counterexample provides separate evidence that time-averaged condition patterns omit order and direction.

## 6. Integrated interpretation

The evidence forms a coherent sequence:

1. Total-parameter-matched Dynamic shows a consistent five-session response-level advantage over Static.
2. Time-aware output-space RSA and CKA also detect the Dynamic–Static difference, so trajectory evaluation is not being compared with an artificially weak conventional baseline.
3. Frozen neural-data-defined GPFA reveals especially strong differences in trajectory position and local direction, while speed-profile evidence remains uncertain.
4. Nearly matching scalar response correlation does not eliminate the position, velocity, and acceleration differences.
5. Time reversal exposes a concrete temporal-order limitation of time-averaged condition-pattern RSA/CKA.
6. Graded temporal-weight attenuation produces a graded trajectory response consistent with sensitivity to learned temporal history, without uniquely isolating temporal causality.

Together, the response-matching and reversal tests show temporal-order and local-direction sensitivity that the particular matched or reversal-invariant summaries do not capture. Time-aware conventional measures also detect model differences and predict substantial trajectory-score variation. Trajectory evaluation adds a diagnostic within this scope; response prediction and population-response similarity remain separate evaluation criteria.

## 7. Evidence scope and interpretation boundaries

- Response-level evidence covers five sessions and 40,034 neurons; detailed RSA/CKA and trajectory evidence covers one session, 512 neurons, and six repeated movies.
- Encoding-model training uses a single seed.
- Total-parameter matching applies to total trainable parameters, not to identical cores or core parameter counts. Static and Dynamic are different complete architectures.
- Speed-profile evidence is weaker than the position-, velocity-, and acceleration-related evidence.
- The GPFA represents a reproducible shared low-dimensional subspace of neural activity, not the complete neural response.
- GPFA latent dimension and timescales are analysis parameters and are not interpreted as unique biological dimensions or time constants.
- The 30 Hz derivative metrics describe the inferred continuous-time GPFA posterior rather than independently observed 30 Hz neural dynamics.
- Temporal-weight attenuation demonstrates graded sensitivity but does not uniquely isolate a temporal causal effect.

These boundaries limit the breadth and causal interpretation of the result without changing the proof-of-concept finding that trajectory evaluation adds a useful temporal diagnostic.

## 8. Detailed evidence and supporting documents

- [Detailed Q1–Q6 evidence](results/Q1_Q6_ANSWERS.md): canonical question-by-question statistics, intervals, controls, and qualifications.
- [Methods](METHODS.md): data preparation, temporal alignment, response/RSA/CKA definitions, GPFA inference, reliability procedures, and stress-test implementation.
- [GPFA Validation](GPFA_VALIDATION.md): reliability, null separation, sensitivity, negative findings, and metric restrictions.
- [Design Rationale](DESIGN_RATIONALE.md): study design, comparison-space rationale, and falsification logic.

## 9. Study 2: Mental-Pong

The completed [Mental-Pong exploration](MENTAL_PONG.md) compares position reconstructed from condition-mean DMFC responses with an objective ball path and a behavior-constrained candidate path. FA50 and GPFA50 use matched ordinary least squares (OLS) readouts and the same condition splits. This task-coordinate comparison is separate from Sensorium's shared-space neural trajectory comparison.

Each readout is first scored against the target it was trained to reconstruct. All four animal-by-representation comparisons over the full evaluated interval favor the objective target on these correlation and RMSE scores. No-collision segments have lower candidate RMSE and lower candidate correlation; post-collision segments favor the objective target on own-target RMSE. Cross-scoring distinguishes a change of target from a change of trained readout. The [versioned final report and tables](../experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_REPORT.md) retain these phase differences and the complete two-by-two comparisons.

The condition-mismatch control keeps each readout fixed and assigns its predictions to other conditions. Matched and mismatched scores use the same valid time points. Random-endpoint controls preserve condition geometry and fit new readouts to each randomized target. The actual candidate exceeds the random-endpoint reference in full, hidden, and endpoint-influence evaluations, with post-collision RMSE exceptions. Neural results over the full evaluated interval remain below the mean-endpoint geometry baseline, while some Perle phase-specific skills are slightly positive. These controls support recoverable task-position structure within the available data scope. They do not establish an overall candidate-path advantage or differences between opposite behavioral choices within the same condition.

The scientific status remains `CLOSED_EXPLORATORY_WITH_LIMITATIONS` and `raw_preprocessing=fail`. Filtering and fit-source checks pass within the published inputs; unresolved upstream filling across conditions and times remains outside those checks. The unknown membership of the neural means also prevents a direct test of neural information lost through behavioral averaging. The [current future plan](FUTURE_DIRECTIONS.md) specifies the paired records needed for that question.
