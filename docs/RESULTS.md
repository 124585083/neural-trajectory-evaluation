# Results

The results proceed from response-level prediction across five Dynamic Sensorium sessions to output-space population-response similarity and, finally, to trajectory agreement in a frozen neural-data-defined GPFA space. The final stress tests examine whether the trajectory findings are fully summarized by scalar response correlation or the tested RSA/CKA measures.

## 1. Evaluation scope and model comparison

The primary comparison approximately matches total trainable parameter count while retaining two different model architectures:

| Model | Core architecture | Core parameters | Total trainable parameters |
|---|---|---:|---:|
| Static | Four-layer framewise 2D core | 50,624 | 2,814,015 |
| Total-parameter-matched Dynamic | Three-stage Factorized3D core with explicit temporal convolutions | 98,672 | 2,862,063 |

The total-count difference is 48,048 parameters, or 1.707%. The readout dominates both totals, and the core parameter counts are not matched. This is therefore a complete-model architectural comparison under approximately matched total trainable parameter counts—not a same-backbone comparison, a core-parameter-matched comparison, or an isolation of temporal history as the only architectural difference.

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

These values are local oracle correlations. The official Dynamic reference uses hidden `final_test_main` labels that cannot be re-evaluated locally, so the hidden server score and the local oracle result are treated as different evaluation settings rather than as directly comparable numerical reproductions.

## 3. Conventional population-response metrics

RSA and CKA here compare predicted neural population responses with recorded population responses; they do not compare hidden neural-network representations. The most informative differences occur in variants that preserve within-movie temporal organization:

| Output-space metric | Static | Dynamic | Dynamic − Static | Condition-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Within-condition temporal CKA | 0.5131 | 0.5985 | +0.0854 | +0.0304–+0.1655 |
| Temporal-difference CKA | 0.0991 | 0.2027 | +0.1036 | +0.0921–+0.1154 |
| Within-condition temporal RSA | 0.5535 | 0.6427 | +0.0892 | +0.0308–+0.1930 |

Other time-resolved and condition-by-time state measures show the same direction. The result is not that RSA or CKA fail to detect the model difference. Rather, they detect it most clearly when temporal structure is retained, providing a substantive conventional baseline against which to evaluate the added sensitivity of trajectory metrics.

## 4. Frozen neural-data-defined GPFA trajectory comparison

The GPFA coordinate system is fitted only to neural training data and its reliability is validated before model comparison. Train-only preprocessing, latent coordinates, and temporal priors are frozen; recorded responses, Static predictions, and Dynamic predictions then enter the same posterior-inference procedure. No model-specific GPFA refit, rotation, scaling, Procrustes alignment, or latent-axis selection is applied.

| Trajectory metric | Static | Dynamic | Oriented Dynamic-advantage 95% interval |
|---|---:|---:|---:|
| Position correlation | 0.5203 | 0.7262 | +0.0883–+0.4513 |
| Normalized position RMSE | 0.8705 | 0.7048 | +0.0717–+0.3457 |
| Velocity-direction cosine | 0.3045 | 0.4985 | +0.1336–+0.2486 |
| Speed-profile correlation | 0.5271 | 0.5375 | −0.0513–+0.0859 |
| Acceleration-direction cosine | 0.1850 | 0.4525 | +0.1822–+0.3395 |

Dynamic shows substantially stronger agreement in trajectory position, normalized position error, local velocity direction, and acceleration/curvature-related direction. The speed-profile difference is inconclusive because its interval crosses zero; the evidence does not support a Dynamic advantage on every trajectory metric.

Static predictions also exceed the sampled model-prediction nulls for the primary trajectory metrics. Static therefore captures nontrivial recorded dynamic structure; the comparative result is that Dynamic more closely matches several aspects of the neural trajectory, not that Static contains no trajectory structure.

The reliability evidence and metric restrictions are documented in [GPFA Validation](GPFA_VALIDATION.md).

## 5. Stress tests: does trajectory evaluation add information?

### 5.1 Response-matching stress test

The response-score-matched output perturbation asks whether trajectory metrics still distinguish the predictions when their scalar response correlations are nearly matched. Perturbation strength is selected on one repeat half; test-half neural responses are not used to select that strength.

On the held-out repeat half:

| Output | Mean response correlation |
|---|---:|
| Static | 0.15651 |
| Response-score-matched Dynamic output | 0.15687 |
| Dynamic − Static | +0.00036 |

The paired-bootstrap interval for the response difference is `[-0.00510, +0.00537]`. The scores are nearly matched under this interval, but no formal equivalence test was performed and they are not described as statistically equivalent.

Despite the close response scores, frozen-GPFA position remains higher for the perturbed Dynamic output (`0.5130 → 0.6914`), as do velocity direction (`0.2892 → 0.4737`) and acceleration direction (`0.1450 → 0.3946`). The speed comparison remains inconclusive. Scalar response correlation can therefore be nearly matched while substantial trajectory-position and local-direction differences remain.

This is a metric-sensitivity stress test applied to model outputs, not a fair ranking of a newly trained response-matched model.

### 5.2 Graded temporal-weight attenuation

The attenuation intervention progressively scales learned off-center temporal-convolution weights at retention levels `1.00`, `0.75`, `0.50`, `0.25`, and `0.00`, while leaving the center temporal slices and remaining model components fixed.

Position, velocity-direction, speed-profile, and acceleration-direction similarity decrease strictly monotonically as history retention decreases. Normalized RMSE follows the overall degradation but is not perfectly monotonic at the strongest attenuation: it changes from `1.0836` to `1.0738` at the final step. The complete curve remains in the detailed evidence ledger.

The appropriate interpretation is: **consistent with a graded temporal-history effect, but temporal specificity remains to be controlled.** The intervention does not cleanly isolate temporal computation or establish that temporal history alone causes the full model difference.

### 5.3 Time reversal and incremental information

Full time reversal provides a strict counterexample for standard condition-averaged representational metrics. It preserves the time-averaged condition patterns, leaving condition CKA and RSA unchanged, while strongly disrupting trajectory agreement:

| Metric | Original Dynamic | Time-reversed Dynamic |
|---|---:|---:|
| Condition CKA | 0.84860060 | 0.84860060 |
| Condition RSA | 0.342857 | 0.342857 |
| GPFA position correlation | 0.7262 | 0.2033 |
| GPFA velocity cosine | 0.4985 | 0.0720 |
| GPFA speed correlation | 0.5375 | 0.1068 |
| GPFA acceleration cosine | 0.4525 | 0.0440 |

This demonstrates that standard condition-averaged RSA/CKA are insufficient to encode temporal order and direction. It does not imply that trajectory metrics are mathematically independent of RSA/CKA.

An enriched conventional battery was also used to predict GPFA metrics across held-out perturbation families:

| GPFA target | Leave-family-out R² |
|---|---:|
| Position | 0.819 |
| Velocity | 0.661 |
| Speed | 0.683 |
| Acceleration | 0.420 |
| RMSE quality | 0.733 |

The response/RSA/CKA battery explains a substantial fraction of trajectory variation, but not all of it, especially for acceleration and local directional information. This supports a sufficiency failure and incremental-information interpretation: trajectory metrics are partially predictable from conventional metrics while retaining temporal-order and local-direction sensitivity that the tested battery does not fully capture.

## 6. Integrated interpretation

The evidence forms a coherent sequence:

1. Total-parameter-matched Dynamic shows a consistent five-session response-level advantage over Static.
2. Time-aware output-space RSA and CKA also detect the Dynamic–Static difference, so trajectory evaluation is not being compared with an artificially weak conventional baseline.
3. Frozen neural-data-defined GPFA reveals especially strong differences in trajectory position and local direction, while speed-profile evidence remains uncertain.
4. Nearly matching scalar response correlation does not eliminate the position, velocity, and acceleration differences.
5. Time reversal exposes a concrete temporal-order limitation of condition-averaged RSA/CKA.
6. Graded temporal-weight attenuation produces a graded trajectory response consistent with sensitivity to learned temporal history, without uniquely isolating temporal causality.

Taken together, the results suggest that trajectory metrics contain temporal-order and local-direction information that is not fully captured by scalar response correlation or the tested RSA/CKA battery. Trajectory evaluation is an additional diagnostic, not a replacement for response prediction or conventional population-response similarity.

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
