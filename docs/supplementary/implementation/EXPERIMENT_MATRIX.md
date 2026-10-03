# Experiment Matrix

This matrix records Study 1 experimental coverage as of 14 August 2026. The completed Study 2 is documented separately in [Mental-Pong](../../MENTAL_PONG.md). `Complete` means that the corresponding analysis has been run and its compact outputs are preserved in the repository. `Supporting` denotes a completed secondary control that is not the primary evidence. `Not run` identifies an analysis not included in the current public evidence, and `N/A` means that the evaluation is not scientifically applicable to that row.

## Official Factorized3D Dynamic baseline (`best.pt`)

| Field | Coverage |
|---|---|
| Parameters | 5,707,743 |
| Temporal information available to the evaluated output | Learned temporal context through the full-width Factorized3D core. |
| Training, fitting, and evaluation data scope | Trained on the five Dynamic Sensorium 2023 sessions. Evaluated locally on the full-sequence oracle tiers from all five sessions. Hidden `final_test_main` labels are unavailable locally. |
| Five-session response | **Complete:** local oracle correlation; exact hidden-test reproduction is not verifiable. |
| Detailed response | **Not run:** not included in the one-session deep comparison. |
| Output-space RSA / CKA | **Not run.** |
| Frozen-GPFA trajectory | **Not run.** |
| Reliability / null validation | **N/A:** the reliability assay was developed independently of this benchmark. |
| Response-score matching | **N/A.** |
| Temporal ablation | **Not run.** |

## Static-on-Dynamic baseline (`best.pt`)

| Field | Coverage |
|---|---|
| Parameters | 2,814,015 |
| Temporal information available to the evaluated output | Current-frame visual processing only; the 2D core has no learned temporal history. |
| Training, fitting, and evaluation data scope | Trained on the same five sessions as the Dynamic models. Five-session full-sequence oracle evaluation is complete. Deep evaluation uses one session, 58 oracle trials from six movies, 512 fixed neurons, and frames 50-299 (250 timestamps). |
| Five-session response | **Complete.** |
| Detailed response | **Complete:** neuron-wise, population-vector, temporal-difference, lag, error, variance, and scale diagnostics. |
| Output-space RSA / CKA | **Complete:** condition, time-resolved, temporal, condition-by-time RSA, and four linear-CKA variants. |
| Frozen-GPFA trajectory | **Complete:** position, normalized error, velocity, speed, acceleration, lag, and descriptive path metrics. |
| Reliability / null validation | **Complete:** model-specific temporal nulls; neural split-half reliability is assay-level and listed below. |
| Response-score matching | **Complete as the target/reference** for the primary and supporting matching controls. |
| Temporal ablation | **N/A:** there are no off-center temporal kernels to ablate. |

## Total-parameter-matched Dynamic (`best.pt`)

| Field | Coverage |
|---|---|
| Parameters | 2,862,063 |
| Temporal information available to the evaluated output | Three-stage reduced-width Factorized3D core with learned temporal convolutions; 98,672 core parameters. Total count is 1.71% above Static because each model has a 2,763,106-parameter readout that dominates its total; learned readout weights are separate. |
| Training, fitting, and evaluation data scope | Trained on the same five sessions, splits, inputs, behaviors, loss, and readout family as Static. Static retains a different four-layer 2D core with 50,624 parameters, so the control matches total count rather than core architecture or computation. Evaluated on all five oracle sessions for response and on the same one-session `58 x 250 x 512` tensor for the deep comparison. |
| Five-session response | **Complete.** |
| Detailed response | **Complete:** same battery as Static. |
| Output-space RSA / CKA | **Complete:** same output-space battery as Static. |
| Frozen-GPFA trajectory | **Complete:** same frozen GPFA and coordinates as Static; no model-specific refit or alignment. |
| Reliability / null validation | **Complete:** model-specific temporal nulls; neural split-half reliability is assay-level and listed below. |
| Response-score matching | **Complete as the source model** for the held-out response-score-matched output perturbation. |
| Temporal ablation | **Complete as the source model:** five retention levels were evaluated. |

## Validation-matched Dynamic, epoch 65 (`epoch_65_validation_matched.pth`)

| Field | Coverage |
|---|---|
| Parameters | 2,862,063 |
| Temporal information available to the evaluated output | Learned temporal context is intact; this is an earlier naturally trained checkpoint, not an output perturbation. |
| Training, fitting, and evaluation data scope | Selected only from the recorded five-session validation history (`0.1638853` versus Static `0.1639558`). Its aligned predictions were evaluated on the same one-session `58 x 250 x 512` oracle tensor. |
| Five-session response | **Supporting:** validation matching is complete; a separate five-session oracle benchmark is not reported. |
| Detailed response | **Complete** on the one-session tensor. |
| Output-space RSA / CKA | **Complete** on the one-session tensor. |
| Frozen-GPFA trajectory | **Complete** on the one-session tensor. |
| Reliability / null validation | **Not run** for this checkpoint. |
| Response-score matching | **Supporting:** this checkpoint is a validation-matched model control, not the response-score-matched output perturbation used for Q4. |
| Temporal ablation | **Not run.** |

## Response-score-matched output perturbation

| Field | Coverage |
|---|---|
| Parameters | 2,862,063 underlying model parameters; no weights changed |
| Temporal information available to the evaluated output | Learned temporal kernels remain intact, but the predicted output is degraded with fixed Gaussian noise scaled by each neuron's Dynamic prediction standard deviation. |
| Training, fitting, and evaluation data scope | No retraining. Within the six oracle movies, 28 trials pooled across the movies select the noise amplitude and a disjoint pooled 28 form the test half excluded from amplitude selection; 512 neurons and 250 timestamps are retained. |
| Five-session response | **N/A:** this is a one-session output-space stress test, not a trained five-session benchmark. |
| Detailed response | **Complete:** test-half mean neuron-response point estimates are close under the selected perturbation; no equivalence test was performed. |
| Output-space RSA / CKA | **Complete:** evaluated but deliberately not forced to match. |
| Frozen-GPFA trajectory | **Complete:** held-out predictions enter the same frozen GPFA posterior inference. |
| Reliability / null validation | **Not run.** |
| Response-score matching | **Complete:** primary Q4 stress test. |
| Temporal ablation | **N/A.** |

## Temporally ablated Dynamic family (retention `1.00, 0.75, 0.50, 0.25, 0.00`)

| Field | Coverage |
|---|---|
| Parameters | 2,862,063 stored parameters at every level; effective off-center weight magnitude changes |
| Temporal information available to the evaluated output | Graded learned history: all off-center temporal weights are multiplied by retention, while center slices, biases, spatial kernels, readout, and shifter remain fixed. Retention `0.00` removes learned off-center temporal history. |
| Training, fitting, and evaluation data scope | Derived from the best total-parameter-matched checkpoint without retraining. Every level is evaluated on the same one-session 58 oracle trials, six movies, 512 neurons, and 250 timestamps. |
| Five-session response | **Not run:** current ablation inference is limited to the one-session pilot. |
| Detailed response | **Complete** at all five levels. |
| Output-space RSA / CKA | **Complete:** condition-average CKA and condition-by-time RSA at all five levels. |
| Frozen-GPFA trajectory | **Complete:** the full trajectory battery and condition bootstrap at all five levels. |
| Reliability / null validation | **Not run** as a separate split-half or model-null series. |
| Response-score matching | **N/A.** |
| Temporal ablation | **Complete:** four similarity metrics degrade strictly monotonically; normalized RMSE contains one small non-monotonic step. |

## Neural-data-defined GPFA assay (not an encoding model)

| Field | Coverage |
|---|---|
| Parameters | N/A as a neural-network count; latent dimension `q = 4` |
| Temporal information available to the evaluated output | A continuous Gaussian-process temporal prior and shared neural observation model learned only from recorded population activity. |
| Training, fitting, and evaluation data scope | One session. Dimension and initialization use neural training/calibration data only. The final comparison GPFA is refitted on a locked 174-of-348 training-trial subset and frozen before evaluation. Oracle validation uses 58 trials from six movies; the primary observation grid is 7.5 Hz-equivalent and the posterior is queried at 30 Hz. The offline posterior uses observations throughout the supplied time window. |
| Five-session response | **N/A.** |
| Detailed response | **N/A.** |
| Output-space RSA / CKA | **N/A.** |
| Frozen-GPFA trajectory | **Complete:** frozen posterior inference is used for brain, Static, and Dynamic observations. |
| Reliability / null validation | **Complete:** 200 balanced split halves, matched nulls, model-specific nulls, and 16 sensitivity profiles. |
| Response-score matching | **N/A.** |
| Temporal ablation | **N/A.** |


The oracle tier also informed encoding-model checkpoint selection. Q4 excludes its test half from noise-amplitude selection, not from that earlier checkpoint selection. The saved Q4 result retains pooled counts but not individual half-membership indices; [Methods](../../METHODS.md#141-response-matching-stress-test) gives the selection procedure and scope.

The five-session result should therefore be interpreted as a response benchmark. The broader output-space RSA/CKA, GPFA trajectory, response-score-matched output perturbation, and temporal-ablation conclusions remain a one-session proof of concept.

Detailed definitions and evidence are available in [Methods](../../METHODS.md), [Results](../../RESULTS.md), [GPFA Validation](../../GPFA_VALIDATION.md), and [Q1-Q6 Answers](../../results/Q1_Q6_ANSWERS.md).
