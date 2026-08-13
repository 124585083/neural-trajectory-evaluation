# Phase 4 — Comparing Static and Dynamic models beyond response correlation

**Research question:** Does a parameter-matched Dynamic model reproduce neural population dynamics better than a Static model, and do trajectory metrics reveal temporal differences not fully summarized by response correlation or output-space RSA/CKA?

## Why this phase exists

Phases 1-3 establish the two prerequisites for a controlled comparison: the encoding pipelines work, the trajectory assay is reliable on neural repeats, and Dynamic capacity is approximately matched to Static. Phase 4 brings these components together without allowing either encoding model to define the latent space.

The comparison has three levels, all applied to the same predicted neural-response outputs:

1. response accuracy, including neuron-wise, population-vector, temporal-difference, and lag diagnostics;
2. output-space population-response geometry, measured by time-aware and condition-level RSA/CKA; and
3. brain-defined GPFA trajectories, measured through position, normalized error, velocity direction, speed profile, and acceleration direction.

RSA and CKA in this phase compare predicted neural population responses with recorded neural population responses. They are **output-space analyses**, not hidden-layer representational analyses.

## Locked proof-of-concept scope

The deep comparison uses one Dynamic Sensorium session, 512 deterministic neurons, original frames 50-299, six repeated natural movies, and all 58 oracle trials. The model-comparison GPFA is fitted on a locked 174-of-348 subset of neural training trials, using 139 trials for fitting and 35 for calibration before the selected four-dimensional model is refitted on all 174 trials.

Oracle responses and encoding-model predictions do not enter GPFA preprocessing, dimensionality selection, initialization selection, or fitting. Once fitted, the observation model, temporal priors, neuron order, scaling, and latent axes are frozen. Static and Dynamic predictions are treated as new observations and transformed by the same GPFA posterior inference; no model-specific refit, rotation, scaling, or Procrustes alignment is allowed.

The five-session response result remains separate from this one-session deep analysis.

## Current result

| Question | Current answer | Main evidence |
|---|---|---|
| Does Dynamic have a response-level gain? | **Yes.** | Five-session oracle correlation improves by +0.0231, with a positive difference in every session. |
| Do output-space RSA/CKA detect a difference? | **Yes.** | Time-aware population-response RSA and CKA favor Dynamic; purely condition-averaged differences are smaller. |
| Does trajectory evaluation detect a difference? | **Yes.** | Dynamic improves GPFA position and direction-sensitive metrics; speed-profile evidence is weaker. |
| Does a trajectory difference remain after response matching? | **Yes, in the current stress test.** | Held-out mean neuron response is matched within the predefined procedure, while position, velocity, and acceleration still favor Dynamic. |
| Do trajectory metrics degrade under temporal ablation? | **Mostly monotonically.** | Position, velocity, speed, and acceleration decrease strictly across five history-retention levels; normalized RMSE has one small non-monotonic step. |
| Is trajectory information fully explained by response/RSA/CKA? | **Current evidence suggests not fully.** | Time reversal, response matching, and held-out perturbation-family prediction leave residual temporal-order and local-direction information. |

For the intact models, GPFA position correlation improves from **0.5203** to **0.7262**, and velocity-direction cosine from **0.3045** to **0.4985**. Acceleration direction also improves strongly, while speed-profile uncertainty overlaps zero. These metrics are reported separately rather than collapsed into a composite score.

## Critical controls

### Revalidated neural reliability

Before ranking models, this phase repeats split-half reliability, null testing, and sensitivity analysis using the subset-trained comparison GPFA. This prevents the model result from relying only on the earlier full-training-tier assay.

### Held-out response matching

Oracle repeats are divided into disjoint selection and test halves. Fixed Gaussian noise, scaled separately by each neuron's predicted Dynamic range, is tuned only on the selection half to match Static's scalar mean-neuron response score. The held-out test then recomputes response, output-space RSA/CKA, and frozen-GPFA metrics. This is a metric-sufficiency stress test, not a new fair-training leaderboard and not a formal equivalence test.

### Temporal-history ablation

All off-center temporal-convolution weights are multiplied by retention values `1.00, 0.75, 0.50, 0.25, 0.00`, while center slices, spatial kernels, biases, readout, and shifter remain fixed. This produces a graded test of sensitivity to learned temporal history.

A magnitude-matched non-temporal weight-damage control has not yet been run. Until it is added, the dose-response result supports temporal sensitivity but cannot completely exclude a contribution from generic progressive network damage.

## Interpretation and limits

The current evidence suggests that trajectory evaluation contributes information beyond scalar response correlation and the tested output-space RSA/CKA battery. It does not establish mathematical independence from every possible similarity metric, identify a biological mechanism, or support a multi-session trajectory conclusion.

The primary inferential units are five sessions for the response benchmark and six movie conditions for the one-session geometry and trajectory analyses. Multi-session trajectory replication, multiple encoding-model seeds, hidden-layer analysis, and the non-temporal damage control remain future work.

## Reproducible assets

- Locked configuration: [`configs/pilot.yaml`](configs/pilot.yaml)
- Static and reduced Dynamic checkpoints: [`../../models/static_on_dynamic/`](../../models/static_on_dynamic/) and [`../../models/parameter_matched_dynamic/`](../../models/parameter_matched_dynamic/)
- Frozen comparison GPFA: [`../../models/gpfa_model_comparison/`](../../models/gpfa_model_comparison/)
- Compact response, RSA/CKA, GPFA, matching, ablation, and Q1-Q6 outputs: [`../../results/tables/04_model_comparison/`](../../results/tables/04_model_comparison/)
- Full evidence ledger: [Model Comparison Results](../../docs/results/MODEL_COMPARISON_RESULTS.md) and [Q1-Q6 Answers](../../docs/results/Q1_Q6_ANSWERS.md)
- Complete method definitions: [Methods](../../docs/METHODS.md)

## Minimal reproduction entry points

Run the commands in order from this phase directory:

```text
python -m trajectory_model_eval.cli lock --config configs/pilot.yaml
python -m trajectory_model_eval.cli predict --config configs/pilot.yaml
python -m trajectory_model_eval.cli traditional --config configs/pilot.yaml
python -m trajectory_model_eval.cli gpfa --config configs/pilot.yaml
python -m trajectory_model_eval.cli sensitivity --config configs/pilot.yaml
python -m trajectory_model_eval.cli gpfa-evaluate --config configs/pilot.yaml
python -m trajectory_model_eval.cli extended-predict --config configs/pilot.yaml
python -m trajectory_model_eval.cli questions --config configs/pilot.yaml
```

Checkpoint inference commands require the official CUDA/Sensorium environment. GPFA analysis uses the Phase 2 scientific environment. The workflow fails closed when a required checkpoint, locked split, prediction tensor, or frozen GPFA artifact is missing. Generated working artifacts are written under `outputs/pilot/`; compact public results are preserved under the repository-level `results/tables/04_model_comparison/` directory.

**Previous phase:** [Phase 3](../03_parameter_matching/README.md) establishes the capacity-controlled Dynamic model.  
**Project-level overview:** [main README](../../README.md) and [experiment matrix](../../docs/EXPERIMENT_MATRIX.md).
