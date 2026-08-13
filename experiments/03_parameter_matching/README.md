# Phase 3 — Matching Dynamic model capacity to the Static control

**Research question:** Does the Dynamic model retain a response advantage when model capacity is approximately matched to the Static baseline?

## Why this phase exists

The full Dynamic model has substantially more parameters than the Static model. A direct comparison could therefore confuse temporal computation with generic model capacity. Phase 3 constructs a reduced Dynamic model that preserves the original temporal architecture while bringing its total parameter count within 2% of Static.

## Controlled architecture change

Only the Factorized3D core width is reduced, using a single predeclared multiplier:

```text
Full Dynamic channels       [32, 64, 128]
Reduced Dynamic channels    [16, 32, 64]
Width multiplier             0.5
```

The three-layer depth, `11/5/5` spatial kernels, `11/5/5` temporal kernels, 19-frame temporal receptive field, activations, normalization, behavior inputs, Gaussian readouts, grid predictors, pupil shifters, and output nonlinearity remain unchanged. The regular 1:2:4 widening rule is preserved rather than searching irregular channel combinations for a numerically closer count.

| Component | Static | Reduced Dynamic | Difference |
|---|---:|---:|---:|
| Core | 50,624 | 98,672 | +48,048 |
| Readout | 2,763,106 | 2,763,106 | 0 |
| Shifter | 285 | 285 | 0 |
| **Total** | **2,814,015** | **2,862,063** | **+48,048 (+1.707%)** |

## Data and training controls

Static and reduced Dynamic use the same five Dynamic Sensorium 2023 sessions, trial identities, neuron order, 36 x 64 movie resolution, 80-frame training snippets, behavioral and pupil inputs, normalization source, effective batch size, Poisson objective, training tiers, seed, and frames 50-299 evaluation interval. The readout parameter count is exactly matched.

This is a capacity-controlled Dynamic model for the present experiment, not an official Sensorium baseline reproduction.

## Current result

The parameter-matched Dynamic model reaches a five-session full-sequence oracle correlation of **0.1875**, compared with **0.1644** for Static. The absolute gain is **+0.0231**, and the Dynamic-minus-Static difference is positive in all five sessions. The reduced model retains approximately **95.35%** of the full Dynamic model's local oracle score of 0.1967.

This result answers the capacity check at the response level: the Dynamic advantage does not disappear when total parameter count is closely matched. It does not by itself show that trajectory metrics contain information beyond response accuracy; that question is reserved for Phase 4.

## Reproducible assets

- Locked configuration: [`configs/dynamic_parameter_matched.yaml`](configs/dynamic_parameter_matched.yaml)
- Architecture, data-lock, training, and evaluation records: [`records/`](records/)
- Published checkpoint: [`../../models/parameter_matched_dynamic/best.pt`](../../models/parameter_matched_dynamic/best.pt)
- Compact oracle result: [`../../results/tables/03_parameter_matching/parameter_matched_dynamic_oracle.json`](../../results/tables/03_parameter_matching/parameter_matched_dynamic_oracle.json)
- Full architecture report: [Parameter-Matched Dynamic Model](../../docs/models/PARAMETER_MATCHED_DYNAMIC.md)

## Minimal reproduction entry points

Run from this phase directory with the Phase 1 scientific environment and this phase's `src` directory available:

```text
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml audit
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml smoke
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml train
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml evaluate
```

Formal training should begin only after both `audit` and `smoke` pass. Phase 1 data and checkpoints are treated as read-only inputs.

**Previous phase:** [Phase 2](../02_gpfa_reliability/README.md) validates the trajectory assay independently of model predictions.  
**Next phase:** [Phase 4](../04_model_comparison/README.md) compares Static and parameter-matched Dynamic predictions in response, output-geometry, and brain-defined trajectory spaces.
