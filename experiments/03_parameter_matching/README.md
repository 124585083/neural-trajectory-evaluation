# Phase 3 — Total-parameter matching

This phase constructs and trains a reduced Factorized3D Dynamic model whose total trainable parameter count approximately matches the Static baseline before the main Static–Dynamic comparison.

## Purpose

Phase 1 provides the Static and full Dynamic baselines, with the full Dynamic model containing substantially more total trainable parameters. Phase 3 applies one predeclared reduction to the Dynamic core and audits the resulting complete-model parameter count against Static. The trained and frozen Total-parameter-matched Dynamic checkpoint becomes the primary Dynamic model used in Phase 4. This is a complete-model architectural comparison, not a same-backbone experiment or a causal isolation of temporal history.

## What is matched

| Component | Static | Total-parameter-matched Dynamic |
|---|---:|---:|
| Core parameters | 50,624 | 98,672 |
| Readout parameters | 2,763,106 | 2,763,106 |
| Shifter parameters | 285 | 285 |
| Temporal adapter | 0 | — |
| **Total trainable parameters** | **2,814,015** | **2,862,063** |

The complete-model difference is 48,048 parameters, or 1.707% relative to Static. The close totals arise because the corresponding session-specific readouts contribute the same 2,763,106 parameters to both models.

The control approximately matches complete-model trainable parameter count; it does not match the cores or isolate temporal history as the only architectural difference.

## Dynamic reduction

```text
Full Dynamic channels                 [32, 64, 128]
Total-parameter-matched channels      [16, 32, 64]
Predeclared width multiplier           0.5
```

The uniform half-width reduction preserves the original 1:2:4 widening rule. The three Factorized3D stages, spatial and temporal kernels, activation and normalization structure, regularizers, readout family, pupil shifter, and output nonlinearity remain unchanged from the full Dynamic definition. The architecture was fixed by the recorded total-parameter-count procedure; response and trajectory performance did not select the width.

## Inputs and prerequisites

- The Phase 1 Static parameter definition and training snapshot, plus the full Dynamic architecture definition.
- The same five official Dynamic Sensorium 2023 sessions used by the baselines.
- The locked Phase 3 config: [`configs/dynamic_parameter_matched.yaml`](configs/dynamic_parameter_matched.yaml).
- Python 3.11, the Phase 1 encoding environment, the Phase 3 package, and a CUDA-capable GPU for smoke testing, training, and evaluation.

See [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) for installation, data layout, and artifact-loading instructions.

## Formal workflow

Run from this directory. Phase 3 exposes the formal module `trajectory_param_match.experiment`; no separate console alias is installed.

```text
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml audit
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml smoke
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml train
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml evaluate
```

`audit` verifies the configuration, data/split identities, architecture, tensor shapes, and total/core/readout/shifter counts. Formal training should begin only after `audit` and `smoke` pass; `evaluate` independently reloads the published checkpoint for full-sequence oracle evaluation.

## Training protocol

The reduced control uses the locked five-session protocol shared with the baseline comparison: the same sessions and training tiers, 80-frame snippets and aligned targets, seed 42, Poisson objective, AdamW and scheduler family, response metric, and best-checkpoint selection rule. The model definition differs from the full Dynamic baseline only through the locked core channels. Full settings and alignment details are in [Methods](../../docs/METHODS.md).

## Outputs

- **Model:** [`../../models/parameter_matched_dynamic/best.pt`](../../models/parameter_matched_dynamic/best.pt), the released frozen Total-parameter-matched Dynamic checkpoint.
- **Parameter and architecture audits:** [`records/architecture_audit.json`](records/architecture_audit.json), [`records/audit_summary.json`](records/audit_summary.json), and the locked config above.
- **Data lock:** [`records/data_split_lock.json`](records/data_split_lock.json), recording the five-session data and tier identities.
- **Training and evaluation records:** [`records/training_summary.json`](records/training_summary.json), [`records/validation_events.jsonl`](records/validation_events.jsonl), and [`records/official_evaluation.json`](records/official_evaluation.json).
- **Compact public evaluation artifact:** [`../../results/tables/03_parameter_matching/parameter_matched_dynamic_oracle.json`](../../results/tables/03_parameter_matching/parameter_matched_dynamic_oracle.json).

Phase 4 consumes the frozen config and `best.pt`, reconstructs the model, and exports its aligned predictions. Phase 3 does not provide a separate prediction-export command.

## What Phase 3 establishes operationally

- The reduced three-stage Factorized3D architecture is fixed before the main model comparison.
- Its total trainable parameter count is within 1.707% of Static while its core architecture and core count remain different.
- Its frozen trained checkpoint and configuration are handed to Phase 4.

## Documentation

- [Methods](../../docs/METHODS.md) — exact architecture, parameter counts, training, and evaluation procedure
- [Results](../../docs/RESULTS.md) — response and trajectory comparison results
- [Design Rationale](../../docs/DESIGN_RATIONALE.md) — purpose and inferential scope of total-parameter matching
- [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) — data, environment, and artifact loading
