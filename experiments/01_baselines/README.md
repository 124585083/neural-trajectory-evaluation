# Phase 1 — Establishing trustworthy Static and Dynamic baselines

**Research question:** Can the official Dynamic Sensorium architecture and a strict frame-wise Static control be trained and evaluated reliably on the same dynamic neural dataset?

## Why this phase exists

The later trajectory experiments are meaningful only if the encoding models, data alignment, checkpoint loading, and response evaluator are already known to work. Phase 1 therefore establishes two five-session reference models before introducing GPFA or any trajectory metric:

- the full-width Factorized3D Dynamic Sensorium baseline; and
- the Sensorium static CNN architecture retrained frame by frame on Dynamic Sensorium 2023.

The Static model is a controlled transfer of the official static architecture, not a native Sensorium 2022 benchmark and not an official Static-on-Dynamic leaderboard entry.

## Experimental design

Both models use the same five Dynamic Sensorium 2023 competition sessions, comprising 1,744 training trials, 293 oracle trials, and 40,034 recorded neurons. They share the official movie preprocessing, behavioral covariates, pupil input, session-specific Gaussian readouts, shifters, training tiers, and response evaluator. Full-sequence oracle evaluation discards the first 50 burn-in frames and retains original frames 50-299.

The critical difference is temporal context. The Dynamic model uses a three-layer Factorized3D core with learned temporal kernels. The Static model processes each movie frame independently through a four-layer 2D core; a parameter-free adapter preserves the official video tensor and timestamp contracts without giving the model access to neighboring frames.

| Model | Parameters | Learned temporal context | Five-session local oracle correlation |
|---|---:|---|---:|
| Full Factorized3D Dynamic | 5,707,743 | Yes | 0.1967 |
| Static-on-Dynamic | 2,814,015 | No | 0.1644 |

## Current result

The implementation, checkpoint-reload, temporal-alignment, and local oracle-evaluation pipeline works for both models. Static training completed through the official early-stopping procedure at epoch 63. Dynamic training was stopped after epoch 103, and the best checkpoint from epoch 97 was recovered from the complete epoch history; the full official early-stopping closure was therefore not allowed to finish. The recovered Dynamic checkpoint reaches a local five-session oracle correlation of **0.1967**, while the Static checkpoint reaches **0.1644** under the same evaluator.

The official Dynamic reference reports **0.1887** on hidden `final_test_main`, but those labels are unavailable locally. The local oracle value and hidden server value are different evaluation splits and must not be compared numerically as a reproduction error. The precise status is: **the protocol and local evaluation pipeline are reproduced; exact hidden-test score reproduction is not locally verifiable**.

## What this phase establishes

Phase 1 shows that a frame-wise model can be trained and evaluated on the dynamic dataset without changing the response target or temporal alignment. It also confirms that the full Dynamic architecture has a clear response-level advantage. It does not yet determine whether that advantage comes from temporal computation or simply from the Dynamic model's larger core. Phase 3 addresses that capacity confound.

No RSA, CKA, or GPFA trajectory claim is made in this phase. The later output-space analyses operate on predicted neural responses, not the optional hidden-layer hooks retained here for implementation inspection.

## Reproducible assets

- Published checkpoints: [`../../models/official_dynamic/best.pt`](../../models/official_dynamic/best.pt) and [`../../models/static_on_dynamic/best.pt`](../../models/static_on_dynamic/best.pt)
- Compact oracle summaries: [`../../results/tables/01_baselines/`](../../results/tables/01_baselines/)
- Dynamic configuration and audit records: [`configs/phase1A_dynamic_official.yaml`](configs/phase1A_dynamic_official.yaml) and [`records/dynamic/`](records/dynamic/)
- Static configuration and audit records: [`configs/static_dynamic_sensorium2023.yaml`](configs/static_dynamic_sensorium2023.yaml) and [`records/static/`](records/static/)
- Full model reports: [Dynamic model](../../docs/models/DYNAMIC_MODEL.md) and [Static model](../../docs/models/STATIC_MODEL.md)

## Minimal reproduction entry points

Run these commands from this phase directory after installing the package and obtaining the authorized Dynamic Sensorium data:

```text
# Full Dynamic baseline
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml audit
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml smoke
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml evaluate

# Static-on-Dynamic baseline
python -m trajectory_eval.static_dynamic --config configs/static_dynamic_sensorium2023.yaml audit
python -m trajectory_eval.static_dynamic --config configs/static_dynamic_sensorium2023.yaml smoke
python -m trajectory_eval.static_dynamic --config configs/static_dynamic_sensorium2023.yaml evaluate
```

The corresponding `train` command reproduces each training run. Audit and smoke checks should pass before formal training begins. Raw data are read in place and are never copied, moved, or modified by this phase.

**Next phase:** [Phase 2](../02_gpfa_reliability/README.md) validates the brain-defined trajectory assay using neural repeat reliability before that assay is allowed to compare encoding models.
