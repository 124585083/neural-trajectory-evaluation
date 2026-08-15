# Phase 1 — Encoding-model baselines

This phase establishes the Static and full Dynamic encoding-model baselines on the five official Dynamic Sensorium sessions under a shared training and evaluation protocol.

## Purpose

Phase 1 trains and evaluates two baseline encoding models. The Static baseline adapts a framewise 2D Sensorium-style model to Dynamic Sensorium, while the full Dynamic baseline uses the full-width Factorized3D architecture with temporal convolutions. Both use the same five-session dataset and the same aligned full-sequence evaluation interval. Their checkpoints provide the starting point for later total-parameter matching and trajectory comparisons.

## Inputs and prerequisites

- The five official Dynamic Sensorium 2023 sessions.
- Python 3.11 and the package versions pinned in [`pyproject.toml`](pyproject.toml), including the specified Sensorium and `neuralpredictors` revisions.
- A CUDA-capable GPU for formal training and full-sequence evaluation.
- The two versioned configs in [`configs/`](configs/).

Run commands from this directory after installing the Phase 1 package:

```text
python -m pip install -e .
```

See [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) for data acquisition, directory layout, environment setup, and checkpoint loading.

## Formal workflows

### Static baseline

Config: [`configs/static_dynamic_sensorium2023.yaml`](configs/static_dynamic_sensorium2023.yaml). Formal module: `trajectory_eval.static_dynamic`. Installed entry point: `trajectory-static-dynamic`.

```text
python -m trajectory_eval.static_dynamic --config configs/static_dynamic_sensorium2023.yaml train
python -m trajectory_eval.static_dynamic --config configs/static_dynamic_sensorium2023.yaml evaluate
```

Training writes run products under `checkpoints/static_dynamic_sensorium2023/` and `logs/static_dynamic_sensorium2023/`. Independent evaluation loads the published checkpoint at [`../../models/static_on_dynamic/best.pt`](../../models/static_on_dynamic/best.pt).

### Full Dynamic baseline

Config: [`configs/phase1A_dynamic_official.yaml`](configs/phase1A_dynamic_official.yaml). Formal module: `trajectory_eval.official_dynamic`. Installed entry point: `trajectory-official-dynamic`.

```text
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml train
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml evaluate
```

Training writes run products under `checkpoints/dynamic_official_reproduction/` and `logs/dynamic_official_reproduction/`. Independent evaluation loads the published checkpoint at [`../../models/official_dynamic/best.pt`](../../models/official_dynamic/best.pt).

The local full Dynamic run was stopped by project decision after epoch 103 validation. The published checkpoint retains the best complete epoch-97 state rather than representing natural completion of the official early-stopping procedure.

## Outputs

### Models

- Static best checkpoint: [`../../models/static_on_dynamic/best.pt`](../../models/static_on_dynamic/best.pt)
- Full Dynamic best checkpoint: [`../../models/official_dynamic/best.pt`](../../models/official_dynamic/best.pt)

### Evaluation

- Compact full-sequence oracle summaries: [`../../results/tables/01_baselines/`](../../results/tables/01_baselines/)
- Model-specific evaluation records: [`records/static/official_evaluation.json`](records/static/official_evaluation.json) and [`records/dynamic/official_evaluation.json`](records/dynamic/official_evaluation.json)

### Audits

- Static architecture and retained training records: [`records/static/`](records/static/); formal training writes its environment snapshot under `logs/static_dynamic_sensorium2023/`.
- Dynamic architecture, temporal-alignment, and retained training records: [`records/dynamic/`](records/dynamic/); formal training writes its environment snapshot under `logs/dynamic_official_reproduction/`.

### Prediction exports

Continuous full Dynamic oracle predictions for downstream analyses can be generated with:

```text
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml export
```

The configured local output is `predictions/dynamic_official_reproduction/`; large prediction arrays are not duplicated in this README.

## What Phase 1 establishes

- Both baseline architectures can be trained and independently evaluated under the same five-session Dynamic Sensorium protocol.
- Full-sequence predictions and neural targets are compared on original frames 50–299 after the shared burn-in.
- The resulting checkpoints and alignment records feed later total-parameter matching and trajectory-evaluation phases.

Numerical results and scientific interpretation are reported in [Results](../../docs/RESULTS.md).

## Documentation

- [Methods](../../docs/METHODS.md) — model architecture, temporal alignment, training, evaluation, and prediction export
- [Results](../../docs/RESULTS.md) — response-level and downstream comparison results
- [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) — data acquisition, environment setup, and artifact loading
- [Design Rationale](../../docs/DESIGN_RATIONALE.md) — reasons for comparing Static, Dynamic, and later controls
