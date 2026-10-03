# Phase 1 — Encoding-model baselines

Model payloads are external. The [artifact guide](../../docs/DATA_AND_REPRODUCIBILITY.md#8-external-model-and-gpfa-artifacts) gives expected restore paths and integrity checks; checkpoint links below point to identity records.

This Study 1 phase establishes Static and full Dynamic encoding-model baselines on the five official Dynamic Sensorium sessions. Both follow the shared training and evaluation protocol below.

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

Training writes run products under `checkpoints/static_dynamic_sensorium2023/` and `logs/static_dynamic_sensorium2023/`. Reloaded oracle evaluation loads the externally restored checkpoint identified at [`../../models/static_on_dynamic/best.pt`](../../results/manifests/model_files.csv).

### Full Dynamic baseline

Config: [`configs/phase1A_dynamic_official.yaml`](configs/phase1A_dynamic_official.yaml). Formal module: `trajectory_eval.official_dynamic`. Installed entry point: `trajectory-official-dynamic`.

```text
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml train
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml evaluate
```

Training writes run products under `checkpoints/dynamic_official_reproduction/` and `logs/dynamic_official_reproduction/`. Reloaded oracle evaluation loads the externally restored checkpoint identified at [`../../models/official_dynamic/best.pt`](../../results/manifests/model_files.csv).

The local full Dynamic run was stopped by project decision after epoch 103 validation. The recorded checkpoint retains the best complete epoch-97 state. The official early-stopping procedure had not terminated naturally.

## Outputs

### Models

- Static best checkpoint: [`../../models/static_on_dynamic/best.pt`](../../results/manifests/model_files.csv)
- Full Dynamic best checkpoint: [`../../models/official_dynamic/best.pt`](../../results/manifests/model_files.csv)

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

- Both baseline architectures can be trained and evaluated after reloading under the same five-session Dynamic Sensorium protocol.
- Full-sequence predictions and neural targets are compared on original frames 50–299 after the shared burn-in.
- The resulting checkpoints and alignment records feed later total-parameter matching and trajectory-evaluation phases.

Numerical results and scientific interpretation are reported in [Results](../../docs/RESULTS.md).

## Documentation

- [Methods](../../docs/METHODS.md) — model architecture, temporal alignment, training, evaluation, and prediction export
- [Results](../../docs/RESULTS.md) — response-level and downstream comparison results
- [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) — data acquisition, environment setup, and artifact loading
- [Design Rationale](../../docs/DESIGN_RATIONALE.md) — reasons for comparing Static, Dynamic, and later controls

## Evaluation data reuse

Encoding checkpoints were selected using the oracle tier that later supplies full-sequence scores. Reloading a checkpoint checks execution without creating an independent test set. The [Methods data-use table](../../docs/METHODS.md#data-use-and-evaluation-independence) identifies model-specific training, selection and scoring uses. Excluding oracle responses from GPFA fitting or one perturbation choice does not undo their earlier use in checkpoint selection.

## Static frame checks

[Frame-permutation tests](tests/test_static_dynamic_locks.py) check equivariance: moving a frame and its covariates moves the corresponding prediction. The separate [frame-independence test](tests/test_static_frame_independence.py) changes one retained frame while keeping all other frames and covariates fixed. It tests visual channels, behavioral channels, the separate behavior argument and pupil input through the actual Static adapter in evaluation mode.

The recorded checks passed for 15 perturbations at three input positions, using 27 frames, an 18-frame crop, nine retained outputs and seven neurons. Unchanged outputs agree within the declared `2e-6` tolerance. A deliberately sequence-coupled control passes permutation equivariance and fails independence. These CPU checks use synthetic inputs and the actual model implementation. They verify evaluation-mode execution, not training-mode batch normalization or upstream data normalization.
