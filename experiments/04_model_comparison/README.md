# Phase 4 — Model comparison

This phase compares Static and Total-parameter-matched Dynamic predictions using response, output-space RSA/CKA, and frozen neural-data-defined trajectory metrics, together with predefined stress tests.

## Purpose

Phase 1 provides the frozen Static checkpoint, and Phase 3 provides the frozen Total-parameter-matched Dynamic checkpoint. Phase 4 reconstructs aligned oracle predictions from both models and evaluates them with one conventional and trajectory-analysis pipeline. The trajectory comparison uses a separately fitted GPFA defined only by locked neural training data. Response matching, temporal-weight attenuation, time reversal, and an enriched conventional-feature battery probe whether trajectory metrics merely restate the tested conventional summaries; scientific interpretation is delegated to the canonical result documents.

## Inputs and frozen artifacts

- **Recorded data:** pilot session `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20`; 58 oracle trials in six repeated-movie conditions; deterministic 512-neuron order; original frames 50–299.
- **Static model:** [`../01_baselines/records/static/training_config.yaml`](../01_baselines/records/static/training_config.yaml) and [`../../models/static_on_dynamic/best.pt`](../../models/static_on_dynamic/best.pt).
- **Total-parameter-matched Dynamic:** [`../03_parameter_matching/configs/dynamic_parameter_matched.yaml`](../03_parameter_matching/configs/dynamic_parameter_matched.yaml) and [`../../models/parameter_matched_dynamic/best.pt`](../../models/parameter_matched_dynamic/best.pt).
- **Comparison protocol:** [`configs/pilot.yaml`](configs/pilot.yaml), with generated locks under `outputs/pilot/` and a compact released record at [`../../results/tables/04_model_comparison/protocol_lock.json`](../../results/tables/04_model_comparison/protocol_lock.json).
- **Auxiliary checkpoint:** [`../../models/parameter_matched_dynamic/epoch_65_validation_matched.pth`](../../models/parameter_matched_dynamic/epoch_65_validation_matched.pth), used only for the separately named validation-matched checkpoint diagnostic.
- **Comparison GPFA training data:** a deterministic 174-of-348 training-trial subset, divided into 139 fit and 35 calibration trials, then refitted on all 174 after initialization selection.

The resulting `q = 4` comparison-subset GPFA is the exact neural-data-defined GPFA used for model evaluation. Phase 2 establishes the assay design on the full train tier; Phase 4 separately fits and revalidates the locked 174-trial comparison GPFA used here. See [GPFA Validation](../../docs/GPFA_VALIDATION.md) for the relationship between the two fits.

## Locked analysis scope

| Element | Fixed scope |
|---|---|
| Oracle tensor | Neural, Static, and Dynamic arrays with shape `[58, 250, 512]` |
| Trial/condition support | 58 trials; six repeated movies |
| Neurons and time | Deterministic seed-42 neuron order; frames 50–299 |
| Comparison GPFA | 174 train trials; 139/35 fit/calibration split; fixed `q = 4` |
| GPFA temporal grids | Every fourth frame: 63 observations at approximately 7.5 Hz; posterior queried at all 250 timestamps at 30 Hz |
| Evaluation families | Response, output-space RSA/CKA, frozen-GPFA trajectory metrics, model-prediction nulls, and paired summaries |
| Stress-test seeds | Balanced repeat split `20260813`; response-perturbation noise seed `123` |
| Temporal attenuation | Off-center temporal-weight retention `1.00`, `0.75`, `0.50`, `0.25`, and `0.00` |

Both model predictions enter the same frozen GPFA posterior inference without model-specific fitting, rotation, scaling, or latent alignment.

## Protocol-lock behavior

The `lock` command deterministically regenerates and overwrites `protocol_lock.npz` and `protocol_lock.json` from the current data and config. Other commands call `load_protocol`: a missing NPZ lock is generated automatically, while an existing NPZ lock is reused without recomputing or comparing its fingerprints to the current config. Rerun `lock` explicitly after an intentional data/config change. Prediction generation requires identical Static/Dynamic sampler order and neural targets and rejects oracle trials absent from the lock; extended predictions must match the primary trial indices and neural tensor exactly. Missing checkpoints or stage-specific prediction/GPFA artifacts stop the dependent stage, but a missing protocol lock alone does not.

## Formal workflow

Run from this directory in the order below. Checkpoint inference uses the encoding environment; GPFA and table analysis use the analysis environment described in [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md).

### 1. Lock the protocol and export aligned predictions — encoding environment

```text
python -m trajectory_model_eval.cli lock --config configs/pilot.yaml
python -m trajectory_model_eval.cli predict --config configs/pilot.yaml
```

### 2. Run conventional metrics, fit/revalidate GPFA, and evaluate trajectories — analysis environment

```text
python -m trajectory_model_eval.cli traditional --config configs/pilot.yaml
python -m trajectory_model_eval.cli gpfa --config configs/pilot.yaml
python -m trajectory_model_eval.cli sensitivity --config configs/pilot.yaml
python -m trajectory_model_eval.cli gpfa-evaluate --config configs/pilot.yaml
```

`gpfa` performs the 139/35 initialization selection, 174-trial refit, and oracle reliability/null gate. `gpfa-evaluate` applies the frozen result to recorded responses and both prediction tensors.

### 3. Generate stress-test predictions — encoding environment

```text
python -m trajectory_model_eval.cli extended-predict --config configs/pilot.yaml
```

This exports the validation-matched checkpoint prediction and five off-center temporal-weight retention levels while preserving the center slices and the rest of the model.

### 4. Produce the predefined Q1–Q6 analysis artifacts — analysis environment

```text
python -m trajectory_model_eval.cli questions --config configs/pilot.yaml
```

This stage implements the response-score-matched output perturbation, temporal attenuation summaries, time reversal, enriched conventional-feature comparisons, and leave-family-out diagnostics. Perturbation strength is chosen using the selection-half response target; test-half neural responses are not used to select that strength. The test half is then used for evaluation.

The CLI also exposes `all`, which runs only `lock` through `gpfa-evaluate` in internal sequence. It does not run `extended-predict` or `questions`, and the split-environment sequence above is the canonical workflow.

## Outputs

- **Aligned predictions and protocol:** generated `outputs/pilot/protocol_lock.npz`, `protocol_lock.json`, and `oracle_predictions.npz`; compact summaries are released in [`../../results/tables/04_model_comparison/`](../../results/tables/04_model_comparison/).
- **Frozen comparison GPFA:** [`../../models/gpfa_model_comparison/gpfa.pkl`](../../models/gpfa_model_comparison/gpfa.pkl), [`../../models/gpfa_model_comparison/preprocessing.npz`](../../models/gpfa_model_comparison/preprocessing.npz), and selection/reliability metadata in the Phase 4 result directory.
- **Conventional metrics:** response and RSA/CKA tables and paired summaries are released in the Phase 4 result directory; detailed distributions are generated under `outputs/pilot/`.
- **Trajectory comparison:** latent trajectory archives are generated under `outputs/pilot/`; model metrics, condition bootstrap summaries, and model-prediction null distributions are released in the Phase 4 result directory.
- **Stress tests and Q1–Q6:** extended-prediction summary, response-score-matching output, temporal-attenuation tables, Q6 candidate/leave-family-out tables, and [`../../results/tables/04_model_comparison/q1_q6_answers.json`](../../results/tables/04_model_comparison/q1_q6_answers.json).
- **Sensitivity and tests:** comparison-GPFA sensitivity outputs under [`../../results/tables/04_model_comparison/sensitivity/`](../../results/tables/04_model_comparison/sensitivity/) and focused contracts under [`tests/`](tests/).

## What Phase 4 establishes operationally

- Recorded responses and both model predictions are aligned on identical trials, neurons, conditions, and timestamps.
- Both models are evaluated in the same separately frozen 174-trial neural-data-defined GPFA coordinate system.
- Conventional, trajectory, sensitivity, and stress-test artifacts are produced under one comparison protocol.

## Documentation

- [Results](../../docs/RESULTS.md) — main scientific result chain
- [Detailed Q1–Q6 evidence](../../docs/results/Q1_Q6_ANSWERS.md) — question-specific statistics, controls, and interpretation boundaries
- [Methods](../../docs/METHODS.md) — response, RSA/CKA, GPFA, resampling, and stress-test procedures
- [GPFA Validation](../../docs/GPFA_VALIDATION.md) — reliability and measurement limits of the comparison assay
- [Design Rationale](../../docs/DESIGN_RATIONALE.md) — reasons for the controls and stress tests
- [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) — environment, data, and artifact setup
