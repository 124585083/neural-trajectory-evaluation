# Phase 2 — GPFA reliability

This phase develops and validates a neural-data-defined GPFA trajectory assay using recorded neural responses before the assay is used for Static–Dynamic model comparison.

## Purpose

Phase 2 asks whether repeated presentations of the same natural movie yield reproducible neural-population trajectories in a GPFA space fitted only from training neural responses. Static and Dynamic predictions do not define or modify this coordinate system. Reliability is evaluated only after model selection, preprocessing, and fitting are frozen. Structured nulls test dependence on movie identity, temporal alignment and order, and coordinated population timing before the assay design is applied to model comparison.

## Inputs and prerequisites

- Pilot session: `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20`.
- All 348 official train-tier neural-response trials for fitting, calibration, and final refitting.
- The 58 official oracle trials, used only for post-freeze repeated-movie reliability.
- The deterministic seed-42 neuron ordering and first 512 selected units.
- Python and dependencies specified in [`pyproject.toml`](pyproject.toml), with operational choices in [`configs/pilot.yaml`](configs/pilot.yaml).

Run commands from this directory after `python -m pip install -e .`. See [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) for dataset installation and environment setup.

## Analysis lock

The executable workflow loads its fixed choices from [`configs/pilot.yaml`](configs/pilot.yaml). The result-blind specification is recorded in [`../../docs/supplementary/protocols/GPFA_PROTOCOL_LOCKED.md`](../../docs/supplementary/protocols/GPFA_PROTOCOL_LOCKED.md) and was **internally locked before reliability-result inspection**; it was not externally preregistered. The CLI does not generate or parse that Markdown record.

## Formal workflow

The installed entry point is `trajectory-reliability`; the equivalent module commands below show the formal order.

### Inspect and smoke-test

```text
python -m trajectory_reliability.cli inspect --config configs/pilot.yaml
python -m trajectory_reliability.cli smoke --config configs/pilot.yaml
```

Run the smoke test before the primary analysis because it writes reduced test artifacts to the configured output directory.

### Fit, select, refit, and evaluate reliability

```text
python -m trajectory_reliability.cli run --config configs/pilot.yaml
```

This command applies the locked 278/70 fit/calibration split, selects GPFA hyperparameters from calibration likelihood, refits on all 348 training trials, freezes the assay, and then evaluates oracle split-half reliability and matched nulls.

### Sensitivity and auxiliary diagnostics

Run these after the primary command because the saturation workflows load its frozen GPFA and preprocessing artifacts:

```text
python -m trajectory_reliability.cli saturation --config configs/pilot.yaml
python -m trajectory_reliability.cli split-saturation --config configs/pilot.yaml
python -m trajectory_reliability.cli condition-prior --config configs/pilot.yaml
```

## What is frozen in Phase 2

| Element | Frozen choice |
|---|---|
| Session and neurons | Pilot session; deterministic seed-42 order; first 512 units |
| Temporal support | Original frames 50–299 |
| Selection/refit trials | 278 fit, 70 calibration; refit on all 348 train trials |
| Latent dimensions | Candidates `4, 8, 12, 16`; released artifact uses calibration-selected `q = 4` |
| Temporal grids | Every fourth frame: 63 observations at approximately 7.5 Hz; posterior queried at all 250 timestamps |
| Selection rule | Smallest dimension within one standard error of the best calibration marginal NLL |
| Reliability | Seed 42; 200 balanced split-half draws after freezing |
| Null families | Movie-condition, circular-shift, frame/block-order, reversal, and independent-neuron timing nulls |

## Outputs

- **Frozen GPFA and preprocessing:** generated under `outputs/pilot/`; released as [`../../models/gpfa_reliability/gpfa.pkl`](../../models/gpfa_reliability/gpfa.pkl) and [`../../models/gpfa_reliability/preprocessing.npz`](../../models/gpfa_reliability/preprocessing.npz).
- **Selection summary:** [`../../results/tables/02_gpfa_reliability/model_selection.csv`](../../results/tables/02_gpfa_reliability/model_selection.csv).
- **Reliability and null outputs:** observed split halves, matched-null distributions, data/run audits, and the compact summary in [`../../results/tables/02_gpfa_reliability/`](../../results/tables/02_gpfa_reliability/).
- **Sensitivity outputs:** saturation and split-count artifacts in [`../../results/tables/02_gpfa_reliability/saturation/`](../../results/tables/02_gpfa_reliability/saturation/), plus the train-tier prior diagnostic in [`../../results/tables/02_gpfa_reliability/behavior_conditioned_prior.json`](../../results/tables/02_gpfa_reliability/behavior_conditioned_prior.json).
- **Tests:** focused data, GPFA, selection, and reliability contracts are in [`tests/`](tests/).

## What Phase 2 establishes operationally

- A neural-data-defined GPFA and its preprocessing can be fitted without encoding-model predictions.
- Balanced repeat reliability and structured-null checks can be applied only after the assay is frozen.
- Phase 4 applies the same validated assay design to a separately fitted and revalidated 174-trial comparison-subset GPFA; the Phase 2 `gpfa.pkl` is not the final model-comparison fit.

## Documentation

- [GPFA Validation](../../docs/GPFA_VALIDATION.md) — reliability evidence, sensitivity, negative findings, and the relationship between the two GPFA fits
- [Methods](../../docs/METHODS.md) — fitting, inference, temporal sampling, metrics, and null procedures
- [Design Rationale](../../docs/DESIGN_RATIONALE.md) — why measurement validation precedes model comparison
- [Data and Reproducibility](../../docs/DATA_AND_REPRODUCIBILITY.md) — data and environment setup
