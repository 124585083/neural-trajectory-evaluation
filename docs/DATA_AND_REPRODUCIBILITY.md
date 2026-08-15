# Data and Reproducibility

Raw Dynamic Sensorium data are not redistributed by this repository. The repository provides code, documentation, compact result tables, and selected encoding-model and GPFA artifacts through Git LFS. Reproduction is organized by experimental phase; this guide covers shared setup and navigation, while each Phase README is the authoritative source for exact commands.

## 1. Reproducibility levels

| Level | Requires raw Sensorium data? | Requires released checkpoints? | What can be done |
|---|---|---|---|
| Documentation and code inspection | No | No | Inspect the scientific documentation, source, configurations, tests, and compact result tables. |
| Lightweight unit and contract tests | No | No | Run the confirmed synthetic, temporary-file, metric, and configuration tests listed in Section 13. |
| Released-artifact inspection | No | Yes | Pull the Git LFS objects, verify sizes and SHA-256 digests, and inspect trusted checkpoint or GPFA files in the appropriate environment. |
| Analysis from released encoding checkpoints | Yes | Yes | Reconstruct models, export aligned predictions, and run downstream evaluation without retraining the encoding models. |
| Full public analysis reproduction | Yes | Released or regenerated | Run the four phase workflows and compare regenerated compact outputs with `results/tables/`. |
| Full encoding-model retraining | Yes | No, if all encoding models are retrained | Recreate Phase 1 and Phase 3 training runs; this is the most compute-intensive path. |

A minimal clone is sufficient for reading and lightweight tests, but not for model inference or scientific recomputation from the raw stimuli and responses.

## 2. Repository clone and Git LFS

Files under `models/` with extensions `.pt`, `.pth`, `.pkl`, and `.npz` are configured for Git LFS in `.gitattributes`.

```text
git clone https://github.com/124585083/neural-trajectory-evaluation.git
cd neural-trajectory-evaluation
git lfs install
git lfs pull
```

Without `git lfs pull`, model paths may contain small pointer files instead of usable checkpoints or fitted GPFA objects.

## 3. Raw Dynamic Sensorium data

Obtain Dynamic Sensorium 2023 from the official providers:

- [Dynamic Sensorium 2023 starter repository](https://github.com/ecker-lab/sensorium_2023)
- [Official GIN data record](https://gin.g-node.org/pollytur/sensorium_2023_dataset)

Provider access, reuse, and citation terms apply. Do not commit downloaded data to this repository; `.gitignore` excludes the repository-level data payload except for `data/.gitkeep`.

The encoding-model workflows use these five official competition sessions:

```text
dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20
dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20
dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20
dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20
dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20
```

The detailed GPFA and model-comparison pilot uses the first session in this list; it is not an additional sixth session.

## 4. Expected local data layout

The committed configurations expect the session directories directly under one shared root:

```text
neural-trajectory-evaluation/
└── data/
    └── sensorium_all_2023/
        ├── dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/
        ├── dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20/
        ├── dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20/
        ├── dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20/
        └── dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20/
```

The checked-in root values are `../../data/sensorium_all_2023` and are resolved relative to the corresponding phase directory. A different location may be supplied by editing the applicable config root to an absolute path or another phase-relative path. A symlink or junction is optional; it is not part of the required repository layout.

## 5. Upstream code and pinned revisions

The Phase 1 package installs the formal encoding stack directly from Git at these revisions:

| Dependency | Pinned revision |
|---|---|
| `ecker-lab/sensorium_2023` | `0e02656220e84a50f3be1b92d6f66c2f9ccd51ef` |
| `sinzlab/neuralpredictors` | `efdda679596517fad95d71f36d0385d7450dd207` |

The selected `neuralpredictors` revision contains the corrected channel wiring required by the variable-width Factorized3D core. Installing `experiments/01_baselines` therefore requires Git and network access unless these exact dependencies are already available in a local package cache.

Original repository code and documentation use the repository [MIT License](../LICENSE). Raw Sensorium data remain governed by provider terms, and upstream source/license boundaries are documented in [Third-Party Notices](../THIRD_PARTY_NOTICES.md).

## 6. Python environments

Python 3.11 is the common tested version. Two environments are required because Phase 1 pins `pandas==2.0.0`, while the Phase 2 analysis package requires `pandas>=2.1`.

### Encoding environment

Use this environment for Phase 1, Phase 3, and the Phase 4 `lock`, `predict`, and `extended-predict` stages.

Create and activate it with the platform-appropriate commands:

```text
# Windows PowerShell
py -3.11 -m venv .venv-encoding
.\.venv-encoding\Scripts\Activate.ps1

# Linux or macOS
python3.11 -m venv .venv-encoding
source .venv-encoding/bin/activate
```

Then install the phase packages from the repository root:

```text
python -m pip install --upgrade pip setuptools wheel
python -m pip install "pytest>=8,<10"
python -m pip install -e ./experiments/01_baselines
python -m pip install -e ./experiments/03_parameter_matching
python -m pip install "scipy>=1.12,<2"
python -m pip install -e ./experiments/02_gpfa_reliability --no-deps
python -m pip install -e ./experiments/04_model_comparison --no-deps
```

The `--no-deps` installs expose the Phase 2 data-contract code and Phase 4 command package without replacing the pinned encoding stack.

### Analysis / GPFA environment

Use this environment for Phase 2 and the Phase 4 `traditional`, `gpfa`, `sensitivity`, `gpfa-evaluate`, and `questions` stages.

```text
# Windows PowerShell
py -3.11 -m venv .venv-analysis
.\.venv-analysis\Scripts\Activate.ps1

# Linux or macOS
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "./experiments/02_gpfa_reliability[test]"
python -m pip install -e ./experiments/04_model_comparison
```

The individual `pyproject.toml` files remain authoritative for direct dependencies and package entry points. Follow the Phase 4 README when switching environments within its ordered workflow.

## 7. Hardware expectations

### Encoding-model training and inference

The formal Phase 1 and Phase 3 training/evaluation workflows and the Phase 4 prediction stages require a CUDA-capable GPU. Development runs used an NVIDIA RTX 4090 Laptop GPU with 16 GB VRAM and approximately 31 GB host RAM; identical hardware is not required. The five-session loader disables full file caching to keep host-memory use manageable.

### GPFA and table analysis

Phase 2 and the non-inference Phase 4 analyses use NumPy/SciPy/scikit-learn and do not require a GPU. GPFA fitting, reliability resampling, null generation, and sensitivity profiles can be CPU- and memory-intensive, but no exact runtime is guaranteed across machines.

## 8. Released model and GPFA artifacts

All paths below are Git LFS-managed.

### Encoding models

| Role | Released path |
|---|---|
| Full Dynamic baseline | [`../models/official_dynamic/best.pt`](../models/official_dynamic/best.pt) |
| Static-on-Dynamic baseline | [`../models/static_on_dynamic/best.pt`](../models/static_on_dynamic/best.pt) |
| Total-parameter-matched Dynamic | [`../models/parameter_matched_dynamic/best.pt`](../models/parameter_matched_dynamic/best.pt) |
| Validation-matched auxiliary checkpoint | [`../models/parameter_matched_dynamic/epoch_65_validation_matched.pth`](../models/parameter_matched_dynamic/epoch_65_validation_matched.pth) |

### GPFA objects

| Role | Model and preprocessing |
|---|---|
| Phase 2 full-train reliability GPFA | [`gpfa.pkl`](../models/gpfa_reliability/gpfa.pkl), [`preprocessing.npz`](../models/gpfa_reliability/preprocessing.npz) |
| Phase 4 comparison-subset GPFA | [`gpfa.pkl`](../models/gpfa_model_comparison/gpfa.pkl), [`preprocessing.npz`](../models/gpfa_model_comparison/preprocessing.npz) |

The Phase 2 and Phase 4 GPFAs are different fitted objects. Their scientific roles are documented in [GPFA Validation](GPFA_VALIDATION.md). Python pickle files may execute code during deserialization; load the `.pkl` objects only from a trusted checkout after verifying their digests.

## 9. Checksums and artifact manifests

[`results/manifests/model_files.csv`](../results/manifests/model_files.csv) is the canonical model-artifact manifest. It records each released path, byte size, SHA-256 digest, and purpose; the guide does not duplicate those values.

Verify an individual file against the corresponding manifest row with either:

```text
# Linux or macOS
sha256sum models/official_dynamic/best.pt

# Windows PowerShell
(Get-FileHash -Algorithm SHA256 models/official_dynamic/best.pt).Hash
```

If a size or digest does not match, rerun `git lfs pull` before using the artifact.

## 10. Phase-by-phase reproduction map

| Phase | Purpose | Environment | Entry documentation |
|---|---|---|---|
| Phase 1 | Static and full Dynamic baselines | Encoding | [Phase 1 README](../experiments/01_baselines/README.md) |
| Phase 2 | Neural-data-defined GPFA reliability | Analysis | [Phase 2 README](../experiments/02_gpfa_reliability/README.md) |
| Phase 3 | Total-parameter-matched Dynamic | Encoding | [Phase 3 README](../experiments/03_parameter_matching/README.md) |
| Phase 4 | Final comparison and stress tests | Both | [Phase 4 README](../experiments/04_model_comparison/README.md) |

The dependency structure is:

```text
data setup → Phase 1 → Phase 3 → Phase 4
          └→ Phase 2 ─────────→ Phase 4 assay design
```

Phase 2 can run independently of encoding-model retraining once its pilot-session data are available. Phase 4 uses the assay design validated in Phase 2 but fits and revalidates a separate comparison-subset GPFA. Exact commands and stage order remain in the Phase READMEs.

## 11. Minimal end-to-end reproduction order

1. Clone the repository and pull Git LFS objects.
2. Obtain the five official Dynamic Sensorium sessions and configure the shared data root.
3. Create the encoding and analysis environments.
4. In the encoding environment, either evaluate the released Phase 1 checkpoints or retrain the Phase 1 models.
5. In the analysis environment, run the Phase 2 primary workflow and any required sensitivity stages.
6. In the encoding environment, either evaluate the released Phase 3 checkpoint or retrain Phase 3.
7. Run Phase 4 in the exact cross-environment sequence documented in its README.
8. Compare regenerated compact outputs with the corresponding directories under `results/tables/`.

Phase 2 may run while encoding-model work is in progress because it uses recorded neural data rather than model predictions.

### Reproduce analysis from released checkpoints

This is the shorter path. Keep the released Phase 1 and Phase 3 checkpoints, use the raw Sensorium data to reconstruct their session-specific readouts and aligned predictions, and then run Phase 4. The pilot session is sufficient for the detailed Phase 4 analysis; all five sessions are required to re-evaluate the five-session response benchmarks.

### Retrain encoding models from raw data

Run the Phase 1 and Phase 3 training commands before their evaluation commands, then continue to Phase 4. Full retraining requires all five sessions and the CUDA encoding environment.

## 12. Using released checkpoints instead of retraining

| Workflow | Released artifact support | Raw data still required? |
|---|---|---|
| Phase 1 evaluation | `evaluate` loads the published Static or full Dynamic checkpoint selected by its config. | Yes, all five sessions. |
| Phase 1 Dynamic prediction export | `export` loads the published full Dynamic checkpoint. | Yes, all five sessions. |
| Phase 3 evaluation | `evaluate` loads the published Total-parameter-matched Dynamic checkpoint. | Yes, all five sessions. |
| Phase 4 prediction generation | `predict` loads the released Static and Total-parameter-matched Dynamic checkpoints; `extended-predict` additionally loads the auxiliary validation-matched checkpoint. | Yes, the pilot session. |

Released checkpoints remove the need to repeat encoding-model training; they do not replace stimulus, neural-response, behavior, pupil, or session metadata files.

The released GPFA objects support integrity checking and independent reuse. The current formal Phase 2 `run` and Phase 4 `gpfa` commands regenerate GPFA files under their ignored `outputs/pilot/` directories, and dependent CLI stages read those generated locations. The current CLIs do not provide a restore command that copies the top-level released GPFA files into those output locations.

## 13. Tests and validation checks

The repository includes focused unit and contract tests for implementation invariants; scientific reliability and statistical evidence are documented separately in [GPFA Validation](GPFA_VALIDATION.md) and [Results](RESULTS.md).

### No raw Sensorium data required

The following tests use synthetic arrays, temporary metadata, or configuration constants and do not open Sensorium sessions or released artifacts:

```text
# Encoding environment: Phase 3 configuration-only contract
python -m pytest experiments/03_parameter_matching/tests -m "not data" -q

# Analysis environment: synthetic GPFA/metric and Phase 4 tests
python -m pytest experiments/02_gpfa_reliability/tests -q
python -m pytest experiments/04_model_comparison/tests -q
```

Phase 1 currently has no raw-data-independent pytest test.

### Raw Sensorium data required

The data-marked Phase 1 tests instantiate the official five-session loaders and construct the two architectures. The data-marked Phase 3 test verifies the real five-session data and split contract.

```text
# Encoding environment, after installing all five sessions
python -m pytest experiments/01_baselines/tests -m data -q
python -m pytest experiments/03_parameter_matching/tests -m data -q
```

These tests do not require a sixth legacy session.

### Artifact and integration requirements

No committed pytest test currently loads a released neural-network checkpoint, frozen GPFA object, or generated prediction archive. Artifact-dependent integration occurs through the formal phase commands: evaluation and prediction require the applicable checkpoints; Phase 2 saturation requires the preceding Phase 2 `run` outputs; and downstream Phase 4 stages require their earlier protocol, prediction, GPFA, or extended-prediction outputs. Consult the relevant Phase README for the exact dependency order.

Tests cover selected contracts such as session/configuration identities, tensor and architecture shapes, GPFA interfaces, metric behavior, split logic, and temporal-perturbation mechanics. Passing them is not evidence by itself for the scientific conclusions.

## 14. Reproducibility boundaries

- Raw Dynamic Sensorium data are externally hosted and not redistributed.
- Hidden `final_test_main` neural responses are unavailable locally; local oracle evaluation is not the same as the official hidden benchmark.
- Large prediction archives and other intermediate outputs are generated locally and excluded by `.gitignore`; curated compact tables are retained under `results/`.
- Current encoding-model training uses a single seed.
- Fixed seeds improve repeatability, but exact floating-point values may vary slightly across compatible hardware, CUDA, BLAS, and package builds.
- Released state dictionaries still require the matching model configuration and Sensorium metadata to reconstruct session-specific readouts.

## 15. Documentation map

- [Project README](../README.md)
- [Methods](METHODS.md)
- [Results](RESULTS.md)
- [GPFA Validation](GPFA_VALIDATION.md)
- [Design Rationale](DESIGN_RATIONALE.md)
- [Detailed Q1–Q6 evidence](results/Q1_Q6_ANSWERS.md)
- [Phase 1 README](../experiments/01_baselines/README.md)
- [Phase 2 README](../experiments/02_gpfa_reliability/README.md)
- [Phase 3 README](../experiments/03_parameter_matching/README.md)
- [Phase 4 README](../experiments/04_model_comparison/README.md)
- [Internally locked GPFA protocol](supplementary/protocols/GPFA_PROTOCOL_LOCKED.md)
- [Detailed model reports](supplementary/model_reports/)
- [Experiment matrix](supplementary/implementation/EXPERIMENT_MATRIX.md)
