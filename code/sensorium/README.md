# Sensorium — Code and Reproduction

This directory contains the code used for the Sensorium analyses in **neural-trajectory-evaluation**. The main comparison belongs to `retrained_v1`, which compares a Static encoder and a reduced Dynamic encoder in a shared, frozen neural-data-defined Gaussian Process Factor Analysis (GPFA) space. The current response-matching control uses the separate `checkpoint_matching_v1` protocol to select a Dynamic training checkpoint.

**The supported one-command workflow starts from saved, aligned neural responses and encoder predictions.** It recomputes the source-run GPFA measurements, model comparisons and retained diagnostics using frozen models and settings. The source-run local replay was verified on 12 October 2026. The new checkpoint-matching control requires its own selected weights, prediction caches and result verification. The numerical bundle is supplied separately from a public clone. A one-command rebuild from the provider’s videos and neural data remains unverified.

For the research questions and findings, see [Sensorium results](../../results/sensorium/) and [models and data](../../results/sensorium/MODELS_AND_DATA.md).

## 1. Run the frozen analysis

From the repository root, with the **analysis environment** active and the authorized frozen bundle installed:

```shell
# Check the source-run bundle without producing results
python code/sensorium/run.py --check

# Recompute goals 02, 03, 04 and 06 from the source bundle
python code/sensorium/run.py --run --output-dir code/sensorium/outputs/replay_main

# Check and include the completed checkpoint control (Goal 05)
python code/sensorium/run.py --check --matching-root code/sensorium/outputs/checkpoint_matching_v1
python code/sensorium/run.py --run --matching-root code/sensorium/outputs/checkpoint_matching_v1 --output-dir code/sensorium/outputs/replay_all
```

Choose a new or empty output directory. The default artifact root is `models/sensorium/retrained_v1/`. To use the same authorized bundle elsewhere, add `--artifact-root /path/to/authorized-bundle` to either command. The separate `--matching-root` must contain the frozen checkpoint selection and aligned development/test prediction caches. It may instead point to a restored bundle such as `models/sensorium/checkpoint_matching_v1/`. Without this option, Goal 05 is explicitly `not_supplied_not_recomputed`; the replay does not substitute the superseded control. Run `python code/sensorium/run.py --help` for the current command-line interface.

The entry point checks input files and identities before running. Missing or mismatched inputs cause an error. **Encoder weights, GPFA parameters, encoder predictions, trial splits and intervention strengths remain fixed during replay.** The workflow covers the retained main GPFA analyses; the auxiliary sensitivity experiments were withdrawn.

### What can be reproduced?

| Available starting material | Supported operation | Limit |
|---|---|---|
| Public code, configurations and published evidence | Inspect settings, run code tests and check recorded provenance | Numerical execution requires separately obtained neural arrays, fitted weights and prediction caches |
| Authorized `retrained_v1` frozen bundle, including aligned and intervention caches | Recompute source-run scores, GPFA posterior trajectories and retained diagnostics | Starts from fixed predictions and GPFA parameters; new Goal 05 also needs the separately selected checkpoint and its caches |
| Source bundle plus completed checkpoint-control bundle | Recompute goals **02–06** with `--matching-root` | Requires the frozen selected checkpoint and both development/test matched-prediction caches |
| Authorized provider data, frozen weights and upstream source environments | Use the existing individual preparation, prediction and evaluation stages | A complete fresh-clone rebuild of all six goals remains unverified |

The historical *full Dynamic* reproduction has separate training and local evaluation records from the reduced Dynamic model used in the current comparison. The frozen replay preserves the historical experiment, whose evidence consists of local evaluation; an official hidden-test score remains unverified.

## 2. Input data and frozen artifacts

### Provider-data interface

The original session folders contain video stimuli, neural responses, behavior, pupil traces, trial tiers and neuron metadata. The recorded arrays use the following layouts:

| Data | Recorded layout |
|---|---|
| Video | `[36, 64, time]` |
| Neural response | `[neuron, time]` |
| Behavior and pupil traces | `[2, time]` each |

The encoder adapter builds inputs of shape `[batch, 3, time, 36, 64]`. Two behavior traces join the visual input channels; pupil coordinates also supply the shifter. The recorded inference uses the first 300 frames, with evaluation on original frames **50–299**.

For *raw-data* inference stages, use an authorized provider-data location (`SENSORIUM_DATA` / `--data-root`) and the pinned Sensorium and neuralpredictors source checkouts (`SENSORIUM_SOURCE`, `NEURALPREDICTORS_SOURCE`). **The recommended cache-based replay uses the frozen bundle directly.** A fresh-install recipe for that advanced route remains to be validated.

### Cached-replay interface

Each aligned prediction NPZ contains `neural`, `static` and `dynamic` arrays of shape `[trial, 250, neuron]`, together with session, trial, movie/condition, dataset-index, neuron-order, frame and prediction-precision metadata. Five-session response scoring uses each session's neurons; the pilot trajectory analyses use a fixed **512-neuron mapping**, **six test movies** and **58 test trials** from session `29515-10-12`.

The installed bundle contains the following main artifacts, with paths relative to `models/sensorium/retrained_v1/`:

| Artifact | Role |
|---|---|
| `encoders/static/best.pt`, `encoders/dynamic/best.pt` | Frozen seed-42 encoder checkpoints (epochs 59 and 109) and their identity records |
| `data/normalization/*.npz`, `data/preprocessing.json`, `data/split_manifest.json` | Training-derived scales, data roles, trial identities and neuron ordering |
| `gpfa/frozen_gpfa.pkl`, `gpfa/frozen_gpfa_arrays.npz` | Fixed four-dimensional GPFA model, scale and observation/query time indices |
| `frozen/frozen_analysis.json`, `frozen/frozen_control.npz` | Immutable source-run evaluation settings and archived control payload |
| `test/<session>/aligned_predictions.npz` and the pilot `development/<session>/aligned_predictions.npz` | Neural and model predictions from which the recommended replay starts |
| `diagnostics/encoder/`, `diagnostics/gpfa/`, `diagnostics/reversal_predictions.npz` | Original intervention, repeat-sampling and reversal inputs for the retained controls |

The exact inventory, file sizes and SHA-256 values are in the [model manifest](../../models/manifest.json); [artifact preparation](../../models/README.md) and the [acquisition record](../../models/acquisition.json) describe access. The manifest records **129 files (8.58 GiB)** in the current local bundle, plus a separate 15-file historical full-Dynamic bundle. Both bundles are Git-ignored. A public download route for the fitted numerical payload remains unconfirmed; the frozen caches are separate inputs from the provider data.

Keep the original relative paths and matching identity records when restoring a bundle. Cached replay requires the aligned predictions, preprocessing and control files in addition to encoder weights. Load serialized GPFA pickle objects only from a trusted, hash-verified source.

## 3. Python environments

Separate recorded environments support analysis and encoder execution. The listed versions describe the original working systems; portability to a clean installation remains to be verified.

| Component | Analysis and cached replay | Encoder training / prediction |
|---|---|---|
| Python | 3.11.9 | 3.11.9 |
| NumPy | 1.26.4 | 1.26.4 |
| SciPy | 1.14.1 | 1.13.1 |
| pandas | 2.2.3 | 2.0.0 |
| PyTorch | 2.5.1+cu124 | 2.5.1+cu124 |
| Additional requirements | scikit-learn 1.5.2; CPU GPFA calculations | CUDA 12.4; nnfabrik 0.2.1; neuralpredictors 0.3.0 |

The encoder environment was recorded with an RTX 4090 Laptop GPU. CPU-only encoder inference remains an unverified execution option. The package defines separate `analysis` and `encoding` extras with conflicting recorded pandas requirements; **install them in separate environments**. Consult [pyproject.toml](pyproject.toml) and the [environment records](environments/) for dependency details. The pinned source commits are Sensorium 2023 `0e02656220e84a50f3be1b92d6f66c2f9ccd51ef` and neuralpredictors `efdda679596517fad95d71f36d0385d7450dd207`.

The original `trajectory_reliability.gpfa.GaussianProcessFactorAnalysis` import path is retained for GPFA pickle compatibility.

## 4. Outputs

The frozen command writes its results and input/output manifests to the chosen new directory, principally under `evaluation/`, `measurement/`, `temporal_metric_tests/` and `temporal_weight_controls/`. With `--matching-root`, `evaluation/pilot/response_matching_*` contains the new checkpoint-control scores, per-movie comparisons, intervals and latents; the pilot summary records its selected checkpoint and test-match status. It regenerates five retained figures, including repeat averaging and weight attenuation. Historical training figures and legacy atlases remain separate saved outputs.

| Goal | Results and methods |
|---|---|
| 01 — Model validation (existing records only) | [Setup](../../results/sensorium/01_model_validation/SETUP.md) · [Results](../../results/sensorium/01_model_validation/RESULTS.md) |
| 02 — Trajectory measurement and reliability | [Setup](../../results/sensorium/02_trajectory_measurement/SETUP.md) · [Results](../../results/sensorium/02_trajectory_measurement/RESULTS.md) |
| 03 — Static/Dynamic model comparison | [Setup](../../results/sensorium/03_model_comparison/SETUP.md) · [Results](../../results/sensorium/03_model_comparison/RESULTS.md) |
| 04 — Temporal metric tests | [Setup](../../results/sensorium/04_temporal_metric_tests/SETUP.md) · [Results](../../results/sensorium/04_temporal_metric_tests/RESULTS.md) |
| 05 — Response matching | [Setup](../../results/sensorium/05_response_matching/SETUP.md) · [Results](../../results/sensorium/05_response_matching/RESULTS.md) |
| 06 — Temporal/Spatial weight controls | [Setup](../../results/sensorium/06_temporal_weight_controls/SETUP.md) · [Results](../../results/sensorium/06_temporal_weight_controls/RESULTS.md) |

Published scientific records remain under [Sensorium evidence](../../results/sensorium/evidence/) and [figures](../../results/sensorium/figures/). **New output is written separately, preserving published reference data.** The older `scripts/make_core_figures.js` uses an external `sharp` dependency to render historical tables as figures. Its output belongs to the archived analysis.

### Redraw the documentation figures

With NumPy, pandas and Matplotlib installed, run:

```shell
python code/sensorium/make_documentation_figures.py
```

This draws the current method diagrams and result panels from retained evidence tables into `results/sensorium/figures/documentation/`. The checkpoint-matching figure reads its separate completed evidence: 91 candidates, selected epoch 63, validation/test response gaps and all five paired trajectory intervals. It uses saved scores and intervals; model weights and analysis outputs remain unchanged. The figure manifest records source files and checksums. Use `--output-dir` to write an alternative copy. These documentation figures are separate from the five figures produced by the frozen numerical replay.

## 5. Other code entry points

Use `run.py` for the default frozen replay. The original stage scripts support individual analysis steps, including stages that **train, fit or select parameters**. Those stages produce new fitted or selected inputs and should run in a workflow separate from frozen replay.

| Script | Purpose |
|---|---|
| `run_checkpoint_matching.py` | Score all declared checkpoints, freeze the validation-selected candidate, export its predictions and evaluate the control. See the stage commands below. |
| `run_retrained.py` | Source-run stage pipeline. `predict-test` uses frozen encoder weights and `evaluate` scores saved predictions. Training, preparation and GPFA fitting produce new artifacts; the source protocol remains preserved. |
| `run_supplementary.py` | Diagnostic-stage analysis, including stages that may select spatial intervention strengths; use `run.py` for frozen replay. |
| `run_unified.py` | Historical `unified_v1` workflow; its source protocol and results are archived. |
| `scripts/prepare_local_bundle.py` | Checks and prepares an authorized project-local frozen bundle without publishing its numerical payload. |

Core model, preprocessing, GPFA and evaluation functions remain in `src/`, with configurations in `configs/` and checks in `tests/`. The source-run replay uses the selected frozen artifacts. The checkpoint-matching workflow additionally records the complete declared candidate set and validates every candidate before freezing its selected epoch.

### Current Dynamic checkpoint matching

[configs/checkpoint_matching_v1.json](configs/checkpoint_matching_v1.json) declares all completed epochs through the earlier of epoch 119 and verified native early stopping. The original training policy remains `max_iter=200`, `patience=5` and four learning-rate decay cycles. A shorter cohort requires a completed training summary with stop reason `patience_cycles_exhausted`; a pause or interrupted epoch does not qualify. The rule and its timing before full-pilot matching scores and new test outputs are recorded in `protocol_revision.json`. The new training run retains its own checkpoint identities because the original intermediate weights were unavailable. Each checkpoint is scored on all 66 pilot validation trials and a fixed common finite subset of the selected 512 neurons. Ranking uses the absolute gap from Static mean per-neuron Pearson correlation, then earlier epoch and checkpoint SHA-256. The selected identity is frozen before test inference; the tolerance is 0.005. Test matching retains the validation common-finite neuron cohort and reports its finite test support; any undefined cohort correlation makes matching unevaluable. GPFA and other metrics continue to use all 512 neurons.

The training replay is implemented in [checkpoint_recovery.py](src/trajectory_model_eval/checkpoint_recovery.py). It requires the preserved original training run, including its initial Dynamic state, training identity and full training curve. The smaller frozen replay bundle alone does not supply those training inputs. The new run verifies the original initialization, but its later weights are not asserted to reproduce the original intermediate states. It writes `recovery/candidates/` and `recovery/candidate_manifest.json`; `--resume` continues a verified epoch-boundary state.

After the complete candidate set exists, use the **encoding environment** for validation inference and selected-checkpoint prediction. From the repository root, the following command uses the default source bundle, matching root `code/sensorium/outputs/checkpoint_matching_v1` and its `recovery/` subdirectory:

```shell
python code/sensorium/run_checkpoint_matching.py score-candidates --data-root /path/to/sensorium-data --sensorium-source /path/to/sensorium-source --neuralpredictors-source /path/to/neuralpredictors-source
python code/sensorium/run_checkpoint_matching.py export-selected --data-root /path/to/sensorium-data --sensorium-source /path/to/sensorium-source --neuralpredictors-source /path/to/neuralpredictors-source
```

`score-candidates` requires the complete declared contiguous cohort—up to 119 epochs, or all epochs through verified native early stopping—and writes the candidate table and frozen selection. It rejects a cohort shortened by a pause or interruption. `export-selected` verifies that selection and writes matched predictions for the pilot development and test trials. `--watch` can wait for recovery to finish before scoring candidates. The paths can be changed with `--source-root`, `--recovery-root` and `--matching-root`; candidate files must remain inside the matching root.

Then use the **analysis environment** to score the saved predictions:

```shell
python code/sensorium/run_checkpoint_matching.py evaluate
```

The default evaluation destination is `<matching-root>/evaluation`. Use `--output-dir` for a new alternative destination. The `run` command combines candidate scoring, selected prediction export and evaluation when one environment provides all required dependencies; it still requires `--data-root`. Frozen replay with `run.py --matching-root ...` reuses the completed caches and performs no candidate selection.

The sidecar configuration and its new result bundle are separate from the original `retrained_v1.json`, checkpoint identities and GPFA. The completed run stopped natively at epoch 91 and selected epoch 63 on validation. Its test mean response-r gap is 0.0004839494; both primary trajectory contrasts remain positive, while speed remains unresolved. The current [setup](../../results/sensorium/05_response_matching/SETUP.md) gives formulas and data roles; [results](../../results/sensorium/05_response_matching/RESULTS.md) reports the new scores and intervals. Archived Goal 05 records remain available for provenance and are not substituted for checkpoint results.

## 6. Recorded source-run verification and current control

The recorded local replay on **12 October 2026** used only the installed project-local bundle; the original training run, baseline workspace and old repository were denied as fallback input roots. Goals **02–06** were recomputed and **five figures** produced. Goal **01** was checked against saved records rather than re-executed. The replay used saved encoder predictions, fixed GPFA parameters and frozen control settings.

Of **36 regenerated CSV files, 35** matched the saved tables exactly. In the remaining `spatial_response_matches.csv`, scientific scores, support and matching states agreed; three historical `original_*` diagnostic fields differed because the newer scoring path records float64 correlations. The largest recorded difference was **0.000553**. All **15 checked main-latent and reversal arrays** matched exactly. The spatial-zero-weight endpoint remains `NA / numerically_degenerate`, with response matching `not_evaluable`; other matched-control outcomes remain unchanged. These results describe the **12 October source-run verification**, including the now-archived Goal 05. They do not verify the newly executed checkpoint-matching control. See [verification details](../../models/verification.json).

The new checkpoint control separately completed native-stop verification, all 91 candidate validation scores, the epoch-63 freeze, aligned prediction export and test scoring with 2,000 paired movie-bootstrap draws. Its [published records](../../results/sensorium/evidence/04_model_comparison/checkpoint_matching_v1/) bind the actual selected weights and prediction caches. Across the specified analysis and encoding environments, **218 checks passed**; the two checks requiring `neuralpredictors` were executed in the encoding environment.

Public reproduction still requires an available fitted bundle, a verified rebuild from provider video and neural arrays, and installation checks on other systems. The historical full-Dynamic work also lacks a verified official hidden-server score. The demonstrated scope is local replay from frozen caches.
