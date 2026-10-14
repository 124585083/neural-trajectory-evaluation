# Mental-Pong: Code and Reproduction

This directory contains the analysis code for the Mental-Pong study. The current scientific analysis uses **FA50 or GPFA50 representations of condition-mean dorsomedial frontal cortex (DMFC) activity**, followed by a fixed linear position readout. The readouts recover physical ball position and compare its reconstruction with that of a mean-paddle-endpoint candidate.

**A public checkout supports inspection of the analysis and its saved evidence.** Numerical replay additionally requires fitted representations and cached neural data, which remain available only in the local installation. Recorded verification covers replay from saved material; a complete rebuild from public data remains to be validated.

For the scientific methods and conclusions, see [models and data](../../results/mental_pong/MODELS_AND_DATA.md), [task-motion analysis](../../results/mental_pong/01_task_motion/SETUP.md) and [behavior analysis](../../results/mental_pong/02_behavior_relation/SETUP.md).

## 1. Available reproduction routes

| Route | Recalculation performed | Required inputs | Scope |
|---|---|---|---|
| Inspect / verify | Display command help and historical execution requirements; check retained scientific file identities and report artifact availability | Organized public source and evidence | Read-only identity and availability checks through `--verify` |
| Saved-head replay | Apply one saved random-endpoint OLS head to saved 50-D features and score it | Authorized local source-artifact bundle | Selected-case scoring from fixed FA/GPFA features and fixed OLS coefficients |
| Mini replay | Recalculate own-target, 2×2, condition-mismatch, random-endpoint and geometry scores for four animal/representation cases at **split 0, random allocation 0** | Authorized seven-file mini bundle | One selected case from the 100-split, 1,000-randomization analysis |
| Mini replay with `--ols-refit` | Also refit 12 small OLS heads on retained training rows and check their held-out predictions | Mini bundle and refit dependencies | OLS refitting with saved FA/GPFA features held fixed |
| Full saved-readout audit (previously executed) | Apply 400 original OLS readouts to their saved held-out features; compare predictions and scores | Full local bundle and audit scripts | Readout-level audit using existing representations and B/C control records |
| Original analysis runners | Implement condition labels, representation/readout stages and the complete control procedures | Authorized datasets, exact source identities and an isolated writable workspace | Individual stages; one-command reproduction from a fresh public clone awaits validation |

The completed project's scientific status is `CLOSED_EXPLORATORY_WITH_LIMITATIONS`. Public software and artifact availability are assessed separately. The current analysis is `condition_endpoint_fa_gpfa_v1`, followed by `closeout_v1`; historical author FA50, GPFA64 and Ridge analyses use different protocols.

## 2. Quick start

Run the following commands from the **repository root**, using the Python environment described below.

### Inspect the available commands

```sh
python code/mental_pong/run.py --help
python code/mental_pong/run.py --full-plan
python code/mental_pong/run.py --verify
```

`--full-plan` displays the historical full-run requirements. `--verify` checks the recorded scientific CSV/JSON/figure identities and retained goal documents, and reports missing numerical inputs. Both commands inspect existing records; the replay commands below perform the numerical calculations while preserving source files.

### Check an installed full artifact bundle

```sh
python code/mental_pong/scripts/prepare_local_bundle.py --check
```

The expected local source bundle is `models/mental_pong/current/source_artifacts/`. This Git-ignored directory requires separate installation; a fresh public clone reports its payload as missing. The fitted files currently require authorized local access because an automatic public download is unavailable.

### Replay one saved random-endpoint head

```sh
python code/mental_pong/run.py --replay --animal mahler --representation FA50 --q 0 --split 0
```

This reads the registered local bundle and recalculates the selected correlation, RMSE and geometry skill from saved OLS coefficients and latent features. The head, latents and model files remain fixed. A missing artifact bundle produces an availability error.

### Run the small numerical example

With an authorized mini bundle installed at `models/mental_pong/current/mini/`, install the recorded refit requirements and choose a **new or empty** output directory:

```sh
python -m pip install -r code/mental_pong/mini_replay/requirements-refit.txt
python code/mental_pong/run.py --mini-replay --ols-refit --output replays/mental_pong_mini_01
```

Omit `--ols-refit` to score the saved heads unchanged. If an authorized mini bundle resides elsewhere, add `--mini-bundle /path/to/authorized-mini`. The mini covers Mahler and Perle with FA50 and GPFA50 at one split and one random allocation. It verifies this selected example; the complete 100-by-1,000 control study has a separate reproduction scope.

## 3. Input data and model artifacts

The upstream release is [Rajalingham, Sohn and Jazayeri's Mental-Pong data (Zenodo v1)](https://zenodo.org/records/13952210). The neural source files contain processed **condition-mean pseudopopulations** assembled across recording sessions. For each animal, the release includes 79 physical conditions, 100 time bins at 50 ms resolution and two stored neural halves. The recorded aligned unit counts are 518 (Mahler) and 2,058 (Perle). Current readouts use half1 and held-out physical conditions; 78 conditions and 3,369 condition-time samples have valid common position labels.

For a full representation rebuild, the code also needs the released neural arrays, global neuron identities, masks, fixed 100 condition splits, source geometry and behavioral metadata. The current behavioral candidate uses a **condition-mean terminal paddle position** and known physical anchors. The constructed label is a condition-level reconstruction target; relating it to an internal estimate requires additional evidence. Individual behavioral trials have no recoverable membership link to these neural means.

The local source bundle was recorded as **2,237 files (2,729,908,953 bytes; 2.54 GiB)**. The important classes are:

| Saved file class | Count | Role |
|---|---:|---|
| `fa50.joblib` | 200 | Fitted FA50 models, one for each animal and split |
| `gpfa50.npz` | 200 | Fitted GPFA50 loadings, offset, diagonal noise and shared time scale |
| `preprocessing.npz` | 200 | Training-side neuron selection, mean/std, neuron order and split identities |
| `latents.npz` | 200 | Saved 50-dimensional features for both neural halves and both representations |
| OLS readout NPZ | 400 | Fitted objective and candidate xy heads, held-out support and identities |
| Held-out prediction NPZ | 400 | Original readout predictions for checking numerical agreement |
| B/C controls and labels | Additional recorded files | Condition permutations, endpoint assignments, randomized OLS weights/scores, physical geometry and masks |

The [artifact manifest](../../models/manifest.json) and [source registry](integration/artifact_registry.csv) record paths, checksums and formats. The source bundle was assembled locally with exact checksums; the separate mini bundle contains seven allowlisted files. **Neither bundle is included in Git or currently has a documented, tested public download URL.** Restoring this analysis requires its fitted parameters, masks, labels and caches in addition to the original Zenodo data.

A holder of the original authorized source tree can inspect the preparation tool's scope with:

```sh
python code/mental_pong/scripts/prepare_local_bundle.py --source-root /path/to/authorized-source --estimate
```

The tool's subsequent `--source-root /path/to/authorized-source` operation copies its checked allowlist into the project-local bundle, preserving the existing data and fits. An alternative source tree can be selected through `MENTAL_PONG_SOURCE_ROOT` or the `artifact_roots.source_pilot` field illustrated in [paths.example.json](configs/paths.example.json). The `source_pilot` identifier is a **runtime path alias** for the authorized artifact tree, not a separate code package. Retain the original source-edition configuration hashes when using translated or path-adjusted public copies.

The dataset release lists **CC BY-NC-ND 4.0**. The distribution status of individual derived artifacts has not been resolved; the numerical bundles remain withheld pending an artifact-specific basis for sharing. This is not a general conclusion that fitted weights can never be published. See [Third-Party Notices](../../THIRD_PARTY_NOTICES.md).

## 4. Environment and code organization

The recorded analysis environment used Python **3.11.9**, NumPy **1.26.4**, SciPy **1.14.1**, pandas **2.2.3**, scikit-learn **1.5.2** and joblib **1.4.2**. Other recorded versions are in [requirements.lock.txt](requirements.lock.txt). The mini scorer needs Python/NumPy; mini OLS refitting uses [requirements-refit.txt](mini_replay/requirements-refit.txt). The logged mini runs were on Windows. Validation on clean Linux/macOS installations and from public raw data remains outstanding.

The main code locations are:

- [`run.py`](run.py): inspection, evidence verification, saved-head replay and mini-replay entry point.
- [`trajectory_project/trial_endpoint_fa_gpfa_v1/`](trajectory_project/trial_endpoint_fa_gpfa_v1/): FA/GPFA representation construction, training-side preprocessing and inference.
- [`trajectory_project/condition_endpoint_fa_gpfa_v1/`](trajectory_project/condition_endpoint_fa_gpfa_v1/): condition-mean target preparation, frozen-split OLS fitting and held-out scoring.
- [`trajectory_project/closeout_v1/`](trajectory_project/closeout_v1/): fixed-head condition mismatch (B), random-endpoint OLS refit (C), geometric references and figure generation.
- [`mini_replay/`](mini_replay/) and [`scripts/`](scripts/): bounded numerical example, artifact checks and private preparation tools.

Historical report templates may have separate rendering dependencies. Saved CSVs supply the values used to regenerate result figures and reports. Neural inference requires the corresponding upstream stages. Run historical full pipelines only in a separately configured, writable workspace to preserve the original frozen artifacts.

### Redraw the documentation figures

With NumPy, pandas and Matplotlib installed, run from the repository root:

```sh
python code/mental_pong/make_method_figures.py
python code/mental_pong/make_result_figures.py
```

These commands draw the method diagrams and saved-result panels into `results/mental_pong/figures/documentation/`; `--output-dir` selects another destination. Result panels use retained scientific tables, including the objective-head condition-mismatch results. Readouts, latents and randomizations remain fixed. The future-study diagram illustrates a proposed experiment.

## 5. Recorded numerical verification

The verification completed on **12 October 2026** reported the following:

- All **2,237** installed source files and project-local copies passed their recorded SHA-256 checks. The verified files contain the fitted FA/GPFA parameters and preprocessing arrays.
- All **400** observed OLS readouts were applied to the saved half1 latents across both animals, both representations and 100 splits. Their **400 held-out prediction files matched exactly**. Rescoring reproduced **19,200 2×2 rows** and **4,800 own-target rows** with maximum absolute discrepancy below **1.08 × 10⁻¹⁴**.
- The installed mini replay covered four animal/representation cases and **604 score rows**, with maximum metric discrepancy **1.64 × 10⁻¹¹**. Its **12 optional OLS refits** reproduced stored held-out predictions to **1.51 × 10⁻¹⁴**.
- The local tests also covered missing artifacts, checksum tampering, path containment and preservation of source files; **46 targeted tests passed**. Network and original pilot-data access were denied during these checks.

These **recorded earlier executions** establish numerical reproducibility *from saved representations, heads and selected controls*. The B and C functional examples cover only **split 0 / allocation 0**. The complete **1,000-randomization × 100-split** distributions are retained from the original execution, outside this verification's replay scope. This verification reused the saved FA/GPFA fits and latent arrays. The `--verify` command checks metadata and file identities; numerical validation uses the separate replay routes.

The released neural means were previously filled across conditions and times. Current GPFA prefix filtering uses only the available prefix of the **released input**, retaining the effects of that upstream preprocessing. The analysis therefore retains **`raw_preprocessing=fail`**. Raw-input independence and within-condition behavioral decoding require the missing pre-imputation responses and paired trials.

## 6. Results and remaining reproduction work

The generated numerical files belong in a new output directory, separate from the retained evidence under [`results/mental_pong/evidence/`](../../results/mental_pong/evidence/). For the saved conclusions, see:

- [Goal 01 — Task-motion setup](../../results/mental_pong/01_task_motion/SETUP.md) and [results](../../results/mental_pong/01_task_motion/RESULTS.md): objective position reconstruction, phases, fixed-head mismatch and geometry baseline.
- [Goal 02 — Behavior relation setup](../../results/mental_pong/02_behavior_relation/SETUP.md) and [results](../../results/mental_pong/02_behavior_relation/RESULTS.md): own-target and cross-target comparisons, randomized endpoints, null-support limitations and geometry.
- [Research questions](../../results/mental_pong/QUESTIONS.md) and [next steps](../../results/mental_pong/NEXT_STEPS.md): what the analyses establish, why behavioral specificity remains unresolved and what paired neural–behavior observations are needed next.

**Remaining reproduction gaps** are access to distributable numerical payloads, a tested neural-response-to-latent rebuild, isolated rerunning of the complete B/C controls, and a clean end-to-end reconstruction on an independent installation. Single-trial neural–behavior pairing and pre-imputation observations would require additional source data beyond the published means. The repository thus provides source code, saved evidence and verified local replay, with full public reproduction as a separate remaining task.
