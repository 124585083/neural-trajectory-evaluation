# Models, replay inputs and acquisition

The local project contains model parameters and replay inputs described in the public [identity inventory](manifest.json). The payload directories below are Git-ignored and available only in the local installation.

| Directory, relative to the repository root | Contents | Supported starting point |
|---|---|---|
| `models/sensorium/retrained_v1/` | Source-run Static and reduced Dynamic checkpoints, main q4 GPFA, five normalization files, neuron/split identities, original control records and aligned prediction caches | Retained Sensorium evaluation from frozen caches |
| `code/sensorium/outputs/checkpoint_matching_v1/` | Selected Dynamic epoch 63, the complete 91-candidate score table, frozen selection and aligned validation/test prediction caches | Current Goal 05; supply this directory as `--matching-root` alongside the source-run bundle |
| `models/sensorium/historical_full_dynamic/` | Original full Dynamic checkpoint wrapper and its training/evaluation records | Historical identity and local-score evidence; separate from the current reduced Dynamic |
| `models/mental_pong/current/source_artifacts/` | 200 FA50 fits, 200 GPFA50 fits, preprocessing, both-half latents, 400 observed readouts/predictions, labels, splits and control inputs | Saved-latent and saved-readout replay of the final condition-average study |
| `models/mental_pong/current/mini/` | Four animal/representation cases at split 0 and random allocation 0 | Small numerical replay, including optional OLS refitting |

The checkpoint control is currently stored at its executed output path; `models/sensorium/checkpoint_matching_v1/` below is an optional copy destination. Its compact [public evidence](../results/sensorium/evidence/04_model_comparison/checkpoint_matching_v1/) supplies the selection and measured results.

Each prepared bundle has a file inventory with exact byte counts and SHA-256 checksums. The copies preserve the original parameters and scientific source records. Representation parameters and preprocessing are retained separately from latents, distinguishing neural-input inference from replay using cached features.

## Run with the installed local inputs

Use the analysis environment specified by each [Sensorium](../code/sensorium/README.md) or [Mental-Pong](../code/mental_pong/README.md) code contract. Run from the repository root and choose output directories that do not contain existing results:

```shell
python code/sensorium/run.py --check
python code/sensorium/run.py --run --output-dir /path/to/new-sensorium-replay
# Include Goal 05 when the separate completed checkpoint bundle is installed
python code/sensorium/run.py --check --matching-root /path/to/checkpoint-matching-v1
python code/sensorium/run.py --run --matching-root /path/to/checkpoint-matching-v1 --output-dir /path/to/new-complete-sensorium-replay
python code/mental_pong/scripts/prepare_local_bundle.py --check
python code/mental_pong/run.py --mini-replay --ols-refit --output /path/to/new-mental-pong-replay
```

Without `--matching-root`, Sensorium recomputes goals 02, 03, 04 and 06 and reports Goal 05 as `not_supplied_not_recomputed`. The checkpoint bundle supplies its separately selected weights, candidate/selection records and development/test prediction caches.

Both project entries locate their installed source inputs by default and accept explicit input-root options. They recompute scores from the specified frozen artifacts. Missing payloads produce an availability error, with no fallback to published scores or replacement-model training.

The recorded source-run verification scope and its limits are in [verification.json](verification.json). The current Dynamic checkpoint-matching control has a separate candidate inventory, selected checkpoint and prediction/result records; the original bundle alone cannot establish that new result. See [checkpoint-matching setup](../results/sensorium/05_response_matching/SETUP.md). The dated verification distinguishes cached replay, small OLS refits, original neural-input inference and a complete rebuild from provider data.

## What a public reader can obtain

[acquisition.json](acquisition.json) records checked upstream links, versions, checksums where known, and the missing delivery route for current fitted artifacts. Original data remain available from their providers. Downloading those data does not supply this study's frozen parameters, labels or prediction caches.

The current fitted bundles have no established public download URL. Redistribution of the numerical payload remains unresolved under the documented publication scope. Project-owned replay code, inventories, source attribution and verification summaries can be distributed separately. This status is not a legal determination that all trained parameters are adapted material. See [Third-Party Notices](../THIRD_PARTY_NOTICES.md).

For an authorized holder restoring a preserved source bundle into a checkout without installed payloads:

```shell
python code/sensorium/scripts/prepare_local_bundle.py --source /path/to/preserved-retrained-v1 --destination models/sensorium/retrained_v1 --copy
# Or copy both completed Sensorium bundles into separate new destinations
python code/sensorium/scripts/prepare_local_bundle.py --source /path/to/preserved-retrained-v1 --destination models/sensorium/retrained_v1 --matching-source /path/to/completed-checkpoint-control --matching-destination models/sensorium/checkpoint_matching_v1 --copy
python code/mental_pong/scripts/prepare_local_bundle.py --source-root /path/to/preserved-mental-pong-artifacts
```

The Sensorium wrapper exposes `sensorium_runtime.bundle` (also available as `python -m sensorium_runtime.bundle` when the package is installed). The two-bundle command keeps the source-run identities unchanged and adds a separate portable control containing the complete candidate score table and frozen selection, the selected checkpoint at its recorded relative path, and development/test prediction caches. Replaying the frozen control does not require copying every candidate weight. Choose either Sensorium copy command above with new destination directories.

These tools preserve source files, check copied bytes and omit unrelated pilot experiments. Their `--help` describes verification and size-estimate options. They prepare local copies only; permission and publication remain separate steps. The bundles exclude raw datasets, complete pilot histories and cancelled auxiliary fits. Both commands require access to the current fitted artifacts at the specified source location.

Historical Sensorium weights remain in earlier Git/LFS history. They are not the current `retrained_v1` inputs and are not promoted as a newly cleared distribution route. No history was rewritten.
