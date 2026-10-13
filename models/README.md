# Models, replay inputs and acquisition

The local project contains model parameters and replay inputs described in the public [identity inventory](manifest.json). The payload directories below are Git-ignored and available only in the local installation.

| Local directory | Contents | Supported starting point |
|---|---|---|
| `sensorium/retrained_v1/` | Current Static and reduced Dynamic checkpoints, main q4 GPFA, five normalization files, neuron/split identities, frozen controls and aligned prediction caches | Retained Sensorium evaluation from frozen caches |
| `sensorium/historical_full_dynamic/` | Original full Dynamic checkpoint wrapper and its training/evaluation records | Historical identity and local-score evidence; separate from the current reduced Dynamic |
| `mental_pong/current/source_artifacts/` | 200 FA50 fits, 200 GPFA50 fits, preprocessing, both-half latents, 400 observed readouts/predictions, labels, splits and control inputs | Saved-latent and saved-readout replay of the final condition-average study |
| `mental_pong/current/mini/` | Four animal/representation cases at split 0 and random allocation 0 | Small numerical replay, including optional OLS refitting |

Each prepared bundle has a file inventory with exact byte counts and SHA-256 checksums. The copies preserve the original parameters and scientific source records. Representation parameters and preprocessing are retained separately from latents, distinguishing neural-input inference from replay using cached features.

## Run with the installed local inputs

Use the analysis environment specified by each [Sensorium](../code/sensorium/README.md) or [Mental-Pong](../code/mental_pong/README.md) code contract. Run from the repository root and choose output directories that do not contain existing results:

```shell
python code/sensorium/run.py --check
python code/sensorium/run.py --run --output-dir /path/to/new-sensorium-replay
python code/mental_pong/scripts/prepare_local_bundle.py --check
python code/mental_pong/run.py --mini-replay --ols-refit --output /path/to/new-mental-pong-replay
```

Both entries locate the installed project-local inputs by default and accept explicit input-root options. They recompute scores from the specified frozen artifacts. Missing payloads produce an availability error, with no fallback to published scores or replacement-model training.

The numerical verification scope and its limits are recorded in [verification.json](verification.json). It separates replay from cached predictions/latents, small OLS refits, original neural-input inference, and a complete rebuild from provider data.

## What a public reader can obtain

[acquisition.json](acquisition.json) records checked upstream links, versions, checksums where known, and the missing delivery route for current fitted artifacts. Original data remain available from their providers. Downloading those data does not supply this study's frozen parameters, labels or prediction caches.

The current fitted bundles have no established public download URL. Redistribution of the numerical payload remains unresolved under the documented publication scope. Project-owned replay code, inventories, source attribution and verification summaries can be distributed separately. This status is not a legal determination that all trained parameters are adapted material. See [Third-Party Notices](../THIRD_PARTY_NOTICES.md).

For an authorized holder restoring a preserved source bundle into a checkout without installed payloads:

```shell
python code/sensorium/scripts/prepare_local_bundle.py --source /path/to/preserved-retrained-v1 --destination models/sensorium/retrained_v1 --copy
python code/mental_pong/scripts/prepare_local_bundle.py --source-root /path/to/preserved-mental-pong-artifacts
```

These tools preserve source files, check copied bytes and omit unrelated pilot experiments. Their `--help` describes verification and size-estimate options. They prepare local copies only; permission and publication remain separate steps. The bundles exclude raw datasets, complete pilot histories and cancelled auxiliary fits. Both commands require access to the current fitted artifacts at the specified source location.

Historical Sensorium weights remain in earlier Git/LFS history. They are not the current `retrained_v1` inputs and are not promoted as a newly cleared distribution route. No history was rewritten.
