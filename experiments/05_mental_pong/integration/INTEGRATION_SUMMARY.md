# Integration summary

The repository contains the completed Mental-Pong study as `experiments/05_mental_pong`, alongside the four existing Sensorium experiment directories. The publication includes English reports, compact results, preserved labels and splits, the actual Python dependency modules, a path adapter, and evidence indexes. The original integration was prepared for review; Git history records subsequent publication.

Scientific status remains **CLOSED_EXPLORATORY_WITH_LIMITATIONS** and **raw_preprocessing=fail**. Integration checks are recorded separately in [verification.json](verification.json). English editions have new hashes; their originals and the pre-edit Sensorium snapshot remain external and unchanged.

## What was integrated

The current source versions are `condition_endpoint_fa_gpfa_v1` and `closeout_v1`. The required `trial_endpoint_fa_gpfa_v1` code remains a working dependency. English [historical summaries](../archive/HISTORY.md) explain earlier versions without presenting them as current results.

The [repository overview](../../../README.md), [Mental-Pong account](../../../docs/MENTAL_PONG.md), [evidence index](../../../results/study_evidence_index.csv), and [unified future plan](../../../docs/FUTURE_DIRECTIONS.md) connect the studies. The plan is a document only. Existing Sensorium prose and figure labels were revised; examples appear in [editorial changes](editorial_changes.md).

Models, latent values, random mappings, labels, masks, seeds, sample identities, and numerical results retain their source values. Compact scientific files were copied byte for byte where possible. Explanatory text and absolute location fields have separate publication editions. The [migration map](migration_map.csv), [artifact registry](artifact_registry.csv), and [dependency manifest](dependency_manifest.csv) distinguish these cases.

## Checks and scope

The verification record covers source hashes, unchanged numerical files, edited JSON/CSV numerical equivalence, Python syntax and focused adapter tests, saved-score aggregation, own-target/2×2 identity, fixed 39/40 splits, documentation links, language scans, and fresh figure inspection. The main tables agree with their saved per-split scores within floating-point precision.

Four saved random-head cases were replayed, one for each animal/representation combination, with q=0 and split=0 fixed in advance. Coefficients and latent arrays were read from the originals. The largest score discrepancy was about 1.64e-11. No model was fitted, no random allocation was regenerated, and no full analysis was rerun. See the [replay record](saved_replay_verification.json).

All 14 available lightweight Sensorium checks passed using the appropriate existing environments. Mental-Pong synthetic and adapter checks are listed in the verification record. Large-delivery tests skipped during that suite are identified separately from the successful saved replay. The three Sensorium and ten Mental-Pong PNGs received new visual inspection. External all79 PDF atlases remain original artifacts and have not been newly visually certified.

## External dependencies and unverified routes

Large readouts, latent arrays, model weights, B/C arrays, raw/prepared data, and original reports remain external. Configure their storage roots using [paths.example.json](../configs/paths.example.json); actual machine locations belong in ignored `paths.local.json` or environment variables. No data were downloaded or packages upgraded.

One historical user-request attachment is unavailable. Its original recorded hash and unavailable status remain in the registry. Its absence does not remove a scientific result file. The neural means' trial membership and strictly pre-feedback endpoint timing remain unknown, and upstream preprocessing remains unresolved; integration does not repair those scientific limitations.

The saved replay and default checks are portable and read-only. Full execution is a retained historical route requiring a separate writable workspace and complete external inputs. That route has not been exercised end to end after relocation. Historical atlas export additionally requires a configured PDF runtime and Poppler. This delivery certifies the available publication and saved-replay scope, not a newly executed full scientific pipeline.

## Reading and commands

Read the [repository README](../../../README.md), [study overview](../../../docs/MENTAL_PONG.md), [final report](../trajectory_project/closeout_v1/FINAL_REPORT.md), and [future plan](../../../docs/FUTURE_DIRECTIONS.md), in that order. From the repository root:

```shell
python scripts/verify_integration.py
python experiments/05_mental_pong/run.py --verify
python experiments/05_mental_pong/run.py --replay --animal mahler --representation FA50 --q 0 --split 0
python experiments/05_mental_pong/run.py --full-plan
```

Use the recorded Mental-Pong environment for replay. Verification also reports external availability when the local roots are not configured. These commands accept execution from another working directory when their script paths are supplied explicitly. Legacy scientific `--all` and `--tests` runners can write outputs; they were not used on the source directories during integration.
