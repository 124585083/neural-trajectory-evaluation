# Runtime and artifact availability

The Mental-Pong module contains the completed `condition_endpoint_fa_gpfa_v1` and `closeout_v1` analyses. The earlier `trial_endpoint_fa_gpfa_v1` implementation remains a computational dependency. The [version history](../archive/HISTORY.md) distinguishes the final FA50/GPFA50 ordinary least squares analysis from earlier GPFA64 and segmented-proxy studies.

The public tree includes reports, source code, fixed split definitions, aggregate score tables, and figures. Condition-label arrays and their condition-by-time CSV equivalents are external pending clarification of redistribution rights. Raw data, model and readout weights, latent arrays, and full randomization arrays also require external storage. The [dependency manifest](dependency_manifest.csv), [artifact registry](artifact_registry.csv), and [rights record](revision_20261003/data_rights_audit.json) identify these boundaries. The [migration map](migration_map.csv) records historical source hashes; the [publication map](revision_20261003/publication_map.csv) identifies retained files, edited technical editions, and explicit exclusions.

Scientific status remains **CLOSED_EXPLORATORY_WITH_LIMITATIONS** and **raw_preprocessing=fail**. Released-input prefix checks do not repair upstream filling across conditions and times. Neural trial membership and strictly pre-feedback endpoint timing remain unresolved. The [study overview](../../../docs/MENTAL_PONG.md), [final report](../trajectory_project/closeout_v1/FINAL_REPORT.md), and [evidence index](../../../results/study_evidence_index.csv) explain the findings. The [future plan](../../../docs/FUTURE_DIRECTIONS.md) has not been executed.

## Supported checks

The verifier checks the declared public file set, syntax, relative links, stored hashes, fixed condition splits, and saved-score aggregation. Its character scan does not assess English grammar. [Historical verification](verification.json) records earlier execution results; each record applies to its stated source version and environment, rather than certifying all later changes.

Four original saved random-head cases were replayed with q=0 and split=0, one per animal and representation. The largest score discrepancy was approximately 1.64e-11; no representation or readout was fitted. See the [saved replay record](saved_replay_verification.json). The separate [mini replay](../mini_replay/README.md) uses a compact private bundle to check both position heads, cross-scoring, and fixed controls. Its numerical payload is not distributed. A source-only clone reports that absence explicitly.

## Commands

From the repository root:

```shell
python scripts/verify_integration.py --skip-external
python experiments/05_mental_pong/run.py --verify
python experiments/05_mental_pong/run.py --replay --animal mahler --representation FA50 --q 0 --split 0
python experiments/05_mental_pong/run.py --full-plan
```

The first command checks public files and saved aggregate scores without external data. The second also reports external availability. Saved-weight replay requires the registered original artifacts and the recorded runtime. Configure storage roots using [paths.example.json](../configs/paths.example.json); actual locations belong in ignored local configuration or environment variables.

`--full-plan` describes dependencies without running the historical scientific pipeline. That full pipeline has not been tested end to end after relocation and requires complete inputs plus a separate writable output location. Legacy analysis runners can fit models and write results. Historical atlas export requires its configured PDF runtime and Poppler. The external PDF atlases have not been regenerated or visually rechecked for this edition. The default mini replay was tested on Windows; Linux and macOS execution remains unverified.
