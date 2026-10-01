# Source versions and evidence boundaries

The completed source of record is `condition_endpoint_fa_gpfa_v1` followed by its sibling `closeout_v1`. Original reports, audit records, models and serialized outputs remain preserved outside this publication tree. English editions have new hashes, documented in the [migration map](../../integration/migration_map.csv).

The filenames shown in downloaded copies as `REPORT(2).md` and `previous_trial_version_comparison(1).csv` correspond to the actual source files [condition-mean report](../condition_endpoint_fa_gpfa_v1/REPORT.md) and [version comparison](../condition_endpoint_fa_gpfa_v1/results/previous_trial_version_comparison.csv). The suffixes are not separate scientific versions.

Key evidence includes the [fixed condition splits](../condition_endpoint_fa_gpfa_v1/configs/condition_splits_100.json), [representation reuse audit](../condition_endpoint_fa_gpfa_v1/results/reused_representation_audit.json), [closeout protocol](configs/closeout_protocol.json), [A replay](results/A/self_reconstruction_main_table.csv), [B completion](results/B/B_validation.json) and [C completion](results/C/completed.json). Exact mean memberships, source hashes and large artifacts are addressed through the [registry](../../integration/artifact_registry.csv) and [dependency manifest](../../integration/dependency_manifest.csv).

The all-79 Mahler and Perle condition atlases retain their original condition order and test-only averaging. Their registered artifact names are `mahler_all79_condition_atlas.pdf` and `perle_all79_condition_atlas.pdf`. They were not selected or re-averaged by null scores. Source user requests and immutable historical manifests stay in external preservation storage; publication metadata identifies them without rewriting their contents.

Earlier GPFA64/Ridge/segmented-proxy results enter the [history](../../archive/HISTORY.md), where their version identities and protocols remain explicit. They do not substitute for current FA50/GPFA50/OLS/condition-mean endpoint results.

**`raw_preprocessing=fail`.** Released-input `provided_input_filtering` and `fit_provenance` checks pass only in their recorded scope. None of the closeout randomizations removes dependence introduced by upstream filling across conditions and times.
