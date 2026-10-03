# Mental-Pong: task-coordinate reconstruction

This independent study compares objective ball paths with behavior-constrained candidate paths using condition-mean DMFC neural representations. Directory number 05 is organizational; no output from Sensorium stages 01–04 is required. Scientific completion is `CLOSED_EXPLORATORY_WITH_LIMITATIONS`, with `raw_preprocessing=fail`. The published-input fit and filtering checks have a narrower passing scope.

## Read the study

1. [Scientific overview](../../docs/MENTAL_PONG.md): question, data level, readouts, results and limits.
2. [Closeout summary](trajectory_project/closeout_v1/FINAL_SUMMARY.md) and [full report](trajectory_project/closeout_v1/FINAL_REPORT.md): own-target scores, cross-scoring, both controls and geometry baseline.
3. [Condition-mean methods/results](trajectory_project/condition_endpoint_fa_gpfa_v1/REPORT.md), [experiment ledger](trajectory_project/closeout_v1/experiment_ledger.md) and [historical versions](archive/HISTORY.md).
4. [Current future plan](../../docs/FUTURE_DIRECTIONS.md): a design only; no future experiment was performed during integration.

Compact numerical results and English reports are published with the module. Large artifacts and original immutable source records are described by the [artifact registry](integration/artifact_registry.csv), [dependency manifest](integration/dependency_manifest.csv) and [migration map](integration/migration_map.csv). Availability is recorded independently from source scientific completion.

## Check available artifacts

Using the recorded Mental-Pong environment, run from any working directory:

```shell
python /path/to/neural-trajectory-evaluation/experiments/05_mental_pong/run.py --verify
```

Invoking `run.py` without an action also performs read-only verification. Configure external storage with the ignored local path file based on [paths.example.json](configs/paths.example.json), the `--paths` option, or the documented `MENTAL_PONG_SOURCE_ROOT` environment variable. The source-root variable identifies the preserved pilot bundle, without putting machine-specific locations in tracked documents.

## Small replay and its current availability

The [mini replay](mini_replay/README.md) now contains a NumPy runner and an exporter for the fixed split 0/q=0 examples from both animals and FA50/GPFA50. It recomputes predictions, the objective/behavior 2×2 scores, random-candidate scores and geometry skill, and the fixed-head mismatch control. A genuine 5.7 MB private export passed numerical, cold-copy and tampering checks. The [dependency inventory](mini_replay/dependencies.csv) separates source code from scientific arrays.

The new public payload is **withheld pending derived-data redistribution permission**. The specific data record lists CC BY-NC-ND 4.0. Permission to redistribute these transformed neural and behavioral arrays has not been established; see [third-party notices](../../THIRD_PARTY_NOTICES.md). The command below therefore returns a clear unavailable status in a public-only checkout:

```shell
python experiments/05_mental_pong/run.py --mini-replay --output /tmp/mental-pong-replay
```

An authorized holder can follow the [private export and replay instructions](mini_replay/README.md#commands-and-availability). The mini runner does not consult `source_pilot`, raw recordings, GPU tools or local path configuration. Its optional `--ols-refit` performs a small check of the original OLS specification. It never fits a neural representation.

The code and public links can be verified now. A saved split has been replayed from a private copy. The full analysis has not been rebuilt during this revision. These are separate verification states.

## Existing local random-head replay

```shell
python /path/to/neural-trajectory-evaluation/experiments/05_mental_pong/run.py --replay --animal mahler --representation FA50 --q 0 --split 0
```

This older route requires the configured external pilot and uses saved coefficients, labels and the corresponding latent values. Missing required storage produces an unavailable result. It does not fit a new model or rerun the 1000 randomizations. Use the mini route when an authorized exported package is available.

## Full historical execution

Print the required dependencies and the explicit historical full-run route without executing it:

```shell
python /path/to/neural-trajectory-evaluation/experiments/05_mental_pong/run.py --full-plan
```

Consult `run.py --help`, the [dependency manifest](integration/dependency_manifest.csv) and [runtime lock](requirements.lock.txt). Full execution requires the external raw/prepared data and representation bundle. Documentation of this route is not a claim that integration repeated the experiments. Default checks remain read-only; legacy `--all` and `--tests` paths can write reports or test records and require separate output handling.

The [integration summary](integration/INTEGRATION_SUMMARY.md) and [integration verification record](integration/verification.json) preserve the earlier integration checks. The [current verification checks](integration/revision_20261003/verification.json) record the available source and numerical checks. [Original acceptance](trajectory_project/closeout_v1/results/final_acceptance_audit.json) and source hashes remain separate. The historical migration map is unchanged; the [publication map](integration/revision_20261003/publication_map.csv) links its hashes to current edited files. Immutable numerical files retain their scientific identity. Source user requests and original-language documents stay in external preservation storage.
