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

## Replay one saved random-head case

```shell
python /path/to/neural-trajectory-evaluation/experiments/05_mental_pong/run.py --replay --animal mahler --representation FA50 --q 0 --split 0
```

This route uses saved coefficients, labels and the corresponding latent values. Missing required storage produces an unavailable result rather than a false pass. It does not fit a new model or rerun the 1000 randomizations.

## Full historical execution

Print the required dependencies and the explicit historical full-run route without executing it:

```shell
python /path/to/neural-trajectory-evaluation/experiments/05_mental_pong/run.py --full-plan
```

Consult `run.py --help`, the [dependency manifest](integration/dependency_manifest.csv) and [runtime lock](requirements.lock.txt). Full execution requires the external raw/prepared data and representation bundle. Documentation of this route is not a claim that integration repeated the experiments. Default checks remain read-only; legacy `--all` and `--tests` paths can write reports or test records and require separate output handling.

The [integration summary](integration/INTEGRATION_SUMMARY.md) and [verification record](integration/verification.json) list checks actually performed now. [Original acceptance](trajectory_project/closeout_v1/results/final_acceptance_audit.json) and source hashes remain separate. English edits change document hashes; immutable numerical files retain their scientific identity. Source user requests and original-language documents stay in external preservation storage.
