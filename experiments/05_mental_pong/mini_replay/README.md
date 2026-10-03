# One saved Mental-Pong split

This replay recomputes position predictions from saved neural features and ordinary least squares (OLS) coefficients. It checks split 0 and random-endpoint allocation q=0 for Mahler and Perle. The features come from factor analysis (FA50) and Gaussian-process factor analysis (GPFA50), each with 50 dimensions. These choices were fixed before inspecting replay scores. One split does not reproduce the 100-split results or the 1,000-allocation distributions.

**Public payload status: withheld pending permission to redistribute derived data.** The [dataset record](https://zenodo.org/records/13952210) specifies CC BY-NC-ND 4.0. Permission to redistribute the transformed neural and behavioral arrays has not been established. Public code and export instructions are available; the scientific arrays remain in private validation storage. The repository's MIT software license does not grant rights to the source data. See [third-party notices](../../../THIRD_PARTY_NOTICES.md).

The private export contains four cases and about 5.7 MB of data. The default NumPy replay passed for all four. A separate copy ran on Windows with Python 3.11.9 and NumPy 1.26.4, outside the source checkout and with network operations and access to source files denied by a Python audit hook. Linux and macOS were not tested. This verifies the code against actual private payloads. The public payload is unavailable, and a complete raw-data rebuild remains unverified.

## What is recomputed

The program checks hashes, array axes, row identities and the original 39/40 condition split. Condition 59920 remains invalid. Split 0 therefore has 39 contributing training conditions and 39 contributing test conditions; its nominal 40 test identities are retained.

For each case, predictions are computed as `X_test @ coefficients.T + intercept`. A 50-dimensional input and intercept give 51 coefficients per coordinate and 102 for an x/y head. Three heads reconstruct objective position, the actual mean-endpoint candidate, and one saved random-endpoint candidate. No scaler, regularization, feature search or additional covariate is introduced.

The replay computes each head's r and RMSE against its own target, the objective/behavior 2×2 table, random-candidate geometry-baseline skill, and one fixed-head condition-mismatch control. The mismatch and its matched reference use the same original timestamps and the intersection of both conditions' valid phase masks. Undefined correlations remain JSON `null`. Expected metrics come from original per-split result arrays and are used only after recomputing predictions and scores. The fixed comparison tolerance is `atol=1e-8, rtol=1e-10`.

The objective and behavior tables identify the coordinate and evaluated interval for each score. Correlation is dimensionless; RMSE uses the saved position-coordinate units. Random-candidate scores concern y. The geometry baseline uses the mean assigned endpoint of valid training conditions. These are single-split checks, so they do not estimate biological trial variability.

Saved time stamps, masks, train/test rows, endpoint assignments, geometry offset/alpha, coefficients and original source hashes are retained without rounding. NPZ arrays contain only numeric dtypes and load with `allow_pickle=False`. Raw trial logs, executable model pickles, machine paths and unrelated source records are omitted.

The released neural inputs include preprocessing across conditions and times. The recorded status remains `raw_preprocessing=fail`. Neural trial membership and exact pre-feedback endpoint timing remain unresolved. A successful numerical replay does not remove these limitations. Read the [study overview](../../../docs/MENTAL_PONG.md) for their scientific consequences.

## Commands and availability

From the repository root, install the small replay dependencies:

```shell
python -m pip install -r experiments/05_mental_pong/mini_replay/requirements.txt
```

The public command is implemented, but currently returns an explicit withheld-payload status and exits with code 1:

```shell
python experiments/05_mental_pong/run.py --mini-replay --output /tmp/mental-pong-replay
```

The Windows equivalent uses a directory chosen by the caller:

```powershell
python experiments/05_mental_pong/run.py --mini-replay --output "$env:TEMP/mental-pong-replay"
```

An authorized holder of the original saved artifacts can make a private export. The exporter verifies source bytes against the existing artifact registry. It requires a new output directory outside the original pilot; it never changes that source.

The condition-mean label NPZ files and their per-bin CSV copies are also excluded from the current public tree while redistribution remains unresolved. The exporter reads their registered private originals. A repository clone therefore supplies code and aggregate results, without the labels needed for a numerical export. Public editions of both condition-label coverage tables and the endpoint-condition audit also omit the `mean_endpoint` input column. Their remaining summary fields are unchanged; complete original tables remain external under their recorded source hashes.

```shell
python experiments/05_mental_pong/scripts/export_mini_replay.py --source-pilot /path/to/preserved/pilot --output /path/to/private-mini
python experiments/05_mental_pong/run.py --mini-replay --mini-bundle /path/to/private-mini --output /path/to/new-replay-output
```

`--mini-bundle` supplies the actual exported package directly. The replay never reads `paths.local.json`, `MENTAL_PONG_SOURCE_ROOT`, raw recordings or a GPFA installation. Only the exporter uses the pilot. Publishing a payload requires a separately documented, passed rights audit; an export alone does not establish that permission.

The explicit optional refit uses the recorded scikit-learn and SciPy versions:

```shell
python -m pip install -r experiments/05_mental_pong/mini_replay/requirements-refit.txt
python experiments/05_mental_pong/run.py --mini-replay --mini-bundle /path/to/private-mini --output /path/to/new-refit-output --ols-refit
```

This check fits the three original OLS targets on the exported training rows, then compares test predictions and metrics. It records design rank and solver versions. It does not require coefficient identity in a rank-deficient system. All 12 checked heads had rank 50; the largest prediction difference from saved coefficients was about `1.51e-14`. No FA/GPFA was refitted and no randomization distribution was rerun.

## Dependencies and tests

[Source and artifact dependencies](dependencies.csv) distinguish the replay, export and full historical execution routes. The retained modules implement the final endpoint constructor, representation fitting and prefix inference, split handling, OLS scoring and both null controls. The historical `capacity_position_baselines.py` is a source fingerprint in the saved configuration; the final pipeline never imports its functions. Checking that historical fingerprint still requires the corresponding original record.

```shell
python -m pytest -q -p no:cacheprovider experiments/05_mental_pong/mini_replay/tests
```

Without a payload, real-data tests skip with a specific unavailable reason. The initial private validation used `MENTAL_PONG_TEST_MINI_BUNDLE` to locate the real export and passed all 20 tests then present, including the optional OLS refit. The cold subprocess clears that variable and custom `PYTHONPATH`; it reads a new temporary copy containing only the entry point, NumPy replay code and actual payload. Tests reject changed features, masks, coefficients, hashes and contradictory axis or coordinate labels. Changed test features or coefficients still fail numerical validation after their checksum records are refreshed. Output must go to a new or empty directory, which prevents overwriting previous records.

The [2025 neural article](https://www.nature.com/articles/s41467-024-54688-y), [2024 dataset](https://zenodo.org/records/13952210) and [official code](https://github.com/jazlab/MentalPong/tree/b976255be73140c759d8f8db0fd8ff551a4e2d73) have distinct roles. This replay uses the completed local condition-mean endpoint analysis. Its protocol differences from the original decoding analysis are documented in the [study methods](../trajectory_project/condition_endpoint_fa_gpfa_v1/protocol_diff.md).
