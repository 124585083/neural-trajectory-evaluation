# Mental-Pong exploratory closeout

[Summary](FINAL_SUMMARY.md) · [Full report](FINAL_REPORT.md) · [Experiment ledger](experiment_ledger.md) · [Sources](SOURCES.md)

This version completed own-target/2x2 replay (A), 1000 fixed-readout condition mismatches (B), 1000 random-endpoint OLS refits (C), and descriptive averaging checks. Every random repeat covers the original 100 nominal 39/40 condition splits. Existing half1 FA50/GPFA50 representations were reused.

`raw_preprocessing=fail` remains visible throughout the record. Released-input filtering and fit provenance do not repair upstream cross-condition/time filling. The [historical future design](future_design.md) was not executed and now links to the single current plan.

## Access and execution

Use the [publication entry point](../../README.md) for reading, default read-only verification and one saved random-head replay. Full historical execution requires the external dependency bundle and explicit invocation. Integration did not rerun the randomization studies. The historical `--tests` option writes a test record; `--all` can regenerate reports and seal a new run. They are not default verification commands.

## Results and array axes

- [Configurations](configs/closeout_protocol.json) preserve the protocol, seeds and splits. B seeds are 314159/314160 and C seeds 271828/271829 for Mahler/Perle, using PCG64.
- [A scores](results/A/self_reconstruction_main_table.csv) and [cross-scores](results/A/cross_2x2_summary.csv) are recomputed from saved held-out predictions, scored per split before aggregation.
- [B summary](results/B/B_summary.csv) averages the 100 splits within each random repeat. Full arrays use `[q,split,epoch,head,target,coordinate,metric]`; mappings use `[q,split,condition]`, with -1 outside valid test identities. Matched and mismatched scores share support.
- [C methods](results/C/README.md) describe `[q,split,epoch,metric]` scores, fixed endpoint allocations and actual new OLS coefficients. Read axis names from each array. Candidate labels can be reconstructed as `offset + alpha * endpoint`; invalid entries remain NaN.
- [Descriptive results](results/descriptive/mean_cancellation_interpretation.md) cover exact behavioral members, sign cancellation, variance decomposition, path separation and fixed post hoc examples.
- [Scientific acceptance](results/final_acceptance_audit.json), [test record](results/test_execution.json) and [manifest](manifest.json) document the original execution. [Integration verification](../../integration/verification.json) documents publication changes separately.

Large mappings, coefficients, latent arrays, predictions, behavioral member records and complete all-condition atlases are registered in the [artifact inventory](../../integration/artifact_registry.csv). Unavailable local storage must be reported as unavailable, not treated as a completed replay.

Each xy head has 102 coefficients. C's 400 multioutput solves represent 400000 independent random-label heads, with one equivalent x solution for identical x labels. These are model fits rather than independent animal experiments. Final behavioral labels are constructed after observing the endpoint; that does not make endpoint behavior an input feature to the neural readout.

English figure copies require their own presentation checks. Original scientific visual-inspection records remain source evidence and are not automatically transferred to a changed figure.
