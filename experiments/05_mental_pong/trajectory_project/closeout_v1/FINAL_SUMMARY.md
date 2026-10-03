# Mental-Pong closeout summary

**Scientific status: `CLOSED_EXPLORATORY_WITH_LIMITATIONS`.** A, B1000 and C1000 are complete; each random repeat covers the original 100 condition splits. Integration and translation are recorded separately.

| animal | representation | r_obj | r_beh | RMSE_obj | RMSE_beh | Delta_r | Delta_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mahler | FA50 | 0.4928 | 0.4696 | 4.5909 | 4.6129 | -0.0232 | -0.0220 |
| mahler | GPFA50 | 0.5374 | 0.5126 | 4.4458 | 4.4781 | -0.0248 | -0.0322 |
| perle | FA50 | 0.7019 | 0.6910 | 3.8586 | 3.8829 | -0.0108 | -0.0242 |
| perle | GPFA50 | 0.7204 | 0.7094 | 3.8005 | 3.8265 | -0.0110 | -0.0261 |

Each readout is scored against the target it was fitted to reconstruct. The representations contain readable task-position structure. Correct condition correspondence outperforms fixed-readout mismatches. Actual candidates also outperform the random-endpoint reference over full, hidden and endpoint-influence ranges.

The full-epoch comparisons do not support a behavior-constrained candidate-path advantage: all four objective own-target results are better. No-collision candidates have lower RMSE and lower correlation. Post-collision RMSE advantages over random endpoints are inconsistent.

All four full-epoch neural reconstructions fall below the mean-endpoint geometry baseline. Some Perle hidden/no-collision configurations have small positive skill. Exceeding random endpoints therefore does not establish a general advantage over shared geometry.

Behavioral averaging cancels opposite endpoint errors in 77/78 Mahler and 78/78 Perle conditions; median cancellation fractions are 54.0%/48.0%. The earlier trial-label comparison also changes condition weights. Neural information lost by averaging has not been measured.

**`raw_preprocessing=fail`.** Upstream filling across conditions and times remains unresolved. Neural mean membership and the final paddle scalar's strictly pre-feedback timing are unknown. Within-condition neural differences between opposite behaviors cannot be tested with these means. The [future design](future_design.md) remains a document.

[Full report](FINAL_REPORT.md) · [Scientific acceptance](results/final_acceptance_audit.json) · [Read-only entry point](../../README.md) · [Current cross-study plan](../../../../docs/FUTURE_DIRECTIONS.md)
