# Behavioral averaging and path separation

The audit uses exactly the behavioral members in the frozen membership JSON and verifies their source-record SHA256 hashes. Endpoint errors are relative to the current geometry's `objective_end_y`, rather than the trial metadata field `yf_mworks`. Signs use exact numerical zero. No near-correct threshold is estimated from these results. All variances use `ddof=0`; empty sign subgroups are NA. Error moments give equal weight to the 78 valid conditions. Endpoint errors, absolute errors and y-path RMS use centered MWorks display-coordinate units; second moments and variances use squared units.

| animal | n_up | n_down | both_sign_conditions | mean_abs | abs_mean | mean_square | squared_mean | within_variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahler | 3536 | 3871 | 77 | 1.7272 | 0.9784 | 5.7743 | 1.9690 | 3.8053 |
| perle | 50765 | 34108 | 78 | 1.3388 | 0.7908 | 3.5272 | 1.4164 | 2.1108 |

Mahler has 7407 contributing records and Perle 84873 across 78 valid conditions each. The median within-condition cancellation fractions are 54.0% and 48.0%. Their condition-mean endpoint error quartiles are -0.7527 to 0.4824 and -0.2209 to 0.8077; medians are -0.1547 and 0.1991. Complete ranges, sign-specific means, variances and SDs remain in the [condition audit](endpoint_condition_audit.csv). Zero-error record counts are zero for both animals.

The identity `mean(e_ci^2) = mean(e_ci)^2 + variance(e_ci)` has maximum residual 8.88e-15. Recomputed mean endpoints differ from existing labels by at most 1.78e-15. Opposite behavioral errors therefore do partly cancel in the means.

| animal | representation | epoch | path_RMS_pooled_bins | RMSE_obj_mean_of_100_splits | RMSE_beh_mean_of_100_splits |
| --- | --- | --- | --- | --- | --- |
| mahler | FA50 | full | 0.6760 | 4.5909 | 4.6129 |
| mahler | GPFA50 | full | 0.6760 | 4.4458 | 4.4781 |
| mahler | FA50 | hidden | 1.0006 | 4.3630 | 4.3735 |
| mahler | GPFA50 | hidden | 1.0006 | 4.1563 | 4.1806 |
| perle | FA50 | full | 0.5612 | 3.8586 | 3.8829 |
| perle | GPFA50 | full | 0.5612 | 3.8005 | 3.8265 |
| perle | FA50 | hidden | 0.8472 | 3.7038 | 3.7421 |
| perle | GPFA50 | hidden | 0.8472 | 3.5718 | 3.6130 |

Path separation is a descriptive RMS across all valid label rows. Reconstruction RMSE first scores each held-out split and then averages 100 scores. This is a magnitude comparison with different aggregation, not a power calculation. The [condition/phase comparison](path_separation_vs_error.csv) preserves detailed support. Primary split scores are not replaced by scores of averaged prediction curves.

The behavioral result does not establish identical neural and behavioral trial membership or quantify neural information removed by averaging. The trial-label comparison also changes condition weighting. A lack of overall candidate advantage cannot be attributed entirely to averaging, and weak separation of condition means does not exclude trial-level behavioral representation. `raw_preprocessing=fail`, unverified strictly pre-feedback endpoint timing and estimated collision anchors remain limits.

Cases 55062 and 241919 are post hoc illustrations. Their [scores](fixed_posthoc_case_scores.csv), [cross-scores](fixed_posthoc_case_2x2.csv) and [interpretation](fixed_posthoc_case_interpretation.md) use current targets. Complete atlases and large member-level records are in the [artifact registry](../../../../integration/artifact_registry.csv). Old segmented-candidate improvements are not transferred into this version.
