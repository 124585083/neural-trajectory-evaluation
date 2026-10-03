# Current post hoc cases

Conditions 55062 and 241919 were selected after earlier aggregate results had been examined. They are illustrations, without independent confirmatory status. Scores below average the original per-split held-out results. The table reports y for the indicated interval. Correlation is dimensionless; RMSE uses centered MWorks display-coordinate units. Each condition's scores average only the splits where it was held out. Delta_r = r_beh - r_obj and Delta_RMSE = RMSE_obj - RMSE_beh; positive differences favor the candidate. Four-curve figures instead show the mean and population SD of those test predictions for visualization.

| animal | representation | condition_id | epoch | r_obj | r_beh | RMSE_obj | RMSE_beh | Delta_r | Delta_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahler | FA50 | 55062 | full | 0.6040 | 0.5243 | 4.7901 | 4.5305 | -0.0797 | 0.2597 |
| mahler | FA50 | 241919 | full | 0.3175 | 0.3818 | 5.0632 | 5.3524 | 0.0644 | -0.2892 |
| mahler | FA50 | 55062 | hidden | 0.7730 | 0.7610 | 3.4881 | 2.6758 | -0.0120 | 0.8122 |
| mahler | FA50 | 241919 | hidden | 0.0723 | 0.0275 | 4.5166 | 5.1045 | -0.0448 | -0.5879 |
| mahler | GPFA50 | 55062 | full | 0.6349 | 0.5462 | 4.5619 | 4.3176 | -0.0887 | 0.2443 |
| mahler | GPFA50 | 241919 | full | 0.4238 | 0.4970 | 4.5394 | 4.9125 | 0.0732 | -0.3731 |
| mahler | GPFA50 | 55062 | hidden | 0.8868 | 0.8722 | 2.9272 | 2.1666 | -0.0147 | 0.7606 |
| mahler | GPFA50 | 241919 | hidden | 0.2486 | 0.1720 | 3.3694 | 4.0386 | -0.0767 | -0.6692 |
| perle | FA50 | 55062 | full | 0.4930 | 0.5308 | 5.2396 | 4.9722 | 0.0378 | 0.2674 |
| perle | FA50 | 241919 | full | 0.5873 | 0.5869 | 4.5094 | 4.5977 | -0.0004 | -0.0882 |
| perle | FA50 | 55062 | hidden | 0.6268 | 0.6345 | 5.2455 | 4.2320 | 0.0077 | 1.0135 |
| perle | FA50 | 241919 | hidden | 0.7408 | 0.7363 | 4.7853 | 4.7865 | -0.0045 | -0.0012 |
| perle | GPFA50 | 55062 | full | 0.5816 | 0.6117 | 5.1289 | 4.8586 | 0.0301 | 0.2703 |
| perle | GPFA50 | 241919 | full | 0.6729 | 0.6693 | 4.4051 | 4.5034 | -0.0036 | -0.0983 |
| perle | GPFA50 | 55062 | hidden | 0.8455 | 0.8543 | 5.1201 | 4.0633 | 0.0088 | 1.0568 |
| perle | GPFA50 | 241919 | hidden | 0.8083 | 0.7942 | 4.7154 | 4.7384 | -0.0142 | -0.0230 |

For condition 55062, candidate own-target RMSE improves across both animals and representations. Correlation over the full evaluated interval decreases in Mahler and increases in Perle. For condition 241919, candidate RMSE worsens in all four groups; full-interval correlation increases in Mahler and decreases slightly in Perle. One error metric and the old stop-proxy result cannot summarize these cases.

The two readout curves are close relative to the reconstruction-error scale. Their separation is smaller than the label separation for both Mahler cases and Perle 55062. Perle 241919 has nearly coincident labels and a larger readout-mean separation. The [curve-separation table](fixed_posthoc_case_curve_separation.csv) describes that distinction without replacing per-split reconstruction scores. [Complete cross-scores](fixed_posthoc_case_2x2.csv) retain both references.

`raw_preprocessing=fail` remains. Neural membership, terminal behavior timing and collision-estimation limits are unchanged.
