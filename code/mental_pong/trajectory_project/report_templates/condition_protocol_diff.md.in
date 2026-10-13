# Protocol changes from the trial-label version

| Component | `trial_endpoint_fa_gpfa_v1` | `condition_endpoint_fa_gpfa_v1` |
| --- | --- | --- |
| Behavioral target | Each record's own endpoint-constrained path | Mean path over the same fixed valid members; equivalent to constructing from their mean endpoint |
| OLS rows | Trial-by-time rows repeat the condition-mean latent input | One row per condition and time |
| Condition contribution | Valid trial count multiplied by valid time rows | Valid time rows; no extra equal-condition weighting |
| Primary scoring | Original held-out trial-by-time labels | Held-out condition-by-time mean labels |
| Representation | Train-39 FA50/GPFA50 for each split, half1 | Identical stored weights and latent values; no representation fitting |
| Splits and support | 100 nominal 39/40 splits, seed 0, audited geometry and masks | Unchanged |
| Readout capacity | Intercept OLS, 50 inputs, 102 coefficients per xy head | Unchanged |
| Atlas | One fixed actual behavioral record per condition | Condition-mean targets and out-of-fold reconstructions |
| Causal audit | Released-input prefix checks pass; raw preprocessing fails | Unchanged |

The published decoding framework expands condition-by-time samples and uses instantaneous unregularized OLS. This version returns to that row unit. Its half1 input, per-split training-side representations, completed-bin timing and behavior-constrained labels remain adaptations. It is not an exact recreation of every paper setting. The [historical protocol account](../../archive/HISTORY.md) distinguishes the original FA50 reproduction from later analyses.

Target variation and condition contribution change together. Their combined effect cannot be assigned solely to behavioral averaging. All numerical comparisons come from saved predictions; no model, condition, time window or seed was selected again.
