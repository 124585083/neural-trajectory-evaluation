# Response Matching: Results

**The validation-selected output perturbation keeps mean per-neuron response correlation within the fixed test tolerance, while both primary trajectory differences remain positive.** The test response-r gap is 0.003286, below the 0.005 tolerance. Position correlation and velocity-direction cosine still favour perturbed Dynamic, with paired intervals above zero. Speed-profile differences remain unresolved.

The trajectory differences retain information beyond this single matched summary. The evidence comes from a fixed-output control; a comparison of independently trained models with statistically equivalent accuracy would address a different question.

## 1. Completed control and comparison scope

The completed `retrained_v1` control uses the same Static, reduced Dynamic and main four-dimensional Gaussian-process factor analysis (GPFA) as the [primary comparison](../03_model_comparison/RESULTS.md). Static predictions are unchanged. Dynamic predictions receive training-scaled, trial-keyed Gaussian noise and a fixed lower clipping bound. Validation selects the amplitude, and test applies the same fixed operation.

The control covers one pilot session, 58 test trials from six movies, 512 fixed neurons and 250 time bins at original frames 50–299. Training has 282 trials and validation has 66. Training predictions determine the per-neuron noise scales, and validation supplies the scalar selection criterion. The [setup](SETUP.md#4-output-perturbation-and-amplitude-selection) specifies the operation, selection rule and common GPFA conversion.

Historical matched-output experiments and the old epoch-65 checkpoint control are separate analyses. The current results use their own scores, repeat splits and calibration. The encoders, GPFA and matching choices remain fixed through test scoring.

![Signed validation and test response gaps within tolerance, with all five trajectory contrasts](../figures/documentation/05_response_matching.png)

*Signed response gaps remain inside the fixed ±0.005 tolerance in validation and test. The right panel plots all five stored perturbed-Dynamic-minus-Static trajectory contrasts; the speed interval crosses zero. These intervals resample six whole movies with noise and calibration fixed. The tolerance band defines the matching rule; statistical equivalence was not tested.*

## 2. The fixed response match transfers to test

The selected amplitude is `sigma=0.8697490026177834`. It is the validation candidate with the smallest absolute gap from Static among 101 stored candidates. Matching uses the fixed rule $|r_{\mathrm{perturbed\ Dynamic}}-r_{\mathrm{Static}}|\leq0.005$.

| Data role | Static response r | Perturbed Dynamic response r | Absolute gap | Within 0.005 tolerance |
|---|---:|---:|---:|---|
| Validation | 0.146299 | 0.144669 | 0.001630 | Yes |
| Test | 0.145129 | 0.148415 | 0.003286 | Yes |

All 512 neurons contribute to this summary. Within each neuron, trial and time samples are pooled before calculating response correlation; the neuron correlations are then averaged. The criterion permits differences between individual neuron scores.

The validation and test gaps have opposite signs but satisfy the same absolute tolerance. `test_matching_retained=true` and `test_retuned=false` record transfer without recalibration.

The [selection table](../evidence/04_model_comparison/retrained_v1/response_matching_selection.csv) retains every candidate and its stored `absolute_gap`. The [freeze record](../evidence/04_model_comparison/retrained_v1/frozen_analysis.json) and [evaluation summary](../evidence/04_model_comparison/retrained_v1/evaluation_summary.json) identify the unchanged amplitude and test check. The reported gaps retain the precision of those records.

The fixed tolerance establishes scalar closeness. The response gap is a point estimate, with statistical equivalence untested. The trajectory comparison asks what remains different under that criterion.

## 3. Primary trajectory differences remain

The table gives agreement with the same neural reference. **Perturbed Dynamic** is the fixed output control. **Unperturbed Dynamic** supplies the original main-score context for interpreting the calibrated control.

| Trajectory metric | Role | Static | Unperturbed Dynamic | Perturbed Dynamic |
|---|---|---:|---:|---:|
| Position correlation | Primary | 0.416545 | 0.781569 | 0.790734 |
| Velocity-direction cosine | Primary | 0.259418 | 0.516995 | 0.497020 |
| Normalised position RMSE | Secondary; lower is better | 0.922172 | 0.647897 | 0.654987 |
| Speed-profile correlation | Secondary | 0.534564 | 0.555709 | 0.551869 |
| Acceleration-direction cosine | Secondary | 0.133099 | 0.428879 | 0.414563 |

All values are dimensionless. Position refers to state in the GPFA latent coordinates. Trajectories use all repeats averaged within each movie and the same frozen offline posterior. The normalised position root mean squared error (RMSE) uses the pair-dependent denominator defined in [Setup](SETUP.md#6-metrics-and-paired-uncertainty).

The control's paired contrast is **perturbed Dynamic minus Static**:

| Trajectory metric | Role | Stored paired difference | 95% paired interval | Valid / undefined draws |
|---|---|---:|---|---:|
| Position correlation | Primary | +0.374188 | [+0.226531, +0.557650] | 2,000 / 0 |
| Velocity-direction cosine | Primary | +0.237603 | [+0.114948, +0.346273] | 2,000 / 0 |
| Normalised position RMSE | Secondary; lower is better | −0.267185 | [−0.465381, −0.166979] | 2,000 / 0 |
| Speed-profile correlation | Secondary | +0.017304 | [−0.047370, +0.082075] | 2,000 / 0 |
| Acceleration-direction cosine | Secondary | +0.281464 | [+0.104865, +0.449732] | 2,000 / 0 |

Both primary intervals favour perturbed Dynamic; error and acceleration also favour it. **Speed remains unresolved**, as its interval crosses zero.

The 2,000 seed-42 draws recompute each pooled trajectory score on the same resampled whole movies. They quantify variability across these movies conditional on the trained model and fixed noise; animal-, training- and noise-level variability require separate repetitions.

The Static and perturbed columns come from [control metrics](../evidence/04_model_comparison/retrained_v1/pilot/response_matching_metrics.csv); the unperturbed column comes from [main metrics](../evidence/04_model_comparison/retrained_v1/pilot/metrics.csv). The differences and intervals come directly from [control bootstrap fields](../evidence/04_model_comparison/retrained_v1/pilot/response_matching_bootstrap.csv). Display rounding and small original/replay numerical differences can prevent subtraction of the displayed columns from reproducing the final reported digit. The stored contrast fields are retained unchanged.

Position r is descriptively higher after perturbation; the other four points worsen. The apparent position increase is descriptive; assessing a noise benefit would require a perturbed-versus-unperturbed interval.

<a id="4-other-response-and-representation-scores-were-not-matched"></a>
## 4. Response properties beyond the matching target

Only mean per-neuron response r determined calibration. The other reported response and representation measures remain separate outcomes:

| Metric | Static | Unperturbed Dynamic | Perturbed Dynamic |
|---|---:|---:|---:|
| Mean per-neuron response r — matching target | 0.145129 | 0.185953 | 0.148415 |
| Response normalised mean squared error (NMSE; lower is better) | 0.978171 | 0.962456 | 0.971422 |
| Condition/time centred kernel alignment (CKA) | 0.456054 | 0.510811 | 0.517646 |
| Temporal-difference CKA | 0.028074 | 0.093155 | 0.075791 |
| Condition/time-state representational similarity analysis (RSA) | 0.424808 | 0.544941 | 0.538857 |

NMSE and time-aware CKA/RSA remain different. The control reports pooled point estimates for these measures; the main comparison’s intervals describe a different contrast.

These measures retain information outside the matching constraint. Their differences show that conventional measures remain unequal and that the perturbed output and Static differ in overall prediction accuracy. Trajectory metrics add a description of latent position and temporal-change agreement to the differences also captured by these conventional measures.

The full shared support comprises 1,500 condition/time states, 1,494 velocity comparisons, 1,488 acceleration comparisons and 512 response neurons. Time-state RSA uses 150 sampled states and 11,175 edges. The trajectory differences are evaluated on the full stated common support. Undefined values retain the `NA` designation.

## 5. Interpretation and limits

This control shows that **similar mean per-neuron response correlations can coexist with differences in both primary trajectory measures.** With the test response gap at 0.003286, position and local-direction agreement still differ, and their paired intervals exclude zero. Reporting the trajectory scores alongside that response summary makes these differences explicit.

The trajectories describe the same response arrays after repeat averaging, fixed scaling and GPFA inference. Their contribution is a complementary description of those response arrays. NMSE and the reported time-aware CKA/RSA values remain different, so explanations based on other response properties remain possible. The separate [time-reversal analysis](../04_temporal_metric_tests/RESULTS.md) tests the invariance of particular time-averaged summaries; this response-matching control uses its own scalar-closeness criterion.

Averaging and smoothing are also relevant to interpretation. The matching score uses individual trial/time samples. Trajectory scores use repeat means and an offline posterior. Added noise may have a stronger effect on individual-trial correlation than on repeat-averaged, smoothed trajectories. Averaging and smoothing therefore remain plausible contributors to the surviving trajectory contrast, with their specific contributions unresolved.

Finally, this is one validation-selected perturbation of one reduced Dynamic model, with one fixed trial-keyed noise setting and one pilot session. Further encoder fits, independent noise calibrations and recordings would establish how broadly this pattern generalises. The underlying test movies were explored historically, and the frozen GPFA remains an offline measurement. The [measurement results](../02_trajectory_measurement/RESULTS.md) delimit its repeat reliability and limited response-variance coverage; the present scores retain their original metric units alongside that reliability reference.

Within this scope, the supported reporting choice is to retain the scalar response score **alongside** trajectory and time-aware representation outcomes, naming what was matched. Position and direction provide additional diagnostic detail in this control; speed remains an unresolved result.

## 6. Result records and available presentation

| Record | Content used here |
|---|---|
| [Validation selection](../evidence/04_model_comparison/retrained_v1/response_matching_selection.csv) | `selected=True`, selected amplitude, validation scores and stored `absolute_gap`; all 101 candidate rows retained |
| [Frozen analysis](../evidence/04_model_comparison/retrained_v1/frozen_analysis.json) | `response_matching.sigma`, `noise_seed`, `tolerance`, control checksum and fixed model identities |
| [Evaluation summary](../evidence/04_model_comparison/retrained_v1/evaluation_summary.json) | `pilot.response_matching`, both gaps, `test_matching_retained=true` and `test_retuned=false` |
| [Control metrics](../evidence/04_model_comparison/retrained_v1/pilot/response_matching_metrics.csv) | `static` and `perturbed_dynamic`, all ten metrics on the test/pilot/seed-42 support |
| [Unperturbed main metrics](../evidence/04_model_comparison/retrained_v1/pilot/metrics.csv) | Contextual `dynamic` values on the same support; not a new control run |
| [Control paired intervals](../evidence/04_model_comparison/retrained_v1/pilot/response_matching_bootstrap.csv) | Five `perturbed_dynamic_minus_static` trajectory contrasts, interval bounds and valid/undefined draw counts |

[Setup](SETUP.md#7-source-records-and-interpretation-scope) links the formulas, configuration and source functions. The inline figure renders the saved validation/test gaps and all five control trajectory intervals; contextual unperturbed values remain in the tables.

Control-specific conventional intervals, a statistical-equivalence design and across-noise-seed variability remain possible extensions. Each would require evidence specific to this control.
