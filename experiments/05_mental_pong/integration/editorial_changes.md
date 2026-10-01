# Editorial and presentation changes

The publication tree now presents two separate studies. The [repository overview](../../../README.md) introduces their shared questions before the result details. The [Mental-Pong account](../../../docs/MENTAL_PONG.md) follows the completed condition-mean endpoint analysis. Earlier GPFA64, stop-proxy, all79, and trial-label versions retain their identities in the [history](../archive/HISTORY.md).

## Existing Sensorium English

The review covered the existing methods, validation, results, model reports, experiment READMEs, figure source, and regenerated figures. Representative changes are:

| Earlier wording | Revised meaning | Location |
|---|---|---|
| “Leakage-free training and inference design” | Separate train/oracle fitting controls from offline full-window inference, which still uses future observations. | [Reliability README](../../02_gpfa_reliability/README.md), [methods](../../../docs/METHODS.md) |
| Existing metrics cannot explain trajectory scores | The tested finite-sample ridge regressions leave residual variation; this does not establish information independence. | [Results](../../../docs/RESULTS.md), [Q1–Q6](../../../docs/results/Q1_Q6_ANSWERS.md) |
| Noise scale range | Noise standard deviation, matching the actual perturbation implementation. | [Experiment matrix](../../../docs/supplementary/implementation/EXPERIMENT_MATRIX.md) |
| “No evaluation-driven fitting” in the workflow figure | Frozen encoding checkpoints, with training-only GPFA fitting described separately. Oracle checkpoint selection remains documented. | [Figure source](../../../scripts/make_core_figures.js) |

The comparison-figure caption now distinguishes observed response/RSA/CKA means from saved bootstrap trajectory means. Point values and intervals are unchanged. The temporal-reversal result remains scoped to condition-pattern RSA/CKA. Inconclusive speed differences and the general-damage alternative remain explicit.

## Mental-Pong English editions

Current and closeout reports, the experiment ledger, descriptive interpretations, historical summaries, and Python report templates are English editions. Comments, docstrings, messages, and plotting labels received the same review. The publication consistently distinguishes objective ball paths, behavior-constrained candidate paths, reconstructed positions, and neural trajectories.

The reports retain the negative full-epoch candidate comparison, low post-collision support in B, C's phase exceptions, negative full-epoch skill against the mean-endpoint geometry baseline, and small positive Perle phase-specific skills. Behavioral cancellation is described as an observation in endpoint means. Unknown neural trial membership prevents measuring the corresponding neural information loss.

Five closeout figures were rendered from saved CSVs to update labels, including “mean-endpoint geometry baseline.” Three Sensorium figures were also rendered with revised English. The other five published Mental-Pong PNGs retain their original English pixels. All 13 published PNGs received a new visual inspection; external historical PDF atlases did not inherit that pass.

## Preservation and code review

Original-language files remain in external source storage. The pre-edit target snapshot includes the existing uncommitted README. The [migration map](migration_map.csv) records original and publication hashes; [numerical verification](numerical_preservation.json) records unchanged serialized values and metadata comparisons. Source hash records inside migrated manifests describe their original artifacts, not the new English reports.

Python parser comparisons distinguish text changes from import and path adapters. The latter have focused functional tests. The code retains numerical fitting and scoring definitions; an imported source module's changed text hash does not certify a new scientific run. Portable verification and saved replay use separate entry points and records.

The [writing guide](../../../docs/WRITING_GUIDE.md) and [language audit](language_audit.json) describe the publication rules and checked scope. Required third-party licenses are preserved verbatim.
