# Mental-Pong experiment history

The final analysis is `condition_endpoint_fa_gpfa_v1` followed by `closeout_v1`. This archive summarizes earlier executed work in English. Original reports, configurations, weights and manifests remain immutable in external preservation storage. The [migration map](../integration/migration_map.csv) identifies source documents and hashes; the [artifact registry](../integration/artifact_registry.csv) gives availability. Historical scores keep their original data level and protocol.

## Original reproduction and representation selection

The pilot reproduced the published FA50 coordinate-decoding outputs. The reference uses all-condition reliability selection/FA and 100 condition-group OLS splits. Its x/y correlations are Perle 0.9598/0.7872 and Mahler 0.9413/0.7002. Later tool evaluation used a separate 47-training/16-validation/16-test protocol, training-half1 preprocessing and validation-selected Ridge. These share a linear decoding framework but cannot isolate an algorithm effect across identical protocols.

The first trajectory project chose causal GPFA64 with a shared learnable RBF time scale for a specific fixed-tool analysis. The one-million-parameter constraint was an upper limit rather than parameter matching. GPFA64 had 135829 Perle and 34189 Mahler representation parameters. Cross-half response correlations were approximately 0.233/0.277 and position-y correlations 0.778/0.522. Increasing from 32 to 64 dimensions retained more y-related information. Seeds 42, 137 and 271 summarized scalar consistency; latent coordinates were not averaged across seeds or animals. This historical choice does not replace the final per-split FA50/GPFA50 comparison.

## Frozen GPFA64 hidden-direction study

`trajectory_project` steps 1–3 asked whether a fixed visible-trained direction readout retained the last visible direction (M) or changed after a hidden collision (U). The 47/16/16 split, visible-only Ridge head and baseline seed 42 were fixed; previously viewed test data made this exploratory rather than preregistered. Physical metadata set three valid pre-event and three post-event bins. A positive pre-window and sufficiently positive/negative post-window had to agree across halves.

Perle's visible direction calibration failed its existing threshold. Mahler had inadequate pre-event or second-half amplitude. Neither animal's two test collision events supported a clear two-half M/U classification; Mahler 463174 showed an M-like pattern only in half1. Seeds 137/271 retained this interpretation. A falling positive score was not called a reversal, and low amplitude was not called information loss.

Hidden readout transfer differed by animal. Perle y correlations were about 0.355/0.308 for half1/half2; Mahler had 0.055/0.062. Both animals retained repeated condition-related latent structure, with condition-null comparisons restricted to shared support. Local event geometry could disagree with full-hidden reliability and task-coordinate readouts. Two physical events, 50 ms means and an upstream filling boundary do not support precise single-trial latency or a unique computational mechanism. This negative finding motivated closer attention to observable behavioral references.

## Early average-behavior proxy

`condition_mean_behavior_preview_v1` constructed one geometrically selected training example: Mahler condition 251629. A fixed average-paddle stability rule yielded two continuous target updates at 1.950 and 2.400 seconds and 18 behavior-active bins. This was a construction preview without a neural readout fit. A stable mean did not establish a single trial's stopped paddle or internal belief. Its historical `AWAITING_USER_REVIEW` status described that preview; the later stage proceeded after authorization.

`condition_mean_behavior_stageB_v1` used frozen GPFA64, Ridge and the original 47/16/16 split on screened short behavior-active windows. The half1 test support was only three Mahler conditions/34 bins and two Perle conditions/27 bins. Mahler delta MSE was 0.5163 versus a zero-delta 0.7278, but it did not beat the saved training-mean baseline. Perle delta MSE was 1.208 versus zero-delta 0.633; its higher delta correlation did not resolve its amplitude error. A same-candidate-target head comparison differs from own-target position reconstruction. These small-set results are retained as historical exploratory evidence.

## Old all-79 segmented-candidate OLS

`all79_official_readout_v1` restored 100 nominal 39/40 condition splits and one full-time OLS mapping using the earlier frozen GPFA64. Candidate paths were physical references until a qualifying average-behavior update, then followed continuous piecewise connections. Behavior-active bins accounted for only about 4.5% of Mahler and 5.6% of Perle support.

Full-epoch half1 objective/candidate correlations were Mahler 0.4435/0.4463 and Perle 0.5598/0.5673; RMSEs were 5.0563/5.0389 and 5.1509/5.0465. Those small aggregate improvements apply to that candidate definition. The older representation's fitting conditions overlapped some randomized readout-test conditions, so this was readout condition holdout on a frozen representation rather than complete-pipeline generalization to unseen conditions. Fixed-case improvements from this version do not become evidence for the final endpoint candidates.

## Trial-endpoint labels

`trial_endpoint_fa_gpfa_v1` used a simpler endpoint connection with no stable-paddle detector: no-collision starting point to individual endpoint, or objective path until collision followed by the original collision anchor to endpoint. Each split fitted training-side FA50 and shared-time-scale GPFA50. OLS used 50 inputs and an intercept.

Released data lacked the response/session/unit/trial membership required for exact pairing. Route B therefore attached different behavioral-record labels to repeated copies of the same condition-mean neural input. It did not create independent neural trials or additional neural rank. Behavioral failures were retained when the endpoint and geometry were valid. Condition 59920 remained unresolved. There were 7407 usable Mahler and 84873 usable Perle candidate records across 78 conditions.

The [preserved full-precision comparison](../trajectory_project/condition_endpoint_fa_gpfa_v1/results/previous_trial_version_comparison.csv) reports this version alongside the final means. Repeated trial rows gave conditions with more records extra fitting and scoring contribution. Averaging labels later changed both target variation and that weighting. The comparison is not a one-factor averaging experiment.

## Final means and closeout

`condition_endpoint_fa_gpfa_v1` reused the same per-split FA50/GPFA50, moved to one condition-by-time row and used the fixed-membership mean endpoint. It contributed 78 valid conditions and 3369 rows per animal. All four full-epoch own-target results favored the objective path, with phase-specific candidate RMSE improvements in no-collision samples.

`closeout_v1` completed A own/cross-scoring, B1000 fixed-head condition mismatches and C1000 random-endpoint OLS fits, each random repeat covering 100 original splits. Correct condition correspondence and actual endpoint structure were readable under these controls. Full-epoch candidate superiority was not supported. The mean-endpoint geometry baseline exceeded full-epoch neural reconstruction, with small positive Perle phase-specific skill exceptions. Behavioral mean cancellation was demonstrated; corresponding neural information loss was not measured.

The final status is `CLOSED_EXPLORATORY_WITH_LIMITATIONS`, with `raw_preprocessing=fail`. The [final report](../trajectory_project/closeout_v1/FINAL_REPORT.md) is the scientific conclusion. The [current future plan](../../../docs/FUTURE_DIRECTIONS.md) is documentation only; it does not restart model selection, proxy construction or grouped neural experiments.
