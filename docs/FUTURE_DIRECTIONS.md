# Future research: preserving behavior contrasts and testing temporal organization

This is the current joint research plan for Dynamic Sensorium and Mental-Pong. It describes future work; none of these experiments was run during repository integration. The completed [Sensorium results](RESULTS.md) and [Mental-Pong study](MENTAL_PONG.md) establish different starting points. Their measurements, noise ceilings, and GPFA implementations remain separate.

**First question:** With the physical condition fixed, do above-target and below-target trial groups show reliable neural readout shifts in the corresponding directions? Answering it requires neural and behavioral records with matching trial identities. The current condition means cannot be separated into those neural groups.

The plan proceeds from recovering identities and timing, to defining reliable behavior groups, testing their position readouts, examining temporal organization, and finally designing observations that separate competing computations.

## 1. What the completed averaging audit shows

The final Mental-Pong candidate uses a condition-mean paddle endpoint. Positive and negative endpoint errors partly cancel before a candidate path is constructed. The following quantities give each valid physical condition equal weight, rather than weighting conditions by their number of behavioral records.

| Quantity | Mahler | Perle |
|---|---:|---:|
| Mean within-condition absolute trial endpoint error | 1.7272 | 1.3388 |
| Mean absolute condition-mean endpoint error | 0.9784 | 0.7908 |
| Conditions with both numerical error signs | 77/78 | 78/78 |
| Median within-condition cancellation fraction | 54.0% | 48.0% |

These values come from the completed [descriptive audit](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/descriptive/mean_cancellation_summary.json), with full condition results in the [endpoint table](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/descriptive/endpoint_condition_audit.csv). “Positive” and “negative” refer to numerical endpoint differences. They are not validated categories of psychological judgment error.

For a trial endpoint \(b_{ci}\) in condition \(c\), define \(e_{ci}=b_{ci}-y_{\mathrm{end},c}\). After the common physical anchor, the linear candidate construction implies

\[
y_{\mathrm{behavior},ci}(t)-F(c,y_{\mathrm{end},c})(t)
=\alpha_c(t)e_{ci}.
\]

Here \(F\) is the fixed geometric interpolation and \(\alpha_c(t)\) is its endpoint weight. In the ideal geometry, \(F(c,y_{\mathrm{end},c})\) is the objective ball path, giving \(y_{\mathrm{behavior},ci}(t)-y_{\mathrm{objective},c}(t)=\alpha_c(t)e_{ci}\). The saved objective trajectory differs slightly from this interpolation because its anchor is estimated from published samples. The [geometry audit](../experiments/05_mental_pong/trajectory_project/closeout_v1/results/C/geometry_feasibility.json) records that numerical discrepancy. The original objective values were retained.

Averaging opposite endpoint errors reduces the candidate-to-objective difference. Full-path RMS separation is 0.6760 for Mahler and 0.5612 for Perle. These magnitudes are smaller than the reported reconstruction RMSE, but the aggregation differs: separation pools valid bins, whereas the readout result averages scores over held-out splits. This is a descriptive scale comparison, not a power calculation.

The observed cancellation concerns behavior. The trial membership of each released neural mean is unknown, so the corresponding neural group differences cannot be measured or recovered from the combined mean. The behavioral cancellation percentages must not be interpreted as percentages of neural information lost. The [earlier trial-label version](../experiments/05_mental_pong/archive/README.md) also changed condition weighting; its comparison with the final version is not a one-factor averaging experiment.

## 2. Recover observations that identify the contrast

The next dataset must link animal, session, physical condition, stable trial identity, neural response, endpoint behavior, task timing, feedback boundaries, unit availability, and membership in every released average. A physical condition fixes the stimulus and objective path. Its repetitions can still differ in behavior.

The relevant quantities are

\[
E[R(t)\mid\mathrm{condition}]
\quad\text{and}\quad
E[R(t)\mid\mathrm{condition},\mathrm{behavioral\ group}].
\]

Repeat averaging remains useful for estimating reproducible structure. Trials should be averaged within groups that preserve the behavioral contrast being tested. Each behavior group must retain enough observations to estimate its mean response and split-half reliability. This connects the repeat-averaged measurement rationale in [Study 1](DESIGN_RATIONALE.md) to a task where choices within one condition matter.

## 3. Compare behavior groups within a physical condition

Define signed endpoint error and group thresholds before inspecting neural differences, then check the reliability of the resulting group averages. Above-target, below-target, and near-correct groups should use measurement precision and task tolerance. Keep endpoints reasonably similar within each group. Match absolute error, session, trial count, and available units where feasible, and report remaining imbalance.

Use the neural records that actually belong to each behavior group. Fit a common representation on training neural data and apply a shared readout protocol. Separate model-fitting trials from evaluation trials and split-half reliability trials. Keep objective paths fixed within a condition, then ask whether reconstructed positions shift in the direction of the group's behavior. Retain own-target comparisons, cross-scoring, and within-condition controls for correspondence.

These are group-average representation tests. Blind single-trial prediction requires paired neural and behavioral observations and its own held-out evaluation. Cross-session pseudopopulations do not establish simultaneously recorded trial-level population dynamics. A missed or inaccurate interception can reflect motor or measurement factors; it does not by itself identify a cognitive estimation error.

## 4. Test what temporal organization adds

Once a reliable group difference is established, ask when it becomes readable, whether it persists, and whether it changes during occlusion or after a physical event. Report effective time support and reliability at the scale of each claim. Compare these results with endpoint summaries and suitable time-resolved baselines.

A current GPFA state can already contain past neural information. A history comparison should therefore distinguish a current state, an equal-duration history summary, and an ordered trajectory description. Use comparable readout capacity and sample support where feasible. This comparison tests whether temporal order adds value beyond additional observations and noise reduction.

Keep both physical and behavioral references. Geometric interpolation supplies part of the candidate path's shape. An interpretable neural reconstruction requires evidence beyond that shared shape, including comparison with the mean-endpoint geometry baseline. The existing full-epoch readouts did not outperform that baseline.

## 5. Make competing explanations predict different observations

This step requires suitable data and explicit predictions. Endpoint groups alone can be consistent with an early biased estimate that is maintained or an estimate that changes later. A design must state which physical, behavioral, and temporal variables are held fixed and which are changed.

| Candidate comparison | Hold fixed where possible | Change or observe | Distinguishing measurement |
|---|---|---|---|
| Maintained early bias versus later update | Physical path, endpoint-error scale, session and neural support | Independent early and later estimates or an informative late event | Onset and change of a shared signed task readout, with event-window reliability |
| Endpoint goal versus evolving hidden state | Final target or behavior group | Hidden path or event sequence | Time-resolved task readouts and correspondence controls beyond endpoint summaries |
| Temporal order versus additional history samples | Duration, inputs, readout capacity and evaluation support | Ordered versus order-insensitive history descriptions | Held-out performance and reliability for the prespecified temporal contrast |

These are conditional design options, not conclusions from the current data. The [Sensorium temporal intervention](METHODS.md#14-stress-test-methods) motivates capacity-matched comparisons and controls for general model damage. An intervention that lowers all predictive performance does not isolate a temporal computation. Future experiments must measure the signature that separates their candidate processes; closer neural geometry alone cannot identify an algorithm or causal mechanism.
