# Evaluating Dynamic Neural Encoding Beyond Response Correlation

This repository studies what neural population trajectories reveal about temporal structure, task variables, and model predictions. A **neural trajectory** is the time course in a neural representation space. Two completed exploratory studies address different parts of that question.

| Study | Question | Analysis |
|---|---|---|
| [Dynamic Sensorium](docs/RESULTS.md) | Which temporal measurements are reliable, and what do they add to encoding-model evaluation? | Repeat-half validation and Static–Dynamic comparisons in a shared neural space |
| [Mental-Pong](docs/MENTAL_PONG.md) | How well can neural readouts reconstruct objective and behavior-constrained ball paths? | Condition-mean DMFC representations, matched linear readouts, cross-scoring, and randomization controls |

The studies share a sequence of questions: establish measurement reliability, relate measurements to observable variables, and identify the data needed to distinguish competing explanations. Their datasets, GPFA implementations, and validation results remain separate. Scores and noise ceilings are not pooled across studies.

```text
Study 1: repeated neural responses → measurement checks
         → aligned neural/model trajectories → temporal sensitivity and limits
Study 2: condition-mean neural responses → saved FA50/GPFA50 representations
         → objective/candidate position readouts → correspondence and endpoint controls
Joint next design: preserve within-condition behavior groups before averaging
                   → test reliable group differences → test their temporal organization
```

The Mental-Pong study is **CLOSED_EXPLORATORY_WITH_LIMITATIONS**. Its input audit retains **raw_preprocessing=fail**: filtering and fit-source checks apply to the released inputs, while upstream filling across conditions and times remains unresolved. The [integration record](experiments/05_mental_pong/integration/INTEGRATION_SUMMARY.md) describes publication changes and verification separately from scientific completion.

<a id="why-trajectory-evaluation"></a>
## Study 1: Dynamic Sensorium

Response accuracy need not describe population temporal structure. This Dynamic Sensorium 2023 proof of concept asks:

1. **Measurement usability:** are neural trajectories reproducible, distinguishable from relevant nulls, and sensitive to their claimed temporal properties?
2. **Complementary diagnosis:** can trajectory evaluation reveal Static–Dynamic differences not adequately summarized by response accuracy and representation similarity?

Different handling of temporal input makes these models a useful test case, without a presumed winner. Condition-level RSA/CKA omit order; time-aware variants retain temporal structure. Complementarity must address the tested formulations; see [design rationale](docs/DESIGN_RATIONALE.md).

## Working hypothesis and distinguishable outcomes

Usable measurements should agree across independent repeat halves and exceed property-relevant nulls. Reliability alone is insufficient: reversal-invariant metrics cannot establish temporal direction.

Complementary value would allow similar conventional scores alongside reliable trajectory differences. Redundancy, mixed metric utility, and inconclusive differences are admissible; unreliable measurement leaves the question unresolved. These are retrospective working hypotheses and interpretation rules, not preregistered predictions.

## Why these data and measurements

Natural movies, population recordings, and repeated presentations support reliability checks and aligned neural/model comparisons at three complementary levels:

- **Response predictivity:** agreement with individual neural responses.
- **Output-space RSA/CKA:** recorded/predicted population geometry, including temporal variants; not hidden-layer analysis.
- **Trajectory evaluation:** position, local direction, and speed in a common neural-data-defined space.

**Brain-defined GPFA** is fitted only to training neural responses and frozen. Responses held out from that fit and model predictions share coordinates, avoiding model-specific alignment. This does not establish V1's intrinsic dimensionality. Frozen full-window smoothing still uses future observations. [Methods](docs/METHODS.md#8-neural-data-defined-gpfa) defines this offline measurement.

Four-layer frame-wise 2D Static and three-stage Factorized3D Dynamic have approximately matched total trainable counts: **2,814,015 versus 2,862,063**, a difference of **48,048 (1.707%)**. Different cores preclude isolating temporal history alone. Static outputs change with inputs; Dynamic uses temporal convolutions, not recurrence.

<a id="experimental-logic"></a>
## Experimental and analysis flow

```text
Aligned data: movies, repeats, neuron identities, and time support
  ├─ Static / Dynamic predictions → response and output-space RSA/CKA
  └─ Training neural data → fit / freeze GPFA
       → independent repeat halves → reliability + nulls + sensitivity
Validated comparison GPFA + aligned neural / model responses
  → trajectory comparison → three-level comparison and targeted controls
  → interpretation
```

Model preparation can parallel neural-only validation, which must precede trajectory interpretation. The 348-trial development and 174-trial comparison GPFAs require separate [validation evidence](docs/GPFA_VALIDATION.md#4-why-two-gpfa-fits-appear-in-the-repository).

Three stress tests ask distinct questions:

| Test | Why it is included |
|---|---|
| Response-score-matched output perturbation | Can trajectory differences remain when one response-accuracy summary is nearly matched? |
| Time reversal | Which metrics detect disrupted temporal ordering and direction? |
| Graded temporal-weight attenuation | How do metrics respond as learned off-center temporal contributions are weakened? |

See [stress-test methods](docs/METHODS.md#14-stress-test-methods) for implementation and selection/test separation.

## Main findings

### A. Measurement usability

Comparison-GPFA repeat-half position correlation is **0.8583**; velocity-direction cosine is **0.6368**. Position, normalized error, velocity, speed, and acceleration exceed matched nulls in **200/200** splits, reusing 58 trials rather than independent biological samples. [Reliability and sensitivity results](docs/GPFA_VALIDATION.md#7-main-reliability-results) show greater stability for position than speed and higher derivatives.

Path length is repeatable but fails timing/direction nulls and remains descriptive. Single-trial reliability is weaker; low-dimensional reliability does not imply full-response coverage. Posterior derivatives are not independently observed 30 Hz dynamics. These [negative findings](docs/GPFA_VALIDATION.md#9-important-negative-findings-and-measurement-limits) restrict measurement scope.

<a id="1-dynamic-models-predict-neural-responses-better"></a>
<a id="2-the-dynamic-advantage-is-especially-visible-in-temporally-structured-population-geometry"></a>
<a id="what-the-current-proof-of-concept-covers"></a>
### B. Static–Dynamic model differences

Across five sessions, mean per-neuron oracle response correlation increases from **0.1644** for Static to **0.1875** for Total-parameter-matched Dynamic (**+0.0231**); Dynamic is higher in each session. Detailed RSA/CKA and trajectory results cover one pilot: 512 neurons, 58 oracle trials, six repeated movies, frames 50–299. Training uses one seed.

Time-aware RSA/CKA already detect model differences. GPFA shows stronger Dynamic agreement in position, velocity direction, and acceleration direction, but the **speed-profile difference is inconclusive**. Static also captures nontrivial trajectory structure. Full estimates, uncertainty, and scope appear in [Results](docs/RESULTS.md#2-response-level-comparison).

![Static–Dynamic comparison across response, CKA, RSA, and GPFA trajectory evaluation](results/figures/figure-2-static-dynamic-comparison.png)

> **Figure 2.** One-session, six-condition comparison. Response, RSA, and CKA points use observed means; trajectory points use saved bootstrap means. Interpret estimates and uncertainty within each metric family. Absolute effect magnitudes are not comparable across families.

<a id="3-trajectory-metrics-provide-additional-sensitivity-to-temporal-structure"></a>
### C. Complementary diagnostic value

- **Response matching:** position, velocity, and acceleration differences remain on the repeat half held out from perturbation-strength selection; speed remains inconclusive. This output stress test is neither a newly trained accuracy-matched model nor a formal equivalence result.
- **Time reversal:** condition-pattern RSA/CKA remain unchanged while trajectory agreement deteriorates. This counterexample concerns those formulations, not all time-aware RSA/CKA.
- **Temporal-weight attenuation:** four similarity metrics degrade monotonically in the observed curve; normalized RMSE is not strictly monotonic. Generic network damage remains an alternative explanation without a matched non-temporal weight control.

The [Q1–Q6 ledger](docs/results/Q1_Q6_ANSWERS.md#q4-when-response-scores-are-similar-can-trajectory-evaluation-still-detect-a-trajectory-difference) gives complete results. Expanded baselines explain substantial trajectory-score variation; ridge residuals do not establish information independence.

## Study 2: Mental-Pong

The completed study asks which of two position targets is more accurately reconstructed from condition-mean neural responses: the **objective ball path** or a **behavior-constrained candidate path**. The candidate joins a physical anchor to the condition-mean final paddle position. Its geometry is specified independently of the neural readout. It is a behavioral proxy, rather than an observed internal estimate.

Mahler and Perle are analyzed separately. The data are DMFC pseudopopulations: condition-mean neural responses assembled across recording sessions. A physical condition specifies a ball path; it is the train/test grouping unit. The completed version retains 79 condition identities, with 78 valid conditions and 3369 condition-by-time samples per animal. Condition 59920 has an unusable collision anchor. The [study overview](docs/MENTAL_PONG.md) explains masks, timing, and the unknown trial membership of the released neural means.

Factor analysis (**FA**) and Gaussian-process factor analysis (**GPFA**) provide 50-dimensional representations for each saved split. Ordinary least squares (**OLS**) fits a single position readout across valid times, with an intercept and no added regularization. Each target uses the same inputs and 100 nominal 39/40 condition splits. The GPFA implementation has one shared learnable RBF time scale. These are the final FA50/GPFA50 analyses; earlier frozen-GPFA64 studies remain in the [version history](experiments/05_mental_pong/archive/README.md).

Full-epoch y reconstruction favors the objective target in all four animal/representation comparisons. The [main table](experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/results/self_reconstruction_main_table.csv) and [final report](experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_REPORT.md) retain the full precision, split variation, and phase results. Candidate RMSE is lower in no-collision segments even though candidate correlation is lower. After collisions, candidate RMSE is higher.

Fixed-head condition mismatch reduces reconstruction on matched time support. Random-endpoint controls retain positive correlation because the constructed paths share geometry. The actual candidate exceeds that reference in full, hidden, and endpoint-influence evaluations, with post-collision RMSE exceptions. All four full-epoch neural readouts remain below the **mean-endpoint geometry baseline**, which combines a training-set mean endpoint with known task geometry; some Perle phase-specific skills are slightly positive. These controls establish correspondence within the measured data scope. They do not establish a full-epoch candidate-path advantage.

Opposite endpoint errors partly cancel in behavioral condition means: 77/78 Mahler conditions and 78/78 Perle conditions contain both signs. The [averaging audit](experiments/05_mental_pong/trajectory_project/closeout_v1/results/descriptive/mean_cancellation_interpretation.md) measures this behavioral effect. Unknown neural trial membership prevents a corresponding test of behavior-group neural differences. Behavioral record counts and repeated splits are not independent neural observations.

## Interpretation and future design

Study 1 supports repeat-averaged measurement and complementary diagnostics within its tested conditions. Study 2 supports recoverable task-position structure, while the full-epoch results favor the objective target. Neither study identifies a unique neural algorithm or causal mechanism.

The [unified future plan](docs/FUTURE_DIRECTIONS.md) first requires neural and behavioral trial identities that preserve opposite choices within the same physical condition. Reliable group differences would support a later comparison of their onset, persistence, and change. That comparison must distinguish information already in a current GPFA state from equal-duration history summaries and ordered trajectories. No experiments in this plan were run during integration.

<a id="documentation"></a>
## Repository orientation and reproduction

```text
docs/          detailed methods, validation, results, and design rationale
experiments/   phase-specific code, configurations, commands, and outputs
models/        encoding-model weights and frozen GPFA objects
results/       figures, compact result tables, and artifact manifests
```

Raw data are not redistributed. The [Data and Reproducibility Guide](docs/DATA_AND_REPRODUCIBILITY.md) covers official data, Git LFS, environments, tests, and phase READMEs.

Read this README first, then [Mental-Pong](docs/MENTAL_PONG.md), its [final scientific report](experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_REPORT.md), and the [future plan](docs/FUTURE_DIRECTIONS.md). The [claim-to-source index](results/study_evidence_index.csv) links both studies to their supporting records.

From the repository root, check published links and artifacts without rerunning science:

```bash
python scripts/verify_integration.py
python experiments/05_mental_pong/run.py
```

A [separate Mental-Pong environment](experiments/05_mental_pong/README.md) supports a small saved-weight replay. Large predictions, latent arrays, and null-study artifacts remain in configured external storage. The adapter reports missing inputs explicitly. Full-analysis requirements are available through `python experiments/05_mental_pong/run.py --full-plan`; the complete scientific run was not repeated during integration.

## License, citation, and contact

Original project code and documentation are released under the [MIT License](LICENSE). Dataset use remains subject to each official provider's terms; upstream attribution and license boundaries are recorded in [Third-Party Notices](THIRD_PARTY_NOTICES.md).

Use [`CITATION.cff`](CITATION.cff) to cite this repository and cite the relevant dataset and paper separately. The primary author is [Xiaotian Zhu](AUTHORS.md); reproducibility questions may be submitted through [GitHub Issues](https://github.com/124585083/neural-trajectory-evaluation/issues).

## References

1. Wang et al. (2024). [Retrospective for the Dynamic Sensorium Competition for predicting large-scale mouse primary visual cortex activity from videos](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html). *NeurIPS 2024 Datasets and Benchmarks Track*.
2. Willeke et al. (2023). [Retrospective on the SENSORIUM 2022 competition](https://proceedings.mlr.press/v220/willeke23a.html). *Proceedings of Machine Learning Research, 220*.
3. Yu et al. (2009). [Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity](https://doi.org/10.1152/jn.90941.2008). *Journal of Neurophysiology, 102*(1), 614–635.
4. Kriegeskorte, Mur, and Bandettini (2008). [Representational similarity analysis—connecting the branches of systems neuroscience](https://doi.org/10.3389/neuro.06.004.2008). *Frontiers in Systems Neuroscience, 2*.
5. Kornblith et al. (2019). [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). *Proceedings of Machine Learning Research, 97*, 3519–3529.
6. Rajalingham et al. (2025). [Dynamic tracking of objects in the macaque dorsomedial frontal cortex](https://www.nature.com/articles/s41467-024-54688-y). Mental-Pong [official code](https://github.com/jazlab/MentalPong) and [released data](https://doi.org/10.5281/zenodo.13952210).
