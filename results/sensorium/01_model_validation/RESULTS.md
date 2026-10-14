# Model Implementation and Reproduction: Results

The historical full Dynamic checkpoint was recovered, reloaded and evaluated on complete local trials, yielding a neuron-weighted response correlation of **0.196673**. The newly trained Static checkpoint passed all **15 single-frame checks**: predictions at other frames were unchanged, while a deliberately coupled control failed.

These findings verify complete-trial local execution and framewise behaviour under the tested perturbations. The full Dynamic run ended before its stopping schedule completed; reproduction of its official hidden-test score remains unverified. The historical Dynamic run and current Static check address separate implementation questions using different checkpoints and preprocessing.

[Setup](SETUP.md) specifies the models, data, training record, scoring and perturbation procedure. The current parameter-matched Static–reduced Dynamic results belong to [Goal 3](../03_model_comparison/RESULTS.md).

## 1. Historical full Dynamic reproduction

### Training and checkpoint recovery

The full-width model used seed 42. Training stopped by project decision after validation at epoch 103, and epoch 104 did not yield a complete checkpoint. Recovery selected the complete state from epoch 97.

| Recorded item | Result |
|---|---|
| Best retained epoch | 97 |
| Last complete epoch | 103 |
| Complete raw epoch checkpoints | 103 |
| Validation events | 213 |
| Best stopping-closure correlation | 0.19800327718257904 |
| Best post-epoch validation correlation | 0.1977536529302597 |
| Natural completion of official early stopping | Not completed |

The stopping closure is the validation calculation used for training decisions. Separate random-snippet calls supplied the stopping closure and post-epoch evaluation, giving different scores. Complete-trial performance is reported separately below. The [training summary](../../../code/sensorium/records/baselines/dynamic/training_summary.json) and [validation events](../../../code/sensorium/records/baselines/dynamic/validation_events.jsonl) preserve these distinct observations.

### Local full-sequence response scores

The reloaded model was evaluated on all 293 labelled repeated-movie trials across five sessions. Predictions and responses were compared at original frames 50–299, using all 40,034 recorded neurons.

| Session identifier | Neurons | Oracle trials | Full-sequence response correlation |
|---|---:|---:|---:|
| 29515-10-12 | 7,863 | 58 | 0.185506448 |
| 29623-4-9 | 7,908 | 56 | 0.203315929 |
| 29647-19-8 | 8,202 | 60 | 0.192512736 |
| 29712-5-9 | 7,939 | 60 | 0.207132056 |
| 29755-2-8 | 8,122 | 59 | 0.194994688 |
| **All neurons, neuron-weighted** | **40,034** | **293** | **0.196673259** |

The exact stored all-neuron score is `0.19667325913906097`. The table rounds scores to nine decimal places. The aggregate averages per-neuron correlations across sessions, weighting each session by its neuron count. Historical scoring details, including the correlation helper's epsilon and handling of NaN values, are defined in [Setup](SETUP.md#6-scoring-and-interpretation).

The [retained evaluation](../evidence/01_baselines/official_dynamic_oracle.json) agrees with the [model-specific local evaluation](../../../code/sensorium/records/baselines/dynamic/official_evaluation.json). The full-sequence fields, rather than `checkpoint_training_validation_score`, supply these results.

<a id="what-was-and-was-not-reproduced"></a>
### Reproduced steps and benchmark coverage

The available evidence supports construction of the full Factorized3D architecture, training of that implementation, recovery and reloading of a complete checkpoint, and local response evaluation with recorded alignment.

The evaluation record also contains a seed-42 hidden `final_test_main` reference of **0.1887**. This reference comes from a different stimulus split. A server evaluation of the local checkpoint remains unrecorded; local hidden arrays contain placeholders rather than the required observed responses. The retained field `official_minus_local=null` marks the cross-split reproduction error as undefined.

The local score establishes performance on the labelled oracle trials, which also informed checkpoint selection, scheduling and stopping. Reloading verifies saved-model execution on those trials. Comparison with the official hidden benchmark would require a separate server-scored test evaluation.

## 2. Complete-trial time alignment

All **293 of 293** historical alignment entries report the same temporal support:

| Alignment check | Recorded value |
|---|---:|
| Source frames per trial | 300 |
| Predictions before the scoring crop | 282 |
| Frames retained for scoring | 250 |
| First scored original frame | 50 |
| Last scored original frame | 299 |

Predictions align to frames 18–299; scoring keeps 50–299. The [alignment record](../../../code/sensorium/records/baselines/dynamic/temporal_alignment.json) verifies the correspondence between prediction and response timestamps.

## 3. Frame independence of the new Static checkpoint

### Tested inputs and outputs

The current check used the newly trained Static checkpoint selected at epoch 59, with train-only preprocessing. It perturbed validation trial `dataset_index=45` from pilot session `29515-10-12` in evaluation mode. Its checkpoint SHA-256 begins `3ded37a0fac494f4`; the full identity is in [Setup](SETUP.md#code-and-checkpoint-identity) and the [execution record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/static_frame_independence.json).

Five pathways were tested at original frames 50, 174 and 299, corresponding to positions 32, 156 and 281 in the adapted output. Across all 15 checks, the maximum absolute change at **other** output frames was exactly **0.0**, against an absolute tolerance of `2e-6` and relative tolerance zero.

The table below reports the maximum absolute effect at the **changed** frame. Values are rounded to six decimal places; full precision is retained in the execution record.

| Perturbed pathway | Frame 50 / output 32 | Frame 174 / output 156 | Frame 299 / output 281 |
|---|---:|---:|---:|
| Movie channel (`input_channel_0`) | 0.541772 | 0.561221 | 0.601010 |
| Appended behaviour channel 1 (`input_channel_1`) | 0.540478 | 0.704840 | 0.701962 |
| Appended behaviour channel 2 (`input_channel_2`) | 0.681345 | 0.838637 | 0.774956 |
| Pupil coordinates (`pupil_center`) | 0.088046 | 0.095166 | 0.045773 |
| Separate behaviour argument (`behavior_argument`) | 0.000000 | 0.000000 | 0.000000 |

Movie, appended behaviour and pupil perturbations affect the corresponding output frame, with other output frames unchanged. The separate behaviour argument is inert in this adapter; the nonzero appended-channel effects show that behaviour remains an active model input.

The execution record reports `mode=eval` and `state_unchanged=true`: model parameters and BatchNorm buffers remained unchanged during these checks.

### Negative control

The deliberately coupled model produced a maximum other-frame error of **0.009258270263671875**, exceeding the same `2e-6` tolerance. The test therefore detected the cross-frame dependence introduced by this control.

The sequence-mean control preserves permutation equivariance while introducing cross-frame dependence. The direct perturbation detects it; the [implementation](../../../code/sensorium/src/trajectory_model_eval/supplementary_encoder.py) and [execution record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/static_frame_independence.json) specify this negative control.

### Scope of the result

The new Static model is frame independent **for the tested pathways and positions on this trial in evaluation mode**. The nonzero changed-frame effects show that the active pathways can alter predictions, while the negative control demonstrates that the check can detect its deliberately introduced coupling.

The result characterises the tested frozen computation. The audit covers evaluation-mode perturbations; training-mode BatchNorm coupling, other inputs and upstream normalisation require separate checks. The synthetic CPU regression adds an implementation check using synthetic inputs. An official Static-on-Dynamic benchmark result for this architecture remains unestablished.

## 4. Historical model identities remain separate

The following scores preserve the identities of the earlier local runs. All three used oracle observations for selection and evaluation. **They describe historical local evaluations; the current held-out comparison has separate model identities.**

| Historical model | Neuron-weighted oracle correlation | Recorded restore path |
|---|---:|---|
| Static | 0.164407775 | `models/static_on_dynamic/best.pt` |
| Reduced Dynamic | 0.187524691 | `models/parameter_matched_dynamic/best.pt` |
| Official full Dynamic | 0.196673259 | `models/official_dynamic/best.pt` |

Scores are rounded to nine decimal places. The historical [Static evaluation](../../../code/sensorium/records/baselines/static/official_evaluation.json), [reduced Dynamic evaluation](../../../code/sensorium/records/parameter_matching/official_evaluation.json) and [model identity table](../evidence/historical_model_records/model_files.csv) retain their original values and paths.

Historical Static training ended at epoch 63, with restored stopping-closure correlation `0.16395580768585205`. Historical reduced Dynamic ended at epoch 99, with closure correlation `0.18789274990558624`. Their training completed separately from the project-stopped full Dynamic run. The relevant [Static](../../../code/sensorium/records/baselines/static/training_summary.json) and [reduced Dynamic](../../../code/sensorium/records/parameter_matching/training_summary.json) summaries remain separate.

The full Dynamic has 5,707,743 parameters. The reduced Dynamic used in the current study has 2,862,063. Scores for the newly trained reduced model and the full Dynamic reproduction describe these distinct architectures and checkpoints. The Static perturbation result above belongs to the newly trained checkpoint.

Current encoder selection histories concern newly trained Static and reduced Dynamic. Section 1 supplies the separate training and validation evidence for historical full Dynamic.

## 5. Interpretation and remaining limits

The implementation evidence supports two conclusions. First, the historical full Factorized3D encoder could be trained, recovered and reloaded to produce the reported complete-trial local score with the declared time alignment. Second, the newly trained Static encoder preserves frame independence under the tested frozen-execution perturbations, and the same test detects a deliberately coupled control.

These checks establish the intended distinction between framewise and temporally convolved predictions. Differences in parameter counts, core operations, initialisation and regularisation leave several possible contributors to performance; the current approximately parameter-matched comparison has its own [settings](../03_model_comparison/SETUP.md) and [results](../03_model_comparison/RESULTS.md).

The retained record distinguishes the local run from the benchmark protocol through its source revisions, corrected channel wiring, disabled file caching, historical `all` normalisation, local software and hardware, and early termination. These differences document execution history; isolating their individual effects on the score would require controlled comparisons.

The reproduced steps cover implementation, checkpoint recovery and local scoring. The full Dynamic schedule ended early, and a same-protocol hidden-server score remains unverified. The stored 0.1887 value remains a separate benchmark reference. The checked sources provide architecture background for Static; its video evaluation is project-specific.

Weights remain separate from their identity records. Loading a model requires its weight payload; the [artifact guide](../../../models/README.md) and [manifest](../../../models/manifest.json) state the applicable locations and access status. Benchmark attribution and implementation references are collected in [Setup](SETUP.md#7-evidence-attribution-and-remaining-limits).
