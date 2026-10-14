# Model Implementation and Reproduction: Setup

## 1. Research question

This analysis establishes the implementation basis for trajectory comparisons: **does the encoder perform the intended computation, and which training and evaluation steps have been reproduced for the official full Dynamic baseline?**

The historical check recovers and evaluates official full-width Dynamic on complete trials. The current check tests cross-frame dependence of the newly trained Static checkpoint.

| Check | Evidence sought | Role in the study |
|---|---|---|
| Historical full Dynamic reproduction | Recorded architecture, training, checkpoint recovery, reloading and local response scores | Establish what was implemented and which benchmark comparisons are justified |
| Current Static frame independence | Effects at the perturbed frame, unchanged predictions elsewhere, and detection of a deliberately coupled control | Check the intended framewise computation during frozen evaluation |
| Parameter and time-alignment checks | Model size, output shape and correspondence to original response frames | Identify model size and the shared evaluation support |

Each check has its own checkpoint, preprocessing and data roles, so they address complementary implementation questions. [Results](RESULTS.md) reports their findings; [Goal 3](../03_model_comparison/SETUP.md) defines the current comparison.

![Current retrained comparison pathway and timestamp alignment](../figures/documentation/01_shared_pathway.png)

*Data roles and timestamp alignment for the current retrained comparison. The historical full Dynamic run described below uses its own preprocessing and oracle-based selection protocol.*

## 2. Data and preprocessing

### Historical full Dynamic run

The historical run used five competition sessions. The upstream tier named **`oracle`** contains labelled repeated-movie trials. In this run it supplied learning-rate scheduling, stopping and checkpoint selection, as well as the subsequent local evaluation.

| Session identifier | Original train trials | Oracle trials | Recorded neurons |
|---|---:|---:|---:|
| 29515-10-12 | 348 | 58 | 7,863 |
| 29623-4-9 | 329 | 56 | 7,908 |
| 29647-19-8 | 354 | 60 | 8,202 |
| 29712-5-9 | 359 | 60 | 7,939 |
| 29755-2-8 | 354 | 59 | 8,122 |
| **Total** | **1,744** | **293** | **40,034** |

Full session keys use `dynamic<identifier>-Video-9b4f6a1a067fe51e15306b9628efea20`. Local hidden final-test arrays are zero placeholders. Hidden-benchmark scoring requires the observed responses held by the benchmark provider.

The source arrays contain grayscale movies, neural responses, two behavioural channels and pupil coordinates. A documented trial from the first session has the following shapes:

| Field | Axes | Example source shape |
|---|---|---|
| `videos` | Height × width × time | 36 × 64 × 324 |
| `responses` | Neuron × time | 7,863 × 324 |
| `behavior` | Channel × time | 2 × 324 |
| `pupil_center` | Coordinate × time | 2 × 324 |

Cropping to the common finite interval produces 300 aligned frames on the released 30 Hz grid. The responses are processed calcium signals whose scale is defined by the recorded preprocessing; absolute spike-rate units are unspecified. The historical description reports approximately 8 Hz acquisition before resampling. The released 30 Hz grid represents resampled calcium signals rather than independent acquisitions at that rate. The records leave the behavioural channels’ calibrated physical units unspecified.

Historical normalisation used the supplied arrays selected by `stats_source="all"`. This setting defines the historical protocol; the new run uses its own train-only preprocessing. Data and scoring details are recorded in the [full Dynamic evaluation](../evidence/01_baselines/official_dynamic_oracle.json) and [training record](../../../code/sensorium/records/baselines/dynamic/training_summary.json).

### Current Static execution check

The new-checkpoint test uses validation trial `dataset_index=45` from pilot session `29515-10-12`. Its input has shape `[1, 3, 300, 36, 64]`: one movie channel and two behavioural traces broadcast over the image. Pupil coordinates enter the separate shifter. A separately passed `behavior` argument is tested independently of the behavioural channels already present in the core input.

This check uses the new train-only preprocessing and the Static checkpoint selected at epoch 59. The tested original frames are 50, 174 and 299. After the adapter removes the first 18 outputs, these correspond to positions 32, 156 and 281 in the 282-frame output, before the final scoring crop.

The repository also contains a small CPU regression using synthetic 27-frame inputs and seven neurons. It supplies a separate implementation test; the 15 new-checkpoint observations come from the recorded validation trial. See the [execution record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/static_frame_independence.json) and [regression tests](../../../code/sensorium/tests/test_static_frame_independence.py).

## 3. Model structures and fixed identities

### Architectures

Full Dynamic applies spatial then temporal convolutions in three Factorized3D stages. Static applies one shared 2D core framewise. It serves as a project-built framewise video control; an identical official video benchmark is not provided.

| Component | Official full Dynamic | Project Static |
|---|---|---|
| Core widths | 32 → 64 → 128 | 64 throughout four layers |
| Spatial operations | Kernels 11 × 11, 5 × 5 and 5 × 5; valid padding | Initial valid 9 × 9 convolution; then three 1 × 1 → depthwise 7 × 7 → 1 × 1 blocks, with padding 3 for the depthwise operations |
| Temporal operations | Kernels of 11, 5 and 5 frames; valid padding; 19-frame receptive field; no recurrent unit | Reshape `[B, 3, T, H, W]` to `[B*T, 3, H, W]`; restore time after framewise processing; crop the first 18 outputs |
| Normalisation and activation | BatchNorm3d, momentum 0.7; ELU | BatchNorm2d, momentum 0.9; AdaptiveELU |
| Core regularisation | Spatial `LaplaceL2norm`: 10; temporal `DepthLaplaceL21d`: 0.01 | Input `LaplaceL2norm`: 6.3831; hidden coefficient: 0 |
| Gaussian readout | Five session-specific `FullGaussian2d` readouts with full covariance; `init_mu_range=0.2`, `init_sigma=1`, `gamma_readout=0` | Same family; `init_mu_range=0.3`, `init_sigma=0.1`, `gamma_readout=0.0076` |
| Anatomical grid predictor | Cortical coordinates: 2 → 30 → 2 → tanh; no shared features or grid | Same structure |
| Pupil shifter | Per-session MLP: 2 → 5 → 5 → 2, with tanh; `gamma=0` | Same structure |
| Output nonlinearity | ELU + 1 | ELU + 1 |

Here, `B`, `T`, `H`, `W` mean batch, time, height and width. Learned parameters, initialisation and regularisation remain specific to each model.

| Parameter group | Official full Dynamic | Project Static |
|---|---:|---:|
| Core | 382,176 | 50,624 |
| Readout | 5,325,282 | 2,763,106 |
| Pupil shifter | 285 | 285 |
| **Total** | **5,707,743** | **2,814,015** |

These counts identify the implementations. **The current approximately parameter-matched comparison uses a separate model from full Dynamic.** That comparison uses reduced Dynamic with widths `[16, 32, 64]` and 2,862,063 parameters, described in [Models and Data](../MODELS_AND_DATA.md#4-models-and-training).

### Code and checkpoint identity

The historical configuration is [phase1A_dynamic_official.yaml](../../../code/sensorium/configs/phase1A_dynamic_official.yaml). Its retained source revisions are:

| Dependency | Source commit |
|---|---|
| `sensorium_2023` | `0e02656220e84a50f3be1b92d6f66c2f9ccd51ef` |
| `neuralpredictors` | `efdda679596517fad95d71f36d0385d7450dd207` |

An earlier `neuralpredictors` pin, `43faededa2d2e76bb904f38a49b9d8b81d287a0a`, had a variable-width channel-wiring issue. The run used the corrected upstream implementation. The [architecture audit](../../../code/sensorium/records/baselines/dynamic/architecture_audit.json) records the source revisions, layers, shapes and counts.

The historical full Dynamic state dictionary has recorded restore path `models/official_dynamic/best.pt` and SHA-256:

```text
263d7abb29956934d529e8ad600337a8ea0930f4ff6d8b0dd0d08e94849e072f
```

The current Static checkpoint used for the perturbation check has SHA-256:

```text
3ded37a0fac494f47a8553a432e04e8be7e95a0a99ed9444b07ae61f31ff7052
```

The historical restore path identifies the artifact; execution requires its separately obtained weights. Current locations and access requirements belong to the [model manifest](../../../models/manifest.json) and [artifact guide](../../../models/README.md). Loading the model requires both the parameter payload and its matching manifest.

## 4. Historical training, recovery and full-sequence evaluation

### Training protocol

The historical full Dynamic run used `sensorium.training.video_training_loop.standard_trainer` with the following settings. These settings apply to that historical run; the later retraining protocol is documented separately.

| Setting | Value |
|---|---|
| Random seed | 42 |
| Training input | Random valid 80-frame snippets |
| Physical batch | Eight snippets per session |
| Gradient accumulation | Five sessions; effective batch of 40 snippets |
| Objective | Summed Poisson loss plus architecture regularisers |
| Optimiser | AdamW; learning rate 0.005; betas (0.9, 0.999); epsilon `1e-8`; weight decay 0.01; AMSGrad off |
| Learning-rate scheduler | ReduceLROnPlateau; multiplier 0.3; up to four decay stages |
| Stopping settings | Patience 5; tolerance `1e-6`; minimum learning rate `1e-4`; maximum 200 epochs; restore best enabled |
| Numerical precision | FP32; no mixed precision or activation checkpointing |
| File caching | Disabled for local memory use; source arrays and transformations retained |

Oracle correlation supplied scheduling, stopping and checkpoint selection. The run was stopped by project decision after validation at epoch 103. Epoch 104 did not produce a complete checkpoint. Recovery retained epoch 97 from the complete epoch states. **Training was stopped by project decision before the official early-stopping schedule completed.** The [training summary](../../../code/sensorium/records/baselines/dynamic/training_summary.json) and [validation-event log](../../../code/sensorium/records/baselines/dynamic/validation_events.jsonl) retain the execution history.

### Time support

| Operation | Input support | Prediction or scoring support |
|---|---|---|
| Training snippet | 80 frames | 62 predictions aligned to snippet frames 18–79 |
| Historical snippet validation | Same snippet convention | Response frames 50–79 |
| Reloaded complete-trial inference | Original frames 0–299 | 282 predictions aligned to original frames 18–299 |
| Local full-sequence scoring | Recorded responses and complete-trial predictions | Original frames 50–299; 250 timestamps |

Output cropping and scoring burn-in have different roles, as shown above. The [alignment records](../../../code/sensorium/records/baselines/dynamic/temporal_alignment.json) retain complete-trial frame indices; no state is carried across trials.

## 5. Current Static frame-independence test

The frozen Static model is tested in evaluation mode at original frames 50, 174 and 299. Separate perturbations of movie, each appended behaviour channel, pupil and the separate behaviour argument give 15 pathway-by-position checks.

At each perturbed position `j`, the analysis records the changed-frame effect and maximum change elsewhere:

```text
other_frame_error   = max(abs(perturbed_output[k, n] - baseline_output[k, n]))
                      over k != j and all neurons n

changed_frame_effect = max(abs(perturbed_output[j, n] - baseline_output[j, n]))
                       over all neurons n
```

Other-frame tolerance is absolute `2e-6`, relative zero. Changed-frame effects identify active inputs. Parameter and BatchNorm-buffer checks verify that the model remains fixed during evaluation.

The negative control adds a sequence mean to each output. Its sequence-mean term couples frames while preserving permutation equivariance, giving the perturbation test a known cross-frame effect to detect. Its [implementation](../../../code/sensorium/src/trajectory_model_eval/supplementary_encoder.py) is `supplementary_encoder._negative_control`.

The [execution record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/static_frame_independence.json) identifies `checks`, `negative_control`, `state_unchanged` and `checkpoint_sha256`. Exact perturbation amplitudes must be read from the linked implementation and execution record; the protocol summary leaves them unspecified.

## 6. Scoring and interpretation

Historical local response scoring uses `sensorium.utility.scores.get_correlations`. It excludes response frames 0–49, end-aligns predictions, concatenates trial and time samples for each neuron, computes each neuron's normalised correlation, and averages those values over all recorded neurons.

If session `s` contains `N_s` neurons and its mean score is `r_s`, the final aggregation is:

$$
r_{\mathrm{all}} = \frac{\sum_s N_s r_s}{\sum_s N_s}.
$$

The historical scalar is therefore **neuron-weighted**; each of the five sessions contributes in proportion to its neuron count. The pinned helper uses standard deviations with `ddof=0` and epsilon `1e-8`; the historical wrapper replaces NaN neuron correlations with zero. These implementation details define the recorded historical score, which retains its own convention alongside the current pilot metric.

The stopping-closure and post-epoch validation scores come from separate random-snippet calls. Both describe snippet-level validation; the complete-sequence local score is reported separately. The retained evaluation also lists a seed-42 hidden `final_test_main` reference of 0.1887, but that reference concerns a different stimulus split. A same-protocol reproduction error would require scores from the same stimulus split and evaluation procedure.

The historical record covers one training seed. Historical paired uncertainty and an independently server-scored hidden result remain unverified. Frame errors and parameter counts are deterministic execution diagnostics; their role is to verify implementation and model identity.

## 7. Evidence, attribution and remaining limits

The main project records for this goal are:

| Evidence | Record and relevant content |
|---|---|
| Local full Dynamic evaluation | [Evaluation JSON](../evidence/01_baselines/official_dynamic_oracle.json): `implementation`, `full_sequence_oracle_single_trial`, `full_sequence_oracle_by_session`, `acceptance`, `reason` |
| Training and recovery | [Training summary](../../../code/sensorium/records/baselines/dynamic/training_summary.json): `best_epoch`, `last_complete_epoch`, `full_official_early_stopping_completed`; [validation events](../../../code/sensorium/records/baselines/dynamic/validation_events.jsonl) |
| Architecture and frame support | [Architecture audit](../../../code/sensorium/records/baselines/dynamic/architecture_audit.json); [temporal alignment](../../../code/sensorium/records/baselines/dynamic/temporal_alignment.json) |
| Model construction and evaluation | [Full Dynamic source](../../../code/sensorium/src/trajectory_eval/official_dynamic.py): `make_official_loaders`, `build_official_model`, `recover_user_stopped_training`, `evaluate_best`; [Static source](../../../code/sensorium/src/trajectory_eval/static_dynamic.py): `build_model`, `audit_frame_independence` |
| Current Static checks and model sizes | [Frame-check record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/static_frame_independence.json); [parameter record](../evidence/04_model_comparison/retrained_v1_diagnostics/encoder/parameter_counts.json): `actual_instances`, `shared_parameters_counted_once` |

Turishcheva et al. (2024), *Retrospective for the Dynamic Sensorium Competition for predicting large-scale mouse primary visual cortex activity from videos*, provides the [Dynamic benchmark context](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html). Willeke et al., *Retrospective on the SENSORIUM 2022 competition*, provides the [static-architecture context](https://proceedings.mlr.press/v220/willeke23a.html), retained in the project reference guide as 2022, PMLR 220:314–333. These sources identify the architecture background; the Static-on-Dynamic evaluation is project-specific.

The retained evidence covers local full Dynamic scoring after a project-directed training stop; hidden-server evaluation remains unverified. The Static perturbation test covers the specified inputs and positions in evaluation mode. Training-mode BatchNorm coupling and upstream normalisation provenance require separate checks. These boundaries define the implementation evidence carried forward to the model comparison. The associated numerical findings and their interpretation are in [Results](RESULTS.md).
