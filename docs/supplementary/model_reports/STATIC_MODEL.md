# Static Encoding Model: Sensorium 2023 Training and Validation Report

Last updated: 2026-08-11

## One-page summary

I trained the full Sensorium+ static CNN architecture on the five official Dynamic Sensorium 2023 competition sessions. This is neither a reduced model nor a total-parameter-matched variant. It preserves the official static baseline's four-layer 2D CNN core, Gaussian readout, cortical-coordinate grid predictor, pupil shifter, and behavioral inputs. To apply the static model to dynamic data, I added only a parameter-free framewise reshape/crop adapter so that its output time indices are exactly aligned with those of the Dynamic Factorized3D model.

Using seed 42, training naturally terminated through the official Dynamic Sensorium 2023 early-stopping procedure at epoch 63. After reloading the frozen `best.pt`, I independently ran inference on complete 300-frame oracle trials and discarded the first 50 frames according to the official evaluation rule. The resulting single-trial correlation, aggregated across five sessions and 40,034 neurons, was:

```text
Static full-sequence oracle correlation = 0.1644077748
Dynamic full-sequence oracle correlation = 0.1966732591
absolute difference                     = 0.0322654843
Static / Dynamic                        = 0.83594
```

This shows that the static model can be trained and evaluated stably on Dynamic Sensorium 2023, but under the same data, neurons, temporal interval, and response metric, its response-prediction performance is lower than that of the current Dynamic Factorized3D model. This is a local oracle comparison, not a Sensorium 2022 static benchmark result and not a Sensorium 2023 hidden `final_test_main` ranking.

## 1. Scientific question and interpretation of the result

The goal is to construct a reference model with no explicit temporal modeling: at each time point, the model reads only the current video frame and current behavioral state, and independently predicts the neural response at that time point. It cannot access past or future frames. Matching data, neurons, readouts, and evaluation intervals removes those particular differences, but Static and Dynamic still use different core architectures, convolutional operations, channel structures, parameterizations, and inductive biases. Their complete-model comparison therefore cannot attribute every observed difference exclusively to temporal history.

The model in this report should be described precisely as:

> The full Sensorium+ static CNN architecture, retrained on the five official Dynamic Sensorium 2023 sessions using the 2023 training protocol.

It is not equivalent to either of the following:

- It is not a local reproduction of the native Sensorium 2022 static benchmark, which uses static-image data and therefore defines a different task.
- It is not an officially released Static-on-Dynamic benchmark from Sensorium. No official target score exists for this specific transfer setting. This project combines the official static architecture with the official 2023 data/trainer and validates comparability through explicit temporal-alignment tests.

The Sensorium 2022 official repository and tutorial provide the static CNN baseline's data, model, and training interfaces, whereas Sensorium 2023 provides the dynamic data loader, five-session training procedure, and evaluation protocol. References: [Sensorium 2022 official repository](https://github.com/sinzlab/sensorium), [Sensorium 2022 competition retrospective](https://proceedings.mlr.press/v220/willeke23a.html), and [Dynamic Sensorium 2023 official starter kit](https://github.com/ecker-lab/sensorium_2023).

## 2. Data scope and on-disk structure

### 2.1 Data location and integrity

Physical data location:

```text
data/sensorium_all_2023
```

The project reads the same data through the directory junction:

```text
data/sensorium_all_2023
```

All Sensorium data and original archives are retained. Each of the five competition-session archives passed verification against the official MD5 checksums. No Sensorium data were deleted, moved, or rewritten during training or report preparation.

### 2.2 Five sessions used for formal training

| Session | Train trials | Oracle trials | Neurons |
|---|---:|---:|---:|
| `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20` | 348 | 58 | 7,863 |
| `dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 329 | 56 | 7,908 |
| `dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 60 | 8,202 |
| `dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 359 | 60 | 7,939 |
| `dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 59 | 8,122 |
| **Total** | **1,744** | **293** | **40,034** |

No sessions or neurons were removed. Responses in the hidden test tiers are withheld/zero placeholders locally, so a formal hidden-test score cannot be computed from the local files.

### 2.3 File structure of a single session

```text
session/
├── data/
│   ├── videos/<trial_index>.npy
│   ├── responses/<trial_index>.npy
│   ├── behavior/<trial_index>.npy
│   └── pupil_center/<trial_index>.npy
├── meta/
│   ├── trials/{tiers,trial_idx,animal_id,session,scan_idx,target_fps}.npy
│   ├── neurons/{unit_ids,cell_motor_coordinates,...}.npy
│   └── statistics/{videos,responses,behavior,pupil_center}/...
└── config.json
```

For example, a raw trial from the first session has:

```text
videos:        [36, 64, 324]     float64
responses:     [7863, 324]       float64
behavior:      [2, 324]          float64
pupil_center:  [2, 324]          float64
```

The tails of the raw arrays may contain invalid regions used for synchronization or padding. The official `CutVideos(max_frame=None)` transform crops all modalities to their common finite interval, yielding 300 valid time points. The target frame rate is 30 Hz.

### 2.4 Tensor contract after loading

During training, a contiguous 80-frame snippet is randomly sampled from each 300-frame trial:

```text
video + behavior channels: [B, 3, 80, 36, 64]
responses:                 [B, N_session, 80]
behavior:                  [B, 2, 80]
pupil_center:              [B, 2, 80]
```

The three core input channels consist of one grayscale video channel plus two behavior traces broadcast over the spatial dimensions. `pupil_center` is not merged into these three channels; it is passed separately as a two-dimensional input to the shifter.

The loader uses the official `NeuroNormalizer(stats_source="all")` transformations. All model comparisons share the same data files, official tiers, sampling schedule, normalization, and neuron ordering.

## 3. How the Static model is applied to dynamic data

### 3.1 Framewise computation

Given an input:

```text
x: [B, 3, T, 36, 64]
```

I first apply a parameter-free reshape:

```text
[B, 3, T, 36, 64]
    -> [B, T, 3, 36, 64]
    -> [B*T, 3, 36, 64]
```

Each frame is then passed independently through the same 2D core:

```text
[B*T, 3, 36, 64]
    -> [B*T, 64, 28, 56]
```

The Gaussian readout for the corresponding session and the pupil-center shifter for the same frame then produce:

```text
[B*T, N_session] -> [B, T, N_session]
```

This procedure contains no temporal convolution, recurrent state, frame mixing, or history buffer.

### 3.2 Temporal alignment with the Dynamic model

The Dynamic Factorized3D model has effective temporal kernels of 11, 5, and 5 frames. An 80-frame input therefore loses 18 frames and produces 62 outputs. To make the Static training target exactly match the Dynamic target, I retain only source frames 18–79 from the Static model's 80 framewise outputs:

```text
all static outputs: [B, 80, N]
aligned outputs:    [B, 62, N] = all_outputs[:, 18:80]
```

This adapter has zero trainable parameters. Discarding 18 frames only matches the valid-convolution support of the Dynamic core; it is distinct from the official 50-frame evaluation burn-in.

For full-trial evaluation:

```text
source input                         300 frames
static adapter output               frames 18–299, 282 predictions
official response burn-in           discard response frames 0–49
official end-crop of predictions    keep last 250 predictions
final aligned interval              original frames 50–299
```

Thus, there is no process in which 80-frame windows are repeatedly predicted and then concatenated. Training uses random 80-frame snippets; final oracle evaluation performs one forward pass over each complete 300-frame trial.

### 3.3 Verification of no temporal leakage

In evaluation mode, I randomly permuted the input frames together with their corresponding pupil-center frames and then inverted the permutation on the outputs. Every prediction matched the original output exactly:

```text
maximum frame-permutation error = 0.0
```

This demonstrates that the Static model's output at a given time point does not depend on any other time point and that no accidental temporal leakage was introduced by batch normalization or reshaping.

## 4. Model architecture

### 4.1 2D core

| Layer | Operation | Output channels | Kernel / padding | Normalization | Nonlinearity |
|---|---|---:|---|---|---|
| 0 | `Conv2d` | 64 | 9×9, valid | BatchNorm2d, momentum 0.9 | AdaptiveELU |
| 1 | depth-separable block | 64 | 1×1 → depthwise 7×7, pad 3 → 1×1 | BatchNorm2d | AdaptiveELU |
| 2 | depth-separable block | 64 | same as above | BatchNorm2d | AdaptiveELU |
| 3 | depth-separable block | 64 | same as above | BatchNorm2d | AdaptiveELU |

Full configuration:

```text
layers                4
hidden_channels       64
input kernel          9×9
hidden kernels        7×7 depth-separable
skip                  0
hidden dilation       1
stack                 last layer only
input regularizer     LaplaceL2norm
gamma_input           6.3831
gamma_hidden          0
final nonlinearity    enabled
```

For a single-frame input `[B,3,36,64]`, the core output is `[B,64,28,56]`.

### 4.2 Readout, shifter, and output

Each session uses an independent `FullGaussian2d` readout:

```text
gauss_type             full
bias                   true
init_mu_range          0.3
init_sigma             0.1
gamma_readout          0.0076
share_features/grid    false
grid predictor         cortical coordinates: 2 -> 30 -> 2 -> tanh
```

The shifter is also session-specific:

```text
input                  normalized pupil center, 2-D
MLP                    2 -> 5 -> 5 -> 2
activation             tanh after each layer
gamma_shifter          0
```

The final firing-rate nonlinearity is:

```text
ELU(x) + 1
```

Therefore, the output is a strictly positive predicted rate:

```text
training output: [B, 62, N_session]
full raw output: [B, 282, N_session]
evaluated output:[B, 250, N_session]
```

### 4.3 Parameter count

| Component | Trainable parameters |
|---|---:|
| 2D core | 50,624 |
| Five Gaussian readouts | 2,763,106 |
| Five pupil shifters | 285 |
| temporal adapter | 0 |
| **Total** | **2,814,015** |

The parameter audit additionally locks the four-layer core, all five sessions, per-session neuron counts, single-frame feature shape, and the zero-parameter property of the adapter. Any change to width, depth, kernel, readout, or neuron count causes the corresponding test to fail.

## 5. Training and optimization

### 5.1 Formal training settings

| Item | Setting |
|---|---|
| Seed | 42 |
| Trainer | `sensorium.training.video_training_loop.standard_trainer` |
| Maximum epochs | 200 |
| Physical batch | 8 80-frame snippets per session |
| Session accumulation | 5 sessions |
| Effective batch | 40 snippets / optimizer step |
| Epoch structure | 225 session microbatches / 45 optimizer steps |
| Loss | summed Poisson loss, `average_loss=false` |
| Dataset scaling | `scale_loss=true`, scaled by the square root of session dataset size / batch size |
| Optimizer | AdamW |
| Initial LR | 0.005 |
| Betas / epsilon | (0.9, 0.999) / 1e-8 |
| Weight decay | 0.01 |
| AMSGrad | false |
| Scheduler | ReduceLROnPlateau, mode=max |
| LR factor | 0.3 |
| Patience | 5 |
| Tolerance | 1e-6 absolute |
| Minimum LR | 1e-4 |
| LR-decay stages | up to 4 |
| Restore best | true; restore the best state at each decay and at the end of training |
| Checkpoint selection | oracle single-trial correlation |
| Precision | FP32; AMP disabled |

`LongCycler` cycles shorter session loaders until they match the length of the longest loader. Each optimizer step accumulates gradients across all five sessions. The model was not reduced, neurons were not removed, and spatial resolution was not lowered to fit GPU memory.

### 5.2 Regularization

The total optimization objective includes:

```text
scaled summed Poisson NLL
+ core LaplaceL2norm regularizer (gamma_input = 6.3831)
+ readout feature regularizer (gamma_readout = 0.0076)
+ shifter regularizer (gamma_shifter = 0)
```

The Static core uses `gamma_hidden=0`. Although some coefficients are zero, the corresponding modules and interfaces remain part of the official architecture.

### 5.3 Hyperparameter selection principle

I did not search model width, depth, or kernels against the final oracle score. Static core/readout hyperparameters are locked directly to the official Sensorium+ static architecture. The Dynamic Sensorium 2023 loader, Poisson objective, AdamW optimizer, batch accumulation, scheduler, and early-stopping procedure are kept consistent with the Dynamic baseline.

The only newly introduced design element is the zero-parameter temporal adapter. Its 18-frame crop is determined exactly by the temporal reduction of the Dynamic model's valid temporal kernels and was not tuned from the results.

### 5.4 Local execution performance

```text
GPU                         NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB
precision                   FP32
measured microbatches       200 after 20 warmups
seconds/session microbatch  0.4288
seconds/optimizer step      2.1438
estimated train epoch       96.47 s, excluding validation
peak allocated VRAM         8.33 GiB
nvidia-smi peak memory      15,995 MiB
actual elapsed training     26,013.44 s = 7.226 h
```

The data-file cache was disabled to avoid filling 31 GB of system RAM with raw arrays from all five sessions. This changes only the I/O caching strategy; samples, transforms, numerical values, ordering, and model behavior are unchanged.

## 6. Training completion, checkpoints, and final validation

### 6.1 Training endpoint

```text
status                     training_complete
last epoch                 63
epochs completed           64 (0–63)
validation events          138
official restored score    0.1639558077
elapsed                    7.226 h
```

The official early-stopping routine terminated naturally and restored the best weights. The highest individual intermediate validation event in the log was `0.1704876721`. Because random positions of the 80-frame oracle subsequences cause event-level variation, this instantaneous value is not the formal reloaded result of the frozen `best.pt` and should not replace the final score.

### 6.2 Independent full-sequence evaluation

In a fresh process, I reconstructed the full architecture, strictly loaded `best.pt`, changed the oracle loader to batch size 1 with `to_cut=false`, passed complete 300-frame trials, and called the official `sensorium.utility.scores.get_correlations` function:

| Session | Full-sequence oracle correlation |
|---|---:|
| `dynamic29515-10-12` | 0.1536800861 |
| `dynamic29623-4-9` | 0.1660582572 |
| `dynamic29647-19-8` | 0.1608951837 |
| `dynamic29712-5-9` | 0.1847600043 |
| `dynamic29755-2-8` | 0.1568398625 |
| **Neuron-weighted all-session mean** | **0.1644077748** |

The official metric aggregates results as follows:

1. Discard response frames 0–49 from each trial.
2. Select the same number of frames from the end of the model output, yielding original frames 50–299.
3. Concatenate all oracle trials and time points within each neuron.
4. Compute Pearson correlation separately for each neuron.
5. Concatenate all 40,034 per-neuron correlations across the five sessions and take the mean.

### 6.3 Comparison with the Dynamic model under the same metric

| Model | Parameters | Full-sequence oracle correlation |
|---|---:|---:|
| Static Sensorium+ framewise | 2,814,015 | 0.1644077748 |
| Dynamic Factorized3D | 5,707,743 | 0.1966732591 |

The Dynamic model exceeds the Static model by `0.0322654843`, corresponding to `19.63%` relative to the Static score. This is a descriptive difference measured on the same five sessions, with the same neuron-weighted metric and the same frames 50–299. It cannot by itself establish that the difference is caused by temporal computation because the models also differ in core architecture, core parameterization, convolutional operations, inductive bias, effective computation, and total parameter count. Downstream trajectory evaluation must therefore retain response correlation as the performance gate while separately testing latent dynamics.

## 7. Reproducibility files

```text
configuration
experiments/01_baselines/configs/static_dynamic_sensorium2023.yaml

implementation
experiments/01_baselines/src/trajectory_eval/static_dynamic.py

architecture audit
experiments/01_baselines/records/static/architecture_audit.json

training logs
experiments/01_baselines/records/static/validation_events.jsonl
experiments/01_baselines/records/static/training_summary.json

full-sequence evaluation
experiments/01_baselines/records/static/official_evaluation.json

checkpoint
models/static_on_dynamic/best.pt
```

## 8. Current conclusion and limitations

I consider this Static-on-Dynamic model to have met the following usability criteria:

- full official static architecture, without reduction;
- full five-session Dynamic Sensorium 2023 data scope;
- behavioral channels, pupil shifters, and Gaussian readouts retained;
- exact alignment with the Dynamic model's training targets and final 250-frame evaluation interval;
- no temporal leakage;
- normal completion of the official early-stopping procedure;
- independently reloadable `best.pt`, reproducing `0.1644077748` on complete oracle trials.

Three qualifications must still be retained:

1. This is a Dynamic Sensorium 2023 oracle result, not a Sensorium 2022 static benchmark reproduction.
2. Hidden `final_test_main` labels are unavailable locally, so there is no Sensorium server hidden-test score.
3. The response comparison alone shows only that the Dynamic model achieves higher predictive correlation. Trajectory-level conclusions are evaluated separately using the frozen brain-based GPFA and the result-blind locked null/reliability protocol.
