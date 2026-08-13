# Dynamic Encoding Model: Sensorium 2023 Factorized3D Reproduction Report

Last updated: 2026-08-11

## One-page summary

I reproduced and trained the full official 3D Factorized baseline from Dynamic Sensorium / Sensorium 2023. The model covers all five competition sessions and 40,034 neurons, while preserving the three-layer `[32,64,128]` Factorized3D core, full spatiotemporal kernels, behavior channels, Gaussian readouts, cortical grid predictors, and pupil shifters. No reduction was made to the architecture, resolution, temporal window, or data scope.

Formal training used seed 42. As specified by the project protocol, training was stopped after validation at epoch 103, and the best weights from epoch 97 were recovered from the complete epoch checkpoints. The frozen `best.pt` was independently evaluated in a fresh process on all 293 full-length oracle trials, yielding:

```text
best epoch                                  97
training-time best closure correlation      0.1980032772
full-sequence oracle correlation             0.1966732591
official seed-42 hidden final_test_main      0.1887
```

`0.1966732591` and `0.1887` come from different splits: the former is the local oracle score, whereas the latter is the official hidden `final_test_main` score. They therefore cannot be directly subtracted, nor can the local result be used to claim that the hidden benchmark has been reproduced. In terms of local model health, architecture, checkpoint reloading, full-sequence prediction, temporal alignment, and oracle response performance all pass. Formal hidden-test reproduction remains pending server evaluation.

## 1. Official references and reproduction scope

This model was built directly from the Dynamic Sensorium 2023 starter repository, benchmark training notebook, official trainer/model/evaluation implementation, and competition retrospective, rather than being reimplemented from secondary descriptions. The official paper describes the Factorized baseline as using a 3D factorized convolutional core and Gaussian readout, trained by accumulating snippets from all five animals, with 8 snippets per session and 80 frames per snippet, giving an effective batch size of 40. See the [Dynamic Sensorium 2023 starter kit](https://github.com/ecker-lab/sensorium_2023) and the [NeurIPS 2024 competition retrospective](https://proceedings.neurips.cc/paper_files/paper/2024/file/d758d7c0a88d741c8ca4637579c9df87-Paper-Datasets_and_Benchmarks_Track.pdf).

Locally locked source versions:

```text
Sensorium 2023 commit       0e02656220e84a50f3be1b92d6f66c2f9ccd51ef
neuralpredictors commit     efdda679596517fad95d71f36d0385d7450dd207
historical NP pin retained  43faededa2d2e76bb904f38a49b9d8b81d287a0a
```

The historical `neuralpredictors` pin contains a channel-wiring issue for the variable-width `[32,64,128]` core. Training therefore used the corrected upstream commit. This fix does not alter any locked scientific hyperparameters. Both source trees are retained for auditability.

## 2. Data usage and data structure

### 2.1 Data location, scope, and integrity

```text
physical data root
data/sensorium_all_2023

project junction
data/sensorium_all_2023
```

All ten Dynamic Sensorium sessions and their verified archives are retained. The formal baseline uses only the five competition sessions specified by the official configuration; the other five OOD sessions were not deleted. The MD5 hashes of all five formal archives match the official values.

| Session | Train | Oracle | Neurons |
|---|---:|---:|---:|
| `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20` | 348 | 58 | 7,863 |
| `dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 329 | 56 | 7,908 |
| `dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 60 | 8,202 |
| `dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 359 | 60 | 7,939 |
| `dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 59 | 8,122 |
| **Total** | **1,744** | **293** | **40,034** |

### 2.2 File tree and raw trials

Each session uses the official `MovieFileTreeDataset` structure:

```text
data/{videos,responses,behavior,pupil_center}/<trial>.npy
meta/trials/*.npy
meta/neurons/*.npy
meta/statistics/{videos,responses,behavior,pupil_center}/...
```

Using session 29515 as an example, a raw trial has the following shapes:

```text
videos       [36,64,324]
responses    [7863,324]
behavior     [2,324]
pupil_center [2,324]
```

The official `CutVideos` transform extracts the common valid temporal interval shared by all four modalities, producing 300 aligned frames. Calcium signals were originally acquired at approximately 8 Hz and were resampled by the official dataset to 30 Hz; this is explicitly accounted for in the downstream GPFA design. Dynamic Sensorium includes video, neural responses, running/pupil measurements, and related behavioral variables. Detailed acquisition and dataset information is provided in the [competition retrospective](https://pmc.ncbi.nlm.nih.gov/articles/PMC11261979/).

### 2.3 Training tensor contract

A contiguous 80-frame snippet is randomly sampled from each full trial:

```text
model input             [B,3,80,36,64]
neural response target  [B,N_session,80]
behavior                [B,2,80]
pupil center            [B,2,80]
sampling grid           30 Hz
```

One grayscale video channel and two behavior traces form the three core input channels. The behavior traces are spatially broadcast over the 36×64 image grid. Pupil center is supplied separately to the shifter. The official `NeuroNormalizer` uses stored `stats_source="all"` statistics for video, responses, behavior, and pupil center.

## 3. Model architecture

### 3.1 Factorized3D core

Each layer factorizes a 3D convolution into a spatial convolution followed by a temporal convolution:

| Block | Spatial convolution | Temporal convolution | Output channels | BN / activation |
|---|---|---|---:|---|
| 0 | `3→32`, kernel `(1,11,11)` | `32→32`, kernel `(11,1,1)` | 32 | BatchNorm3d momentum 0.7 + ELU |
| 1 | `32→64`, kernel `(1,5,5)` | `64→64`, kernel `(5,1,1)` | 64 | BatchNorm3d momentum 0.7 + ELU |
| 2 | `64→128`, kernel `(1,5,5)` | `128→128`, kernel `(5,1,1)` | 128 | BatchNorm3d momentum 0.7 + ELU |

All convolutions use stride 1 and padding 0. Effective temporal kernels `11,5,5` reduce the sequence by a total of 18 frames; spatial dimensions shrink from 36×64 to 18×46.

```text
core input   [B,3,80,36,64]
core output  [B,128,62,18,46]
```

`[B,128,62,18,46]` is a feature tensor, not the neuron-response prediction. Only after the session-specific readout does the model produce:

```text
model response output [B,62,N_session]
```

Core regularization:

```text
first spatial convolution   10.0 × LaplaceL2norm
first temporal convolution  0.01 × DepthLaplaceL21d
```

### 3.2 Gaussian readout

The five sessions share the core, but each session has its own `FullGaussian2d` readout, producing 7,863, 7,908, 8,202, 7,939, and 8,122 neurons, respectively.

```text
gauss_type                 full
bias                       true
init_mu_range              0.2
init_sigma                 1.0
gamma_readout              0
share_features / grid      false
cortical grid predictor    2 -> 30 -> 2 -> tanh
```

The readout uses neuron cortical coordinates to predict or constrain the initial sampling location, then performs Gaussian spatial sampling from the `[128,18,46]` feature map.

### 3.3 Pupil shifter and output nonlinearity

Each session has an independent MLP shifter:

```text
normalized pupil center 2-D
2 -> 5 -> 5 -> 2
tanh nonlinearities
gamma_shifter = 0
```

The output nonlinearity is:

```text
ELU(x) + 1
```

The model does not contain a GRU; temporal dependence is learned entirely by the temporal kernels in the Factorized3D core.

### 3.4 Parameter count

| Component | Trainable parameters |
|---|---:|
| Factorized3D core | 382,176 |
| Five FullGaussian2d readouts | 5,325,282 |
| Five MLP shifters | 285 |
| **Total** | **5,707,743** |

The architecture audit locks channels, layers, kernels, input/output shapes, the session/neuron inventory, and parameter counts. No Factorized-Lite or single-mouse substitute was used.

## 4. Temporal windows, training targets, and full-trial evaluation

### 4.1 80-frame training window

After receiving 80 input frames, the model outputs 62 frames. The official trainer does not use the first 62 response frames; instead, it automatically selects the final 62 target frames according to the model output length:

```text
input source frames        0–79
raw model output indices   0–61
aligned response frames    18–79
```

Thus, the first temporal-convolution output is aligned with original frame 18. The training loss covers 62 response frames.

### 4.2 Validation within an 80-frame window

The official metric first discards response frames 0–49, leaving 30 frames, and then selects the final 30 frames from the model's 62 outputs. Both correspond to original frames 50–79.

### 4.3 Full 300-frame evaluation

Final evaluation does not repeatedly process 80-frame windows and concatenate the outputs. Instead, the complete trial is passed through the model in a single forward pass:

```text
input                       [B,3,300,36,64]
core feature output         [B,128,282,18,46]
readout response output     [B,282,N]
response after burn-in      [B,N,250], original frames 50–299
prediction after end-crop   last 250 of 282, original frames 50–299
```

The model has a temporal receptive field of 19 frames; the first 18 source frames therefore have no valid output. The official 50-frame burn-in is longer than this structural 18-frame reduction, so every retained final prediction has complete temporal context.

Temporal-alignment audits passed for all 293 oracle trials: source frame indices, response indices, prediction indices, timestamps, trial boundaries, and burn-in handling were consistent. No state is carried across trial boundaries.

## 5. Training and optimization

### 5.1 Formal protocol

| Item | Setting |
|---|---|
| Seed | 42, single-seed reproduction |
| Trainer | `sensorium.training.video_training_loop.standard_trainer` |
| Max epochs | 200 |
| Snippet | 80 consecutive frames, random valid offset |
| Physical batch | 8 / session |
| Sessions per optimizer step | 5 |
| Effective batch | 40 |
| Epoch | 225 session microbatches / 45 optimizer steps |
| Loss | summed Poisson loss |
| Loss scaling | official square-root dataset-size scaling |
| Optimizer | AdamW |
| LR | 0.005 |
| Betas / eps | (0.9,0.999) / 1e-8 |
| Weight decay | 0.01 |
| AMSGrad | false |
| Scheduler | ReduceLROnPlateau, factor 0.3 |
| Patience / tolerance | 5 / 1e-6 absolute |
| Minimum LR | 1e-4 |
| Decay stages | up to 4 |
| Restore best | true |
| Stop metric | oracle single-trial correlation |
| Precision | FP32, AMP disabled |
| Activation checkpointing | disabled |

`LongCycler` aligns loader lengths `44,42,45,45,45` to the longest session. Gradients are accumulated sequentially across all five sessions before each optimizer step.

### 5.2 Source of hyperparameters

Channels, kernels, regularization, readout, shifter, batch size, learning rate, scheduler, and early stopping all come from the official benchmark configuration/tutorial; no architecture search was performed against the local oracle result. Seed 42 is the official tutorial/reference seed. The official retrospective additionally reports stability across 14 single-model seeds.

### 5.3 Local hardware and execution differences

```text
GPU                       RTX 4090 Laptop, 16,376 MiB
GPU power limit observed  up to 125 W profile-dependent
CPU                       Intel Core Ultra 9 185H
RAM                       31.42 GiB
PyTorch                   2.5.1+cu124
CUDA / cuDNN              12.4 / 9.1.0
```

Runtime benchmark:

```text
seconds/session microbatch   2.8069
seconds/optimizer step       14.0343
estimated epoch              631.55 s excluding validation
peak allocated VRAM          7.81 GiB
nvidia-smi peak memory       15,922 MiB
```

To prevent the five-session raw-array cache from exceeding 31 GB of RAM, the file cache was disabled. The same `.npy` files are still loaded on demand and processed with the same transforms. Model architecture, data scope, effective batch, loss, and numerical precision are unchanged.

## 6. Training progress, best model, and validation results

### 6.1 Training stop and checkpoint recovery

By project decision, training was stopped after validation at epoch 103; epoch 104 did not produce a complete checkpoint. The official early-stopping routine had not yet naturally terminated, so this run cannot be described as having completed the full official schedule to natural stopping.

```text
best epoch                              97
best early-stopping closure score       0.1980032772
best post-epoch validation score        0.1977536529
last complete epoch                     103
raw complete epoch checkpoints          103
partial epoch 104 state used            no
```

Both `best.pt` and `last.pt` were reconstructed from complete raw states; the checkpoint kind of `best.pt` is `official_early_stopping_best`.

### 6.2 Independent full-sequence oracle evaluation

In a fresh process, the model was rebuilt, `best.pt` was loaded strictly, and the official `get_correlations` function was run on complete 300-frame oracle trials with batch size 1:

| Session | Correlation |
|---|---:|
| `dynamic29515-10-12` | 0.1855064481 |
| `dynamic29623-4-9` | 0.2033159286 |
| `dynamic29647-19-8` | 0.1925127357 |
| `dynamic29712-5-9` | 0.2071320564 |
| `dynamic29755-2-8` | 0.1949946880 |
| **Neuron-weighted all-session mean** | **0.1966732591** |

This result does not depend on temporary model state from the training process and is consistent with the best validation level observed during training.

### 6.3 Official target scores and correct interpretation

The Dynamic Sensorium retrospective reports:

```text
Factorized baseline, Table 1 hidden main single-trial       0.164
Factorized baseline, Table 1 hidden main trial-average      0.321
seed-42 hidden final_test_main single-trial                 0.1887
seed-42 hidden final_test_main trial-average                0.3569
14 single-model seeds, single-trial mean ± SD               0.1828 ± 0.0094
14-model ensemble hidden main single-trial                  0.197
```

Official table source: [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/d758d7c0a88d741c8ca4637579c9df87-Paper-Datasets_and_Benchmarks_Track.pdf).

The local `0.1966732591` is an oracle score, whereas the official `0.1887` is from hidden `final_test_main`. Because they use different splits and stimulus trials, I do not compute their difference. A valid hidden final-test submission has been generated but has not yet been uploaded; formal hidden benchmark status therefore remains pending.

### 6.4 Continuous prediction export

Continuous outputs have been exported for all 293 oracle trials; each file retains:

```text
predicted_response [250,N]
ground_truth_response [250,N]
timestamps [250]
valid_mask [250]
neuron_ids [N]
session_id
checkpoint and architecture metadata
```

All arrays preserve the temporal order of original frames 50–299. The total export size is approximately 4.29 GB. The official final-test submission parquet has also been generated, with SHA-256:

```text
646ebbd5d164a39275897fe47e085662c22830c93bb9780e96c743a90266ee9e
```

## 7. Comparability with the Static model

The two models share:

- the same five sessions and all 40,034 neurons;
- the same training/oracle tiers;
- the same 36×64 input resolution;
- the same random 80-frame training snippets;
- the same video/behavior/pupil transforms;
- the same family of session-specific Gaussian readouts and pupil shifters;
- the same Poisson loss, AdamW optimizer, effective batch size of 40, scheduler, and stopping metric;
- the same full-trial evaluation interval, original frames 50–299;
- the same neuron-weighted single-trial correlation implementation.

Primary architectural differences:

| Feature | Static | Dynamic |
|---|---|---|
| Core | 4-layer 2D CNN | 3-block Factorized3D |
| Temporal access | current frame | 19-frame temporal receptive field |
| Core channels | 64 | 32→64→128 |
| Total parameters | 2,814,015 | 5,707,743 |
| Oracle correlation | 0.1644077748 | 0.1966732591 |

The two models can therefore be compared at both the response level and the brain-based trajectory level, but the difference cannot be attributed entirely to temporal modeling because model capacity also differs and must be reported as a limitation.

## 8. Reproducibility files

```text
configuration
experiments/01_baselines/configs/phase1A_dynamic_official.yaml

implementation
experiments/01_baselines/src/trajectory_eval/official_dynamic.py

architecture audit
experiments/01_baselines/records/dynamic/architecture_audit.json

training logs and environment
experiments/01_baselines/records/dynamic/

checkpoint
models/official_dynamic/best.pt

independent evaluation
experiments/01_baselines/records/dynamic/official_evaluation.json

temporal alignment
experiments/01_baselines/records/dynamic/temporal_alignment.json

continuous predictions
regenerated locally by the evaluation command; full tensors are excluded from Git because of size
```

## 9. Final assessment

I consider the current Dynamic checkpoint to be a suitable full-width benchmark for the controlled model comparison: the full architecture, complete five-session data scope, oracle performance, independent reload evaluation, and temporal alignment have all been confirmed.

The rigorous status should still be stated as:

```text
Local oracle reproduction: PASS
Architecture/data/alignment: PASS
Official hidden final_test_main reproduction: PENDING
Training to natural official early-stop termination: NOT COMPLETED BY PROJECT DECISION
```

Downstream trajectory evaluation should always report response correlation alongside trajectory metrics. GPFA trajectory metrics should not replace the original response-prediction benchmark.
