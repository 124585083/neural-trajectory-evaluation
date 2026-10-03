# Dynamic–Static Total-Parameter-Matching Design

## Selected model

```text
Full Dynamic channels       [32,64,128]
Matched Dynamic channels    [16,32,64]
Width multiplier            0.5
Spatial kernels             [11,5,5], unchanged
Temporal kernels            [11,5,5], unchanged
Core depth                  3, unchanged
Temporal reduction          18 frames, unchanged
Temporal receptive field    19 frames, unchanged
```

## Total-parameter match

| Component | Static | Matched Dynamic | Difference |
|---|---:|---:|---:|
| Core | 50,624 | 98,672 | +48,048 |
| Readout | 2,763,106 | 2,763,106 | 0 |
| Shifter | 285 | 285 | 0 |
| **Total** | **2,814,015** | **2,862,063** | **+48,048 (+1.707%)** |

The half-width model is selected instead of numerically closer irregular channel
triples because it preserves the original 1:2:4 widening rule and applies one
predeclared width multiplier to every core layer. The final 64-channel feature
map also makes the five-session Gaussian readout parameter count exactly equal
to the Static model.

## Controlled variables

The total-parameter-matched Dynamic model uses exactly the Static-on-Dynamic model's five
sessions, train/oracle trial IDs, neuron IDs, 36x64 resolution, 80-frame random
snippets, behavior and pupil inputs, normalization source, effective batch 40,
Poisson objective, AdamW settings, scheduler, seed, and frames 50--299 evaluation.

It retains the full Dynamic baseline's Factorized3D kernels, activations,
normalization, regularizers, Gaussian-readout hyperparameters, shifter, and output
nonlinearity. This defines the **total-parameter-matched Dynamic control** used
in the project. The original full-width baseline is documented separately.

The match applies to total trainable count. Core structure remains different: Static uses a
four-layer 2D frame-wise core with 50,624 parameters; Dynamic uses a three-stage
Factorized3D core with 98,672 parameters and learned temporal convolutions. The
readouts each contain 2,763,106 parameters and dominate both totals; their learned weights are separate. The comparison therefore
does not equate core parameterization, convolutional operations, inductive bias,
effective computation, or temporal access.

## Acceptance thresholds

```text
preferred total-parameter difference  <= 5%
absolute maximum                       <= 10%
selected difference                     1.707%
```

## Selection and evaluation data

The training entry point uses the official train tier for gradients and the oracle correlation closure for scheduling, stopping decisions and checkpoint selection. Complete-sequence evaluation reloads the selected weights and scores the same oracle tier. It verifies saved-checkpoint execution but does not provide a new independent test set. Hidden server scoring was unavailable for this local comparison. The [Methods data-use table](../../METHODS.md#data-use-and-evaluation-independence) distinguishes these uses from train-only GPFA fitting and perturbation-half selection. The [reference guide](../../REFERENCES.md) credits the original benchmark and reused software.
