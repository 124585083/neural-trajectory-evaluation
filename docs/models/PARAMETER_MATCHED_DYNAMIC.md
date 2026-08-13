# Dynamic–Static parameter-matching design

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

## Parameter match

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

The matched Dynamic model uses exactly the Static-on-Dynamic model's five
sessions, train/oracle trial IDs, neuron IDs, 36x64 resolution, 80-frame random
snippets, behavior and pupil inputs, normalization source, effective batch 40,
Poisson objective, AdamW settings, scheduler, seed, and frames 50--299 evaluation.

It retains the full Dynamic baseline's Factorized3D kernels, activations,
normalization, regularizers, Gaussian-readout hyperparameters, shifter, and output
nonlinearity. It is therefore a capacity-matched Dynamic control, not an official
Sensorium baseline reproduction.

## Acceptance thresholds

```text
preferred total-parameter difference  <= 5%
absolute maximum                       <= 10%
selected difference                     1.707%
```

