# Sensorium: Models and Data

The current study, `retrained_v1`, compares two newly trained neural encoding models: **Static**, which processes frames independently, and **reduced Dynamic**, which combines frames through temporal convolutions. Both predict the same recorded neurons. A Gaussian-process factor analysis (GPFA) model fitted to training neural responses defines the common space used to compare recorded and predicted trajectories. Its parameters remain fixed during evaluation.

This document describes the data, model structures and transformations connecting those comparisons. The earlier full-width Dynamic reproduction is a separate historical run. Detailed analysis settings and findings are linked from the [project overview](../../README.md).

## 1. Dataset and analysis scope

The data come from the [Sensorium 2023 release](https://gin.g-node.org/pollytur/sensorium_2023_dataset): mouse visual-cortex recordings during natural movies, with behavioural channels and pupil measurements. The released neural arrays contain **processed calcium responses, not spike counts**. Calibrated physical units for the response amplitudes and behavioural channels are not specified in the checked run records; the response scales used here are defined in Section 5.

The comparison includes five configured competition sessions. Response-level evaluation uses all 40,034 recorded neurons across these sessions. Detailed trajectory and representational comparisons use one pilot session, `29515-10-12`, with 512 fixed neurons and six repeated movie conditions. Five other sessions appeared in the earlier local inventory but are not evaluated here.

The [benchmark description](https://doi.org/10.48550/arXiv.2305.19654) is the 2024 revision of Turishcheva et al.'s *The Dynamic Sensorium competition for predicting large-scale mouse visual cortex activity from videos*, first submitted in 2023. The [2024 NeurIPS retrospective](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html) is a separate publication. These papers describe the benchmark; this project's numerical results come from its saved run records.

Source-file hashes identify the actual inputs; a separate provider release-version tag is absent from the run records. The [source inventory](evidence/04_model_comparison/unified_v1/split_manifest.json) preserves original trial and metadata identities; the [current split manifest](evidence/04_model_comparison/retrained_v1/split_manifest.json) assigns their roles in `retrained_v1`.

![Current data roles, model inputs and common evaluation support](figures/documentation/01_shared_pathway.png)

*Current data roles, model inputs and common evaluation support. The pupil shifter is separate from the video/behaviour core; Static crops its first 18 outputs to match Dynamic. GPFA is fitted to training neural responses and uses an offline posterior. The released 30 Hz grid does not represent independent neural acquisitions.*

## 2. Observations, array shapes and time alignment

The encoders receive a grayscale movie and two supplied behavioural traces. Each behavioural trace is broadcast over the image, producing three input channels. The two pupil coordinates supply a separate pupil shifter.

| Data or processing stage | Array axes and support |
|---|---|
| Source movie | Height × width × time; 36 × 64 pixels, first 300 frames used |
| Source neural responses | Neuron × time; neuron order is session-specific |
| Source behaviour | 2 × time |
| Source pupil centre | 2 × time |
| Encoder input | Batch × 3 × time × 36 × 64 |
| Encoder output | Batch × (input time − 18) × session neurons |
| Scoring arrays | Trial × 250 × neuron |
| Pilot condition-average responses or predictions | 6 × 250 × 512 |
| Pilot condition-average GPFA trajectories | 6 × 250 × 4 |

Training uses 80-frame snippets, with eight snippets per session. Gradient accumulation over five sessions gives 40 snippets per update cycle. Dynamic's valid temporal convolutions produce 62 outputs from an 80-frame input. Static uses a parameter-free crop to remove its first 18 outputs and match the same target timestamps.

Evaluation processes complete trials on the aligned window shown above, without stitching training snippets. Scoring timestamps run from `50/30` to `299/30` seconds. The historical description reports approximately 8 Hz calcium acquisition before resampling, but raw acquisition metadata was not checked.

Array shapes and transformation formulas are recorded in [normalization provenance](evidence/04_model_comparison/retrained_v1/normalization_provenance.json). The corresponding implementation is in the [comparison package](../../code/sensorium/src/trajectory_model_eval), including `retrained_data.fit_preprocessing`, `RetrainedDataset`, `retrained_predictions.export_predictions`, `select_neurons` and `unified_analysis.condition_means`.

## 3. Training, validation and test data

Training-tier movie groups were assigned to train or validation using seed 42. A movie group has the same role across sessions. The upstream tier called `oracle` contains labelled repeated-movie trials; these supply the current test set.

| Session identifier | Neurons | Train trials | Validation trials | Test trials | Test movie conditions |
|---|---:|---:|---:|---:|---:|
| 29515-10-12 | 7,863 | 282 | 66 | 58 | 6 |
| 29623-4-9 | 7,908 | 252 | 77 | 56 | 6 |
| 29647-19-8 | 8,202 | 281 | 73 | 60 | 6 |
| 29712-5-9 | 7,939 | 287 | 72 | 60 | 6 |
| 29755-2-8 | 8,122 | 293 | 61 | 59 | 6 |
| **Total** | **40,034** | **1,395** | **349** | **293** | **6 per session** |

Full session keys use `dynamic<identifier>-Video-9b4f6a1a067fe51e15306b9628efea20`. Movie counts are within-session counts; the table does not establish 30 different source films. The 1,518 hidden-tier response placeholders are excluded, rather than treated as observed zero activity.

| Data role | Use in the current study |
|---|---|
| **Train** | Encoder parameter fitting, input and response moments, readout activity initialisation, and pilot GPFA fitting |
| **Validation** | Encoder scheduling and checkpoint selection, GPFA initialisation selection, and control-strength calibration |
| **Test** | Frozen model comparison, repeat reliability and null comparisons |

The pilot's six test conditions contain 9, 9, 10, 10, 10 and 10 repeats. Its 512 neurons are selected by sorting stable unit identities by the SHA-256 hash of `42:<unit_id>`. Selection does not use model scores or neural reliability. No test conditions are removed on the basis of test reliability. Counts and identities are retained in the [split summary](evidence/04_model_comparison/retrained_v1/split_summary.csv) and [split manifest](evidence/04_model_comparison/retrained_v1/split_manifest.json).

This is **held-out evaluation after retraining**, not untouched external validation: the test movies were explored in earlier work. Approximate screening of shifted content examined 7,725 candidate development/test pairs without verifying an overlap. Missing source-film identities and the approximate search limit claims about unseen source films.

## 4. Models and training

### Static and reduced Dynamic

Static is a project-built adaptation of the Sensorium 2022 architecture to video data, not an official video benchmark with an identical architecture. The current Dynamic model uses reduced channel widths to bring its total parameter count close to Static's.

| Component | Static | Reduced Dynamic |
|---|---|---|
| Core | Four shared 2D layers, applied frame by frame | Three Factorized3D stages |
| Channels | 64 throughout | 16 → 32 → 64 |
| Spatial operations | First 9 × 9 convolution; then three pointwise → depthwise 7 × 7 → pointwise blocks | Spatial kernels 11 × 11, 5 × 5 and 5 × 5 |
| Temporal operations | No frame mixing in the core; leading crop removes 18 output frames | Temporal kernels of 11, 5 and 5 frames; stride 1, no padding |
| Final spatial feature map | 64 × 28 × 56 per frame | 64 × 18 × 46 per output time |
| Readout | Per-session full Gaussian readout with an anatomical grid predictor | Same readout family |
| Pupil input | Separate pupil shifter | Same shifter family |
| Output | ELU + 1 | ELU + 1 |

The models use the same readout and shifter families with separately learned parameters. Counts from the actual model instances are:

| Component | Static | Reduced Dynamic |
|---|---:|---:|
| Core | 50,624 | 98,672 |
| Readout | 2,763,106 | 2,763,106 |
| Pupil shifter | 285 | 285 |
| **Total** | **2,814,015** | **2,862,063** |

The totals differ by 48,048 parameters, or 1.707% relative to Static. Readout parameters dominate both totals. The different cores and computations leave several possible contributors to a model difference, even with similar total parameter counts.

Construction is defined by `retrained_training.build_encoder` and the retained [model source](../../code/sensorium/src). The [Static construction configuration](../../code/sensorium/records/baselines/static/training_config.yaml), [reduced Dynamic configuration](../../code/sensorium/configs/dynamic_parameter_matched.yaml), [current protocol](../../code/sensorium/configs/retrained_v1.json) and [parameter-count record](evidence/04_model_comparison/retrained_v1_diagnostics/encoder/parameter_counts.json) specify the actual models. Some configuration references retain historical names; the selected checkpoints belong to the new run.

### Encoder fitting and selection

Both encoders were initialised from scratch with seed 42. Training used AdamW, learning rate 0.005, weight decay 0.01, summed Poisson loss and architecture-specific regularisers. The stopping protocol used patience 5, a learning-rate multiplier of 0.3, up to four cycles and a 200-epoch cap. The local resumable implementation follows the pinned official stopping protocol.

| Encoder | Selected epoch | Last complete epoch | Stop reason |
|---|---:|---:|---|
| Static | 59 | 69 | Exhausted patience cycles |
| Reduced Dynamic | 109 | 119 | Exhausted patience cycles |

In this implementation, the validation loader retains the key `oracle`, but it contains only the **new validation trial IDs**. It does not contain the original oracle trials reserved for test. Full training and comparison settings are given in [model-comparison setup](03_model_comparison/SETUP.md).

### Shared GPFA model

The pilot uses four latent dimensions for 512 observed neurons:

$$
y_t = Cz_t + d + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,R).
$$

Here, $y_t$ is the scaled neural response, $z_t$ is the four-dimensional state, $C$ is a 512 × 4 loading matrix, $d$ is a 512-element offset and $R$ is diagonal observation noise. Each latent dimension has an independent radial-basis-function (RBF) temporal prior with its own timescale.

Initial timescales of 0.125, 0.25, 0.5 and 1 second were compared using validation neural likelihood after 12-iteration candidate fits. The selected 0.25-second initialisation was refitted on train alone. The final fit converged in 19 of at most 20 expectation–maximisation (EM) iterations, with tolerance `1e-6` and timescale bounds of 0.1–3 seconds. The selected value is an **initialisation**, not a claim that all learned or biological timescales equal 0.25 seconds. Encoder predictions did not fit or select GPFA.

The exact Gaussian posterior uses the full observation window. Later observations can therefore affect an earlier inferred state: this is **offline inference**. Measurement settings, reliability and reconstruction results belong in [trajectory-measurement setup](02_trajectory_measurement/SETUP.md) and [results](02_trajectory_measurement/RESULTS.md).

## 5. Frozen preprocessing and the evaluation pathway

### Encoder scales

Normalisation moments are estimated separately for each session from its new training trials. Calculation across trials at each original array element preserves time dependence in the stored moments for each neuron or channel.

Let $x$ denote a source array and $\mu_x$ and $\sigma_x$ its stored training mean and standard deviation. Standard deviations use `ddof=0`.

| Field | Fixed transformation |
|---|---|
| Movie and pupil centre | $(x-\mu_x)/\sigma_x$ |
| Behaviour | $x/\sigma_x$ |
| Neural response | $x/\max(\sigma_x,\,0.01\operatorname{mean}(\sigma_x))$ |

For non-response fields, standard deviations at or below float32 machine epsilon are replaced by one. Neural responses are not centred, and small negative source responses are retained. Validation and test apply these fixed transformations without estimating new moments. The stored array shapes and formulas are in [normalization provenance](evidence/04_model_comparison/retrained_v1/normalization_provenance.json).

### Conversion to the common GPFA scale

GPFA uses a separate, time-independent scale for each selected neuron. It is estimated from raw pilot training responses over frames 50–299 using `ddof=1`, with a floor of 1% of the mean standard deviation.

Observed responses and both encoder predictions are converted from the encoder response scale to this common GPFA scale:

```text
GPFA_input[t, n] = encoder_scaled[t, n]
                   / encoder_train_precision[t, n]
                   * GPFA_train_precision[n]
```

The variables named `precision` are the stored multiplicative scaling factors. The division undoes the encoder response scaling; the multiplication applies the GPFA scaling. Both factors are fixed from training data. The encoder factor depends on time and neuron; the GPFA factor depends only on neuron.

### From aligned responses to trajectories

The two branches in the diagram score the same aligned responses: direct response/RSA/CKA comparisons, and within-movie repeat means passed through the common fixed scaling and GPFA.

GPFA observes bin indices `0, 4, …, 248` relative to the 250-bin scoring window: 63 observations. Its posterior is queried at all 250 bins. Neural and model condition means enter this posterior separately, using identical loadings, offsets, noise, priors and neuron mapping. No model-specific rotation or Procrustes alignment is applied.

Derivatives of queried trajectories describe the inferred posterior, not independently observed neural dynamics at 30 Hz.

## 6. Historical full Dynamic reproduction

The earlier full Dynamic run documents implementation and reproduction checks. The current comparisons use a separate reduced Dynamic model.

| Item | Historical full Dynamic |
|---|---|
| Core widths | [32, 64, 128] |
| Total parameters | 5,707,743 |
| Retained checkpoint | Epoch 97 |
| Training stopped | After epoch 103, before natural early stopping |
| Local oracle correlation | 0.196673, over 40,034 neurons |
| Data use | Oracle informed both checkpoint selection and evaluation |
| Normalisation | Supplied statistics with `stats_source="all"` |

The source record lists an official hidden `final_test_main` reference of 0.1887. It concerns a different evaluation population, and no server-scored reproduction is recorded for this local run. The local correlation therefore does not establish reproduction of that hidden-test score. See [model-validation setup](01_model_validation/SETUP.md) and [results](01_model_validation/RESULTS.md).

Earlier oracle-selected Static and reduced Dynamic checkpoints, and two historical GPFA objects, retain separate model hashes. Their scores do not establish the current GPFA's reliability. Withdrawn auxiliary GPFA sensitivity records remain provenance only and are outside the current conclusions.

## 7. Model files, access and reproduction status

The [model manifest](../../models/manifest.json) identifies the checkpoints, GPFA arrays, five session normalisation files, split and neuron mappings, and fixed controls. The current local copies are under `models/sensorium/retrained_v1/`. The historical full Dynamic checkpoint wrapper and 14 associated source records are under `models/sensorium/historical_full_dynamic/`; the wrapper and its state dictionary have separate identities.

| Current artifact | Recorded SHA-256 prefix |
|---|---|
| Static checkpoint | `3ded37a0fac494f4…` |
| Reduced Dynamic checkpoint | `c6eff392ec412494…` |
| Main GPFA | `98de0cef99973c8a…` |

Full digests and sizes are in the manifest. Scale, neuron order and split identities must also match. The manifest includes main and intervention cache metadata, the frozen spatial-control selection, the sampling plan and source diagnostics.

These fitted payloads are stored locally and ignored by Git; a public clone does not contain them. The checked acquisition records do not identify an authorised public download for the fitted payloads. The manifest inventories the files; running the models requires the weight payloads. Locations and access information are documented in the [artifact guide](../../models/README.md) and [acquisition record](../../models/acquisition.json); redistribution conditions are recorded in [Third-Party Notices](../../THIRD_PARTY_NOTICES.md).

### What has been verified

The recorded preflight checked all 43 main, intervention and reversal cache payload hashes. The recommended `run.py` entry checks these required frozen files; it does not train models or recalibrate controls to replace missing inputs.

The [verification record](../../models/verification.json) reports a complete cached replay with access to the original source directories denied. Of 36 regenerated CSVs, 35 matched exactly. The remaining table matched all current scientific scores, support and status fields, differing only in three prior-calculation diagnostic columns. Fifteen latent and reversal arrays matched exactly. Original evidence files were left unchanged.

This verification covers the retained **cached replay**. A full rebuild from provider data remains unverified. A tested single command that starts with provider data and restored frozen models and generates all six retained research goals is still missing. The [code README](../../code/sensorium/README.md) states the supported inputs, commands and outputs.

Other unresolved items are upstream source-film identities, calibrated physical units in the checked records, and public delivery of the fitted payloads. The saved numerical results remain intact; these gaps concern source interpretation and independent reproduction.

## 8. Method sources

The following sources identify the underlying methods. Project-specific validation and implemented settings are documented separately; correspondence with every original-paper setting has not been established. Bibliographic identities follow the project's retained reference guide; implemented settings are specified by the linked project configurations and analysis setups.

| Method or model background | Source |
|---|---|
| GPFA | Yu et al. (2009), *Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity*. **Journal of Neurophysiology**, 102(1), 614–635. [DOI](https://doi.org/10.1152/jn.90941.2008) |
| Representational similarity analysis (RSA) | Kriegeskorte, Mur and Bandettini (2008), *Representational similarity analysis—connecting the branches of systems neuroscience*. **Frontiers in Systems Neuroscience**, 2. [DOI](https://doi.org/10.3389/neuro.06.004.2008) |
| Centered kernel alignment (CKA) | Kornblith et al. (2019), *Similarity of Neural Network Representations Revisited*. **PMLR**, 97. [Publication](https://proceedings.mlr.press/v97/kornblith19a.html) |
| Static architecture background | Willeke et al. (2022), *Retrospective on the SENSORIUM 2022 competition*. **PMLR**, 220, 314–333. [Publication](https://proceedings.mlr.press/v220/willeke23a.html) |
