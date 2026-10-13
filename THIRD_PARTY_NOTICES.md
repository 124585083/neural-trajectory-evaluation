# Third-Party Notices and Attribution

## Mental-Pong

The Mental-Pong study uses the data and analysis framework accompanying Rajalingham, Sohn, and Jazayeri (2025), [Dynamic tracking of objects in the macaque dorsomedial frontal cortex](https://www.nature.com/articles/s41467-024-54688-y), *Nature Communications* 16, 346, published 2 January 2025. Cite this paper and the separate 2024 [Zenodo release 13952210](https://doi.org/10.5281/zenodo.13952210), version v1, when using these data. The earlier task and behavioral study is Rajalingham, Piccato, and Jazayeri (2022), DOI `10.1038/s41467-022-33581-6`. The [Mental-Pong source brief](results/mental_pong/MODELS_AND_DATA.md) gives full source identities.

The local audit inspected [jazlab/MentalPong](https://github.com/jazlab/MentalPong) at commit `b976255be73140c759d8f8db0fd8ff551a4e2d73`. Its code carries the MIT License, copyright (c) 2024 JazLab. The [unmodified license](third_party/licenses/MentalPong-MIT.txt) is preserved here. Source-code licensing does not replace the dataset provider's terms.

The public module contains project analysis code, English source briefs, and compact derived results. Raw Mental-Pong data remain external. Exact model, preprocessing, latent, readout and control copies are installed in the local project's ignored `models/mental_pong/current/` directory; they are not part of the public module. The [artifact registry](code/mental_pong/integration/artifact_registry.csv) and [official source audit](code/mental_pong/integration/official_source_audit.json) retain provenance. No affiliation with or endorsement by the original authors is implied.

The locally preserved provider JSON for this specific Zenodo record specifies **CC BY-NC-ND 4.0**. The record's public landing page was checked on 3 October 2026. The API refresh recorded in the audit returned HTTP 403, so the license assertion is tied to the saved metadata and its checksum in the [data-rights audit](code/mental_pong/integration/revision_20261003/data_rights_audit.json). The downloaded archive contains no separate license granting broader data rights. Neither the GitHub code's MIT license nor Zenodo's platform default establishes permission for derived data redistribution.

On 12 October 2026, the original archive download returned HTTP 200 to a HEAD request with the expected 6,145,726,435-byte length. The record metadata endpoint remained inaccessible. Archive access is therefore verified separately from license-metadata refresh; neither supplies a public route to this study's fitted artifacts. Checked URLs and archive identity are in [the acquisition record](models/acquisition.json).

The [CC BY-NC-ND terms](https://creativecommons.org/licenses/by-nc-nd/4.0/) restrict redistribution of modified material. This project has not resolved their application to the transformed neural and behavioral arrays. Accordingly, the new numerical mini-replay payload is withheld from the publication tree; the exporter, replay source, dependency inventory, and local verification record remain available. The two condition-label NPZ files and their two condition-by-time CSV equivalents are also withheld because their redistribution basis has not been established. Aggregate scores, reports, and source identities remain available; their presence does not grant rights to the underlying dataset or establish a blanket permission for numerical derivatives. This is a documented publication limitation, not a claim that every numerical derivative has a settled legal classification. Rights-holder clarification would be needed before expanding the public payload.

This repository contains original project code, documentation, configurations, model identity records, and compact result tables. It also depends on external data and software. The repository-level [MIT License](LICENSE) does not replace or broaden any third-party license, dataset access condition, model-use condition, or citation requirement.

## Dynamic Sensorium 2023

**Role in this project:** natural-movie stimuli, mouse V1 population responses, behavioral covariates, the official video loader/trainer/evaluator, and the full Factorized3D Dynamic baseline specification.

- Upstream repository: [ecker-lab/sensorium_2023](https://github.com/ecker-lab/sensorium_2023)
- Source revision pinned by Phase 1: `0e02656220e84a50f3be1b92d6f66c2f9ccd51ef`
- Official data record: [GIN — sensorium_2023_dataset](https://gin.g-node.org/pollytur/sensorium_2023_dataset)
- Benchmark description: [Turishcheva et al., arXiv:2305.19654v2](https://arxiv.org/abs/2305.19654v2), revised 12 July 2024; version 1 was submitted 31 May 2023.
- Competition retrospective: [Turishcheva et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html)

Raw Dynamic Sensorium data are external. Appendix A.8.4–A.8.5 of the [2024 retrospective](https://proceedings.neurips.cc/paper_files/paper/2024/file/d758d7c0a88d741c8ca4637579c9df87-Paper-Datasets_and_Benchmarks_Track.pdf) identifies CC BY-NC-ND 4.0 terms for data and code. The pinned upstream checkout has no root `LICENSE`; that absence does not establish unrestricted use. Upstream code is an external dependency, and this project's MIT license does not broaden its terms.

Historical and current fitted encoding checkpoints, GPFA objects and preprocessing arrays are withheld pending a documented redistribution basis. Their identities and actual availability are distinguished in the [model manifest](models/manifest.json). This conservative availability decision does not determine whether trained parameters are legally adapted material. Existing Git/LFS history is unchanged; inference from a public-only clone requires separately obtained copies. The local installation described below already contains the current parameters.

The provider's [README](https://gin.g-node.org/pollytur/sensorium_2023_dataset/src/master/README.md) and [DataCite metadata](https://gin.g-node.org/pollytur/sensorium_2023_dataset/src/master/datacite.yml) were retrieved successfully on 12 October 2026 and both specify CC BY-NC-ND 4.0. The README also asks users to contact the provider before using the data in their own research and to obtain permission before publishing new analyses. This records the provider's stated condition; it does not independently settle the legal classification of analysis results or trained weights. A separate permission document was not found in the checked materials.

Current replay inputs and the original historical full Dynamic wrapper now have exact local copies under `models/sensorium/`. Those ignored copies resolve local file dependencies but do not establish public download availability. Older weights remain technically accessible in published Git/LFS history and do not replace the current retrained models. The [artifact access guide](models/README.md) separates local replay, original-data acquisition and the unresolved distribution of current fitted payloads.

## Sensorium 2022

**Role in this project:** source of the Static Sensorium/Sensorium+ CNN architecture that is retrained frame by frame on Dynamic Sensorium 2023.

- Upstream repository: [sinzlab/sensorium](https://github.com/sinzlab/sensorium)
- Static source revision recorded by Phase 1: `c433fed25f234724fd9adf0cef3c260a2068b1fa`
- Upstream license: MIT
- Upstream copyright notice: Copyright (c) 2024 Sensorium GitHub Contributors
- Local copy of the upstream notice: [`third_party/licenses/SENSORIUM-2022-MIT.txt`](third_party/licenses/SENSORIUM-2022-MIT.txt)
- Competition retrospective: [Willeke et al., 2022](https://proceedings.mlr.press/v220/willeke23a.html)

The Static-on-Dynamic model in this repository is a project-specific transfer experiment. It is not an official Sensorium 2022 score and not an official Static-on-Dynamic benchmark released by the Sensorium organizers.

## neuralpredictors

**Role in this project:** neural-system-identification components used by the official Sensorium implementations, including loaders, readouts, training utilities, and model support code.

- Upstream repository: [sinzlab/neuralpredictors](https://github.com/sinzlab/neuralpredictors)
- Source revision pinned by Phase 1: `efdda679596517fad95d71f36d0385d7450dd207`
- Upstream license: MIT
- Upstream copyright notice: Copyright (c) 2019 Sinz Lab
- Local copy of the upstream notice: [`third_party/licenses/NEURALPREDICTORS-MIT.txt`](third_party/licenses/NEURALPREDICTORS-MIT.txt)

## Other dependencies

PyTorch, torchvision, NumPy, SciPy, pandas, scikit-learn, PyYAML, matplotlib, nnfabrik, DataJoint, and other packages are installed as external dependencies and remain governed by their respective licenses. They are not relicensed by this repository. Direct dependencies remain in the retained code packages; the project code briefs record checked runtime versions.

## No endorsement

Use of the Sensorium names and upstream project names is solely for scientific attribution and reproducibility. It does not imply endorsement of this project by the dataset authors, competition organizers, Sinz Lab, Ecker Lab, or other upstream contributors.
