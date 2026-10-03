# Third-Party Notices and Attribution

## Mental-Pong

The Mental-Pong study uses the data and analysis framework accompanying Rajalingham, Sohn, and Jazayeri (2025), [Dynamic tracking of objects in the macaque dorsomedial frontal cortex](https://www.nature.com/articles/s41467-024-54688-y), *Nature Communications* 16, 346, published 2 January 2025. Cite this paper and the separate 2024 [Zenodo release 13952210](https://doi.org/10.5281/zenodo.13952210), version v1, when using these data. The earlier task and behavioral study is Rajalingham, Piccato, and Jazayeri (2022), DOI `10.1038/s41467-022-33581-6`. The [reference guide](docs/REFERENCES.md) gives full source identities.

The local audit inspected [jazlab/MentalPong](https://github.com/jazlab/MentalPong) at commit `b976255be73140c759d8f8db0fd8ff551a4e2d73`. Its code carries the MIT License, copyright (c) 2024 JazLab. The [unmodified license](third_party/licenses/MentalPong-MIT.txt) is preserved here. Source-code licensing does not replace the dataset provider's terms.

The integrated module contains project analysis code, English report editions, and compact derived results. Raw Mental-Pong data and large saved representations, readouts, and null arrays remain external. The [artifact registry](experiments/05_mental_pong/integration/artifact_registry.csv) and [official source audit](experiments/05_mental_pong/integration/official_source_audit.json) retain provenance. No affiliation with or endorsement by the original authors is implied.

The locally preserved provider JSON for this specific Zenodo record specifies **CC BY-NC-ND 4.0**. The record's public landing page was checked on 3 October 2026. The API refresh recorded in the audit returned HTTP 403, so the license assertion is tied to the saved metadata and its checksum in the [data-rights audit](experiments/05_mental_pong/integration/revision_20261003/data_rights_audit.json). The downloaded archive contains no separate license granting broader data rights. Neither the GitHub code's MIT license nor Zenodo's platform default establishes permission for derived data redistribution.

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

The four fitted encoding checkpoints, two GPFA objects, and two preprocessing arrays are withheld pending a documented redistribution basis. Their original hashes and historical paths remain in the [model inventory](results/manifests/model_files.csv). This conservative availability decision does not determine whether trained parameters are legally adapted material. Existing Git/LFS history is unchanged; future inference requires appropriately obtained external copies.

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

PyTorch, torchvision, NumPy, SciPy, pandas, scikit-learn, PyYAML, matplotlib, nnfabrik, DataJoint, and other packages are installed as external dependencies and remain governed by their respective licenses. They are not relicensed by this repository. Exact direct dependencies are declared in each phase's `pyproject.toml`.

## No endorsement

Use of the Sensorium names and upstream project names is solely for scientific attribution and reproducibility. It does not imply endorsement of this project by the dataset authors, competition organizers, Sinz Lab, Ecker Lab, or other upstream contributors.
