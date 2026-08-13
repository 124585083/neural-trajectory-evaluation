# Third-Party Notices and Attribution

This repository contains original project code, documentation, configurations, trained artifacts, and compact result tables. It also depends on external data and software. The repository-level [MIT License](LICENSE) does not replace or broaden any third-party license, dataset access condition, model-use condition, or citation requirement.

## Dynamic Sensorium 2023

**Role in this project:** natural-movie stimuli, mouse V1 population responses, behavioral covariates, the official video loader/trainer/evaluator, and the full Factorized3D Dynamic baseline specification.

- Upstream repository: [ecker-lab/sensorium_2023](https://github.com/ecker-lab/sensorium_2023)
- Source revision pinned by Phase 1: `0e02656220e84a50f3be1b92d6f66c2f9ccd51ef`
- Official data record: [GIN — sensorium_2023_dataset](https://gin.g-node.org/pollytur/sensorium_2023_dataset)
- Dataset white paper: [Turishcheva et al., 2024](https://arxiv.org/abs/2305.19654)
- Competition retrospective: [Wang et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html)

Raw Dynamic Sensorium data are not included in this repository. Users must obtain them from the official source and comply with the terms displayed there. As of 14 August 2026, the upstream `sensorium_2023` repository does not expose a root-level `LICENSE` file. This project therefore treats it as an external dependency and scientific reference; it does not claim that its own MIT license grants rights to upstream Sensorium 2023 code or data.

The released checkpoints were trained on Dynamic Sensorium 2023. Their inclusion does not redistribute raw recordings and does not waive any conditions that the data providers may apply to trained derivatives. Users are responsible for confirming that their intended reuse complies with the current official terms.

## Sensorium 2022

**Role in this project:** source of the Static Sensorium/Sensorium+ CNN architecture that is retrained frame by frame on Dynamic Sensorium 2023.

- Upstream repository: [sinzlab/sensorium](https://github.com/sinzlab/sensorium)
- Static source revision recorded by Phase 1: `c433fed25f234724fd9adf0cef3c260a2068b1fa`
- Upstream license: MIT
- Upstream copyright notice: Copyright (c) 2024 Sensorium GitHub Contributors
- Local copy of the upstream notice: [`third_party/licenses/SENSORIUM-2022-MIT.txt`](third_party/licenses/SENSORIUM-2022-MIT.txt)
- Competition retrospective: [Willeke et al., 2023](https://proceedings.mlr.press/v220/willeke23a.html)

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
