# Data and Reproducibility Guide

This guide describes how to obtain the external data, reproduce the repository layout, install the four experimental phases, verify the public artifacts, and load the released model weights. Raw Sensorium data are not redistributed by this repository.

## Official data source

Dynamic Sensorium 2023 data are distributed by the competition organizers:

- [Official Dynamic Sensorium 2023 starter repository](https://github.com/ecker-lab/sensorium_2023)
- [Official Dynamic Sensorium 2023 GIN download page](https://gin.g-node.org/pollytur/sensorium_2023_dataset)

The starter repository links directly to the GIN record and documents the official file-tree loader. Obtain the data from the official source, review the current access and reuse terms there, and download the required archives. Do not commit downloaded data to this repository.

This project uses the five main competition sessions listed below. Additional Sensorium sessions may remain in the same data root, but they are not used in the reported five-session comparison.

| Session ID | Training trials | Oracle trials | Neurons |
|---|---:|---:|---:|
| `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20` | 348 | 58 | 7,863 |
| `dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 329 | 56 | 7,908 |
| `dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 60 | 8,202 |
| `dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20` | 359 | 60 | 7,939 |
| `dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20` | 354 | 59 | 8,122 |
| **Total** | **1,744** | **293** | **40,034** |

The one-session GPFA and detailed model-comparison pilot uses the first session in this table.

## Required data layout

Extract each session so that the repository-level data root directly contains the session directories. Do not leave an extra archive-name directory between `sensorium_all_2023` and a session ID.

```text
neural-trajectory-evaluation/
`-- data/
    `-- sensorium_all_2023/
        |-- dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/
        |   |-- config.json
        |   |-- data/
        |   |   |-- videos/<trial>.npy
        |   |   |-- responses/<trial>.npy
        |   |   |-- behavior/<trial>.npy
        |   |   `-- pupil_center/<trial>.npy
        |   `-- meta/
        |       |-- trials/{tiers,trial_idx,...}.npy
        |       |-- neurons/{unit_ids,...}.npy
        |       `-- statistics/{videos,responses,behavior,pupil_center}/...
        |-- dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20/
        |-- dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20/
        |-- dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20/
        `-- dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20/
```

All phase configurations resolve this same repository-level directory. The training and audit code reads it in place and does not move, rename, or rewrite the official files. The entire `data/` payload except `.gitkeep` is excluded by `.gitignore`.

## Software environment

Use **Python 3.11** for the complete pipeline. It satisfies all four phase constraints and matches the released baseline environment. The reference encoding-model runs used:

```text
Python              3.11
PyTorch             2.5.1
torchvision         0.20.1
CUDA / cuDNN        12.4 / 9.1.0
GPU                 NVIDIA RTX 4090 Laptop, 16 GB
host memory         approximately 31 GB
```

The exact NVIDIA driver may differ, but it must support the CUDA runtime used by the installed PyTorch build.

- **CUDA is required** for Phase 1 and Phase 3 training/evaluation and for Phase 4 checkpoint inference (`predict` and `extended-predict`).
- **CPU execution is sufficient** for Phase 2 GPFA/reliability analysis and for Phase 4 table analysis after aligned prediction tensors have been generated.
- A **16 GB GPU is recommended** for the locked five-session training configuration. Reference peak PyTorch allocation was 7.81-8.33 GiB, while observed total device occupancy approached 16 GB.
- Approximately **32 GB system RAM is recommended**. The official file-tree cache is disabled in the locked configuration because caching all five sessions exceeded the reference host memory; this changes I/O caching, not samples or transforms.
- Reserve additional disk space for the official archives, extracted sessions, environments, and regenerated outputs. Dataset size is controlled by the official distribution and is not duplicated by this repository.

## Installation order

Model files use Git LFS. Install Git and Git LFS before cloning or pulling weights.

```text
git clone https://github.com/124585083/neural-trajectory-evaluation.git
cd neural-trajectory-evaluation
git lfs install
git lfs pull
```

Use two Python 3.11 environments. This separation is intentional: the official Phase 1 encoding stack pins `pandas==2.0.0`, whereas the Phase 2 analysis package declares `pandas>=2.1`. Installing every phase with dependency resolution into one environment would therefore produce a version conflict or silently alter the official baseline stack.

### Environment A: encoding and checkpoint inference

On Windows PowerShell:

```text
py -3.11 -m venv .venv-encoding
.\.venv-encoding\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./experiments/01_baselines
python -m pip install -e ./experiments/03_parameter_matching
python -m pip install "scipy>=1.12,<2"
python -m pip install -e ./experiments/02_gpfa_reliability --no-deps
python -m pip install -e ./experiments/04_model_comparison --no-deps
```

On Linux or macOS:

```text
python3.11 -m venv .venv-encoding
source .venv-encoding/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./experiments/01_baselines
python -m pip install -e ./experiments/03_parameter_matching
python -m pip install "scipy>=1.12,<2"
python -m pip install -e ./experiments/02_gpfa_reliability --no-deps
python -m pip install -e ./experiments/04_model_comparison --no-deps
```

The two `--no-deps` installs expose the Phase 2 metadata and Phase 4 command packages without replacing the official Phase 1 dependency pins. Use this environment for Phase 1/3 training and evaluation and for Phase 4 `lock`, `predict`, and `extended-predict`. The `lock` command imports only the Phase 2 data-contract code; full GPFA fitting belongs in Environment B.

### Environment B: GPFA and statistical analysis

Create a second Python 3.11 environment from the repository root:

```text
python -m venv .venv-analysis
```

Activate `.venv-analysis` using the platform-specific command above, then install:

```text
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "./experiments/02_gpfa_reliability[test]"
python -m pip install -e ./experiments/04_model_comparison
python -m pip install "pytest>=8,<10"
```

Use this environment for Phase 2 and for the Phase 4 `traditional`, `gpfa`, `sensitivity`, `gpfa-evaluate`, and `questions` commands after Environment A has generated the locked protocol and aligned prediction tensors. Both environments write to the same ignored `outputs/pilot/` directories.

Phase 1 pins the official scientific stack, including the project-tested Sensorium 2023 and `neuralpredictors` source revisions. Its installation therefore requires network access and Git unless those dependencies are already available from a local package cache.

## Minimal verification

### Code and analysis contracts without raw data

Run the public unit and contract tests from the repository root. In Environment A:

```text
python -m pip install "pytest>=8,<10"
python -m pytest experiments/01_baselines/tests -q
python -m pytest experiments/03_parameter_matching/tests -q
```

In Environment B:

```text
python -m pytest experiments/02_gpfa_reliability/tests -q
python -m pytest experiments/04_model_comparison/tests -q
```

The curated public tree contains 19 contract and analysis tests. Existing results can be inspected without downloading the raw data:

```text
python -c "import json; p=json.load(open('results/tables/04_model_comparison/q1_q6_answers.json')); print(p['status'])"
```

Expected output:

```text
q1_q6_analysis_complete
```

### Data-aware audit commands

After placing the five sessions under `data/sensorium_all_2023/`, run these minimal audits in phase order:

```text
cd experiments/01_baselines
python -m trajectory_eval.official_dynamic --config configs/phase1A_dynamic_official.yaml audit
python -m trajectory_eval.static_dynamic --config configs/static_dynamic_sensorium2023.yaml audit

cd ../02_gpfa_reliability
python -m trajectory_reliability.cli inspect --config configs/pilot.yaml

cd ../03_parameter_matching
python -m trajectory_param_match.experiment --config configs/dynamic_parameter_matched.yaml audit

cd ../04_model_comparison
python -m trajectory_model_eval.cli lock --config configs/pilot.yaml
```

These checks validate session presence, trial tiers, neuron counts, architecture contracts, parameter matching, and locked analysis paths before expensive inference or training begins.

## Weight integrity and loading

Released weights and frozen GPFA objects are tracked with Git LFS. Their sizes and SHA-256 digests are recorded in [`../results/manifests/model_files.csv`](../results/manifests/model_files.csv) and [`../results/manifests/model_checksums.csv`](../results/manifests/model_checksums.csv). If a checkpoint is only a small text pointer, run `git lfs pull` before loading it.

The neural-network files contain public state dictionaries rather than machine-specific optimizer or trainer state. They can be inspected safely on CPU:

```python
from pathlib import Path
import torch

checkpoint = Path("models/parameter_matched_dynamic/best.pt")
state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

print(len(state_dict), "tensors")
print(next(iter(state_dict)))
```

To run inference, reconstruct the exact architecture and dataloader metadata before loading the state dictionary. This example rebuilds the parameter-matched Dynamic model from the repository root:

```python
from pathlib import Path
import torch

from trajectory_param_match.experiment import (
    build_official_model,
    load_config,
    make_official_loaders,
)

config = load_config(
    "experiments/03_parameter_matching/configs/dynamic_parameter_matched.yaml"
)
dataloaders = make_official_loaders(
    config,
    cuda=False,
    batch_size=1,
    to_cut=False,
    offset=0,
)
model = build_official_model(config, dataloaders)
state_dict = torch.load(
    Path("models/parameter_matched_dynamic/best.pt"),
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict, strict=True)
model.eval()
```

The model builder needs the authorized dataset metadata because the five session-specific readouts depend on neuron counts and session keys. For standard evaluation, prefer the phase commands rather than duplicating preprocessing or temporal alignment in a new script.

Python pickle files can execute code during deserialization. Load the released GPFA `.pkl` files only from a trusted checkout after verifying their recorded SHA-256 digests.

## Reproduction boundaries

- Hidden `final_test_main` labels are unavailable locally. The official Dynamic hidden score cannot be numerically re-evaluated without the external competition service.
- The five-session result is the response benchmark. Detailed output-space RSA/CKA, GPFA, response-matching, and temporal-ablation analyses currently use one session, 512 neurons, and six movie conditions.
- Raw data and full prediction tensors are intentionally excluded. Checkpoints, frozen GPFA objects, configurations, audit records, and compact result tables are included.
- See the [experiment matrix](EXPERIMENT_MATRIX.md) for the exact completion status of every model and evaluation.

## Data citation

Use of Dynamic Sensorium data should cite the dataset/competition papers independently of this software repository:

1. Turishcheva et al. (2024), [The Dynamic Sensorium competition for predicting large-scale mouse visual cortex activity from videos](https://arxiv.org/abs/2305.19654).
2. Wang et al. (2024), [Retrospective for the Dynamic Sensorium Competition](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html).

Licensing and attribution details are separated in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
