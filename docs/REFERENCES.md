# References and source identities

These references distinguish the experiments, released data, and software used in this secondary analysis. The year in a DOI suffix is not necessarily the publication year. The repository's own software citation remains in [CITATION.cff](../CITATION.cff); attribution of the present work is described in [AUTHORS](../AUTHORS.md).

## Mental-Pong

- **Neural study:** Rajalingham, R., Sohn, H., and Jazayeri, M. (2025). [Dynamic tracking of objects in the macaque dorsomedial frontal cortex](https://www.nature.com/articles/s41467-024-54688-y). *Nature Communications*, **16**, 346. Published 2 January 2025. DOI: `10.1038/s41467-024-54688-y`. This is the neural recording and decoding paper; the `2024` DOI suffix does not make it a 2024 publication.
- **Data release:** Rajalingham, R., Sohn, H., and Jazayeri, M. (2024). [Rajalingham_Sohn_Jazayeri_MentalPong_NHP_DorsoMedialFrontalCortex](https://zenodo.org/records/13952210), version v1. Zenodo. DOI: `10.5281/zenodo.13952210`. The record displays 18 November 2024 as its publication date. The archive is `MentalPong.zip`, MD5 `382e420c6f053d8ac2bdd4acd2fb02b2`. Its title and historical README date are preserved as source identifiers.
- **Earlier behavioral task and RNN study:** Rajalingham, R., Piccato, A., and Jazayeri, M. (2022). [Recurrent neural networks with explicit representation of dynamic latent variables can mimic behavioral patterns in a physical inference task](https://www.nature.com/articles/s41467-022-33581-6). *Nature Communications*, **13**, 5865. Published 4 October 2022. DOI: `10.1038/s41467-022-33581-6`. This reference supplies task and behavioral-model context; it is separate from the DMFC neural study.
- **Analysis software:** [jazlab/MentalPong](https://github.com/jazlab/MentalPong), audited revision [`b976255be73140c759d8f8db0fd8ff551a4e2d73`](https://github.com/jazlab/MentalPong/tree/b976255be73140c759d8f8db0fd8ff551a4e2d73). The [source audit](../experiments/05_mental_pong/integration/official_source_audit.json) records the actual decoder and released-artifact checks. The Git checkout and the Zenodo archive are distinct sources; their identity is not inferred from a matching project name.

The code checkout has an MIT license. The saved Zenodo provider metadata identifies the dataset license as **CC BY-NC-ND 4.0**. Permission to redistribute a newly transformed numerical replay bundle has not been established. See [Third-Party Notices](../THIRD_PARTY_NOTICES.md) and the [data-rights audit](../experiments/05_mental_pong/integration/revision_20261003/data_rights_audit.json) for the evidence and its limits.

## Dynamic Sensorium and Static Sensorium

- **Dynamic benchmark description:** Turishcheva, P., Fahey, P. G., Hansel, L., Froebe, R., Ponder, K., Vystrčilová, M., Willeke, K. F., Bashiri, M., Wang, E., Ding, Z., Tolias, A. S., Sinz, F. H., and Ecker, A. S. (2024 version). [The Dynamic Sensorium competition for predicting large-scale mouse visual cortex activity from videos](https://arxiv.org/abs/2305.19654v2). arXiv:2305.19654v2. Version 1 was submitted on 31 May 2023; the cited version 2 is dated 12 July 2024. DOI: `10.48550/arXiv.2305.19654`.
- **Dynamic competition retrospective:** Wang et al. (2024). [Retrospective for the Dynamic Sensorium Competition for predicting large-scale mouse primary visual cortex activity from videos](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d758d7c0a88d741c8ca4637579c9df87-Abstract-Datasets_and_Benchmarks_Track.html). *NeurIPS 2024 Datasets and Benchmarks Track*. This retrospective and the benchmark description have different roles.
- **Static competition retrospective:** Willeke et al. (2023). [Retrospective on the SENSORIUM 2022 competition](https://proceedings.mlr.press/v220/willeke23a.html). *Proceedings of Machine Learning Research*, **220**.

The exact Sensorium and neuralpredictors revisions and their licensing boundaries are recorded in [Third-Party Notices](../THIRD_PARTY_NOTICES.md). The [data-use audit](METHODS.md) distinguishes training gradients, checkpoint selection, GPFA fitting, perturbation selection, and final scoring.

## Measurement methods

- Yu et al. (2009). [Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity](https://doi.org/10.1152/jn.90941.2008). *Journal of Neurophysiology*, **102**(1), 614–635.
- Kriegeskorte, N., Mur, M., and Bandettini, P. A. (2008). [Representational similarity analysis—connecting the branches of systems neuroscience](https://doi.org/10.3389/neuro.06.004.2008). *Frontiers in Systems Neuroscience*, **2**.
- Kornblith et al. (2019). [Similarity of neural network representations revisited](https://proceedings.mlr.press/v97/kornblith19a.html). *Proceedings of Machine Learning Research*, **97**.

Bibliographic corrections do not change the scientific results or declare a new software release. The revision record separately identifies the public sources checked and any inaccessible endpoints.
