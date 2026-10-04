# Authors and contributions

**Xiaotian Zhu** led this independent secondary analysis of public neural and behavioral data. He set the research aims, revised the hypotheses and comparison targets, specified analysis constraints, and evaluated the reported results. He is responsible for the final study decisions and the claims presented here.

The analysis and repository were developed with assistance from **ChatGPT and Codex**. ChatGPT supported literature discussion, analysis planning, examination of interpretations and part of writing. Codex supported implementation, debugging, local execution, tests, and preparation of reports. 

| Area | Human research role | Tool assistance | Evidence |
|---|---|---|---|
| Study questions and scope | Define and revise questions; make final scope decisions | Discussion and alternative suggestions | [Design rationale](docs/DESIGN_RATIONALE.md), [Mental-Pong version history](experiments/05_mental_pong/archive/HISTORY.md) |
| Analysis implementation | Specify required behavior and evaluate relevant outputs | Code generation, adaptation, debugging and execution | [Study modules](experiments/), [retained training records](experiments/01_baselines/records/), [Mental-Pong methods](experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/REPORT.md) |
| Evaluation and interpretation | Compare results, preserve limitations and decide supported claims | Numerical summaries, counterexamples and review suggestions | [Sensorium results](docs/RESULTS.md), [Mental-Pong closeout](experiments/05_mental_pong/trajectory_project/closeout_v1/FINAL_REPORT.md) |
| Writing and packaging | Select the final scientific account and its claims | Drafting, language revision, plots and packaging | [Scientific entry point](README.md), [writing guidance](docs/WRITING_GUIDE.md) |

The original studies' authors collected the recordings, designed the tasks and released their data. Published FA, GPFA, OLS, RSA, CKA and baseline architectures are credited to their sources. This repository contributes secondary-analysis questions, comparison design, implementation and adaptation, checks, and interpretation. Upstream data and software authorship is documented separately in [References](docs/REFERENCES.md) and [Third-Party Notices](THIRD_PARTY_NOTICES.md). AI tools are described as assistance; the citation author field identifies the human author.

## Contact

- **Institution:** No institutional affiliation
- **Email:** [xiaotian_zhu@outlook.com](mailto:xiaotian_zhu@outlook.com)
- **ORCID:** [0009-0003-2956-8476](https://orcid.org/0009-0003-2956-8476)
- **GitHub:** [124585083](https://github.com/124585083)

For reproducibility questions, use [GitHub Issues](https://github.com/124585083/neural-trajectory-evaluation/issues). Future human contributors should be credited with their actual roles and added to `CITATION.cff` when formal authorship is warranted.
