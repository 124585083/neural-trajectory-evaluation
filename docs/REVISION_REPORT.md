# Repository revision, 3 October 2026

Publication note: repository publication was authorized after this revision handoff. Statements below about uncommitted work describe the earlier handoff. The restriction on the new numerical replay payload remains in effect.

This revision improves the research explanation, attribution, source inventory and verification of the completed Sensorium and Mental-Pong studies. It does not change their scientific results. The work remains in the local working tree; no commit, push, public release or DOI was created.

The starting branch was `main` at `0637c08bca6661e32753f7f4c52d141bcd76f61b`, with no uncommitted changes. All eight Git LFS payloads were present. An external snapshot preserved the 388 tracked files before editing. Its inventory contains original hashes and locations; private filesystem paths and revision instructions are excluded from publication. The [baseline checks](../experiments/05_mental_pong/integration/revision_20261003/baseline_checks.json) passed on that unedited snapshot.

## Research entry, attribution and references

The [README](../README.md) now introduces the two research questions before execution details. Its scientific introduction is approximately 1,000 words. It explains the tasks, comparisons, principal findings and immediate limits, with links to supporting methods and results. The [contribution statement](../AUTHORS.md) separates Xiaotian Zhu's research decisions from substantial ChatGPT discussion/writing assistance and Codex implementation/execution assistance. It does not assign unsupported percentages or claim complete manual verification of every artifact. The citation author remains human, and the recorded release version and date are unchanged.

The [reference guide](REFERENCES.md) distinguishes the 2025 DMFC article, the 2024 Zenodo release and the 2022 behavioral/RNN paper. The Sensorium benchmark reference identifies arXiv version 2, dated 12 July 2024, alongside the competition retrospective. Existing method references remain available. The audited Mental-Pong Git revision remains `b976255be73140c759d8f8db0fd8ff551a4e2d73`; Git code and the downloaded archive are treated as distinct sources.

## Data selection and measurement meaning

Source and training-log inspection confirmed that Static, full-width Dynamic and parameter-matched Dynamic used train-tier observations for gradients and oracle correlation for scheduling, stopping decisions and checkpoint selection. Final complete-sequence scoring reused the oracle tier. The [seven-row data-use table](METHODS.md#data-use-and-evaluation-independence) separates those decisions from train-only GPFA fitting, perturbation-half selection and final measurement. Reloading a checkpoint demonstrates executable saved weights; it does not create an independent evaluation set. The hidden test responses remain unavailable locally.

Acceleration-direction cosine is consistently described as agreement in how the velocity vector changes. It includes changes in speed along a straight path and is not a direct curvature measure. The equations, masks, small-vector handling and numerical values are unchanged. Posterior-query derivatives remain distinguished from independently sampled neural activity. The speed-profile model comparison remains inconclusive.

The English review covered the research entry, major documents, study and model reports, comments, docstrings, CLI descriptions, report templates and plotting text. Exact source identifiers and historical scientific records were retained. Six affected PNG figures were regenerated from saved values and inspected: the Sensorium comparison and five Mental-Pong presentation figures. The Sensorium figure separates metric families and warns against comparing their absolute effect sizes. The Mental-Pong figures define the four curves, valid times, collision interval and split-variability shading. The averaging figure describes behavioral cancellation without assigning a percentage of neural information loss. Source files and new figure hashes are in the [writing and figure record](../experiments/05_mental_pong/integration/revision_20261003/writing_and_figures.json).

The [future plan](FUTURE_DIRECTIONS.md) first asks whether above-target and below-target trials within a physical condition show corresponding neural readout shifts. It requires matched neural/behavioral trial identities. The current condition means cannot recover those groups. The five-step plan remains conditional; none of its experiments was run.

## Source availability and numerical replay

The [dependency inventory](../experiments/05_mental_pong/mini_replay/dependencies.csv) separately covers numerical replay, export from preserved artifacts, and full historical execution. The actual project modules for data loading, endpoint construction, representation fitting, prefix inference, OLS, scoring and both controls were already present. An import audit checked 35 internal runtime modules without finding missing project imports. A historical capacity module occurs only as a saved provenance fingerprint; its unused AE dependencies were not copied into the final pipeline.

A genuine private export contains the four preselected split-0/q=0 cases: Mahler and Perle, each with FA50 and GPFA50, using half1. The export is 5,713,262 bytes. All 27 parent artifact hashes were checked before and after export. The bundle retains actual training/test features, labels, masks, times, coefficients, endpoint allocation and condition-mismatch pairs. Numeric NPZ loading disables pickle. JSON encodes undefined metrics as `null`.

Default replay recomputes `X_test @ coefficient.T + intercept` and then scores the predictions. It checks each path against its own reference, the complete objective/behavior 2×2 matrix, the random-endpoint geometry baseline and a fixed-head mismatch on identical valid rows. Across 604 score rows, the largest absolute difference from the original per-split values was `1.6370904631912708e-11`, below the fixed `1e-8` absolute tolerance. This checks one saved split and allocation; it does not reproduce the 100-split or 1,000-randomization distributions.

The explicitly requested optional OLS check refitted only the 12 original linear heads on the exported training rows. All had design rank 50. The largest held-out prediction difference was approximately `1.51e-14`. No neural representation was refitted. No new label, split, seed, dimension search or scientific control was introduced. The [mini-replay record](../experiments/05_mental_pong/integration/revision_20261003/mini_replay.json) separates saved-weight replay from this optional check.

A temporary copy with a different working directory and a path containing spaces passed on Windows, Python 3.11.9 and NumPy 1.26.4. The test removed project environment variables and custom `PYTHONPATH`, denied access to the original source/data directories, blocked network operations, and required no PyTorch, SciPy or scikit-learn import for default replay. Linux and macOS execution remains unverified.

### Commands and actual public status

From the repository root:

```shell
python scripts/verify_integration.py
python experiments/05_mental_pong/run.py --mini-replay --output /path/to/new-output
```

The second command currently exits with an explicit **payload withheld** message. The public tree contains source and instructions, but no new scientific mini bundle. An authorized holder of the preserved inputs can use the [export instructions](../experiments/05_mental_pong/mini_replay/README.md#commands-and-availability), then run:

```shell
python experiments/05_mental_pong/run.py --mini-replay --mini-bundle /path/to/private-mini --output /path/to/new-output
```

The saved provider metadata for Zenodo record 13952210 specifies **CC BY-NC-ND 4.0**. The current landing page agrees on the archive identity and date; an API refresh returned HTTP 403. Both archive READMEs and the complete member list were inspected without finding broader data-redistribution terms. The [rights audit](../experiments/05_mental_pong/integration/revision_20261003/data_rights_audit.json) preserves these distinctions. Permission to distribute the transformed bundle has not been established. This is a conservative publication decision, not a definitive legal classification of numerical derivatives. Existing scientific artifacts remain preserved, with their redistribution status also left unresolved. A code license or successful private export does not settle that question.

## Safeguards and verification

The [safeguard record](../experiments/05_mental_pong/integration/revision_20261003/safeguards.json) reports two focused changes:

- The protocol lock compares scientific settings, source/checkpoint identities, data bytes, neuron order, trial groups and time support. Path-only changes do not invalidate otherwise identical inputs. Mismatches produce precise errors without overwriting a completed lock. Historical locks without sufficient identity metadata require an explicit diagnostic override and remain labelled unverified.
- The actual Static adapter, pinned 2D core, Gaussian readout and shifter passed evaluation-mode single-frame perturbation tests using synthetic inputs. All 15 frame/covariate changes left other retained outputs unchanged. A sequence-mean counterexample passed permutation equivariance but failed independence. These checks do not establish training-mode BatchNorm independence or properties of upstream normalization.

| Check | Result and boundary |
|---|---|
| Protocol guard and existing Phase 4 tests | 18 passed, including 13 new guard tests |
| Actual Static adapter regression tests | 4 passed; CPU synthetic inputs, no training data |
| Existing Mental-Pong suite plus publication-map tests | 115 passed, 3 skipped because full C arrays are not materialized in the publication tree |
| Mini suite with real private bundle | 20 passed, including cold-copy, tampering, wrong metadata, overwrite protection and optional OLS checks |
| Mini suite without bundle | 1 passed, 19 explicitly skipped; public numerical replay not claimed |
| Original numeric/artifact/record preservation | 223 files byte-identical, including eight LFS payloads |
| Saved result aggregation | 48 own-target and 192 cross-target summary rows verified against 100 rounds; maximum difference `3.33e-15` |

The first broad Mental-Pong test command was launched from the repository root and encountered eight import collection errors. Running the documented command from `experiments/05_mental_pong` passed. Matplotlib deprecation warnings and a short synthetic FA fixture's convergence warning remain recorded in the [test report](../experiments/05_mental_pong/integration/revision_20261003/existing_mental_pong_tests.json). No scientific fitting was used to resolve them.

The final publication check found no language, private-path, syntax or relative-link errors across 384 text files and 101 parsed Python files. It checked 486 relative links. Artifact verification confirmed 7,347 registered hashes; another 226 large files were present but were not rehashed in that default scan. The 27 mini-export parents were separately hash-verified. One historical instruction attachment remains unavailable; it is not an input to the numerical replay. The missing prior review, that attachment and unchecked large-file hashes are reported separately from test failures. `CITATION.cff` parses as YAML, and its human author, preferred citation, version and release date match the baseline.

Original scientific completion records and the previous integration verification remain unchanged. The additive [publication map](../experiments/05_mental_pong/integration/revision_20261003/publication_map.csv) records baseline and current hashes, with reasons for edits. The verifier checks this chain without replacing historical source hashes. Current language, links, numerical preservation and availability are summarized in the separate [revision verification](../experiments/05_mental_pong/integration/revision_20261003/verification.json).

## R1–R12 review correspondence and completion

`Repository_Review_2026-10-02.md` was not found in the available workspace, Downloads or attachment locations. Its exact R1–R12 wording and numbering therefore cannot be certified. The following local checklist covers the supplied revision specification; the C identifiers are deliberately not presented as the missing review's R identifiers.

| Local check | Work completed | Status |
|---|---|---|
| C1 | Starting state, external snapshot and baseline verification | Complete |
| C2 | Research-first README and evidence links | Complete |
| C3 | Human/tool contributions and unchanged software release citation | Complete; author can review final wording |
| C4 | Paper, dataset, benchmark and upstream reference identities | Complete with failed API refresh disclosed |
| C5 | Three model routes and seven-row data-use audit | Complete |
| C6 | English prose, templates, comments and terminology | Complete for available publication files |
| C7 | Acceleration interpretation and six inspected presentation renders | Complete; scientific values unchanged |
| C8 | Focused five-step future plan | Complete; no experiments run |
| C9 | Source/import inventory and actual licensing evidence | Complete for available source |
| C10 | Four-case numerical replay, export, cold run and tampering checks | Verified privately; public payload blocked by unresolved permission |
| C11 | Protocol guard and actual Static independence test | Complete within stated test boundaries |
| C12 | Separate records and original-to-edited publication hash chain | Complete; exact R1–R12 mapping unavailable |

Batch 1 and Batch 2 are complete for the available files. Batch 3's source, exporter, safeguards and private verification are complete; public numerical payload delivery remains blocked by rights uncertainty. Full scientific reconstruction, untested operating systems and exact prior-review mapping remain unverified. The original `raw_preprocessing=fail`, unresolved neural trial membership and endpoint-timing limits are unchanged. Publication of the working-tree changes, any release, and expansion of the numerical payload await separate authorization and the applicable data rights.
