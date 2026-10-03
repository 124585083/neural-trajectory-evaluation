# C: random-endpoint candidates with new OLS fits

Each animal has 1000 fixed endpoint permutations. OLS denotes ordinary least squares; FA and GPFA denote factor analysis and Gaussian-process factor analysis. A given q is held constant across all times, 100 original condition splits and FA50/GPFA50. The 78 valid conditions receive endpoints; condition 59920 retains only its split identity. Each valid condition-by-time pair contributes one training row, without behavioral record-count weighting.

The [geometry audit](geometry_feasibility.json) checks all 78-by-78 anchor-endpoint combinations before neural randomization scores. There are no compatibility strata, clipped endpoints, extra reflections or score-dependent redraws. The existing candidate function replays the original behavioral labels with identical x and support. `endpoint_influence` combines the original post-collision support with samples after the initial anchor for no-collision conditions and changes scoring only.

Each split and representation calls float64 `sklearn.linear_model.LinearRegression(fit_intercept=True, positive=False)` once to fit 1000 random y outputs, each with its own coefficient vector, and one shared x output. Each column is an OLS solution; comparisons with separate OLS and the original candidate target verify equivalence. No scaling, weights, regularization or tuning are added. FA/GPFA remain fixed. Each q has 51 y coefficients and a 102-coefficient xy head, with the identical x solution shared. Random predictions are evaluated against their corresponding random labels.

| Artifact | Meaning |
| --- | --- |
| `*_all_scores.npz` | `random_scores[q,split,epoch,metric]`, shape 1000x100x7x14; `observed_scores[objective/behavior,split,epoch,metric]`; support counts included |
| `*_per_q_mean100.csv` | Each q's scores averaged across its original 100 splits |
| [random_endpoint_summary.csv](random_endpoint_summary.csv) | Distribution of 1000 q statistics, observed scores, differences and empirical tail proportions |
| `*_endpoint_assignments.npz` | Endpoint assignments, geometric offset/alpha, masks and identities; labels reconstruct as `offset + alpha * endpoints[q,:,None]` |
| `*_OLS_weights.npz` | New y coefficients `[split,q,dimension]`, intercepts, shared x coefficients, rank and training-set mean endpoints |
| `*_fixed_examples.npz` | Preselected q=0/1/2 test-only predictions; training and invalid samples remain NaN |

Read saved axis names before indexing. Large arrays and their access locations are listed in the [artifact registry](../../../../integration/artifact_registry.csv). A prediction band describes readout-split stability, not trial variability. The 100000 q-by-split scores are not independent experiments.

The mean-endpoint geometry baseline combines the valid training conditions' mean endpoint with each test condition's known anchor and horizontal position. For the objective label it uses the training mean of `objective_end_y`; the objective scoring target remains the original path. `skill = 1 - SSE_neural / SSE_baseline`, with NA for a zero denominator. This baseline includes known task geometry.

Endpoint permutations preserve the endpoint marginal distribution, not every aspect of path difficulty. Random paths retain starts, collisions, timing and horizontal structure. Target SD, range and distance from the objective path are saved on scoring support; endpoint SD/range are in the geometry audit. Tail values are empirical comparisons under this allocation rule. `raw_preprocessing=fail` remains unchanged.

Use the [publication entry point](../../../../README.md) for a read-only saved-case replay. Full preparation/fitting uses the historical dependency bundle and explicit commands documented there; integration did not repeat it. [completed.json](completed.json) records completion of all four branches and saved-weight replay, while project-wide status is recorded separately.
