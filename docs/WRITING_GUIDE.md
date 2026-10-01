# Writing and evidence guide

Write project-owned reports, comments, templates, messages, configuration explanations, and figure labels in English. Preserve required third-party notices and accurate names. Keep original non-English source records outside the publication tree and link their identifiers through the migration records.

## Explain the claim and its support

Give each paragraph one main point. State the subject, action, and meaning, then explain the evidence. Split sentences that combine several methods and qualifications. Put limits near the claims they restrict. Define technical terms when they first matter, including factor analysis (FA), Gaussian-process factor analysis (GPFA), ordinary least squares (OLS), a physical condition, and a pseudopopulation.

Use tables to compare parallel quantities and short lists for procedures. Preserve equations, assumptions, units, time conventions, missing values, and aggregation rules. Shorter prose must still explain the reasoning.

## Use consistent terms

| Term | Meaning |
|---|---|
| Neural trajectory | Time course in a neural representation space |
| Objective ball path | Physical reference path |
| Behavior-constrained candidate path | Path constructed from the observed endpoint and stated geometry |
| Reconstructed position | Output of the neural readout |
| Condition-mean neural response | Neural observations averaged within the specified condition |
| Single-trial neural response | Response of an identified individual trial |
| Mean-endpoint geometry baseline | Training-set mean endpoint combined with known task geometry |

“Mean-endpoint geometry” does not mean a geometric mean. Distinguish the neural trajectory from its task-coordinate readout. Distinguish an own-target score from cross-scoring or a difference-decoding score.

## Edit sentences without changing the science

Avoid stock praise, unsupported adjectives, and repeated contrast formulas. Words such as “robust,” “validated,” or “mechanistic” need a stated test. Preserve necessary negative results explicitly.

| Before | After |
|---|---|
| The framework provides a powerful bridge to behaviorally grounded computation. | We compared two position targets using the same linear readout. The analysis tests which target is easier to reconstruct. |
| Averaging erased behavior-specific neural information. | Opposite endpoint errors partly canceled in behavioral averages. Unknown neural trial membership prevents measuring the corresponding neural differences. |
| Existing metrics cannot explain trajectory scores. | The tested finite-sample ridge regressions left residual variation. This result does not establish information independence. |
| The causal analysis is leakage-free. | Filtering used only completed published-input bins. Upstream filling across conditions and times remains unresolved. |

## Keep evidence and versions visible

Link substantive claims to their protocol and saved result, using repository-relative links. The [evidence index](../results/study_evidence_index.csv) is a source map, not a ranking across datasets. Keep Study 1 and Study 2 validation scopes separate. Repeated splits and randomizations are not independent animals or neural trials.

Treat English reports and presentation renders as new publication versions. Preserve source hashes separately from destination hashes. Record whether a change is editorial, a translation, a path adapter, or a presentation render. Check that numerical arrays and identifiers remain unchanged.

Scan text and review flagged passages manually; do not automatically rewrite statements using a readability score or sentence-length cutoff. Inspect rendered figures as well as their source strings. A prior visual pass does not certify a newly rendered figure. Record unavailable inputs and unrun checks precisely.
