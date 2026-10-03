# Writing and evidence guide

Write project-owned reports, comments, templates, messages, configuration explanations, and figure labels in English. Preserve required third-party notices and accurate names. Keep original non-English source records outside the publication tree and link their identifiers through the migration records.

## Explain the claim and its support

Give each paragraph one main point, using an explicit subject and verb. A useful order is the question or action, the data and comparison, the result, then the limitation needed to interpret it. State the subject, action, and meaning, then explain the evidence. Split sentences that combine several methods and qualifications. Put limits near the claims they restrict. Define technical terms when they first matter, including factor analysis (FA), Gaussian-process factor analysis (GPFA), ordinary least squares (OLS), a physical condition, and a pseudopopulation.

Use tables to compare parallel quantities and short lists for procedures. Preserve equations, assumptions, units, time conventions, missing values, and aggregation rules. Shorter prose must still explain the reasoning.

## Use consistent terms

| Term | Meaning |
|---|---|
| Neural trajectory | Sequence of neural population states over time |
| Objective ball path | Physical reference path |
| Behavior-constrained candidate path | Path constructed from the observed endpoint and stated geometry |
| Reconstructed position | Output of the neural readout |
| Condition-mean neural response | Neural observations averaged within the specified condition |
| Single-trial neural response | Response of an identified individual trial |
| Mean-endpoint geometry baseline | Training-set mean endpoint combined with known task geometry |

“Mean-endpoint geometry” does not mean a geometric mean. Distinguish the neural trajectory from its task-coordinate readout. Explain an own-target score as the score of a readout against the target it was trained to reconstruct. In a cross-score, name the fixed readout and the reference path separately. A score for their difference is a supplementary measurement.

## Edit sentences without changing the science

Avoid stock praise, unsupported adjectives, and repeated contrast formulas. Words such as “robust,” “validated,” or “mechanistic” need a stated test. Preserve necessary negative results explicitly. Avoid routine “not X but Y” and similar contrast frames; retain a contrast when it conveys a scientific distinction. Explain valid time points or sample coverage instead of using “support” ambiguously. Name the excluded perturbation family instead of relying on “held-family”. Identify the data that fitted a model when discussing fit provenance.

| Before | After |
|---|---|
| The framework provides a powerful bridge to behaviorally grounded computation. | We compared two position targets using the same linear readout. The analysis tests which target is easier to reconstruct. |
| Averaging erased behavior-specific neural information. | Opposite endpoint errors partly canceled in behavioral averages. Unknown neural trial membership prevents measuring the corresponding neural differences. |
| Existing metrics cannot explain trajectory scores. | The tested finite-sample ridge regressions left residual variation. This result does not establish information independence. |
| The causal analysis is leakage-free. | Filtering used only completed published-input bins. Upstream filling across conditions and times remains unresolved. |

## Describe the measured quantity

Use **acceleration-direction agreement** or **acceleration-direction cosine** for the second-difference metric. It measures agreement in how the velocity vector changes over time, including speed changes along a straight path. It is not a direct curvature measurement. Distinguish physical ball velocity, neural-response differences and derivatives of an inferred GPFA posterior.

Describe a fresh-process checkpoint run as reloaded evaluation. Independence requires observations excluded from the relevant earlier selection steps. A test half excluded from one perturbation choice may still have been used for encoding-model checkpoint selection.

Keep tool assistance and authorship explicit. Do not invent percentages of generated work or complete manual validation. Replace conversational references to requests or approvals with the actual procedure. Mark examples selected after results were examined as post hoc.

## Keep evidence and versions visible

Link substantive claims to their protocol and saved result, using repository-relative links. The [evidence index](../results/study_evidence_index.csv) is a source map, not a ranking across datasets. Keep Study 1 and Study 2 validation scopes separate. Repeated splits and randomizations are not independent animals or neural trials.

Treat English reports and presentation renders as new publication versions. Preserve source hashes separately from destination hashes. Record whether a change is editorial, a translation, a path adapter, or a presentation render. Check that numerical arrays and identifiers remain unchanged.

Scan text and review flagged passages manually; do not automatically rewrite statements using a readability score or sentence-length cutoff. Inspect rendered figures as well as their source strings. A prior visual pass does not certify a newly rendered figure. Record unavailable inputs and unrun checks precisely.
