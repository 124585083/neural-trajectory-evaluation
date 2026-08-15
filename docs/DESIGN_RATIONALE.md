# Design Rationale

The project is not designed merely to ask whether a Dynamic model predicts neural responses better. It asks whether different evaluation levels reveal different aspects of the Dynamic–Static difference, and whether trajectory-based evaluation provides interpretable temporal information beyond standard response and population-similarity summaries.

## 1. The inference problem: predictive gain is not the same as dynamic explanation

A gain in neuron-wise response correlation establishes better prediction under that metric, but it does not identify which aspect of the population response improved. In particular, response correlation alone does not show whether a model visits the correct population states at the correct times, reproduces local direction through state space, or preserves temporally organized population geometry.

The study therefore separates three evaluation levels:

1. **Response prediction:** whether individual recorded responses are predicted accurately.
2. **Output-space population-response similarity:** whether predicted and recorded population patterns share representational structure.
3. **Neural-population trajectory agreement:** whether population states occupy and move through a shared neural coordinate system in corresponding ways over time.

Dynamic Sensorium is useful for this separation because continuous natural movies make temporal order meaningful, while repeated presentations provide an empirical basis for testing whether the proposed trajectory measurement is reproducible before it is applied to model predictions.

These levels address related but non-identical questions. A model can improve on more than one level, but agreement at one level should not be assumed to explain agreement at the others.

## 2. Why response correlation, RSA/CKA, and trajectory metrics are all retained

### Response correlation

Response correlation is the basic prediction-performance gate. A trajectory score should not substitute for showing that a model predicts recorded neural responses. If basic predictive agreement were absent, a favorable low-dimensional trajectory comparison would be difficult to interpret as successful neural encoding.

### Output-space RSA and CKA

RSA and CKA test whether population-response geometry already summarizes the model difference. They are applied to predicted and recorded neural population responses, not to hidden network features.

This is deliberate: the project compares evaluation methods for neural-response predictions rather than attempting to map an arbitrarily selected internal network layer onto cortex. Output-space analysis holds the evaluated object fixed across response, RSA/CKA, and trajectory metrics and avoids introducing hidden-layer selection and correspondence choices.

### Trajectory evaluation

Trajectory metrics add explicitly time-resolved questions: where the population state is, how it moves, whether its local direction is correct, when it speeds up or slows down, and how its direction changes locally.

The intended contribution is complementarity. Trajectory evaluation is not assumed to be intrinsically superior to RSA/CKA or response correlation in every setting; the question is whether it supplies useful temporal diagnostics that those summaries do not fully provide in the present comparison.

## 3. Why trajectory measurement is defined by neural data

Static and Dynamic should not each receive a separately optimized latent space. Independent manifold fits could differ by arbitrary rotations, scales, axis choices, noise models, and temporal priors. Post-hoc alignment could then reward flexibility in the comparison rather than fidelity to the recorded neural population.

The project instead uses one **neural-data-defined GPFA** and fixes the direction of reference:

```text
recorded training neural responses
    → define one shared coordinate system

recorded oracle responses / Static predictions / Dynamic predictions
    → enter that same frozen coordinate system
```

The coordinate system is therefore determined by the biological target. Neither encoding model can refit, rotate, scale, or select axes to improve its own trajectory agreement.

GPFA is used because it combines a shared low-dimensional observation model with an explicit smooth temporal prior, allowing position and local temporal derivatives to be evaluated in one probabilistic coordinate system. This is a pragmatic measurement choice, not a claim that GPFA identifies the complete or uniquely correct cortical state space. The fitting and posterior-inference procedure is defined in [Methods](METHODS.md).

## 4. Why GPFA reliability is established before model comparison

Introducing a trajectory assay only after observing a favorable model result would create a circularity: a reader could not distinguish a genuinely reliable neural measurement from an analysis chosen because it ranks one model highly.

The project therefore separates two questions:

- **Measurement question:** do repeated presentations of the same movie produce reproducible neural trajectories under the proposed assay?
- **Model question:** after that assay is fixed, which encoding prediction better reproduces those trajectories?

The reliability gate precedes the Static–Dynamic trajectory comparison. Its protocol was internally locked before result inspection; it was not externally preregistered.

High raw split-half similarity is not sufficient by itself. Smoothness, autocorrelation, or repeated marginal statistics could produce apparently stable trajectories without preserving the information of interest. Structured nulls therefore test whether the assay depends on stimulus identity, absolute timing, temporal direction, local order, and coordinated population structure.

If the proposed measurement cannot separate recorded repeat halves from these nulls, it should not serve as a primary basis for model comparison. The validation evidence and measurement restrictions are reported in [GPFA Validation](GPFA_VALIDATION.md).

## 5. Why condition-average trajectories are primary

Repeated-movie data allow trajectory reliability to be measured rather than assumed. Validation shows that trajectories formed after averaging independent repeat groups are substantially more reliable than single-repeat trajectories.

The primary model comparison therefore asks whether an encoding model reproduces the **repeatable, stimulus-linked component** of population dynamics. It does not claim to reconstruct every trial-specific fluctuation.

Condition averaging narrows the interpretation appropriately: the trajectory result concerns reproducible condition-linked population organization. Single-trial response prediction remains a separate and necessary part of the evaluation. The empirical reliability distinction is documented in [GPFA Validation](GPFA_VALIDATION.md).

## 6. Why total-parameter matching is used, and what it controls

The full Dynamic baseline has substantially more trainable parameters than Static. A direct full-model comparison therefore mixes temporal architecture with a conspicuous whole-model size difference.

The reduced control approximately matches **total trainable parameter count**, making the primary comparison less reducible to “larger complete model versus smaller complete model.” This removes one simple explanation without redefining the models as otherwise identical.

The control does not make the cores equivalent. Static is a four-layer framewise 2D architecture; Total-parameter-matched Dynamic is a three-stage Factorized3D architecture with explicit temporal convolutions. Their core parameter counts, convolutional operations, feature transformations, optimization geometry, inductive biases, and temporal access remain different.

The appropriate inferential scope is therefore:

> A Dynamic architecture with explicit temporal convolutions is compared with a framewise Static architecture under approximately matched total trainable parameter counts.

This is a complete-model architectural comparison, not a same-backbone, core-parameter-matched, or single-variable causal experiment. Any observed difference cannot be attributed solely to temporal history.

## 7. Why multiple trajectory metrics are reported separately

Trajectory properties are not interchangeable:

- **Position** asks whether the model occupies the corresponding latent state.
- **Normalized position error** measures the magnitude of state mismatch relative to the recorded trajectory scale.
- **Velocity direction** asks whether local motion points in the corresponding direction.
- **Speed profile** asks whether fast and slow movement occurs at corresponding times.
- **Acceleration direction** measures local changes in direction and curvature-related motion.
- **Path length** summarizes total distance traveled.

Collapsing these quantities into one composite would obscure metric-specific successes and failures and would introduce arbitrary weights. They are therefore reported as a battery rather than as a single trajectory score.

Path length illustrates why reliability and diagnostic value are distinct. Total traveled distance can be reproducible while remaining nearly insensitive to temporal shifts or reversal. It can describe trajectory magnitude without establishing correct temporal alignment or direction.

Position is generally the most stable quantity. Speed and higher derivatives are more sensitive to latent dimensionality, temporal sampling, fit seed, and noise, so their exact magnitudes require greater caution. These measurement-specific restrictions are established in [GPFA Validation](GPFA_VALIDATION.md).

## 8. Why the response-matching stress test is included

When one model has both higher scalar response correlation and higher trajectory agreement, the trajectory difference may simply reflect generally better prediction. The response-matching stress test asks a narrower question:

> Can trajectory metrics remain sensitive when a major scalar response-correlation summary is made nearly the same?

The test deliberately degrades the Dynamic output using a perturbation strength selected on one repeat half, then evaluates the comparison using held-out neural responses. Selection and testing are separated so the perturbation is not chosen to maximize a test-side trajectory difference.

This is a **metric-sensitivity stress test**, not a newly trained model comparison. It does not match every neuron, response variance, RSA, CKA, or other property of the outputs, and no formal equivalence test is implied. Its purpose is to test whether the chosen scalar response summary is sufficient to explain the trajectory distinction. The procedure is specified in [Methods](METHODS.md), with findings in [Results](RESULTS.md) and the detailed evidence ledger.

## 9. Why time reversal is an important counterexample

Some condition-averaged representational summaries operate on state sets or time-averaged patterns that do not uniquely encode order. Complete time reversal can preserve the visited population states, condition-level averages, and some representational summaries while reversing local direction, sequence, and phase.

Time reversal therefore provides a controlled sufficiency test. If a conventional summary remains unchanged after reversal while an order-sensitive trajectory metric changes, that conventional summary is not sufficient to encode the temporal property destroyed by reversal.

The inference is deliberately limited to **sufficiency failure for the tested metric formulation**. It does not establish mathematical independence, universal statistical independence, or the inability of every possible time-aware RSA/CKA construction to represent temporal information.

## 10. Why graded temporal-weight attenuation is used

An intact-versus-fully-ablated comparison would provide only one perturbation endpoint. Progressively attenuating learned off-center temporal weights instead tests whether trajectory quality varies systematically with the retained strength of learned temporal-history weighting.

A graded intervention is more informative about dose response than a single endpoint, but it does not isolate a unique cause. Increasing attenuation also increases the magnitude of perturbation to a trained network. The observed pattern is therefore interpreted as:

> **Consistent with a graded temporal-history effect, but temporal specificity remains to be controlled.**

Without an independently magnitude-matched non-temporal perturbation, loss of temporal-history information cannot be uniquely separated from generic degradation caused by progressively stronger weight disruption. The current experiment is not described as clean causal isolation or as proof that temporal history alone causes the complete-model difference.

## 11. Why the enriched conventional-metric battery is used

Incremental value should not be assessed only against one scalar response score, one simple RSA, or one CKA. The project therefore combines response summaries with time-aware RSA and CKA measures to create a stronger conventional baseline.

The leave-perturbation-family-out analysis asks how well this enriched battery predicts trajectory-metric variation for a transformation family that was excluded from regression fitting. Holding out entire perturbation families tests whether the relationship generalizes across kinds of transformation rather than merely memorizing one degradation curve.

If the conventional battery predicts substantial but incomplete trajectory variation, the appropriate interpretation is **partial predictability**, **incremental information**, and **failure of complete sufficiency**. It is not independence, orthogonality, or evidence for a wholly separate information source.

## 12. Falsification and decision logic

The design includes outcomes that would weaken its methodological interpretation:

| Design question | Pattern supporting usefulness | Pattern weakening the interpretation | Consequence |
|---|---|---|---|
| Is the GPFA measurement trustworthy? | Repeated-movie trajectories are reproducible and exceed structured condition/time/population nulls | Recorded repeat halves are not more consistent than the structured nulls | The trajectory assay should not serve as a primary model-comparison measure |
| Is there a response-level gain to explain? | Dynamic shows a reproducible response advantage | Little or no response advantage | Trajectory evaluation may still compare temporal structure, but it is no longer explaining an established predictive gain |
| Do conventional population metrics detect relevant structure? | Time-aware RSA/CKA detect some model differences, while trajectory metrics expose additional order/direction sensitivity | RSA/CKA fully capture the same structure and trajectory behavior | Trajectory evaluation offers little incremental value; RSA/CKA detecting a difference alone is not a failure |
| Does trajectory sensitivity remain after response matching? | Trajectory differences remain when the selected scalar response summary is nearly matched | Trajectory differences disappear with response-score matching | The trajectory result may largely restate scalar predictive quality |
| Is the trajectory battery sensitive to temporal order? | Reversal changes direction/order-sensitive trajectory metrics | Trajectory metrics remain largely invariant to reversal | The battery is not adequately diagnostic of temporal order or direction |
| Does the assay track graded temporal-weight perturbation? | Trajectory quality changes systematically with attenuation severity | No consistent relation between attenuation and trajectory quality | The proposed link to learned temporal-history weighting is weak; monotonicity alone would still not establish temporal specificity |
| Does the enriched conventional battery fully explain trajectory variation? | Conventional features predict some but not all held-family trajectory variation | Held-family trajectory metrics are almost completely predictable | Trajectory evaluation has limited incremental diagnostic value for the tested perturbations |

Assay failure and hypothesis weakening are not identical. Poor neural reliability, leakage, or dependence on arbitrary model-specific alignment would mean that the measurement cannot adjudicate the question; they would not demonstrate that temporal population organization is scientifically irrelevant. This is why measurement validation is placed before model comparison.

## 13. What the project can and cannot claim

Within its evidence scope, the design can support claims about:

- response-level Dynamic–Static differences;
- output-space population-response representational differences;
- reproducible condition-average neural trajectories;
- model differences in trajectory position and local direction;
- temporal-order sensitivity of the tested trajectory metrics;
- incremental diagnostic information relative to the tested conventional battery.

The design does not by itself identify:

- one unique cortical dynamical mechanism;
- temporal history as the sole cause of the complete-model difference;
- a unique biological GPFA dimensionality or timescale;
- mathematical independence from all RSA/CKA formulations;
- the complete neural-response state;
- dataset-wide biological conclusions from the one-session trajectory pilot.

Response prediction, population-response representational similarity, and trajectory evaluation answer different but complementary questions.

## 14. Documentation pointers

- [README](../README.md): concise scientific story.
- [Methods](METHODS.md): complete procedures.
- [Results](RESULTS.md): main scientific result chain.
- [GPFA Validation](GPFA_VALIDATION.md): measurement reliability and limitations.
- [Detailed Q1–Q6 evidence](results/Q1_Q6_ANSWERS.md): question-by-question statistics and qualifications.
