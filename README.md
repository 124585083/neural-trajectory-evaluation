# Neural Trajectories for Dynamic Visual Representation

## Research motivation

I want to understand how neural populations keep visual features associated with particular objects and selectively update those representations. When two objects cross paths, the system must detect a change in motion and determine which object's estimated state should change. This matters for predicting events and acting on the right target.

Visual representations evolve as objects move, disappear, and reappear. I describe these changes as **trajectories of neural population activity** in a lower-dimensional space to examine shared population dynamics, compare model predictions with neural recordings, and relate neural states to task variables. The two studies in this repository develop the measurements and evidence needed for that longer-term research. **Sensorium** evaluates trajectory-based model comparisons; **Mental-Pong** examines task-motion and behavior-related representations.

Each study is organized by research goal. **Setup** explains the question, data, methods, and controls; **Results** presents the observations and their interpretation. The model/data guides, research question collections, and runnable code have separate entry points.

## 1. Sensorium — evaluating dynamic neural representations

### Data and models

The Sensorium 2023 data contain mouse visual-cortex responses to natural movies, together with behavioral and pupil measurements. Response prediction is evaluated across **five sessions (40,034 recorded neurons)**. The detailed trajectory analysis uses **one session, 512 fixed neurons, and six repeated movies**. A frame-based **Static** encoder and a temporally convolutional **Dynamic** encoder predict the same neural response targets. Recorded and predicted responses are then measured in one shared, frozen **four-dimensional GPFA** space. See [Models and Data](results/sensorium/MODELS_AND_DATA.md) for architectures, data roles, and preprocessing.

### Research goals, methods, and findings

| Goal | Method | Main finding | Detailed documents |
|---|---|---|---|
| **1. Check the model implementations** | Reproduce a historical full Dynamic run; test whether the current Static model processes frames independently. | The full Dynamic checkpoint gives local response **r = 0.1967**; the current Static model passes **15 frame-independence checks**. | [Setup](results/sensorium/01_model_validation/SETUP.md) · [Results](results/sensorium/01_model_validation/RESULTS.md) |
| **2. Establish a reliable trajectory measurement** | Fit a neural-data-defined GPFA; compare independent repeat halves and temporal/condition nulls. | Balanced repeat-half agreement reaches **0.853** for position and **0.641** for movement direction; the primary metrics separate the specified nulls. | [Setup](results/sensorium/02_trajectory_measurement/SETUP.md) · [Results](results/sensorium/02_trajectory_measurement/RESULTS.md) |
| **3. Compare Static and Dynamic models** | Evaluate neural-response prediction and trajectory agreement using frozen models and common measurements. | Dynamic has higher mean response **r** across five sessions (**0.1911 vs. 0.1596**) and better pilot trajectory position and direction agreement. | [Setup](results/sensorium/03_model_comparison/SETUP.md) · [Results](results/sensorium/03_model_comparison/RESULTS.md) |
| **4. Test sensitivity to temporal order** | Reverse predicted time sequences; compare time-averaged RSA/CKA with time-aware and trajectory metrics. | The selected time-averaged summaries stay unchanged under reversal, while trajectory and time-aware scores detect the change. | [Setup](results/sensorium/04_temporal_metric_tests/SETUP.md) · [Results](results/sensorium/04_temporal_metric_tests/RESULTS.md) |
| **5. Test trajectory differences after matching mean response correlation** | Select an output perturbation on validation data and apply it unchanged on test. | The test response-r gap is **0.0033**; Dynamic retains its advantage in trajectory position and direction. | [Setup](results/sensorium/05_response_matching/SETUP.md) · [Results](results/sensorium/05_response_matching/RESULTS.md) |
| **6. Examine temporal and spatial contributions** | Weaken temporal convolution weights and compare with spatial-weight controls calibrated on validation data. | Temporal weakening changes trajectory agreement; response-matched spatial controls produce **different outcomes across metrics and strengths**. | [Setup](results/sensorium/06_temporal_weight_controls/SETUP.md) · [Results](results/sensorium/06_temporal_weight_controls/RESULTS.md) |

The [Sensorium Questions](results/sensorium/QUESTIONS.md) explore the reasoning behind these choices. One example is: **“Do trajectory metrics reveal new information, or describe the same responses differently?”**

**Code:** [Execution and reproduction guide](code/sensorium/README.md) · [Main entry point](code/sensorium/run.py). The guide identifies required input arrays, frozen models, supported commands, and generated results.

## 2. Mental-Pong — neural representations of task motion and behavior

### Data and models

In Mental-Pong, two macaques (**Mahler** and **Perle**) track a moving ball through visible and occluded periods and respond by positioning a paddle. The released dorsomedial frontal cortex (DMFC) data provide **condition-averaged neural pseudopopulations**. The analysis uses separate **FA50 and GPFA50** representations and fixed linear position readouts across **100 whole-condition splits**. The neural dataset and behavioral records are summarized in [Models and Data](results/mental_pong/MODELS_AND_DATA.md).

### Research goals, methods, and findings

| Goal | Method | Main finding | Detailed documents |
|---|---|---|---|
| **1. Can neural trajectories represent the ball's motion?** | Decode the physical ball position from FA50/GPFA50 across visible, hidden, and collision-related periods; compare with shuffled conditions and task geometry. | Objective vertical position is recoverable during both visible and hidden periods. Full-interval **r = 0.493–0.720** across animals and representations; task geometry remains a strong reference. | [Setup](results/mental_pong/01_task_motion/SETUP.md) · [Results](results/mental_pong/01_task_motion/RESULTS.md) |
| **2. Do condition-averaged neural trajectories align with a behavior-constrained path?** | Compare the physical path with a candidate path based on mean paddle endpoints; use cross-scoring, condition mismatch, randomized endpoints, and a geometry baseline. | All four full-path comparisons favor the **physical path**. Real behavioral candidates outperform randomized endpoint assignments in several phases, while the full-path geometry baseline remains stronger. | [Setup](results/mental_pong/02_behavior_relation/SETUP.md) · [Results](results/mental_pong/02_behavior_relation/RESULTS.md) |

### How the unexpected result shapes the next study

The second analysis favored physical trajectories over behavior-constrained candidates. It also showed that **above-target and below-target behavioral errors partly cancel when trials are averaged within a condition**. The released neural condition means cannot be separated into the corresponding behavioral groups, leaving the possible neural effect of that averaging unresolved.

The proposed next study uses **paired neural and behavioral trials** to compare **above-target and below-target** responses within the same physical ball condition. It would first test whether neural trajectories reliably follow behavioral direction **before feedback**, then whether their differences are **persistent** or change around a relevant task event. Positive and negative outcomes would refine the interpretation of behavior-related state and updating. The rationale, controls, and predicted outcomes are in [Next Steps](results/mental_pong/NEXT_STEPS.md).

The [Mental-Pong Questions](results/mental_pong/QUESTIONS.md) examine the evidence behind both goals. For example: **“How does averaging across trials limit what we can say about behavior?”**

**Code:** [Execution and reproduction guide](code/mental_pong/README.md) · [Main entry point](code/mental_pong/run.py). The guide separates saved-model scoring, small OLS refits, and the data needed for a full reconstruction.

## Code, sources, and attribution

Both projects are independent secondary analyses of published datasets. Their code guides describe executable workflows and input requirements; the [artifact manifest](models/manifest.json) records the frozen models, numerical inputs, and their availability.

Research questions, study decisions, and scientific interpretation were directed by **Xiaotian Zhu**. ChatGPT assisted with research discussion and writing; Codex assisted extensively with implementation, testing, figures, and documentation. The original research teams are credited in the study documents.

The [MIT license](LICENSE) covers original repository software and documentation; [Third-Party Notices](THIRD_PARTY_NOTICES.md) records external data and code sources and their terms. [Citation](CITATION.cff) · [Contact](mailto:xiaotian_zhu@outlook.com)
