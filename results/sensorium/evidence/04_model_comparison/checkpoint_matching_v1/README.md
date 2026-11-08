# Dynamic checkpoint-matching evidence

This directory is the active response-matching control, separate from the immutable `retrained_v1` source run. [Setup](../../../05_response_matching/SETUP.md) specifies the validation-only checkpoint choice and frozen test comparison; [Results](../../../05_response_matching/RESULTS.md) reports the completed measured outcomes.

[source_audit.json](source_audit.json) records the initial audit of Git-tracked published files at commit `f738eec69e30a58d5310814b220a9b6156b4fa60`.

The candidate rule retains every completed epoch through `min(119, verified native early-stop epoch)`, without changing the original training stopping policy. A paused run is not a completed cohort. `protocol_revision.json` records this clarification before full-pilot checkpoint matching and new test scoring.

The run stopped natively at epoch 91, and validation selected epoch 63. The validation response-r gap is 0.00006768159 and the test gap is 0.00048394940, both within 0.005. Position and local-direction agreement favour the selected Dynamic checkpoint; speed remains unresolved.

The completed control contains `checkpoint_candidates.csv`, `checkpoint_selection.json`, `summary.json`, the `response_matching_*` score tables and their provenance records. These contain newly measured checkpoint results. Historical output-noise scores are not used to populate them.

[Verification](verification.json) records the successful full frozen-input check, 218 passed tests across the analysis and encoding environments, byte-identical neural and Static latent arrays, and exact agreement of nine Static metrics. The [NMSE layout check](nmse_layout_verification.json) independently attributes the remaining `9.826e-7` difference from the original main CSV to float32 summation under different memory layouts: consistent C-contiguous layout reproduces the original value exactly. The input data and model parameters are unchanged.
