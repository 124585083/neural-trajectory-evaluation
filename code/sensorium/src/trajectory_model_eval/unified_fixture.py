"""Small artificial data for testing the part-three execution contract."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .unified_contract import canonical_digest, manifest_digest


def synthetic_fixture(config: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Keep q=4; reduce data, EM iterations and resampling for a smoke test.

    Predictions are constructed arrays, not trained Sensorium encoders. Nothing
    produced here is evidence about Static or Dynamic performance on mice.
    """
    config = deepcopy(config)
    session = "synthetic_session"
    config["data"].update(session=session, sessions=[session], frame_start=0,
                          frame_stop=20, neurons=8)
    config["gpfa"].update(selection_em_iterations=2, final_em_iterations=3,
                          initial_lengthscales_seconds=[0.125, 0.5])
    config["reliability"].update(splits=4, block_sizes_frames=[2, 4])
    config["model_nulls"].update(permutations=4, block_size_frames=4)
    config["traditional"].update(paired_bootstrap_samples=20, temporal_rsa_stride=2)
    config["response_matching"]["sigma_grid"] = [0.0, 0.1, 0.3, 1.0]
    config["project"]["synthetic"] = True
    rng = np.random.default_rng(9182)
    t = np.arange(20) / 30
    loading = rng.normal(0, 0.3, (4, 8))
    precision = 0.8 + 0.3 * rng.random((20, 8))
    rows, neural, static, dynamic = [], [], [], []
    for role, conditions, repeats, tier in (("train", 8, 2, "train"),
                                           ("validation", 3, 4, "oracle"),
                                           ("test", 4, 4, "heldout")):
        for condition in range(conditions):
            phase = rng.uniform(-np.pi, np.pi, 4)
            latent = np.sin(t[:, None] * np.array([2, 4, 6, 8]) + phase)
            mean = 3 + latent @ loading
            for repeat in range(repeats):
                index = len(rows)
                rows.append({"session": session, "dataset_index": index,
                             "trial_id": f"artificial_trial_{index}",
                             "movie_id": f"{role}_movie_{condition}", "role": role,
                             "source_tier": tier, "response_status": "synthetic",
                             "modalities_complete": True})
                neural.append((mean + rng.normal(0, 0.12, mean.shape)) * precision)
                static.append((np.roll(mean, 2, axis=0) + rng.normal(0, 0.2, mean.shape)) * precision)
                dynamic.append((mean + rng.normal(0, 0.22, mean.shape)) * precision)
    checkpoints = {
        name: {"sha256": canonical_digest(f"artificial_{name}"),
               "trained_sessions": [session], "training_tiers": ["train"],
               "selection_tiers": ["oracle"], "synthetic": True}
        for name in ("static", "dynamic")}
    manifest = {
        "schema_version": "sensorium_part3_v1", "synthetic": True,
        "config_sha256": canonical_digest(config), "session": session,
        "evaluation_scope": "unseen_movie", "checkpoint_identities": checkpoints,
        "preprocessing": {"status": "verified", "provenance": "Constructed positive artificial scale; no real data.",
                          "sha256": canonical_digest(precision)},
        "neurons": {"indices": list(range(8)), "ids": list(range(100, 108))},
        "time": {"frame_start": 0, "frame_stop": 20, "frame_rate_hz": 30.0, "observation_step": 4},
        "trials": rows, "eligibility": {"status": "ready", "reasons": []},
    }
    manifest["split_sha256"] = manifest_digest(manifest)
    bundle = {"neural": np.asarray(neural), "static": np.asarray(static), "dynamic": np.asarray(dynamic),
              "session": np.asarray(session), "encoder_precision": precision,
              "dataset_indices": np.asarray([r["dataset_index"] for r in rows]),
              "trial_ids": np.asarray([r["trial_id"] for r in rows]),
              "movie_ids": np.asarray([r["movie_id"] for r in rows]),
              "conditions": np.asarray([r["movie_id"] for r in rows]),
              "roles": np.asarray([r["role"] for r in rows]),
              "neuron_indices": np.arange(8), "neuron_ids": np.arange(100, 108),
              "frame_indices": np.arange(20)}
    return bundle, manifest, config
