import copy

import numpy as np
import pytest

from trajectory_model_eval.supplementary_gpfa import (
    NULL_SPECS, reconstruction_scores, sampling_plan, training_indices,
    verify_additive_guard,
)
from trajectory_model_eval.unified_contract import canonical_digest
from trajectory_reliability.reliability import _balanced_half, _null_response


def test_sampling_exactly_replays_main_rng_and_disjoint_nested_sides():
    counts = [4, 5, 6]
    plan = sampling_plan(counts, n_times=40, n_neurons=5, n_splits=3)
    rng = np.random.default_rng(42)
    value = np.arange(3*40*5).reshape(3, 40, 5)
    for item in plan:
        main = [_balanced_half(c, rng) for c in counts]
        for saved, (a,b) in zip(item["halves"], main):
            assert np.array_equal(np.sort(saved["a"]), a)
            assert np.array_equal(np.sort(saved["b"]), b)
            assert not set(saved["a"]) & set(saved["b"])
            for k in (1, 2):
                assert len(saved["a"][:k]) == len(saved["b"][:k]) == k
                assert not set(saved["a"][:k]) & set(saved["b"][:k])
        for label, name, block in NULL_SPECS:
            replay = np.random.default_rng()
            replay.bit_generator.state = copy.deepcopy(item["null_rng_states"][label])
            assert np.array_equal(_null_response(value, name, rng, block),
                                  _null_response(value, name, replay, block))


def test_reconstruction_denominators_and_undefined_are_explicit():
    x = np.asarray([[[1., 1.], [2., 1.], [3., 1.]]])
    scores, neurons = reconstruction_scores(x, x.copy(), np.zeros(2))
    assert scores["population_r2_evaluation_mean"] == 1
    assert scores["population_r2_train_intercept"] == 1
    assert neurons[0] == 1 and np.isnan(neurons[1])
    assert scores["defined_neurons"] == 1
    scores, _ = reconstruction_scores(x, np.zeros_like(x), np.zeros(2))
    assert scores["population_r2_train_intercept"] == 0
    assert scores["population_r2_evaluation_mean"] < 0
    with pytest.raises(ValueError):
        reconstruction_scores(x, np.zeros((1, 3, 1)), np.zeros(2))


def test_reduced_training_membership_never_uses_validation_or_test():
    bundle = {"roles": np.asarray(["train"]*10 + ["validation"]*3 + ["test"]*5)}
    a, validation = training_indices(bundle, .5)
    b, _ = training_indices(bundle, .5)
    assert np.array_equal(a, b) and len(a) == 5
    assert np.array_equal(validation, [10, 11, 12])
    assert np.all(a < 10)
    changed = {"roles": bundle["roles"].copy(), "neural": np.random.default_rng(1).normal(size=18)}
    assert np.array_equal(training_indices(changed, .5)[0], a)


def test_gpfa_query_is_linear_and_does_not_mutate_parameters():
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    rng = np.random.default_rng(4)
    train = rng.normal(size=(6, 6, 5))
    model = GaussianProcessFactorAnalysis(2, np.arange(6)/30, max_em_iterations=2,
                                         random_seed=42).fit(train, split_name="train")
    digest = model.parameter_digest()
    queries = np.arange(11)/60
    full = model.transform_query(train, queries)
    average = model.transform_query(train.mean(0, keepdims=True), queries)
    assert np.allclose(full.mean(0), average[0], atol=1e-10)
    assert model.parameter_digest() == digest


def test_additive_guard_rejects_changed_development_and_main_table_sources():
    protocol = {"seed": 42}
    sources = {"development/cache.npz": "old", "main/reliability.csv": "same"}
    saved = {"supplementary_config_sha256": canonical_digest(protocol), "sources": sources}
    verify_additive_guard(saved, dict(sources), protocol)
    for key in sources:
        changed = {**sources, key: "replacement"}
        with pytest.raises(ValueError, match="changed"):
            verify_additive_guard(saved, changed, protocol)
    with pytest.raises(ValueError, match="changed"):
        verify_additive_guard(saved, sources, {"seed": 43})
