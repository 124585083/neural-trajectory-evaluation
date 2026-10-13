import numpy as np

from trajectory_model_eval.traditional import evaluate_traditional, linear_cka


def test_identical_prediction_scores_one_on_similarity_metrics():
    rng = np.random.default_rng(1)
    neural = rng.normal(size=(12, 20, 16))
    conditions = np.repeat(np.arange(3), 4)
    metrics, per_neuron, per_sample = evaluate_traditional(
        neural, neural.copy(), conditions, cka_max_samples=100
    )
    assert np.nanmin(per_neuron) > 0.999999
    assert np.nanmin(per_sample) > 0.999999
    assert metrics["cka_condition_average_time_aligned"] > 0.999999
    assert metrics["rsa_condition_time_state_spearman"] > 0.999999


def test_cka_is_invariant_to_orthogonal_feature_rotation():
    rng = np.random.default_rng(2)
    left = rng.normal(size=(100, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert linear_cka(left, left @ q) > 0.999999

