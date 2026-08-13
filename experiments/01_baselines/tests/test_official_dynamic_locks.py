from trajectory_eval.official_dynamic import (
    EXPECTED_NEURONS,
    EXPECTED_PARAMETER_COUNTS,
    EXPECTED_SESSIONS,
    audit_architecture,
    load_official_config,
)


def test_full_official_dynamic_architecture_is_locked() -> None:
    audit = audit_architecture(load_official_config(), save=False)
    assert audit["status"] == "pass"
    assert tuple(audit["sessions"]) == EXPECTED_SESSIONS
    assert tuple(audit["neuron_counts"]) == EXPECTED_NEURONS
    assert audit["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert audit["input_shape"] == [3, 80, 36, 64]
    assert audit["output_shape"] == [62, 7863]

