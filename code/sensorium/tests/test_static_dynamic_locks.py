import pytest

from trajectory_eval.static_dynamic import (
    EXPECTED_CORE_OUTPUT,
    EXPECTED_CORE_PARAMETERS,
    EXPECTED_NEURONS,
    EXPECTED_SESSIONS,
    TEMPORAL_REDUCTION,
    audit_locked_config,
    load_config,
)


@pytest.mark.data
def test_static_on_dynamic_configuration_is_locked() -> None:
    audit = audit_locked_config(load_config())
    assert audit["status"] == "pass"
    assert tuple(audit["sessions"]) == EXPECTED_SESSIONS
    assert tuple(audit["neuron_counts"]) == EXPECTED_NEURONS
    assert audit["core_parameters"] == EXPECTED_CORE_PARAMETERS
    assert tuple(audit["core_output_per_frame"]) == EXPECTED_CORE_OUTPUT
    assert audit["temporal_adapter_parameters"] == 0
    assert audit["temporal_reduction"] == TEMPORAL_REDUCTION == 18
