import pytest

from trajectory_param_match.experiment import (
    EXPECTED_CHANNELS,
    EXPECTED_PARAMETER_COUNTS,
    STATIC_PARAMETER_COUNTS,
    audit_config,
    audit_data_contract,
    load_config,
)


def test_parameter_matching_and_architecture_config_are_locked() -> None:
    result = audit_config(load_config())
    assert result["status"] == "pass"
    assert tuple(result["channels"]) == EXPECTED_CHANNELS == (16, 32, 64)
    assert result["expected_parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert abs(result["relative_total_difference"]) < 0.05
    assert result["readout_parameter_difference"] == 0


@pytest.mark.data
def test_data_and_split_contract_match_static() -> None:
    result = audit_data_contract(load_config(), save=False)
    assert result["status"] == "pass"
    assert result["same_physical_data_root"] is True
    assert result["static_reference_total_parameters"] == STATIC_PARAMETER_COUNTS["total"]
    assert result["total_train_trials"] == 1744
    assert result["total_oracle_trials"] == 293
    assert result["total_neurons"] == 40034
