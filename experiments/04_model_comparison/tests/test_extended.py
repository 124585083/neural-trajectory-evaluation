import torch

from trajectory_model_eval.extended_predictions import ablate_temporal_state


def test_temporal_ablation_preserves_center_and_scales_history():
    original = {
        "core.layer.conv_temporal.weight": torch.arange(5.0).reshape(1, 1, 5, 1, 1),
        "core.layer.conv_temporal.bias": torch.ones(1),
        "readout.weight": torch.ones(2, 2),
    }
    result, names = ablate_temporal_state(original, 0.25)
    assert names == ["core.layer.conv_temporal.weight"]
    assert result[names[0]][0, 0, 2, 0, 0] == original[names[0]][0, 0, 2, 0, 0]
    assert result[names[0]][0, 0, 4, 0, 0] == original[names[0]][0, 0, 4, 0, 0] * 0.25
    assert torch.equal(result["core.layer.conv_temporal.bias"], original["core.layer.conv_temporal.bias"])

