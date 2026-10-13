import numpy as np
import pytest
import torch

from trajectory_model_eval.unified_predictions import aligned_prediction, predict_loader


def test_alignment_uses_original_frames_instead_of_last_retained_frames():
    output = torch.arange(18, 324).reshape(1, -1, 1)
    selected = aligned_prediction(output, 324, 50, 300)
    assert selected[0, 0, 0] == 50
    assert selected[0, -1, 0] == 299
    with pytest.raises(ValueError, match="reduction"):
        aligned_prediction(output, 325, 50, 300)


def test_frozen_prediction_keeps_batchnorm_and_target_alignment():
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)

        def forward(self, videos, **kwargs):
            values = videos[:, 0, :, 0, :].transpose(1, 2)
            return self.bn(values).transpose(1, 2)[:, 18:]

    model = TinyEncoder().train()
    batch = {"videos": torch.randn(1, 1, 40, 1, 2), "responses": torch.arange(80).reshape(1, 2, 40)}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    prediction, target = predict_loader(model, [batch], "example", np.array([1, 0]), 20, 35, "cpu")
    assert prediction.shape == target.shape == (1, 15, 2)
    np.testing.assert_array_equal(target[0, :, 0], np.arange(60, 75))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert model.training
