"""Small CPU execution checks using the actual adapter and pinned 2D core."""

from collections import namedtuple
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from trajectory_eval.static_dynamic import audit_frame_independence, build_model, load_config


@pytest.fixture(scope="module")
def synthetic_static():
    torch.manual_seed(42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    batch_type = namedtuple("Batch", "videos responses behavior pupil_center")
    behavior = torch.randn(1, 2, 27)
    video = torch.randn(1, 3, 27, 36, 64)
    video[:, 1:] = behavior[:, :, :, None, None]
    batch = batch_type(video, torch.rand(1, 7, 27), behavior, torch.randn(1, 2, 27))

    class Loader:
        dataset = SimpleNamespace(neurons=SimpleNamespace(cell_motor_coordinates=np.random.default_rng(42).normal(size=(7, 3))))

        def __iter__(self):
            return iter([batch])

    model = build_model(load_config(), {"train": {"synthetic": Loader()}}).eval()
    yield model, batch
    torch.set_num_threads(previous_threads)


def test_actual_static_adapter_permutation_equivariance(synthetic_static):
    model, batch = synthetic_static
    permutation = torch.randperm(27, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        expected = model.predict_all_frames(batch.videos, data_key="synthetic", pupil_center=batch.pupil_center)
        observed = model.predict_all_frames(batch.videos[:, :, permutation], data_key="synthetic", pupil_center=batch.pupil_center[:, :, permutation])[:, torch.argsort(permutation)]
    torch.testing.assert_close(observed, expected, atol=2e-6, rtol=0)


def test_actual_static_adapter_frame_and_covariate_independence(synthetic_static, record_property):
    model, batch = synthetic_static
    result = audit_frame_independence(model, batch.videos, data_key="synthetic", pupil_center=batch.pupil_center, behavior=batch.behavior, positions=(18, 22, 26))
    record_property("frame_independence", json.dumps(result, sort_keys=True))
    assert len(result["checks"]) == 15
    assert result["absolute_tolerance"] == 2e-6
    assert max(row["max_other_frame_error"] for row in result["checks"]) <= 2e-6
    for pathway in ("input_channel_0", "input_channel_1", "input_channel_2", "pupil_center"):
        assert any(row["max_changed_frame_effect"] > 2e-6 for row in result["checks"] if row["pathway"] == pathway)
    assert all(row["max_changed_frame_effect"] == 0 for row in result["checks"] if row["pathway"] == "behavior_argument")
    assert not model.training


class SequenceMeanCoupled(nn.Module):
    """A permutation-equivariant counterexample with cross-frame dependence."""

    temporal_reduction = 18

    def predict_all_frames(self, inputs, **kwargs):
        values = inputs.mean(dim=(1, 3, 4)).unsqueeze(-1)
        return values + values.mean(dim=1, keepdim=True)

    def forward(self, inputs, **kwargs):
        return self.predict_all_frames(inputs, **kwargs)[:, self.temporal_reduction:]


def test_coupled_counterexample_passes_equivariance_but_fails_independence():
    model = SequenceMeanCoupled().eval()
    values = torch.arange(27, dtype=torch.float32).reshape(1, 1, 27, 1, 1)
    permutation = torch.randperm(27, generator=torch.Generator().manual_seed(42))
    expected = model.predict_all_frames(values)
    actual = model.predict_all_frames(values[:, :, permutation])[:, torch.argsort(permutation)]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
    with pytest.raises(AssertionError, match="frame independence failed"):
        audit_frame_independence(model, values, data_key="synthetic", pupil_center=torch.zeros(1, 2, 27))


def test_independence_rejects_cropped_positions(synthetic_static):
    model, batch = synthetic_static
    with pytest.raises(ValueError, match="retained input frames"):
        audit_frame_independence(model, batch.videos, data_key="synthetic", pupil_center=batch.pupil_center, positions=(0,))
