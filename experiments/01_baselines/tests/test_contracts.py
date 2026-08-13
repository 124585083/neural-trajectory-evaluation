from __future__ import annotations

import numpy as np
import torch

from trajectory_eval.config import load_config
from trajectory_eval.data import DynamicSensoriumDataModule
from trajectory_eval.evaluation.response import pearson_per_neuron
from trajectory_eval.hooks import RepresentationHooks
from trajectory_eval.models import build_model


def test_shared_data_contract_and_replay() -> None:
    config = load_config()
    data = DynamicSensoriumDataModule(config)
    first = data.make_loader("train", batch_size=1)
    second = data.make_loader("train", batch_size=1)
    first.set_epoch(3)
    second.set_epoch(3)
    batch_a = next(iter(first.loader))
    batch_b = next(iter(second.loader))
    assert batch_a["video"].shape[1:] == (3, 80, 36, 64)
    assert batch_a["behavior"].shape[1:] == (2, 80)
    assert batch_a["responses"].shape[1:] == (7495, 80)
    assert torch.equal(batch_a["trial_index"], batch_b["trial_index"])
    assert torch.equal(batch_a["window_offset"], batch_b["window_offset"])


def test_static_is_frame_independent_and_outputs_align() -> None:
    config = load_config()
    data = DynamicSensoriumDataModule(config)
    bundle = data.make_loader("oracle", batch_size=1, shuffle=False)
    batch = next(iter(bundle.loader))
    static, _ = build_model("static_reference", config, data, "cpu")
    dynamic, _ = build_model("dynamic_reference", config, data, "cpu")
    static.eval()
    dynamic.eval()
    with torch.no_grad():
        original = static(batch["video"], batch["behavior"], batch["pupil_center"])
        changed = batch["video"].clone()
        changed[:, :, 0] += 100
        perturbed = static(changed, batch["behavior"], batch["pupil_center"])
        dynamic_output = dynamic(batch["video"], batch["behavior"], batch["pupil_center"])
    # Frame zero is outside the aligned static outputs; no later prediction may change.
    assert torch.equal(original, perturbed)
    assert original.shape == dynamic_output.shape == (1, 62, 7495)


def test_representation_hooks_and_metric() -> None:
    config = load_config()
    data = DynamicSensoriumDataModule(config)
    batch = next(iter(data.make_loader("oracle", batch_size=1, shuffle=False).loader))
    model, _ = build_model("dynamic_reference", config, data, "cpu")
    with RepresentationHooks(model) as hooks, torch.no_grad():
        model(batch["video"], batch["behavior"], batch["pupil_center"])
        assert len(hooks.metadata(80)) == 3
    target = np.arange(30, dtype=np.float32).reshape(10, 3)
    assert np.allclose(pearson_per_neuron(target, target), 1)

