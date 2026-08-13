from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from neuralpredictors.layers.cores import Stacked2dCore
from neuralpredictors.layers.cores.conv3d import Factorized3dCore
from neuralpredictors.layers.readouts import FullGaussian2d
from neuralpredictors.layers.shifters import MLPShifter
from torch import nn

from .data import DynamicSensoriumDataModule


ModelName = Literal["dynamic_reference", "static_reference"]


@dataclass(frozen=True)
class ModelShapes:
    input: tuple[int, int, int, int]
    core_output: tuple[int, int, int]
    output_time: int
    temporal_reduction: int


class SensoriumReferenceModel(nn.Module):
    def __init__(
        self,
        *,
        name: ModelName,
        core: nn.Module,
        readout: FullGaussian2d,
        shifter: MLPShifter,
        session_key: str,
        temporal_reduction: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.session_key = session_key
        self.temporal_reduction = int(temporal_reduction)
        self.output_nonlinearity = nn.ELU()

    def _readout(self, features: torch.Tensor, pupil: torch.Tensor) -> torch.Tensor:
        shifts = self.shifter[self.session_key](pupil)
        rates = self.readout(features, shift=shifts)
        return self.output_nonlinearity(rates) + 1.0

    def forward(
        self,
        video: torch.Tensor,
        behavior: torch.Tensor | None = None,
        pupil_center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        if pupil_center is None:
            raise ValueError("pupil_center is required by the official MLP shifter")
        batch, _, time_points, _, _ = video.shape

        if self.name == "dynamic_reference":
            features = self.core(video)
            output_time = features.shape[2]
            flat_features = features.transpose(1, 2).reshape(-1, *features.shape[1:2], *features.shape[3:])
            flat_pupil = pupil_center[:, :, -output_time:].transpose(1, 2).reshape(-1, pupil_center.shape[1])
            rates = self._readout(flat_features, flat_pupil)
            return rates.reshape(batch, output_time, -1)

        # Every frame is an independent sample. No operation below mixes time.
        flat_video = video.transpose(1, 2).reshape(batch * time_points, video.shape[1], video.shape[3], video.shape[4])
        features = self.core(flat_video)
        flat_pupil = pupil_center.transpose(1, 2).reshape(batch * time_points, pupil_center.shape[1])
        rates = self._readout(features, flat_pupil).reshape(batch, time_points, -1)
        return rates[:, self.temporal_reduction :, :]

    def regularizer(self) -> torch.Tensor:
        core_reg = self.core.regularizer()
        if isinstance(core_reg, tuple):
            core_reg = sum(core_reg)
        return core_reg + self.readout.regularizer(reduction="sum") + self.shifter.regularizer(self.session_key)


def _core_output_shape(core: nn.Module, input_tensor: torch.Tensor, dynamic: bool) -> tuple[int, int, int, int]:
    was_training = core.training
    core.eval()
    with torch.no_grad():
        output = core(input_tensor)
    core.train(was_training)
    if dynamic:
        return tuple(int(v) for v in output.shape[1:])
    return (int(output.shape[1]), 1, int(output.shape[2]), int(output.shape[3]))


def build_model(
    name: ModelName,
    config: dict,
    data: DynamicSensoriumDataModule,
    device: torch.device | str = "cpu",
) -> tuple[SensoriumReferenceModel, ModelShapes]:
    torch.manual_seed(int(config["project"]["seed"]))
    channels, height, width = data.metadata.input_shape
    frames = int(config["data"]["frames"])

    if name == "dynamic_reference":
        cfg = config[name]
        core = Factorized3dCore(
            input_channels=int(cfg["input_channels"]),
            hidden_channels=[int(v) for v in cfg["hidden_channels"]],
            spatial_input_kernel=tuple(cfg["spatial_input_kernel"]),
            temporal_input_kernel=int(cfg["temporal_input_kernel"]),
            spatial_hidden_kernel=tuple(cfg["spatial_hidden_kernel"]),
            temporal_hidden_kernel=int(cfg["temporal_hidden_kernel"]),
            stride=int(cfg["stride"]),
            layers=int(cfg["layers"]),
            gamma_input_spatial=float(cfg["gamma_input_spatial"]),
            gamma_input_temporal=float(cfg["gamma_input_temporal"]),
            bias=bool(cfg["bias"]),
            hidden_nonlinearities=cfg["hidden_nonlinearities"],
            batch_norm=bool(cfg["batch_norm"]),
            padding=bool(cfg["padding"]),
            final_nonlin=bool(cfg["final_nonlin"]),
            momentum=float(cfg["momentum"]),
            laplace_padding=None,
            input_regularizer="LaplaceL2norm",
        )
        dummy = torch.zeros(1, channels, frames, height, width)
        core_shape = _core_output_shape(core, dummy, dynamic=True)
        readout_shape = (core_shape[0], core_shape[2], core_shape[3])
        output_time = core_shape[1]
    elif name == "static_reference":
        cfg = config[name]
        core = Stacked2dCore(
            input_channels=int(cfg["input_channels"]),
            hidden_channels=int(cfg["hidden_channels"]),
            input_kern=int(cfg["input_kernel"]),
            hidden_kern=int(cfg["hidden_kernel"]),
            layers=int(cfg["layers"]),
            gamma_input=float(cfg["gamma_input"]),
            gamma_hidden=0,
            skip=0,
            final_nonlinearity=bool(cfg["final_nonlinearity"]),
            bias=True,
            momentum=float(cfg["momentum"]),
            pad_input=bool(cfg["pad_input"]),
            batch_norm=bool(cfg["batch_norm"]),
            hidden_dilation=1,
            laplace_padding=None,
            input_regularizer="LaplaceL2norm",
            stack=int(cfg["stack"]),
            depth_separable=bool(cfg["depth_separable"]),
            linear=False,
            attention_conv=False,
            hidden_padding=None,
            use_avg_reg=False,
        )
        dummy = torch.zeros(1, channels, height, width)
        core_shape = _core_output_shape(core, dummy, dynamic=False)
        readout_shape = (core_shape[0], core_shape[2], core_shape[3])
        # Crop static predictions to the exact output timepoints of the valid 3D core.
        dyn = config["dynamic_reference"]
        reduction = int(dyn["temporal_input_kernel"]) - 1
        reduction += (int(dyn["layers"]) - 1) * (int(dyn["temporal_hidden_kernel"]) - 1)
        output_time = frames - reduction
    else:
        raise ValueError(f"unknown model {name}")

    temporal_reduction = frames - output_time
    readout_cfg = config["shared_readout"]
    source_grid = np.asarray(data.metadata.cell_motor_coordinates[:, :2], dtype=np.float32)
    readout = FullGaussian2d(
        in_shape=readout_shape,
        outdims=data.n_neurons,
        bias=bool(readout_cfg["bias"]),
        init_mu_range=float(readout_cfg["init_mu_range"]),
        init_sigma=float(readout_cfg["init_sigma"]),
        gamma_readout=float(readout_cfg["gamma_readout"]),
        gauss_type=readout_cfg["gauss_type"],
        grid_mean_predictor=dict(readout_cfg["grid_mean_predictor"]),
        source_grid=source_grid,
        mean_activity=data.reference_mean_activity(),
    )
    shift_cfg = readout_cfg["shifter"]
    shifter = MLPShifter(
        data_keys=[data.session_key],
        input_channels=int(shift_cfg["input_channels"]),
        hidden_channels_shifter=int(shift_cfg["hidden_channels"]),
        shift_layers=int(shift_cfg["layers"]),
        gamma_shifter=float(shift_cfg["gamma"]),
    )
    model = SensoriumReferenceModel(
        name=name,
        core=core,
        readout=readout,
        shifter=shifter,
        session_key=data.session_key,
        temporal_reduction=temporal_reduction,
    ).to(device)
    shapes = ModelShapes(
        input=(channels, frames, height, width),
        core_output=(core_shape[0], core_shape[2], core_shape[3]),
        output_time=output_time,
        temporal_reduction=temporal_reduction,
    )
    return model, shapes


def parameter_counts(model: SensoriumReferenceModel) -> dict[str, int]:
    core = sum(p.numel() for p in model.core.parameters())
    readout = sum(p.numel() for p in model.readout.parameters())
    shifter = sum(p.numel() for p in model.shifter.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"core": core, "readout": readout, "other": shifter, "total": total}

