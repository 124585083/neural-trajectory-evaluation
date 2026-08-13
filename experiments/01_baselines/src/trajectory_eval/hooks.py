from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from .models import SensoriumReferenceModel


class RepresentationHooks:
    """Capture layer representations only; no RSA/CKA/trajectory analysis."""

    def __init__(self, model: SensoriumReferenceModel) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, module in model.core.named_modules():
            if name.startswith("features.layer") and isinstance(module, nn.Sequential):
                self.handles.append(module.register_forward_hook(self._capture(f"core.{name}")))

    def _capture(self, name: str):
        def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.outputs[name] = output.detach()

        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def metadata(self, input_time: int) -> list[dict[str, Any]]:
        result = []
        for layer, tensor in self.outputs.items():
            shape = tuple(int(value) for value in tensor.shape)
            if self.model.name == "static_reference":
                result.append({"model": self.model.name, "layer": layer, "time": input_time, "shape": shape})
            else:
                result.append({"model": self.model.name, "layer": layer, "time": shape[2], "shape": shape})
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "RepresentationHooks":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

