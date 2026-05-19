from collections.abc import Sequence

import torch
from torch import nn, Tensor
import numpy as np


class ModelOutput(dict):
    """Dict subclass that enforces Tensor values."""

    def __init__(self, *, data: dict[str, Tensor] | None = None, **kwargs):
        if data is None:
            data = kwargs
        elif kwargs:
            raise ValueError("Cannot use both 'data' argument and keyword arguments.")
        for key, value in data.items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"ModelOutput['{key}'] must be a Tensor, got {type(value)}."
                )
        super().__init__(data)

    def __setitem__(self, key: str, value: Tensor):
        if not isinstance(value, Tensor):
            raise TypeError(
                f"ModelOutput['{key}'] must be a Tensor, got {type(value)}."
            )
        super().__setitem__(key, value)

    def detach(self) -> "ModelOutput":
        return ModelOutput(data={k: v.detach() for k, v in self.items()})

    def to(self, device: torch.device, *, non_blocking: bool = True) -> "ModelOutput":
        return ModelOutput(
            data={k: v.to(device, non_blocking=non_blocking) for k, v in self.items()}
        )

    def numpy(self) -> dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in self.items()}


def cat_model_outputs(outputs: Sequence[ModelOutput], *, dim: int = 0) -> ModelOutput:
    """Concatenate a sequence of ModelOutputs along a dimension."""
    if not outputs:
        raise ValueError("The outputs sequence is empty.")
    return ModelOutput(
        data={k: torch.cat([o[k] for o in outputs], dim=dim) for k in outputs[0]}
    )


class BaseModel(nn.Module):
    """Base class for models."""

    def forward(self, x: Tensor) -> ModelOutput:
        raise NotImplementedError

    def for_loss(self, output: ModelOutput, target: Tensor) -> tuple[Tensor, Tensor]:
        """Prepare (prediction, target) for the loss function. Override as needed."""
        return output["logits"], target


class ComposableClassifier(BaseModel):
    """Classifier composed of an encoder + linear head."""

    def __init__(self, encoder_module: nn.Module, head_module: nn.Module) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.head_module = head_module

    def forward(self, x: Tensor) -> ModelOutput:
        z = self.encoder_module(x)
        return ModelOutput(logits=self.head_module(z), z=z)

    def for_loss(
        self, output: ModelOutput, target: Tensor, *args
    ) -> tuple[Tensor, ...]:
        return (output["logits"], target, *args)


class ComposableTabularClassifier(ComposableClassifier):
    """Classifier for tabular (numerical + categorical) input."""

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> ModelOutput:
        z = self.encoder_module(x_numerical, x_categorical)
        return ModelOutput(logits=self.head_module(z), z=z)
