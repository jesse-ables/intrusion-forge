import torch
import torch.nn.functional as F
from torch import Tensor

from . import LossFactory
from .base import BaseLoss


def _make_class_weight(
    class_weight: Tensor | list[float] | None, device: torch.device
) -> Tensor | None:
    if class_weight is None:
        return None
    if not isinstance(class_weight, torch.Tensor):
        class_weight = torch.tensor(list(class_weight), dtype=torch.float32)
    return class_weight.to(device=device)


@LossFactory.register()
class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for classification with hard labels."""

    def __init__(
        self,
        *,
        reduction: str = "mean",
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        class_weight: Tensor | list[float] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(reduction=reduction)
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        w = _make_class_weight(class_weight, device)
        if w is not None:
            self.register_buffer("class_weight", w)
        else:
            self.class_weight = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(
            x,
            target,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        return self._reduce(loss[target != self.ignore_index])


@LossFactory.register()
class FocalLoss(BaseLoss):
    """Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(
        self,
        *,
        class_weight: Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(reduction=reduction)
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        w = _make_class_weight(class_weight, device)
        if w is not None:
            self.register_buffer("class_weight", w)
        else:
            self.class_weight = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(
            x,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )
        p_t = torch.softmax(x, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.class_weight is not None:
            loss = self.class_weight.gather(0, target) * loss
        return self._reduce(loss[target != self.ignore_index])
