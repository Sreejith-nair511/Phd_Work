"""Loss functions for depression detection."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for regularisation.

    Args:
        smoothing: Smoothing factor in [0, 1).
        weight: Optional class weights tensor.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (B, C) unnormalised logits.
            targets: (B,) integer class labels.
        Returns:
            Scalar loss.
        """
        n_classes = logits.shape[-1]
        log_prob = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_prob, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        if self.weight is not None:
            w = self.weight[targets].unsqueeze(-1)
            loss = -(smooth_targets * log_prob * w).sum(dim=-1).mean()
        else:
            loss = -(smooth_targets * log_prob).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling severe class imbalance.

    Args:
        alpha: Weighting factor for the rare class.
        gamma: Focusing parameter.
        weight: Optional class weights tensor.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


def get_loss_fn(
    task: str = "binary",
    loss_type: str = "label_smoothing",
    smoothing: float = 0.1,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Return a loss function given task and type.

    Args:
        task: 'binary' or 'regression'.
        loss_type: 'ce', 'label_smoothing', 'focal'.
        smoothing: Label smoothing factor (for label_smoothing type).
        class_weights: Optional class weight tensor.

    Returns:
        Loss module.
    """
    if task == "regression":
        return nn.MSELoss()

    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
    elif loss_type == "focal":
        return FocalLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
