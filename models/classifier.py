"""Depression classifier head (MLP)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DepressionClassifier(nn.Module):
    """Multi-layer perceptron classifier for depression detection.

    Supports both binary classification and PHQ-8 score regression.

    Args:
        input_dim: Dimension of incoming fused representation.
        hidden_dims: List of hidden layer sizes.
        num_classes: 2 for binary, 1 for regression.
        dropout: Dropout probability.
        task: 'binary' or 'regression'.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = None,
        num_classes: int = 2,
        dropout: float = 0.4,
        task: str = "binary",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.task = task
        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, input_dim) fused embedding.
        Returns:
            Logits (B, num_classes) or scores (B, 1) for regression.
        """
        return self.net(x)
