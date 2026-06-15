"""Abstract base class for all modality encoders."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Abstract encoder that all modality-specific encoders must inherit.

    Contract:
        - ``forward`` receives raw modality input (tensor or dict).
        - ``forward`` returns a fixed-size embedding of shape ``(B, output_dim)``.
        - If input is ``None`` (missing modality), ``forward`` returns ``None``.
        - ``output_dim`` attribute must be set in ``__init__``.
    """

    def __init__(self, output_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def encode(self, x: Any) -> torch.Tensor:
        """Encode a single modality input to a fixed-size embedding.

        Args:
            x: Modality-specific input tensor or dict.

        Returns:
            Embedding tensor of shape ``(B, output_dim)``.
        """
        ...

    def forward(self, x: Optional[Any]) -> Optional[torch.Tensor]:
        """Wrap encode with NULL modality guard.

        Args:
            x: Input tensor/dict or None for missing modality.

        Returns:
            Embedding of shape ``(B, output_dim)`` or None.
        """
        if x is None:
            return None
        return self.encode(x)

    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim
