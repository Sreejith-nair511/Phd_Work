"""Abstract base class for all fusion strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseFusion(ABC, nn.Module):
    """All fusion modules must inherit this class.

    Contract:
        - ``forward`` accepts a dict mapping modality name → embedding tensor (B, D)
          or None for missing modalities.
        - ``forward`` returns a single fused tensor (B, output_dim).
        - Missing modalities (None values) must be handled gracefully.
    """

    def __init__(self, embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

    @abstractmethod
    def fuse(
        self, embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Fuse multimodal embeddings into a single representation.

        Args:
            embeddings: Dict of modality_name → (B, embedding_dim) or None.

        Returns:
            Fused tensor (B, output_dim).
        """
        ...

    def forward(
        self, embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        # Filter out None embeddings before fusing
        available = {k: v for k, v in embeddings.items() if v is not None}
        if not available:
            raise ValueError("All modalities are None — cannot fuse empty inputs.")
        return self.fuse(available)

    @staticmethod
    def get_available(
        embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Return list of non-None embedding tensors."""
        return [v for v in embeddings.values() if v is not None]
