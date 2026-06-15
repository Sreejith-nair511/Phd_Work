"""Late Fusion: independent classifier per modality, then ensemble."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.base_fusion import BaseFusion


class LateFusion(BaseFusion):
    """Each modality produces its own logit-level prediction; outputs are
    averaged (or learned-weighted) across available modalities.

    The ``fuse`` method returns a combined representation (not logits) so it
    integrates naturally with the shared classifier head.

    Args:
        embedding_dim: Input embedding size per modality.
        output_dim: Output representation size.
        dropout: Dropout probability.
        learnable_weights: If True, learn per-modality importance weights.
    """

    MODALITY_ORDER = ["speech", "text", "eeg", "facial"]

    def __init__(
        self,
        embedding_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3,
        learnable_weights: bool = True,
    ) -> None:
        super().__init__(embedding_dim, output_dim)
        self.learnable_weights = learnable_weights

        # Per-modality projection head
        self.heads = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embedding_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for m in self.MODALITY_ORDER
            }
        )

        if learnable_weights:
            # One scalar weight per modality; softmax over available ones
            self.modal_weights = nn.ParameterDict(
                {m: nn.Parameter(torch.ones(1)) for m in self.MODALITY_ORDER}
            )

    def fuse(
        self, embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Average (weighted) modality-specific projections.

        Args:
            embeddings: Dict modality → (B, embedding_dim).

        Returns:
            (B, output_dim)
        """
        available_modalities = [m for m in self.MODALITY_ORDER if embeddings.get(m) is not None]

        projected: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []

        for m in available_modalities:
            proj = self.heads[m](embeddings[m])  # (B, output_dim)
            projected.append(proj)
            if self.learnable_weights:
                weights.append(self.modal_weights[m])

        stacked = torch.stack(projected, dim=1)  # (B, N, output_dim)

        if self.learnable_weights:
            w = torch.cat(weights)            # (N,)
            w = F.softmax(w, dim=0)           # normalise
            w = w.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
            fused = (stacked * w).sum(dim=1)  # (B, output_dim)
        else:
            fused = stacked.mean(dim=1)

        return fused
