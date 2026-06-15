"""Early Fusion: concatenate all available embeddings then project."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fusion.base_fusion import BaseFusion


class EarlyFusion(BaseFusion):
    """Concatenates all available modality embeddings and projects to output_dim.

    Because the number of available modalities can vary at runtime, we use
    a per-modality projection to a common dim first, then concatenate and
    project through an MLP.  This keeps the downstream MLP dimension fixed
    regardless of how many modalities are present.

    Args:
        embedding_dim: Input embedding size per modality.
        output_dim: Output representation size.
        num_modalities: Maximum number of modalities (used to size MLP).
        dropout: Dropout probability.
    """

    MODALITY_ORDER = ["speech", "text", "eeg", "facial"]

    def __init__(
        self,
        embedding_dim: int = 256,
        output_dim: int = 256,
        num_modalities: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(embedding_dim, output_dim)

        # Per-modality projection to a common intermediate dim
        self.modality_proj = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                )
                for m in self.MODALITY_ORDER
            }
        )

        # Final MLP — input is concatenation of projected embeddings
        max_concat_dim = embedding_dim * num_modalities
        self.mlp = nn.Sequential(
            nn.Linear(max_concat_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._max_concat_dim = max_concat_dim
        self._embedding_dim = embedding_dim

    def fuse(
        self, embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Concatenate available projections with zero-padding for missing.

        Args:
            embeddings: Dict modality → (B, embedding_dim) or None.

        Returns:
            (B, output_dim)
        """
        batch_size = next(v for v in embeddings.values() if v is not None).shape[0]
        device = next(v for v in embeddings.values() if v is not None).device

        parts: List[torch.Tensor] = []
        for m in self.MODALITY_ORDER:
            emb = embeddings.get(m)
            if emb is not None:
                parts.append(self.modality_proj[m](emb))
            else:
                # Zero-pad for missing modality to keep fixed concat size
                parts.append(
                    torch.zeros(batch_size, self._embedding_dim, device=device)
                )

        concat = torch.cat(parts, dim=-1)  # (B, embedding_dim * 4)
        return self.mlp(concat)
