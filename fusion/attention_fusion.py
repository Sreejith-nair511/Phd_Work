"""Attention-based multimodal fusion (default strategy).

Each available modality embedding is treated as a token; multi-head
self-attention over the token set produces a context-aware fused
representation.  A learnable [FUSE] token aggregates information.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fusion.base_fusion import BaseFusion


class AttentionFusion(BaseFusion):
    """Transformer-style attention fusion over available modality tokens.

    Architecture:
        1. Project each modality embedding → common dim.
        2. Add learnable modality-type embeddings.
        3. Prepend a learnable [FUSE] token.
        4. Run multi-head self-attention.
        5. Output is the [FUSE] token's representation projected to output_dim.

    Args:
        embedding_dim: Input embedding size per modality.
        output_dim: Output representation size.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
    """

    MODALITY_ORDER = ["speech", "text", "eeg", "facial"]

    def __init__(
        self,
        embedding_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(embedding_dim, output_dim)

        # Per-modality input projections → model_dim
        model_dim = embedding_dim
        self.input_proj = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embedding_dim, model_dim),
                    nn.LayerNorm(model_dim),
                )
                for m in self.MODALITY_ORDER
            }
        )

        # Learnable modality-type embeddings (like token-type in BERT)
        self.type_embeddings = nn.Embedding(
            len(self.MODALITY_ORDER) + 1,  # +1 for [FUSE] token
            model_dim,
        )
        self._modal_idx = {m: i + 1 for i, m in enumerate(self.MODALITY_ORDER)}
        self._fuse_idx = 0  # [FUSE] token index

        # Learnable [FUSE] token
        self.fuse_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        # Store attention weights for visualization
        self._last_attn_weights: Optional[torch.Tensor] = None

    def fuse(
        self, embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Attend over available modality tokens and return [FUSE] output.

        Args:
            embeddings: Dict modality → (B, embedding_dim).

        Returns:
            (B, output_dim)
        """
        batch_size = next(iter(embeddings.values())).shape[0]
        device = next(iter(embeddings.values())).device

        tokens: List[torch.Tensor] = []

        # [FUSE] token first
        fuse_tok = self.fuse_token.expand(batch_size, -1, -1)  # (B, 1, D)
        type_emb = self.type_embeddings(
            torch.tensor([self._fuse_idx], device=device)
        )  # (1, D)
        tokens.append(fuse_tok + type_emb.unsqueeze(0))

        # Modality tokens
        for m in self.MODALITY_ORDER:
            emb = embeddings.get(m)
            if emb is None:
                continue
            proj = self.input_proj[m](emb)          # (B, D)
            type_emb = self.type_embeddings(
                torch.tensor([self._modal_idx[m]], device=device)
            )  # (1, D)
            tokens.append((proj + type_emb).unsqueeze(1))  # (B, 1, D)

        x = torch.cat(tokens, dim=1)  # (B, 1+N, D)
        x = self.transformer(x)       # (B, 1+N, D)

        fuse_out = x[:, 0, :]         # [FUSE] token output (B, D)
        return self.out_proj(fuse_out)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last computed attention weights for visualization."""
        return self._last_attn_weights
