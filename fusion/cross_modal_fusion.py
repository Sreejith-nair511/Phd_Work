"""Cross-Modal Attention Fusion.

Each modality attends to every other modality via pairwise cross-attention,
producing richer inter-modal representations before final aggregation.

Architecture per modality pair (A → B):
    query  = modality A embedding
    key/value = modality B embedding
    → cross-attention output enriches A with context from B

All enriched representations are then pooled via self-attention.
"""
from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.base_fusion import BaseFusion


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: query from modality A, key/value from B."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,   # (B, D)
        context: torch.Tensor, # (B, D)
    ) -> torch.Tensor:         # (B, D)
        q = self.norm_q(query).unsqueeze(1)    # (B, 1, D)
        kv = self.norm_kv(context).unsqueeze(1) # (B, 1, D)
        attn_out, _ = self.attn(q, kv, kv)    # (B, 1, D)
        x = query + attn_out.squeeze(1)        # residual
        x = x + self.ffn(self.norm_out(x))
        return x


class CrossModalFusion(BaseFusion):
    """Bidirectional cross-modal attention followed by self-attention pooling.

    For each available modality, it attends to all other available modalities
    (pairwise cross-attention).  The enriched modality representations are then
    aggregated with a self-attention pooling layer.

    Args:
        embedding_dim: Input embedding size per modality.
        output_dim: Output representation size.
        num_heads: Attention heads for both cross- and self-attention.
        dropout: Dropout probability.
    """

    MODALITY_ORDER = ["speech", "text", "eeg", "facial"]

    def __init__(
        self,
        embedding_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(embedding_dim, output_dim)

        # Input projection per modality
        self.input_proj = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )
                for m in self.MODALITY_ORDER
            }
        )

        # Cross-attention blocks: one per ordered pair (A, B) where A ≠ B
        self.cross_attn = nn.ModuleDict()
        for a, b in permutations(self.MODALITY_ORDER, 2):
            key = f"{a}_from_{b}"
            self.cross_attn[key] = CrossAttentionBlock(
                dim=embedding_dim, num_heads=num_heads, dropout=dropout
            )

        # Self-attention aggregation over enriched tokens
        self.agg_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.agg_norm = nn.LayerNorm(embedding_dim)

        # Learnable aggregation query token
        self.query_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def fuse(
        self, embeddings: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Cross-modal attention followed by query-based aggregation.

        Args:
            embeddings: Dict modality → (B, embedding_dim).

        Returns:
            (B, output_dim)
        """
        available = [m for m in self.MODALITY_ORDER if embeddings.get(m) is not None]
        batch_size = embeddings[available[0]].shape[0]
        device = embeddings[available[0]].device

        # Project all available modalities
        projected: Dict[str, torch.Tensor] = {
            m: self.input_proj[m](embeddings[m]) for m in available
        }

        # Cross-attention enrichment
        enriched: Dict[str, torch.Tensor] = {m: projected[m].clone() for m in available}

        for a in available:
            for b in available:
                if a == b:
                    continue
                key = f"{a}_from_{b}"
                enriched[a] = self.cross_attn[key](enriched[a], projected[b])

        # Stack enriched modality tokens → self-attention aggregation
        tokens = torch.stack(
            [enriched[m] for m in available], dim=1
        )  # (B, N, D)

        # Use learnable query token to extract fused representation
        query = self.query_token.expand(batch_size, -1, -1)  # (B, 1, D)
        tokens_norm = self.agg_norm(tokens)
        agg_out, _ = self.agg_attn(query, tokens_norm, tokens_norm)  # (B, 1, D)
        fused = agg_out.squeeze(1)  # (B, D)

        return self.out_proj(fused)
