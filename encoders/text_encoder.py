"""Text modality encoder using RoBERTa or BERT.

Input: tokenized text dict (input_ids, attention_mask, token_type_ids)
Output: (B, output_dim) sentence embedding
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from encoders.base_encoder import BaseEncoder


class TextEncoder(BaseEncoder):
    """Sentence-level text encoder backed by RoBERTa or BERT.

    Pooling strategies:
        - 'cls'  : Use [CLS] token representation.
        - 'mean' : Mean of all token embeddings (masked).
        - 'max'  : Max-pool over sequence dimension.

    Args:
        config: Dictionary from configs/base_config.yaml under encoders.text
    """

    def __init__(self, config: dict) -> None:
        output_dim = config.get("output_dim", 256)
        dropout = config.get("dropout", 0.3)
        super().__init__(output_dim=output_dim, dropout=dropout)

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("transformers package required for TextEncoder.") from e

        model_name = config.get("model_name", "roberta-base")
        self.pooling = config.get("pooling", "cls")
        self.fine_tune = config.get("fine_tune", False)
        hidden_dim = config.get("hidden_dim", 512)

        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        lm_hidden = self.transformer.config.hidden_size  # 768 for base models

        if not self.fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(lm_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _pool(
        self,
        last_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply pooling strategy over sequence dimension.

        Args:
            last_hidden: (B, T, H) token embeddings.
            attention_mask: (B, T) mask (1 = real token, 0 = padding).

        Returns:
            Pooled tensor (B, H).
        """
        if self.pooling == "cls":
            return last_hidden[:, 0, :]

        if attention_mask is None:
            attention_mask = torch.ones(
                last_hidden.shape[:2], device=last_hidden.device
            )

        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)

        if self.pooling == "mean":
            return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if self.pooling == "max":
            # Replace padding positions with -inf before max
            masked = last_hidden.masked_fill(mask == 0, float("-inf"))
            return masked.max(dim=1).values

        raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def encode(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized text inputs.

        Args:
            x: Dict with 'input_ids', 'attention_mask', and optionally
               'token_type_ids' (for BERT-style models).

        Returns:
            Sentence embedding (B, output_dim).
        """
        outputs = self.transformer(
            input_ids=x["input_ids"],
            attention_mask=x.get("attention_mask"),
            token_type_ids=x.get("token_type_ids"),  # None for RoBERTa
        )
        pooled = self._pool(outputs.last_hidden_state, x.get("attention_mask"))
        return self.proj(pooled)
