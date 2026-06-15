"""EEG modality encoder.

Supports three backbone architectures:
  1. CNN  : 1-D convolutional blocks over time dimension.
  2. Transformer: Multi-head self-attention over EEG channels/time patches.
  3. BiLSTM: Bidirectional LSTM over the temporal sequence.

Input shape: (B, C, T)  – C = EEG channels, T = time samples
Output: (B, output_dim)
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base_encoder import BaseEncoder


# ─── Sub-architectures ───────────────────────────────────────────

class EEG_CNN(nn.Module):
    """1-D CNN feature extractor for EEG signals."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, T) → (B, output_dim)"""
        return self.proj(self.conv_blocks(x))


class EEG_BiLSTM(nn.Module):
    """BiLSTM encoder for EEG time-series."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, T) – channels treated as features, T as time steps.
        Transposes to (B, T, C) for LSTM.
        Returns: (B, output_dim)
        """
        x = x.permute(0, 2, 1)  # (B, T, C)
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden_dim)
        return self.proj(h_cat)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EEG_Transformer(nn.Module):
    """Transformer encoder for EEG channel-time representations."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Project EEG channels to model dimension
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, T)
        Returns: (B, output_dim)
        """
        x = x.permute(0, 2, 1)           # (B, T, C)
        x = self.input_proj(x)            # (B, T, hidden)
        x = self.pos_enc(x)
        x = self.transformer(x)           # (B, T, hidden)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # (B, hidden)
        return self.proj(x)


# ─── Top-level encoder ───────────────────────────────────────────

class EEGEncoder(BaseEncoder):
    """EEG modality encoder.

    Args:
        config: Dictionary from configs/base_config.yaml under encoders.eeg
    """

    def __init__(self, config: dict) -> None:
        output_dim = config.get("output_dim", 256)
        dropout = config.get("dropout", 0.3)
        super().__init__(output_dim=output_dim, dropout=dropout)

        enc_type = config.get("encoder_type", "transformer")
        in_channels = config.get("input_channels", 64)
        hidden_dim = config.get("hidden_dim", 512)
        num_layers = config.get("num_layers", 4)
        num_heads = config.get("num_heads", 8)

        if enc_type == "cnn":
            self.backbone = EEG_CNN(in_channels, hidden_dim, output_dim, dropout)
        elif enc_type == "bilstm":
            self.backbone = EEG_BiLSTM(in_channels, hidden_dim, num_layers, output_dim, dropout)
        elif enc_type == "transformer":
            self.backbone = EEG_Transformer(
                in_channels, hidden_dim, num_heads, num_layers, output_dim, dropout
            )
        else:
            raise ValueError(f"Unknown EEG encoder_type: {enc_type}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG tensor.

        Args:
            x: EEG tensor of shape (B, C, T).

        Returns:
            Embedding (B, output_dim).
        """
        return self.backbone(x)
