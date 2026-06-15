"""Speech modality encoder.

Supports two feature extraction backends:
  1. MFCC features (torchaudio)  → BiLSTM encoder
  2. Wav2Vec2 embeddings (HuggingFace) → projection head

Also extracts prosodic features: speech rate, pause duration,
response latency, energy, and pitch.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base_encoder import BaseEncoder


class ProsodicFeatureExtractor(nn.Module):
    """Extracts and projects prosodic/paralinguistic features.

    Input dict keys expected:
        speech_rate, pause_duration, response_latency, energy, pitch
    Each is a tensor of shape (B,) or (B, T) depending on extraction.
    """

    FEATURE_KEYS = ["speech_rate", "pause_duration", "response_latency", "energy", "pitch"]

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        # 5 scalar prosodic features → projected to output_dim
        self.proj = nn.Sequential(
            nn.Linear(len(self.FEATURE_KEYS), output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
        )

    def forward(self, prosodic: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate and project prosodic features.

        Args:
            prosodic: Dict with scalar tensors per feature.

        Returns:
            Tensor of shape (B, output_dim).
        """
        feats = []
        for key in self.FEATURE_KEYS:
            v = prosodic.get(key)
            if v is None:
                # Fill with zeros if feature missing
                sample = next(iter(prosodic.values()))
                v = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
            if v.dim() > 1:
                v = v.mean(dim=-1)  # aggregate over time
            feats.append(v.unsqueeze(1))
        x = torch.cat(feats, dim=1)  # (B, 5)
        return self.proj(x)


class MFCCEncoder(nn.Module):
    """Encode MFCC feature sequences with a BiLSTM.

    Input: (B, T, n_mfcc) feature tensor.
    Output: (B, output_dim)
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        hidden_dim: int = 512,
        num_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, n_mfcc)
        Returns:
            (B, output_dim)
        """
        out, (h, _) = self.lstm(x)
        # Concatenate final forward + backward hidden states
        h_fwd = h[-2]  # (B, hidden//2)
        h_bwd = h[-1]  # (B, hidden//2)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden)
        return self.norm(self.drop(self.proj(h_cat)))


class Wav2Vec2Encoder(nn.Module):
    """Wrap HuggingFace Wav2Vec2 model as an encoder.

    Input: dict with keys:
        'input_values': (B, T) raw waveform float tensor
        'attention_mask': (B, T) optional mask
    Output: (B, output_dim)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        output_dim: int = 256,
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as e:
            raise ImportError("transformers package required for Wav2Vec2Encoder.") from e

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_extractor:
            # Only fine-tune transformer layers, freeze CNN feature extractor
            self.wav2vec2.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec2.config.hidden_size  # typically 768
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Args:
            x: dict with 'input_values' and optionally 'attention_mask'.
        Returns:
            (B, output_dim)
        """
        input_values = x["input_values"]
        attention_mask = x.get("attention_mask")

        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        # Mean-pool over time dimension, respecting mask
        hidden = outputs.last_hidden_state  # (B, T, hidden)
        if attention_mask is not None:
            # Create mask in hidden state time dimension
            # (approximate: wav2vec2 downsamples by ~320x)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)
        return self.proj(pooled)


class SpeechEncoder(BaseEncoder):
    """Top-level speech encoder combining acoustic + prosodic features.

    Args:
        config: Dictionary from configs/base_config.yaml under encoders.speech
    """

    def __init__(self, config: dict) -> None:
        output_dim = config.get("output_dim", 256)
        dropout = config.get("dropout", 0.3)
        super().__init__(output_dim=output_dim, dropout=dropout)

        self.feature_type = config.get("feature_type", "wav2vec2")
        self.extract_prosodic = config.get("extract_prosodic", True)
        hidden_dim = config.get("hidden_dim", 512)
        num_layers = config.get("num_layers", 2)

        # ── Acoustic encoder ──────────────────────────────────────
        if self.feature_type == "mfcc":
            self.acoustic_encoder = MFCCEncoder(
                n_mfcc=config.get("mfcc_n_mfcc", 40),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout,
            )
        elif self.feature_type == "wav2vec2":
            self.acoustic_encoder = Wav2Vec2Encoder(
                model_name=config.get("wav2vec2_model", "facebook/wav2vec2-base-960h"),
                output_dim=output_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        # ── Prosodic encoder ──────────────────────────────────────
        if self.extract_prosodic:
            self.prosodic_encoder = ProsodicFeatureExtractor(output_dim=output_dim)
            # Fusion of acoustic + prosodic
            self.fusion_proj = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def encode(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode speech input.

        Args:
            x: Dict containing:
                - For mfcc: 'mfcc' tensor (B, T, n_mfcc)
                - For wav2vec2: 'input_values' (B, T), optional 'attention_mask'
                - Optional prosodic keys: speech_rate, pause_duration, etc.

        Returns:
            Embedding (B, output_dim)
        """
        # Acoustic encoding
        if self.feature_type == "mfcc":
            acoustic_emb = self.acoustic_encoder(x["mfcc"])
        else:
            acoustic_emb = self.acoustic_encoder(x)

        if self.extract_prosodic:
            prosodic_emb = self.prosodic_encoder(x)
            combined = torch.cat([acoustic_emb, prosodic_emb], dim=-1)
            return self.fusion_proj(combined)

        return acoustic_emb
