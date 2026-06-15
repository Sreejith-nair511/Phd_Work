"""Facial modality encoder.

Supports:
  1. CNN backbone (ResNet-inspired) for image sequences.
  2. Vision Transformer (ViT) for image sequences.
  3. Landmark-based MLP encoder.
  4. Hybrid: landmark + image.

Input options:
  - 'frames'    : (B, T, C, H, W) video frame tensor
  - 'landmarks' : (B, T, L*2) flattened 2D landmark coordinates
  - Both simultaneously (hybrid)

Output: (B, output_dim)
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base_encoder import BaseEncoder


# ─── Sub-architectures ───────────────────────────────────────────

class LandmarkEncoder(nn.Module):
    """MLP encoder for facial landmark sequences."""

    def __init__(
        self,
        num_landmarks: int = 68,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = num_landmarks * 2  # x, y per landmark
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, num_landmarks*2) landmark sequences.
        Returns:
            (B, output_dim) – mean over time.
        """
        # Process each frame, then aggregate
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        out = self.mlp(x_flat).view(B, T, -1)
        return out.mean(dim=1)


class FrameCNNEncoder(nn.Module):
    """CNN encoder applied to each video frame, then temporal pooling."""

    def __init__(
        self,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Lightweight CNN inspired by ResNet blocks
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, C, H, W) frame sequence.
        Returns:
            (B, output_dim).
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.cnn(x_flat)
        feats = self.proj(feats).view(B, T, -1)
        return feats.mean(dim=1)  # temporal mean pooling


class ViTFrameEncoder(nn.Module):
    """Vision Transformer encoder applied to video frames.

    Uses PyTorch's built-in or timm ViT. Falls back to CNN if timm unavailable.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            import timm
            self.vit = timm.create_model(
                "vit_small_patch16_224",
                pretrained=False,
                num_classes=0,  # return features, no classifier
            )
            vit_dim = self.vit.num_features
        except ImportError:
            # Fallback: simple patch embedding + transformer
            self.vit = None
            num_patches = (image_size // patch_size) ** 2
            patch_dim = 3 * patch_size * patch_size
            self._patch_size = patch_size
            self._embed = nn.Linear(patch_dim, hidden_dim)
            self._pos = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim) * 0.02)
            self._cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True
            )
            self._transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
            vit_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(vit_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch tokens. x: (B, C, H, W)"""
        p = self._patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p * C)
        return x

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Simple ViT forward without timm."""
        patches = self._patchify(x)      # (B, N, patch_dim)
        tokens = self._embed(patches)     # (B, N, hidden)
        B = tokens.shape[0]
        cls = self._cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self._pos
        out = self._transformer(tokens)
        return out[:, 0]  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, C, H, W).
        Returns:
            (B, output_dim).
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        if self.vit is not None:
            feats = self.vit(x_flat)
        else:
            feats = self._forward_fallback(x_flat)
        feats = feats.view(B, T, -1).mean(dim=1)
        return self.proj(feats)


# ─── Top-level encoder ───────────────────────────────────────────

class FacialEncoder(BaseEncoder):
    """Facial modality encoder supporting CNN, ViT, landmarks, or hybrid.

    Args:
        config: Dictionary from configs/base_config.yaml under encoders.facial
    """

    def __init__(self, config: dict) -> None:
        output_dim = config.get("output_dim", 256)
        dropout = config.get("dropout", 0.3)
        super().__init__(output_dim=output_dim, dropout=dropout)

        enc_type = config.get("encoder_type", "cnn")
        use_landmarks = config.get("use_landmarks", True)
        use_images = config.get("use_image_sequence", True)
        hidden_dim = config.get("hidden_dim", 512)
        image_size = config.get("image_size", 224)
        patch_size = config.get("vit_patch_size", 16)

        self.use_landmarks = use_landmarks
        self.use_images = use_images

        # ── Image encoder ─────────────────────────────────────────
        if use_images:
            if enc_type == "cnn":
                self.image_encoder = FrameCNNEncoder(hidden_dim, output_dim, dropout)
            elif enc_type == "vit":
                self.image_encoder = ViTFrameEncoder(
                    image_size, patch_size, hidden_dim, output_dim, dropout
                )
            else:
                raise ValueError(f"Unknown facial encoder_type: {enc_type}")

        # ── Landmark encoder ───────────────────────────────────────
        if use_landmarks:
            self.landmark_encoder = LandmarkEncoder(
                num_landmarks=68, hidden_dim=hidden_dim,
                output_dim=output_dim, dropout=dropout
            )

        # ── Fusion if both ─────────────────────────────────────────
        if use_images and use_landmarks:
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def encode(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode facial input.

        Args:
            x: Dict with optional keys:
                - 'frames'    : (B, T, C, H, W)
                - 'landmarks' : (B, T, num_landmarks*2)

        Returns:
            Embedding (B, output_dim).
        """
        embs = []

        if self.use_images and "frames" in x:
            embs.append(self.image_encoder(x["frames"]))

        if self.use_landmarks and "landmarks" in x:
            embs.append(self.landmark_encoder(x["landmarks"]))

        if len(embs) == 0:
            raise ValueError("FacialEncoder received no valid input keys.")
        if len(embs) == 1:
            return embs[0]
        # Both available → fuse
        return self.fusion(torch.cat(embs, dim=-1))
