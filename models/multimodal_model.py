"""Core multimodal depression detection model.

Orchestrates:
  1. Per-modality encoders (only active modalities are run)
  2. Fusion layer (handles missing modalities automatically)
  3. Classifier head

Any combination of modalities can be passed; missing ones are simply None.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from encoders.encoder_factory import build_all_encoders
from fusion.fusion_factory import build_fusion
from models.classifier import DepressionClassifier


class MultimodalDepressionModel(nn.Module):
    """End-to-end modality-agnostic depression detection model.

    Args:
        config: Full framework configuration dictionary.

    Input (forward):
        inputs: Dict[str, Optional[any]] mapping modality name to its input
                data (tensor or dict).  Pass None for unavailable modalities.

    Output (forward):
        logits: (B, num_classes) classification logits or (B, 1) regression.
        embeddings: Dict of per-modality embeddings + 'fused' key.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.modality_flags: Dict[str, bool] = config.get("modalities", {})

        # ── Encoders ────────────────────────────────────────────────
        encoder_dict = build_all_encoders(config)
        self.encoders = nn.ModuleDict(
            {k: v for k, v in encoder_dict.items() if v is not None}
        )
        self.disabled_modalities = [k for k, v in encoder_dict.items() if v is None]

        # ── Fusion ──────────────────────────────────────────────────
        self.fusion = build_fusion(config)

        # ── Classifier ──────────────────────────────────────────────
        clf_cfg = config.get("classifier", {})
        embedding_dim = config.get("fusion", {}).get("embedding_dim", 256)
        self.classifier = DepressionClassifier(
            input_dim=embedding_dim,
            hidden_dims=clf_cfg.get("hidden_dims", [512, 256, 128]),
            num_classes=clf_cfg.get("num_classes", 2),
            dropout=clf_cfg.get("dropout", 0.4),
            task=clf_cfg.get("task", "binary"),
        )

    # ------------------------------------------------------------------ #
    #  Forward pass                                                         #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        inputs: Dict[str, Optional[any]],
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Run the full model pipeline.

        Args:
            inputs: Dict mapping modality name → modality-specific input or None.
                    Supported keys: 'speech', 'text', 'eeg', 'facial'.

        Returns:
            logits: Classification/regression output (B, num_classes).
            embeddings: Dict with per-modality embeddings and 'fused' key.
        """
        # ── Step 1: encode each available modality ─────────────────
        embeddings: Dict[str, Optional[torch.Tensor]] = {}

        for modality, encoder in self.encoders.items():
            raw_input = inputs.get(modality)
            embeddings[modality] = encoder(raw_input)  # returns None if input is None

        # Ensure disabled modalities are explicitly None
        for modality in self.disabled_modalities:
            embeddings[modality] = None

        # ── Step 2: fuse available embeddings ──────────────────────
        fused = self.fusion(embeddings)   # (B, embedding_dim)
        embeddings["fused"] = fused

        # ── Step 3: classify ───────────────────────────────────────
        logits = self.classifier(fused)

        return logits, embeddings

    # ------------------------------------------------------------------ #
    #  Utility                                                              #
    # ------------------------------------------------------------------ #
    def get_active_modalities(self) -> List[str]:
        """Return list of modalities with active encoders."""
        return list(self.encoders.keys())

    def count_parameters(self) -> Dict[str, int]:
        """Parameter count breakdown by component."""
        counts: Dict[str, int] = {}
        for name, encoder in self.encoders.items():
            counts[f"encoder_{name}"] = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad
            )
        counts["fusion"] = sum(
            p.numel() for p in self.fusion.parameters() if p.requires_grad
        )
        counts["classifier"] = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )
        counts["total"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts

    def freeze_encoders(self, modalities: Optional[List[str]] = None) -> None:
        """Freeze encoder weights (useful for fine-tuning fusion/classifier only).

        Args:
            modalities: List of modality names to freeze, or None for all.
        """
        targets = modalities or list(self.encoders.keys())
        for m in targets:
            if m in self.encoders:
                for p in self.encoders[m].parameters():
                    p.requires_grad = False

    def unfreeze_encoders(self, modalities: Optional[List[str]] = None) -> None:
        """Unfreeze encoder weights."""
        targets = modalities or list(self.encoders.keys())
        for m in targets:
            if m in self.encoders:
                for p in self.encoders[m].parameters():
                    p.requires_grad = True
