"""Factory function to build encoders from config."""
from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from encoders.speech_encoder import SpeechEncoder
from encoders.text_encoder import TextEncoder
from encoders.eeg_encoder import EEGEncoder
from encoders.facial_encoder import FacialEncoder


ENCODER_REGISTRY = {
    "speech": SpeechEncoder,
    "text": TextEncoder,
    "eeg": EEGEncoder,
    "facial": FacialEncoder,
}


def build_encoder(modality: str, config: dict) -> nn.Module:
    """Instantiate an encoder for a given modality.

    Args:
        modality: One of 'speech', 'text', 'eeg', 'facial'.
        config: Full framework config dict (encoders sub-dict is extracted).

    Returns:
        Instantiated encoder module.

    Raises:
        ValueError: If modality is not registered.
    """
    if modality not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    encoder_cfg = config.get("encoders", {}).get(modality, {})
    # Ensure output_dim is consistent across all encoders
    encoder_cfg["output_dim"] = config.get("encoders", {}).get("embedding_dim", 256)
    return ENCODER_REGISTRY[modality](encoder_cfg)


def build_all_encoders(config: dict) -> Dict[str, Optional[nn.Module]]:
    """Build encoders only for enabled modalities.

    Args:
        config: Full framework config dict.

    Returns:
        Dict mapping modality name → encoder (or None if disabled).
    """
    modalities_cfg = config.get("modalities", {})
    encoders: Dict[str, Optional[nn.Module]] = {}
    for modality in ENCODER_REGISTRY:
        if modalities_cfg.get(modality, False):
            encoders[modality] = build_encoder(modality, config)
        else:
            encoders[modality] = None
    return encoders
