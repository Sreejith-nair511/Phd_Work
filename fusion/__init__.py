"""Fusion strategies for multimodal depression detection."""
from fusion.early_fusion import EarlyFusion
from fusion.late_fusion import LateFusion
from fusion.attention_fusion import AttentionFusion
from fusion.cross_modal_fusion import CrossModalFusion
from fusion.fusion_factory import build_fusion

__all__ = [
    "EarlyFusion",
    "LateFusion",
    "AttentionFusion",
    "CrossModalFusion",
    "build_fusion",
]
