"""Factory to build fusion modules from config."""
from __future__ import annotations

import torch.nn as nn

from fusion.early_fusion import EarlyFusion
from fusion.late_fusion import LateFusion
from fusion.attention_fusion import AttentionFusion
from fusion.cross_modal_fusion import CrossModalFusion

FUSION_REGISTRY = {
    "early": EarlyFusion,
    "late": LateFusion,
    "attention": AttentionFusion,
    "cross_modal": CrossModalFusion,
}


def build_fusion(config: dict) -> nn.Module:
    """Instantiate a fusion module from config.

    Args:
        config: Full framework config dict.

    Returns:
        Fusion module instance.
    """
    fusion_cfg = config.get("fusion", {})
    fusion_type = fusion_cfg.get("type", "attention")

    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion type '{fusion_type}'. "
            f"Available: {list(FUSION_REGISTRY.keys())}"
        )

    embedding_dim = fusion_cfg.get("embedding_dim", 256)
    output_dim = fusion_cfg.get("embedding_dim", 256)
    num_heads = fusion_cfg.get("num_heads", 8)
    dropout = fusion_cfg.get("dropout", 0.3)

    kwargs = dict(
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        dropout=dropout,
    )

    if fusion_type in ("attention", "cross_modal"):
        kwargs["num_heads"] = num_heads

    return FUSION_REGISTRY[fusion_type](**kwargs)
