"""Optimizer and learning-rate scheduler builders."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)


def build_optimizer(
    model: nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    """Build optimizer with optional layer-wise learning rates.

    Pre-trained transformer encoders (speech/text) use a smaller LR to
    prevent catastrophic forgetting.

    Args:
        model: The full MultimodalDepressionModel.
        config: Full framework config dict.

    Returns:
        Configured optimizer instance.
    """
    train_cfg = config.get("training", {})
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    # Separate parameters into groups
    encoder_params: List[Dict[str, Any]] = []
    other_params: List[Dict[str, Any]] = []

    pretrained_enc_names = ("transformer", "wav2vec2", "bert", "roberta")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Lower LR for pre-trained backbone parameters
        is_pretrained = any(enc in name for enc in pretrained_enc_names)
        if is_pretrained:
            encoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": base_lr},
        {"params": encoder_params, "lr": base_lr * 0.1},  # 10× smaller for backbones
    ]

    return AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: int,
) -> Any:
    """Build learning-rate scheduler.

    Args:
        optimizer: Configured optimizer.
        config: Full framework config dict.
        steps_per_epoch: Number of batches per training epoch.

    Returns:
        LR scheduler instance.
    """
    train_cfg = config.get("training", {})
    scheduler_type = train_cfg.get("scheduler", "cosine")
    total_epochs = int(train_cfg.get("epochs", 50))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    if scheduler_type == "cosine":
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-7,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=total_epochs // 3, gamma=0.1)

    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_type}")
