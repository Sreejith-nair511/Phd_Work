"""Factory to build train/val/test DataLoaders from config."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.collate import multimodal_collate_fn
from dataset.daic_woz_dataset import DAICWOZDataset
from dataset.modma_dataset import MODMADataset

DATASET_REGISTRY = {
    "daic_woz": DAICWOZDataset,
    "modma": MODMADataset,
}


def build_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from config.

    Args:
        config: Full framework configuration dictionary.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    ds_cfg = config.get("dataset", {})
    dataset_name = ds_cfg.get("name", "daic_woz")
    root = ds_cfg.get("root", "dataset/") + dataset_name
    augmentation = ds_cfg.get("augmentation", True)
    cache = ds_cfg.get("cache_features", True)
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    use_class_weights = config.get("training", {}).get("class_weights", True)
    phq8_threshold = ds_cfg.get("phq8_threshold", 10)

    # Active modalities from config
    modalities_cfg = config.get("modalities", {})
    active_modalities = [m for m, on in modalities_cfg.items() if on]

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]

    # Common kwargs (both datasets accept these)
    common_kwargs = dict(
        root=root,
        active_modalities=active_modalities,
        phq8_threshold=phq8_threshold,
        cache_features=cache,
    )

    train_ds = DatasetClass(split="train", augmentation=augmentation, **common_kwargs)
    val_ds = DatasetClass(split="val", augmentation=False, **common_kwargs)
    test_ds = DatasetClass(split="test", augmentation=False, **common_kwargs)

    # ── Weighted sampler for class imbalance ───────────────────────
    train_sampler = None
    if use_class_weights:
        class_weights = train_ds.get_class_weights()
        sample_weights = torch.tensor(
            [class_weights[label] for label in train_ds.get_labels()],
            dtype=torch.float,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
