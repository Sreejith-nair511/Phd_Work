"""Abstract base dataset for depression detection."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class BaseDepressionDataset(ABC, Dataset):
    """Base class all depression datasets must inherit.

    Defines the contract for:
        - Loading participant metadata and labels.
        - Returning per-sample multimodal inputs + label.
        - Handling missing modalities gracefully (return None).

    Subclasses must implement:
        - ``_load_metadata()``
        - ``_get_speech_input(idx)``
        - ``_get_text_input(idx)``
        - ``_get_eeg_input(idx)``
        - ``_get_facial_input(idx)``

    Each modality method should return None if the modality is unavailable
    for a given sample.
    """

    MODALITY_KEYS = ["speech", "text", "eeg", "facial"]

    def __init__(
        self,
        root: str | Path,
        split: str,                      # 'train' | 'val' | 'test'
        active_modalities: List[str],
        phq8_threshold: int = 10,
        augmentation: bool = False,
        cache_features: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.active_modalities = active_modalities
        self.phq8_threshold = phq8_threshold
        self.augmentation = augmentation and (split == "train")
        self.cache_features = cache_features

        self._cache: Dict[int, Any] = {}
        self.samples: List[Dict[str, Any]] = []   # populated by _load_metadata
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        """Populate self.samples with dicts containing at minimum:
            {'participant_id': str, 'label': int, 'phq8_score': float}
        """
        ...

    @abstractmethod
    def _get_speech_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Return speech input dict or None."""
        ...

    @abstractmethod
    def _get_text_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Return tokenised text dict or None."""
        ...

    @abstractmethod
    def _get_eeg_input(self, idx: int) -> Optional[torch.Tensor]:
        """Return EEG tensor (C, T) or None."""
        ...

    @abstractmethod
    def _get_facial_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Return facial input dict (frames and/or landmarks) or None."""
        ...

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                    #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], int]:
        if self.cache_features and idx in self._cache:
            return self._cache[idx]

        sample_meta = self.samples[idx]
        label = int(sample_meta["label"])

        inputs: Dict[str, Any] = {}

        modality_getters = {
            "speech": self._get_speech_input,
            "text": self._get_text_input,
            "eeg": self._get_eeg_input,
            "facial": self._get_facial_input,
        }

        for modality in self.MODALITY_KEYS:
            if modality in self.active_modalities:
                inputs[modality] = modality_getters[modality](idx)
            else:
                inputs[modality] = None

        result = (inputs, label)

        if self.cache_features:
            self._cache[idx] = result

        return result

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        labels = [int(s["label"]) for s in self.samples]
        num_classes = max(labels) + 1
        counts = torch.zeros(num_classes)
        for l in labels:
            counts[l] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum() * num_classes

    def get_labels(self) -> List[int]:
        """Return all labels (useful for stratified splitting)."""
        return [int(s["label"]) for s in self.samples]
