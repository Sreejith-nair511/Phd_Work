"""MODMA (Multi-modal Open Dataset for Mental-disorder Analysis) loader.

MODMA contains:
  - EEG recordings (128-channel, 250 Hz)
  - Audio recordings (speech)
  - Labels: depression vs. healthy control

Directory structure expected:
    dataset/modma/
        metadata.csv          (participant_id, label, eeg_file, audio_file)
        eeg/
            <participant_id>.npy    (channels × time)
        audio/
            <participant_id>.wav

Reference: Cai et al., 2020. doi:10.1038/s41597-022-01211-x
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dataset.base_dataset import BaseDepressionDataset
from dataset.preprocessing import (
    load_audio,
    extract_mfcc,
    extract_prosodic_features,
)


class MODMADataset(BaseDepressionDataset):
    """PyTorch Dataset for the MODMA corpus.

    Args:
        root: Path to dataset/modma/ directory.
        split: 'train' | 'val' | 'test'.
        active_modalities: List of modalities to load.
        feature_type: 'mfcc' | 'wav2vec2' for speech.
        eeg_channels: Number of EEG channels to use (subset from start).
        eeg_seq_len: Fixed EEG sequence length (samples); pad or truncate.
        max_audio_len: Max audio length in seconds.
        augmentation: Apply augmentation on train split.
        cache_features: Cache features in memory.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        active_modalities: Optional[List[str]] = None,
        feature_type: str = "mfcc",
        eeg_channels: int = 64,
        eeg_seq_len: int = 1000,
        max_audio_len: int = 120,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        augmentation: bool = False,
        cache_features: bool = True,
    ) -> None:
        self.feature_type = feature_type
        self.eeg_channels = eeg_channels
        self.eeg_seq_len = eeg_seq_len
        self.max_audio_len = max_audio_len
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        if active_modalities is None:
            active_modalities = ["speech", "eeg"]

        super().__init__(
            root=root,
            split=split,
            active_modalities=active_modalities,
            augmentation=augmentation,
            cache_features=cache_features,
        )

    # ------------------------------------------------------------------ #
    #  Metadata                                                             #
    # ------------------------------------------------------------------ #
    def _load_metadata(self) -> None:
        meta_file = self.root / "metadata.csv"

        if not meta_file.exists():
            self.samples = self._create_dummy_samples(n=20)
            return

        all_samples: List[Dict[str, Any]] = []
        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["participant_id"]
                label = int(row["label"])
                all_samples.append(
                    {
                        "participant_id": pid,
                        "label": label,
                        "phq8_score": float(label * 15),  # approx if not available
                        "eeg_path": self.root / "eeg" / f"{pid}.npy",
                        "audio_path": self.root / "audio" / f"{pid}.wav",
                    }
                )

        # Deterministic split by participant index
        n = len(all_samples)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        splits = {
            "train": all_samples[:train_end],
            "val": all_samples[train_end:val_end],
            "test": all_samples[val_end:],
        }
        self.samples = splits.get(self.split, all_samples)

    def _create_dummy_samples(self, n: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "participant_id": f"MODMA_{i:03d}",
                "label": int(i % 2),
                "phq8_score": float((i % 2) * 15),
                "eeg_path": None,
                "audio_path": None,
            }
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    #  Modality loaders                                                     #
    # ------------------------------------------------------------------ #
    def _get_speech_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        audio_path = sample.get("audio_path")

        if audio_path is None or not Path(audio_path).exists():
            return self._dummy_speech()

        waveform, sr = load_audio(audio_path, target_sr=self.sample_rate)
        max_samples = self.max_audio_len * self.sample_rate
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        else:
            waveform = F.pad(waveform, (0, max_samples - waveform.shape[-1]))

        result: Dict[str, torch.Tensor] = {}
        if self.feature_type == "mfcc":
            result["mfcc"] = extract_mfcc(waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        else:
            result["input_values"] = waveform.squeeze(0)

        result.update(extract_prosodic_features(waveform, sr=self.sample_rate))
        return result

    def _get_text_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        # MODMA does not include transcripts
        return None

    def _get_eeg_input(self, idx: int) -> Optional[torch.Tensor]:
        sample = self.samples[idx]
        eeg_path = sample.get("eeg_path")

        if eeg_path is None or not Path(eeg_path).exists():
            return torch.zeros(self.eeg_channels, self.eeg_seq_len)

        eeg = np.load(str(eeg_path)).astype(np.float32)  # (C, T)

        # Select channels
        if eeg.shape[0] > self.eeg_channels:
            eeg = eeg[: self.eeg_channels]

        # Pad or truncate time dimension
        T = eeg.shape[1]
        if T >= self.eeg_seq_len:
            eeg = eeg[:, : self.eeg_seq_len]
        else:
            pad = np.zeros((eeg.shape[0], self.eeg_seq_len - T), dtype=np.float32)
            eeg = np.concatenate([eeg, pad], axis=1)

        # Z-score normalise per channel (preserve relative magnitudes)
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-8
        eeg = (eeg - mean) / std

        return torch.from_numpy(eeg)  # (C, T)

    def _get_facial_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        # MODMA does not include facial data
        return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _dummy_speech(self) -> Dict[str, torch.Tensor]:
        max_s = self.max_audio_len * self.sample_rate
        if self.feature_type == "mfcc":
            return {
                "mfcc": torch.zeros(max_s // 160, self.n_mfcc),
                "speech_rate": torch.tensor(0.0),
                "pause_duration": torch.tensor(0.0),
                "response_latency": torch.tensor(0.0),
                "energy": torch.tensor(0.0),
                "pitch": torch.tensor(0.0),
            }
        return {
            "input_values": torch.zeros(max_s),
            "speech_rate": torch.tensor(0.0),
            "pause_duration": torch.tensor(0.0),
            "response_latency": torch.tensor(0.0),
            "energy": torch.tensor(0.0),
            "pitch": torch.tensor(0.0),
        }
