"""DAIC-WOZ dataset loader.

DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz) contains:
  - Audio interviews (.wav)
  - Transcripts (.csv with word-level timestamps)
  - PHQ-8 questionnaire scores

Directory structure expected:
    dataset/daic_woz/
        train_split_Depression_AVEC2017.csv
        dev_split_Depression_AVEC2017.csv
        test_split_Depression_AVEC2017.csv
        <participant_id>/
            <participant_id>_AUDIO.wav
            <participant_id>_TRANSCRIPT.csv
            <participant_id>_COVAREP.csv          (acoustic features, optional)

Reference: Gratch et al., 2014.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset.base_dataset import BaseDepressionDataset
from dataset.preprocessing import (
    load_audio,
    extract_mfcc,
    extract_prosodic_features,
    load_transcript,
    tokenize_text,
    clean_transcript,
)


class DAICWOZDataset(BaseDepressionDataset):
    """PyTorch Dataset for DAIC-WOZ.

    Args:
        root: Path to dataset/daic_woz/ directory.
        split: 'train' | 'val' | 'test'.
        active_modalities: List of modalities to load.
        tokenizer_name: HuggingFace tokenizer for text.
        feature_type: 'mfcc' or 'wav2vec2' for speech.
        phq8_threshold: PHQ-8 score >= threshold → depressed (label=1).
        max_audio_len: Max audio duration in seconds (truncate/pad).
        max_text_len: Max token length for text.
        augmentation: Apply data augmentation on train split.
        cache_features: Cache loaded features in memory.
    """

    SPLIT_FILES = {
        "train": "train_split_Depression_AVEC2017.csv",
        "val": "dev_split_Depression_AVEC2017.csv",
        "test": "test_split_Depression_AVEC2017.csv",
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        active_modalities: Optional[List[str]] = None,
        tokenizer_name: str = "roberta-base",
        feature_type: str = "mfcc",
        phq8_threshold: int = 10,
        max_audio_len: int = 300,      # seconds
        max_text_len: int = 512,
        augmentation: bool = False,
        cache_features: bool = True,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.feature_type = feature_type
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self._tokenizer = None   # lazy-loaded

        if active_modalities is None:
            active_modalities = ["speech", "text"]

        super().__init__(
            root=root,
            split=split,
            active_modalities=active_modalities,
            phq8_threshold=phq8_threshold,
            augmentation=augmentation,
            cache_features=cache_features,
        )

    # ------------------------------------------------------------------ #
    #  Metadata loading                                                     #
    # ------------------------------------------------------------------ #
    def _load_metadata(self) -> None:
        split_key = self.split if self.split != "val" else "val"
        csv_file = self.root / self.SPLIT_FILES.get(split_key, self.SPLIT_FILES["train"])

        if not csv_file.exists():
            # Graceful degradation: create dummy metadata for unit testing
            self.samples = self._create_dummy_samples(n=10)
            return

        self.samples = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("Participant_ID", row.get("participant_id", ""))
                phq8_score = float(row.get("PHQ8_Score", row.get("PHQ_Score", 0)))
                label = int(phq8_score >= self.phq8_threshold)
                self.samples.append(
                    {
                        "participant_id": str(pid),
                        "phq8_score": phq8_score,
                        "label": label,
                        "audio_path": self.root / str(pid) / f"{pid}_AUDIO.wav",
                        "transcript_path": self.root / str(pid) / f"{pid}_TRANSCRIPT.csv",
                    }
                )

    def _create_dummy_samples(self, n: int = 10) -> List[Dict[str, Any]]:
        """Generate dummy samples for testing without real data."""
        return [
            {
                "participant_id": f"P{300 + i:03d}",
                "phq8_score": float(i * 2),
                "label": int(i >= 5),
                "audio_path": None,
                "transcript_path": None,
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
            return self._dummy_speech_input()

        waveform, sr = load_audio(audio_path, target_sr=self.sample_rate)

        # Truncate/pad to max_audio_len seconds
        max_samples = self.max_audio_len * self.sample_rate
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        else:
            pad_len = max_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_len))

        result: Dict[str, torch.Tensor] = {}

        if self.feature_type == "mfcc":
            mfcc = extract_mfcc(waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            result["mfcc"] = mfcc  # (T, n_mfcc)
        else:
            # wav2vec2 expects raw waveform
            result["input_values"] = waveform.squeeze(0)   # (T,)

        # Prosodic features (depression-relevant silence, energy, pitch, etc.)
        prosodic = extract_prosodic_features(waveform, sr=self.sample_rate)
        result.update(prosodic)

        return result

    def _get_text_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        transcript_path = sample.get("transcript_path")

        if transcript_path is None or not Path(transcript_path).exists():
            return self._dummy_text_input()

        raw_text = load_transcript(transcript_path)
        cleaned = clean_transcript(raw_text)

        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()

        return tokenize_text(
            cleaned,
            tokenizer=self._tokenizer,
            max_length=self.max_text_len,
        )

    def _get_eeg_input(self, idx: int) -> Optional[torch.Tensor]:
        # DAIC-WOZ does not include EEG data
        return None

    def _get_facial_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        # DAIC-WOZ facial features optional (COVAREP/OpenFace outputs)
        # Placeholder – extend if OpenFace landmarks are available
        return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def _dummy_speech_input(self) -> Dict[str, torch.Tensor]:
        """Zero tensor dummy for missing audio."""
        max_samples = self.max_audio_len * self.sample_rate
        if self.feature_type == "mfcc":
            # (T, n_mfcc) where T ≈ max_samples / hop_length
            T = max_samples // 160
            return {
                "mfcc": torch.zeros(T, self.n_mfcc),
                "speech_rate": torch.tensor(0.0),
                "pause_duration": torch.tensor(0.0),
                "response_latency": torch.tensor(0.0),
                "energy": torch.tensor(0.0),
                "pitch": torch.tensor(0.0),
            }
        return {
            "input_values": torch.zeros(max_samples),
            "speech_rate": torch.tensor(0.0),
            "pause_duration": torch.tensor(0.0),
            "response_latency": torch.tensor(0.0),
            "energy": torch.tensor(0.0),
            "pitch": torch.tensor(0.0),
        }

    def _dummy_text_input(self) -> Dict[str, torch.Tensor]:
        """Zero tensor dummy for missing transcripts."""
        return {
            "input_ids": torch.zeros(self.max_text_len, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_text_len, dtype=torch.long),
        }
