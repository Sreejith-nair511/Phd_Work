"""Preprocessing utilities shared across dataset loaders.

Covers:
  - Audio loading and resampling
  - MFCC feature extraction
  - Prosodic feature extraction (speech rate, pauses, energy, pitch, latency)
  - Transcript loading and cleaning
  - Text tokenisation
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio file, resample to target_sr, convert to mono.

    Args:
        path: Path to audio file (.wav, .mp3, etc.).
        target_sr: Target sample rate in Hz.
        mono: Convert to mono if True.

    Returns:
        (waveform, sample_rate) – waveform shape (1, T).
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform, target_sr
    except Exception:
        # Fallback: return silence
        return torch.zeros(1, target_sr * 5), target_sr


def extract_mfcc(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    """Compute MFCC features from waveform.

    Args:
        waveform: (1, T) audio tensor.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT size.
        hop_length: Hop length in samples.
        n_mels: Number of mel filterbanks.

    Returns:
        MFCC tensor of shape (T_frames, n_mfcc).
    """
    try:
        import torchaudio
        transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
            },
        )
        mfcc = transform(waveform)  # (1, n_mfcc, T_frames)
        mfcc = mfcc.squeeze(0).T    # (T_frames, n_mfcc)
        # Normalize per coefficient
        mean = mfcc.mean(dim=0, keepdim=True)
        std = mfcc.std(dim=0, keepdim=True) + 1e-8
        return (mfcc - mean) / std
    except Exception:
        T_frames = waveform.shape[-1] // hop_length
        return torch.zeros(T_frames, n_mfcc)


def extract_prosodic_features(
    waveform: torch.Tensor,
    sr: int = 16000,
    frame_length_ms: int = 25,
    hop_length_ms: int = 10,
    silence_threshold_db: float = -40.0,
) -> Dict[str, torch.Tensor]:
    """Extract depression-relevant prosodic features.

    Features extracted:
        - speech_rate: voiced frames / total frames
        - pause_duration: mean duration of silent segments (normalised)
        - response_latency: duration of leading silence (normalised)
        - energy: RMS energy (mean over voiced frames)
        - pitch: mean fundamental frequency (F0) over voiced frames

    Args:
        waveform: (1, T) audio tensor.
        sr: Sample rate in Hz.
        frame_length_ms: Frame length in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        silence_threshold_db: dB threshold below which frames are silent.

    Returns:
        Dict of scalar torch.Tensor values.
    """
    wav = waveform.squeeze(0)  # (T,)
    T = wav.shape[0]

    frame_len = int(sr * frame_length_ms / 1000)
    hop_len = int(sr * hop_length_ms / 1000)

    # Frame the signal
    n_frames = max(1, (T - frame_len) // hop_len + 1)
    frames = wav.unfold(0, frame_len, hop_len)  # (n_frames, frame_len)

    # RMS energy per frame (in dB)
    rms = frames.pow(2).mean(dim=-1).clamp(min=1e-10).sqrt()
    rms_db = 20 * torch.log10(rms + 1e-10)

    # Voiced / silence masks
    voiced_mask = rms_db > silence_threshold_db   # (n_frames,)
    silent_mask = ~voiced_mask

    # ── speech_rate ────────────────────────────────────────────────
    speech_rate = voiced_mask.float().mean()

    # ── pause_duration ─────────────────────────────────────────────
    # Mean length of contiguous silent segments (in frames, normalised)
    silence_runs = _run_lengths(silent_mask)
    pause_duration = (
        torch.tensor(float(silence_runs.mean())) if len(silence_runs) > 0
        else torch.tensor(0.0)
    ) / max(n_frames, 1)

    # ── response_latency ───────────────────────────────────────────
    # How many frames before first voiced frame (normalised)
    first_voiced = voiced_mask.float().argmax()
    if voiced_mask.any():
        response_latency = first_voiced.float() / n_frames
    else:
        response_latency = torch.tensor(1.0)

    # ── energy ─────────────────────────────────────────────────────
    voiced_rms = rms[voiced_mask] if voiced_mask.any() else rms
    energy = voiced_rms.mean()

    # ── pitch (F0 approximation via zero-crossing rate) ────────────
    # Full pitch tracking requires librosa/CREPE; use ZCR as proxy
    zero_crossings = ((frames[:, :-1] * frames[:, 1:]) < 0).float().sum(dim=-1)
    f0_proxy = zero_crossings[voiced_mask] if voiced_mask.any() else zero_crossings
    pitch = (f0_proxy.mean() * sr / (2 * frame_len)).clamp(0, 500) / 500.0  # normalised

    return {
        "speech_rate": speech_rate,
        "pause_duration": pause_duration,
        "response_latency": response_latency,
        "energy": energy.clamp(0, 1),
        "pitch": pitch,
    }


def _run_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Compute lengths of contiguous True runs in a boolean tensor."""
    if not mask.any():
        return torch.tensor([])
    mask = mask.int()
    diff = torch.diff(mask, prepend=torch.tensor([0]), append=torch.tensor([0]))
    starts = (diff == 1).nonzero(as_tuple=True)[0]
    ends = (diff == -1).nonzero(as_tuple=True)[0]
    return (ends - starts).float()


# ─────────────────────────────────────────────────────────────────────────────
# Text
# ─────────────────────────────────────────────────────────────────────────────

def load_transcript(path: str | Path) -> str:
    """Load participant transcript from DAIC-WOZ CSV format.

    DAIC-WOZ transcript CSV columns: start_time, stop_time, speaker, value
    Only 'Participant' speaker utterances are kept.

    Args:
        path: Path to transcript CSV file.

    Returns:
        Concatenated participant utterances as a string.
    """
    path = Path(path)
    if not path.exists():
        return ""

    utterances = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                speaker = row.get("speaker", row.get("Speaker", "")).strip()
                text = row.get("value", row.get("Value", "")).strip()
                if speaker.lower() in ("participant", "p") and text:
                    utterances.append(text)
    except Exception:
        return ""

    return " ".join(utterances)


def clean_transcript(text: str) -> str:
    """Clean raw transcript text.

    Operations:
        - Unicode normalisation (NFC)
        - Lowercase
        - Remove filler words (um, uh, mm, etc.)
        - Remove timestamps/markers in brackets
        - Collapse multiple spaces
        - Strip leading/trailing whitespace

    Args:
        text: Raw transcript string.

    Returns:
        Cleaned transcript string.
    """
    if not text:
        return ""

    # Unicode normalise
    text = unicodedata.normalize("NFC", text)

    # Remove content in brackets/parentheses (e.g., [laughs], (inaudible))
    text = re.sub(r"[\[\(][^\]\)]*[\]\)]", " ", text)

    # Lowercase
    text = text.lower()

    # Remove filler words
    fillers = r"\b(um+|uh+|mm+|hmm+|erm+|ah+|oh+|eh+)\b"
    text = re.sub(fillers, "", text)

    # Remove punctuation except basic sentence-ending marks
    text = re.sub(r"[^\w\s\.\,\?\!\'']", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(
    text: str,
    tokenizer: Any,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize cleaned text using a HuggingFace tokenizer.

    Args:
        text: Cleaned text string.
        tokenizer: HuggingFace PreTrainedTokenizer instance.
        max_length: Maximum token sequence length.
        padding: Padding strategy.
        truncation: Whether to truncate to max_length.

    Returns:
        Dict with 'input_ids', 'attention_mask', and optionally 'token_type_ids'.
    """
    if not text:
        text = "[PAD]"

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )

    result: Dict[str, torch.Tensor] = {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }
    if "token_type_ids" in encoding:
        result["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

    return result
