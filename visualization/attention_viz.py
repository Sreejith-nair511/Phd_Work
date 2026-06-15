"""Attention weight visualization for fusion layers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_weights(
    attention_weights: np.ndarray,
    modality_names: List[str],
    sample_ids: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Modality Attention Weights",
    show: bool = False,
) -> None:
    """Plot attention weights heatmap over modalities.

    Args:
        attention_weights: (N, M) array — N samples, M modalities.
        modality_names: List of modality name strings (length M).
        sample_ids: Optional list of sample identifiers (length N).
        save_path: If provided, save figure here.
        title: Plot title.
        show: Display interactively.
    """
    import seaborn as sns

    if sample_ids is None:
        sample_ids = [f"Sample {i}" for i in range(len(attention_weights))]

    # Show at most 30 samples for readability
    if len(attention_weights) > 30:
        attention_weights = attention_weights[:30]
        sample_ids = sample_ids[:30]

    fig, ax = plt.subplots(figsize=(max(6, len(modality_names) * 2), max(4, len(sample_ids) * 0.35)))
    sns.heatmap(
        attention_weights,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=modality_names,
        yticklabels=sample_ids,
        linewidths=0.3,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Sample")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    importance_scores: Dict[str, float],
    save_path: Optional[str | Path] = None,
    title: str = "Modality Feature Importance",
    show: bool = False,
) -> None:
    """Bar chart of per-modality feature importance scores.

    Args:
        importance_scores: Dict modality_name → importance score.
        save_path: If provided, save figure here.
        title: Plot title.
        show: Display interactively.
    """
    names = list(importance_scores.keys())
    scores = [importance_scores[n] for n in names]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, scores, color=colors[: len(names)], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1.0)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
