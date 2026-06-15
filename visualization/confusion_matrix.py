"""Confusion matrix visualization."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    show: bool = False,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (n_classes x n_classes).
        class_names: List of class label strings.
        save_path: If provided, save figure to this path.
        title: Plot title.
        normalize: Show normalized (percentage) values if True.
        show: Display interactively.
    """
    if class_names is None:
        class_names = ["Not Depressed", "Depressed"]

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
