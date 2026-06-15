"""ROC curve and AUC visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    experiment_name: str = "Model",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> float:
    """Plot ROC curve and return AUC score.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for positive class.
        experiment_name: Label for the curve.
        save_path: If provided, save figure here.
        show: Display interactively.

    Returns:
        AUC score.
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{experiment_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return roc_auc


def plot_multi_roc(
    results: Dict[str, Dict],
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Plot multiple ROC curves on one figure for experiment comparison.

    Args:
        results: Dict experiment_name → {y_true, y_prob} dict.
        save_path: If provided, save figure here.
        show: Display interactively.
    """
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, lw=2,
            color=colors[i % len(colors)],
            label=f"{name} (AUC={roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Experiment Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
