"""Plot training and validation loss/accuracy curves."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str | Path] = None,
    title: str = "Training Curves",
    show: bool = False,
) -> None:
    """Plot loss and accuracy curves from training history.

    Args:
        history: Dict with keys train_loss, val_loss, train_acc, val_acc, lr.
        save_path: If provided, save figure to this path.
        title: Figure title.
        show: Display the plot interactively.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[2]
    if "lr" in history and history["lr"]:
        ax.plot(epochs, history["lr"], color="green", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
