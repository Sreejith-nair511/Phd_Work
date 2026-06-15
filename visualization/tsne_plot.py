"""t-SNE embedding visualization."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    modality_name: str = "fused",
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    show: bool = False,
) -> None:
    """Visualize embeddings in 2-D using t-SNE.

    Args:
        embeddings: (N, D) float array of embeddings.
        labels: (N,) integer class labels.
        modality_name: Name of the modality/layer being visualized.
        class_names: List of class label strings.
        save_path: If provided, save figure here.
        title: Plot title (auto-generated if None).
        perplexity: t-SNE perplexity parameter.
        n_iter: t-SNE iterations.
        random_state: Seed for reproducibility.
        show: Display interactively.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn required for t-SNE visualization.")
        return

    if class_names is None:
        class_names = ["Not Depressed", "Depressed"]

    if title is None:
        title = f"t-SNE: {modality_name} embeddings"

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    reduced = tsne.fit_transform(embeddings)

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=colors[i % len(colors)],
            label=class_names[label] if label < len(class_names) else str(label),
            alpha=0.65,
            s=30,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
