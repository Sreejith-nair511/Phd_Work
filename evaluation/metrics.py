"""Evaluation metrics for depression detection."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities for positive class (for ROC AUC).
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dict with accuracy, precision, recall, f1, and optionally roc_auc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_prob is not None:
        try:
            if average == "binary":
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def classification_report_str(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Return a formatted sklearn classification report string.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        target_names: Optional class names.

    Returns:
        Formatted report string.
    """
    if target_names is None:
        target_names = ["Not Depressed", "Depressed"]
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )


def get_confusion_matrix(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> np.ndarray:
    """Return confusion matrix as a numpy array."""
    return confusion_matrix(y_true, y_pred)
