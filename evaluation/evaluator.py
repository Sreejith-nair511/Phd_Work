"""Full evaluation pipeline: inference + metrics + report generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from evaluation.metrics import (
    compute_metrics,
    classification_report_str,
    get_confusion_matrix,
)
from utils.logger import get_logger


class Evaluator:
    """Runs inference on a DataLoader and computes all evaluation metrics.

    Args:
        model: Trained MultimodalDepressionModel.
        config: Full framework config dict.
    """

    def __init__(self, model: nn.Module, config: dict) -> None:
        self.model = model
        self.config = config

        proj_cfg = config.get("project", {})
        device_str = proj_cfg.get("device", "cuda")
        self.device = torch.device(
            device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
        )
        self.model = self.model.to(self.device)
        self.mixed_precision = bool(proj_cfg.get("mixed_precision", True))
        self.logger = get_logger(__name__)

    @torch.no_grad()
    def run(
        self,
        loader: DataLoader,
        return_embeddings: bool = False,
    ) -> Dict:
        """Run full inference and return results dict.

        Args:
            loader: DataLoader to evaluate on.
            return_embeddings: Whether to collect per-sample embeddings.

        Returns:
            Dict with keys: metrics, report, confusion_matrix,
            y_true, y_pred, y_prob, and optionally embeddings.
        """
        self.model.eval()

        all_labels: List[int] = []
        all_preds: List[int] = []
        all_probs: List[float] = []
        all_embeddings: List[np.ndarray] = []

        for batch_inputs, labels in loader:
            labels = labels.to(self.device)
            batch_inputs = self._to_device(batch_inputs)

            with autocast(enabled=self.mixed_precision):
                logits, embeddings = self.model(batch_inputs)

            probs = torch.softmax(logits, dim=-1)          # (B, C)
            preds = probs.argmax(dim=-1)                   # (B,)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            # Probability of positive class (index 1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

            if return_embeddings and "fused" in embeddings:
                all_embeddings.append(
                    embeddings["fused"].cpu().float().numpy()
                )

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        report = classification_report_str(y_true, y_pred)
        cm = get_confusion_matrix(y_true, y_pred)

        self.logger.info("Evaluation results:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        self.logger.info(f"\n{report}")

        result = {
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        if return_embeddings and all_embeddings:
            result["embeddings"] = np.concatenate(all_embeddings, axis=0)
            result["embedding_labels"] = y_true

        return result

    def save_report(self, results: Dict, output_dir: str | Path) -> None:
        """Save evaluation report, confusion matrix, and metrics to disk.

        Args:
            results: Output dict from ``run()``.
            output_dir: Directory to save outputs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Classification report
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(results["report"])
        self.logger.info(f"Saved classification report to {report_path}")

        # Metrics JSON
        import json
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results["metrics"], f, indent=2)
        self.logger.info(f"Saved metrics to {metrics_path}")

        # Confusion matrix (numpy)
        cm_path = output_dir / "confusion_matrix.npy"
        np.save(str(cm_path), results["confusion_matrix"])

    def _to_device(self, inputs: Dict) -> Dict:
        result = {}
        for k, v in inputs.items():
            if v is None:
                result[k] = None
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                result[k] = self._to_device(v)
            else:
                result[k] = v
        return result
