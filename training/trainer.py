"""Main training loop for the multimodal depression detection framework."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.losses import get_loss_fn
from training.optimizers import build_optimizer, build_scheduler
from utils.logger import get_logger


class Trainer:
    """Handles training, validation, checkpointing, and early stopping.

    Args:
        model: MultimodalDepressionModel instance.
        config: Full framework config dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_weights: Optional tensor for weighted loss.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        train_cfg = config.get("training", {})
        proj_cfg = config.get("project", {})

        self.epochs = int(train_cfg.get("epochs", 50))
        self.grad_clip = float(train_cfg.get("gradient_clip", 1.0))
        self.patience = int(train_cfg.get("early_stopping_patience", 10))
        self.mixed_precision = bool(proj_cfg.get("mixed_precision", True))
        self.checkpoint_dir = Path(proj_cfg.get("checkpoint_dir", "models/checkpoints/"))
        self.experiment_name = proj_cfg.get("name", "experiment")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        device_str = proj_cfg.get("device", "cuda")
        self.device = torch.device(
            device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
        )
        self.model = self.model.to(self.device)

        # Loss
        clf_cfg = config.get("classifier", {})
        task = clf_cfg.get("task", "binary")
        smoothing = float(train_cfg.get("label_smoothing", 0.1))
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = get_loss_fn(
            task=task,
            loss_type="label_smoothing",
            smoothing=smoothing,
            class_weights=class_weights,
        )

        # Optimizer & Scheduler
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(
            self.optimizer, config, steps_per_epoch=len(train_loader)
        )
        self.scaler = GradScaler(enabled=self.mixed_precision)

        # Logging
        log_dir = Path(proj_cfg.get("log_dir", "runs/"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(
            __name__, log_file=log_dir / f"{self.experiment_name}.log"
        )

        # State
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, list]:
        """Run the full training loop.

        Returns:
            Training history dict with loss and accuracy curves.
        """
        self.logger.info(
            f"Starting training: {self.experiment_name} | "
            f"Epochs={self.epochs} | Device={self.device}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._val_epoch(epoch)

            # Scheduler step
            sched = self.scheduler
            if hasattr(sched, "step"):
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_loss)
                else:
                    sched.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch [{epoch:03d}/{self.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={current_lr:.2e} time={elapsed:.1f}s"
            )

            # Checkpointing + early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs."
                    )
                    break

        self.logger.info("Training complete.")
        return self.history

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Run evaluation on test set and return loss and accuracy."""
        return self._val_epoch(epoch=0, loader=test_loader)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_inputs, labels in self.train_loader:
            labels = labels.to(self.device)
            batch_inputs = self._inputs_to_device(batch_inputs)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.mixed_precision):
                logits, _ = self.model(batch_inputs)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _val_epoch(
        self,
        epoch: int,
        loader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        self.model.eval()
        loader = loader or self.val_loader
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_inputs, labels in loader:
            labels = labels.to(self.device)
            batch_inputs = self._inputs_to_device(batch_inputs)

            with autocast(enabled=self.mixed_precision):
                logits, _ = self.model(batch_inputs)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively move all tensors in inputs dict to device."""
        result = {}
        for k, v in inputs.items():
            if v is None:
                result[k] = None
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                result[k] = self._inputs_to_device(v)
            else:
                result[k] = v
        return result

    def _save_checkpoint(self, epoch: int, val_loss: float, best: bool = False) -> None:
        tag = "best" if best else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"{self.experiment_name}_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
                "history": self.history,
            },
            str(path),
        )
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the saved epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]
