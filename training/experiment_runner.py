"""Run a single experiment end-to-end from a config file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from dataset.dataset_factory import build_dataloaders
from evaluation.evaluator import Evaluator
from models.multimodal_model import MultimodalDepressionModel
from training.trainer import Trainer
from utils.config import load_config, get_modality_list
from utils.logger import get_logger
from utils.seed import set_seed
from visualization.confusion_matrix import plot_confusion_matrix
from visualization.roc_curve import plot_roc_curve
from visualization.training_curves import plot_training_curves
from visualization.tsne_plot import plot_tsne


def run_experiment(config_path: str) -> Dict:
    """Run a complete train → evaluate → visualize experiment.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict with metrics, history, and output paths.
    """
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    exp_name = config.get("project", {}).get("name", "experiment")
    output_dir = Path("visualization/outputs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__, log_file=f"runs/{exp_name}.log")
    logger.info(f"Running experiment: {exp_name}")
    logger.info(f"Active modalities: {get_modality_list(config)}")

    # ── Data ───────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # ── Model ──────────────────────────────────────────────────────
    model = MultimodalDepressionModel(config)
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: {param_counts}")

    # Class weights from training set
    class_weights = None
    if config.get("training", {}).get("class_weights", True):
        class_weights = train_loader.dataset.get_class_weights()

    # ── Train ──────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )
    history = trainer.fit()

    # ── Training curves ────────────────────────────────────────────
    plot_training_curves(
        history,
        save_path=output_dir / "training_curves.png",
        title=f"Training Curves: {exp_name}",
    )

    # ── Load best model ────────────────────────────────────────────
    ckpt_path = (
        Path(config.get("project", {}).get("checkpoint_dir", "models/checkpoints/"))
        / f"{exp_name}_best.pt"
    )
    if ckpt_path.exists():
        trainer.load_checkpoint(str(ckpt_path))

    # ── Evaluate ───────────────────────────────────────────────────
    evaluator = Evaluator(model, config)
    results = evaluator.run(test_loader, return_embeddings=True)
    evaluator.save_report(results, output_dir)

    # ── Confusion matrix ───────────────────────────────────────────
    plot_confusion_matrix(
        results["confusion_matrix"],
        save_path=output_dir / "confusion_matrix.png",
        title=f"Confusion Matrix: {exp_name}",
    )

    # ── ROC curve ──────────────────────────────────────────────────
    plot_roc_curve(
        results["y_true"],
        results["y_prob"],
        experiment_name=exp_name,
        save_path=output_dir / "roc_curve.png",
    )

    # ── t-SNE ──────────────────────────────────────────────────────
    if "embeddings" in results:
        plot_tsne(
            results["embeddings"],
            results["embedding_labels"],
            modality_name="fused",
            save_path=output_dir / "tsne.png",
            title=f"t-SNE: {exp_name}",
        )

    logger.info(f"All outputs saved to {output_dir}")
    return {"metrics": results["metrics"], "history": history}


def run_ablation_study(experiment_configs: Dict[str, str]) -> Dict[str, Dict]:
    """Run multiple experiments and collect results for comparison.

    Args:
        experiment_configs: Dict mapping experiment name → config path.

    Returns:
        Dict mapping experiment name → results dict.
    """
    all_results: Dict[str, Dict] = {}
    for name, cfg_path in experiment_configs.items():
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        try:
            result = run_experiment(cfg_path)
            all_results[name] = result["metrics"]
        except Exception as e:
            print(f"Experiment {name} failed: {e}")
            all_results[name] = {}

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Experiment':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 70)
    for name, metrics in all_results.items():
        acc = f"{metrics.get('accuracy', 0):.4f}"
        f1 = f"{metrics.get('f1', 0):.4f}"
        auc = f"{metrics.get('roc_auc', 0):.4f}"
        print(f"{name:<30} {acc:>10} {f1:>10} {auc:>10}")
    print("=" * 70)

    # Save comparison
    out_path = Path("visualization/outputs/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
