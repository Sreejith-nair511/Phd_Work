"""Run all defined experiments (A through F) and compare results.

Usage:
    python run_experiments.py                        # run all experiments
    python run_experiments.py --exp speech_only      # run a single experiment
    python run_experiments.py --ablation             # print comparison table only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.experiment_runner import run_ablation_study, run_experiment
from utils.seed import set_seed

EXPERIMENTS = {
    "A_speech_only": "configs/experiments/speech_only.yaml",
    "B_text_only": "configs/experiments/text_only.yaml",
    "C_speech_text": "configs/experiments/speech_text.yaml",
    "D_speech_eeg": "configs/experiments/speech_eeg.yaml",
    "E_speech_text_eeg": "configs/experiments/speech_text_eeg.yaml",
    "F_all_modalities": "configs/experiments/all_modalities.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Depression Detection - Experiment Runner"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Run a single experiment by key (e.g. A_speech_only)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config YAML file",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all experiments and print comparison",
    )
    args = parser.parse_args()

    set_seed(42)

    if args.config:
        run_experiment(args.config)
    elif args.exp:
        if args.exp not in EXPERIMENTS:
            print(f"Unknown experiment '{args.exp}'. Available: {list(EXPERIMENTS.keys())}")
            return
        run_experiment(EXPERIMENTS[args.exp])
    else:
        # Default: run full ablation
        run_ablation_study(EXPERIMENTS)


if __name__ == "__main__":
    main()
