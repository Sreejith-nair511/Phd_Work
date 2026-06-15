# Multimodal Depression Detection Framework

A research-grade, modality-agnostic framework for automated depression detection
using Speech, Text, EEG, and Facial modalities. Built for PhD-level research with
a focus on reproducibility, extensibility, and publication-ready evaluation.

---

## Overview

Depression affects over 280 million people globally and remains underdiagnosed.
This framework provides a unified architecture to detect depression from clinical
interview data using any combination of four modalities. Missing modalities are
handled natively at the architecture level, requiring no changes to the core model.

---

## Architecture

```
Input Modalities
    Speech      Text        EEG         Facial
       |           |          |            |
  SpeechEncoder  TextEncoder EEGEncoder FacialEncoder
  (MFCC/Wav2Vec2)(RoBERTa/BERT)(CNN/Transformer/BiLSTM)(CNN/ViT)
       |           |          |            |
       +-----+-----+----+-----+
                   |
            Fusion Layer
     (Early | Late | Attention | Cross-Modal)
                   |
          Depression Classifier
          (Binary / PHQ-8 Score)
```

Each encoder produces a fixed-size embedding (default 256-d). The fusion layer
accepts any subset of available modalities and produces a single representation
for the classifier. NULL modality inputs are silently ignored at every stage.

---

## Supported Modality Combinations

Any of the 15 subsets of {Speech, Text, EEG, Facial} are supported out of the box.
Enabling or disabling a modality requires only a one-line config change.

---

## Repository Structure

```
configs/                  YAML configuration files
    base_config.yaml      Master config with all defaults
    experiments/          Per-experiment overrides (A through F)
dataset/                  Dataset loaders and preprocessing
    base_dataset.py       Abstract base dataset class
    daic_woz_dataset.py   DAIC-WOZ loader (Speech + Text + PHQ-8)
    modma_dataset.py      MODMA loader (Speech + EEG)
    preprocessing.py      Audio, text, and feature preprocessing
    collate.py            Custom collate for missing modality batches
    dataset_factory.py    Build DataLoaders from config
encoders/                 Modality-specific encoder modules
    base_encoder.py       Abstract encoder interface
    speech_encoder.py     MFCC / Wav2Vec2 + prosodic features
    text_encoder.py       RoBERTa / BERT sentence encoder
    eeg_encoder.py        CNN / Transformer / BiLSTM for EEG
    facial_encoder.py     CNN / ViT + landmark MLP for facial
    encoder_factory.py    Build encoders from config
fusion/                   Fusion strategy implementations
    early_fusion.py       Concatenation + MLP
    late_fusion.py        Per-modality heads + weighted average
    attention_fusion.py   Transformer attention over modality tokens
    cross_modal_fusion.py Pairwise cross-attention + aggregation
    fusion_factory.py     Build fusion module from config
models/                   Core model and classifier
    multimodal_model.py   End-to-end orchestration model
    classifier.py         MLP classifier / regression head
training/                 Training loop, losses, optimizers
    trainer.py            Training loop with early stopping, checkpointing
    losses.py             Cross-entropy, label smoothing, focal loss
    optimizers.py         AdamW with layer-wise LR + schedulers
    experiment_runner.py  Single experiment and ablation study runner
evaluation/               Metrics and evaluation pipeline
    metrics.py            Accuracy, Precision, Recall, F1, ROC AUC
    evaluator.py          Inference + report generation
visualization/            Plotting utilities
    training_curves.py    Loss and accuracy curves
    confusion_matrix.py   Heatmap visualization
    tsne_plot.py          t-SNE embedding visualization
    attention_viz.py      Attention weights and feature importance
    roc_curve.py          ROC curve (single and multi-experiment)
utils/                    Shared utilities
    config.py             YAML config loader with inheritance
    seed.py               Full reproducibility seed setting
    logger.py             Structured logging to console and file
    device.py             GPU / CPU device selection
run_experiments.py        CLI entry point for all experiments
requirements.txt          Python dependencies
```

---

## Experiments

| ID | Modalities                        | Fusion        |
|----|-----------------------------------|---------------|
| A  | Speech only                       | Attention     |
| B  | Text only                         | Attention     |
| C  | Speech + Text                     | Attention     |
| D  | Speech + EEG                      | Attention     |
| E  | Speech + Text + EEG               | Cross-Modal   |
| F  | Speech + Text + EEG + Facial      | Cross-Modal   |

---

## Datasets

**DAIC-WOZ** (Gratch et al., 2014)
- Speech audio and transcripts from clinical depression interviews.
- PHQ-8 questionnaire scores as ground-truth labels.
- Modalities: Speech, Text.
- Download: https://dcapswoz.ict.usc.edu

**MODMA** (Cai et al., 2020)
- 128-channel EEG and audio recordings.
- Binary depression / healthy control labels.
- Modalities: Speech, EEG.
- Download: https://modma.lzu.edu.cn

Place datasets under `dataset/daic_woz/` and `dataset/modma/` respectively.
See comments inside each dataset loader for the expected directory structure.

---

## Installation

```bash
git clone https://github.com/Sreejith-nair511/Phd_Work.git
cd Phd_Work
pip install -r requirements.txt
```

Python 3.10 or later is recommended. A CUDA-capable GPU is strongly advised.

---

## Quick Start

Run all six experiments with default settings:

```bash
python run_experiments.py
```

Run a single experiment:

```bash
python run_experiments.py --exp A_speech_only
python run_experiments.py --exp F_all_modalities
```

Run with a custom config:

```bash
python run_experiments.py --config configs/experiments/speech_text.yaml
```

---

## Configuration

All behaviour is controlled through YAML config files. Experiment configs inherit
from `configs/base_config.yaml` and override only the relevant fields.

Key parameters:

```yaml
modalities:
  speech: true
  text: true
  eeg: false
  facial: false

encoders:
  embedding_dim: 256
  speech:
    feature_type: wav2vec2   # or mfcc
  text:
    model_name: roberta-base

fusion:
  type: attention            # early | late | attention | cross_modal

training:
  epochs: 50
  batch_size: 16
  learning_rate: 1.0e-4
```

---

## Evaluation Outputs

After each experiment the following are saved under `visualization/outputs/<exp_name>/`:

- `metrics.json` - Accuracy, Precision, Recall, F1, ROC AUC
- `classification_report.txt` - Full sklearn report
- `confusion_matrix.png` - Normalized heatmap
- `roc_curve.png` - ROC curve with AUC
- `training_curves.png` - Loss, accuracy, and learning rate
- `tsne.png` - t-SNE of fused embeddings
- `ablation_results.json` - Cross-experiment comparison table

---

## Research Features

- **Missing modality handling**: Any modality can be None at inference time without model changes.
- **Attention visualization**: Modality-level attention weights for interpretability.
- **t-SNE embedding visualization**: Inspect learned representations per experiment.
- **Feature importance**: Modality contribution scoring via ablation.
- **Ablation studies**: Built-in multi-experiment comparison runner.
- **Reproducible**: Fixed seeds for Python, NumPy, and PyTorch including CUDNN.
- **Mixed precision**: Automatic FP16 training with gradient scaling.
- **Weighted sampling**: Class imbalance handled via inverse-frequency weighting.

---

## Extending the Framework

Adding a new dataset:
1. Subclass `BaseDepressionDataset` in `dataset/`.
2. Implement `_load_metadata` and the four modality loader methods.
3. Register the class in `dataset/dataset_factory.py`.

Adding a new encoder:
1. Subclass `BaseEncoder` in `encoders/`.
2. Implement the `encode` method returning `(B, output_dim)`.
3. Register in `encoders/encoder_factory.py`.

Adding a new fusion strategy:
1. Subclass `BaseFusion` in `fusion/`.
2. Implement the `fuse` method.
3. Register in `fusion/fusion_factory.py`.

---

## Citation

If you use this framework in your research, please cite:

```
@misc{multimodal_depression_2026,
  title  = {Modality-Agnostic Multimodal Depression Detection Framework},
  author = {Sreejith Nair},
  year   = {2026},
  url    = {https://github.com/Sreejith-nair511/Phd_Work}
}
```

---

## License

MIT License. See LICENSE file for details.
