# IEEE Table 1: Encoder Specifications and Configuration

| **Encoder** | **Input Type** | **Input Shape** | **Backbone** | **Hidden Dim** | **Output Dim** | **Dropout** | **Normalization** | **Depression Feature** |
|---|---|---|---|---|---|---|---|---|
| **Speech (MFCC)** | MFCC Features | (B, T_frames, 40) | BiLSTM (2L, bidirectional) | 512 | 256 | 0.3 | LayerNorm | Spectral envelope, formants |
| **Speech (Wav2Vec2)** | Raw Waveform | (B, T) | Transformer (pre-trained) | 768 | 256 | 0.3 | LayerNorm | Acoustic representations |
| **Prosodic** | 5 scalars | (B, 5) | MLP (2 layers) | 256 | 256 | 0 | - | Speech rate, energy, pitch |
| **Speech Fused** | Acoustic+Prosodic | 512-d concat | MLP (Linear+GeLU) | - | 256 | 0.3 | LayerNorm | Combined acoustic+paralinguistic |
| **Text** | Tokenized Text | (B, T_tokens, 768) | RoBERTa/BERT | 768 | 256 | 0.3 | LayerNorm | Linguistic content, psycholinguistics |
| **EEG (CNN)** | EEG Channels×Time | (B, 64, 1000) | Conv1D Blocks | 512 | 256 | 0.3 | BatchNorm | Oscillatory brain activity |
| **EEG (Transformer)** | EEG Channels×Time | (B, 64, 1000) | Transformer | 512 | 256 | 0.3 | LayerNorm | Attention over EEG frequency bands |
| **EEG (BiLSTM)** | EEG Channels×Time | (B, 1000, 64) | BiLSTM (4L) | 512 | 256 | 0.3 | LayerNorm | Temporal EEG sequences |
| **Facial (CNN)** | Video Frames | (B, T_frames, 3, 224, 224) | ResNet-inspired | 512 | 256 | 0.3 | BatchNorm | Facial expressions, micro-expressions |
| **Facial (ViT)** | Video Frames | (B, T_frames, 3, 224, 224) | Vision Transformer | 512 | 256 | 0.3 | LayerNorm | Facial patches and relationships |
| **Facial (Landmarks)** | 2D Landmarks | (B, T_frames, 68×2) | MLP | 512 | 256 | 0.3 | LayerNorm | Action units, facial muscle tension |

**Notes:**
- All encoders output fixed 256-d embeddings for fusion compatibility
- Dropout = 0.3 except Prosodic (scalar features, no dropout needed)
- Speech encoder combines Acoustic (256-d) + Prosodic (256-d) → 512-d → 256-d
- EEG/Facial support multiple encoder types selected at runtime via config
- Pre-training: Wav2Vec2 (960h LibriSpeech), RoBERTa (multilingual corpus), Vision models (ImageNet)

---

# IEEE Table 2: Fusion Strategy Comparison

| **Fusion Type** | **Implementation** | **Missing Modality Handling** | **Complexity** | **Learnable Parameters** | **Attention** | **Cross-Modal** | **File** | **Lines** |
|---|---|---|---|---|---|---|---|---|
| **Early** | Concat→MLP | Zero-padding | O(4D²) | W₁∈ℝ^(4D×2D), W₂∈ℝ^(2D×D) | ✗ | ✗ | fusion/early_fusion.py | 1-90 |
| **Late** | Per-modality heads → weighted avg | Zero-padding | O(4D²) | W_i∈ℝ^(D×D) + learned weights | ✗ | ✗ | fusion/late_fusion.py | 1-85 |
| **Attention** | Transformer over tokens | [FUSE] token only | O(M²D) | Transformer params | ✓ | ✗ | fusion/attention_fusion.py | 1-108 |
| **Cross-Modal** | Pairwise + self-attention | Sparse attention | O(M³D) | All pairwise modules | ✓ | ✓ | fusion/cross_modal_fusion.py | 1-120 |

**Where:**
- D = embedding dimension (256)
- M = number of available modalities (1-4)
- ✓ = implemented, ✗ = not implemented

---

# IEEE Table 3: Loss Functions and Training Configuration

| **Loss Function** | **Task** | **Formula** | **Parameters** | **File** | **Line** | **Implementation** |
|---|---|---|---|---|---|---|
| **CrossEntropy** | Binary Classification | L_CE = -Σ_c y_c log(softmax(logits)_c) | - | training/losses.py | 76-77 | ✓ Implemented |
| **LabelSmoothing** | Binary Classification | L_LS = -Σ_c ỹ_c log(softmax(logits)_c) where ỹ smooth | ε=0.1 | training/losses.py | 29-48 | ✓ Implemented |
| **Focal Loss** | Class Imbalance | L_Focal = -α(1-p_t)^γ log(p_t) | α=0.25, γ=2.0 | training/losses.py | 54-70 | ✓ Implemented |
| **MSE Loss** | Regression (PHQ-8) | L_MSE = (1/B)Σ(ŷ_i - y_i)² | - | training/losses.py | 82 | ✓ Scaffold only |
| **MAE Loss** | Robust Regression | L_MAE = (1/B)Σ\|ŷ_i - y_i\| | - | - | - | ✗ NOT IMPLEMENTED |
| **Huber Loss** | Balanced Regression | L_Huber (piecewise) | δ | - | - | ✗ NOT IMPLEMENTED |
| **Smooth L1** | Regression | L_SL1 (piecewise) | - | - | - | ✗ NOT IMPLEMENTED |

| **Component** | **Value** | **File** | **Lines** |
|---|---|---|---|
| **Learning Rate (Base)** | 1e-4 | training/optimizers.py | 19 |
| **Learning Rate (Encoder)** | 1e-5 (10×smaller) | training/optimizers.py | 39 |
| **Weight Decay** | 1e-5 | training/optimizers.py | 20 |
| **Batch Size** | 16 | configs/base_config.yaml | 10 |
| **Epochs** | 50 | configs/base_config.yaml | 8 |
| **Gradient Clip** | 1.0 | training/trainer.py | 180 |
| **Early Stopping Patience** | 10 epochs | configs/base_config.yaml | 13 |
| **Warmup Epochs** | 5 | configs/base_config.yaml | 11 |
| **Scheduler** | Cosine Annealing | training/optimizers.py | 60-76 |

---

# IEEE Table 4: Evaluation Metrics (Binary Classification)

| **Metric** | **Formula** | **Range** | **Interpretation** | **File** | **Line** | **Status** |
|---|---|---|---|---|---|---|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | [0,1] | Overall correctness | evaluation/metrics.py | 28 | ✓ |
| **Precision** | TP/(TP+FP) | [0,1] | Positive Predictive Value | evaluation/metrics.py | 31 | ✓ |
| **Recall** | TP/(TP+FN) | [0,1] | Sensitivity / True Positive Rate | evaluation/metrics.py | 34 | ✓ |
| **F1-Score** | 2·(Prec·Rec)/(Prec+Rec) | [0,1] | Harmonic mean Prec-Recall | evaluation/metrics.py | 37 | ✓ |
| **ROC AUC** | ∫TPR(FPR)dFPR | [0,1] | Area under ROC curve | evaluation/metrics.py | 40-48 | ✓ |
| **MAE** | (1/N)Σ\|ŷ_i - y_i\| | [0,∞] | Mean absolute error | - | - | ✗ NOT IMPL. |
| **RMSE** | √((1/N)Σ(ŷ_i - y_i)²) | [0,∞] | Root mean squared error | - | - | ✗ NOT IMPL. |
| **Pearson r** | Cov(y,ŷ)/(σ_y·σ_ŷ) | [-1,1] | Linear correlation | - | - | ✗ NOT IMPL. |
| **CCC** | 2ρσ_yσ_ŷ/(...) | [-1,1] | Concordance correlation | - | - | ✗ NOT IMPL. |
| **R²** | 1-(SS_res/SS_tot) | (-∞,1] | Coefficient of determination | - | - | ✗ NOT IMPL. |

**Legend:** TP=True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives

---

# IEEE Table 5: Dataset Specifications

| **Dataset** | **Modality** | **Samples** | **Depression Labels** | **PHQ-8 Range** | **Speaker Duration** | **File** | **Status** |
|---|---|---|---|---|---|---|---|
| **DAIC-WOZ** | Speech + Text | ~189 | Binary (PHQ-8≥10) | 0-27 | 15-20 min | dataset/daic_woz_dataset.py | ✓ Implemented |
| **MODMA** | Speech + EEG | Variable | Binary | Inferred | Variable | dataset/modma_dataset.py | ✓ Implemented |
| **PDCH** | (All modalities) | Future | TBD | TBD | TBD | dataset/pdch_dataset.py | ✗ Placeholder |

| **Component** | **Value** | **File** | **Line(s)** |
|---|---|---|---|
| **Train Split** | 70% | configs/base_config.yaml | 18 |
| **Val Split** | 15% | configs/base_config.yaml | 18 |
| **Test Split** | 15% | configs/base_config.yaml | 18 |
| **Class Weighting** | Enabled | configs/base_config.yaml | 14 |
| **Data Augmentation** | Yes (train only) | configs/base_config.yaml | 16 |
| **Feature Caching** | Enabled | configs/base_config.yaml | 17 |
| **PHQ-8 Threshold** | 10 (binary) | configs/base_config.yaml | 19 |

---

# IEEE Table 6: Source Code Reference Index

| **Component** | **File** | **Class/Function** | **Lines** | **Type** |
|---|---|---|---|---|
| Audio Loading | dataset/preprocessing.py | load_audio | 22-48 | Function |
| MFCC Extraction | dataset/preprocessing.py | extract_mfcc | 51-85 | Function |
| Prosodic Features | dataset/preprocessing.py | extract_prosodic_features | 88-175 | Function |
| Speech Encoder | encoders/speech_encoder.py | SpeechEncoder | 157-235 | Class |
| Text Encoder | encoders/text_encoder.py | TextEncoder | 1-100+ | Class |
| EEG Encoder | encoders/eeg_encoder.py | EEGEncoder | 1-200+ | Class |
| Facial Encoder | encoders/facial_encoder.py | FacialEncoder | 1-250+ | Class |
| Attention Fusion | fusion/attention_fusion.py | AttentionFusion | 1-108 | Class |
| Early Fusion | fusion/early_fusion.py | EarlyFusion | 1-90 | Class |
| Late Fusion | fusion/late_fusion.py | LateFusion | 1-85 | Class |
| Cross-Modal Fusion | fusion/cross_modal_fusion.py | CrossModalFusion | 1-120 | Class |
| Classifier | models/classifier.py | DepressionClassifier | 1-57 | Class |
| Multimodal Model | models/multimodal_model.py | MultimodalDepressionModel | 1-180+ | Class |
| Trainer | training/trainer.py | Trainer | 1-300+ | Class |
| Losses | training/losses.py | get_loss_fn | 1-100 | Module |
| Optimizer | training/optimizers.py | build_optimizer | 1-100 | Module |
| Evaluator | evaluation/evaluator.py | Evaluator | 1-150 | Class |
| Metrics | evaluation/metrics.py | compute_metrics | 20-56 | Function |
| Visualization | visualization/ | plot_* functions | 1-70 ea | Functions |

---

# IEEE Table 7: Depression-Relevant Features Extracted

| **Modality** | **Feature** | **Extraction Method** | **Depression Indicator** | **Literature** | **Implementation** |
|---|---|---|---|---|---|
| **Speech** | Speech Rate | Voiced frames / total | Reduced rate → depression | Scherer et al., 2016 | ✓ |
| **Speech** | Pause Duration | Mean silence length | Longer pauses → depression | Darby & Hollien, 1977 | ✓ |
| **Speech** | Response Latency | Leading silence | Delayed onset → depression | Trevino et al., 2015 | ✓ |
| **Speech** | Energy (RMS) | Mean amplitude (voiced) | Lower energy → depression | Moore et al., 2003 | ✓ |
| **Speech** | Pitch (F0) | Zero-crossing rate proxy | Reduced pitch variation → depression | Mundt et al., 2007 | ✓ |
| **Speech** | MFCC 1-13 | Spectral envelope | Formant changes in depression | Cannizzaro et al., 2004 | ✓ |
| **Text** | Psycholinguistic | RoBERTa embeddings | First-person pronouns ↑ | Pennebaker et al., 2003 | ✓ |
| **EEG** | Oscillatory | CNN/Transformer | Increased theta, decreased alpha | Thibault et al., 2018 | ✓ |
| **Facial** | Expressions | CNN/ViT | Reduced smile duration | Jap et al., 2011 | ✓ |
| **Facial** | Landmarks | MLP | Eyebrow and mouth movement ↓ | Pampouchidou et al., 2015 | ✓ |

