# Mathematical Equations: Fusion, Aggregation and Regression

---

## 1. Speech Encoder Equations

### 1.1 MFCC Feature Extraction

**Z-Score Normalization** (preprocessing.py:71-74)
```
MFCC_norm(i,j) = (MFCC(i,j) - μ_j) / (σ_j + ε)

where:
  i ∈ {1, ..., T_frames}
  j ∈ {1, ..., 40}
  μ_j = mean(MFCC(:, j))
  σ_j = std(MFCC(:, j))
  ε = 1e-8 (numerical stability)
```

### 1.2 BiLSTM Encoder for MFCC

**Forward Pass** (speech_encoder.py:65-78)
```
h_t, c_t = LSTM_cell(x_t, h_{t-1}, c_{t-1})
h_final = [h_forward_T || h_backward_1]  (concatenate)

embedding = LayerNorm(Projection(h_final))
         = LayerNorm(W·[h_fwd || h_bwd] + b)

where:
  x_t ∈ ℝ^40  (MFCC at time t)
  h_t ∈ ℝ^256  (hidden state, bidirectional)
  W ∈ ℝ^(256×512)
  embedding ∈ ℝ^256
```

### 1.3 Wav2Vec2 Embedding with Masking

**Mean Pooling with Attention Mask** (speech_encoder.py:144-159)
```
h = Wav2Vec2(input_values, attention_mask)  ∈ ℝ^(B×T'×768)

pooled = (h ⊙ mask) · 1 / (mask · 1 + δ)
       = Σ_t (h_t · mask_t) / (Σ_t mask_t)

embedding = Projection(pooled)
          = LayerNorm(W·pooled + b)

where:
  ⊙ = element-wise multiplication
  mask ∈ {0,1}^(B×T'×1)  (attention mask)
  W ∈ ℝ^(256×768)
  δ = 1e-9 (numerical stability)
  embedding ∈ ℝ^(B×256)
```

---

## 2. Prosodic Feature Equations

### 2.1 Voice Activity Detection

**RMS-based VAD** (preprocessing.py:115-120)
```
RMS_t = sqrt(mean(frame_t²))

RMS_db = 20 * log10(RMS + 1e-10)

voiced_mask_t = {1  if RMS_db_t > τ
                 0  otherwise

where:
  τ = -40 dB  (silence threshold)
  frame_t ∈ ℝ^400  (25ms frame at 16kHz)
```

### 2.2 Speech Rate

**Definition** (preprocessing.py:130-131)
```
speech_rate = (Σ_t voiced_mask_t) / n_frames
            ∈ [0, 1]

Higher values: more continuous speech (healthy)
Lower values: more pauses/silence (depressed)
```

### 2.3 Pause Duration

**Run-length Analysis** (preprocessing.py:134-138)
```
silence_runs = {run_lengths of contiguous silent frames}
             = [l_1, l_2, ..., l_k]  where l_i > 0

pause_duration = mean(silence_runs) / n_frames
               ∈ [0, 1]

Higher values: longer pauses (depressed)
Lower values: shorter pauses (healthy)
```

### 2.4 Response Latency

**Leading Silence Duration** (preprocessing.py:141-146)
```
first_voiced = argmin_t{t: voiced_mask_t = 1}

response_latency = first_voiced / n_frames
                 ∈ [0, 1]

Higher values: delayed speech onset (depressed)
Lower values: immediate response (healthy)
```

### 2.5 Energy

**RMS Energy of Voiced Frames** (preprocessing.py:149-151)
```
energy = mean({RMS_t : voiced_mask_t = 1})

energy_normalized = clamp(energy, 0, 1)

Higher values: louder speech (healthy)
Lower values: quieter speech (depressed)
```

### 2.6 Pitch Approximation via Zero-Crossing Rate

**Zero-Crossing Rate** (preprocessing.py:154-158)
```
ZCR_t = Σ_{i=0}^{L-1} |sign(frame_t[i]) - sign(frame_t[i+1])|

F0_approx = (ZCR * sample_rate) / (2 * frame_length)
          ∈ [0, 500] Hz

pitch_normalized = F0_approx / 500
                 ∈ [0, 1]

Higher values: higher pitch (healthy)
Lower values: monotone/flat pitch (depressed)
```

### 2.7 Prosodic Feature Projection

**MLP Projection** (speech_encoder.py:28-35)
```
prosodic_raw = [speech_rate, pause_duration, response_latency, energy, pitch]ᵀ
              ∈ ℝ^5

prosodic_emb = ReLU(W₁·prosodic_raw + b₁)
             = Linear(W₂·ReLU(·) + b₂)

where:
  W₁ ∈ ℝ^(128×5),   b₁ ∈ ℝ^128
  W₂ ∈ ℝ^(256×128), b₂ ∈ ℝ^256
  prosodic_emb ∈ ℝ^256
```

---

## 3. Speech Encoder Fusion (Acoustic + Prosodic)

**Concatenation and MLP Fusion** (speech_encoder.py:221-230)
```
acoustic_emb ∈ ℝ^256     (from MFCC BiLSTM or Wav2Vec2)
prosodic_emb ∈ ℝ^256     (from prosodic MLP)

concatenated = [acoustic_emb || prosodic_emb] ∈ ℝ^512

speech_embedding = MLP_fusion(concatenated)
                 = Dropout(GELU(LayerNorm(Linear(concatenated))))
                 ∈ ℝ^256

where:
  Linear: ℝ^512 → ℝ^256
  LayerNorm: normalize mean=0, var=1
  GELU: Gaussian Error Linear Unit activation
  Dropout: p=0.3
```

---

## 4. Multimodal Fusion Equations

### 4.1 Attention Fusion (Default)

**Architecture:** Transformer-style attention over modality tokens

**File:** `fusion/attention_fusion.py` (Lines 1-108)

```
FUSE_token = Parameter(θ_fuse) ∈ ℝ^256
speech_emb ∈ ℝ^256
text_emb ∈ ℝ^256
[eeg_emb ∈ ℝ^256]  (optional)
[facial_emb ∈ ℝ^256]  (optional)

tokens = {FUSE_token ⊕ type_emb_0}
       ∪ {x_i ⊕ type_emb_i for each available modality i}

where:
  ⊕ = element-wise addition
  type_emb_i = learned modality type embedding
  tokens ∈ ℝ^(M×256)  where M = 1 + n_available_modalities
```

**Self-Attention** (fusion/attention_fusion.py:90-106)
```
Z = Transformer(tokens, num_heads=8, num_layers=2)

fused = Z[:, 0, :]  (extract FUSE token output)
      ∈ ℝ^256

Classification_logits = MLP_classifier(fused)
```

**Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
  Q, K, V ∈ ℝ^(M×d_k)
  d_k = 32 (head dimension, total 256/8 = 32)
  output ∈ ℝ^(M×256)
```

### 4.2 Cross-Modal Attention Fusion

**Pairwise Cross-Attention** (fusion/cross_modal_fusion.py)

```
For each modality pair (A, B):
  enriched_A = CrossAttention(query=A, context=B)
  enriched_A ← A + enriched_A  (residual connection)
  enriched_A ← A + MLP(LayerNorm(enriched_A))  (FFN)

enriched_modalities = {enriched_speech, enriched_text, ...}

tokens = stack(enriched_modalities) ∈ ℝ^(N×256)

query_token = Parameter(θ_query) ∈ ℝ^256
fused = MultiheadAttention(query_token, tokens, tokens)
      ∈ ℝ^256
```

### 4.3 Early Fusion

**Concatenation + MLP** (fusion/early_fusion.py)

```
For each modality m ∈ {speech, text, eeg, facial}:
  projected_m = Linear_m(embedding_m)  ∈ ℝ^256
  [or zeros if modality unavailable]

concatenated = [projected_speech || projected_text || ... || projected_facial]
             ∈ ℝ^(256×4)

fused = MLP(concatenated)
      = Dropout(GELU(LayerNorm(Linear(concatenated))))
      ∈ ℝ^256

where:
  Linear: ℝ^(256×4) → ℝ^512
  (hidden layer before final projection)
```

### 4.4 Late Fusion

**Per-Modality Heads + Weighted Average** (fusion/late_fusion.py)

```
For each available modality m:
  head_output_m = MLP_m(embedding_m) ∈ ℝ^256
  
If learnable_weights=True:
  weight_m = softmax(w_m)  where w_m ∈ ℝ  (learned parameter)
  
  fused = Σ_m (weight_m · head_output_m)
        = weighted_average(head_outputs)

If learnable_weights=False:
  fused = mean(head_outputs)

where:
  w_m are learnable scalars initialized to 1
  softmax ensures Σ_m weight_m = 1
```

---

## 5. Classification Head

**MLP Classifier** (models/classifier.py:20-57)

```
For BINARY CLASSIFICATION (task="binary"):
  logits = MLP_binary(fused_embedding)
  
  MLP_binary = Sequential(
    Linear(256 → 512),
    LayerNorm(512),
    GELU(),
    Dropout(0.4),
    Linear(512 → 256),
    LayerNorm(256),
    GELU(),
    Dropout(0.4),
    Linear(256 → 128),
    LayerNorm(128),
    GELU(),
    Dropout(0.4),
    Linear(128 → 2)  (output: 2 class logits)
  )
  
  logits ∈ ℝ^2
  
For REGRESSION (task="regression"):
  score = MLP_regression(fused_embedding)
  
  MLP_regression = Sequential(
    ... same layers ...
    Linear(128 → 1)  (output: continuous PHQ-8 score)
  )
  
  score ∈ ℝ^1
```

---

## 6. Loss Functions

### 6.1 Binary Classification: Cross-Entropy

**Standard Cross-Entropy** (training/losses.py:76-77)

```
L_CE = -Σ_c y_c * log(softmax(logits)_c)

where:
  y ∈ {0,1}^2  (one-hot encoded label)
  logits ∈ ℝ^2
```

### 6.2 Binary Classification: Label Smoothing

**Label Smoothing Cross-Entropy** (training/losses.py:29-48)

```
ỹ_c = {(1 - ε)  if c = true_label
       ε/(C-1)  otherwise

where:
  ε = 0.1  (smoothing factor)
  C = 2    (number of classes)

L_LS = -Σ_c ỹ_c * log(softmax(logits)_c)

Effect: Softens hard labels, prevents overconfidence
```

### 6.3 Binary Classification: Focal Loss

**Focal Loss for Class Imbalance** (training/losses.py:54-70)

```
L_Focal = -α(1 - p_t)^γ * log(p_t)

where:
  p_t = probability of true class
  α = 0.25  (weight factor)
  γ = 2.0   (focusing parameter)

Effect: Emphasizes hard negatives, down-weights easy negatives
Suitable for highly imbalanced datasets
```

### 6.4 Regression: Mean Squared Error

**MSE Loss for PHQ-8 Regression** (training/losses.py:82)

```
L_MSE = (1/B) * Σ_b (ŷ_b - y_b)²

where:
  ŷ_b ∈ [0, 27]  (predicted PHQ-8 score)
  y_b ∈ [0, 27]  (ground truth PHQ-8 score)
  B = batch size
```

**Properties:**
- Penalizes large errors quadratically
- Differentiable everywhere
- Sensitive to outliers

---

## 7. Training Equations

### 7.1 Gradient Descent with Momentum

**AdamW Optimizer** (training/optimizers.py:19-44)

```
m_t ← β₁ * m_{t-1} + (1 - β₁) * ∇L_t
v_t ← β₂ * v_{t-1} + (1 - β₂) * (∇L_t)²

m̂_t = m_t / (1 - β₁^t)  (bias correction)
v̂_t = v_t / (1 - β₂^t)  (bias correction)

θ_t ← θ_{t-1} - α * m̂_t / (√v̂_t + ε) - λ * θ_{t-1}

where:
  α = 1e-4    (learning rate)
  β₁ = 0.9    (momentum)
  β₂ = 0.999  (RMSprop)
  λ = 1e-5    (weight decay)
  ε = 1e-8    (numerical stability)
```

### 7.2 Gradient Clipping

**Norm-based Clipping** (training/trainer.py:180)

```
g ← ∇L
if ||g||₂ > clip_norm:
  g ← g * (clip_norm / ||g||₂)

where:
  clip_norm = 1.0
  ||g||₂ = sqrt(Σ g_i²)

Effect: Prevents gradient explosion in RNNs/Transformers
```

### 7.3 Learning Rate Scheduling: Cosine Annealing

**Cosine Schedule with Warmup** (training/optimizers.py:60-76)

```
For warmup_epochs:
  lr_t = lr_base * (t / warmup_epochs)

For t ≥ warmup_epochs:
  lr_t = lr_min + (lr_base - lr_min) * 
         (1 + cos(π * (t - warmup_epochs) / (T - warmup_epochs))) / 2

where:
  t = current epoch
  T = total epochs
  lr_base = 1e-4
  lr_min = 1e-7
  warmup_epochs = 5
```

---

## 8. Evaluation Metrics (Classification)

### 8.1 Accuracy

**Definition** (evaluation/metrics.py:28)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         ∈ [0, 1]

where:
  TP = True Positives (correctly identified depressed)
  TN = True Negatives (correctly identified not depressed)
  FP = False Positives (healthy misclassified as depressed)
  FN = False Negatives (depressed misclassified as healthy)
```

### 8.2 Precision and Recall

**Precision** (evaluation/metrics.py:31)

```
Precision = TP / (TP + FP)
          ∈ [0, 1]

Interpretation: Of positive predictions, how many correct?
Relevant for: Minimizing false alarms (clinical specificity)
```

**Recall** (evaluation/metrics.py:34)

```
Recall = TP / (TP + FN)
       ∈ [0, 1]

Interpretation: Of actual positives, how many found?
Relevant for: Minimizing missed cases (clinical sensitivity)
```

### 8.3 F1-Score

**Harmonic Mean of Precision and Recall** (evaluation/metrics.py:37)

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ∈ [0, 1]

Balanced metric when classes are imbalanced
```

### 8.4 ROC AUC

**Area Under ROC Curve** (evaluation/metrics.py:40-48)

```
For varying threshold τ:
  TPR(τ) = TP / (TP + FN)
  FPR(τ) = FP / (FP + TN)

AUC = ∫₀¹ TPR(FPR) dFPR
    ∈ [0, 1]

Interpretation:
  0.5 = random classifier
  1.0 = perfect classifier
  > 0.7 = good performance
```

---

## 9. NOT IMPLEMENTED: Regression Metrics

**Missing equations (no implementation found in codebase):**

```
MAE = (1/N) * Σ_i |ŷ_i - y_i|

RMSE = √((1/N) * Σ_i (ŷ_i - y_i)²)

Pearson_r = Cov(y, ŷ) / (σ_y * σ_ŷ)

CCC = 2ρ * σ_y * σ_ŷ / (σ_y² + σ_ŷ² + (μ_y - μ_ŷ)²)

R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
```

**Status in Codebase:** NOT IMPLEMENTED

---

## 10. Summary Table

| Component | Equation Type | File | Lines | Status |
|-----------|---|---|---|---|
| MFCC normalization | Z-score | preprocessing.py | 71-74 | ✓ |
| BiLSTM encoding | RNN | speech_encoder.py | 65-78 | ✓ |
| Wav2Vec2 pooling | Mean with mask | speech_encoder.py | 144-159 | ✓ |
| VAD | Threshold | preprocessing.py | 115-120 | ✓ |
| Prosodic features | Statistical | preprocessing.py | 130-158 | ✓ |
| Speech fusion | Concatenation+MLP | speech_encoder.py | 221-230 | ✓ |
| Attention fusion | Transformer | attention_fusion.py | 1-108 | ✓ |
| Cross-modal fusion | Pairwise attention | cross_modal_fusion.py | 1-120 | ✓ |
| Early fusion | Concat+MLP | early_fusion.py | 1-90 | ✓ |
| Late fusion | Weighted avg | late_fusion.py | 1-85 | ✓ |
| Classification | MLP | classifier.py | 20-57 | ✓ |
| Binary CE | CrossEntropy | losses.py | 76-77 | ✓ |
| Label smoothing | LS-CE | losses.py | 29-48 | ✓ |
| Focal loss | Focal | losses.py | 54-70 | ✓ |
| Regression MSE | MSE | losses.py | 82 | ✓ |
| Accuracy | Binary | metrics.py | 28 | ✓ |
| Precision/Recall | Binary | metrics.py | 31-34 | ✓ |
| F1-Score | Binary | metrics.py | 37 | ✓ |
| ROC AUC | ROC | metrics.py | 40-48 | ✓ |
| MAE | Regression | N/A | N/A | ✗ |
| RMSE | Regression | N/A | N/A | ✗ |
| Pearson r | Correlation | N/A | N/A | ✗ |
| CCC | Regression | N/A | N/A | ✗ |
| R² | Regression | N/A | N/A | ✗ |

