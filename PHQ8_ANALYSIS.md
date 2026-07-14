# PHQ-8 Severity Prediction - Source Code Analysis

Complete analysis of the Multimodal Depression Detection Framework based on actual source code review.

---

## 1. Is PHQ-8 Severity Prediction (Regression) Implemented?

**Status:** PARTIALLY IMPLEMENTED (Binary Classification Only, Regression Scaffolding Exists)

The framework has the STRUCTURAL CAPACITY for PHQ-8 regression, but the FULL PIPELINE is NOT fully operational.

### Evidence:

**File:** `models/classifier.py` (Lines 1-57)
```python
class DepressionClassifier(nn.Module):
    """Multi-layer perceptron classifier for depression detection.

    Supports both binary classification and PHQ-8 score regression.  # <-- Claims regression support
    ...
    """
    def __init__(self, ..., num_classes: int = 2, task: str = "binary") -> None:
        ...
        self.task = task  # Can be 'binary' or 'regression'
        layers.append(nn.Linear(in_dim, num_classes))
        # For regression: num_classes=1 outputs (B, 1) raw predictions
        # For binary: num_classes=2 outputs (B, 2) logits
```

**File:** `training/losses.py` (Lines 80-100)
```python
def get_loss_fn(task: str = "binary", ...) -> nn.Module:
    if task == "regression":
        return nn.MSELoss()  # <-- Regression loss is MSELoss (Line 82)
    # Binary classification losses follow...
```

**File:** `configs/base_config.yaml`
```yaml
classifier:
    num_classes: 2
    task: "binary"  # Can be changed to "regression"
```

### What IS Implemented:

1. **Classifier Head:** Can output regression scores (1D) instead of binary logits
2. **Loss Function:** MSE Loss for regression (Line 82, `training/losses.py`)
3. **Config Support:** Task parameter in classifier config
4. **Trainer:** Loads the regression loss if task="regression"

### What IS NOT Implemented:

1. **Utterance-Level PHQ-8 Prediction:** NOT IMPLEMENTED
2. **Utterance Segmentation:** NOT IMPLEMENTED
3. **Participant-Level Aggregation:** NOT IMPLEMENTED
4. **Regression Metrics:** NOT IMPLEMENTED
5. **Regression Evaluation:** NOT IMPLEMENTED
6. **Utterance-to-Participant Pipeline:** NOT IMPLEMENTED

---

## 2. How Are Speech Recordings Preprocessed?

### Speech Audio Processing:

**File:** `dataset/preprocessing.py` (Lines 22-48)

```python
def load_audio(path: str | Path, target_sr: int = 16000, mono: bool = True):
    """Load audio file, resample to target_sr, convert to mono."""
    import torchaudio
    waveform, sr = torchaudio.load(str(path))
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr
```

**Audio Preprocessing Steps:**
1. Load WAV/MP3 file using `torchaudio.load()`
2. Convert to mono (if stereo)
3. Resample to 16 kHz (default)
4. Return tensor shape: `(1, T)` where T = number of samples

### MFCC Feature Extraction:

**File:** `dataset/preprocessing.py` (Lines 51-85)

```python
def extract_mfcc(waveform: torch.Tensor, sr: int = 16000, 
                 n_mfcc: int = 40, n_fft: int = 512, 
                 hop_length: int = 160) -> torch.Tensor:
    """Compute MFCC features from waveform."""
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 80}
    )
    mfcc = transform(waveform)  # (1, 40, T_frames)
    mfcc = mfcc.squeeze(0).T    # (T_frames, 40)
    # Z-score normalization per coefficient
    mean = mfcc.mean(dim=0, keepdim=True)
    std = mfcc.std(dim=0, keepdim=True) + 1e-8
    return (mfcc - mean) / std
```

**MFCC Parameters:**
- 40 MFCC coefficients
- FFT size: 512
- Hop length: 160 samples (10ms at 16kHz)
- Output shape: `(T_frames, 40)` normalized

### Prosodic Feature Extraction:

**File:** `dataset/preprocessing.py` (Lines 88-175)

```python
def extract_prosodic_features(waveform: torch.Tensor, sr: int = 16000,
                              frame_length_ms: int = 25,
                              hop_length_ms: int = 10,
                              silence_threshold_db: float = -40.0):
    """Extract depression-relevant prosodic features."""
    # Returns Dict with:
    return {
        "speech_rate": speech_rate,           # voiced frames / total
        "pause_duration": pause_duration,     # normalized mean pause length
        "response_latency": response_latency, # leading silence duration
        "energy": energy,                      # RMS energy
        "pitch": pitch,                        # Zero-crossing-based F0 proxy
    }
```

**Prosodic Features Extracted:**

| Feature | Calculation | Depression Relevance |
|---------|-------------|----------------------|
| **speech_rate** | voiced_frames / total_frames | Lower in depression |
| **pause_duration** | mean(silent_segment_lengths) / n_frames | Higher pauses in depression |
| **response_latency** | first_voiced_frame / n_frames | Delayed response in depression |
| **energy** | mean(RMS[voiced_frames]) | Lower energy in depression |
| **pitch** | zero_crossing_rate / (2 * frame_len) | Lower pitch in depression |

### Speech Preprocessing LIMITATIONS:

**What IS Implemented:**
- Whole-utterance preprocessing (entire AUDIO file at once)
- No utterance-level segmentation

**What IS NOT Implemented:**
- Speech Utterance Segmentation (continuous audio → multiple utterances)
- Timestamp-based segment extraction
- Per-utterance label assignment
- Voice Activity Detection (VAD)
- Silence-based segmentation

---

## 3. During Inference, How Are Utterance-Level Predictions Aggregated?

**Status:** NOT IMPLEMENTED

The framework processes ENTIRE AUDIO FILES as single samples. There is NO utterance segmentation and therefore NO utterance-level aggregation.

### Current Approach:

**File:** `dataset/daic_woz_dataset.py` (Lines 145-182)

```python
def _get_speech_input(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
    sample = self.samples[idx]
    audio_path = sample.get("audio_path")
    
    waveform, sr = load_audio(audio_path, target_sr=self.sample_rate)
    
    # Truncate/pad to max_audio_len seconds
    max_samples = self.max_audio_len * self.sample_rate  # 300 seconds default
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]  # Truncate to 5 minutes
    else:
        pad_len = max_samples - waveform.shape[-1]
        waveform = F.pad(waveform, (0, pad_len))  # Pad with silence
    
    # Extract features from ENTIRE audio
    result["mfcc"] = extract_mfcc(waveform, ...)  # Shape: (T_frames, 40)
    result.update(extract_prosodic_features(waveform, ...))
```

### Key Limitation:

**ONE AUDIO FILE → ONE PHQ-8 LABEL**

There is NO mechanism for:
- Dividing the audio into utterances
- Computing per-utterance predictions
- Aggregating utterance predictions

### What Would Be Needed for Utterance Aggregation:

1. **Utterance Segmentation:** Silence detection or VAD
2. **Per-Utterance Features:** Extract MFCC/prosodic for each utterance
3. **Per-Utterance Predictions:** Forward each utterance through the model
4. **Aggregation Functions:**
   - Mean averaging
   - Weighted averaging (by utterance confidence)
   - Attention pooling
   - Median/mode selection
5. **Aggregation Implementation:** Would require new module in `fusion/` or evaluation pipeline

---

## 4. Where Is Participant-Level PHQ-8 Score Computed?

**Status:** NOT COMPUTED

Participant-level PHQ-8 scores are NOT computed. The system operates at the SAMPLE level (one audio file = one participant).

### Current Data Flow:

**File:** `dataset/daic_woz_dataset.py` (Lines 80-110)

```python
def _load_metadata(self) -> None:
    """Load metadata from CSV."""
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("Participant_ID")
            phq8_score = float(row.get("PHQ8_Score"))
            label = int(phq8_score >= self.phq8_threshold)  # Binary: 0 or 1
            self.samples.append({
                "participant_id": str(pid),
                "phq8_score": phq8_score,  # <-- Ground truth (not computed)
                "label": label,             # <-- Binary label
                ...
            })
```

**The PHQ-8 score comes from the DATASET CSV, not from model prediction.**

### Model Output:

**File:** `training/trainer.py` (Lines 178-179)

```python
with autocast(enabled=self.mixed_precision):
    logits, _ = self.model(batch_inputs)  # For binary: shape (B, 2)
                                          # For regression: shape (B, 1)
    loss = self.criterion(logits, labels)
```

**Labels are indices (0 or 1) for binary classification, not PHQ-8 scores.**

### Inference Stage:

**File:** `evaluation/evaluator.py` (Lines 60-65)

```python
probs = torch.softmax(logits, dim=-1)    # (B, C=2)
preds = probs.argmax(dim=-1)             # (B,) - predicted class index
all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # P(depressed)
```

**Output is probability of depression, NOT PHQ-8 severity score.**

---

## 5. Which Regression Loss Function Is Used?

**Implemented:** MSE Loss Only

**File:** `training/losses.py` (Lines 80-82)

```python
def get_loss_fn(task: str = "binary", loss_type: str = "label_smoothing", 
                smoothing: float = 0.1, class_weights = None):
    if task == "regression":
        return nn.MSELoss()  # <-- Only regression loss
    # Binary classification losses follow...
```

**MSE Loss Definition:**
```
Loss = Mean((y_pred - y_true)^2)
```

### Regression Loss Functions NOT Implemented:

- MAE (Mean Absolute Error) - NOT implemented
- Huber Loss - NOT implemented
- Smooth L1 Loss - NOT implemented
- Any robust regression loss - NOT implemented

### Regression Loss Characteristics:

| Loss Type | Implemented | Sensitivity to Outliers |
|-----------|-------------|------------------------|
| MSE | YES | High (quadratic) |
| MAE | NO | Robust |
| Huber | NO | Balanced |
| Smooth L1 | NO | Balanced |

---

## 6. Which Regression Evaluation Metrics Are Calculated?

**Status:** NOT IMPLEMENTED

The metrics module (evaluation/metrics.py) calculates ONLY CLASSIFICATION METRICS.

**File:** `evaluation/metrics.py` (Lines 20-56)

```python
def compute_metrics(y_true, y_pred, y_prob=None, average="binary"):
    """Compute standard CLASSIFICATION metrics."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(...)),
        "recall": float(recall_score(...)),
        "f1": float(f1_score(...)),
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    
    return metrics
```

### Regression Metrics NOT Implemented:

| Metric | Status | Would Calculate |
|--------|--------|-----------------|
| MAE | NOT IMPLEMENTED | mean(abs(y_pred - y_true)) |
| RMSE | NOT IMPLEMENTED | sqrt(mean((y_pred - y_true)^2)) |
| Pearson Correlation | NOT IMPLEMENTED | pearsonr(y_pred, y_true) |
| CCC | NOT IMPLEMENTED | Concordance correlation coefficient |
| R² Score | NOT IMPLEMENTED | 1 - SS_res/SS_tot |

### Classification Metrics Implemented (for binary task only):

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

---

## 7. Complete PHQ-8 Prediction Pipeline

### ACTUAL IMPLEMENTED PIPELINE (Binary Classification):

```
Participant Audio File
        ↓
[load_audio] (preprocessing.py:22)
    Convert to mono, resample to 16kHz
        ↓
Waveform (1, T)
        ↓
[extract_mfcc] (preprocessing.py:51)
    40 MFCC coefficients, normalized
        ↓
MFCC Features (T_frames, 40)
        ↓
[extract_prosodic_features] (preprocessing.py:88)
    5 prosodic features: speech_rate, pause_duration, 
    response_latency, energy, pitch
        ↓
Feature Dictionary
        ↓
[SpeechEncoder] (encoders/speech_encoder.py)
    BiLSTM or Wav2Vec2 encoder
    Fuses acoustic + prosodic features
        ↓
Speech Embedding (B, 256)
        ↓
[TextEncoder] (encoders/text_encoder.py)
    RoBERTa/BERT tokenization and encoding
        ↓
Text Embedding (B, 256)
        ↓
[Fusion Layer] (fusion/*.py)
    Attention-based or Late/Early fusion
        ↓
Fused Representation (B, 256)
        ↓
[DepressionClassifier] (models/classifier.py)
    MLP: 256 → 512 → 256 → 128 → 2 (binary)
        ↓
Logits (B, 2)
        ↓
[Binary Prediction]
    Class 0: Not Depressed
    Class 1: Depressed
        ↓
Final Output: Binary Label (0 or 1)
```

### MISSING STEPS FOR FULL PHQ-8 REGRESSION:

```
NOT IMPLEMENTED IN CURRENT CODEBASE:
    ├─ Utterance Segmentation (split audio into segments)
    ├─ Per-Utterance Feature Extraction
    ├─ Per-Utterance Model Inference
    ├─ Utterance-Level PHQ-8 Prediction (regression)
    ├─ Utterance Aggregation
    └─ Participant-Level PHQ-8 Score
```

---

## 8. Summary: What IS and IS NOT Implemented

### IMPLEMENTED:

1. **Binary Depression Classification** - FULL
2. **Speech Feature Extraction** - FULL
   - MFCC extraction with normalization
   - Prosodic features (5 types)
   - Full audio processing pipeline
3. **Multimodal Fusion** - FULL
   - 4 fusion strategies implemented
   - Handles missing modalities
4. **Training Pipeline** - FULL (for binary classification)
   - Mixed precision training
   - Early stopping
   - Checkpointing
5. **Evaluation** - PARTIAL
   - Classification metrics: ✓
   - Regression metrics: ✗
6. **Visualization** - FULL (for classification)
   - Training curves
   - Confusion matrix
   - ROC curve
   - t-SNE embeddings

### NOT IMPLEMENTED:

1. **PHQ-8 Regression Inference** - NOT IMPLEMENTED
2. **Utterance Segmentation** - NOT IMPLEMENTED
3. **Per-Utterance Predictions** - NOT IMPLEMENTED
4. **Utterance Aggregation** - NOT IMPLEMENTED
5. **Regression Metrics** - NOT IMPLEMENTED
   - No MAE, RMSE, Pearson, CCC, R² Score
6. **Regression Visualization** - NOT IMPLEMENTED
7. **Participant-Level Aggregation** - NOT IMPLEMENTED

---

## WHAT YOU NEED TO IMPLEMENT FOR FULL PHQ-8 REGRESSION:

### 1. Utterance Segmentation Module

**Create:** `dataset/utterance_segmentation.py`
```python
def segment_by_silence(audio, sr, threshold_db=-40):
    """Split audio into utterances based on silence."""
    # Voice Activity Detection
    # Returns: List[Tuple[start_sample, end_sample]]

def segment_by_transcript_timestamps(audio, sr, transcript_df):
    """Split audio using transcript timestamps."""
    # Requires: transcript with start_time, stop_time
    # Returns: List[np.ndarray] - one audio chunk per utterance
```

### 2. Per-Utterance Inference

**Modify:** `models/multimodal_model.py`
```python
def forward_utterance_batch(self, utterances_batch, 
                            utterance_lengths):
    """Forward pass for batch of variable-length utterances."""
    # Must handle variable utterance lengths
    # Return: (utterance_embeddings, utterance_predictions)
```

### 3. Utterance Aggregation

**Create:** `fusion/utterance_aggregation.py`
```python
class UtteranceAggregator(nn.Module):
    """Aggregate utterance-level predictions to participant level."""
    
    def forward(self, utterance_preds, utterance_embeddings):
        # Options:
        # 1. Mean pooling
        # 2. Attention pooling
        # 3. Weighted by utterance confidence
        return participant_phq8_score
```

### 4. Regression Metrics

**Expand:** `evaluation/metrics.py`
```python
def compute_regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "pearson_r": pearsonr(y_true, y_pred)[0],
        "ccc": compute_ccc(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
```

### 5. Update Trainer for Regression

**Modify:** `training/trainer.py`
```python
def _train_epoch_regression(self, epoch):
    """Handle regression task with continuous PHQ-8 targets."""
    # Use regression metrics instead of accuracy
    # Handle continuous targets instead of class indices
```

### 6. Update Evaluator for Regression

**Modify:** `evaluation/evaluator.py`
```python
def run_regression(self, loader):
    """Inference for regression task."""
    all_preds_raw = []
    for logits, _ in model(batch):
        all_preds_raw.extend(logits.squeeze().cpu().numpy())
    
    return compute_regression_metrics(y_true, all_preds_raw)
```

---

## RECOMMENDED IMPLEMENTATION PRIORITY:

1. **Utterance Segmentation** (critical)
2. **Regression Metrics** (quick win)
3. **Per-Utterance Inference** (moderate complexity)
4. **Utterance Aggregation** (moderate complexity)
5. **Update Trainer/Evaluator** (refactoring)

---

## CONCLUSION:

The framework provides an excellent FOUNDATION for PHQ-8 regression but LACKS the UTTERANCE-LEVEL PIPELINE. It currently operates as a **PARTICIPANT-LEVEL BINARY CLASSIFIER** using full-audio features.

To achieve top-class regression research, you need to implement utterance segmentation and aggregation, which is beyond the current scope but architecturally feasible within this framework.
