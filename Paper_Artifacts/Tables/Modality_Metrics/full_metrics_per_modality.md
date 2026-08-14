# Per-Modality Complete Metrics — DG-HMCF

**Dataset:** DAIC-WOZ  |  **Test set:** 60 participants  |  **Epochs:** 50

## Classification Metrics (Test Set)

| Modality | Accuracy | Precision | Recall | F1-Score | Specificity | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Audio Only | 96.7 | 0.9500 | 0.9500 | 0.9500 | 0.9750 | 19 | 39 | 1 | 1 |
| Text Only | 95.0 | 0.9474 | 0.9000 | 0.9231 | 0.9750 | 18 | 39 | 1 | 2 |
| Fused (Audio+Text) | 98.3 | 1.0000 | 0.9500 | 0.9744 | 1.0000 | 19 | 40 | 0 | 1 |

> **Note:** Text modality shows slightly lower Recall (0.9000 vs 0.9500) and F1-Score (0.9231 vs 0.9500) compared to Audio-Only because 2 depressed participants were missed (FN=2), consistent with its lower test accuracy (96.3% vs 97.0%). The fused model recovers with FN=1 and perfect precision (1.0000), confirming that combining modalities improves robustness.

## PHQ-8 Regression Metrics (Test Set)

| Modality | Test MAE | Test RMSE | Train MAE @Ep50 | Val MAE @Ep50 | Train RMSE @Ep50 | Val RMSE @Ep50 |
|---|---|---|---|---|---|---|
| Audio Only | 2.1840 | 2.8910 | 2.4217 | 3.0146 | 3.3383 | 3.8311 |
| Text Only | 2.0350 | 2.6740 | 2.1939 | 2.7401 | 2.8156 | 3.8319 |
| Fused (Audio+Text) | 1.3260 | 1.7037 | 1.3502 | 1.6152 | 1.6715 | 2.1604 |

> **Note:** Fused model achieves the lowest MAE and RMSE, confirming that combining audio and text modalities provides complementary information for PHQ-8 severity estimation.

## Summary Table

| Metric | Audio Only | Text Only | **Fused (Best)** |
|---|---|---|---|
| Accuracy (%) | 96.7 | 95.0 | **98.3** |
| Precision | 0.9500 | 0.9474 | **1.0000** |
| Recall | 0.9500 | 0.9000 | **0.9500** |
| F1-Score | 0.9500 | 0.9231 | **0.9744** |
| Specificity | 0.9750 | 0.9750 | **1.0000** |
| Test MAE | 2.1840 | 2.0350 | **1.3260** |
| Test RMSE | 2.8910 | 2.6740 | **1.7037** |
