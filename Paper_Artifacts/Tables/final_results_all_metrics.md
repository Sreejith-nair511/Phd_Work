# Final Results: Complete Metrics

**Dataset:** DAIC-WOZ  
**Training Utterances:** 16,906  
**Validation Utterances:** 6,678  
**Test Participants:** 60  
**Aggregation:** Mean over utterances per participant  

## Split-Level Accuracy

| Split | Accuracy | Loss | MAE | RMSE |
|---|---|---|---|---|
| Train | 98.9% | 0.0672 | 2.4106 | 3.0822 |
| Validation | 98.4% | 0.0710 | 2.4960 | 3.2762 |
| **Test** | **98.3%** | -- | **1.3260** | **1.7037** |

## Classification Metrics (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 98.3% |
| Precision | 1.0000 |
| Recall | 0.9500 |
| F1-Score | 0.9744 |
| Specificity | 1.0000 |
| ROC AUC | 0.9974 |
| TP | 19 |
| TN | 40 |
| FP | 0 |
| FN | 1 |

## Regression Metrics (Test Set — PHQ-8 Severity)

| Metric | Value | Description |
|---|---|---|
| MAE | 1.3260 | Mean Absolute Error |
| RMSE | 1.7037 | Root Mean Squared Error |
| Pearson r | 0.9694 | Linear correlation |
| CCC | 0.9687 | Concordance Correlation Coefficient |
| R2 Score | 0.9356 | Coefficient of Determination |
