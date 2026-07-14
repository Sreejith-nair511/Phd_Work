# Classification Metrics: Binary Depression Detection

Dataset: DAIC-WOZ | PHQ-8 Threshold >= 10 | Test: 56 participants
Confusion Matrix: TP=16, TN=34, FP=4, FN=2

| Metric | Value | Formula | Interpretation |
|---|---|---|---|
| Accuracy | 0.8929 | (TP+TN)/(TP+TN+FP+FN) | Higher is better |
| Precision | 0.8000 | TP/(TP+FP) | Higher is better |
| Recall | 0.8889 | TP/(TP+FN) | Higher is better |
| F1-Score | 0.8421 | 2*(Prec*Rec)/(Prec+Rec) | Higher is better |
| ROC AUC | 0.9576 | Area under ROC curve | Higher is better |
| Specificity | 0.8947 | TN/(TN+FP) | Higher is better |
