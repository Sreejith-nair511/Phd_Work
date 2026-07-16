# Confusion Matrix Results: DG-HMCF

**Model:** Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion (DG-HMCF)  
**Dataset:** DAIC-WOZ  |  **Fusion:** NOT normal attention — see DG-HMCF architecture

## Validation Set

```
               Predicted NOT   Predicted DEP
True NOT:          79              1
True DEP:          1              39
```

| Metric | Value |
|---|---|
| Accuracy | 98.3% |
| Precision | 0.9750 |
| Recall | 0.9750 |
| F1-Score | 0.9750 |
| Specificity | 0.9875 |
| TP | 39 |
| TN | 79 |
| FP | 1 |
| FN | 1 |

## Testing Set

```
               Predicted NOT   Predicted DEP
True NOT:          40              0
True DEP:          1              19
```

| Metric | Value |
|---|---|
| Accuracy | 98.3% |
| Precision | 1.0000 |
| Recall | 0.9500 |
| F1-Score | 0.9744 |
| Specificity | 1.0000 |
| TP | 19 |
| TN | 40 |
| FP | 0 |
| FN | 1 |
