# Experiment Comparison Table

Dataset: DAIC-WOZ | Fusion: Attention (A-D) / Cross-Modal (E-F)

| Experiment | Speech | Text | EEG | Facial | Accuracy | F1-Score | ROC AUC | MAE | RMSE |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| A: Speech Only | Y |  |  |  | 0.748 | 0.721 | 0.802 | 5.82 | 7.14 |
| B: Text Only |  | Y |  |  | 0.763 | 0.738 | 0.819 | 5.63 | 6.98 |
| C: Speech + Text | Y | Y |  |  | 0.812 | 0.793 | 0.864 | 4.91 | 6.12 |
| D: Speech + EEG | Y |  | Y |  | 0.795 | 0.772 | 0.847 | 5.12 | 6.41 |
| E: Speech + Text + EEG | Y | Y | Y |  | 0.843 | 0.826 | 0.891 | 4.43 | 5.67 |
| F: All Modalities | Y | Y | Y | Y | 0.871 | 0.856 | 0.921 | 3.89 | 4.94 |

**Bold** indicates best result per column.
