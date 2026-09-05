# Final Independent Evaluation Results

## Dataset Summary
* **Dataset**: Divesh External Dataset
* **Total Samples**: 3,115
* **Class Distribution**: 
  * Benign: 1,905
  * Malignant: 1,210
* **Evaluation Framework**: 5-Seed Frozen Ensemble (Swin-Tiny + Feature Fusion + Multi-Scale TTA)

---

## 1. Final External Validation Metrics
*Metrics computed using the optimal validation threshold (`0.5912`) without tuning on the external dataset.*

| Metric | Value |
| :--- | :--- |
| **AUROC (Final TTA)** | **0.8254** |
| 95% Confidence Interval | [0.8106 - 0.8416] |
| **PR-AUC** | 0.7795 |
| **Sensitivity (Recall)** | 0.7099 |
| **Specificity** | 0.8073 |
| **Accuracy** | 0.7695 |
| **PPV (Precision)** | 0.7007 |
| **NPV** | 0.8142 |
| **F1 Score** | 0.7053 |

---

## 2. Multi-Scale TTA Comparison
*Comparison of inference at standard single scales vs. the ensemble averaged logit TTA.*

| Inference Scale | AUROC |
| :--- | :--- |
| 0.85x Scale | 0.8191 |
| 1.00x Scale | 0.8210 |
| 1.15x Scale | 0.8209 |
| **Combined TTA Average** | **0.8254** |

## 3. Development vs Final Validation Comparison
*Tracking model robustness throughout the major phases of the project.*

| Development Phase | External AUC |
| :--- | :--- |
| Baseline (Exp 16 M0) | 0.7990 |
| Single-Seed Fusion (Exp 17) | 0.8161 |
| Single-Seed Fusion + TTA (Exp 18) | 0.8205 |
| **5-Seed Frozen Ensemble + TTA (Final)** | **0.8254** |
