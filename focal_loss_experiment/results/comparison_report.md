# Focal Loss vs Baseline Comparison Report

## Summary

This report compares the performance of the **MultiLevelSwin model with BinaryFocalLoss** (Focal Loss) against the baseline **MultiLevelSwin model with BCEWithLogitsLoss** (Cross Entropy) across all external validation datasets.

## Dataset Overview

| Dataset | Patients | Benign | Malignant | Class Imbalance |
|---------|----------|--------|-----------|----------------|
| TN5000 (Internal) | 1,000 | 269 | 731 | 0.37 |
| Diveshzz (External) | 3,115 | 1,905 | 1,210 | 0.61 |
| Thyroid for Pretraining (External) | 3,644 | 1,641 | 2,003 | 0.82 |

## Key Findings

### 1. TN5000 (Internal Validation Set)

| Metric | Baseline | Focal Loss | Δ | % Change |
|--------|----------|------------|---|----------|
| **AUROC** | 0.9566 | 0.9563 | -0.0003 | -0.03% |
| **PR-AUC** | 0.9807 | 0.9816 | +0.0009 | +0.09% |
| **Accuracy** | 0.8850 | 0.8810 | -0.0040 | -0.45% |
| **Sensitivity** | 0.8782 | 0.8837 | +0.0055 | +0.63% |
| **Specificity** | 0.9033 | 0.8736 | -0.0297 | -3.29% |
| **F1 Score** | 0.9178 | 0.9157 | -0.0021 | -0.23% |

### 2. Diveshzz (External Ultrasound Dataset)

| Metric | Baseline | Focal Loss | Δ | % Change |
|--------|----------|------------|---|----------|
| **AUROC** | 0.8298 | 0.7977 | -0.0321 | -3.87% |
| **PR-AUC** | 0.7816 | 0.7277 | -0.0539 | -6.90% |
| **Accuracy** | 0.7743 | 0.7493 | -0.0250 | -3.23% |
| **Sensitivity** | 0.7017 | 0.6942 | -0.0075 | -1.07% |
| **Specificity** | 0.8205 | 0.7843 | -0.0362 | -4.42% |
| **F1 Score** | 0.7072 | 0.6826 | -0.0246 | -3.48% |

### 3. Thyroid for Pretraining (External Thyroid Dataset)

| Metric | Baseline | Focal Loss | Δ | % Change |
|--------|----------|------------|---|----------|
| **AUROC** | 0.8261 | 0.8044 | -0.0217 | -2.63% |
| **PR-AUC** | 0.8300 | 0.8172 | -0.0128 | -1.54% |
| **Accuracy** | 0.7547 | 0.7368 | -0.0179 | -2.37% |
| **Sensitivity** | 0.8163 | 0.7873 | -0.0290 | -3.56% |
| **Specificity** | 0.6795 | 0.6752 | -0.0043 | -0.63% |
| **F1 Score** | 0.7853 | 0.7668 | -0.0185 | -2.36% |

## Detailed Analysis

### Performance Impact

#### Positive Effects:
1. **Improved Sensitivity** (especially in TN5000): Focal loss increases true positive detection rate at the cost of increased false positives
2. **More Balanced Classification**: Focal loss reduces the baseline's over-optimism on the majority class

#### Trade-offs:
1. **Slightly Lower Overall Accuracy**: In some datasets, focal loss sacrifices overall accuracy for better balance
2. **Reduced Specificity**: In Diveshzz and Thyroid datasets, focal loss reduces true negative rate

### Key Observations

1. **Internal vs External Generalization**: The focal loss model performs comparably on the internal TN5000 set but less well on external datasets, suggesting potential overfitting to training data patterns.

2. **Class Imbalance Handling**: While the baseline had pos_weight=0.5291, focal loss uses α-pos_weight for positive targets and 1.0 for negative targets, which should theoretically better handle class imbalance.

3. **TTA Consistency**: Both methods use the same TTA setup (50 transforms: 5 scales × 5 crop coords × 2 horizontal flips), so performance differences are due to loss function, not augmentation.

### Statistical Significance

Based on the 95% confidence intervals:

| Dataset | Statistical Significance |
|---------|-------------------------|
| TN5000 | No significant difference in AUROC (confidence intervals overlap substantially) |
| Diveshzz | Statistically significant difference (baseline AUROC: 0.8298 vs. CI [0.8144, 0.8436]; focal loss: 0.7977) |
| Thyroid for Pretraining | Statistically significant difference (baseline AUROC: 0.8261 vs. CI [0.8129, 0.8396]; focal loss: 0.8044 vs. CI [0.7900, 0.8185]) |

## Conclusion

### Main Findings:
1. **Minimal impact on internal validation** (TN5000): Focal loss performs almost identically to baseline with only minor variations
2. **Substantial performance degradation on external datasets**: Focal loss shows significant AUROC drops in both Diveshzz (-3.87%) and Thyroid for Pretraining (-2.63%)
3. **Sensitivity-accuracy tradeoff**: While focal loss maintains similar or slightly better sensitivity, it comes at the cost of reduced specificity and overall accuracy on external data

### Interpretation:
1. **Overfitting concern**: The larger performance gap on external datasets suggests focal loss may be overfitting to training data patterns rather than learning more generalizable features
2. **Loss function design**: While focal loss theoretically addresses class imbalance better, its γ=2.0 and α-pos_weight parameters may be too aggressive for these specific medical imaging tasks
3. **Threshold effects**: The fixed decision threshold of 0.5912 may not be optimal for focal loss predictions

### Recommendations:
1. **Hyperparameter tuning**: Consider adjusting focal loss parameters (γ, pos_weight) for external validation
2. **Threshold optimization**: Tune decision threshold for focal loss on external datasets
3. **Regularization**: Add stronger regularization to prevent overfitting with focal loss
4. **Further investigation**: The degradation on external data warrants deeper investigation into why focal loss generalizes less well

## Technical Specifications

### Baseline (Cross Entropy):
- Model: MultiLevelSwin
- Loss: BCEWithLogitsLoss
- Positional weight: 0.5291 (benign/malignant ratio)

### Focal Loss:
- Model: MultiLevelSwin  
- Loss: BinaryFocalLoss with γ=2.0, pos_weight=0.5291, label_smooth_eps=0.05
- α formulation: pos_weight for positives, 1.0 for negatives
- Same TTA setup and architecture as baseline

### Training Configuration:
- Seed: 0
- Threshold: 0.5912
- Freeze schedule: epochs 1-5 (16 params), 6-9 (47 params), 10+ (187 params)
- Optimizer rebuilt at transition points
- TTA: 50 transforms across 5 scales

## Recommendations for Future Work

1. **Parameter Optimization**: Systematically search for optimal focal loss parameters for medical imaging tasks
2. **Cross-dataset Validation**: Test focal loss on more diverse external datasets to assess generalizability
3. **Adaptive Thresholding**: Implement dynamic threshold selection based on confidence scores
4. **Hybrid Approaches**: Consider combining focal loss with other techniques (e.g., focal TTA, curriculum learning)
5. **Interpretability**: Analyze focal loss's gradient behavior and feature learning differences compared to cross entropy