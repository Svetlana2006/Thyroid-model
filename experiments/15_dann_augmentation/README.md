# Experiment 15: Ultrasound Appearance Augmentation + Domain-Adversarial Learning (DANN)

## Scientific Objective

Test whether the remaining external-validation generalization gap (after aspect-ratio preservation) is driven by **dataset-specific ultrasound appearance differences** (scanner characteristics, image processing, gain settings).

## Methodology

We evaluate four models using the **Swin-Tiny** architecture and **Aspect-Ratio-Preserving** preprocessing (Pipeline B from Exp 14).

*   **M0: AR Baseline**: Control model, directly corresponding to Exp 14 (Swin-Tiny, Pipeline B). Uses standard data augmentation.
*   **M1: AR + Appearance Augmentation**: Replaces standard ColorJitter/Blur with a medically plausible, ultrasound-specific appearance augmentation pipeline (multiplicative noise, gaussian noise, mild blur/sharpen, gamma/brightness/contrast) that does not alter lesion morphology.
*   **M2: AR + DANN**: Adds a Gradient Reversal Layer (GRL) and domain-classification head (TN5000 vs AUITD) to penalize domain-specific feature representations.
*   **M3: AR + Both**: Combines ultrasound appearance augmentation and DANN.

## DANN Implementation & Confounders

A critical confounder in domain-adversarial learning is that different datasets often have different disease prevalences. If not addressed, the domain classifier can simply learn to predict "benign vs malignant" to infer the domain, inadvertently penalizing disease-relevant features.

To prevent this:
1. We compute **sample-level domain weights**.
2. We weight the domain `CrossEntropyLoss` such that the four subgroups (`TN5000-Benign`, `TN5000-Malignant`, `AUITD-Benign`, `AUITD-Malignant`) contribute equally to the domain loss.
3. This completely decouples domain prediction from disease prevalence, forcing the domain head to focus on image texture/appearance.

## Post-Hoc Analysis

For all four models, we extract frozen backbone features and train a balanced Logistic Regression classifier to distinguish TN5000 from AUITD. This allows us to quantify the true domain separability of the learned representations.

## Output Structure

*   `run_experiment15.py`: The unified training and evaluation script.
*   `configs/`: Saved hyperparameter configurations.
*   `checkpoints/`: Model weights.
*   `metrics/`: Detailed per-run metrics.
*   `results.csv`: Combined metric table.
*   `RESULTS.md`: Interpretation and recommendations based on the findings.
