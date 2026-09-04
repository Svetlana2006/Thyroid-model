# Experiment 18: Multi-Level Fusion + Multi-Scale TTA

## Objective
To determine if applying Multi-Scale Test-Time Augmentation (TTA) to the Multi-Level Feature Fusion model provides complementary benefits, establishing our strongest possible inference pipeline.

## 1. Final Metrics
* **Internal AUC:** 0.9481
* **External AUC:** **0.8205**
* **Generalization gap:** 0.1276

## 2. Scale-Specific Metrics
* **AUC @ 0.85x:** 0.8129
* **AUC @ 1.00x:** 0.8161
* **AUC @ 1.15x:** 0.8135
* **AUC @ averaged (TTA):** **0.8205**

## 3. Prediction Correlations (Scale Sensitivity)
* **0.85x vs 1.00x:** 0.9351 *(Baseline was 0.9323)*
* **0.85x vs 1.15x:** 0.9050 *(Baseline was 0.8953)*
* **1.00x vs 1.15x:** 0.9416 *(Baseline was 0.9297)*

## 4. Comparison

| Configuration | External AUC | Δ vs AR baseline |
| :--- | :--- | :--- |
| AR baseline (Exp 16 M0) | 0.7990 | — |
| AR + TTA (Exp 16 M2) | 0.8138 | +0.0148 |
| AR + Fusion (Exp 17 M1) | 0.8161 | +0.0171 |
| **AR + Fusion + TTA** (Exp 18) | **0.8205** | **+0.0215** |

## 5. Interpretation

1. **Does TTA improve the multi-level fusion model?**
   Yes. Averaging the logits across the three spatial scales boosted the fusion model's external AUC from 0.8161 to 0.8205.
2. **Is the combined model better than both individual improvements?**
   Yes. It definitively outperforms the TTA-only approach (0.8138) and the Fusion-only approach (0.8161), setting a new peak performance record for this project.
3. **Are the gains additive, partially additive, or redundant?**
   They are **partially additive**. TTA added ~0.015 and Fusion added ~0.017. If perfectly additive, we would expect a ~0.032 gain (0.831). Instead we saw a ~0.022 gain. This indicates that while both methods correct some of the same structural uncertainties, they also provide distinct, non-overlapping improvements.
4. **Does feature fusion reduce scale sensitivity?**
   **Yes.** Across every scale combination, the pairwise prediction correlation increased compared to the standard model (e.g., the extreme `0.85x vs 1.15x` correlation rose from 0.895 to 0.905). By grounding its predictions in low- and mid-level structural features (Stages 1 and 2), the fusion model's confidence fluctuates less when the nodule's apparent physical scale changes in the frame.
5. **Is the combined model currently our strongest candidate?**
   Without a doubt. Combining architectural structural awareness (Feature Fusion) with inference-time spatial awareness (TTA) yielded the most robust ultrasound classifier we have developed.
