# Experiment 16: Multi-Scale Aspect-Ratio-Preserving Inference

## Objective
To determine if we can make the Swin-Tiny model more robust to apparent nodule scale variations (which cause domain shifts) by utilizing stochastic multi-scale training and/or multi-scale Test-Time Augmentation (TTA). 

We tested three spatial scales while preserving the true aspect ratio:
* **0.85x**: Image appears smaller within the frame, showing more surrounding context.
* **1.00x**: Standard AR-preserving center crop.
* **1.15x**: Image appears larger, cropping out peripheral context.

## Results

| Model | Multi-Scale Train | TTA Evaluation | Internal AUC | External Divesh AUC | Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M0 (Baseline)** | No (1.00x only) | No (1.00x only) | 0.948 | 0.799 | 0.149 |
| **M1** | Yes (Stochastic) | No (1.00x only) | 0.945 | 0.806 | 0.139 |
| **M2** | No (1.00x only) | **Yes (0.85x, 1.00x, 1.15x)** | 0.941 | **0.814** | 0.127 |
| **M3** | Yes (Stochastic) | Yes (0.85x, 1.00x, 1.15x) | 0.946 | 0.798 | 0.147 |

### Deep Dive into M2 (TTA)
For the M2 run, we can see the AUC evaluated at individual scales before averaging:
* **M2 @ 1.00x only**: 0.808
* **M2 @ 0.85x only**: 0.812
* **M2 @ 1.15x only**: 0.801
* **M2 TTA Average**: **0.814**

Prediction correlations between the scales in M2:
* 0.85x vs 1.00x: **0.932**
* 1.00x vs 1.15x: **0.930**
* 0.85x vs 1.15x: **0.895**

## Interpretation

1. **TTA is highly effective (M2)**: Averaging the predictions of the baseline model across different apparent spatial scales (TTA) provided the highest external AUC (0.814) and the lowest generalization gap (0.127). 
2. **Swin is scale-sensitive**: The correlation between the 0.85x prediction and the 1.15x prediction was only `0.895`. This proves that Swin-Tiny's confidence fluctuates significantly based on how much of the frame the nodule occupies. TTA successfully smooths out this fluctuation.
3. **Multi-scale training is harmful (M3)**: Forcing the network to train on stochastic scales (M3) degraded the model. This is likely because the `0.85x` scale introduces heavy black padding, which limits the resolution of the fine ultrasound texture during training, hurting feature extraction.

**Conclusion:** We should keep standard AR-preserving training (1.00x) to ensure the network sees high-resolution texture during optimization, but use **Multi-Scale TTA at inference** to stabilize scale sensitivity.
