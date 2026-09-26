# Experiment Audit: Development Path to Frozen Model

**Purpose**: Summarize the empirical development path that culminated in the final frozen model configuration, documenting the hypothesis, outcomes, and conclusions of each major research phase.

---

## 1. Baseline Evaluation (ResNet50 / EfficientNet-B3 / Swin-Tiny)
* **What changed**: Evaluated three fundamentally different architectures using standard ImageNet preprocessing.
* **Why it was tested**: To establish a baseline of generalization capability across CNNs and Vision Transformers.
* **Internal AUC**: ~0.94 - 0.98
* **Divesh AUC**: ~0.76 - 0.77
* **Conclusion**: Swin-Tiny performed the best among baselines, but all models suffered a massive generalization gap (~0.20) when evaluated on an external dataset. 
* **Retained**: Swin-Tiny backbone.

## 2. Aspect-Ratio-Preserving Preprocessing
* **What changed**: Replaced standard anisotropic resizing (`Resize(224, 224)`) with an aspect-ratio-preserving padding and center-crop strategy.
* **Why it was tested**: Suspected that squashing/stretching ultrasound images distorted key structural features of nodules (e.g., shape metrics like "taller-than-wide"), causing the model to learn artifactual decision boundaries.
* **Internal AUC**: ~0.94
* **Divesh AUC**: ~0.79 - 0.80
* **Conclusion**: Preserving the physical aspect ratio of the nodule significantly improved external generalization (+0.03 AUC) without sacrificing internal performance.
* **Retained**: Yes. Made mandatory for all subsequent experiments.

## 3. Domain-Adversarial Neural Networks (DANN) (Experiments 14–15)
* **What changed**: Introduced a Gradient Reversal Layer (GRL) and a domain classifier to force the backbone to learn domain-invariant features between TN5000 and AUITD.
* **Why it was tested**: To actively penalize the network for learning machine-specific artifacts.
* **Internal AUC**: ~0.94
* **Divesh AUC**: ~0.77
* **Conclusion**: DANN failed to improve external generalization and actually caused severe performance instability. The gradient reversal conflicted heavily with the staged unfreezing schedule, causing catastrophic forgetting or failing to converge optimally.
* **Retained**: No. Abandoned domain-adversarial approaches.

## 4. Multi-Scale TTA (Experiment 16)
* **What changed**: Evaluated the trained baseline model at three different spatial scales (`0.85x`, `1.00x`, `1.15x`) during inference and averaged the logits.
* **Why it was tested**: Hypothesized that Swin-Tiny predictions were brittle to the apparent spatial scale of the nodule in the image (since different machines use different zoom levels).
* **Internal AUC**: 0.941
* **Divesh AUC**: 0.814
* **Conclusion**: TTA proved that Swin-Tiny's confidence fluctuates based on scale (inter-scale correlations ~0.89-0.93). Averaging these predictions smoothed out spatial uncertainty and provided a massive boost to external generalization (+0.015 AUC).
* **Retained**: Yes. Made mandatory for inference. (Multi-scale training was also tested in this phase but regressed performance to 0.798, so it was discarded).

## 5. Multi-Level Feature Fusion (Experiment 17)
* **What changed**: Extracted hierarchical feature maps from intermediate Swin stages (`layers.1`, `layers.2`), pooled them, and concatenated them with the final abstract representation (`layers.3`) into a single fused classification head.
* **Why it was tested**: Deep abstractions in the final layer are often highly overfitted to the training domain's specific contrast/physics. Grounding predictions in early- and mid-level structural features (texture, edges) was hypothesized to improve robustness.
* **Internal AUC**: 0.945
* **Divesh AUC**: 0.816
* **Conclusion**: The structural grounding provided by fusion yielded the highest single-model external performance to date (+0.017 AUC) while only adding 0.27% more parameters.
* **Retained**: Yes. Replaced the standard classification head.

## 6. Multi-Seed Confirmation of Fusion + TTA (Experiments 18 & 19)
* **What changed**: Combined the architectural fix (Feature Fusion) with the inference fix (TTA) and evaluated it strictly across 5 random seeds to ensure the gains were reproducible and not due to lucky initialization.
* **Why it was tested**: To establish statistical confidence in the final configuration before freezing it.
* **Internal AUC**: 0.9438 ± 0.0060
* **Divesh AUC (TTA)**: 0.8140 ± 0.0077 (Max: 0.8282)
* **Conclusion**: TTA strictly improved predictions for all 5 random seeds. The combination of Fusion + TTA proved partially additive, consistently outperforming the original baseline's 0.799 AUC even in the worst-case random initialization (0.8069). 
* **Retained**: Yes. 

## 7. MODEL FROZEN
The development phase concluded. The pipeline was frozen for final independent external evaluation on a completely unseen dataset.
