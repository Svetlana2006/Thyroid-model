# Research Findings

## Status: Steps 1-12 EXPERIMENTS COMPLETED

---

# What We Know So Far

## Confirmed
*(Directly demonstrated via empirical measurement or code execution)*

- **Exact anisotropic preprocessing**: `A.Resize(256, 256)` performs anisotropic scaling. All images are stretched to a square, causing massive geometric distortion. For the TN5000 test set, median relative distortion is **43.6%**. Our new Aspect-Ratio-Preserving (B) and Letterbox (E) pipelines reduce this distortion to ~0.0%.
- **Existence/nonexistence of cross-dataset duplicates**: Zero exact (MD5) or near-duplicate (dHash ≤10) images exist between the local TN5000 and AUITD datasets.
- **Actual DDTI sample size**: DDTI-unique contains exactly 1 image due to aggressive pHash deduplication.
- **Internal/External performance**: Current models achieve high internal AUC (0.90-0.95) on TN5000 but fail on the external Divesh set (AUC 0.58-0.66).
- **Measured dataset separability**: A Random Forest using only simple image stats achieves 1.000 accuracy in distinguishing TN5000 vs AUITD. Deep features (ResNet50, EffNet, Swin) achieve >0.98 accuracy.
- **Measured occlusion effects**: Masking the nodule drops AUC by ~0.20 to 0.27 across architectures, demonstrating that the models heavily utilize the nodule region.

## Strong Evidence
*(Multiple experiments support the explanation)*

- **Substantial TN5000/AUITD domain difference**: The domain shift gap is fundamental to the ultrasound texture. Progressive standardization (cropping 15% outer borders to remove UI/logos, preserving aspect ratio, and matching normalization) still allows a ResNet50 classifier to separate the datasets with **0.9965 accuracy**. 
- **Learned representations contain dataset-specific information**: Frozen deep features map the datasets to entirely separable clusters, making domain shift a leading explanation for the observed external-validation degradation.

## Suggestive
*(Result is consistent with the explanation but confounded)*

- **Possible contextual shortcut**: Masking everything outside a padded bounding box drops AUC by ~0.07 to 0.16. This suggests some dependence on contextual tissue or scanner background, especially for EfficientNet-B3.
- **Possible scale dependence**: Our decomposed crop experiment shows that resizing a nodule crop to a standardized scale (full frame) craters AUC by **0.29 to 0.40**, whereas cropping but preserving the original apparent scale (letterboxing on a mean background) only drops AUC by ~0.08 to 0.25. This strongly suggests models rely on the relative scale of the nodule within the frame.
- **Possible acquisition-related shortcut**: We analyzed spatial-frequency characteristics (low, mid, high-frequency energy fractions). The distributions are nearly identical across TN5000, AUITD, and Divesh (e.g., LF is ~0.99 for all). Therefore, while high-frequency energy was a strong predictor in the image-stats classifier, simple global frequency bands do not fully explain the domain gap.

## Unknown
*(No experiment currently distinguishes the hypotheses)*

- **Whether preserving aspect ratio improves external generalization**: The ablation script (`run_preprocessing_ablation.py`) is complete, including diagnostic tracking, but full 25-epoch training on all pipelines is pending execution.
- **Whether explicit geometry supervision improves robustness**: Multitask learning/feature probing experiments are pending.
- **Whether architecture-specific feature representation is responsible**: Pending deeper probing.
- **Whether radiomics add complementary information**: Pending.
