# Experiment 00: Methodological Audit

## Verified Claims

### DDTI-unique dataset
- **Confirmed**: Only 1 image (benign/2_1.jpg, 33KB) survived deduplication.
- The `setup_ddti_unique.py` uses pHash with Hamming distance ≤5, which was too aggressive.
- The "three-dataset training" is effectively TN5000 (3500) + AUITD (2118) + 1 DDTI image.

### Preprocessing (Albumentations Resize)
- **Confirmed**: `A.Resize(256,256)` performs anisotropic scaling — stretches all images to square.
- TN5000 has 9 distinct image sizes; 79.5% are 718×500 (AR_HW=0.696).
- Anisotropic distortion: 30.4% for the majority class, up to 55% for 818×368 images.
- Bounding boxes are NOT passed through Albumentations (no BboxParams) — bbox AR in metadata reflects original pixel coordinates, but the image content the model sees has distorted geometry.

### Class weights
- TN5000-only pos_weight = 0.4182 (1032 benign / 2468 malignant)
- Combined pos_weight = 0.5294 (1945 benign / 3674 malignant)
- Difference = 0.1112 — AUITD has more balanced classes, shifting the combined weight.

### EfficientNet-B3
- timm native resolution: 288×288 (not 300 as previously stated)
- 224×224 is functional (adaptive pooling), producing same 1536-D output
- Using 224 instead of 288 = 39.5% fewer input pixels, but not invalid

### Checkpoints
- **Corrected**: All 9 checkpoints ARE present locally (previous analysis was wrong)
- ResNet50: 3 × 104.7 MB, EfficientNet-B3: 3 × 44.9 MB, Swin-Tiny: 3 × 110.9 MB

### Image dimensions
| Size | Count | % | AR_HW |
|------|-------|---|-------|
| 718×500 | 3973 | 79.5% | 0.696 |
| 818×628 | 650 | 13.0% | 0.768 |
| 498×429 | 205 | 4.1% | 0.861 |
| 439×368 | 124 | 2.5% | 0.838 |
| Other (5 sizes) | 48 | 0.9% | 0.638–0.914 |
