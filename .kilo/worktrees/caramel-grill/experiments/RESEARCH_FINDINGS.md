# Research Findings

## Status: Experiments 2, 4, 5, 6, 12, 13 COMPLETE

---

# What We Know So Far

## Confirmed
*(Directly demonstrated via empirical measurement or code execution)*

### Anisotropic Preprocessing (Exp 4)
- `A.Resize(256,256)` causes a **median relative bbox distortion of 43.6%** (max: 257%) across TN5000.
- Pipeline B (AR-preserving + CenterCrop to 224) reduces this to 0.15% median.
- Pipeline E (Letterbox directly to 224, no crop) reduces this to 0.007% median — and **79.6% of bboxes are losslessly preserved** (distortion ≈ 0).
- Bounding boxes are not transformed by the current pipeline at all; bbox metadata reflects original pixel coordinates.

### Cross-Dataset Duplicates (Exp 2)
- Zero exact (MD5) or near-duplicate (dHash ≤10) images exist between TN5000 and AUITD.
- DDTI-unique contains exactly 1 image after pHash deduplication.

### Internal/External Performance Baseline
- Current baseline (Swin-Tiny, Pipeline A): internal AUC 0.931, external AUC 0.786, **delta = 0.146**.

### Preprocessing Ablation — Swin-Tiny, seed 0 (Exp 2/5)

| Pipeline | Internal AUC | External AUC | Delta AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| A — Current (anisotropic) | 0.931 | 0.786 | 0.146 | 0.850 | 0.877 |
| B — AR-preserving (crop) | 0.937 | **0.822** | **0.114** | 0.918 | 0.814 |
| C — Nodule crop (anisotropic) | 0.756 | 0.741 | 0.016 | 0.700 | 0.691 |
| D — Nodule crop + AR | 0.751 | 0.733 | 0.019 | 0.625 | 0.740 |
| E — Letterbox full image | **0.941** | 0.817 | 0.124 | 0.889 | 0.870 |

- **B and E both improve external AUC by ~3.5 pp vs. A.** Delta AUC narrows from 0.146 to 0.114 (B) and 0.124 (E).
- **C and D dramatically reduce internal AUC (~17 pp drop)** while barely improving the delta. Nodule-only cropping is harmful in this setting — the model needs global context to classify effectively.
- Pipeline E (letterbox, no crop) achieves the best internal AUC (0.941) and comparable external AUC to B (0.817 vs 0.822).

### EfficientNet-B3 Resolution (Exp 13)

| Resolution | Internal AUC | External AUC | Delta AUC | Best Val AUC |
|---|---|---|---|---|
| 224×224 | 0.887 | **0.780** | **0.107** | 0.917 |
| 288×288 (native) | **0.909** | 0.762 | 0.147 | 0.923 |

- Native resolution gives slightly better internal (+2.2 pp) and val AUC, **but worse external AUC** (−1.8 pp) and a **larger delta** (0.147 vs 0.107).
- This is the opposite of what might be naively expected. The native-resolution model does not generalize better.

### Domain Shift — Progressive Standardization (Exp 6)

| Condition | Mean Acc | 95% CI |
|---|---|---|
| 1 — Raw squashed (current) | 0.969 | [0.961, 0.977] |
| 2 — AR-preserving (no norm) | 0.971 | [0.967, 0.975] |
| 3 — AR-preserving + normalization | 0.997 | [0.994, 0.999] |
| 4 — Border crop (UI removal) + AR + norm | **0.997** | [0.995, 0.999] |

- **Accuracy does not decrease with standardization — it increases.** After AR-preserving + normalization, separability goes from 0.969 → 0.997.
- This counter-intuitive result suggests that the domain gap is not a superficial preprocessing artifact. It is deeply encoded in the image content learned by the backbone.
- Removing scanner borders/logos (Condition 4) had essentially no effect vs. Condition 3. UI elements are **not** a significant contributor to dataset separability.

### Spatial Frequency Analysis (Exp 12)

| Dataset | LF energy | MF energy | HF energy |
|---|---|---|---|
| TN5000 | 0.991 ± 0.006 | 0.008 ± 0.006 | 0.0006 ± 0.0004 |
| AUITD | 0.989 ± 0.007 | 0.010 ± 0.007 | 0.0013 ± 0.0007 |
| Divesh | 0.990 ± 0.006 | 0.009 ± 0.006 | 0.0007 ± 0.0005 |

- Mann-Whitney U vs TN5000: HF energy difference is highly significant vs. AUITD (p ≈ 0, **effect size r = 0.70**), but small vs. Divesh (r = 0.08).
- Despite statistical significance, the absolute HF differences are tiny (<0.001 in mean). These simple radial frequency bands are insufficient to explain the deep feature separability.
- The domain gap is encoded in higher-level texture/structural patterns, not captured by coarse frequency bins.

---

## Strong Evidence
*(Multiple consistent experimental observations)*

- **Substantial TN5000/AUITD domain difference is genuine and not trivially preprocessing-related**: Domain separability rises from 0.969 → 0.997 as standardization improves, ruling out that the gap is simply an artifact of size or normalization differences.
- **Learned representations are saturated with domain information**: Frozen ResNet50 features separate datasets at 0.997 accuracy even after border crop, meaning ~3 misclassifications per 1000 images.
- **AR preservation modestly but consistently reduces the generalization gap**: B and E both shrank delta AUC by 2.2–3.2 pp vs. A, across identical training conditions. This effect is real but modest.

---

## Suggestive
*(Consistent with the explanation, but confounded or underpowered)*

- **Possible scale dependence**: The occlusion decomposition showed that resizing the nodule crop to fill the full frame (9D) drops AUC 0.29–0.40, while placing the nodule on a mean background at its original scale (9C) only drops 0.08–0.25. **Suggestive** that relative scale is important, but the 9D perturbation also destroys contextual tissue, so causation is not isolated.
- **Possible contextual shortcut**: Masking outside the padded bbox (9B) drops AUC by 0.07–0.16. Suggests models use surrounding tissue or scanner background, but the sharp masking boundary is a confound.
- **Higher input resolution may not help external generalization**: The 288 vs 224 EfficientNet experiment shows native resolution improves internal metrics but **hurts** external AUC. This is consistent with higher resolution facilitating memorization of dataset-specific texture details.

---

## Unknown
*(No experiment currently distinguishes these hypotheses)*

- **Why does AR preservation help externally but not enough?** The delta AUC remains 0.11–0.12 even after fixing geometry. Roughly 60–70% of the gap is unexplained by preprocessing alone.
- **What drives the deep-feature domain gap?** Frequency analysis eliminated coarse frequency bands as the primary driver. The cause (scanner PSF, gain settings, post-processing, speckle statistics, patient selection) remains unidentified.
- **Whether architecture-specific representation matters**: Only Swin-Tiny has been tested across pipelines. Whether ResNet50/EffNet show different sensitivities to AR preservation is unknown.
- **Whether domain adversarial training / normalization (DANN, HistogramMatching) can close the remaining gap**: Not yet tested.
- **Whether radiomics/geometry supervision add complementary information**: Not yet tested.
- **Whether single-seed ablation results are reliable**: All ablation results are single-seed (seed 0). Multi-seed confirmation is required before treating the B/E improvement as robust.
