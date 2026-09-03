# Experiment 14 — Results

> **Status: COMPLETE — GPU training finished**

---

## Results

### ResNet50

| Metric | Current (A) | AR-preserving (B) | Change |
|---|---|---|---|
| Internal AUC | 0.9166 | 0.9273 | +0.0107 |
| Internal AUC 95% CI | [0.8968–0.9341] | [0.9095–0.9432] | |
| External AUC | 0.7534 | 0.7806 | +0.0272 |
| External AUC 95% CI | [0.7377–0.7708] | [0.7649–0.7980] | |
| Ext. Sensitivity | 0.6488 | 0.6959 | +0.0471 |
| Ext. Specificity | 0.7375 | 0.7318 | -0.0057 |
| Ext. F1 | 0.6293 | 0.6570 | +0.0277 |
| Delta AUC (gap) | 0.1632 | 0.1467 | -0.0165 |
| DeLong p (exploratory) | | | p = 0.0294 |

### EfficientNet-B3

| Metric | Current (A) | AR-preserving (B) | Change |
|---|---|---|---|
| Internal AUC | 0.8930 | 0.9108 | +0.0178 |
| Internal AUC 95% CI | [0.8714–0.9155] | [0.8898–0.9312] | |
| External AUC | 0.7633 | 0.7753 | +0.0120 |
| External AUC 95% CI | [0.7458–0.7813] | [0.7585–0.7928] | |
| Ext. Sensitivity | 0.6421 | 0.6992 | +0.0571 |
| Ext. Specificity | 0.7612 | 0.7176 | -0.0436 |
| Ext. F1 | 0.6364 | 0.6523 | +0.0159 |
| Delta AUC (gap) | 0.1297 | 0.1356 | +0.0059 |
| DeLong p (exploratory) | | | p = 0.3325 |

### Swin-Tiny

| Metric | Current (A) | AR-preserving (B) | Change |
|---|---|---|---|
| Internal AUC | 0.9412 | 0.9402 | -0.0010 |
| Internal AUC 95% CI | [0.9230–0.9580] | [0.9230–0.9575] | |
| External AUC | 0.7983 | 0.8072 | +0.0089 |
| External AUC 95% CI | [0.7808–0.8158] | [0.7902–0.8235] | |
| Ext. Sensitivity | 0.5661 | 0.7066 | +0.1405 |
| Ext. Specificity | 0.8877 | 0.7795 | -0.1082 |
| Ext. F1 | 0.6496 | 0.6881 | +0.0385 |
| Delta AUC (gap) | 0.1429 | 0.1329 | -0.0100 |
| DeLong p (exploratory) | | | p = 0.4461 |

---

## Cross-Architecture Comparison Table

| Architecture | Cur Int | AR Int | dInt | Cur Ext | AR Ext | dExt | Cur Gap | AR Gap | Gap Red |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | 0.9166 | 0.9273 | +0.0107 | 0.7534 | 0.7806 | +0.0272 | 0.1632 | 0.1467 | +0.0165 |
| EfficientNet-B3 | 0.8930 | 0.9108 | +0.0178 | 0.7633 | 0.7753 | +0.0120 | 0.1297 | 0.1356 | -0.0059 |
| Swin-Tiny | 0.9412 | 0.9402 | -0.0010 | 0.7983 | 0.8072 | +0.0089 | 0.1429 | 0.1329 | +0.0100 |

---

## Interpretation

### Does AR preservation consistently improve external AUC?

**Yes.** Aspect-ratio-preserving preprocessing improved external AUC across all three tested architectures (ResNet50: +0.0272, EfficientNet-B3: +0.0120, Swin-Tiny: +0.0089).

### Does it consistently reduce the internal→external gap?

**Mostly.** The generalization gap was reduced for ResNet50 (by 0.0165) and Swin-Tiny (by 0.0100). However, for EfficientNet-B3, the internal AUC increased more than the external AUC, causing the absolute gap to slightly widen (by 0.0059), even though external performance improved.

### Is the effect similar in magnitude across architectures?

**No, the magnitude varies.** The improvement was largest for ResNet50 (where the DeLong test showed exploratory significance, p=0.029). The improvement for Swin-Tiny and EfficientNet-B3 was more modest (~0.01 AUC) and not statistically significant under DeLong testing on this single run. In all cases, the models traded external specificity for substantial gains in external sensitivity (e.g., Swin-Tiny sensitivity jumped from 0.5661 to 0.7066).

---

## Limitations

- **Single seed (seed 0)**: Results represent one training run per condition. Variance due to random initialization and data ordering is not quantified. Multi-seed confirmation is required before treating any improvement as robust.
- **Single external dataset (Divesh)**: External generalization is evaluated on one dataset from one source. Performance on other external datasets may differ.
- **No claim of universal generalization**: The experiment tests the specific question of whether AR preservation helps on these architectures with this training data and this external set.
- **Preprocessing is only one intervention**: Even if AR preservation consistently improves external AUC, domain shift remains the dominant source of the generalization gap (estimated ~60–70% of the gap in Exp 02).
- **DeLong comparisons are exploratory**: The two models being compared were trained separately under different preprocessing; this is not a paired statistical test in the strict sense.

---

## Decision for Next Experiment

**C. AR preservation is too weak or inconsistent → proceed to domain-robust representation learning.**

*Rationale*: While AR preservation technically improved external AUC on all three architectures, the magnitude is small (0.0089 to 0.0272) and only significant on ResNet50. Even with AR preservation, the remaining generalization gap is massive (~0.13 to 0.15). Since geometric distortion only accounts for a fraction of the domain shift, and the improvement is relatively weak, multi-seed validation of preprocessing alone is not the most scientifically urgent next step. We should adopt AR preservation as the new standard preprocessing baseline, but immediately proceed to algorithmic interventions (e.g., domain adversarial training, geometry fusion) that directly target the much larger feature-representation domain gap.
