# Experiment 14 — Results

> **Status: PENDING — fill in after GPU training completes**

---

## Results

### ResNet50

| Metric | Current (A) | AR-preserving (B) | Change |
|---|---|---|---|
| Internal AUC | | | |
| Internal AUC 95% CI | | | |
| External AUC | | | |
| External AUC 95% CI | | | |
| Sensitivity | | | |
| Specificity | | | |
| F1 | | | |
| Delta AUC (gap) | | | |
| DeLong p (exploratory) | | | |

### EfficientNet-B3

| Metric | Current (A) | AR-preserving (B) | Change |
|---|---|---|---|
| Internal AUC | | | |
| Internal AUC 95% CI | | | |
| External AUC | | | |
| External AUC 95% CI | | | |
| Sensitivity | | | |
| Specificity | | | |
| F1 | | | |
| Delta AUC (gap) | | | |
| DeLong p (exploratory) | | | |

### Swin-Tiny

Reference from Exp 02. Reproduced or updated values to be filled here.

| Metric | Current (A) | AR-preserving (B) | Change |
|---|---|---|---|
| Internal AUC | 0.9313 | 0.9366 | +0.0053 |
| External AUC | 0.7858 | 0.8222 | +0.0364 |
| Delta AUC (gap) | 0.1456 | 0.1144 | −0.0312 |

---

## Cross-Architecture Comparison Table

| Architecture | Cur Int | AR Int | dInt | Cur Ext | AR Ext | dExt | Cur Gap | AR Gap | Gap Red |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | | | | | | | | | |
| EfficientNet-B3 | | | | | | | | | |
| Swin-Tiny | 0.9313 | 0.9366 | +0.0053 | 0.7858 | 0.8222 | +0.0364 | 0.1456 | 0.1144 | +0.0312 |

---

## Interpretation

### Does AR preservation consistently improve external AUC?

*To be filled after results.*

### Does it consistently reduce the internal→external gap?

*To be filled after results.*

### Is the effect similar in magnitude across architectures?

*To be filled after results.*

---

## Limitations

- **Single seed (seed 0)**: Results represent one training run per condition. Variance due to random initialization and data ordering is not quantified. Multi-seed confirmation is required before treating any improvement as robust.
- **Single external dataset (Divesh)**: External generalization is evaluated on one dataset from one source. Performance on other external datasets may differ.
- **No claim of universal generalization**: The experiment tests the specific question of whether AR preservation helps on these architectures with this training data and this external set.
- **Preprocessing is only one intervention**: Even if AR preservation consistently improves external AUC, domain shift remains the dominant source of the generalization gap (estimated ~60–70% of the gap in Exp 02).
- **DeLong comparisons are exploratory**: The two models being compared were trained separately under different preprocessing; this is not a paired statistical test in the strict sense.

---

## Decision for Next Experiment

*(Fill in after all 6 runs complete)*

Based only on the measured results, select one:

**A.** AR preservation is sufficiently strong and consistent → proceed with multi-seed confirmation.

**B.** AR preservation effect is architecture-specific → investigate which architectures benefit and why.

**C.** AR preservation is too weak or inconsistent → proceed to domain-robust representation learning (DANN, histogram matching, etc.).

Do not choose based on which recommendation leads to a higher target AUC. Choose based only on what the data shows.
