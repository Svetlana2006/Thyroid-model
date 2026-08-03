# Thyroid Nodule Classification — Exact Model Plan (TN5000)

## 0. Task definition
- Input: single B-mode ultrasound image, 224×224×3 (already resized per dataset docs).
- Output: binary probability P(malignant), threshold at 0.5 default (re-tuned via Youden's J on val set).
- Loss target: biopsy-confirmed label (ground truth), not TIRADS score.

---

## 1. Preprocessing (fixed, identical across all models)
- Resize: already 224×224 — verify; if not, `Resize(256) → CenterCrop(224)`.
- Normalize: ImageNet stats, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (matches dataset's own preprocessing convention).
- Color: convert grayscale US images to 3-channel by replication (not colorization) — preserves intensity fidelity.
- No CLAHE/contrast-stretching in v1 (avoid confounding architecture comparison); add as ablation later if needed.

---

## 2. Data splits — exact protocol
- If patient ID recoverable from metadata: **GroupShuffleSplit** by patient, stratified by label.
  - 70% train+val pool / 15% held-out internal test / 15% reserved as a second "shifted" test (different acquisition batch if identifiable).
- If no patient ID: stratified image-level split, fixed `random_state=42`, documented explicitly as a limitation.
- **Outer loop**: 5-fold StratifiedKFold on the 70% train+val pool (fold assignment fixed once, reused for all 3 architectures — controls for fold-luck).
- **Inner loop** (hyperparameter search only): within each outer training fold, an 80/20 sub-split for Optuna trials.
- Test set touched exactly once, at the very end, per architecture and for the final ensemble.

---

## 3. Architectures — exact configs

### Model A — ResNet-50
- Backbone: `torchvision.models.resnet50(weights=IMAGENET1K_V2)`.
- Head: replace `fc` with `Linear(2048→256) → ReLU → Dropout(p=0.3) → Linear(256→1)`.
- Freeze schedule: freeze all backbone layers for epochs 1–3 (train head only), unfreeze layer4 at epoch 4, unfreeze layer3 at epoch 8, full unfreeze at epoch 12.

### Model B — EfficientNet-B3
- Backbone: `timm.create_model('efficientnet_b3', pretrained=True)`.
- Head: replace classifier with `Linear(1536→256) → ReLU → Dropout(p=0.3) → Linear(256→1)`.
- Freeze schedule: same staged unfreeze pattern as Model A, scaled to EfficientNet block groups (unfreeze last 2 blocks at epoch 4, next 2 at epoch 8, rest at epoch 12).

### Model C — Swin-Tiny
- Backbone: `timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)`.
- Head: `Linear(768→256) → ReLU → Dropout(p=0.3) → Linear(256→1)`.
- Freeze schedule: freeze all but head + last stage for epochs 1–5 (transformers need longer warm-up), full unfreeze at epoch 10.
- Stochastic depth rate: 0.1 (timm default for tiny variant).

All three: single logit output (`BCEWithLogitsLoss`), sigmoid at inference.

---

## 4. Loss function — exact
- **Class-weighted BCE**: weight = inverse class frequency, `pos_weight = n_benign / n_malignant ≈ 0.40` (since malignant is majority here — reweight toward benign sensitivity, not malignant).
- Alternative to test in ablation: **Focal loss** (γ=2, α=0.25) — compare AUC/F1 against weighted BCE, keep whichever wins on inner-CV val set.
- Label smoothing: ε=0.05 applied to hard 0/1 labels before loss computation (reduces overconfidence, helps calibration).

---

## 5. Optimizer & schedule — exact
- Optimizer: `AdamW`, weight_decay=1e-4 (tunable, see §7).
- LR schedule: **Cosine annealing with warm restarts** — `T_0=10`, `T_mult=2`.
- Base LR: 3e-4 for head-only phase, drop to 1e-5 once backbone unfreezes (discriminative LR: backbone gets base_lr × 0.1, head gets base_lr).
- Batch size: 32 (ResNet/EfficientNet), 16 (Swin, due to memory).
- Mixed precision (`torch.cuda.amp`) throughout.
- Max epochs: 60, **early stopping** on val AUC, patience=10, min_delta=0.001.
- Gradient clipping: max_norm=1.0.

---

## 6. Augmentation — exact, label-preserving only
Applied via `torchvision.transforms` or `albumentations`, identical pipeline for all 3 models:
- `RandomRotation(degrees=15)`
- `RandomHorizontalFlip(p=0.5)` (no vertical flip — anatomically implausible for US probe orientation)
- `ColorJitter(brightness=0.15, contrast=0.15)`
- `RandomResizedCrop(224, scale=(0.9, 1.0))` — mild only, avoid cropping out nodule
- `GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))`, p=0.2 (simulates probe/gain variation)
- No mixup/cutmix in v1 (blends nodule with background — changes clinical meaning; explicitly avoided per Tiger Model paper's finding on morphology-altering augmentation).
- Test-time augmentation (inference only): 5-crop + horizontal flip, average sigmoid outputs.

---

## 7. Hyperparameter search — exact
- Tool: **Optuna**, TPE sampler, 40 trials per architecture, pruning via `MedianPruner` (prune at epoch 15 if trial trending below median).
- Search space:
  - `lr_head`: log-uniform [1e-4, 1e-2]
  - `weight_decay`: log-uniform [1e-5, 1e-2]
  - `dropout`: uniform [0.1, 0.5]
  - `pos_weight_scale`: uniform [0.8, 1.5] (multiplier on the computed class-weight)
  - `batch_size`: categorical [16, 32, 64] (ResNet/EfficientNet only; Swin fixed at 16 for memory)
- Objective: mean val AUC across the inner 80/20 split, repeated ×3 seeds per trial, objective = mean − 0.5×std (penalize instability, not just peak performance).
- Output: best config per architecture, **plus** a full trial-history CSV and hyperparameter-vs-AUC sensitivity plots (partial dependence) — this is the deliverable that shows the model isn't fragile to small perturbations.

---

## 8. Interpretability — exact methods
- **Grad-CAM++** for ResNet-50 and EfficientNet-B3, hooked at the last conv block before global pooling.
- **Attention rollout** for Swin-Tiny (average attention weights across heads and layers, following Abnar & Zuidema 2020 method).
- Generate heatmaps for: (a) 20 random correctly-classified malignant, (b) 20 random correctly-classified benign, (c) all misclassified test cases.
- Quantitative check: compute % of heatmap "hot" pixels falling inside the nodule segmentation mask (need a nodule bounding box/mask — if not provided in TN5000, generate via a lightweight U-Net trained on any subset with masks, or via radiologist spot-check on 50 images) — this converts "heatmap looks plausible" into a **measured localization accuracy**, addressing the review's specific criticism that heatmaps are often clinically unhelpful without such grounding.
- Secondary interpretable model: `XGBoost` on handcrafted features (aspect ratio, mean/std intensity inside vs. outside a coarse nodule mask, edge irregularity via contour perimeter/area ratio) — report agreement (Cohen's κ) between XGBoost and the deep ensemble; flag disagreement cases for manual review.

---

## 9. Ensembling — exact
- Soft-vote: average sigmoid outputs of Model A, B, C (equal weight v1).
- Weighted variant: weights optimized on inner-CV val AUC via constrained grid search (weights sum to 1, step 0.1) — report both simple and optimized ensemble.
- Deep ensemble uncertainty: train each architecture with 3 different seeds (9 models total: 3 arch × 3 seeds) → predictive mean = ensemble output, predictive variance = uncertainty score.
- Uncertainty thresholding: flag cases with variance above the 90th percentile as "low confidence — recommend specialist review" rather than forcing a binary call.

---

## 10. Evaluation & reporting — exact
- Metrics: AUC, accuracy, sensitivity, specificity, PPV, NPV, F1, Brier score — computed per fold, then aggregated.
- CI: **bootstrap, 1000 resamples**, report 2.5th/97.5th percentile.
- Calibration: reliability diagram + Brier score, before and after **temperature scaling** (fit temperature on held-out val fold only).
- Statistical comparison across architectures: **DeLong's test** for AUC differences (paired, same test set).
- Final report table: one row per model (A, B, C, simple ensemble, weighted ensemble) × columns (AUC±CI, Sens, Spec, PPV, NPV, Brier, ECE).

---

## 11. Reproducibility artifacts to produce
- `config.yaml` per architecture with final chosen hyperparameters.
- `splits.json` recording exact train/val/test indices (or patient ID lists) and the random seed used.
- Optuna study database (`.db` file) — full trial history, not just winners.
- Model checkpoints (best val AUC epoch, per fold, per architecture).
