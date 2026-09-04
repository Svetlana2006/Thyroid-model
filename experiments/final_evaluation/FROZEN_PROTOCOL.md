# FROZEN PROTOCOL: Thyroid Ultrasound Classification Model

**Status**: FROZEN
**Purpose**: Final Independent External Validation

## 1. Development Path
The frozen configuration defined in this document was informed by a strict ablation sequence documented in Experiments 1–19, culminating in the Multi-Seed Confirmation (Experiment 19). No changes to this protocol will be made following its finalization. The final independent external test set (DDTI) was completely unseen and untouched during the development phase.

## 2. Final Architecture
* **Backbone**: Swin-Tiny (`swin_tiny_patch4_window7_224` from `timm`)
* **Pre-trained Weights**: ImageNet pre-training for backbone
* **Fusion Module**:
  * Extracts intermediate hierarchical features from three specific stages:
    * `layers.1` (192 channels)
    * `layers.2` (384 channels)
    * `layers.3` (768 channels)
  * Applies Global Average Pooling to each stage.
  * Normalizes each stage independently via `LayerNorm`.
  * Projects each stage to `PROJ_DIM = 128` using a bias-free `nn.Linear` layer.
  * Concatenates the three projected stages into a single 384-dimensional feature vector.
* **Fusion Head**:
  * Linear(384 → 256)
  * GELU activation
  * Dropout (p=0.3)
  * Linear(256 → 1)
* **Total Parameter Count**: 27,792,891

## 3. Data Preprocessing
Images are resized strictly while preserving their native aspect ratios. No arbitrary nodule cropping is performed; the entire ultrasound context is maintained.
* `TARGET_RES = 256`
* `CROP_SIZE = 224`
* Resize: `LongestMaxSize(max_size=256)`
* Padding: `PadIfNeeded(min_height=256, min_width=256, border_mode=0)` (black padding)
* Cropping:
  * Training: `RandomCrop(224, 224)`
  * Evaluation: `CenterCrop(224, 224)`
* Normalization: ImageNet Mean and Std `Normalize()`
* Tensor Conversion: `ToTensorV2()`

## 4. Test-Time Augmentation (TTA) Procedure
Inference will be performed using a 3-scale spatial ensemble.
* **Scales**: `0.85x`, `1.00x`, `1.15x`
* **Implementation**: The `max_size` for `LongestMaxSize` is multiplied by the scale factor, yielding sizes of `218`, `256`, and `294` respectively. The image is padded (if necessary) to at least `224`, and a center crop of `224x224` is extracted.
* **Ensembling**: The model will independently process each scale variant. The three resulting raw logits will be averaged to produce the final `p_final` prediction for the sample.

## 5. Training Protocol
* **Training Dataset**: TN5000 (Train Split) + AUITD (Full)
* **Validation Strategy**: TN5000 (Val Split) at 1.00x scale
* **Optimizer**: AdamW
* **Learning Rate**: `lr_head = 3e-4`, `lr_backbone = 3e-5`
* **Scheduler**: CosineAnnealingWarmRestarts (`T_0=10`, `T_mult=2`)
* **Weight Decay**: `1e-4`
* **Loss Function**: Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`)
* **Label Smoothing**: `eps = 0.05`
* **Positional Weighting**: Dynamically calculated based on class imbalance in the training set (`pos_weight`).
* **Gradient Clipping**: `max_norm = 1.0`
* **Batch Size**: 16
* **Max Epochs**: 25
* **Early Stopping**: Patience of 10 epochs, minimum delta of `0.001` on validation AUC.
* **Precision**: Automatic Mixed Precision (AMP) `float16` via `torch.amp` and `GradScaler`.

### 5.1 Training Augmentations
* Random Rotation (limit=15, p=1.0)
* Horizontal Flip (p=0.5)
* Color Jitter (brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0)
* Gaussian Blur (blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2)

### 5.2 Freeze Schedule
* **Epochs 1-5**: Entire backbone frozen. Only fusion module and head are trainable.
* **Epochs 6-9**: `layers.3` (Stage 3) and backbone norm layers are unfrozen.
* **Epochs 10-25**: Entire backbone unfrozen.

## 6. Final Model Selection Rule
The final evaluation will employ a **Multi-Seed Ensemble**.
* **Seed Policy**: 5 independent runs will be trained from scratch using Python/NumPy/PyTorch seeds `0, 1, 2, 3, 4`.
* **Checkpoint Selection**: For each seed, the checkpoint yielding the highest Internal Validation AUC (TN5000 Val) will be selected.
* **Final Inference**: 
  1. Each sample from the final external dataset (DDTI) will be processed by all 5 seeds.
  2. Each seed will evaluate the sample using the 3-scale TTA (`0.85x`, `1.00x`, `1.15x`).
  3. The 3 scale logits will be averaged to produce a *seed-level prediction*.
  4. The 5 seed-level predictions will be averaged to produce the *final ensemble prediction*.
* **Threshold Selection**: The classification threshold (for sensitivity, specificity, accuracy, F1) will be determined via Youden's J statistic calculated on the internal validation set predictions (at 1.00x). The threshold will NOT be tuned on the final external dataset.
