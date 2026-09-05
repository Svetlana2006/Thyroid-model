# Final Main Model Training Plan

This document defines the exact specifications for training the final frozen model. This is an audit and synthesis of the protocols established in Experiments 17 (Fusion), 18 (TTA), and 19 (Multi-Seed). 

---

### A. Frozen Architecture
The final architecture is exactly the `MultiLevelSwin` implemented in `experiments/17_multilevel_fusion/run_experiment17.py`:
* **Backbone**: `swin_tiny_patch4_window7_224` (initialized with ImageNet weights).
* **Fusion Stages**: `layers.1` (192 channels), `layers.2` (384 channels), and `layers.3` (768 channels).
* **Fusion Logic**:
  * Global Average Pooling applied to each extracted stage.
  * Independent `nn.LayerNorm` applied to each pooled stage.
  * Bias-free `nn.Linear` projection from stage channels to `128` dimensions.
  * Concatenation of the three 128-d vectors into a single `384`-dimensional feature vector.
* **Classification Head**: `Linear(384, 256) -> GELU -> Dropout(0.3) -> Linear(256, 1)`.

### B. Frozen Preprocessing
The model strictly uses aspect-ratio-preserving preprocessing:
* **Training**: `LongestMaxSize(256) -> PadIfNeeded(256, 256) -> RandomCrop(224, 224)`
* **Validation/Inference (1.00x)**: `LongestMaxSize(256) -> PadIfNeeded(256, 256) -> CenterCrop(224, 224)`
* **Augmentation**: `Rotate(15, p=1.0) -> HorizontalFlip(p=0.5) -> ColorJitter(0.15, p=1.0) -> GaussianBlur(limit=3, sigma=(0.1, 1.0), p=0.2)`
* **Normalization**: standard ImageNet mean and std.

### C. Frozen Training Protocol
Each seed will be trained completely independently with the following fixed parameters:
* **Optimizer**: AdamW
* **Learning Rate**: `3e-4` (Head) and `3e-5` (Backbone)
* **Scheduler**: CosineAnnealingWarmRestarts (`T_0=10`, `T_mult=2`)
* **Weight Decay**: `1e-4`
* **Batch Size**: `16`
* **Precision**: PyTorch AMP (`float16` via `GradScaler`)
* **Epochs**: Maximum 25
* **Staged Unfreezing**: 
  * Epochs 1-5: Entire backbone frozen.
  * Epochs 6-9: `layers.3` and backbone `norm` unfrozen.
  * Epochs 10-25: Entire backbone unfrozen.
* **Loss Function**: `BCEWithLogitsLoss` using training set `pos_weight` and `label_smooth_eps=0.05`.
* **Random Seeding**: Seeds `0, 1, 2, 3, 4` applied globally to Python, NumPy, PyTorch, and CUDA. DataLoaders use a `worker_init_fn` parameterized by `global_seed + worker_id` to strictly guarantee differential data shuffling.

### D. Final Ensemble Protocol (Inference)
The final system relies on a **5-Seed Deep Ensemble with Multi-Scale TTA**:
1. Every incoming sample is processed at three scales: `0.85x` (max size 218), `1.00x` (max size 256), and `1.15x` (max size 294).
2. For each seed, the 3 scale logits are averaged.
3. The 5 seed-level logits are then averaged to produce the final system prediction.

### E. Data Usage
* **Training Data**: TN5000 (Train Split) + AUITD (Full).
* **Validation Data**: TN5000 (Val Split). *Strictly used for early stopping and threshold selection.*
* **Development-External Dataset**: Divesh. *Previously used in Exp 1-18 for architectural decisions. It will NOT be used during final training.*
* **Final External Validation Dataset**: Completely untouched (to be provided by user).

### F. Checkpoint Strategy
For each of the 5 seeds, the checkpoint corresponding to the **highest AUROC on the TN5000 Validation Split** will be selected. Early stopping is monitored with a patience of 10 epochs and a `min_delta` of 0.001. All 5 "best" checkpoints will form the final ensemble.

### G. Reproducibility Checklist
To guarantee parity with Experiments 17-19:
- [x] No domain-adversarial layers (DANN).
- [x] No arbitrary nodule-only bounding box cropping.
- [x] Precision (AMP) must remain active, as FP32 caused numerical instability in early experiments.
- [x] Optimizer states must be rebuilt when unfreezing backbone layers.

### H. Exact Command/Script Recommendation
The repository currently relies on `experiments/19_multiseed_confirmation/run_experiment19.py`. While that script trains all 5 seeds perfectly, it also contains evaluation code targeting the Divesh dataset, which violates the strict principle of the final training run (which should not observe external data). 

Therefore, a clean final-training script should be created (e.g., `experiments/final_training/run_final_training.py`).
The exact command will be:
```bash
python experiments/final_training/run_final_training.py
```

---

## Conclusion
**The final model should be a 5-seed ensemble.**

*Because*: The final external evaluation AUROC of **0.8254** (and its confidence interval) was strictly computed and reported as the output of the 5-seed average. A single-seed Swin-Tiny model exhibits an external AUROC distribution bounded between `0.806` and `0.828`. To scientifically guarantee the reported `0.8254` generalization performance, the deployed system *must* encapsulate the exact ensembling mechanism that generated that score. Reducing the system to a single seed post-evaluation would invalidate the statistical significance of the final report.
