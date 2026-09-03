# Experiment 14 — Cross-Architecture Validation of AR-Preserving Preprocessing

## Purpose

Determine whether the external-AUC improvement observed with AR-preserving preprocessing (Exp 02, Swin-Tiny) generalizes across all three architectures, or is architecture-specific.

## Design

Controlled ablation: **one variable only** (preprocessing), three architectures, seed 0.

| Run | Architecture    | Preprocessing | Seed |
|-----|----------------|---------------|------|
| 1   | ResNet50        | Current (A)   | 0    |
| 2   | ResNet50        | AR-preserving (B) | 0 |
| 3   | EfficientNet-B3 | Current (A)   | 0    |
| 4   | EfficientNet-B3 | AR-preserving (B) | 0 |
| 5   | Swin-Tiny       | Current (A)   | 0    |
| 6   | Swin-Tiny       | AR-preserving (B) | 0 |

EfficientNet-B3 uses 224×224 for both pipelines (see Exp 13 for 288 vs 224 analysis).

## Datasets

- **Training**: TN5000 train split + AUITD (full)
- **Validation (model selection)**: TN5000 val split
- **Internal evaluation**: TN5000 test split
- **External evaluation**: Divesh (Kaggle) — used only for final evaluation, never for model selection

DDTI-unique is excluded as a meaningful training source (1 image).

## Hyperparameters

All identical to Exp 02 baseline:
- batch_size = 16, max_epochs = 25, patience = 10, min_delta = 0.001
- AdamW + CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- dropout = 0.3, label_smooth_eps = 0.05, grad_clip = 1.0
- pos_weight from combined TN5000+AUITD labels

## Sanity Checks

Run before training begins:
1. Same source images for both pipelines
2. 20 side-by-side diagnostic images (original / Pipeline A / Pipeline B)
3. Transformed bbox overlays for both pipelines on 20 TN5000 images
4. Label/split identity check
5. Divesh isolation check

## Outputs

- `configs/` — per-run config JSON
- `checkpoints/` — best model per run
- `logs/` — per-epoch training curves
- `metrics/` — per-run metrics JSON (internal + external + bootstrap CI)
- `plots/` — training curves + AUC comparison bar chart
- `diagnostics/` — sanity check images
- `results.csv` — final comparison table
- `RESULTS.md` — interpretation document

## Usage

```bash
python experiments/14_cross_arch_ar/run_experiment14.py --dry-run   # sanity checks only
python experiments/14_cross_arch_ar/run_experiment14.py              # full training
```

**Important**: Run on GPU. On CPU this will take many hours per model.
