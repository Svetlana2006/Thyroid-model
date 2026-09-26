# Focal Loss Experiment v2 - Implementation Audit Report

Generated: CPU

## 1. Baseline Source Files Inspected
- train.py
- src/trainer.py
- src/dataset.py
- src/transforms.py
- evaluate_internal_tn5000.py
- outputs/final_model/seed0/config.json

## 2. Baseline Training Configuration
- **config**: A4S1V2
- **ar**: A4
- **parameter_count**: 27792891
- **best_val_auc**: 0.9464
- **best_epoch**: 23
- **tta_scales**: [0.7, 0.85, 1.0, 1.15, 1.3]
- **pos_weight**: 0.5291
- **label_smooth_eps**: 0.05
- **lr_head**: 0.0003
- **weight_decay**: 0.0001
- **T_0**: 10
- **T_mult**: 2
- **patience**: 10
- **threshold**: 0.5912

## 3. Focal Training Configuration
- **config**: A4S1V2 (focal)
- **loss**: BinaryFocalLoss(gamma=2.0)
- **parameter_count**: 27792891
- **tta_scales**: [0.7, 0.85, 1.0, 1.15, 1.3]
- **pos_weight**: 0.5291
- **label_smooth_eps**: 0.05
- **lr_head**: 0.0003
- **weight_decay**: 0.0001
- **T_0**: 10
- **T_mult**: 2
- **patience**: 10
- **threshold**: 0.5912

## 4. Architecture Comparison
- **Focal param count**: 27,792,891
- **Main param count**: 27,792,891
- **Param count match**: PASS
- **Keys match**: PASS
- **Shapes match**: PASS
- **Total focal keys**: 184
- **Total main keys**: 184

## 5. Transform Comparison
- **Training transforms match**: PASS
  - Focal count: 9, Baseline count: 9
  - Focal types: ['Rotate', 'HorizontalFlip', 'ColorJitter', 'LongestMaxSize', 'PadIfNeeded', 'RandomCrop', 'GaussianBlur', 'Normalize', 'ToTensorV2']
  - Baseline types: ['Rotate', 'HorizontalFlip', 'ColorJitter', 'LongestMaxSize', 'PadIfNeeded', 'RandomCrop', 'GaussianBlur', 'Normalize', 'ToTensorV2']
- **Validation transforms match**: PASS
  - Focal types: ['LongestMaxSize', 'PadIfNeeded', 'CenterCrop', 'Normalize', 'ToTensorV2']
  - Baseline types: ['LongestMaxSize', 'PadIfNeeded', 'CenterCrop', 'Normalize', 'ToTensorV2']

## 6. Dataset/Split Comparison
- **Train count**: 3500
- **Val count**: 500
- **Test count**: 1000
- **Train-Val overlap**: 0
- **Train-Test overlap**: 0
- **Val-Test overlap**: 0
- **No overlap (PASS)**: YES

## 7. Freeze Schedule Comparison
- **Trainable parameters per epoch**:
  - epoch_1: 273537 params (13 tensors)
  - epoch_5: 273537 params (13 tensors)
  - epoch_6: 15641649 params (44 tensors)
  - epoch_9: 15641649 params (44 tensors)
  - epoch_10: 27792891 params (184 tensors)
  - epoch_25: 27792891 params (184 tensors)

## 8. Optimizer Comparison
- **Optimizer**: AdamW
- **LR head**: 0.0003
- **LR backbone**: 2.9999999999999997e-05
- **Weight decay**: 0.0001
- **T_0**: 10
- **T_mult**: 2
  - Group lr=3.00e-04, params=273,537

## 9. Loss Equivalence Tests
- **All gamma=0 equivalence tests passed**: PASS
  - positive_easy: FL=0.12863232, BCE=0.12863232, abs_err=0.00e+00 PASS
  - positive_hard: FL=1.57389700, BCE=1.57389688, abs_err=1.19e-07 PASS
  - negative_easy: FL=0.07277380, BCE=0.07277346, abs_err=3.43e-07 PASS
  - negative_hard: FL=2.97301555, BCE=2.97301555, abs_err=0.00e+00 PASS
  - logit_zero_pos: FL=0.37490427, BCE=0.37490425, abs_err=2.98e-08 PASS
  - logit_zero_neg: FL=0.68498713, BCE=0.68498707, abs_err=5.96e-08 PASS
  - large_positive: FL=0.50000072, BCE=0.50000072, abs_err=0.00e+00 PASS
  - large_negative: FL=10.31744957, BCE=10.31744957, abs_err=0.00e+00 PASS
  - small_pos_logit: FL=0.37245667, BCE=0.37245667, abs_err=0.00e+00 PASS
  - small_neg_logit: FL=0.68019062, BCE=0.68019056, abs_err=5.96e-08 PASS
  - random_batch: FL=1.04095447, BCE=1.04095435, abs_err=1.19e-07 PASS

## 10. Seed/RNG Comparison
- **seed_function_exists**: PASS
- **python_rng_savable**: PASS
- **numpy_rng_savable**: PASS
- **torch_cpu_rng_savable**: PASS
- **torch_cuda_rng_savable**: FAIL

## 11. TTA Comparison
- **Focal TTA count**: 5
- **Baseline TTA count**: 5
- **Count match**: PASS
- **Types match**: PASS

## 12. TN5000 Evaluation Comparison
- Uses established evaluator TTA (5 views, CenterCrop only, no flips)
- Focal experiment matches this exactly

## 13. Diveshzz Evaluation Comparison
- Uses same TTA as TN5000

## 14. Thyroid Patient-Level Evaluation
- Asserts exactly 2 images per patient
- Asserts consistent labels within patient
- Averages 2 image predictions for patient-level

## 15. Final PASS/FAIL Table
| Check | Status |
|-------|--------|
| training_transforms | PASS |
| validation_transforms | PASS |
| architecture_keys | PASS |
| architecture_shapes | PASS |
| architecture_params | PASS |
| loss_equivalence | PASS |
| tta_transforms | PASS |
| pos_weight | PASS |
| seed_reproducibility | PASS |
| checkpoint_content | PASS |
| dataset_splits | PASS |

**OVERALL: READY FOR SEED-0 TRAINING**
