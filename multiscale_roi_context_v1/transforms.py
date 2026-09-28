"""
Transforms for multiscale_roi_context_v1.

Full-image train transform: exact A4 preprocessing from main model.
ROI transform: same normalization, but geometry-safe for crops
  (LongestMaxSize + PadIfNeeded + RandomCrop/CenterCrop — no independent
   geometric random ops that would misalign views).
TTA: 5-scale established protocol, no horizontal flip.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
TTA_SCALES    = [0.70, 0.85, 1.00, 1.15, 1.30]


def make_full_train_transform() -> A.Compose:
    """
    A4 full-image training transform — identical to main model train.py.
    Applied to the full image BEFORE ROI crops are extracted.
    """
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.RandomCrop(224, 224),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def make_roi_train_transform() -> A.Compose:
    """
    ROI crop training transform.

    Uses LongestMaxSize + PadIfNeeded + RandomCrop to handle variable-size
    crops while preserving aspect ratio. NO independent Rotate/Flip — these
    are applied to the source image before cropping so all three views remain
    geometrically consistent with each other.
    ColorJitter and GaussianBlur are retained for photometric augmentation.
    """
    return A.Compose([
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.RandomCrop(224, 224),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def make_val_transform(scale: float = 1.0) -> A.Compose:
    """
    Validation/test transform for a single scale.
    Applied to full image and ROI crops alike.
    """
    max_size = round(256 * scale)
    return A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(
            min_height=max(max_size, 256),
            min_width=max(max_size, 256),
            border_mode=0,
        ),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def make_tta_transforms() -> list:
    """
    5-scale TTA transforms matching the established main-model TTA.
    No horizontal flip. Each transform is safe for ROI crops.
    """
    return [make_val_transform(scale) for scale in TTA_SCALES]
