"""
Transforms for TN5000 (albumentation-based).
Returns callables that work with `image` keyword from albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Training augmentations per §6 of the plan:
      - RandomRotation(15°)
      - RandomHorizontalFlip(p=0.5)   [no vertical flip]
      - ColorJitter(brightness=0.15, contrast=0.15)
      - RandomResizedCrop(224, scale=(0.9, 1.0))
      - GaussianBlur(p=0.2)
      - Normalize (ImageNet)
      - ToTensor
    """
    return A.Compose(
        [
            A.Rotate(limit=15, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.9, 1.0),
                ratio=(0.75, 1.333),
                p=1.0,
            ),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Validation / test transforms: resize + center crop + normalize.
    """
    return A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(img_size: int = 224):
    """
    Test-time augmentation: 5 crops + horizontal flip variants.
    Returns a list of transforms for TTA ensemble.
    """
    norm_tensor = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    # (x_min, y_min, x_max, y_max) offsets within a 256x256 resized image
    crop_coords = [
        (16, 16, 16 + img_size, 16 + img_size),   # centre
        (0,  0,  img_size,      img_size),          # top-left
        (32, 0,  32 + img_size, img_size),          # top-right
        (0,  32, img_size,      32 + img_size),     # bottom-left
        (32, 32, 32 + img_size, 32 + img_size),     # bottom-right
    ]
    transforms_list = []
    for (x0, y0, x1, y1) in crop_coords:
        transforms_list.append(
            A.Compose([A.Resize(256, 256), A.Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1), *norm_tensor])
        )
        transforms_list.append(
            A.Compose([A.Resize(256, 256), A.Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1), A.HorizontalFlip(p=1.0), *norm_tensor])
        )
    return transforms_list
