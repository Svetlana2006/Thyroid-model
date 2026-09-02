"""
Experiment 02b: Preprocessing ablation.

Trains Swin-Tiny (single seed) under four preprocessing conditions to
isolate the effect of geometric distortion and nodule context on
internal vs external validation performance.

Pipeline A: Current (anisotropic Resize(256,256) + CenterCrop(224))
Pipeline B: Aspect-ratio-preserving (LongestMaxSize(256) + Pad + CenterCrop(224))
Pipeline C: Nodule-centered crop (bbox + 20% padding, TN5000 only)
Pipeline D: Nodule-centered crop + AR-preserving

IMPORTANT: This experiment requires training 4 models. On CPU this may
take many hours. Use --dry-run to verify configs without training.

Usage:
    python experiments/02_preprocessing_ablation/run_preprocessing_ablation.py
    python experiments/02_preprocessing_ablation/run_preprocessing_ablation.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image as PILImage

import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset import TN5000Dataset, AUITDDataset
from src.models import build_model
from src.trainer import train_model, evaluate
from src.metrics import compute_metrics, sigmoid, youden_threshold
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = Path("experiments/02_preprocessing_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("data_raw/TN5000_forReview")
TRAIN_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "train.txt")
VAL_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "val.txt")
TEST_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "test.txt")

ARCH = "swin_tiny"
SEED = 0
BATCH_SIZE = 16

_USE_GPU = torch.cuda.is_available()
_PIN_MEM = _USE_GPU
_N_WORK = 4 if _USE_GPU else 0


def pipeline_a_train(img_size=224):
    """Current pipeline: anisotropic resize."""
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.9, 1.0),
                            ratio=(0.75, 1.333), p=1.0),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_a_val(img_size=224):
    """Current pipeline: anisotropic resize."""
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_b_train(img_size=224):
    """Aspect-ratio-preserving resize."""
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.RandomCrop(img_size, img_size),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_b_val(img_size=224):
    """Aspect-ratio-preserving resize."""
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_e_train(img_size=224):
    """AR-preserving full image with explicit letterboxing."""
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_e_val(img_size=224):
    """AR-preserving full image with explicit letterboxing."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import glob
import kagglehub
from torch.utils.data import Dataset

class DiveshDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        for cls_dir, label in [(os.path.join(dataset_dir, "0"), 0), (os.path.join(dataset_dir, "1"), 1)]:
            if os.path.exists(cls_dir):
                for img_file in glob.glob(os.path.join(cls_dir, "*.*")):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append({"img_path": img_file, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(PILImage.open(sample["img_path"]).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(sample["label"], dtype=torch.float32)

def run_single_pipeline(pipeline_name, train_transform, val_transform,
                        crop_nodule, device, dry_run=False, divesh_root=None):
    """Train and evaluate a single preprocessing pipeline."""
    print(f"\n{'='*60}")
    print(f"Pipeline {pipeline_name}")
    print(f"{'='*60}")

    set_seed(SEED)

    # Build datasets
    train_ds_tn = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT,
                                 transform=train_transform,
                                 crop_nodule=crop_nodule)
    auitd_ds = AUITDDataset("data_raw/auitd_dataset", transform=train_transform)
    train_ds = torch.utils.data.ConcatDataset([train_ds_tn, auitd_ds])

    val_ds = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=val_transform)
    test_ds = TN5000Dataset(str(DATA_ROOT), TEST_TXT, transform=val_transform)
    
    divesh_ds = None
    if divesh_root:
        divesh_ds = DiveshDataset(divesh_root, transform=val_transform)

    # Class weighting (same formula as original)
    tn_labels = train_ds_tn.get_labels()
    au_labels = auitd_ds.get_labels()
    combined_labels = np.concatenate([tn_labels, au_labels])
    pos_weight = float((combined_labels == 0).sum() / (combined_labels == 1).sum())

    config = {
        "lr_head": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "pos_weight": pos_weight,
        "pos_weight_scale": 1.0,
        "batch_size": BATCH_SIZE,
        "max_epochs": 25,
        "patience": 10,
        "min_delta": 0.001,
        "T_0": 10,
        "T_mult": 2,
        "label_smooth_eps": 0.05,
        "grad_clip_norm": 1.0,
    }

    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if divesh_ds:
        print(f"  External (Divesh): {len(divesh_ds)} samples")
    print(f"  pos_weight: {pos_weight:.4f}")
    print(f"  crop_nodule: {crop_nodule}")

    if dry_run:
        print("  [DRY RUN] Skipping training.")
        return None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=_N_WORK, pin_memory=_PIN_MEM)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=_N_WORK, pin_memory=_PIN_MEM)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=_N_WORK, pin_memory=_PIN_MEM)
                             
    model = build_model(ARCH, dropout=config["dropout"])
    ckpt_dir = OUTPUT_DIR / "checkpoints"

    history = train_model(
        model, train_loader, val_loader, config,
        checkpoint_dir=str(ckpt_dir),
        run_name=f"{ARCH}_{pipeline_name}_seed{SEED}",
        device=device,
    )

    # Evaluate
    test_metrics = evaluate(model, test_loader, device,
                            config["pos_weight"], config["pos_weight_scale"])
    val_metrics = evaluate(model, val_loader, device,
                           config["pos_weight"], config["pos_weight_scale"])

    threshold = youden_threshold(val_metrics["logits"], val_metrics["labels"])
    metrics = compute_metrics(test_metrics["logits"], test_metrics["labels"],
                              threshold=threshold)
                              
    ext_auc = 0.0
    if divesh_ds:
        divesh_loader = DataLoader(divesh_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                 num_workers=_N_WORK, pin_memory=_PIN_MEM)
        ext_metrics_raw = evaluate(model, divesh_loader, device,
                                config["pos_weight"], config["pos_weight_scale"])
        ext_metrics = compute_metrics(ext_metrics_raw["logits"], ext_metrics_raw["labels"],
                                  threshold=threshold)
        ext_auc = float(ext_metrics["auc"])

    result = {
        "pipeline": pipeline_name,
        "arch": ARCH,
        "seed": SEED,
        "crop_nodule": crop_nodule,
        "internal_auc": float(metrics["auc"]),
        "external_auc": ext_auc,
        "delta_auc": float(metrics["auc"]) - ext_auc,
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "best_val_auc": float(history["best_val_auc"]),
        "config": config,
    }

    print(f"\n  Internal Test AUC: {metrics['auc']:.4f}")
    if divesh_ds:
        print(f"  External Test AUC: {ext_auc:.4f}")
        print(f"  Delta AUC: {result['delta_auc']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify configs without training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Downloading/Locating Divesh Dataset...")
    divesh_root = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')

    pipelines = [
        ("A_current", pipeline_a_train(), pipeline_a_val(), False),
        ("B_ar_preserving", pipeline_b_train(), pipeline_b_val(), False),
        ("C_nodule_crop", pipeline_a_train(), pipeline_a_val(), True),
        ("D_nodule_crop_ar", pipeline_b_train(), pipeline_b_val(), True),
        ("E_letterbox", pipeline_e_train(), pipeline_e_val(), False),
    ]

    results = []
    for name, train_t, val_t, crop in pipelines:
        r = run_single_pipeline(name, train_t, val_t, crop, device, args.dry_run, divesh_root)
        if r is not None:
            results.append(r)

    if results:
        with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print comparison table
        print(f"\n{'='*90}")
        print("PREPROCESSING ABLATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Pipeline':<25} {'Int AUC':>10} {'Ext AUC':>10} {'dAUC':>10} {'Sens':>8} {'Spec':>8} {'F1':>8}")
        print("-" * 90)
        for r in results:
            print(f"{r['pipeline']:<25} {r['internal_auc']:>10.4f} "
                  f"{r['external_auc']:>10.4f} {r['delta_auc']:>10.4f} "
                  f"{r['sensitivity']:>8.4f} {r['specificity']:>8.4f} "
                  f"{r['f1']:>8.4f}")

        print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
