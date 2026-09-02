"""
Experiment 5: 224 vs Native (288) EfficientNet-B3 Resolution
Investigates whether downsampling to 224x224 instead of the native 288x288
degrades internal or external performance.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import TN5000Dataset, AUITDDataset
from src.models import build_model
from src.trainer import train_model, evaluate
from src.metrics import compute_metrics, youden_threshold
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = Path("experiments/13_efficientnet_resolution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("data_raw/TN5000_forReview")
TRAIN_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "train.txt")
VAL_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "val.txt")
TEST_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "test.txt")

import glob
from torch.utils.data import Dataset
from PIL import Image as PILImage

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

def get_transforms(img_size):
    """Current preprocessing pipeline (A.Resize -> CenterCrop) adapted to img_size."""
    train_t = A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.9, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    # Validation uses a slightly larger initial resize before crop, 
    # maintaining the ratio 256/224 ~ 1.14
    val_resize = int(img_size * 1.14)
    val_t = A.Compose([
        A.Resize(val_resize, val_resize),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_t, val_t

def run_experiment(resolution, device, divesh_root, dry_run=False, max_epochs=25):
    print(f"\n{'='*60}")
    print(f"Testing Resolution: {resolution}x{resolution}")
    print(f"{'='*60}")
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    train_t, val_t = get_transforms(resolution)
    
    train_ds_tn = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t)
    auitd_ds = AUITDDataset("data_raw/auitd_dataset", transform=train_t)
    train_ds = torch.utils.data.ConcatDataset([train_ds_tn, auitd_ds])
    val_ds = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=val_t)
    test_ds = TN5000Dataset(str(DATA_ROOT), TEST_TXT, transform=val_t)
    divesh_ds = DiveshDataset(divesh_root, transform=val_t)
    
    tn_labels = train_ds_tn.get_labels()
    au_labels = auitd_ds.get_labels()
    combined_labels = np.concatenate([tn_labels, au_labels])
    pos_weight = float((combined_labels == 0).sum() / (combined_labels == 1).sum())
    
    config = {
        "lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3, "pos_weight": pos_weight, "pos_weight_scale": 1.0,
        "batch_size": 16, "max_epochs": max_epochs, "patience": 10, "min_delta": 0.001,
        "T_0": 10, "T_mult": 2, "label_smooth_eps": 0.05, "grad_clip_norm": 1.0,
    }
    
    if dry_run:
        print("  [DRY RUN] Skipping training.")
        return None
        
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"]*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"]*2, shuffle=False, num_workers=0)
    divesh_loader = DataLoader(divesh_ds, batch_size=config["batch_size"]*2, shuffle=False, num_workers=0)
    
    model = build_model("efficientnet_b3", dropout=config["dropout"])
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    
    history = train_model(
        model, train_loader, val_loader, config,
        checkpoint_dir=str(ckpt_dir),
        run_name=f"efficientnet_b3_{resolution}_seed0",
        device=device,
    )
    
    # Eval internal
    val_metrics = evaluate(model, val_loader, device, config["pos_weight"], config["pos_weight_scale"])
    threshold = youden_threshold(val_metrics["logits"], val_metrics["labels"])
    test_metrics = evaluate(model, test_loader, device, config["pos_weight"], config["pos_weight_scale"])
    int_results = compute_metrics(test_metrics["logits"], test_metrics["labels"], threshold=threshold)
    
    # Eval external
    ext_metrics = evaluate(model, divesh_loader, device, config["pos_weight"], config["pos_weight_scale"])
    ext_results = compute_metrics(ext_metrics["logits"], ext_metrics["labels"], threshold=threshold)
    
    result = {
        "resolution": resolution,
        "internal_auc": float(int_results["auc"]),
        "external_auc": float(ext_results["auc"]),
        "delta_auc": float(int_results["auc"]) - float(ext_results["auc"]),
        "best_val_auc": float(history["best_val_auc"]),
    }
    
    print(f"\n  Internal Test AUC: {result['internal_auc']:.4f}")
    print(f"  External Test AUC: {result['external_auc']:.4f}")
    print(f"  Delta AUC:         {result['delta_auc']:.4f}")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    divesh_root = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
    
    results = []
    for res in [224, 288]:
        r = run_experiment(res, device, divesh_root, dry_run=args.dry_run, max_epochs=args.max_epochs)
        if r:
            results.append(r)
            
    if results:
        with open(OUTPUT_DIR / "resolution_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
