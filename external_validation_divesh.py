"""
External Validation Script for Divesh Dataset
Evaluates the frozen 5-seed MultiLevelSwin ensemble on the diveshzz dataset.
"""

import argparse
import glob
import os
import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import kagglehub
import cv2

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import timm
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Architecture (Must match train.py exactly) ────────────────────────────────
PROJ_DIM = 128
FUSION_DIM = 256

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        n_stages = len(self.STAGE_CHANNELS)
        self.stage_norms = nn.ModuleDict()
        self.stage_projs = nn.ModuleDict()
        for name, ch in self.STAGE_CHANNELS.items():
            key = name.replace(".", "_")
            self.stage_norms[key] = nn.LayerNorm(ch)
            self.stage_projs[key] = nn.Linear(ch, PROJ_DIM, bias=False)
        self.fusion_head = nn.Sequential(
            nn.Linear(PROJ_DIM * n_stages, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(FUSION_DIM, 1),
        )
        self._stage_feats = {}
        self._hooks = []

    def _register_hooks(self):
        for name in self.STAGE_CHANNELS:
            module = dict(self.backbone.named_modules())[name]
            handle = module.register_forward_hook(lambda mod, inp, out, n=name: self._stage_feats.update({n: out}))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._stage_feats.clear()
        self._register_hooks()
        _ = self.backbone(x)
        self._remove_hooks()
        pooled = []
        for name in self.STAGE_CHANNELS:
            key = name.replace(".", "_")
            feat = self._stage_feats[name].mean(dim=(1, 2))
            feat = self.stage_norms[key](feat)
            feat = self.stage_projs[key](feat)
            pooled.append(feat)
        fused = torch.cat(pooled, dim=-1)
        return self.fusion_head(fused)

# ── Multi-Scale TTA Transforms ────────────────────────────────────────────────
def make_val_transform(scale: float = 1.0):
    max_size = round(256 * scale)
    return A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(min_height=max(max_size, 224), min_width=max(max_size, 224), border_mode=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

TTA_SCALES = [0.85, 1.00, 1.15]
TTA_TRANSFORMS = [make_val_transform(s) for s in TTA_SCALES]

# ── Divesh Dataset Loader ─────────────────────────────────────────────────────
class DiveshDataset(Dataset):
    def __init__(self, data_root: str, transforms: list):
        self.data_root = data_root
        self.transforms = transforms
        self.samples = []
        
        # Traverse dataset directory
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        
        # Benign = 0
        class_0_dir = os.path.join(dataset_dir, "0")
        if os.path.exists(class_0_dir):
            for img_file in glob.glob(os.path.join(class_0_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        "img_path": img_file,
                        "label": 0
                    })
                    
        # Malignant = 1
        class_1_dir = os.path.join(dataset_dir, "1")
        if os.path.exists(class_1_dir):
            for img_file in glob.glob(os.path.join(class_1_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        "img_path": img_file,
                        "label": 1
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        label = sample["label"]
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply TTA transforms
        tensors = []
        for t in self.transforms:
            aug = t(image=image)
            tensors.append(aug["image"])
        
        # Stack into shape: (num_tta, C, H, W)
        tensors = torch.stack(tensors)
        return tensors, label

# ── Evaluation Logic ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_seed(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    
    iterator = tqdm(loader, desc="Evaluating", leave=False) if HAS_TQDM else loader
    for tensors, labels in iterator:
        tensors = tensors.to(device) # Shape: (B, num_tta, C, H, W)
        B, num_tta, C, H, W = tensors.shape
        tensors = tensors.view(B * num_tta, C, H, W)
        
        logits = model(tensors).squeeze(-1) # (B * num_tta)
        logits = logits.view(B, num_tta)
        
        # Average logits across TTA scales
        avg_logits = logits.mean(dim=1)
        
        all_logits.extend(avg_logits.cpu().float().tolist())
        all_labels.extend(labels.tolist())
        
    return np.array(all_logits), np.array(all_labels)

def get_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = np.percentile(sorted_scores, (100 - ci) / 2)
    upper = np.percentile(sorted_scores, 100 - (100 - ci) / 2)
    return lower, upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="outputs/final_model", help="Path to seed folders")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Downloading dataset...")
    data_root = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
    print(f"Dataset path: {data_root}")
    
    # Build dataset
    dataset = DiveshDataset(data_root, TTA_TRANSFORMS)
    print(f"Loaded {len(dataset)} samples from Divesh dataset.")
    if len(dataset) == 0:
        print("No samples found. Please check dataset extraction.")
        return

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)

    # Accumulate ensemble logits
    ensemble_logits = np.zeros(len(dataset))
    targets = None

    seeds = [0, 1, 2, 3, 4]
    
    print("\nStarting evaluation of 5-seed ensemble with Multi-Scale TTA...")
    for seed in seeds:
        ckpt_path = Path(args.models_dir) / f"seed{seed}" / "best.pt"
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for seed {seed}: {ckpt_path}")
            continue

        model = MultiLevelSwin(dropout=0.0).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
        logits, labels = evaluate_seed(model, loader, device)
        if targets is None:
            targets = labels
            
        seed_auc = roc_auc_score(labels, logits)
        print(f"Seed {seed} | TTA AUROC: {seed_auc:.4f}")
        
        ensemble_logits += logits

    ensemble_logits /= len(seeds)
    ensemble_auc = roc_auc_score(targets, ensemble_logits)
    lower, upper = get_bootstrap_ci(targets, ensemble_logits)
    
    print("\n" + "="*50)
    print("DIVESH EXTERNAL VALIDATION RESULTS")
    print("="*50)
    print(f"Ensemble TTA AUROC : {ensemble_auc:.4f}")
    print(f"95% Confidence Int : [{lower:.4f}, {upper:.4f}]")
    print("="*50)
    
    out_file = Path(args.models_dir) / "divesh_results.json"
    if Path(args.models_dir).exists():
        with open(out_file, "w") as f:
            json.dump({
                "dataset": "diveshzz",
                "samples": len(targets),
                "ensemble_auc": ensemble_auc,
                "ci_lower": lower,
                "ci_upper": upper
            }, f, indent=2)
        print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
