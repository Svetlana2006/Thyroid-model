"""
Final Main Training Script for Thyroid Model (Frozen Configuration)
This replaces the old development train.py with the strict, frozen 5-seed 
MultiLevelSwin ensemble established by Experiments 17-19.

No external evaluation logic, TTA inference, or Optuna search is performed here.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

# Ensure src is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent))
import timm
from src.dataset import TN5000Dataset, AUITDDataset
from src.trainer import train_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── CPU Heat Mitigation ───────────────────────────────────────────────────────
_CPU_COUNT = os.cpu_count() or 4
torch.set_num_threads(max(1, _CPU_COUNT // 2))
torch.set_num_interop_threads(max(1, _CPU_COUNT // 4))

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs/final_model")
DATA_ROOT = Path("data_raw/TN5000_forReview")
AUITD_ROOT = "data_raw/auitd_dataset"
TRAIN_TXT = str(DATA_ROOT / "ImageSets/Main/train.txt")
VAL_TXT = str(DATA_ROOT / "ImageSets/Main/val.txt")

_USE_GPU = torch.cuda.is_available()
_N_WORK = 4 if _USE_GPU else 0


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_CURRENT_SEED = 0

def worker_init_fn(worker_id):
    global _CURRENT_SEED
    np.random.seed(_CURRENT_SEED + worker_id)


# ── Transforms (Aspect-Ratio Preserving) ──────────────────────────────────────
def make_train_transform():
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

def make_val_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Final Architecture (MultiLevelSwin) ───────────────────────────────────────
PROJ_DIM = 128
FUSION_DIM = 256

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # Pretrained = True exactly as Experiment 19
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
        
        for param in self.backbone.parameters():
            param.requires_grad = False

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
        self._stage_feats: dict = {}
        self._hooks = []

    def _register_hooks(self):
        for name in self.STAGE_CHANNELS:
            module = dict(self.backbone.named_modules())[name]
            handle = module.register_forward_hook(lambda mod, inp, out, n=name: self._stage_feats.update({n: out}))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._stage_feats.clear()
        self._register_hooks()
        _ = self.backbone(x)
        self._remove_hooks()
        pooled = []
        for name in self.STAGE_CHANNELS:
            key  = name.replace(".", "_")
            feat = self._stage_feats[name].mean(dim=(1, 2))
            feat = self.stage_norms[key](feat)
            feat = self.stage_projs[key](feat)
            pooled.append(feat)
        fused = torch.cat(pooled, dim=-1)
        return self.fusion_head(fused)

    def freeze_epoch(self, epoch: int):
        if epoch >= 10:
            for param in self.backbone.parameters(): param.requires_grad = True
        elif epoch >= 6:
            for param in self.backbone.parameters(): param.requires_grad = False
            if hasattr(self.backbone, "layers"):
                for param in self.backbone.layers[-1].parameters(): param.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters(): param.requires_grad = True
        else:
            for param in self.backbone.parameters(): param.requires_grad = False

    def get_param_groups(self, lr_head: float, lr_backbone: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (list(self.stage_norms.parameters()) +
                       list(self.stage_projs.parameters()) +
                       list(self.fusion_head.parameters()))
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups


# ── Training Procedure ────────────────────────────────────────────────────────
def run_seed(seed: int, device: torch.device, sanity_check: bool = False):
    print(f"\n{'='*60}\n  Training Final Model - Seed {seed}\n{'='*60}")
    
    global _CURRENT_SEED
    _CURRENT_SEED = seed
    set_seed(seed)

    # 1. Datasets
    train_t = make_train_transform()
    val_t = make_val_transform()

    tn5000_train = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t)
    auitd_train = AUITDDataset(AUITD_ROOT, transform=train_t)
    train_ds = torch.utils.data.ConcatDataset([tn5000_train, auitd_train])
    
    val_ds = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=val_t)

    # Sanity truncation
    if sanity_check:
        print("  [SANITY CHECK] Truncating dataset to 2 batches...")
        train_ds = Subset(train_ds, list(range(32)))
        val_ds = Subset(val_ds, list(range(32)))

    # Pos Weight
    all_labels = np.concatenate([
        TN5000Dataset(str(DATA_ROOT), TRAIN_TXT).get_labels(),
        AUITDDataset(AUITD_ROOT).get_labels()
    ])
    pos_weight = float((all_labels == 0).sum() / (all_labels == 1).sum())

    # 2. Config
    config = {
        "lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
        "pos_weight": pos_weight, "batch_size": 16,
        "max_epochs": 1 if sanity_check else 25, 
        "patience": 10, "min_delta": 0.001,
        "T_0": 10, "T_mult": 2, "grad_clip_norm": 1.0, "label_smooth_eps": 0.05,
    }

    model = MultiLevelSwin(dropout=config["dropout"])
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Config logged:")
    print(f"  Seed: {seed}")
    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"  Model: MultiLevelSwin | Total Params: {total_params:,}")
    print(f"  Optimizer: AdamW | LR Head: {config['lr_head']} | LR Backbone: {config['lr_head']*0.1}")
    print(f"  Batch size: {config['batch_size']} | Max Epochs: {config['max_epochs']}")

    # 3. Loaders
    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"]*2, shuffle=False, **kw)

    # 4. Train using existing src.trainer (which matches Exp 19 perfectly)
    seed_dir = OUTPUTS_DIR / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=str(seed_dir),
        run_name="final",
        device=device
    )
    train_time = time.time() - t0

    # Ensure checkpoint is named best.pt instead of final_best.pt
    final_best_ckpt = seed_dir / "final_best.pt"
    best_ckpt = seed_dir / "best.pt"
    if final_best_ckpt.exists():
        final_best_ckpt.rename(best_ckpt)

    # Save outputs
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(seed_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "seed": seed,
        "best_epoch": history.get("best_epoch", history.get("epoch", 0)),
        "best_val_auc": history.get("best_val_auc", 0.0),
        "train_time_sec": train_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run 1 epoch with 2 batches to verify pipeline.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    SEEDS = [0] if args.sanity_check else [0, 1, 2, 3, 4]
    results = []

    for s in SEEDS:
        r = run_seed(s, device, sanity_check=args.sanity_check)
        results.append(r)

    if not args.sanity_check:
        summary_path = OUTPUTS_DIR / "summary.csv"
        fieldnames = ["seed", "best_epoch", "best_val_auc", "train_time_sec"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\nFinal training complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
