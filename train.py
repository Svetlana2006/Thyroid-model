"""Final Main Training Script for Thyroid Model (A4S1V2 Configuration)

This script is exactly identical to the A4S1V2 configuration in
training_experiments/train_stage2.py. The only difference is the
output directory: outputs/final_model/ instead of training_experiments/.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch.nn as nn
import timm


class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # Pretrained = True exactly as Experiment 19
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        n_stages = len(self.STAGE_CHANNELS)
        self.stage_norms = nn.ModuleDict()
        self.stage_projs = nn.ModuleDict()
        for name, ch in self.STAGE_CHANNELS.items():
            key = name.replace(".", "_")
            self.stage_norms[key] = nn.LayerNorm(ch)
            self.stage_projs[key] = nn.Linear(ch, 128, bias=False)

        self.fusion_head = nn.Sequential(
            nn.Linear(128 * n_stages, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
        self._stage_feats: dict = {}
        self._hooks = []

    def _register_hooks(self):
        for name in self.STAGE_CHANNELS:
            module = dict(self.backbone.named_modules())[name]
            handle = module.register_forward_hook(
                lambda mod, inp, out, n=name: self._stage_feats.update({n: out})
            )
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
            key = name.replace(".", "_")
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
from src.dataset import AUITDDataset, TN5000Dataset
from src.trainer import train_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parent
TN5000_ROOT = ROOT / "data_raw" / "TN5000_forReview"
AUITD_ROOT = ROOT / "data_raw" / "auitd_dataset"
SEEDS = (0, 1, 2, 3, 4)
OUT = Path("outputs/final_model")
NUM_WORKERS = min(2, os.cpu_count() or 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ar_definition(ar: str) -> str:
    return {
        "A1": "Longest side 256; pad only as needed for a 224 crop; crop 224.",
        "A4": "Longest side 256; pad canvas to 256x256; crop 224.",
    }[ar]


def _geometry(ar: str, crop_cls):
    if ar == "A1":
        return [A.LongestMaxSize(max_size=256),
                A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
                crop_cls(224, 224)]
    if ar == "A4":
        return [A.LongestMaxSize(max_size=256),
                A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
                crop_cls(224, 224)]
    raise ValueError(f"Unsupported selected AR setting: {ar}")


def make_train_transform(ar: str):
    return A.Compose([
        A.Rotate(limit=15, p=1.0), A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        *_geometry(ar, A.RandomCrop),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(),
    ])


def make_val_transform(ar: str):
    return A.Compose([*_geometry(ar, A.CenterCrop),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


@torch.no_grad()
def seed_logits(checkpoint: Path, dataset, device, description, batch_size: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS if device.type == "cuda" else 0)
    model = MultiLevelSwin(dropout=0.0).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    result = {}
    for tensors, labels, ids in loader:
        b, views, c, h, w = tensors.shape
        logits = model(tensors.to(device).reshape(b * views, c, h, w)).squeeze(1).reshape(b, views).cpu().float().numpy()
        for i, item_id in enumerate(ids):
            result[item_id] = (int(labels[i]), logits[i])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_seed(ar, scales, seed, base_dir, sanity=False):
    seed_dir = base_dir / f"seed{seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {seed_dir}")
    seed_dir.mkdir(parents=True)
    print(f"\n{'=' * 70}\n[RUN] A4S1V2 | seed {seed}/4 | AR=A4 | TTA={scales}\n{'=' * 70}", flush=True)
    set_seed(seed)
    train_set = ConcatDataset([TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/train.txt"), make_train_transform(ar)),
                               AUITDDataset(str(AUITD_ROOT), make_train_transform(ar))])
    val_set = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/val.txt"), make_val_transform(ar))
    if sanity:
        train_set, val_set = Subset(train_set, range(32)), Subset(val_set, range(32))
    labels = np.concatenate([TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/train.txt")).get_labels(),
                              AUITDDataset(str(AUITD_ROOT)).get_labels()])
    config = {"lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
              "pos_weight": float((labels == 0).sum() / (labels == 1).sum()), "batch_size": 16,
              "max_epochs": 1 if sanity else 25, "patience": 10, "min_delta": 0.001,
              "T_0": 10, "T_mult": 2, "grad_clip_norm": 1.0, "label_smooth_eps": 0.05}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = NUM_WORKERS if device.type == "cuda" else 0
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model = MultiLevelSwin(dropout=0.3)
    started = time.time()
    history = train_model(model, train_loader, val_loader, config, str(seed_dir), "final", device)
    elapsed = time.time() - started
    generated = seed_dir / "final_best.pt"
    generated.rename(seed_dir / "best.pt")
    best_epoch = int(np.argmax(history["val_auc"]) + 1)
    metadata = {"config": "A4S1V2", "ar": "A4", "ar_definition": ar_definition(ar), "tta_scales": scales,
                "tta_views": len(scales), "seed": seed, "best_epoch": best_epoch,
                "best_val_auc": history["best_val_auc"], "training_time_sec": elapsed,
                "training_samples": len(train_set), "validation_samples": len(val_set),
                "checkpoint": str((seed_dir / "best.pt").resolve()), "model": "MultiLevelSwin / Swin-Tiny",
                "parameter_count": sum(p.numel() for p in model.parameters()), "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(), "device": str(device), "platform": platform.platform(), "sanity": sanity}
    (seed_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (seed_dir / "training_log.json").write_text(json.dumps(history), encoding="utf-8")
    print(f"[RUN COMPLETE] A4S1V2 | seed {seed}/4 | best epoch={best_epoch} | "
          f"best validation AUC={history['best_val_auc']:.4f}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run 1 epoch with 2 batches to verify pipeline.")
    args = parser.parse_args()
    print(f"{'=' * 70}")
    print(f"A4S1V2 Training — outputs/final_model/")
    print(f"{'=' * 70}", flush=True)

    ar = "A4"
    scales = [0.70, 0.85, 1.00, 1.15, 1.30]
    metadata = []
    for seed in SEEDS:
        seed_dir = OUT / f"seed{seed}"
        config_path = seed_dir / "config.json"
        checkpoint = seed_dir / "best.pt"
        if config_path.is_file() and checkpoint.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if not existing.get("sanity", False):
                print(f"[RESUME] A4S1V2 | seed {seed}/4 already complete; preserving it.", flush=True)
                metadata.append(existing)
                continue
        metadata.append(run_seed(ar, scales, seed, OUT, sanity=args.sanity_check))

    if not args.sanity_check:
        training_summary = {"config": "A4S1V2", "ar": "A4", "ar_definition": ar_definition(ar),
                            "tta_scales": scales, "tta_views": len(scales), "seeds": metadata,
                            "total_training_time_sec": sum(x["training_time_sec"] for x in metadata),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (OUT / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
        print(f"[TRAINING COMPLETE] A4S1V2. Summary saved to {OUT / 'training_summary.json'}", flush=True)


if __name__ == "__main__":
    main()