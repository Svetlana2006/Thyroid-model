#!/usr/bin/env python3
"""
Supervised Training Script for Thyroid Model with SSL-Pretrained Backbone
"""

import os
import json
import time
import random
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import timm
import xml.etree.ElementTree as ET
from sklearn.metrics import roc_auc_score
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.transforms import IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SSL_EXPERIMENT_ROOT = PROJECT_ROOT / "ssl_pretraining_experiment"
SSL_DATA_ROOT = PROJECT_ROOT / "data_raw"
TN5000_ROOT = SSL_DATA_ROOT / "TN5000_forReview"
AUITD_ROOT = SSL_DATA_ROOT / "auitd_dataset"
SSL_BACKBONE_PATH = SSL_EXPERIMENT_ROOT / "ssl_pretrain" / "best_ssl.pt"

SUPERVISED_LR_HEAD = 3e-4
SUPERVISED_LR_BACKBONE = 3e-5
SUPERVISED_WEIGHT_DECAY = 1e-4
SUPERVISED_BATCH_SIZE = 16
SUPERVISED_MAX_EPOCHS = 25
SUPERVISED_PATIENCE = 10
SUPERVISED_MIN_DELTA = 0.001
SUPERVISED_T_0 = 10
SUPERVISED_T_MULT = 2
SUPERVISED_GRAD_CLIP = 1.0
SUPERVISED_LABEL_SMOOTH = 0.05

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TN5000Dataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.transform = transform
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.samples = []
        for img_id in self.ids:
            img_path = TN5000_ROOT / "JPEGImages" / f"{img_id}.jpg"
            ann_path = TN5000_ROOT / "Annotations" / f"{img_id}.xml"
            label = self._parse_label(ann_path)
            self.samples.append((str(img_path), label))

    def _parse_label(self, ann_path):
        if not ann_path.exists():
            return 0
        tree = ET.parse(ann_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is not None:
            return int(obj.find("name").text)
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"], torch.tensor(label, dtype=torch.float32)

class AUITDDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        dataset_dir = AUITD_ROOT / "dataset thyroid" / "train"
        if dataset_dir.exists():
            for class_name in os.listdir(dataset_dir):
                class_dir = dataset_dir / class_name
                if not class_dir.is_dir():
                    continue
                label = 0 if "benign" in class_name.lower() else 1
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.samples.append((os.path.join(root, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"], torch.tensor(label, dtype=torch.float32)

def make_train_transform():
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(224, 224),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def make_val_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}
    def __init__(self, backbone_state_dict=None, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        if backbone_state_dict:
            self.backbone.load_state_dict(backbone_state_dict)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.stage_norms = nn.ModuleDict()
        self.stage_projs = nn.ModuleDict()
        for name, ch in self.STAGE_CHANNELS.items():
            key = name.replace(".", "_")
            self.stage_norms[key] = nn.LayerNorm(ch)
            self.stage_projs[key] = nn.Linear(ch, 128, bias=False)
        self.fusion_head = nn.Sequential(
            nn.Linear(128 * len(self.STAGE_CHANNELS), 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
        self._stage_feats = {}
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

    def forward(self, x):
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

    def freeze_epoch(self, epoch):
        if epoch >= 10:
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif epoch >= 6:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if hasattr(self.backbone, "layers"):
                for p in self.backbone.layers[-1].parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def get_param_groups(self, lr_head, lr_backbone):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.stage_norms.parameters()) + list(self.stage_projs.parameters()) + list(self.fusion_head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups

def train_epoch(model, loader, optimizer, scaler, device, pos_weight, label_smooth_eps, grad_clip):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images).squeeze(1)
                targets = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            targets = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset) if len(loader.dataset) else 0.0

def evaluate(model, loader, device, pos_weight, label_smooth_eps, scaler):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images).squeeze(1)
                    targets = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            else:
                logits = model(images).squeeze(1)
                targets = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            total_loss += loss.item() * images.size(0)
            all_logits.extend(logits.cpu().float().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) else 0.0
    return {"loss": avg_loss, "auc": roc_auc_score(all_labels, all_logits)}

def get_pos_weights():
    train_set1_labels = []
    train_set2_labels = []
    with open(TN5000_ROOT / "ImageSets" / "Main" / "train.txt", "r") as f:
        for img_id in [l.strip() for l in f if l.strip()]:
            ann_path = TN5000_ROOT / "Annotations" / f"{img_id}.xml"
            if ann_path.exists():
                tree = ET.parse(ann_path)
                obj = tree.getroot().find("object")
                if obj is not None:
                    train_set1_labels.append(int(obj.find("name").text))
    for class_name in os.listdir(AUITD_ROOT / "dataset thyroid" / "train"):
        class_dir = AUITD_ROOT / "dataset thyroid" / "train" / class_name
        if not class_dir.is_dir():
            continue
        label = 0 if "benign" in class_name.lower() else 1
        for root, _, files in os.walk(class_dir):
            train_set2_labels.extend([label] * len([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]))
    all_labels = np.array(train_set1_labels + train_set2_labels)
    return float((all_labels == 0).sum() / (all_labels == 1).sum()) if len(all_labels) > 0 else 1.0

def run_supervised():
    for seed in range(5):
        os.makedirs(SSL_EXPERIMENT_ROOT / f"supervised" / f"seed{seed}", exist_ok=True)

    ssl_ckpt_path = SSL_BACKBONE_PATH
    if ssl_ckpt_path.exists():
        ssl_ckpt = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
        backbone_state_dict = ssl_ckpt.get("backbone_state_dict", ssl_ckpt.get("model_state_dict", {}))
    else:
        print(f"ERROR: SSL backbone not found at {ssl_ckpt_path}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading SSL backbone from {ssl_ckpt_path}", flush=True)

    for seed in range(5):
        print(f"\n{'='*70}", flush=True)
        print(f"Supervised Seed {seed}", flush=True)
        print(f"{'='*70}", flush=True)
        set_seed(seed)

        train_transform = make_train_transform()
        val_transform = make_val_transform()

        train_set1 = TN5000Dataset(TN5000_ROOT / "ImageSets" / "Main" / "train.txt", train_transform)
        train_set2 = AUITDDataset(train_transform)
        train_set = ConcatDataset([train_set1, train_set2])

        val_set = TN5000Dataset(TN5000_ROOT / "ImageSets" / "Main" / "val.txt", val_transform)

        pos_weight = get_pos_weights()

        config = {
            "lr_head": SUPERVISED_LR_HEAD, "lr_backbone": SUPERVISED_LR_BACKBONE,
            "weight_decay": SUPERVISED_WEIGHT_DECAY, "pos_weight": pos_weight,
            "batch_size": SUPERVISED_BATCH_SIZE, "max_epochs": SUPERVISED_MAX_EPOCHS,
            "patience": SUPERVISED_PATIENCE, "min_delta": SUPERVISED_MIN_DELTA,
            "T_0": SUPERVISED_T_0, "T_mult": SUPERVISED_T_MULT,
            "grad_clip": SUPERVISED_GRAD_CLIP, "label_smooth": SUPERVISED_LABEL_SMOOTH,
        }

        workers = min(2, os.cpu_count() or 1) if device.type == "cuda" else 0
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=workers, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"))

        model = MultiLevelSwin(backbone_state_dict=backbone_state_dict, dropout=0.3).to(device)
        optimizer = optim.AdamW(model.get_param_groups(config["lr_head"], config["lr_backbone"]), weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["T_0"], T_mult=config["T_mult"])
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        pos_weight_tensor = torch.tensor([config["pos_weight"]], dtype=torch.float32, device=device)

        best_auc = 0.0
        patience_counter = 0
        history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

        for epoch in range(1, config["max_epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, pos_weight_tensor, config["label_smooth"], config["grad_clip"])
            val_metrics = evaluate(model, val_loader, device, pos_weight_tensor, config["label_smooth"], scaler)
            scheduler.step()
            history["train_loss"].append(train_loss)
            history["train_auc"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_auc"].append(val_metrics["auc"])

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                patience_counter = 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "history": history}, SSL_EXPERIMENT_ROOT / f"supervised" / f"seed{seed}" / "best.pt")
            else:
                patience_counter += 1
            print(f"Seed {seed} Epoch {epoch:03d}/{config['max_epochs']} | Train Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f}", flush=True)
            if patience_counter >= config["patience"]:
                print(f"Seed {seed} early stopping at epoch {epoch}", flush=True)
                break

        epoch += 1
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "history": history}, SSL_EXPERIMENT_ROOT / f"supervised" / f"seed{seed}" / "final_last.pt")
        with open(SSL_EXPERIMENT_ROOT / f"supervised" / f"seed{seed}" / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"Seed {seed} complete. Best AUC: {best_auc:.4f}", flush=True)

    print("\n" + "="*70 + "\nSUPERVISED TRAINING COMPLETE\n" + "="*70, flush=True)

if __name__ == "__main__":
    run_supervised()