#!/usr/bin/env python3
"""
SSL Pretraining Script for Thyroid Ultrasound
Multi-view contrastive SSL pretraining using Swin-Tiny.
"""

import random
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import timm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.transforms import IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SSL_EXPERIMENT_ROOT = PROJECT_ROOT / "ssl_pretraining_experiment"
SSL_DATA_ROOT = PROJECT_ROOT / "data_raw"
TN5000_ROOT = SSL_DATA_ROOT / "TN5000_forReview"
AUITD_ROOT = SSL_DATA_ROOT / "auitd_dataset"

SSL_BATCH_SIZE = 16
SSL_GRADIENT_ACCUMULATION_STEPS = 4
SSL_EPOCHS = 30
SSL_LR = 1e-4
SSL_WEIGHT_DECAY = 1e-4
SSL_TEMPERATURE = 0.2
SSL_SEED = 0

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SSLDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.image_paths = []
        self._collect_images()

    def _collect_images(self):
        tn5000_train_txt = TN5000_ROOT / "ImageSets" / "Main" / "train.txt"
        if tn5000_train_txt.exists():
            with open(tn5000_train_txt, "r") as f:
                ids = [line.strip() for line in f if line.strip()]
            for img_id in ids:
                img_path = TN5000_ROOT / "JPEGImages" / f"{img_id}.jpg"
                if img_path.exists():
                    self.image_paths.append(str(img_path))

        auitd_train = AUITD_ROOT / "dataset thyroid" / "train"
        if auitd_train.exists():
            for class_name in os.listdir(auitd_train):
                class_dir = auitd_train / class_name
                if not class_dir.is_dir():
                    continue
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_np = np.array(img)
        view1 = self.transform(image=img_np)["image"]
        view2 = self.transform(image=img_np)["image"]
        return view1, view2, self.image_paths[idx]

def make_ssl_transform():
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=0.8),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(224, 224),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

class SSLProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)

class SSLModel(nn.Module):
    def __init__(self, backbone, projection_head, temperature=0.2):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.temperature = temperature

    def forward(self, view1, view2):
        feat1 = self.backbone(view1)
        feat2 = self.backbone(view2)
        proj1 = self.projection_head(feat1)
        proj2 = self.projection_head(feat2)
        return proj1, proj2

def contrastive_loss(proj1, proj2, temperature):
    """Numerically stable NT-Xent contrastive loss."""
    bsz = proj1.shape[0]
    z = torch.cat([proj1, proj2], dim=0)  # 2N x D (already normalized by projection head)
    sim = torch.mm(z, z.T) / temperature  # 2N x 2N
    logits = torch.log_softmax(sim, dim=1)  # numerically stable log-probabilities

    pos_mask = torch.zeros(2 * bsz, 2 * bsz, device=z.device)
    for i in range(bsz):
        pos_mask[i, bsz + i] = 1.0
        pos_mask[bsz + i, i] = 1.0
    pos_logits = (pos_mask * logits).sum(dim=1)
    loss = -pos_logits.mean()

    with torch.no_grad():
        cross_sim = torch.mm(proj1, proj2.T) / temperature
        mean_pos = cross_sim.diag().mean().item()
        full_sim = torch.mm(z, z.T) / temperature
        neg_mask = torch.ones_like(full_sim, dtype=torch.bool)
        neg_mask.fill_diagonal_(False)
        for i in range(bsz):
            neg_mask[i, bsz + i] = False
            neg_mask[bsz + i, i] = False
        mean_neg = full_sim[neg_mask].mean().item()
    return loss, mean_pos, mean_neg

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    model.backbone.load_state_dict(ckpt["model_state_dict"])
    model.projection_head.load_state_dict(ckpt["projection_head_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return start_epoch, ckpt.get("history", {})

def run_ssl(epochs=None, batch_size=None, gradient_accumulation_steps=None, max_samples=None):
    epochs = SSL_EPOCHS if epochs is None else epochs
    batch_size = SSL_BATCH_SIZE if batch_size is None else batch_size
    gradient_accumulation_steps = (
        SSL_GRADIENT_ACCUMULATION_STEPS
        if gradient_accumulation_steps is None
        else gradient_accumulation_steps
    )
    set_seed(SSL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    SSL_EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    ssl_pretrain_dir = SSL_EXPERIMENT_ROOT / "ssl_pretrain"
    ssl_pretrain_dir.mkdir(parents=True, exist_ok=True)

    transform = make_ssl_transform()
    dataset = SSLDataset(transform)
    if max_samples is not None and 0 < max_samples < len(dataset):
        dataset.image_paths = dataset.image_paths[:max_samples]
    print(f"SSL dataset size: {len(dataset)}", flush=True)

    if len(dataset) == 0:
        print("ERROR: SSL dataset is empty!", flush=True)
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
    for param in backbone.parameters():
        param.requires_grad = True
    projection_head = SSLProjectionHead()

    model = SSLModel(backbone, projection_head, SSL_TEMPERATURE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=SSL_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 1
    history = {"train_loss": [], "train_pos_sim": [], "train_neg_sim": [], "lr": []}
    last_ckpt_path = ssl_pretrain_dir / "last_ssl.pt"
    if last_ckpt_path.exists():
        try:
            start_epoch, history = load_checkpoint(
                last_ckpt_path, model, optimizer, scheduler, scaler
            )
            print(f"[RESUME] Resuming from epoch {start_epoch}", flush=True)
        except Exception as e:
            print(f"[RESUME FAILED] {e}, starting fresh", flush=True)

    best_loss = float("inf")
    total_accumulation_steps = gradient_accumulation_steps

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_pos = 0.0
        running_neg = 0.0
        optimizer.zero_grad()

        for batch_idx, (view1, view2, _) in enumerate(loader):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            if device.type == "cuda" and scaler is not None:
                with torch.cuda.amp.autocast():
                    proj1, proj2 = model(view1, view2)
                    loss, mean_pos, mean_neg = contrastive_loss(proj1, proj2, SSL_TEMPERATURE)
                    loss = loss / total_accumulation_steps
                scaler.scale(loss).backward()
            else:
                proj1, proj2 = model(view1, view2)
                loss, mean_pos, mean_neg = contrastive_loss(proj1, proj2, SSL_TEMPERATURE)
                loss = loss / total_accumulation_steps
                loss.backward()

            running_loss += loss.item() * total_accumulation_steps
            running_pos += mean_pos
            running_neg += mean_neg

            if (batch_idx + 1) % total_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                if device.type == "cuda" and scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        elapsed = time.time() - epoch_start
        avg_loss = running_loss / len(loader)
        avg_pos = running_pos / len(loader)
        avg_neg = running_neg / len(loader)

        history["train_loss"].append(avg_loss)
        history["train_pos_sim"].append(avg_pos)
        history["train_neg_sim"].append(avg_neg)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch - 1,
                "model_state_dict": model.backbone.state_dict(),
                "projection_head_state_dict": model.projection_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "history": history,
                "config": {
                    "lr": SSL_LR, "weight_decay": SSL_WEIGHT_DECAY,
                    "temperature": SSL_TEMPERATURE, "epochs": epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": total_accumulation_steps,
                    "seed": SSL_SEED,
                    "backbone": "swin_tiny_patch4_window7_224",
                },
            }, ssl_pretrain_dir / "best_ssl.pt")

        torch.save({
            "epoch": epoch - 1,
            "model_state_dict": model.backbone.state_dict(),
            "projection_head_state_dict": model.projection_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "history": history,
            "config": {
                "lr": SSL_LR, "weight_decay": SSL_WEIGHT_DECAY,
                "temperature": SSL_TEMPERATURE, "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": total_accumulation_steps,
                "seed": SSL_SEED,
            },
        }, ssl_pretrain_dir / "last_ssl.pt")

        print(f"SSL Epoch {epoch:03d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | PosSim: {avg_pos:.4f} | "
              f"NegSim: {avg_neg:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s", flush=True)

    torch.save(model.backbone.state_dict(), ssl_pretrain_dir / "backbone.pt")
    torch.save(model.projection_head.state_dict(), ssl_pretrain_dir / "projection_head.pt")
    with open(ssl_pretrain_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(ssl_pretrain_dir / "config.json", "w") as f:
        json.dump({
            "lr": SSL_LR, "weight_decay": SSL_WEIGHT_DECAY,
            "temperature": SSL_TEMPERATURE, "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": total_accumulation_steps,
            "seed": SSL_SEED, "backbone": "swin_tiny_patch4_window7_224",
        }, f, indent=2)

    print(f"\nSSL pretraining complete. Best loss: {best_loss:.4f}")
    print(f"Backbone saved to {ssl_pretrain_dir / 'backbone.pt'}")
    print(f"Training log: {ssl_pretrain_dir / 'training_log.json'}")
    print(f"Config: {ssl_pretrain_dir / 'config.json'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SSL Pretraining for Thyroid Ultrasound")
    parser.add_argument("--epochs", type=int, default=SSL_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=SSL_BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=SSL_GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to use (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()
    run_ssl(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_samples=args.max_samples,
    )