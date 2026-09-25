"""
Focal Loss Experiment Runner

Controlled experiment: MultiLevelSwin (same as main model) with Binary Focal Loss
instead of BCEWithLogitsLoss. Trains seed 0 only, evaluates on TN5000, Diveshzz,
and Thyroid for Pretraining.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from focal_loss_experiment.scripts.focal_loss import BinaryFocalLoss, test_focal_loss
from src.dataset import TN5000Dataset, AUITDDataset
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
import timm

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "focal_loss_experiment"
SEED0_DIR = EXP_DIR / "seed0"
RESULTS_DIR = EXP_DIR / "results"
LOG_DIR = EXP_DIR / "logs"

TN5000_ROOT = ROOT / "data_raw" / "TN5000_forReview"
AUITD_ROOT = ROOT / "data_raw" / "auitd_dataset"

NUM_WORKERS = min(2, os.cpu_count() or 1) if torch.cuda.is_available() else 0
EPOCHS = 25
BATCH_SIZE = 16
LR_HEAD = 3e-4
LABEL_SMOOTH_EPS = 0.05
GAMMA = 2.0
WARMUP_EPOCHS = 10
TTA_SCALES = [0.70, 0.85, 1.00, 1.15, 1.30]
THRESHOLD = 0.5912

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
HAS_TQDM = False
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class MultiLevelSwin(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.stage1_norm = nn.LayerNorm(192)
        self.stage1_proj = nn.Linear(192, 128)
        self.stage2_norm = nn.LayerNorm(384)
        self.stage2_proj = nn.Linear(384, 128)
        self.stage3_norm = nn.LayerNorm(768)
        self.stage3_proj = nn.Linear(768, 128)
        self.fusion_head = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )
        self._feature_cache = {}

    def _register_hooks(self):
        def _make_hook(name):
            def _hook(module, inputs, output):
                self._feature_cache[name] = output
            return _hook
        for name in ["layers.1", "layers.2", "layers.3"]:
            module = dict(self.backbone.named_modules())[name]
            module.register_forward_hook(_make_hook(name))

    def forward(self, x):
        self._feature_cache = {}
        self._register_hooks()
        _ = self.backbone(x)
        f1 = self.stage1_norm(self._feature_cache["layers.1"].mean(dim=(1, 2)))
        f1 = self.stage1_proj(f1)
        f2 = self.stage2_norm(self._feature_cache["layers.2"].mean(dim=(1, 2)))
        f2 = self.stage2_proj(f2)
        f3 = self.stage3_norm(self._feature_cache["layers.3"].mean(dim=(1, 2)))
        f3 = self.stage3_proj(f3)
        fused = torch.cat([f1, f2, f3], dim=-1)
        return self.fusion_head(fused)

    def freeze_epoch(self, epoch):
        if epoch >= 10:
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif epoch >= 6:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, "layers"):
                for param in self.backbone.layers[-1].parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def get_param_groups(self, lr_head, lr_backbone):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (list(self.stage1_norm.parameters()) + list(self.stage1_proj.parameters()) +
                       list(self.stage2_norm.parameters()) + list(self.stage2_proj.parameters()) +
                       list(self.stage3_norm.parameters()) + list(self.stage3_proj.parameters()) +
                       list(self.fusion_head.parameters()))
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups


def make_val_transform(scale: float = 1.0):
    max_size = round(256 * scale)
    return A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(min_height=max(max_size, 256), min_width=max(max_size, 256), border_mode=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(),
    ])


def make_train_transform():
    return A.Compose([
        A.Rotate(limit=15, p=1.0), A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=256), A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.CenterCrop(224, 224), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(),
    ])


def make_tta_transforms():
    norm_tensor = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    crop_coords = [
        (16, 16, 16 + 224, 16 + 224), (0, 0, 224, 224),
        (32, 0, 32 + 224, 224), (0, 32, 224, 32 + 224),
        (32, 32, 32 + 224, 32 + 224),
    ]
    transforms_list = []
    for scale in TTA_SCALES:
        max_size = round(256 * scale)
        for (x0, y0, x1, y1) in crop_coords:
            transforms_list.append(A.Compose([
                A.LongestMaxSize(max_size=max_size),
                A.PadIfNeeded(min_height=max(max_size, 256), min_width=max(max_size, 256), border_mode=0),
                A.Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1), *norm_tensor,
            ]))
            transforms_list.append(A.Compose([
                A.LongestMaxSize(max_size=max_size),
                A.PadIfNeeded(min_height=max(max_size, 256), min_width=max(max_size, 256), border_mode=0),
                A.Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1), A.HorizontalFlip(p=1.0), *norm_tensor,
            ]))
    return transforms_list


TTA_TRANSFORMS = make_tta_transforms()


class TN5000TestDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        split_file = self.data_root / "ImageSets" / "Main" / "test.txt"
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.samples = []
        for img_id in self.ids:
            ann_path = self.data_root / "Annotations" / f"{img_id}.xml"
            img_path = self.data_root / "JPEGImages" / f"{img_id}.jpg"
            label = _parse_xml_label(ann_path)
            self.samples.append({"id": img_id, "img_path": str(img_path), "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
        return tensors, torch.tensor(s["label"], dtype=torch.float32), s["id"]


class DiveshzzDataset(Dataset):
    def __init__(self, data_root: str):
        self.samples = []
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        for cls, label in [("0", 0), ("1", 1)]:
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.exists(cls_dir):
                for img_file in glob.glob(os.path.join(cls_dir, "*.*")):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append({"img_path": img_file, "label": label})
        for i, s in enumerate(self.samples):
            s["id"] = f"{i:04d}_{os.path.basename(s['img_path'])}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
        return tensors, torch.tensor(s["label"], dtype=torch.float32), s["id"]


class ThyroidPretrainingDataset(Dataset):
    def __init__(self, data_root: str):
        self.patients = defaultdict(list)
        dataset_dir = Path(data_root)
        for class_name in ["benign", "malignant"]:
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
            label = 0 if class_name == "benign" else 1
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, file)
                        pid = Path(img_path).stem.split('_')[0]
                        self.patients[pid].append({"img_path": img_path, "label": label})

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = list(self.patients.keys())[idx]
        images = []
        labels = []
        for item in self.patients[pid]:
            img = cv2.imread(item["img_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
            images.append(tensor)
            labels.append(item["label"])
        return torch.stack(images), torch.tensor(labels, dtype=torch.float32), pid


def _parse_xml_label(ann_path: Path) -> int:
    import xml.etree.ElementTree as ET
    tree = ET.parse(ann_path)
    root = tree.getroot()
    obj = root.find("object")
    return int(obj.find("name").text)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def get_pos_weight():
    tn_ds = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets" / "Main" / "train.txt"))
    au_ds = AUITDDataset(str(AUITD_ROOT))
    labels = np.concatenate([tn_ds.get_labels(), au_ds.get_labels()])
    return int((labels == 0).sum()) / int((labels == 1).sum())


def get_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95):
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_pred[idx]))
    scores.sort()
    return float(np.percentile(scores, (100 - ci) / 2)), float(np.percentile(scores, 100 - (100 - ci) / 2))


def compute_metrics(y_true, logits, threshold=THRESHOLD):
    probs = _expit(logits)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    ci_low, ci_high = get_bootstrap_ci(y_true, logits)
    return {
        "N": len(y_true), "Benign": int(np.sum(y_true == 0)), "Malignant": int(np.sum(y_true == 1)),
        "AUROC": roc_auc_score(y_true, logits), "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "PR-AUC": average_precision_score(y_true, probs), "Accuracy": accuracy_score(y_true, preds),
        "Sensitivity": recall_score(y_true, preds), "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "PPV": precision_score(y_true, preds, zero_division=0), "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "F1": f1_score(y_true, preds), "Balanced Accuracy": balanced_accuracy_score(y_true, preds),
        "MCC": matthews_corrcoef(y_true, preds), "Cohen's Kappa": cohen_kappa_score(y_true, preds),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def save_report(metrics, output_path: Path, title: str):
    report = f"# {title}\n\n"
    for k, v in metrics.items():
        if isinstance(v, float):
            report += f"**{k}:** {v:.4f}\n"
        else:
            report += f"**{k}:** {v}\n"
    output_path.write_text(report, encoding="utf-8")


def _train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler,
                     grad_clip, epoch, run_name):
    model.train()
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_labels = []
    t0 = time.time()
    iterator = tqdm(loader, desc=f"Training {run_name}", leave=False) if HAS_TQDM else loader
    for images, targets, _ in iterator:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda" if USE_AMP else "cpu", enabled=USE_AMP):
            logits = model(images).view(-1)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * targets.size(0)
        total_count += targets.size(0)
        all_logits.extend(logits.cpu().float().tolist())
        all_labels.extend(targets.cpu().tolist())
    scheduler.step()
    train_auc = roc_auc_score(np.array(all_labels), np.array(all_logits))
    return {"loss": total_loss / max(1, total_count), "auc": train_auc, "elapsed": time.time() - t0}


def _validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            with torch.amp.autocast("cuda" if USE_AMP else "cpu", enabled=USE_AMP):
                logits = model(images).view(-1)
                loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_count += targets.size(0)
            all_logits.extend(logits.cpu().float().tolist())
            all_labels.extend(targets.cpu().tolist())
    return {"loss": total_loss / max(1, total_count), "logits": np.array(all_logits), "labels": np.array(all_labels)}


def _save_checkpoint(path: Path, epoch: int, model: nn.Module, loss_fn: nn.Module,
                     optimizer: torch.optim.Optimizer, scheduler, scaler,
                     history: dict, config: dict, val_auc: float, best_val_auc: float):
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": val_auc,
        "best_val_auc": best_val_auc,
        "config": config, "history": history,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }, path)


def train_seed0():
    print("=" * 70)
    print("FOCAL LOSS EXPERIMENT - SEED 0 TRAINING")
    print("=" * 70)
    SEED0_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tn_ds = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets" / "Main" / "train.txt"), make_train_transform())
    au_ds = AUITDDataset(str(AUITD_ROOT), make_train_transform())
    print(f"TN5000 train samples: {len(tn_ds)}")
    print(f"AUITD train samples: {len(au_ds)}")
    train_ds = torch.utils.data.ConcatDataset([tn_ds, au_ds])
    print(f"Total train samples: {len(train_ds)}")

    pos_weight = get_pos_weight()
    print(f"Positive-class weight (n_benign/n_malignant): {pos_weight:.4f}")

    model = MultiLevelSwin(dropout=0.3).to(DEVICE)
    loss_fn = BinaryFocalLoss(gamma=GAMMA, pos_weight=pos_weight, label_smooth_eps=LABEL_SMOOTH_EPS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.get_param_groups(LR_HEAD, LR_HEAD * 0.1), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=WARMUP_EPOCHS, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_AMP)

    last_ckpt = SEED0_DIR / "focal_last.pt"
    best_ckpt = SEED0_DIR / "focal_best.pt"
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}
    start_epoch = 1
    best_val_auc = -1e9
    patience = 10
    counter = 0
    _prev_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    if last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history = ckpt.get("history", history)
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", -1e9)
        print(f"Resuming from epoch {start_epoch}")

    config = {
        "model": "MultiLevelSwin (same as main model)", "loss": "BinaryFocalLoss", "gamma": GAMMA,
        "pos_weight": pos_weight, "label_smooth_eps": LABEL_SMOOTH_EPS, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "lr_head": LR_HEAD, "weight_decay": 1e-4, "warmup_epochs": WARMUP_EPOCHS,
        "scheduler": "CosineAnnealingWarmRestarts", "T_0": WARMUP_EPOCHS, "T_mult": 2,
        "grad_clip": 1.0, "patience": patience, "min_delta": 0.001,
        "tta_scales": TTA_SCALES, "threshold": THRESHOLD, "seed": 0, "optimizer": "AdamW",
    }

    for epoch in range(start_epoch, EPOCHS + 1):
        model.freeze_epoch(epoch)
        _curr_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        if _curr_trainable != _prev_trainable:
            print(f"  [unfreeze] Trainable params {_prev_trainable}→{_curr_trainable}. Rebuilding optimizer.")
            optimizer = torch.optim.AdamW(model.get_param_groups(LR_HEAD, LR_HEAD * 0.1), weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=WARMUP_EPOCHS, T_mult=2)
            _prev_trainable = _curr_trainable
        train_metrics = _train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, 1.0, epoch, "seed0")
        val_ds = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets" / "Main" / "val.txt"), make_val_transform())
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        val_result = _validate(model, val_loader, loss_fn)
        val_auc = roc_auc_score(val_result["labels"], val_result["logits"])
        history["train_loss"].append(train_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(val_result["loss"])
        history["val_auc"].append(val_auc)

        print(f"[seed0] Epoch {epoch:03d}/{EPOCHS} | Train Loss={train_metrics['loss']:.4f} | "
              f"Val Loss={val_result['loss']:.4f} | Val AUC={val_auc:.4f} | {train_metrics['elapsed'] / 60:.1f}min")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            _save_checkpoint(best_ckpt, epoch, model, loss_fn, optimizer, scheduler, scaler, history, config, best_val_auc, best_val_auc)
            print(f"  New best checkpoint saved.")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val AUC = {best_val_auc:.4f}")
                break
        _save_checkpoint(last_ckpt, epoch, model, loss_fn, optimizer, scheduler, scaler, history, config, val_auc, best_val_auc)

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_auc = ckpt["val_auc"]

    summary = {"best_val_auc": best_val_auc, "epochs_trained": len(history["val_auc"]), "history": history, "config": config}
    with open(SEED0_DIR / "focal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Training complete. Best val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint: {best_ckpt}")
    print("=" * 70)
    return summary


def evaluate_tn5000(checkpoint_path: str):
    print("=" * 70)
    print("EVALUATING ON TN5000")
    print("=" * 70)
    model = MultiLevelSwin(dropout=0.0).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ds = TN5000TestDataset(str(TN5000_ROOT))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=NUM_WORKERS)
    all_preds = defaultdict(lambda: {"label": None, "logits": [[] for _ in TTA_SCALES]})
    with torch.no_grad():
        for tensors, labels, ids in loader:
            tensors = tensors.to(DEVICE)
            B, num_tta, C, H, W = tensors.shape
            tensors = tensors.view(B * num_tta, C, H, W)
            logits = model(tensors).squeeze(-1).view(B, num_tta).cpu().float().numpy()
            for i in range(B):
                obj_id = ids[i]
                all_preds[obj_id]["label"] = int(labels[i])
                for s_idx in range(num_tta):
                    all_preds[obj_id]["logits"][s_idx].append(logits[i, s_idx])

    final_preds = {}
    for obj_id, data in all_preds.items():
        final_preds[obj_id] = {
            "label": data["label"],
            "scale_logits": [np.mean(lst) for lst in data["logits"]],
        }

    ids = sorted(list(final_preds.keys()))
    y_true = np.array([final_preds[i]["label"] for i in ids])
    seed_tta_logits = []
    for s_idx in range(5):
        s_logits = [final_preds[i]["scale_logits"][s_idx] for i in ids]
        seed_tta_logits.append(np.array(s_logits))
    ensemble_logits = np.mean(seed_tta_logits, axis=0)
    metrics = compute_metrics(y_true, ensemble_logits)
    save_report(metrics, RESULTS_DIR / "tn5000_focal_report.md", "TN5000 Evaluation (Focal Loss)")
    print(f"TN5000 AUROC: {metrics['AUROC']:.4f}")
    return metrics


def _run_sanity():
    print("=" * 70)
    print("FOCAL LOSS EXPERIMENT — SANITY CHECKS")
    print("=" * 70)

    # 1. Focal loss unit tests
    print("\n[1/6] Running focal loss unit tests...")
    ok = test_focal_loss()
    if not ok:
        print("Focal loss unit tests FAILED — aborting.")
        raise SystemExit(1)

    # 2. Model initialization
    print("\n[2/6] Model initialization...")
    model = MultiLevelSwin(dropout=0.3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,} | Trainable (epoch 1): {n_trainable:,}")
    x = torch.randn(2, 3, 224, 224).to(DEVICE)
    out = model(x)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  Forward output shape: {out.shape} [OK]")

    # 3. Freeze schedule
    print("\n[3/6] Freeze schedule...")
    for ep in [1, 5, 6, 9, 10, 25]:
        model.freeze_epoch(ep)
        n_tr = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Epoch {ep}: trainable params={n_tr}")

    # 4. Dataset loading
    print("\n[4/6] Dataset loading...")
    train_t = make_train_transform()
    tn_ds = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets" / "Main" / "train.txt"), train_t)
    au_ds = AUITDDataset(str(AUITD_ROOT), train_t)
    print(f"  TN5000 train: {len(tn_ds)} | AUITD: {len(au_ds)}")
    sample = tn_ds[0]
    assert sample[0].shape[0] == 3, f"Expected 3-channel image, got {sample[0].shape}"
    print(f"  Sample image shape: {sample[0].shape} [OK]")
    val_ds = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets" / "Main" / "val.txt"), make_val_transform())
    print(f"  TN5000 val: {len(val_ds)}")
    test_ds = TN5000TestDataset(str(TN5000_ROOT))
    print(f"  TN5000 test: {len(test_ds)}")

    # 5. TTA transforms
    print("\n[5/6] TTA transforms...")
    print(f"  TTA scales: {TTA_SCALES}")
    print(f"  Total TTA transforms: {len(TTA_TRANSFORMS)}")
    sample_id = test_ds[0][2]  # __getitem__ returns (tensors, label, id)
    img = cv2.imread(str(TN5000_ROOT / "JPEGImages" / f"{sample_id}.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensors = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
    assert tensors.shape[1:] == (3, 224, 224), f"Unexpected shape: {tensors.shape}"
    print(f"  TTA tensor shape: {tensors.shape} [OK]")

    # 6. External datasets
    print("\n[6/6] External datasets...")
    divesh_path = ROOT / "data_raw" / "divesh"
    if divesh_path.exists():
        divesh_ds = DiveshzzDataset(str(divesh_path))
        print(f"  Diveshzz: {len(divesh_ds)} samples")
    else:
        print(f"  Diveshzz: not found at {divesh_path}")

    thyroid_path = ROOT / "data_raw" / "Thyroid_for_Pretraining"
    if thyroid_path.exists():
        thyroid_ds = ThyroidPretrainingDataset(str(thyroid_path))
        print(f"  Thyroid for Pretraining: {len(thyroid_ds)} patients")
    else:
        print(f"  Thyroid for Pretraining: not found at {thyroid_path}")

    # 7. Pos weight
    print("\n[7/7] Pos weight...")
    pw = get_pos_weight()
    print(f"  pos_weight = {pw:.4f}")

    print("\n" + "=" * 70)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sanity", "train", "eval", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    if args.command == "sanity":
        _run_sanity()
    elif args.command == "train":
        train_seed0()
    elif args.command == "eval":
        ckpt = args.checkpoint or str(SEED0_DIR / "focal_best.pt")
        evaluate_tn5000(ckpt)
    elif args.command == "all":
        train_seed0()
        evaluate_tn5000(str(SEED0_DIR / "focal_best.pt"))
