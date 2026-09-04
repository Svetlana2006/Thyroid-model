"""
Experiment 16: Multi-Scale Aspect-Ratio-Preserving Swin Inference

M0: AR baseline (control, reproduces Exp14/Exp15-M0)
M1: AR + stochastic multi-scale training (0.85x/1.0x/1.15x, random per sample)
M2: AR baseline training + multi-scale TTA at inference
M3: Multi-scale training + multi-scale TTA

All models use Swin-Tiny, AR-preserving preprocessing, seed 0.
Divesh is held out entirely.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import TN5000Dataset, AUITDDataset
from src.metrics import compute_metrics, youden_threshold
from src.models import build_model
from src.trainer import EarlyStopping, build_optimizer_and_scheduler, evaluate
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR   = Path("experiments/16_multiscale_ar")
DATA_ROOT = Path("data_raw/TN5000_forReview")
AUITD_ROOT = "data_raw/auitd_dataset"
TRAIN_TXT = str(DATA_ROOT / "ImageSets/Main/train.txt")
VAL_TXT   = str(DATA_ROOT / "ImageSets/Main/val.txt")
TEST_TXT  = str(DATA_ROOT / "ImageSets/Main/test.txt")

for d in ["checkpoints", "logs", "metrics", "plots"]:
    (EXP_DIR / d).mkdir(parents=True, exist_ok=True)

SEED       = 0
BATCH_SIZE = 16
SCALES     = [0.85, 1.00, 1.15]   # fixed, not tuned on Divesh
TARGET_RES = 256   # intermediate resize target before crop
CROP_SIZE  = 224   # model input size
_USE_GPU   = torch.cuda.is_available()
_N_WORK    = 4 if _USE_GPU else 0


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)


# ── Transforms ────────────────────────────────────────────────────────────────
def _make_val_transform(scale: float = 1.0):
    """
    AR-preserving val/test transform at a given scale factor.
    scale < 1.0 → image appears smaller within the frame (zoom out)
    scale > 1.0 → image is larger, cropping removes some outer context (zoom in)
    """
    max_size = round(TARGET_RES * scale)
    return A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(min_height=max(max_size, CROP_SIZE),
                      min_width=max(max_size, CROP_SIZE),
                      border_mode=0),
        A.CenterCrop(CROP_SIZE, CROP_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def make_baseline_train_transform():
    """Standard AR-preserving training transform (1.0x scale)."""
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.LongestMaxSize(max_size=TARGET_RES),
        A.PadIfNeeded(min_height=TARGET_RES, min_width=TARGET_RES, border_mode=0),
        A.RandomCrop(CROP_SIZE, CROP_SIZE),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class MultiScaleDataset(Dataset):
    """
    Wraps TN5000Dataset or AUITDDataset to randomly pick one of the given
    scale transforms per __getitem__ call.  Used for M1 / M3 training.
    """
    def __init__(self, base_dataset, scale_transforms):
        self.base = base_dataset
        self.scale_transforms = scale_transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base returns (tensor, label) if transform already set
        # We bypass the base transform and apply our own randomly
        sample = self.base.samples[idx]
        img = np.array(PILImage.open(sample["img_path"]).convert("RGB"))
        label = sample["label"]
        t = random.choice(self.scale_transforms)
        img = t(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.float32)


class DiveshDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        self.samples = []
        td = os.path.join(data_root, "Thyroid Data")
        for label, sub in [(0, "0"), (1, "1")]:
            d = os.path.join(td, sub)
            if not os.path.exists(d):
                continue
            for f in glob.glob(os.path.join(d, "*.*")):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({"img_path": f, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(PILImage.open(s["img_path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(s["label"], dtype=torch.float32)


# ── Multi-scale training augmentation helpers ─────────────────────────────────
def make_scale_train_transforms():
    """Return three AR-preserving train transforms, one per scale."""
    transforms = []
    for scale in SCALES:
        max_size = round(TARGET_RES * scale)
        t = A.Compose([
            A.Rotate(limit=15, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max(max_size, CROP_SIZE),
                          min_width=max(max_size, CROP_SIZE),
                          border_mode=0),
            A.RandomCrop(CROP_SIZE, CROP_SIZE),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        transforms.append(t)
    return transforms


# ── Training ──────────────────────────────────────────────────────────────────
def train_model_standard(model, train_loader, val_loader, config, run_name, device):
    """Standard training loop with staged unfreezing (mirrors src.trainer)."""
    from src.trainer import train_one_epoch
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    model = model.to(device)
    lr_head = config["lr_head"]
    lr_backbone = lr_head * 0.1
    pos_weight = config["pos_weight"]
    max_epochs = config["max_epochs"]

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr_head, lr_backbone, config["weight_decay"],
        config["T_0"], config["T_mult"], -1
    )
    early_stopping = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    _prev_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    for epoch in range(1, max_epochs + 1):
        model.freeze_epoch(epoch)
        _curr = sum(1 for p in model.parameters() if p.requires_grad)
        if _curr != _prev_trainable:
            print(f"  [unfreeze] {_prev_trainable}→{_curr} trainable params. Rebuilding optimizer.")
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, lr_head, lr_backbone, config["weight_decay"],
                config["T_0"], config["T_mult"], last_epoch=epoch - 1
            )
            _prev_trainable = _curr

        scaler = None
        if _USE_GPU:
            from torch.amp import GradScaler
            scaler = GradScaler("cuda")

        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            pos_weight, config.get("pos_weight_scale", 1.0),
            config.get("label_smooth_eps", 0.05),
            config.get("grad_clip_norm", 1.0),
            epoch=epoch, run_name=run_name
        )
        scheduler.step()
        val_m = evaluate(model, val_loader, device, pos_weight, 1.0)

        history["train_loss"].append(train_m["loss"])
        history["train_auc"].append(train_m["auc"])
        history["val_loss"].append(val_m["loss"])
        history["val_auc"].append(val_m["auc"])

        print(f"[{run_name}] Epoch {epoch:02d} | Train AUC {train_m['auc']:.4f} | Val AUC {val_m['auc']:.4f}")

        if early_stopping(val_m["auc"], model):
            print(f"[{run_name}] Early stop at epoch {epoch}.")
            break
        if early_stopping.counter == 0:
            torch.save(model.state_dict(), EXP_DIR / "checkpoints" / f"{run_name}_best.pt")

    early_stopping.restore_best(model)
    history["best_val_auc"] = early_stopping.best_score
    return history


# ── Inference helpers ─────────────────────────────────────────────────────────
def _infer_logits(model, loader, device):
    """Return logits and labels as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_logits), np.array(all_labels)


def tta_infer(model, dataset_cls, data_root_or_samples, device,
              scale_transforms, batch_size=32, is_divesh=False):
    """
    Run inference at each scale and average logits.
    Returns dict: {scale: logits}, averaged logits, labels.
    """
    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU)
    all_scale_logits = {}
    labels = None

    for scale, t in zip(SCALES, scale_transforms):
        if is_divesh:
            ds = DiveshDataset(data_root_or_samples, transform=t)
        else:
            ds = dataset_cls(str(DATA_ROOT), data_root_or_samples, transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **kw)
        logits, lbl = _infer_logits(model, loader, device)
        all_scale_logits[f"{scale:.2f}x"] = logits
        if labels is None:
            labels = lbl

    avg_logits = np.mean(list(all_scale_logits.values()), axis=0)
    return all_scale_logits, avg_logits, labels


def scale_aucs(scale_logits, labels):
    return {k: float(roc_auc_score(labels, v)) for k, v in scale_logits.items()}

def pairwise_corr(scale_logits):
    keys = list(scale_logits.keys())
    result = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            r = float(np.corrcoef(scale_logits[k1], scale_logits[k2])[0, 1])
            result[f"{k1}_vs_{k2}"] = round(r, 4)
    return result


# ── Single experiment run ─────────────────────────────────────────────────────
def run_single(run_name, use_multiscale_train, use_tta, divesh_root, device, dry_run=False):
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")
    set_seed(SEED)

    val_transform_1x   = _make_val_transform(1.00)
    scale_val_transforms = [_make_val_transform(s) for s in SCALES]

    # Build training dataset
    if use_multiscale_train:
        scale_train_transforms = make_scale_train_transforms()
        train_tn_base = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=None)
        train_au_base = AUITDDataset(AUITD_ROOT, transform=None)
        train_tn = MultiScaleDataset(train_tn_base, scale_train_transforms)
        train_au = MultiScaleDataset(train_au_base, scale_train_transforms)
    else:
        train_t = make_baseline_train_transform()
        train_tn = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t)
        train_au = AUITDDataset(AUITD_ROOT, transform=train_t)

    train_ds = torch.utils.data.ConcatDataset([train_tn, train_au])
    val_ds   = TN5000Dataset(str(DATA_ROOT), VAL_TXT,  transform=val_transform_1x)

    all_labels = np.concatenate([
        TN5000Dataset(str(DATA_ROOT), TRAIN_TXT).get_labels(),
        AUITDDataset(AUITD_ROOT).get_labels()
    ])
    pos_weight = float((all_labels == 0).sum() / (all_labels == 1).sum())

    config = {
        "lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
        "pos_weight": pos_weight, "pos_weight_scale": 1.0, "batch_size": BATCH_SIZE,
        "max_epochs": 25, "patience": 10, "min_delta": 0.001,
        "T_0": 10, "T_mult": 2, "grad_clip_norm": 1.0, "label_smooth_eps": 0.05,
    }

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Multi-scale train: {use_multiscale_train}  TTA: {use_tta}")

    if dry_run:
        print("  [DRY RUN] skipping training.")
        return None

    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)

    model = build_model("swin_tiny", dropout=config["dropout"])
    history = train_model_standard(model, train_loader, val_loader, config, run_name, device)

    with open(EXP_DIR / "logs" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Threshold from val ────────────────────────────────────────────────────
    val_logits, val_labels = _infer_logits(model, val_loader, device)
    threshold = youden_threshold(val_logits, val_labels)

    # ── Internal evaluation ───────────────────────────────────────────────────
    test_ds = TN5000Dataset(str(DATA_ROOT), TEST_TXT, transform=val_transform_1x)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)

    if use_tta:
        int_scale_logits, int_logits, int_labels = tta_infer(
            model, TN5000Dataset, TEST_TXT, device, scale_val_transforms)
    else:
        int_logits, int_labels = _infer_logits(model, test_loader, device)
        int_scale_logits = {"1.00x": int_logits}

    int_m = compute_metrics(int_logits, int_labels, threshold=threshold)

    # ── External evaluation ───────────────────────────────────────────────────
    ext_metrics, ext_scale_logits, ext_logits, ext_labels = {}, {}, None, None
    if divesh_root:
        if use_tta:
            ext_scale_logits, ext_logits, ext_labels = tta_infer(
                model, None, divesh_root, device, scale_val_transforms, is_divesh=True)
        else:
            divesh_ds = DiveshDataset(divesh_root, transform=val_transform_1x)
            dl = DataLoader(divesh_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
            ext_logits, ext_labels = _infer_logits(model, dl, device)
            ext_scale_logits = {"1.00x": ext_logits}

        ext_m = compute_metrics(ext_logits, ext_labels, threshold=threshold)
        ext_individual_aucs = scale_aucs(ext_scale_logits, ext_labels)
        ext_avg_auc = float(roc_auc_score(ext_labels, ext_logits))
        corr = pairwise_corr(ext_scale_logits)
    else:
        ext_m = {}
        ext_individual_aucs = {}
        ext_avg_auc = None
        corr = {}

    out = {
        "model": run_name,
        "multiscale_train": use_multiscale_train,
        "tta": use_tta,
        "seed": SEED,
        "best_val_auc": history["best_val_auc"],
        "internal_auc": int_m["auc"],
        "internal_accuracy": int_m["accuracy"],
        "internal_sensitivity": int_m["sensitivity"],
        "internal_specificity": int_m["specificity"],
        "internal_f1": int_m["f1"],
        "external_auc": ext_m.get("auc"),
        "external_accuracy": ext_m.get("accuracy"),
        "external_sensitivity": ext_m.get("sensitivity"),
        "external_specificity": ext_m.get("specificity"),
        "external_f1": ext_m.get("f1"),
        "generalization_gap": (int_m["auc"] - ext_m["auc"]) if ext_m.get("auc") else None,
        "ext_scale_aucs": ext_individual_aucs,
        "ext_avg_auc_tta": ext_avg_auc,
        "prediction_correlations": corr,
    }

    print(f"  Internal AUC: {int_m['auc']:.4f}")
    if ext_m.get("auc"):
        print(f"  External AUC: {ext_m['auc']:.4f}  Gap: {out['generalization_gap']:.4f}")
    if ext_individual_aucs:
        for k, v in ext_individual_aucs.items():
            print(f"    AUC @ {k}: {v:.4f}")
        print(f"    Correlations: {corr}")

    with open(EXP_DIR / "metrics" / f"{run_name}.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import kagglehub
    divesh_root = kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")

    RUNS = [
        # (run_name, multiscale_train, tta)
        ("M0_AR_Baseline",      False, False),
        ("M1_MultiScale_Train", True,  False),
        ("M2_MultiScale_TTA",   False, True),
        ("M3_MultiScale_Both",  True,  True),
    ]

    results = []
    for name, ms_train, tta in RUNS:
        r = run_single(name, ms_train, tta, divesh_root, device, dry_run=args.dry_run)
        if r:
            results.append(r)

    if results:
        outpath = EXP_DIR / "results.csv"
        # Flatten nested fields for CSV
        flat = []
        for r in results:
            row = {k: v for k, v in r.items() if not isinstance(v, dict)}
            for k2, v2 in r.get("ext_scale_aucs", {}).items():
                row[f"ext_auc_{k2}"] = v2
            for k2, v2 in r.get("prediction_correlations", {}).items():
                row[f"corr_{k2}"] = v2
            flat.append(row)
        
        # Collect all unique fieldnames across all rows
        fieldnames = []
        for row in flat:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
                    
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(flat)
        with open(EXP_DIR / "metrics" / "all_runs.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {outpath}")


if __name__ == "__main__":
    main()
