"""
Experiment 19: Multi-Seed Confirmation of Winning Configuration
Configuration: Swin-Tiny + Multi-Level Fusion + Multi-Scale TTA
"""

import argparse
import csv
import glob
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
from PIL import Image as PILImage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import timm
from src.dataset import TN5000Dataset, AUITDDataset
from src.metrics import compute_metrics, youden_threshold
from src.trainer import EarlyStopping, build_optimizer_and_scheduler, evaluate, train_one_epoch
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR    = Path("experiments/19_multiseed_confirmation")
DATA_ROOT  = Path("data_raw/TN5000_forReview")
AUITD_ROOT = "data_raw/auitd_dataset"
TRAIN_TXT  = str(DATA_ROOT / "ImageSets/Main/train.txt")
VAL_TXT    = str(DATA_ROOT / "ImageSets/Main/val.txt")
TEST_TXT   = str(DATA_ROOT / "ImageSets/Main/test.txt")

for d in ["checkpoints", "logs", "metrics", "predictions"]:
    (EXP_DIR / d).mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
SCALES     = [0.85, 1.00, 1.15]
TARGET_RES = 256
CROP_SIZE  = 224
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

_CURRENT_SEED = 0

def worker_init_fn(worker_id):
    # This must strictly use the global worker_id and seed offset
    # We use a global variable _CURRENT_SEED set during the loop
    global _CURRENT_SEED
    np.random.seed(_CURRENT_SEED + worker_id)


# ── Transforms ────────────────────────────────────────────────────────────────
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

def _make_val_transform(scale: float = 1.0):
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


# ── Model ─────────────────────────────────────────────────────────────────────
PROJ_DIM = 128
FUSION_DIM = 256

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}

    def __init__(self, dropout: float = 0.3):
        super().__init__()
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


# ── Divesh Dataset ────────────────────────────────────────────────────────────
class DiveshDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        self.samples = []
        td = os.path.join(data_root, "Thyroid Data")
        for label, sub in [(0, "0"), (1, "1")]:
            d = os.path.join(td, sub)
            if not os.path.exists(d): continue
            for f in glob.glob(os.path.join(d, "*.*")):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({"img_path": f, "label": label})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(PILImage.open(s["img_path"]).convert("RGB"))
        if self.transform: img = self.transform(image=img)["image"]
        return img, torch.tensor(s["label"], dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────
def train_model_standard(model, train_loader, val_loader, config, run_name, device):
    model = model.to(device)
    lr_head = config["lr_head"]
    lr_backbone = lr_head * 0.1

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr_head, lr_backbone, config["weight_decay"],
        config["T_0"], config["T_mult"], -1
    )
    early_stopping = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    _prev_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    for epoch in range(1, config["max_epochs"] + 1):
        model.freeze_epoch(epoch)
        _curr = sum(1 for p in model.parameters() if p.requires_grad)
        if _curr != _prev_trainable:
            print(f"    [unfreeze] {_prev_trainable}→{_curr} trainable params. Rebuilding optimizer.")
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
            config["pos_weight"], 1.0,
            config.get("label_smooth_eps", 0.05),
            config.get("grad_clip_norm", 1.0),
            epoch=epoch, run_name=run_name
        )
        scheduler.step()
        val_m = evaluate(model, val_loader, device, config["pos_weight"], 1.0)

        history["train_loss"].append(train_m["loss"])
        history["train_auc"].append(train_m["auc"])
        history["val_loss"].append(val_m["loss"])
        history["val_auc"].append(val_m["auc"])

        print(f"  Epoch {epoch:02d} | Train AUC {train_m['auc']:.4f} | Val AUC {val_m['auc']:.4f}")

        if early_stopping(val_m["auc"], model):
            print(f"  Early stop at epoch {epoch}.")
            break
        if early_stopping.counter == 0:
            torch.save(model.state_dict(), EXP_DIR / "checkpoints" / f"{run_name}_best.pt")

    early_stopping.restore_best(model)
    history["best_val_auc"] = early_stopping.best_score
    history["best_epoch"] = epoch - early_stopping.counter
    return history


# ── Inference helpers ─────────────────────────────────────────────────────────
def _infer_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_logits), np.array(all_labels)

def tta_infer(model, dataset_cls, data_root_or_samples, device, scale_transforms, is_divesh=False):
    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU)
    all_scale_logits = {}
    labels = None
    for scale, t in zip(SCALES, scale_transforms):
        if is_divesh: ds = DiveshDataset(data_root_or_samples, transform=t)
        else: ds = dataset_cls(str(DATA_ROOT), data_root_or_samples, transform=t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
        logits, lbl = _infer_logits(model, loader, device)
        all_scale_logits[f"{scale:.2f}x"] = logits
        if labels is None: labels = lbl
    avg_logits = np.mean(list(all_scale_logits.values()), axis=0)
    return all_scale_logits, avg_logits, labels

def pairwise_corr(scale_logits):
    keys = list(scale_logits.keys())
    result = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            r = float(np.corrcoef(scale_logits[k1], scale_logits[k2])[0, 1])
            result[f"{k1}_vs_{k2}"] = round(r, 4)
    return result


# ── Single Run ────────────────────────────────────────────────────────────────
def run_seed(seed, divesh_root, device, dry_run=False):
    run_name = f"Fusion_TTA_Seed{seed}"
    print(f"\n{'='*60}\n  Starting {run_name}\n{'='*60}")
    
    global _CURRENT_SEED
    _CURRENT_SEED = seed
    set_seed(seed)

    train_t = make_train_transform()
    val_t_1x = _make_val_transform(1.0)
    scale_transforms = [_make_val_transform(s) for s in SCALES]

    train_ds = torch.utils.data.ConcatDataset([
        TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t),
        AUITDDataset(AUITD_ROOT, transform=train_t),
    ])
    val_ds = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=val_t_1x)

    all_labels = np.concatenate([
        TN5000Dataset(str(DATA_ROOT), TRAIN_TXT).get_labels(),
        AUITDDataset(AUITD_ROOT).get_labels()
    ])
    pos_weight = float((all_labels == 0).sum() / (all_labels == 1).sum())

    config = {
        "lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
        "pos_weight": pos_weight, "batch_size": BATCH_SIZE,
        "max_epochs": 25, "patience": 10, "min_delta": 0.001,
        "T_0": 10, "T_mult": 2, "grad_clip_norm": 1.0, "label_smooth_eps": 0.05,
    }

    model = MultiLevelSwin(dropout=config["dropout"])
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Config logged:")
    print(f"  Seed: {seed}")
    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"  Model: MultiLevelSwin | Total Params: {total_params:,}")
    print(f"  Optimizer: AdamW | LR Head: {config['lr_head']} | LR Backbone: {config['lr_head']*0.1}")
    print(f"  Scheduler: CosineAnnealingWarmRestarts")
    print(f"  Batch size: {config['batch_size']} | Max Epochs: {config['max_epochs']}")
    print(f"  Precision: {'AMP' if _USE_GPU else 'FP32'}")
    print(f"  TTA Scales: {SCALES}")

    if dry_run:
        print("  [DRY RUN] Skipping training.")
        return None

    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)

    t0 = time.time()
    history = train_model_standard(model, train_loader, val_loader, config, run_name, device)
    train_time = time.time() - t0

    with open(EXP_DIR / "logs" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Inference ─────────────────────────────────────────────────────────────
    # Val threshold
    val_logits, val_labels = _infer_logits(model, val_loader, device)
    threshold = youden_threshold(val_logits, val_labels)

    # Internal TTA
    int_scale_logits, int_logits, int_labels = tta_infer(
        model, TN5000Dataset, TEST_TXT, device, scale_transforms, is_divesh=False
    )
    int_m = compute_metrics(int_logits, int_labels, threshold=threshold)

    # External Divesh TTA
    ext_scale_logits, ext_logits, ext_labels = tta_infer(
        model, None, divesh_root, device, scale_transforms, is_divesh=True
    )
    ext_m = compute_metrics(ext_logits, ext_labels, threshold=threshold)

    ext_aucs = {k: float(roc_auc_score(ext_labels, v)) for k, v in ext_scale_logits.items()}
    ext_avg_auc = float(roc_auc_score(ext_labels, ext_logits))
    corr = pairwise_corr(ext_scale_logits)

    out = {
        "seed": seed,
        "best_epoch": history["best_epoch"],
        "train_time_sec": train_time,
        "param_count": total_params,
        "internal_auc": int_m["auc"],
        "external_auc_1.00x": ext_aucs["1.00x"],
        "external_auc_avg": ext_avg_auc,
        "generalization_gap": int_m["auc"] - ext_avg_auc,
        "ext_auc_0.85x": ext_aucs["0.85x"],
        "ext_auc_1.15x": ext_aucs["1.15x"],
        "corr_0.85x_vs_1.00x": corr.get("0.85x_vs_1.00x"),
        "corr_0.85x_vs_1.15x": corr.get("0.85x_vs_1.15x"),
        "corr_1.00x_vs_1.15x": corr.get("1.00x_vs_1.15x"),
    }

    with open(EXP_DIR / "metrics" / f"{run_name}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"  Internal AUC:       {out['internal_auc']:.4f}")
    print(f"  External AUC 1.00x: {out['external_auc_1.00x']:.4f}")
    print(f"  External AUC TTA:   {out['external_auc_avg']:.4f}")
    print(f"  Gap:                {out['generalization_gap']:.4f}")

    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import kagglehub
    divesh_root = kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")

    SEEDS = [0, 1, 2, 3, 4]
    results = []

    for s in SEEDS:
        r = run_seed(s, divesh_root, device, dry_run=args.dry_run)
        if r:
            results.append(r)

    if not results:
        return

    # ── Aggregate Statistics ──────────────────────────────────────────────────
    metrics_to_agg = [
        "internal_auc", "external_auc_1.00x", "external_auc_avg", "generalization_gap"
    ]
    summary = {}
    for m in metrics_to_agg:
        vals = [r[m] for r in results]
        summary[f"{m}_mean"] = np.mean(vals)
        summary[f"{m}_std"]  = np.std(vals)
        summary[f"{m}_min"]  = np.min(vals)
        summary[f"{m}_max"]  = np.max(vals)
        summary[f"{m}_median"] = np.median(vals)

    print("\n" + "="*60)
    print("  MULTI-SEED SUMMARY")
    print("="*60)
    print(f"External AUC (TTA):   {summary['external_auc_avg_mean']:.4f} ± {summary['external_auc_avg_std']:.4f}")
    print(f"External AUC (1.00x): {summary['external_auc_1.00x_mean']:.4f} ± {summary['external_auc_1.00x_std']:.4f}")
    print(f"Internal AUC:         {summary['internal_auc_mean']:.4f} ± {summary['internal_auc_std']:.4f}")
    print(f"Generalization Gap:   {summary['generalization_gap_mean']:.4f} ± {summary['generalization_gap_std']:.4f}")

    # Save to CSV
    outpath = EXP_DIR / "per_seed_results.csv"
    fieldnames = list(results[0].keys())
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    with open(EXP_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved full results to {outpath}")


if __name__ == "__main__":
    main()
