"""
Experiment 17: Multi-Level Feature Fusion with Swin-Tiny

Swin-Tiny stage output shapes (confirmed experimentally):
  Stage 0: [B, 56, 56,  96]  ← fine spatial features
  Stage 1: [B, 28, 28, 192]  ← intermediate
  Stage 2: [B, 14, 14, 384]  ← higher-level
  Stage 3: [B,  7,  7, 768]  ← final semantic (current model)

M0: AR Swin baseline (control)
M1: AR Swin + multi-level fusion head (Stages 2+3+4 → fused classifier)
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
import timm
from src.dataset import TN5000Dataset, AUITDDataset
from src.metrics import compute_metrics, youden_threshold
from src.trainer import EarlyStopping, build_optimizer_and_scheduler, evaluate, train_one_epoch
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR    = Path("experiments/17_multilevel_fusion")
DATA_ROOT  = Path("data_raw/TN5000_forReview")
AUITD_ROOT = "data_raw/auitd_dataset"
TRAIN_TXT  = str(DATA_ROOT / "ImageSets/Main/train.txt")
VAL_TXT    = str(DATA_ROOT / "ImageSets/Main/val.txt")
TEST_TXT   = str(DATA_ROOT / "ImageSets/Main/test.txt")

for d in ["checkpoints", "logs", "metrics"]:
    (EXP_DIR / d).mkdir(parents=True, exist_ok=True)

SEED       = 0
BATCH_SIZE = 16
_USE_GPU   = torch.cuda.is_available()
_N_WORK    = 4 if _USE_GPU else 0
PROJ_DIM   = 128   # each stage projects to this dimension before fusion
FUSION_DIM = 256   # size of the fusion layer


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


# ── Multi-Level Fusion Model ──────────────────────────────────────────────────
class MultiLevelSwin(nn.Module):
    """
    Swin-Tiny backbone with a lightweight multi-level feature fusion head.

    Uses forward hooks to capture intermediate stage outputs:
      Stage 1 (layers[1]): [B, 28, 28, 192]
      Stage 2 (layers[2]): [B, 14, 14, 384]
      Stage 3 (layers[3]): [B,  7,  7, 768]  ← final stage

    Each stage output is:
      1. GAP over spatial dimensions  → [B, C]
      2. LayerNorm                    → normalise per stage
      3. Linear projection            → [B, PROJ_DIM]

    Concatenate → [B, PROJ_DIM * n_stages]
    → Linear(PROJ_DIM*n, FUSION_DIM) → GELU → Dropout → Linear(FUSION_DIM, 1)
    """

    # Confirmed stage channel sizes from live model inspection
    STAGE_CHANNELS = {
        "layers.1": 192,
        "layers.2": 384,
        "layers.3": 768,
    }

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )

        # Freeze initially (same as SwinTinyClassifier)
        for param in self.backbone.parameters():
            param.requires_grad = False

        n_stages = len(self.STAGE_CHANNELS)

        # Per-stage: LayerNorm → Linear projection
        self.stage_norms = nn.ModuleDict()
        self.stage_projs = nn.ModuleDict()
        for name, ch in self.STAGE_CHANNELS.items():
            key = name.replace(".", "_")
            self.stage_norms[key] = nn.LayerNorm(ch)
            self.stage_projs[key] = nn.Linear(ch, PROJ_DIM, bias=False)

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(PROJ_DIM * n_stages, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(FUSION_DIM, 1),
        )

        # Storage for hook outputs (populated during forward)
        self._stage_feats: dict = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks on the required Swin stages."""
        for name in self.STAGE_CHANNELS:
            # name is e.g. "layers.1" → access backbone.layers[1]
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
        _ = self.backbone(x)   # runs the full Swin forward; hooks populate _stage_feats
        self._remove_hooks()

        pooled = []
        for name in self.STAGE_CHANNELS:
            key  = name.replace(".", "_")
            feat = self._stage_feats[name]          # [B, H, W, C] (Swin uses HWC layout)
            # GAP over spatial dims
            feat = feat.mean(dim=(1, 2))            # [B, C]
            feat = self.stage_norms[key](feat)      # LayerNorm
            feat = self.stage_projs[key](feat)      # [B, PROJ_DIM]
            pooled.append(feat)

        fused = torch.cat(pooled, dim=-1)           # [B, PROJ_DIM * n_stages]
        return self.fusion_head(fused)              # [B, 1]

    def freeze_epoch(self, epoch: int):
        """Identical staged unfreezing as SwinTinyClassifier."""
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

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_only = (sum(p.numel() for p in self.stage_norms.parameters()) +
                     sum(p.numel() for p in self.stage_projs.parameters()) +
                     sum(p.numel() for p in self.fusion_head.parameters()))
        return {"total": total, "trainable_at_init": trainable, "fusion_head": head_only}


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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(PILImage.open(s["img_path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(s["label"], dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, config, run_name, device):
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

        print(f"[{run_name}] Epoch {epoch:02d} | Train AUC {train_m['auc']:.4f} | Val AUC {val_m['auc']:.4f}")

        if early_stopping(val_m["auc"], model):
            print(f"[{run_name}] Early stop at epoch {epoch}.")
            break
        if early_stopping.counter == 0:
            torch.save(model.state_dict(), EXP_DIR / "checkpoints" / f"{run_name}_best.pt")

    early_stopping.restore_best(model)
    history["best_val_auc"] = early_stopping.best_score
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────
def _infer(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_logits), np.array(all_labels)


# ── Single run ────────────────────────────────────────────────────────────────
def run_single(run_name, use_fusion, divesh_root, device, dry_run=False):
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")
    set_seed(SEED)

    train_t = make_train_transform()
    val_t   = make_val_transform()

    train_ds = torch.utils.data.ConcatDataset([
        TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t),
        AUITDDataset(AUITD_ROOT, transform=train_t),
    ])
    val_ds  = TN5000Dataset(str(DATA_ROOT), VAL_TXT,  transform=val_t)
    test_ds = TN5000Dataset(str(DATA_ROOT), TEST_TXT, transform=val_t)

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

    if use_fusion:
        model = MultiLevelSwin(dropout=config["dropout"])
        param_info = model.count_params()
        print(f"  Fusion head params: {param_info['fusion_head']:,}")
        print(f"  Total params:       {param_info['total']:,}")
    else:
        from src.models import build_model
        model = build_model("swin_tiny", dropout=config["dropout"])
        total  = sum(p.numel() for p in model.parameters())
        head_p = sum(p.numel() for p in model.head.parameters())
        param_info = {"total": total, "trainable_at_init": head_p, "fusion_head": head_p}
        print(f"  Head params:  {head_p:,}")
        print(f"  Total params: {total:,}")

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    if dry_run:
        print("  [DRY RUN] skipping training.")
        return None

    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False, **kw)

    history = train_model(model, train_loader, val_loader, config, run_name, device)

    with open(EXP_DIR / "logs" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Threshold from val ────────────────────────────────────────────────────
    val_logits, val_labels = _infer(model, val_loader, device)
    threshold = youden_threshold(val_logits, val_labels)

    # ── Internal test ─────────────────────────────────────────────────────────
    int_logits, int_labels = _infer(model, test_loader, device)
    int_m = compute_metrics(int_logits, int_labels, threshold=threshold)

    # ── External Divesh ───────────────────────────────────────────────────────
    ext_m = {}
    if divesh_root:
        divesh_ds = DiveshDataset(divesh_root, transform=val_t)
        dl = DataLoader(divesh_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
        ext_logits, ext_labels = _infer(model, dl, device)
        ext_m = compute_metrics(ext_logits, ext_labels, threshold=threshold)

    out = {
        "model": run_name,
        "use_fusion": use_fusion,
        "seed": SEED,
        "param_info": param_info,
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
    }

    print(f"  Internal AUC: {int_m['auc']:.4f}")
    if ext_m.get("auc"):
        print(f"  External AUC: {ext_m['auc']:.4f}  Gap: {out['generalization_gap']:.4f}")

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
        # (run_name, use_fusion)
        ("M0_AR_Baseline",      False),
        ("M1_Multilevel_Fusion", True),
    ]

    results = []
    for name, use_fusion in RUNS:
        r = run_single(name, use_fusion, divesh_root, device, dry_run=args.dry_run)
        if r:
            results.append(r)

    if results:
        outpath = EXP_DIR / "results.csv"
        flat = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "param_info"}
            row.update({f"param_{k}": v for k, v in r.get("param_info", {}).items()})
            flat.append(row)
        fieldnames = list(flat[0].keys())
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(flat)
        print(f"\nSaved results to {outpath}")


if __name__ == "__main__":
    main()
