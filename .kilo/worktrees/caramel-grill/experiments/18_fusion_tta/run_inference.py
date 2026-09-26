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
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import timm
from src.dataset import TN5000Dataset
from src.metrics import compute_metrics, youden_threshold
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR = Path("experiments/18_fusion_tta")
DATA_ROOT = Path("data_raw/TN5000_forReview")
VAL_TXT   = str(DATA_ROOT / "ImageSets/Main/val.txt")
TEST_TXT  = str(DATA_ROOT / "ImageSets/Main/test.txt")

EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
BATCH_SIZE = 16
SCALES = [0.85, 1.00, 1.15]
TARGET_RES = 256
CROP_SIZE = 224
_USE_GPU = torch.cuda.is_available()
_N_WORK = 4 if _USE_GPU else 0


# ── Model Definition (copied exactly from Exp 17) ─────────────────────────────
PROJ_DIM = 128
FUSION_DIM = 256

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}

    def __init__(self, dropout: float = 0.3):
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


# ── Transforms ────────────────────────────────────────────────────────────────
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


# ── Inference ─────────────────────────────────────────────────────────────────
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
        if is_divesh:
            ds = DiveshDataset(data_root_or_samples, transform=t)
        else:
            ds = dataset_cls(str(DATA_ROOT), data_root_or_samples, transform=t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
        logits, lbl = _infer_logits(model, loader, device)
        all_scale_logits[f"{scale:.2f}x"] = logits
        if labels is None:
            labels = lbl

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


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Exp 17 Fusion Model
    ckpt_path = Path("experiments/17_multilevel_fusion/checkpoints/M1_Multilevel_Fusion_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    model = MultiLevelSwin(dropout=0.3)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    import kagglehub
    divesh_root = kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")

    scale_transforms = [_make_val_transform(s) for s in SCALES]

    # Threshold from Validation Set (1.00x)
    val_ds = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=_make_val_transform(1.00))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=_N_WORK, pin_memory=_USE_GPU)
    val_logits, val_labels = _infer_logits(model, val_loader, device)
    threshold = youden_threshold(val_logits, val_labels)

    print("Running Internal Test TTA...")
    int_scale_logits, int_logits, int_labels = tta_infer(
        model, TN5000Dataset, TEST_TXT, device, scale_transforms, is_divesh=False
    )
    int_m = compute_metrics(int_logits, int_labels, threshold=threshold)

    print("Running External Divesh TTA...")
    ext_scale_logits, ext_logits, ext_labels = tta_infer(
        model, None, divesh_root, device, scale_transforms, is_divesh=True
    )
    ext_m = compute_metrics(ext_logits, ext_labels, threshold=threshold)

    # Scale-specific AUCs
    ext_aucs = {k: float(roc_auc_score(ext_labels, v)) for k, v in ext_scale_logits.items()}
    ext_avg_auc = float(roc_auc_score(ext_labels, ext_logits))
    corr = pairwise_corr(ext_scale_logits)

    out = {
        "model": "Exp18_Fusion_TTA",
        "checkpoint": str(ckpt_path),
        "internal_auc": int_m["auc"],
        "external_auc": ext_m["auc"], # This uses the threshold on averaged logits, but we also save ext_avg_auc (which is threshold-independent)
        "generalization_gap": int_m["auc"] - ext_m["auc"],
        "ext_auc_0.85x": ext_aucs["0.85x"],
        "ext_auc_1.00x": ext_aucs["1.00x"],
        "ext_auc_1.15x": ext_aucs["1.15x"],
        "ext_auc_avg": ext_avg_auc,
        "corr_0.85x_vs_1.00x": corr.get("0.85x_vs_1.00x"),
        "corr_0.85x_vs_1.15x": corr.get("0.85x_vs_1.15x"),
        "corr_1.00x_vs_1.15x": corr.get("1.00x_vs_1.15x"),
    }

    # Save summary metrics
    with open(EXP_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        w.writeheader()
        w.writerow(out)

    with open(EXP_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # Save detailed predictions for Divesh
    preds_out = []
    for i in range(len(ext_labels)):
        preds_out.append({
            "idx": i,
            "label": int(ext_labels[i]),
            "logit_0.85x": float(ext_scale_logits["0.85x"][i]),
            "logit_1.00x": float(ext_scale_logits["1.00x"][i]),
            "logit_1.15x": float(ext_scale_logits["1.15x"][i]),
            "logit_avg": float(ext_logits[i]),
        })
    with open(EXP_DIR / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds_out[0].keys()))
        w.writeheader()
        w.writerows(preds_out)

    print("\n--- RESULTS ---")
    print(f"Internal AUC:       {out['internal_auc']:.4f}")
    print(f"External AUC (Avg): {out['ext_auc_avg']:.4f}")
    print(f"Generalization Gap: {out['generalization_gap']:.4f}")
    print("\nScale-Specific AUCs:")
    print(f"  AUC @ 0.85x: {out['ext_auc_0.85x']:.4f}")
    print(f"  AUC @ 1.00x: {out['ext_auc_1.00x']:.4f}")
    print(f"  AUC @ 1.15x: {out['ext_auc_1.15x']:.4f}")
    print(f"  AUC @ Avg:   {out['ext_auc_avg']:.4f}")
    print("\nCorrelations:")
    for k, v in corr.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
