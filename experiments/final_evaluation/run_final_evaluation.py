"""
Final Independent External Evaluation
Evaluates the Frozen Multi-Seed Ensemble (Swin-Tiny + Fusion + TTA) on DDTI.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import timm
from src.dataset import DDTIUniqueDataset, TN5000Dataset
from src.metrics import compute_metrics, youden_threshold
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR    = Path("experiments/final_evaluation")
DATA_ROOT  = Path("data_raw/TN5000_forReview")
VAL_TXT    = str(DATA_ROOT / "ImageSets/Main/val.txt")
DDTI_ROOT  = Path("data_raw/ddti_unique_dataset")

for d in ["predictions", "figures"]:
    (EXP_DIR / d).mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
SCALES     = [0.85, 1.00, 1.15]
SEEDS      = [0, 1, 2, 3, 4]
TARGET_RES = 256
CROP_SIZE  = 224
_USE_GPU   = torch.cuda.is_available()
_N_WORK    = 4 if _USE_GPU else 0


# ── Model ─────────────────────────────────────────────────────────────────────
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
        for h in self._hooks: h.remove()
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


# ── Transforms ────────────────────────────────────────────────────────────────
def _make_val_transform(scale: float = 1.0):
    max_size = round(TARGET_RES * scale)
    return A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(min_height=max(max_size, CROP_SIZE), min_width=max(max_size, CROP_SIZE), border_mode=0),
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

def get_predictions(model, dataset, device, scales):
    scale_logits = {}
    labels = None
    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU)
    for scale in scales:
        dataset.transform = _make_val_transform(scale)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
        logits, lbl = _infer_logits(model, loader, device)
        scale_logits[scale] = logits
        if labels is None: labels = lbl
    return scale_logits, labels


# ── Confidence Interval (DeLong / Bootstrap) ──────────────────────────────────
def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return ci_lower, ci_upper


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_roc(y_true, y_pred, auc_val, ci_lower, ci_upper, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr(y_true, y_pred, out_path):
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, color='green', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm(y_true, y_pred_prob, threshold, out_path):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Threshold={threshold:.3f})')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration(y_true, y_pred, out_path):
    prob_true, prob_pred = calibration_curve(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy(), n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution(y_true, y_pred, out_path):
    probs = torch.sigmoid(torch.tensor(y_pred)).numpy()
    plt.figure(figsize=(7,5))
    sns.histplot(probs[y_true==0], color='blue', alpha=0.5, label='Benign', bins=30, stat='density', kde=True)
    sns.histplot(probs[y_true==1], color='red', alpha=0.5, label='Malignant', bins=30, stat='density', kde=True)
    plt.xlabel('Predicted Probability (Malignant)'); plt.ylabel('Density')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_dataset = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=None)
    ext_dataset = DDTIUniqueDataset(str(DDTI_ROOT), transform=None)

    seed_ext_preds = {s: {scale: None for scale in SCALES} for s in SEEDS}
    seed_val_preds = {s: None for s in SEEDS}
    ext_labels = None
    val_labels = None

    model = MultiLevelSwin(dropout=0.3).to(device)

    for s in SEEDS:
        ckpt_path = EXP_DIR / f"checkpoints/Fusion_TTA_Seed{s}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        
        print(f"\nProcessing Seed {s}...")
        
        # Validation logits at 1.00x (for optimal threshold finding)
        val_logits_dict, v_lbl = get_predictions(model, val_dataset, device, [1.00])
        seed_val_preds[s] = val_logits_dict[1.00]
        if val_labels is None: val_labels = v_lbl

        # External logits at 0.85x, 1.00x, 1.15x
        ext_logits_dict, e_lbl = get_predictions(model, ext_dataset, device, SCALES)
        for scale in SCALES:
            seed_ext_preds[s][scale] = ext_logits_dict[scale]
        if ext_labels is None: ext_labels = e_lbl

    # ── Ensemble Aggregation ──────────────────────────────────────────────────
    print("\nAggregating Ensemble...")
    # 1. Validation Ensemble Logits (Average across seeds)
    val_ensemble_logits = np.mean([seed_val_preds[s] for s in SEEDS], axis=0)
    best_threshold = youden_threshold(val_ensemble_logits, val_labels)
    print(f"Ensemble Threshold (from Val): {best_threshold:.4f}")

    # 2. External Ensemble Logits
    # Average scales within each seed, then average across seeds
    final_seed_tta = {}
    for s in SEEDS:
        final_seed_tta[s] = np.mean([seed_ext_preds[s][scale] for scale in SCALES], axis=0)
    
    ext_ensemble_tta_logits = np.mean([final_seed_tta[s] for s in SEEDS], axis=0)
    
    # Also compute ensemble standard 1.00x for comparison
    ext_ensemble_1x_logits = np.mean([seed_ext_preds[s][1.00] for s in SEEDS], axis=0)
    ext_ensemble_085x_logits = np.mean([seed_ext_preds[s][0.85] for s in SEEDS], axis=0)
    ext_ensemble_115x_logits = np.mean([seed_ext_preds[s][0.15] if 0.15 in seed_ext_preds[s] else seed_ext_preds[s][1.15] for s in SEEDS], axis=0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def get_metrics(logits, threshold, name):
        m = compute_metrics(logits, ext_labels, threshold=threshold)
        auc_val = float(roc_auc_score(ext_labels, logits))
        return auc_val, m

    auc_tta, m_tta = get_metrics(ext_ensemble_tta_logits, best_threshold, "TTA")
    auc_1x, m_1x   = get_metrics(ext_ensemble_1x_logits, best_threshold, "1.00x")
    auc_085, _     = get_metrics(ext_ensemble_085x_logits, best_threshold, "0.85x")
    auc_115, _     = get_metrics(ext_ensemble_115x_logits, best_threshold, "1.15x")

    ci_l, ci_u = bootstrap_auc(ext_labels, ext_ensemble_tta_logits)
    
    prec, rec, _ = precision_recall_curve(ext_labels, ext_ensemble_tta_logits)
    pr_auc = auc(rec, prec)

    print(f"\nFINAL INDEPENDENT VALIDATION RESULTS (DDTI)")
    print(f"Total Samples: {len(ext_labels)} (Benign: {(ext_labels==0).sum()}, Malignant: {(ext_labels==1).sum()})")
    print(f"AUROC (TTA):   {auc_tta:.4f} (95% CI: {ci_l:.4f} - {ci_u:.4f})")
    print(f"AUROC (1.00x): {auc_1x:.4f}")
    print(f"Sensitivity:   {m_tta['sensitivity']:.4f}")
    print(f"Specificity:   {m_tta['specificity']:.4f}")
    print(f"F1 Score:      {m_tta['f1']:.4f}")

    # ── Outputs ───────────────────────────────────────────────────────────────
    out = {
        "dataset": "DDTI",
        "samples": len(ext_labels),
        "benign_count": int((ext_labels==0).sum()),
        "malignant_count": int((ext_labels==1).sum()),
        "threshold": float(best_threshold),
        "auc_0.85x": float(auc_085),
        "auc_1.00x": float(auc_1x),
        "auc_1.15x": float(auc_115),
        "auc_tta": float(auc_tta),
        "auc_tta_ci_lower": float(ci_l),
        "auc_tta_ci_upper": float(ci_u),
        "pr_auc": float(pr_auc),
        "accuracy": float(m_tta['accuracy']),
        "sensitivity": float(m_tta['sensitivity']),
        "specificity": float(m_tta['specificity']),
        "ppv": float(m_tta.get('ppv', m_tta['precision'])),
        "npv": float(m_tta['npv']),
        "f1": float(m_tta['f1'])
    }
    
    with open(EXP_DIR / "FINAL_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        w.writeheader(); w.writerow(out)
        
    # Save per-seed final predictions
    preds = []
    for i in range(len(ext_labels)):
        row = {"idx": i, "label": int(ext_labels[i]), "ensemble_1.00x": float(ext_ensemble_1x_logits[i]), "ensemble_tta": float(ext_ensemble_tta_logits[i])}
        for s in SEEDS:
            row[f"seed{s}_tta"] = float(final_seed_tta[s][i])
        preds.append(row)
    with open(EXP_DIR / "predictions" / "final_external_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        w.writeheader(); w.writerows(preds)

    # Plots
    plot_roc(ext_labels, ext_ensemble_tta_logits, auc_tta, ci_l, ci_u, EXP_DIR / "figures/roc.png")
    plot_pr(ext_labels, ext_ensemble_tta_logits, EXP_DIR / "figures/precision_recall.png")
    plot_cm(ext_labels, ext_ensemble_tta_logits, best_threshold, EXP_DIR / "figures/confusion_matrix.png")
    plot_calibration(ext_labels, ext_ensemble_tta_logits, EXP_DIR / "figures/calibration.png")
    plot_distribution(ext_labels, ext_ensemble_tta_logits, EXP_DIR / "figures/distribution.png")

    print("\nSaved all predictions, metrics, and figures to experiments/final_evaluation/")

if __name__ == "__main__":
    main()
