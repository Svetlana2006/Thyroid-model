"""
External Validation Script for Divesh Dataset
Evaluates the frozen 5-seed MultiLevelSwin ensemble on the diveshzz dataset.
"""

import argparse
import csv
import glob
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, recall_score, precision_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_curve, precision_recall_curve
)
from torch.utils.data import DataLoader, Dataset
import kagglehub

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import timm
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

PROJ_DIM = 128
FUSION_DIM = 256
THRESHOLD = 0.5912
DIVESH_EXPECTED_AUC = 0.8244150886

OUTPUT_DIR = Path("outputs/final_model/evaluation/diveshzz")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}
    def __init__(self, dropout: float = 0.0):
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
        self._stage_feats = {}
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
            key = name.replace(".", "_")
            feat = self._stage_feats[name].mean(dim=(1, 2))
            feat = self.stage_norms[key](feat)
            feat = self.stage_projs[key](feat)
            pooled.append(feat)
        fused = torch.cat(pooled, dim=-1)
        return self.fusion_head(fused)

def make_val_transform(scale: float = 1.0):
    max_size = round(256 * scale)
    return A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(min_height=max(max_size, 224), min_width=max(max_size, 224), border_mode=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

TTA_SCALES = [0.85, 1.00, 1.15]
TTA_TRANSFORMS = [make_val_transform(s) for s in TTA_SCALES]

class DiveshDataset(Dataset):
    def __init__(self, data_root: str):
        self.samples = []
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        class_0_dir = os.path.join(dataset_dir, "0")
        if os.path.exists(class_0_dir):
            for img_file in glob.glob(os.path.join(class_0_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({"img_path": img_file, "label": 0})
        class_1_dir = os.path.join(dataset_dir, "1")
        if os.path.exists(class_1_dir):
            for img_file in glob.glob(os.path.join(class_1_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({"img_path": img_file, "label": 1})
                    
        for i, s in enumerate(self.samples):
            s["id"] = f"{i:04d}_{os.path.basename(s['img_path'])}"

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
        return tensors, s["label"], s["id"]

@torch.no_grad()
def get_predictions(model, loader, device, seed_idx):
    from collections import defaultdict
    model.eval()
    preds = defaultdict(lambda: {"label": None, "logits": [[] for _ in TTA_SCALES]})
    
    desc_str = f"Evaluating Seed {seed_idx}"
    iterator = tqdm(loader, desc=desc_str, leave=False) if HAS_TQDM else loader
    for tensors, labels, ids in iterator:
        tensors = tensors.to(device)
        B, num_tta, C, H, W = tensors.shape
        tensors = tensors.view(B * num_tta, C, H, W)
        
        logits = model(tensors).squeeze(-1).view(B, num_tta).cpu().float().numpy()
        labels = labels.numpy()
        
        for i in range(B):
            obj_id = ids[i]
            preds[obj_id]["label"] = int(labels[i])
            for s_idx in range(num_tta):
                preds[obj_id]["logits"][s_idx].append(logits[i, s_idx])
                
    final_preds = {}
    for obj_id, data in preds.items():
        final_preds[obj_id] = {
            "label": data["label"],
            "scale_logits": [img_logits[0] for img_logits in data["logits"]]
        }
    return final_preds

def expit(x): return 1 / (1 + np.exp(-x))

def get_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95):
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[idx])) < 2: continue
        scores.append(roc_auc_score(y_true[idx], y_pred[idx]))
    scores.sort()
    return np.percentile(scores, (100-ci)/2), np.percentile(scores, 100-(100-ci)/2)

def evaluate(preds_dict):
    ids = sorted(list(preds_dict.keys()))
    y_true = np.array([preds_dict[i]["label"] for i in ids])
    
    seed_tta_logits = []
    for s_idx in range(5):
        s_logits = [np.mean(preds_dict[i]["seeds"][s_idx]) for i in ids]
        seed_tta_logits.append(np.array(s_logits))
    
    ensemble_logits = np.mean(seed_tta_logits, axis=0)
    y_pred_prob = expit(ensemble_logits)
    y_pred_class = (y_pred_prob >= THRESHOLD).astype(int)
    
    scale_ensemble_logits = []
    for scale_idx in range(3):
        scale_preds = [np.mean([preds_dict[i]["seeds"][s][scale_idx] for s in range(5)]) for i in ids]
        scale_ensemble_logits.append(np.array(scale_preds))
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    ci_l, ci_u = get_bootstrap_ci(y_true, ensemble_logits)
    
    metrics = {
        "N": len(y_true), "Benign": int(np.sum(y_true == 0)), "Malignant": int(np.sum(y_true == 1)),
        "AUROC": roc_auc_score(y_true, ensemble_logits),
        "95% CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
        "PR-AUC": average_precision_score(y_true, y_pred_prob),
        "Accuracy": accuracy_score(y_true, y_pred_class),
        "Sensitivity": recall_score(y_true, y_pred_class),
        "Specificity": tn / (tn + fp) if (tn+fp)>0 else 0,
        "PPV": precision_score(y_true, y_pred_class, zero_division=0),
        "NPV": tn / (tn + fn) if (tn+fn)>0 else 0,
        "F1": f1_score(y_true, y_pred_class),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred_class),
        "MCC": matthews_corrcoef(y_true, y_pred_class),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred_class),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "AUROC_0.85x": roc_auc_score(y_true, scale_ensemble_logits[0]),
        "AUROC_1.00x": roc_auc_score(y_true, scale_ensemble_logits[1]),
        "AUROC_1.15x": roc_auc_score(y_true, scale_ensemble_logits[2]),
    }
    
    seed_aurocs = [roc_auc_score(y_true, s_logits) for s_logits in seed_tta_logits]
    metrics["Seed_AUROCs"] = seed_aurocs
    metrics["Seed_Mean"] = np.mean(seed_aurocs)
    metrics["Seed_SD"] = np.std(seed_aurocs)
    metrics["Seed_Min"] = np.min(seed_aurocs)
    metrics["Seed_Max"] = np.max(seed_aurocs)
    metrics["Seed_Median"] = np.median(seed_aurocs)

    return metrics, y_true, y_pred_prob

def generate_outputs(preds_dict, metrics, y_true, y_pred_prob):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {metrics['AUROC']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"Diveshzz External ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(FIG_DIR / "Diveshzz_ROC.png", bbox_inches='tight')
    plt.close()
    
    # PR
    p, r, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(r, p, label=f"PR-AUC = {metrics['PR-AUC']:.4f}")
    plt.title(f"Diveshzz External PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(FIG_DIR / "Diveshzz_PR.png", bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title("Diveshzz External Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(FIG_DIR / "Diveshzz_CM.png", bbox_inches='tight')
    plt.close()

    # CSV Predictions
    ids = sorted(list(preds_dict.keys()))
    with open(OUTPUT_DIR / "diveshzz_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "true_label"]
        for s in range(5):
            header.extend([f"seed{s}_0.85x", f"seed{s}_1.00x", f"seed{s}_1.15x", f"seed{s}_TTA"])
        header.extend(["ensemble_TTA_logit", "ensemble_prob", "predicted_class"])
        writer.writerow(header)
        for i in ids:
            row = [i, preds_dict[i]["label"]]
            seed_tta_logits = []
            for s in range(5):
                scales = preds_dict[i]["seeds"][s]
                tta = np.mean(scales)
                seed_tta_logits.append(tta)
                row.extend([scales[0], scales[1], scales[2], tta])
            ens_logit = np.mean(seed_tta_logits)
            ens_prob = expit(ens_logit)
            pred_class = 1 if ens_prob >= THRESHOLD else 0
            row.extend([ens_logit, ens_prob, pred_class])
            writer.writerow(row)

    # Error Analysis
    for i in preds_dict:
        scale_ens = [np.mean([preds_dict[i]["seeds"][s][s_idx] for s in range(5)]) for s_idx in range(3)]
        preds_dict[i]["max_disagreement"] = max(scale_ens) - min(scale_ens)
        preds_dict[i]["prob"] = expit(np.mean([np.mean(preds_dict[i]["seeds"][s]) for s in range(5)]))
        preds_dict[i]["pred_class"] = 1 if preds_dict[i]["prob"] >= THRESHOLD else 0

    fps = sorted([(i, p) for i, p in preds_dict.items() if p["label"]==0 and p["pred_class"]==1], key=lambda x: x[1]["prob"], reverse=True)
    fns = sorted([(i, p) for i, p in preds_dict.items() if p["label"]==1 and p["pred_class"]==0], key=lambda x: x[1]["prob"])
    
    with open(OUTPUT_DIR / "diveshzz_error_analysis.txt", "w") as f:
        f.write(f"Total False Positives: {len(fps)}\nTotal False Negatives: {len(fns)}\n\n")
        f.write("--- Top False Positives ---\n")
        for i, p in fps[:20]: f.write(f"ID: {i} | Prob: {p['prob']:.4f} | Max TTA Disagreement: {p['max_disagreement']:.4f}\n")
        f.write("\n--- Top False Negatives ---\n")
        for i, p in fns[:20]: f.write(f"ID: {i} | Prob: {p['prob']:.4f} | Max TTA Disagreement: {p['max_disagreement']:.4f}\n")

    # Evaluation Report
    report = "# Diveshzz External Dataset Evaluation\n\n"
    for k, v in metrics.items():
        if isinstance(v, float): report += f"**{k}:** {v:.4f}\n"
        elif isinstance(v, list): report += f"**{k}:** {[round(x, 4) for x in v]}\n"
        else: report += f"**{k}:** {v}\n"
        
    report += f"\n### Comparison with Established Result\nEstablished AUROC: {DIVESH_EXPECTED_AUC:.4f}\nRerun AUROC: {metrics['AUROC']:.4f}\n"
    with open(OUTPUT_DIR / "diveshzz_report.md", "w") as f: f.write(report)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    divesh_path = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
    ds = DiveshDataset(divesh_path)
    loader = DataLoader(ds, batch_size=8, num_workers=4 if torch.cuda.is_available() else 0)
    
    from collections import defaultdict
    all_preds = defaultdict(lambda: {"label": None, "seeds": []})
    
    for seed in range(5):
        ckpt_path = Path("outputs/final_model") / f"seed{seed}" / "best.pt"
        model = MultiLevelSwin(dropout=0.0).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
        preds = get_predictions(model, loader, device, seed_idx=seed)
        
        # Quick per-seed AUROC calculation for console output
        seed_y_true = []
        seed_y_scores = []
        for obj_id, data in preds.items():
            all_preds[obj_id]["label"] = data["label"]
            all_preds[obj_id]["seeds"].append(data["scale_logits"])
            seed_y_true.append(data["label"])
            seed_y_scores.append(np.mean(data["scale_logits"]))
            
        print(f"Seed {seed} | TTA AUROC: {roc_auc_score(seed_y_true, seed_y_scores):.4f}")
            
    metrics, y_true, y_pred_prob = evaluate(all_preds)
    generate_outputs(all_preds, metrics, y_true, y_pred_prob)
    
    print("\n" + "="*50)
    print("DIVESHZZ EXTERNAL VALIDATION RESULTS")
    print("="*50)
    print(f"Ensemble TTA AUROC : {metrics['AUROC']:.4f}")
    print(f"95% Confidence Int : {metrics['95% CI']}")
    print(f"PR-AUC             : {metrics['PR-AUC']:.4f}")
    print(f"Reference AUROC    : {DIVESH_EXPECTED_AUC:.4f}")
    print("="*50)
    print(f"Diveshzz evaluation complete. Artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__": main()
