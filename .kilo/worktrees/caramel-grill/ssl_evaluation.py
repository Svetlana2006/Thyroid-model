#!/usr/bin/env python3
"""
SSL Pretraining Evaluation Script
Evaluates the SSL-pretrained 5-seed MultiLevelSwin ensemble on TN5000 internal,
Diveshzz external, and Thyroid-for-Pretraining external datasets.

This script adapts the baseline evaluation scripts (evaluate_internal_tn5000.py,
external_validation_divesh.py) to work with SSL pretrained models.

SSL experiment supervised checkpoints are located at:
  ssl_pretraining_experiment/supervised/seed{seed}/best.pt  (seed 1)
  ssl_pretraining_experiment/supervised/seed{seed}/final_last.pt (seeds 0, 2, 3, 4)

These checkpoints are created by ssl_pretraining_experiment/scripts/supervised_train.py
which loads the SSL backbone from ssl_pretraining_experiment/ssl_pretrain/best_ssl.pt
"""

import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
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
import cv2

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.transforms import IMAGENET_MEAN, IMAGENET_STD
import kagglehub

THRESHOLD = 0.5912

OUTPUT_DIR = Path("ssl_pretraining_experiment/evaluation")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Import MultiLevelSwin from the SSL experiment's supervised_train module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.supervised_train import MultiLevelSwin, make_val_transform

TTA_SCALES = [0.70, 0.85, 1.00, 1.15, 1.30]
TTA_TRANSFORMS = [make_val_transform(s) for s in TTA_SCALES]

# Baseline results (A4S1V2, ImageNet-pretrained) for comparison
BASELINE_INTERNAL_AUROC = 0.9566
BASELINE_INTERNAL_N = 1000
BASELINE_INTERNAL_MALIGNANT = 731
BASELINE_DIVESHZZ_AUROC = 0.8298
BASELINE_DIVESHZZ_N = 3115
BASELINE_DIVESHZZ_MALIGNANT = 1210
BASELINE_THYROID_FOR_PRETRAINING_AUROC = 0.8261
BASELINE_THYROID_FOR_PRETRAINING_N = 3644
BASELINE_THYROID_FOR_PRETRAINING_MALIGNANT = 2003

class TN5000TestDataset(Dataset):
    """Internal TN5000 test dataset (same as baseline)"""
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.img_dir = self.data_root / "JPEGImages"
        self.ann_dir = self.data_root / "Annotations"
        split_file = self.data_root / "ImageSets" / "Main" / "test.txt"
        
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        
        self.samples = []
        for img_id in self.ids:
            ann_path = self.ann_dir / f"{img_id}.xml"
            img_path = self.img_dir / f"{img_id}.jpg"
            tree = ET.parse(ann_path)
            root = tree.getroot()
            obj = root.find("object")
            label = int(obj.find("name").text)
            self.samples.append({"id": img_path.name, "img_path": str(img_path), "label": label})
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
        return tensors, s["label"], s["id"]

class DiveshDataset(Dataset):
    """Diveshzz external dataset (same as baseline)"""
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

class ThyroidForPretrainingDataset(Dataset):
    """Thyroid for Pretraining dataset (from Kaggle for Thyroid-Pretraining)"""
    def __init__(self, data_root: str):
        self.samples = []
        dataset_dir = os.path.join(data_root, "train")
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = 0 if "benign" in class_name.lower() else 1
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append({"img_path": os.path.join(root, file), "label": label})
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors = torch.stack([t(image=img)["image"] for t in TTA_TRANSFORMS])
        return tensors, s["label"], s["img_path"]

@torch.no_grad()
def get_predictions(model, loader, device, seed_idx):
    """Get predictions for all test items"""
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

def evaluate_ssl(predictions_dict, dataset_name):
    """Evaluate SSL-pretrained 5-seed ensemble"""
    ids = sorted(list(predictions_dict.keys()))
    y_true = np.array([predictions_dict[i]["label"] for i in ids])
    
    seed_tta_logits = []
    for s_idx in range(5):
        s_logits = [np.mean(predictions_dict[i]["seeds"][s_idx]) for i in ids]
        seed_tta_logits.append(np.array(s_logits))
    
    ensemble_logits = np.mean(seed_tta_logits, axis=0)
    y_pred_prob = expit(ensemble_logits)
    y_pred_class = (y_pred_prob >= THRESHOLD).astype(int)
    
    scale_ensemble_logits = []
    for scale_idx in range(5):
        scale_preds = [np.mean([predictions_dict[i]["seeds"][s][scale_idx] for s in range(5)]) for i in ids]
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
        "AUROC_0.70x": roc_auc_score(y_true, scale_ensemble_logits[0]),
        "AUROC_0.85x": roc_auc_score(y_true, scale_ensemble_logits[1]),
        "AUROC_1.00x": roc_auc_score(y_true, scale_ensemble_logits[2]),
        "AUROC_1.15x": roc_auc_score(y_true, scale_ensemble_logits[3]),
        "AUROC_1.30x": roc_auc_score(y_true, scale_ensemble_logits[4]),
    }
    
    seed_aurocs = [roc_auc_score(y_true, s_logits) for s_logits in seed_tta_logits]
    metrics["Seed_AUROCs"] = seed_aurocs
    metrics["Seed_Mean"] = np.mean(seed_aurocs)
    metrics["Seed_SD"] = np.std(seed_aurocs)
    metrics["Seed_Min"] = np.min(seed_aurocs)
    metrics["Seed_Max"] = np.max(seed_aurocs)
    metrics["Seed_Median"] = np.median(seed_aurocs)
    
    return metrics, y_true, y_pred_prob

def generate_outputs(predictions_dict, metrics, dataset_name, y_true, y_pred_prob):
    """Generate output files for SSL evaluation"""
    fig_dir = FIG_DIR / dataset_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {metrics['AUROC']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"SSL Pretraining {dataset_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(fig_dir / f"{dataset_name.lower()}_ROC.png", bbox_inches='tight')
    plt.close()
    
    # PR
    p, r, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(r, p, label=f"PR-AUC = {metrics['PR-AUC']:.4f}")
    plt.title(f"SSL Pretraining {dataset_name} PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(fig_dir / f"{dataset_name.lower()}_PR.png", bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f"SSL Pretraining {dataset_name} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(fig_dir / f"{dataset_name.lower()}_CM.png", bbox_inches='tight')
    plt.close()
    
    # CSV Predictions
    ids = sorted(list(predictions_dict.keys()))
    with open(OUTPUT_DIR / f"{dataset_name.lower()}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "true_label"]
        for s in range(5):
            header.extend([f"seed{s}_0.70x", f"seed{s}_0.85x", f"seed{s}_1.00x", f"seed{s}_1.15x", f"seed{s}_1.30x", f"seed{s}_TTA"])
        header.extend(["ensemble_TTA_logit", "ensemble_prob", "predicted_class"])
        writer.writerow(header)
        for i in ids:
            row = [i, predictions_dict[i]["label"]]
            seed_tta_logits = []
            for s in range(5):
                scales = predictions_dict[i]["seeds"][s]
                tta = np.mean(scales)
                seed_tta_logits.append(tta)
                row.extend([scales[0], scales[1], scales[2], scales[3], scales[4], tta])
            ens_logit = np.mean(seed_tta_logits)
            ens_prob = expit(ens_logit)
            pred_class = 1 if ens_prob >= THRESHOLD else 0
            row.extend([ens_logit, ens_prob, pred_class])
            writer.writerow(row)
    
    # Error Analysis
    for i, pred in predictions_dict.items():
        scale_ens = [np.mean([pred["seeds"][s][s_idx] for s in range(5)]) for s_idx in range(5)]
        pred["max_disagreement"] = max(scale_ens) - min(scale_ens)
        pred["prob"] = expit(np.mean([np.mean(pred["seeds"][s]) for s in range(5)]))
        pred["pred_class"] = 1 if pred["prob"] >= THRESHOLD else 0
    
    fps = sorted([(i, p) for i, p in predictions_dict.items() if p["label"]==0 and p["pred_class"]==1], key=lambda x: x[1]["prob"], reverse=True)
    fns = sorted([(i, p) for i, p in predictions_dict.items() if p["label"]==1 and p["pred_class"]==0], key=lambda x: x[1]["prob"])
    
    with open(OUTPUT_DIR / f"{dataset_name.lower()}_error_analysis.txt", "w") as f:
        f.write(f"Total False Positives: {len(fps)}\nTotal False Negatives: {len(fns)}\n\n")
        f.write("--- Top False Positives ---\n")
        for i, p in fps[:20]: f.write(f"ID: {i} | Prob: {p['prob']:.4f} | Max TTA Disagreement: {p['max_disagreement']:.4f}\n")
        f.write("\n--- Top False Negatives ---\n")
        for i, p in fns[:20]: f.write(f"ID: {i} | Prob: {p['prob']:.4f} | Max TTA Disagreement: {p['max_disagreement']:.4f}\n")
    
    # Evaluation Report
    report = f"# SSL Pretraining {dataset_name} Evaluation\n\n"
    for k, v in metrics.items():
        if isinstance(v, float): report += f"**{k}:** {v:.4f}\n"
        elif isinstance(v, list): report += f"**{k}:** {[round(x, 4) for x in v]}\n"
        else: report += f"**{k}:** {v}\n"
    
    # Comparison with baseline
    baseline_auroc = BASELINE_INTERNAL_AUROC
    if dataset_name == "Diveshzz":
        baseline_auroc = BASELINE_DIVESHZZ_AUROC
    elif dataset_name == "ThyroidForPretraining":
        baseline_auroc = BASELINE_THYROID_FOR_PRETRAINING_AUROC
    
    report += f"\n### Comparison with Baseline (A4S1V2, ImageNet-pretrained)\n"
    report += f"Baseline AUROC: {baseline_auroc:.4f}\n"
    report += f"SSL AUROC: {metrics['AUROC']:.4f}\n"
    report += f"Difference: {metrics['AUROC'] - baseline_auroc:.4f}\n"
    
    with open(OUTPUT_DIR / f"{dataset_name.lower()}_report.md", "w") as f: f.write(report)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL-pretrained 5-seed ensemble")
    parser.add_argument("--datasets", nargs="+", default=["TN5000", "Diveshzz", "ThyroidForPretraining"],
                       help="Datasets to evaluate: TN5000, Diveshzz, ThyroidForPretraining")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"SSL evaluation of 5-seed ensemble: {args.datasets}")
    
    # Load all 5 seeds' checkpoints
    checkpoints = []
    for seed in range(5):
        if seed == 1:
            ckpt_path = Path("ssl_pretraining_experiment/supervised/seed1/best.pt")
        else:
            ckpt_path = Path(f"ssl_pretraining_experiment/supervised/seed{seed}/final_last.pt")
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        checkpoints.append((seed, ckpt_path))
    
    print(f"Loaded {len(checkpoints)} checkpoints: {[f'seed{s}' for s,_ in checkpoints]}")
    
    # Prepare datasets
    all_results = {}
    
    # Evaluate TN5000
    if "TN5000" in args.datasets:
        print("\n" + "="*50)
        print("EVALUATION: TN5000 (Internal)")
        print("="*50)
        
        ds = TN5000TestDataset("data_raw/TN5000_forReview")
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=min(2, os.cpu_count() or 1) if torch.cuda.is_available() else 0)
        
        all_preds = defaultdict(lambda: {"label": None, "seeds": []})
        
        for seed, ckpt_path in checkpoints:
            model = MultiLevelSwin(dropout=0.0).to(device)
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            
            preds = get_predictions(model, loader, device, seed_idx=seed)
            
            # Store per-seed predictions
            for obj_id, data in preds.items():
                all_preds[obj_id]["label"] = data["label"]
                all_preds[obj_id]["seeds"].append(data["scale_logits"])
            
            seed_y_true = [preds[obj_id]["label"] for obj_id in preds.keys()]
            seed_y_scores = [np.mean(preds[obj_id]["scale_logits"]) for obj_id in preds.keys()]
            print(f"Seed {seed} | TTA AUROC: {roc_auc_score(seed_y_true, seed_y_scores):.4f}")
        
        metrics, y_true, y_pred_prob = evaluate_ssl(all_preds, "TN5000")
        print(f"\nSSL TN5000 Results:")
        print(f"  N: {metrics['N']}")
        print(f"  AUROC: {metrics['AUROC']:.4f} (Baseline: {BASELINE_INTERNAL_AUROC:.4f})")
        print(f"  Difference: {metrics['AUROC'] - BASELINE_INTERNAL_AUROC:.4f}")
        
        generate_outputs(all_preds, metrics, "TN5000", y_true, y_pred_prob)
        all_results["TN5000"] = metrics
    
    # Evaluate Diveshzz
    if "Diveshzz" in args.datasets:
        print("\n" + "="*50)
        print("EVALUATION: Diveshzz (External)")
        print("="*50)
        
        ds_path = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
        ds = DiveshDataset(ds_path)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=min(2, os.cpu_count() or 1) if torch.cuda.is_available() else 0)
        
        all_preds = defaultdict(lambda: {"label": None, "seeds": []})
        
        for seed, ckpt_path in checkpoints:
            model = MultiLevelSwin(dropout=0.0).to(device)
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            
            preds = get_predictions(model, loader, device, seed_idx=seed)
            
            for obj_id, data in preds.items():
                all_preds[obj_id]["label"] = data["label"]
                all_preds[obj_id]["seeds"].append(data["scale_logits"])
            
            print(f"Seed {seed} | TTA AUROC: {roc_auc_score([all_preds[obj_id]['label'] for obj_id in all_preds.keys()],
                            [np.mean(all_preds[obj_id]['seeds'][seed]) for obj_id in all_preds.keys()]):.4f}")
        
        metrics, y_true, y_pred_prob = evaluate_ssl(all_preds, "Diveshzz")
        print(f"\nSSL Diveshzz Results:")
        print(f"  N: {metrics['N']}")
        print(f"  AUROC: {metrics['AUROC']:.4f} (Baseline: {BASELINE_DIVESHZZ_AUROC:.4f})")
        print(f"  Difference: {metrics['AUROC'] - BASELINE_DIVESHZZ_AUROC:.4f}")
        
        generate_outputs(all_preds, metrics, "Diveshzz", y_true, y_pred_prob)
        all_results["Diveshzz"] = metrics
    
    # Evaluate ThyroidForPretraining
    if "ThyroidForPretraining" in args.datasets:
        print("\n" + "="*50)
        print("EVALUATION: ThyroidForPretraining (External)")
        print("="*50)
        
        ds_path = kagglehub.dataset_download('tingzen/thyroid-for-pretraining')
        ds = ThyroidForPretrainingDataset(ds_path)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=min(2, os.cpu_count() or 1) if torch.cuda.is_available() else 0)
        
        all_preds = defaultdict(lambda: {"label": None, "seeds": []})
        
        for seed, ckpt_path in checkpoints:
            model = MultiLevelSwin(dropout=0.0).to(device)
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            
            preds = get_predictions(model, loader, device, seed_idx=seed)
            
            for obj_id, data in preds.items():
                all_preds[obj_id]["label"] = data["label"]
                all_preds[obj_id]["seeds"].append(data["scale_logits"])
            
            print(f"Seed {seed} | TTA AUROC: {roc_auc_score([all_preds[obj_id]['label'] for obj_id in all_preds.keys()],
                            [np.mean(all_preds[obj_id]['seeds'][seed]) for obj_id in all_preds.keys()]):.4f}")
        
        metrics, y_true, y_pred_prob = evaluate_ssl(all_preds, "ThyroidForPretraining")
        print(f"\nSSL ThyroidForPretraining Results:")
        print(f"  N: {metrics['N']}")
        print(f"  AUROC: {metrics['AUROC']:.4f} (Baseline: {BASELINE_THYROID_FOR_PRETRAINING_AUROC:.4f})")
        print(f"  Difference: {metrics['AUROC'] - BASELINE_THYROID_FOR_PRETRAINING_AUROC:.4f}")
        
        generate_outputs(all_preds, metrics, "ThyroidForPretraining", y_true, y_pred_prob)
        all_results["ThyroidForPretraining"] = metrics
    
    # Generate comparison table
    print("\n" + "="*50)
    print("SSL PRETRAINING EVALUATION COMPARISON")
    print("="*50)
    
    comparison_data = []
    for dataset, metrics in all_results.items():
        baseline_key = dataset
        if dataset == "TN5000":
            baseline_auroc = BASELINE_INTERNAL_AUROC
        elif dataset == "Diveshzz":
            baseline_auroc = BASELINE_DIVESHZZ_AUROC
        elif dataset == "ThyroidForPretraining":
            baseline_auroc = BASELINE_THYROID_FOR_PRETRAINING_AUROC
        
        comparison_data.append({
            "Dataset": dataset,
            "SSL_AUROC": f"{metrics['AUROC']:.4f}",
            "Baseline_AUROC": f"{baseline_auroc:.4f}",
            "Difference": f"{metrics['AUROC'] - baseline_auroc:+.4f}",
            "N": metrics['N'],
            "Seeds_AUROC": f"[{', '.join([f'{x:.4f}' for x in metrics['Seed_AUROCs']])}]",
            "Seed_Mean": f"{metrics['Seed_Mean']:.4f}",
            "Seed_SD": f"{metrics['Seed_SD']:.4f}"
        })
    
    # Save comparison CSV
    with open(OUTPUT_DIR / "comparison.csv", "w", newline="") as f:
        fieldnames = ["Dataset", "SSL_AUROC", "Baseline_AUROC", "Difference", "N", "Seeds_AUROC", "Seed_Mean", "Seed_SD"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_data)
    
    print("\nComparison saved to:", OUTPUT_DIR / "comparison.csv")
    
    # Print summary table
    print("\n" + "="*70)
    print("SSL PRETRAINING EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Dataset':<25} {'SSL_AUROC':>12} {'Baseline_AUROC':>17} {'Difference':>12} {'Seed_Mean':>12}")
    print("-"*70)
    for row in comparison_data:
        print(f"{row['Dataset']:<25} {row['SSL_AUROC']:>12} {row['Baseline_AUROC']:>17} {row['Difference']:>12} {row['Seed_Mean']:>12}")
    
    print("\n" + "="*70)
    print("SSL PRETRAINING EVALUATION COMPLETE")
    print("="*70)
    print(f"All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ssl_evaluation.py --datasets TN5000 Diveshzz ThyroidForPretraining")
        print("      or: python ssl_evaluation.py  # evaluates all datasets by default")
        sys.exit(1)
    main()
