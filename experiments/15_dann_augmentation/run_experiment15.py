"""
Experiment 15: Ultrasound Appearance Augmentation + Domain-Adversarial Learning

Compares:
M0: AR Baseline (Swin-Tiny, AR-preserving preprocessing)
M1: AR + Ultrasound Appearance Augmentation
M2: AR + Domain-Adversarial Learning (DANN)
M3: AR + Appearance Aug + DANN

Validates on TN5000 test and Divesh (held out).
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
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import TN5000Dataset, AUITDDataset
from src.metrics import bootstrap_metrics, compute_metrics, youden_threshold
from src.models import build_model
from src.trainer import EarlyStopping, build_optimizer_and_scheduler
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── TPU / GPU / CPU device setup ──────────────────────────────────────────────
try:
    import torch_xla.core.xla_model as xm
    _USE_TPU = True
    _USE_GPU = False
except ImportError:
    xm = None
    _USE_TPU = False
    _USE_GPU = torch.cuda.is_available()

def get_device():
    if _USE_TPU:
        return xm.xla_device()
    elif _USE_GPU:
        return torch.device("cuda")
    return torch.device("cpu")

def optimizer_step(optimizer):
    """Use XLA optimizer step on TPU, standard step otherwise."""
    if _USE_TPU:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()

def mark_step():
    """XLA requires explicit graph execution after each batch on TPU."""
    if _USE_TPU:
        xm.mark_step()

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR    = Path("experiments/15_dann_augmentation")
DATA_ROOT  = Path("data_raw/TN5000_forReview")
AUITD_ROOT = "data_raw/auitd_dataset"
TRAIN_TXT  = str(DATA_ROOT / "ImageSets/Main/train.txt")
VAL_TXT    = str(DATA_ROOT / "ImageSets/Main/val.txt")
TEST_TXT   = str(DATA_ROOT / "ImageSets/Main/test.txt")

for d in ["configs","checkpoints","logs","metrics","diagnostics"]:
    (EXP_DIR / d).mkdir(parents=True, exist_ok=True)

SEED = 0
BATCH_SIZE = 16
_N_WORK  = 4 if (_USE_GPU or _USE_TPU) else 0


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


# ── Augmentations ─────────────────────────────────────────────────────────────
def ultrasound_appearance_augmentation(p=1.0):
    """Mild ultrasound-specific appearance augmentations."""
    return A.Compose([
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=0.3), # speckle-like
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=1.0)
        ], p=0.3),
        A.GaussNoise(p=0.3),
    ], p=p)

def pipeline_m0_train():
    """Baseline AR-preserving train (matches Exp 14 Pipeline B)."""
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

def pipeline_m1_train():
    """AR-preserving + Ultrasound Appearance Augmentation."""
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        ultrasound_appearance_augmentation(p=1.0), # Replaces generic ColorJitter + GaussianBlur
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.RandomCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_val():
    """Shared AR-preserving validation."""
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Datasets ──────────────────────────────────────────────────────────────────
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

class DomainDatasetWrapper(Dataset):
    """Wraps a dataset to also return a domain label (0 or 1)."""
    def __init__(self, dataset, domain_label):
        self.dataset = dataset
        self.domain_label = domain_label
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, self.domain_label


# ── Domain-Adversarial Architecture ───────────────────────────────────────────
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grl(x, alpha):
    return GradientReversalLayer.apply(x, alpha)

class DomainAdversarialSwin(nn.Module):
    """Wraps SwinClassifier to add a domain head with GRL."""
    def __init__(self, base_model, num_domains=2, dropout=0.3):
        super().__init__()
        self._base_model = base_model   # keep reference for freeze_epoch delegation
        self.backbone = base_model.backbone
        self.head = base_model.head
        self.domain_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_domains)
        )
        self.use_dann = False

    def freeze_epoch(self, epoch: int):
        """Delegate staged unfreezing to the underlying base model."""
        self._base_model.freeze_epoch(epoch)

    def get_param_groups(self, lr_head: float, lr_backbone: float):
        """Discriminative LRs: backbone at lr_backbone, head+domain_head at lr_head."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters()) + list(self.domain_head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups

    def forward(self, x, alpha=None):
        features = self.backbone(x)
        logits = self.head(features)

        if self.use_dann and self.training and alpha is not None:
            reverse_features = grl(features, alpha)
            domain_logits = self.domain_head(reverse_features)
            return logits, domain_logits, features
        return logits, features


# ── Training Engine ───────────────────────────────────────────────────────────
def get_domain_sample_weights(domains, labels):
    """
    Computes sample-level weights for the domain loss so that TN5000-benign,
    TN5000-malignant, AUITD-benign, and AUITD-malignant all contribute equally.
    This removes disease prevalence as a shortcut for the domain classifier.
    """
    weights = np.zeros(len(domains))
    for d in [0, 1]:
        for c in [0, 1]:
            mask = (domains == d) & (labels == c)
            count = mask.sum()
            if count > 0:
                weights[mask] = 1.0 / count
    weights = weights / weights.mean() # normalize to mean 1
    return torch.tensor(weights, dtype=torch.float32)

def evaluate_dann(model, loader, device, pos_weight):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            # Loader might return (img, label) or (img, label, domain)
            images = batch[0].to(device)
            labels = batch[1].to(device).float()
            logits, _ = model(images)
            logits = logits.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(pos_weight).to(device))
            total_loss += loss.item() * len(labels)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    auc = roc_auc_score(all_labels, all_logits)
    return {"loss": total_loss / len(all_labels), "auc": auc, "logits": np.array(all_logits), "labels": np.array(all_labels)}

def train_dann_model(
    model: DomainAdversarialSwin, train_loader: DataLoader, val_loader: DataLoader,
    config: dict, sample_domain_weights: torch.Tensor, run_name: str, device: torch.device
):
    model = model.to(device)
    sample_domain_weights = sample_domain_weights.to(device)
    
    lr_head = config.get("lr_head", 3e-4)
    lr_backbone = lr_head * 0.1
    weight_decay = config.get("weight_decay", 1e-4)
    pos_weight = torch.tensor(config.get("pos_weight", 0.4)).to(device)
    max_epochs = config.get("max_epochs", 25)
    
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr_head, lr_backbone, weight_decay, config["T_0"], config["T_mult"], -1
    )

    early_stopping = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    total_steps = max_epochs * len(train_loader)
    current_step = 0
    
    _prev_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    for epoch in range(1, max_epochs + 1):
        # --- Staged unfreezing (mirrors src.trainer.train_model) ---
        model.freeze_epoch(epoch)
        _curr_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        if _curr_trainable != _prev_trainable:
            print(f"  [unfreeze] Trainable params {_prev_trainable}→{_curr_trainable}. Rebuilding optimizer.")
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, lr_head, lr_backbone, weight_decay, config["T_0"], config["T_mult"], last_epoch=epoch - 1
            )
            _prev_trainable = _curr_trainable

        model.train()
        total_loss, total_cls_loss, total_dom_loss = 0.0, 0.0, 0.0
        all_logits, all_labels = [], []
        try:
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        except ImportError:
            pbar = train_loader

        for idx, (images, labels, domains, global_indices) in enumerate(pbar):
            images, labels, domains = images.to(device), labels.to(device).float(), domains.to(device).long()

            p = float(current_step) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1  # GRL schedule
            current_step += 1

            optimizer.zero_grad()

            # Label smoothing (matches src.trainer)
            smooth_eps = config.get("label_smooth_eps", 0.05)

            if model.use_dann:
                logits, domain_logits, _ = model(images, alpha=alpha)
                logits = logits.squeeze(1)

                labels_smooth = labels * (1.0 - smooth_eps) + 0.5 * smooth_eps
                loss_cls = F.binary_cross_entropy_with_logits(logits, labels_smooth, pos_weight=pos_weight)

                batch_weights = sample_domain_weights[global_indices]
                loss_dom_per_sample = F.cross_entropy(domain_logits, domains, reduction='none')
                loss_dom = (loss_dom_per_sample * batch_weights).mean()

                loss = loss_cls + loss_dom
            else:
                logits, _ = model(images)
                logits = logits.squeeze(1)
                labels_smooth = labels * (1.0 - smooth_eps) + 0.5 * smooth_eps
                loss_cls = F.binary_cross_entropy_with_logits(logits, labels_smooth, pos_weight=pos_weight)
                loss_dom = torch.tensor(0.0)
                loss = loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer_step(optimizer)
            mark_step()  # required on TPU; no-op on GPU/CPU

            total_loss += loss.item() * len(labels)
            total_cls_loss += loss_cls.item() * len(labels)
            total_dom_loss += loss_dom.item() * len(labels)
            all_logits.extend(logits.detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
        scheduler.step()
        val_metrics = evaluate_dann(model, val_loader, device, pos_weight.item())
        
        train_auc = roc_auc_score(all_labels, all_logits)
        train_loss = total_loss / len(all_labels)
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        
        print(f"[{run_name}] Epoch {epoch:02d} | Train AUC: {train_auc:.4f} "
              f"(Cls: {total_cls_loss/len(all_labels):.4f}, Dom: {total_dom_loss/len(all_labels):.4f}) | "
              f"Val AUC: {val_metrics['auc']:.4f}")
        
        should_stop = early_stopping(val_metrics["auc"], model)
        if early_stopping.counter == 0:
            torch.save(model.state_dict(), EXP_DIR / "checkpoints" / f"{run_name}_best.pt")
            
        if should_stop:
            print(f"[{run_name}] Early stopping at epoch {epoch}.")
            break
            
    early_stopping.restore_best(model)
    history["best_val_auc"] = early_stopping.best_score
    return history


# ── Feature Extraction for Post-Hoc Analysis ──────────────────────────────────
def extract_features(model, loader, device):
    model.eval()
    features, labels, domains = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            lbl = batch[1]
            dom = batch[2] if len(batch) > 2 else torch.zeros(len(lbl))
            _, feats = model(images)
            # flatten features
            features.append(feats.view(feats.size(0), -1).cpu().numpy())
            labels.append(lbl.numpy())
            domains.append(dom.numpy())
    return np.concatenate(features), np.concatenate(labels), np.concatenate(domains)

def post_hoc_domain_classification(model, loader, device):
    """Trains a logistic regression to distinguish domains from frozen features."""
    features, labels, domains = extract_features(model, loader, device)
    # Balance classes for the LR by setting class_weight='balanced'
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(features, domains)
    preds = lr.predict(features)
    probs = lr.predict_proba(features)[:, 1]
    acc = (preds == domains).mean()
    auc = roc_auc_score(domains, probs)
    return {"accuracy": float(acc), "auc": float(auc)}


# ── Evaluation helper ─────────────────────────────────────────────────────────
def full_eval(model, loader, device, pos_weight, threshold=0.5):
    raw = evaluate_dann(model, loader, device, pos_weight)
    m = compute_metrics(raw["logits"], raw["labels"], threshold=threshold)
    return {"point": m, "logits": raw["logits"], "labels": raw["labels"]}


# ── Single run ────────────────────────────────────────────────────────────────
def run_single(run_name, use_aug, use_dann, divesh_root, device, dry_run=False):
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")
    set_seed(SEED)
    
    train_t = pipeline_m1_train() if use_aug else pipeline_m0_train()
    val_t = pipeline_val()

    # Create datasets
    train_tn = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t)
    train_au = AUITDDataset(AUITD_ROOT, transform=train_t)
    train_ds = torch.utils.data.ConcatDataset([
        DomainDatasetWrapper(train_tn, 0),
        DomainDatasetWrapper(train_au, 1)
    ])
    
    # We need to return global indices to fetch sample weights
    class IndexedDataset(Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx): return self.ds[idx] + (idx,)
    
    train_ds_idx = IndexedDataset(train_ds)
    
    val_ds = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=val_t)
    test_ds = TN5000Dataset(str(DATA_ROOT), TEST_TXT, transform=val_t)
    divesh_ds = DiveshDataset(divesh_root, transform=val_t) if divesh_root else None

    # Calculate domain sample weights
    all_labels = np.concatenate([train_tn.get_labels(), train_au.get_labels()])
    all_domains = np.concatenate([np.zeros(len(train_tn)), np.ones(len(train_au))])
    sample_domain_weights = get_domain_sample_weights(all_domains, all_labels)
    pos_weight = float((all_labels == 0).sum() / (all_labels == 1).sum())

    config = {
        "lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
        "pos_weight": pos_weight, "batch_size": BATCH_SIZE, "max_epochs": 25,
        "patience": 10, "min_delta": 0.001, "T_0": 10, "T_mult": 2,
        "grad_clip_norm": 1.0, "label_smooth_eps": 0.05,
    }
    
    if dry_run:
        print("  [DRY RUN] skipping training.")
        return None

    # pin_memory only applies to CUDA; must be False on TPU
    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds_idx, batch_size=BATCH_SIZE, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)
    divesh_loader = DataLoader(divesh_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw) if divesh_ds else None

    # We also need a loader without augmentation for post-hoc feature extraction
    clean_tn = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=val_t)
    clean_au = AUITDDataset(AUITD_ROOT, transform=val_t)
    clean_ds = torch.utils.data.ConcatDataset([
        DomainDatasetWrapper(clean_tn, 0),
        DomainDatasetWrapper(clean_au, 1)
    ])
    feat_loader = DataLoader(clean_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw)

    # Build model
    base_model = build_model("swin_tiny", dropout=config["dropout"])
    model = DomainAdversarialSwin(base_model, num_domains=2, dropout=config["dropout"])
    model.use_dann = use_dann
    
    history = train_dann_model(model, train_loader, val_loader, config, sample_domain_weights, run_name, device)
    
    with open(EXP_DIR / "logs" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Eval
    val_raw = evaluate_dann(model, val_loader, device, pos_weight)
    threshold = youden_threshold(val_raw["logits"], val_raw["labels"])
    
    int_eval = full_eval(model, test_loader, device, pos_weight, threshold)
    ext_eval = full_eval(model, divesh_loader, device, pos_weight, threshold) if divesh_loader else None
    
    # Post-hoc domain classification
    domain_metrics = post_hoc_domain_classification(model, feat_loader, device)
    print(f"  Domain Separability AUC: {domain_metrics['auc']:.4f}")

    metrics_out = {
        "experiment": "15_dann_augmentation",
        "model": run_name,
        "internal_auc": int_eval["point"]["auc"],
        "internal_accuracy": int_eval["point"]["accuracy"],
        "internal_sensitivity": int_eval["point"]["sensitivity"],
        "internal_specificity": int_eval["point"]["specificity"],
        "internal_f1": int_eval["point"]["f1"],
        "external_auc": ext_eval["point"]["auc"] if ext_eval else None,
        "external_accuracy": ext_eval["point"]["accuracy"] if ext_eval else None,
        "external_sensitivity": ext_eval["point"]["sensitivity"] if ext_eval else None,
        "external_specificity": ext_eval["point"]["specificity"] if ext_eval else None,
        "external_f1": ext_eval["point"]["f1"] if ext_eval else None,
        "generalization_gap": int_eval["point"]["auc"] - ext_eval["point"]["auc"] if ext_eval else None,
        "domain_classifier_auc": domain_metrics["auc"],
        "domain_classifier_accuracy": domain_metrics["accuracy"],
        "seed": SEED
    }
    
    print(f"  Internal AUC: {int_eval['point']['auc']:.4f}")
    if ext_eval:
        print(f"  External AUC: {ext_eval['point']['auc']:.4f}")
        print(f"  Gap:          {metrics_out['generalization_gap']:.4f}")
        
    return metrics_out

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device} | TPU={_USE_TPU} | GPU={_USE_GPU}")
    
    import kagglehub
    divesh_root = kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")
    
    RUNS = [
        # (run_name, use_aug, use_dann)
        ("M0_AR_Baseline", False, False),
        ("M1_AR_Appearance", True, False),
        ("M2_AR_DANN", False, True),
        ("M3_AR_Both", True, True),
    ]
    
    results = []
    for name, use_aug, use_dann in RUNS:
        r = run_single(name, use_aug, use_dann, divesh_root, device, dry_run=args.dry_run)
        if r: results.append(r)
        
    if results:
        outpath = EXP_DIR / "results.csv"
        fieldnames = list(results[0].keys())
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(results)
        print(f"\nSaved results to {outpath}")
        
if __name__ == "__main__":
    main()
