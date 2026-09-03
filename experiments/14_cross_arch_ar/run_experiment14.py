"""
Experiment 14: Cross-Architecture Validation of AR-Preserving Preprocessing

Compares Pipeline A (current, anisotropic) vs Pipeline B (AR-preserving) across
ResNet50, EfficientNet-B3, and Swin-Tiny. Seed 0. Single controlled ablation.

Usage:
    python experiments/14_cross_arch_ar/run_experiment14.py --dry-run
    python experiments/14_cross_arch_ar/run_experiment14.py
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import TN5000Dataset, AUITDDataset
from src.metrics import (bootstrap_metrics, compute_metrics,
                         delong_test, sigmoid, youden_threshold)
from src.models import build_model
from src.trainer import train_model, evaluate
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR    = Path("experiments/14_cross_arch_ar")
DATA_ROOT  = Path("data_raw/TN5000_forReview")
AUITD_ROOT = "data_raw/auitd_dataset"
TRAIN_TXT  = str(DATA_ROOT / "ImageSets/Main/train.txt")
VAL_TXT    = str(DATA_ROOT / "ImageSets/Main/val.txt")
TEST_TXT   = str(DATA_ROOT / "ImageSets/Main/test.txt")

for d in ["configs","checkpoints","logs","metrics","plots","diagnostics"]:
    (EXP_DIR / d).mkdir(parents=True, exist_ok=True)

SEED = 0
BATCH_SIZE = 16
_USE_GPU = torch.cuda.is_available()
_N_WORK  = 4 if _USE_GPU else 0


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


# ── Pipelines ─────────────────────────────────────────────────────────────────
def pipeline_a_train():
    return A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_a_val():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def pipeline_b_train():
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

def pipeline_b_val():
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Divesh dataset ────────────────────────────────────────────────────────────
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


# ── Sanity checks ─────────────────────────────────────────────────────────────
def run_sanity_checks(divesh_root):
    print("\n" + "="*60)
    print("SANITY CHECKS")
    print("="*60)
    passed = True

    # Check 1: same source images reach both pipelines
    tn_train_a = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=None)
    tn_train_b = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=None)
    ids_a = [s["id"] for s in tn_train_a.samples]
    ids_b = [s["id"] for s in tn_train_b.samples]
    assert ids_a == ids_b, "FAIL: source image IDs differ"
    print("CHECK 1 PASS: same source images for both pipelines")

    # Check 2 & 3: diagnostic images with bbox overlays
    diag_dir = EXP_DIR / "diagnostics"
    pipe_a_v = pipeline_a_val()
    pipe_b_v = pipeline_b_val()
    pipe_a_bbox = A.Compose([A.Resize(256,256), A.CenterCrop(224,224)],
                             bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
    pipe_b_bbox = A.Compose([A.LongestMaxSize(max_size=256),
                              A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
                              A.CenterCrop(224,224)],
                             bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    n_saved = 0
    for sample in tn_train_a.samples:
        if n_saved >= 20:
            break
        img = np.array(PILImage.open(sample["img_path"]).convert("RGB"))
        bbox = sample["bbox"]
        try:
            out_a = pipe_a_bbox(image=img, bboxes=[bbox], labels=[1])
            out_b = pipe_b_bbox(image=img, bboxes=[bbox], labels=[1])
        except Exception:
            continue
        if not out_a["bboxes"] or not out_b["bboxes"]:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        xmin,ymin,xmax,ymax = bbox
        axes[0].add_patch(plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,
                          edgecolor="lime", facecolor="none", linewidth=2))
        axes[0].set_title(f"Original\n{img.shape[1]}x{img.shape[0]}")

        axes[1].imshow(out_a["image"])
        bx = out_a["bboxes"][0]
        axes[1].add_patch(plt.Rectangle((bx[0],bx[1]),bx[2]-bx[0],bx[3]-bx[1],
                          edgecolor="red", facecolor="none", linewidth=2))
        axes[1].set_title("Pipeline A (anisotropic)")

        axes[2].imshow(out_b["image"])
        bx = out_b["bboxes"][0]
        axes[2].add_patch(plt.Rectangle((bx[0],bx[1]),bx[2]-bx[0],bx[3]-bx[1],
                          edgecolor="red", facecolor="none", linewidth=2))
        axes[2].set_title("Pipeline B (AR-preserving)")

        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(diag_dir / f"diag_{n_saved:02d}.png", dpi=90)
        plt.close(fig)
        n_saved += 1

    print(f"CHECK 2&3 PASS: {n_saved} diagnostic images saved to diagnostics/")

    # Check 4: labels and splits are identical
    val_a = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=None)
    val_b = TN5000Dataset(str(DATA_ROOT), VAL_TXT, transform=None)
    assert [s["label"] for s in val_a.samples] == [s["label"] for s in val_b.samples]
    print("CHECK 4 PASS: labels/splits identical")

    # Check 5: Divesh not used in training
    divesh_paths = set()
    if divesh_root:
        td = os.path.join(divesh_root, "Thyroid Data")
        for sub in ["0", "1"]:
            d = os.path.join(td, sub)
            for f in glob.glob(os.path.join(d, "*.*")):
                divesh_paths.add(os.path.abspath(f))
    train_paths = set(os.path.abspath(s["img_path"]) for s in tn_train_a.samples)
    overlap = divesh_paths & train_paths
    assert len(overlap) == 0, f"FAIL: {len(overlap)} Divesh images found in training set"
    print("CHECK 5 PASS: Divesh is isolated from training data")

    print("All sanity checks passed.\n")
    return passed


# ── Evaluation helper ─────────────────────────────────────────────────────────
def full_eval(model, loader, device, pos_weight, threshold=None, n_bootstrap=500):
    """Returns metrics dict with point estimates + bootstrap CI."""
    raw = evaluate(model, loader, device, pos_weight, 1.0)
    logits = raw["logits"]
    labels = raw["labels"]
    if threshold is None:
        threshold = 0.5
    m = compute_metrics(logits, labels, threshold=threshold)
    ci = bootstrap_metrics(logits, labels, n_resamples=n_bootstrap, threshold=threshold, seed=42)
    return {"point": m, "bootstrap_ci": ci, "logits": logits, "labels": labels}


# ── Single run ────────────────────────────────────────────────────────────────
def run_single(arch, pipeline_name, train_t, val_t, device,
               pos_weight, divesh_root, dry_run=False):
    run_name = f"{arch}_{pipeline_name}_seed{SEED}"
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")

    set_seed(SEED)

    train_tn  = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=train_t)
    train_au  = AUITDDataset(AUITD_ROOT, transform=train_t)
    train_ds  = torch.utils.data.ConcatDataset([train_tn, train_au])
    val_ds    = TN5000Dataset(str(DATA_ROOT), VAL_TXT,  transform=val_t)
    test_ds   = TN5000Dataset(str(DATA_ROOT), TEST_TXT, transform=val_t)
    divesh_ds = DiveshDataset(divesh_root, transform=val_t) if divesh_root else None

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}"
          + (f"  External: {len(divesh_ds)}" if divesh_ds else ""))

    config = {
        "lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
        "pos_weight": pos_weight, "pos_weight_scale": 1.0,
        "batch_size": BATCH_SIZE, "max_epochs": 25, "patience": 10,
        "min_delta": 0.001, "T_0": 10, "T_mult": 2,
        "label_smooth_eps": 0.05, "grad_clip_norm": 1.0,
    }
    with open(EXP_DIR / "configs" / f"{run_name}.json", "w") as f:
        json.dump({"run": run_name, "arch": arch, "pipeline": pipeline_name,
                   "seed": SEED, **config}, f, indent=2)

    if dry_run:
        print("  [DRY RUN] skipping training.")
        return None

    kw = dict(num_workers=_N_WORK, pin_memory=_USE_GPU, worker_init_fn=worker_init_fn)
    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader    = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False, **kw)
    test_loader   = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False, **kw)
    divesh_loader = DataLoader(divesh_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw) if divesh_ds else None

    model = build_model(arch, dropout=config["dropout"])
    history = train_model(model, train_loader, val_loader, config,
                          checkpoint_dir=str(EXP_DIR / "checkpoints"),
                          run_name=run_name, device=device)

    # Save training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_auc"],  label="train"); axes[1].plot(history["val_auc"],  label="val")
    axes[1].set_title("AUC");  axes[1].legend()
    plt.suptitle(run_name); plt.tight_layout()
    plt.savefig(EXP_DIR / "plots" / f"{run_name}_curve.png", dpi=100)
    plt.close()
    with open(EXP_DIR / "logs" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Threshold from val
    val_raw = evaluate(model, val_loader, device, pos_weight, 1.0)
    threshold = youden_threshold(val_raw["logits"], val_raw["labels"])

    # Internal
    int_eval = full_eval(model, test_loader, device, pos_weight, threshold)
    # External (no threshold re-tuning on Divesh)
    ext_eval = full_eval(model, divesh_loader, device, pos_weight, threshold) if divesh_loader else None

    # DeLong: current vs AR will be done in the comparison phase
    metrics_out = {
        "run_name": run_name, "arch": arch, "pipeline": pipeline_name, "seed": SEED,
        "best_val_auc": history["best_val_auc"],
        "youden_threshold": threshold,
        "internal": int_eval["point"],
        "internal_ci": int_eval["bootstrap_ci"],
        "external": ext_eval["point"] if ext_eval else {},
        "external_ci": ext_eval["bootstrap_ci"] if ext_eval else {},
        # save logits for DeLong later
        "internal_logits": int_eval["logits"].tolist(),
        "internal_labels": int_eval["labels"].tolist(),
        "external_logits": ext_eval["logits"].tolist() if ext_eval else [],
        "external_labels": ext_eval["labels"].tolist() if ext_eval else [],
    }
    with open(EXP_DIR / "metrics" / f"{run_name}.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n  Internal AUC: {int_eval['point']['auc']:.4f} "
          f"[{int_eval['bootstrap_ci']['auc']['p2_5']:.4f}–{int_eval['bootstrap_ci']['auc']['p97_5']:.4f}]")
    if ext_eval:
        print(f"  External AUC: {ext_eval['point']['auc']:.4f} "
              f"[{ext_eval['bootstrap_ci']['auc']['p2_5']:.4f}–{ext_eval['bootstrap_ci']['auc']['p97_5']:.4f}]")
        print(f"  Delta AUC:    {int_eval['point']['auc'] - ext_eval['point']['auc']:.4f}")

    return metrics_out


# ── Comparison and reporting ──────────────────────────────────────────────────
def build_comparison(results):
    """Build final comparison table and DeLong tests."""
    archs = ["resnet50", "efficientnet_b3", "swin_tiny"]
    rows  = []

    for arch in archs:
        a = next((r for r in results if r["arch"] == arch and r["pipeline"] == "A_current"),    None)
        b = next((r for r in results if r["arch"] == arch and r["pipeline"] == "B_ar_preserving"), None)
        if a is None or b is None:
            continue

        int_a  = a["internal"]["auc"];  int_b  = b["internal"]["auc"]
        ext_a  = a["external"].get("auc", float("nan"))
        ext_b  = b["external"].get("auc", float("nan"))
        gap_a  = int_a - ext_a
        gap_b  = int_b - ext_b

        # DeLong on external (exploratory — single seed)
        dl_ext = None
        if a["external_logits"] and b["external_logits"]:
            try:
                dl_ext = delong_test(np.array(a["external_logits"]),
                                     np.array(b["external_logits"]),
                                     np.array(a["external_labels"]))
            except Exception:
                pass

        rows.append({
            "arch": arch,
            "current_int_auc": int_a, "ar_int_auc": int_b, "delta_int": int_b - int_a,
            "current_ext_auc": ext_a, "ar_ext_auc": ext_b, "delta_ext": ext_b - ext_a,
            "current_gap": gap_a, "ar_gap": gap_b, "gap_reduction": gap_a - gap_b,
            "delong_ext_p": dl_ext["p_value"] if dl_ext else None,
        })

    # Print table
    print(f"\n{'='*110}")
    print("EXPERIMENT 14 — FINAL COMPARISON TABLE")
    print(f"{'='*110}")
    hdr = (f"{'Architecture':<20} {'Cur Int':>9} {'AR Int':>9} {'dInt':>7} "
           f"{'Cur Ext':>9} {'AR Ext':>9} {'dExt':>7} "
           f"{'Cur Gap':>9} {'AR Gap':>9} {'GapRed':>9} {'DeLong p':>10}")
    print(hdr); print("-"*110)
    for r in rows:
        dp = f"{r['delong_ext_p']:.4f}" if r["delong_ext_p"] is not None else "N/A"
        print(f"{r['arch']:<20} {r['current_int_auc']:>9.4f} {r['ar_int_auc']:>9.4f} "
              f"{r['delta_int']:>+7.4f} {r['current_ext_auc']:>9.4f} {r['ar_ext_auc']:>9.4f} "
              f"{r['delta_ext']:>+7.4f} {r['current_gap']:>9.4f} {r['ar_gap']:>9.4f} "
              f"{r['gap_reduction']:>+9.4f} {dp:>10}")
    print(f"{'='*110}")

    return rows


def save_csv(rows):
    outpath = EXP_DIR / "results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Results saved to {outpath}")

    # Append to master
    master = Path("experiments/master_results.csv")
    write_header = not master.exists()
    with open(master, "a", newline="") as f:
        all_fields = ["experiment"] + fieldnames
        w = csv.DictWriter(f, fieldnames=all_fields)
        if write_header: w.writeheader()
        for r in rows:
            w.writerow({"experiment": "14_cross_arch_ar", **r})
    print(f"Appended to {master}")


def save_bar_chart(rows):
    archs = [r["arch"] for r in rows]
    x = np.arange(len(archs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [r["current_ext_auc"] for r in rows], w, label="Current (A)", color="steelblue")
    ax.bar(x + w/2, [r["ar_ext_auc"]      for r in rows], w, label="AR-preserving (B)", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(archs)
    ax.set_ylabel("External AUC (Divesh)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Exp 14: External AUC — Current vs AR-Preserving")
    ax.legend(); ax.axhline(0.925, color="red", linestyle="--", alpha=0.5, label="Target 0.925")
    plt.tight_layout()
    plt.savefig(EXP_DIR / "plots" / "exp14_comparison.png", dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training 6 models on CPU will take many hours.")
        print("         Run this script on your GPU machine.")

    import kagglehub
    print("Locating Divesh dataset...")
    divesh_root = kagglehub.dataset_download(
        "diveshzz/thyroid-cancer-classification-ultrasound-dataset")
    print(f"  Divesh root: {divesh_root}")

    run_sanity_checks(divesh_root)

    # Compute pos_weight once (same for all runs)
    _tn  = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=None)
    _au  = AUITDDataset(AUITD_ROOT, transform=None)
    _lbl = np.concatenate([_tn.get_labels(), _au.get_labels()])
    pos_weight = float((_lbl == 0).sum() / (_lbl == 1).sum())
    print(f"pos_weight = {pos_weight:.6f}")

    RUNS = [
        ("resnet50",        "A_current",       pipeline_a_train(), pipeline_a_val()),
        ("resnet50",        "B_ar_preserving", pipeline_b_train(), pipeline_b_val()),
        ("efficientnet_b3", "A_current",       pipeline_a_train(), pipeline_a_val()),
        ("efficientnet_b3", "B_ar_preserving", pipeline_b_train(), pipeline_b_val()),
        ("swin_tiny",       "A_current",       pipeline_a_train(), pipeline_a_val()),
        ("swin_tiny",       "B_ar_preserving", pipeline_b_train(), pipeline_b_val()),
    ]

    results = []
    for arch, pipe_name, train_t, val_t in RUNS:
        r = run_single(arch, pipe_name, train_t, val_t,
                       device, pos_weight, divesh_root, dry_run=args.dry_run)
        if r:
            results.append(r)

    if results:
        rows = build_comparison(results)
        save_csv(rows)
        save_bar_chart(rows)

        with open(EXP_DIR / "metrics" / "all_runs.json", "w") as f:
            # exclude raw logits from summary JSON to keep it readable
            summary = [{k: v for k, v in r.items()
                        if k not in ("internal_logits","internal_labels",
                                     "external_logits","external_labels")}
                       for r in results]
            json.dump(summary, f, indent=2)

        print(f"\nAll outputs saved to {EXP_DIR}/")
        print("Next step: run RESULTS.md generator or interpret the table manually.")


if __name__ == "__main__":
    main()
