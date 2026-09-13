"""Stage 2 retraining protocol for the selected thyroid-US configurations.

All output is deliberately contained in ``training_experiments``.  This module
never reads from or writes to ``outputs/final_model`` except through the shared
source-code imports; it trains fresh ImageNet-initialized models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import kagglehub
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import AUITDDataset, TN5000Dataset
from src.trainer import train_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from train import MultiLevelSwin

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_): return iterable

OUT = Path(__file__).resolve().parent
TN5000_ROOT = ROOT / "data_raw" / "TN5000_forReview"
AUITD_ROOT = ROOT / "data_raw" / "auitd_dataset"
SEEDS = (0, 1, 2, 3, 4)
THRESHOLD = 0.5912  # Prespecified threshold used by the factorial ablation.
# The GPU host reports a safe maximum of two worker processes. Exceeding that
# limit is a documented cause of DataLoader stalls in notebook environments.
NUM_WORKERS = min(2, os.cpu_count() or 1)

CONFIGS = {
    "A1S4V2": ("A1", [0.60, 0.75, 1.00, 1.15, 1.40]),
    "A4S4V2": ("A4", [0.60, 0.75, 1.00, 1.15, 1.40]),
    "A4S2V2": ("A4", [0.60, 0.80, 1.00, 1.20, 1.40]),
    "A1S2V2": ("A1", [0.60, 0.80, 1.00, 1.20, 1.40]),
    "A1S5V2": ("A1", [0.60, 0.85, 1.00, 1.25, 1.40]),
    "A4S1V2": ("A4", [0.70, 0.85, 1.00, 1.15, 1.30]),
    "A4S5V2": ("A4", [0.60, 0.85, 1.00, 1.25, 1.40]),
    "A1S1V2": ("A1", [0.70, 0.85, 1.00, 1.15, 1.30]),
    "A4S4V1": ("A4", [0.75, 1.00, 1.15]),
    "A1S4V1": ("A1", [0.75, 1.00, 1.15]),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ar_definition(ar: str) -> str:
    return {
        "A1": "Longest side 256; pad only as needed for a 224 crop; crop 224.",
        "A4": "Longest side 256; pad canvas to 256x256; crop 224.",
    }[ar]


def _geometry(ar: str, crop_cls):
    if ar == "A1":
        return [A.LongestMaxSize(max_size=256),
                A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
                crop_cls(224, 224)]
    if ar == "A4":
        return [A.LongestMaxSize(max_size=256),
                A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
                crop_cls(224, 224)]
    raise ValueError(f"Unsupported selected AR setting: {ar}")


def make_train_transform(ar: str):
    return A.Compose([
        A.Rotate(limit=15, p=1.0), A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=1.0),
        *_geometry(ar, A.RandomCrop),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(),
    ])


def make_val_transform(ar: str):
    return A.Compose([*_geometry(ar, A.CenterCrop),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def make_tta_transform(ar: str, scale: float):
    # Exact factorial implementation: the scale modifies the proportional resize.
    max_size = round(256 * scale)
    if ar == "A1":
        geom = [A.LongestMaxSize(max_size=max_size),
                A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
                A.CenterCrop(224, 224)]
    elif ar == "A4":
        geom = [A.LongestMaxSize(max_size=max_size),
                A.PadIfNeeded(min_height=max(max_size, 256), min_width=max(max_size, 256), border_mode=0),
                A.CenterCrop(224, 224)]
    else:
        raise ValueError(ar)
    return A.Compose([*geom, A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class EvalDataset(Dataset):
    def __init__(self, samples, ar, scales):
        self.samples = samples
        self.transforms = [make_tta_transform(ar, s) for s in scales]
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        item = self.samples[index]
        image = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
        return torch.stack([t(image=image)["image"] for t in self.transforms]), item["label"], item["id"]


def tn5000_test_samples():
    ds = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/test.txt"))
    return [{"path": x["img_path"], "label": x["label"], "id": x["id"]} for x in ds.samples]


def divesh_samples(root: str):
    samples = []
    for label in (0, 1):
        directory = Path(root) / "Thyroid Data" / str(label)
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append({"path": str(path), "label": label, "id": f"{len(samples):05d}_{path.name}"})
    return samples


def pretrain_samples(root: str):
    samples = []
    for directory, _, files in os.walk(root):
        name = Path(directory).name.lower().strip()
        label = 0 if name in {"benign", "0", "normal"} else (1 if name in {"malignant", "1"} else None)
        if label is not None:
            for filename in sorted(files):
                if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    samples.append({"path": str(Path(directory) / filename), "label": label,
                                    "id": filename.split("_")[0]})
    return samples


@torch.no_grad()
def seed_logits(checkpoint: Path, dataset: Dataset, device: torch.device, description: str):
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=NUM_WORKERS if device.type == "cuda" else 0)
    model = MultiLevelSwin(dropout=0.0).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    result = {}
    for tensors, labels, ids in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        b, views, c, h, w = tensors.shape
        logits = model(tensors.to(device).reshape(b * views, c, h, w)).squeeze(1).reshape(b, views).cpu().float().numpy()
        for i, item_id in enumerate(ids):
            result[item_id] = (int(labels[i]), logits[i])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metrics(labels, logits):
    probs = 1 / (1 + np.exp(-np.asarray(logits)))
    labels = np.asarray(labels)
    pred = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"AUC": float(roc_auc_score(labels, probs)), "Accuracy": float(accuracy_score(labels, pred)),
            "F1": float(f1_score(labels, pred, zero_division=0)), "Sensitivity": float(recall_score(labels, pred, zero_division=0)),
            "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "PRAUC": float(average_precision_score(labels, probs)), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn), "N": int(len(labels))}


def evaluate_ensemble(config_dir: Path, ar: str, scales, device: torch.device):
    print("[EVALUATION] Locating external datasets (the first run may download them)...", flush=True)
    sources = {
        "Internal": (tn5000_test_samples(), False),
        "Diveshzz": (divesh_samples(kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")), False),
        "ThyroidPretrain": (pretrain_samples(kagglehub.dataset_download("tingzen/thyroid-for-pretraining")), True),
    }
    all_metrics, per_seed_internal = {}, []
    started = time.time()
    for name, (samples, patient_level) in sources.items():
        print(f"[EVALUATION] {name}: {len(samples)} images, {len(scales)} TTA views, 5 seeds.", flush=True)
        dataset = EvalDataset(samples, ar, scales)
        seed_predictions = [seed_logits(config_dir / f"seed{s}" / "best.pt", dataset, device,
                                        f"[EVALUATION] {name} seed{s}") for s in SEEDS]
        ids = sorted(seed_predictions[0])
        labels, ensemble_logits, per_scale = [], [], []
        for item_id in ids:
            labels.append(seed_predictions[0][item_id][0])
            scale_logits = np.mean([x[item_id][1] for x in seed_predictions], axis=0)
            per_scale.append(scale_logits)
            ensemble_logits.append(float(np.mean(scale_logits)))
        if name == "Internal":
            for preds in seed_predictions:
                per_seed_internal.append(float(roc_auc_score(labels, [np.mean(preds[i][1]) for i in ids])))
        if patient_level:
            grouped = defaultdict(lambda: {"label": None, "logits": [], "scales": []})
            for item_id, label, logit, scale_logit in zip(ids, labels, ensemble_logits, per_scale):
                grouped[item_id]["label"] = label
                grouped[item_id]["logits"].append(logit)
                grouped[item_id]["scales"].append(scale_logit)
            labels = [v["label"] for v in grouped.values()]
            ensemble_logits = [float(np.mean(v["logits"])) for v in grouped.values()]
            per_scale = [np.mean(v["scales"], axis=0) for v in grouped.values()]
        value = metrics(labels, ensemble_logits)
        value["PatientLevel"] = patient_level
        value["PatientCount"] = len(labels) if patient_level else None
        value["PerScaleAUC"] = [float(roc_auc_score(labels, np.asarray(per_scale)[:, i])) for i in range(len(scales))]
        all_metrics[name] = value
    elapsed = time.time() - started
    print(f"[EVALUATION] Complete in {elapsed / 60:.1f} minutes.", flush=True)
    return all_metrics, per_seed_internal, elapsed


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def update_status(base_dir: Path, config: str, **values) -> None:
    """Persist progress independently of transient notebook-cell output."""
    path = base_dir / config / "run_status.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    previous.update(values)
    previous["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, previous)
    print(f"[STATUS] {config}: {previous.get('state')} | seed={previous.get('active_seed', '-')}", flush=True)


def completed_seed_metadata(seed_dir: Path):
    """Return metadata only for a fully completed, non-sanity seed run."""
    metadata_path, checkpoint = seed_dir / "config.json", seed_dir / "best.pt"
    if metadata_path.is_file() and checkpoint.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("sanity", False):
            return metadata
    return None


def run_seed(config_name, ar, scales, seed, base_dir, sanity=False):
    seed_dir = base_dir / config_name / f"seed{seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {seed_dir}")
    seed_dir.mkdir(parents=True)
    update_status(base_dir, config_name, state="training", active_seed=seed,
                  completed_seeds=list(range(seed)), sanity=sanity)
    print(f"\n{'=' * 70}\n[RUN] {config_name} | seed {seed}/4 | AR={ar} | TTA={scales}\n{'=' * 70}", flush=True)
    set_seed(seed)
    train_set = ConcatDataset([TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/train.txt"), make_train_transform(ar)),
                               AUITDDataset(str(AUITD_ROOT), make_train_transform(ar))])
    val_set = TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/val.txt"), make_val_transform(ar))
    if sanity:
        train_set, val_set = Subset(train_set, range(32)), Subset(val_set, range(32))
    labels = np.concatenate([TN5000Dataset(str(TN5000_ROOT), str(TN5000_ROOT / "ImageSets/Main/train.txt")).get_labels(),
                             AUITDDataset(str(AUITD_ROOT)).get_labels()])
    config = {"lr_head": 3e-4, "weight_decay": 1e-4, "dropout": 0.3,
              "pos_weight": float((labels == 0).sum() / (labels == 1).sum()), "batch_size": 16,
              "max_epochs": 1 if sanity else 25, "patience": 10, "min_delta": 0.001,
              "T_0": 10, "T_mult": 2, "grad_clip_norm": 1.0, "label_smooth_eps": 0.05}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = NUM_WORKERS if device.type == "cuda" else 0
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model = MultiLevelSwin(dropout=0.3)
    started = time.time()
    history = train_model(model, train_loader, val_loader, config, str(seed_dir), config_name, device)
    elapsed = time.time() - started
    generated = seed_dir / f"{config_name}_best.pt"
    generated.rename(seed_dir / "best.pt")
    best_epoch = int(np.argmax(history["val_auc"]) + 1)
    metadata = {"config": config_name, "ar": ar, "ar_definition": ar_definition(ar), "tta_scales": scales,
                "tta_views": len(scales), "seed": seed, "best_epoch": best_epoch,
                "best_val_auc": history["best_val_auc"], "training_time_sec": elapsed,
                "training_samples": len(train_set), "validation_samples": len(val_set),
                "checkpoint": str((seed_dir / "best.pt").resolve()), "model": "MultiLevelSwin / Swin-Tiny",
                "parameter_count": sum(p.numel() for p in model.parameters()), "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(), "device": str(device), "platform": platform.platform(), "sanity": sanity}
    write_json(seed_dir / "config.json", metadata)
    write_json(seed_dir / "training_log.json", history)
    print(f"[RUN COMPLETE] {config_name} | seed {seed}/4 | best epoch={best_epoch} | "
          f"best validation AUC={history['best_val_auc']:.4f}", flush=True)
    return metadata


def append_master(row):
    path = OUT / "results" / "master_training_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def run_config(name, base_dir=OUT):
    ar, scales = CONFIGS[name]
    config_dir = base_dir / name
    metadata = []
    try:
        for seed in SEEDS:
            existing = completed_seed_metadata(config_dir / f"seed{seed}")
            if existing is not None:
                print(f"[RESUME] {name} | seed {seed}/4 already complete; preserving it.", flush=True)
                metadata.append(existing)
                continue
            seed_dir = config_dir / f"seed{seed}"
            if seed_dir.exists() and any(seed_dir.iterdir()):
                raise RuntimeError(
                    f"{seed_dir} is incomplete. It will not be overwritten. Move it to a dated "
                    "failed_attempt directory before rerunning this configuration.")
            metadata.append(run_seed(name, ar, scales, seed, base_dir))
        update_status(base_dir, name, state="evaluating", active_seed=None,
                      completed_seeds=list(SEEDS), tta_scales=scales)
    except Exception as exc:
        update_status(base_dir, name, state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result, seed_internal_auc, inference_time = evaluate_ensemble(config_dir, ar, scales, device)
    summary = {"config": name, "ar": ar, "ar_definition": ar_definition(ar), "tta_scales": scales,
               "tta_views": len(scales), "seeds": metadata, "per_seed_internal_auc": seed_internal_auc,
               "ensemble": result, "total_training_time_sec": sum(x["training_time_sec"] for x in metadata),
               "total_inference_time_sec": inference_time, "timestamp": datetime.now(timezone.utc).isoformat()}
    write_json(config_dir / "summary.json", summary)
    (config_dir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    row = {"Config": name, "AR": ar, "TTA_Scales": json.dumps(scales), "TTA_Views": len(scales)}
    for item in metadata:
        prefix = f"Seed{item['seed']}"
        row[f"{prefix}_BestEpoch"] = item["best_epoch"]
        row[f"{prefix}_BestValAUC"] = item["best_val_auc"]
        row[f"{prefix}_TestAUC_Internal"] = seed_internal_auc[item["seed"]]
    for label, values in result.items():
        for metric in ("AUC", "Accuracy", "F1", "Sensitivity", "Specificity", "PRAUC", "TP", "TN", "FP", "FN", "N", "PatientCount", "PerScaleAUC"):
            row[f"Ensemble_{label}_{metric}"] = values.get(metric)
    row["TotalTrainingTime"] = summary["total_training_time_sec"]
    row["TotalInferenceTime"] = inference_time
    row["Timestamp"] = summary["timestamp"]
    append_master(row)
    update_status(base_dir, name, state="complete", active_seed=None,
                  completed_seeds=list(SEEDS), summary=str((config_dir / "summary.json").resolve()))
    print(f"[CONFIG COMPLETE] {name}. Summary: {config_dir / 'summary.json'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="One epoch, 32 training/validation samples; no evaluation datasets.")
    parser.add_argument("--config", choices=CONFIGS, help="Run exactly one full selected configuration.")
    parser.add_argument("--run-all", action="store_true", help="Run the 10 configurations in protocol order.")
    parser.add_argument("--status", action="store_true", help="Show persisted configuration/seed status and exit.")
    args = parser.parse_args()
    if args.status:
        for name in CONFIGS:
            path = OUT / name / "run_status.json"
            if path.exists():
                state = json.loads(path.read_text(encoding="utf-8"))
            else:
                complete = [seed for seed in SEEDS if completed_seed_metadata(OUT / name / f"seed{seed}")]
                existing = [seed for seed in SEEDS if (OUT / name / f"seed{seed}").exists()]
                state = {"config": name,
                         "state": "partial_legacy_run" if existing else "not_started",
                         "completed_seeds": complete,
                         "incomplete_seed_directories": sorted(set(existing) - set(complete))}
            print(json.dumps(state, indent=2))
    elif args.sanity:
        item = run_seed("A1S4V2", *CONFIGS["A1S4V2"], 0, OUT / "_sanity", sanity=True)
        write_json(OUT / "_sanity" / "sanity_result.json", item)
        print(json.dumps(item, indent=2))
    elif args.config:
        run_config(args.config)
    elif args.run_all:
        for name in CONFIGS: run_config(name)
    else:
        parser.error("choose --sanity, --config, or --run-all")


if __name__ == "__main__":
    main()
