"""
Main training script for TN5000.
Runs the full pipeline:
1. Optuna hyperparameter search (optional)
2. Final 5-fold cross-validation on 70% train+val pool (plan §2)
3. Test set evaluation with bootstrap CI
4. Ensemble evaluation
5. Results table

Usage:
    python train.py --arch resnet50 --search --n_trials 40
    python train.py --arch efficientnet_b3
    python train.py --arch swin_tiny
    python train.py --ensemble  # evaluate all 3 after training
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import TN5000Dataset
from src.ensemble import (
    deep_ensemble_uncertainty,
    flag_low_confidence,
    soft_vote_ensemble,
    weighted_ensemble,
)
from src.metrics import (
    TemperatureScaling,
    bootstrap_metrics,
    compute_metrics,
    delong_test,
    ece_score,
    format_results_table,
    sigmoid,
    youden_threshold,
)
from src.models import build_model
from src.trainer import train_model
from src.transforms import get_train_transforms, get_val_transforms

# ── CPU heat mitigation ───────────────────────────────────────────────────────
# Limit PyTorch to half the logical cores. Without this, PyTorch saturates all
# cores at 100% for the entire run, which triggers thermal shutdowns on laptops.
# On GPU this has no effect (CUDA bypasses this setting).
_CPU_COUNT = os.cpu_count() or 4
torch.set_num_threads(max(1, _CPU_COUNT // 2))
torch.set_num_interop_threads(max(1, _CPU_COUNT // 4))


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_ROOT = Path("data_raw/TN5000_forReview")
SPLIT_DIR = DATA_ROOT / "ImageSets" / "Main"
TRAIN_TXT = str(SPLIT_DIR / "train.txt")
VAL_TXT = str(SPLIT_DIR / "val.txt")
TEST_TXT = str(SPLIT_DIR / "test.txt")
TRAINVAL_TXT = str(SPLIT_DIR / "trainval.txt")
OUTPUTS = Path("outputs")

ARCH_BATCH_SIZE = {
    "resnet50": 32,
    "efficientnet_b3": 32,
    "swin_tiny": 16,
}

# DataLoader settings — pin_memory and workers only help with GPU
_USE_GPU = torch.cuda.is_available()
_PIN_MEM  = _USE_GPU
_N_WORK   = 4 if _USE_GPU else 0   # 0 = main process only (required on CPU/Windows)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(arch: str) -> dict:
    """Load best config from Optuna search if available, else use defaults."""
    config_path = OUTPUTS / "optuna" / f"tn5000_{arch}_best_config.json"
    if config_path.exists():
        print(f"Loading Optuna config from {config_path}")
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        print(f"No Optuna config found for {arch}, using defaults.")
        cfg = {}

    defaults = {
        "lr_head": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "pos_weight_scale": 1.0,
        "batch_size": ARCH_BATCH_SIZE[arch],
        "max_epochs": 25,
        "patience": 10,
        "min_delta": 0.001,
        "T_0": 10,
        "T_mult": 2,
        "label_smooth_eps": 0.05,
        "grad_clip_norm": 1.0,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


def run_training(
    arch: str,
    seed: int = 42,
    device=None,
    search_first: bool = False,
    n_trials: int = 40,
):
    """Train one architecture with one seed on the official train split."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training {arch} | seed={seed} | device={device}")
    print(f"{'='*60}")

    if search_first:
        from src.hparam_search import run_hparam_search
        print(f"\nRunning Optuna search for {arch} ...")
        run_hparam_search(
            arch=arch,
            data_root=str(DATA_ROOT),
            train_txt=TRAIN_TXT,
            val_txt=VAL_TXT,
            n_trials=n_trials,
            n_seeds=3,
            n_epochs_for_trial=20,
            output_dir=str(OUTPUTS / "optuna"),
            device=device,
        )

    config = load_config(arch)

    train_dataset = TN5000Dataset(
        str(DATA_ROOT), TRAIN_TXT, transform=get_train_transforms()
    )
    val_dataset = TN5000Dataset(
        str(DATA_ROOT), VAL_TXT, transform=get_val_transforms()
    )
    test_dataset = TN5000Dataset(
        str(DATA_ROOT), TEST_TXT, transform=get_val_transforms()
    )

    pos_weight = train_dataset.get_class_weights()
    config["pos_weight"] = float(pos_weight)

    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=_N_WORK, pin_memory=_PIN_MEM
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=_N_WORK, pin_memory=_PIN_MEM
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=_N_WORK, pin_memory=_PIN_MEM
    )

    model = build_model(arch, dropout=config["dropout"])
    ckpt_dir = OUTPUTS / "checkpoints"

    history = train_model(
        model, train_loader, val_loader, config,
        checkpoint_dir=str(ckpt_dir),
        run_name=f"{arch}_seed{seed}",
        device=device,
    )

    # Evaluate on test set
    from src.trainer import evaluate
    test_metrics = evaluate(model, test_loader, device, config["pos_weight"], config["pos_weight_scale"])
    test_logits = test_metrics["logits"]
    test_labels = test_metrics["labels"]

    # Find Youden's J threshold on val set
    val_metrics = evaluate(model, val_loader, device, config["pos_weight"], config["pos_weight_scale"])
    threshold = youden_threshold(val_metrics["logits"], val_metrics["labels"])

    # Temperature scaling
    ts = TemperatureScaling()
    ts.fit(val_metrics["logits"], val_metrics["labels"])
    calibrated_logits = ts.transform(test_logits)
    calibrated_probs = sigmoid(calibrated_logits)

    # Metrics
    metrics = compute_metrics(test_logits, test_labels, threshold=threshold)
    metrics["ece_before"] = ece_score(sigmoid(test_logits), test_labels)
    metrics["ece_after"] = ece_score(calibrated_probs, test_labels)
    metrics["temperature"] = ts.temperature

    ci = bootstrap_metrics(test_logits, test_labels, threshold=threshold)

    print(f"\n[{arch}] Test Results (threshold={threshold:.3f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    result_dir = OUTPUTS / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / f"{arch}_seed{seed}_logits.npy", test_logits)
    np.save(result_dir / f"{arch}_seed{seed}_labels.npy", test_labels)
    with open(result_dir / f"{arch}_seed{seed}_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "ci": ci, "config": config, "history": {k: list(v) for k, v in history.items() if isinstance(v, list)}}, f, indent=2)

    print(f"\nResults saved to {result_dir}")
    return model, test_logits, test_labels, metrics


def run_ensemble_evaluation():
    """Load saved logits and run ensemble analysis."""
    result_dir = OUTPUTS / "results"
    if not result_dir.exists():
        print("No results found. Run training first.")
        return

    archs = ["resnet50", "efficientnet_b3", "swin_tiny"]
    all_logits = {}
    labels = None

    for arch in archs:
        for seed in range(3):
            path = result_dir / f"{arch}_seed{seed}_logits.npy"
            if path.exists():
                all_logits[f"{arch}_s{seed}"] = np.load(path)
                if labels is None:
                    labels = np.load(result_dir / f"{arch}_seed{seed}_labels.npy")

    if not all_logits:
        print("No logits found. Run training first.")
        return

    print(f"\nFound {len(all_logits)} models for ensemble evaluation.")

    # Per-architecture results
    all_results = {}
    for arch in archs:
        arch_logits = [v for k, v in all_logits.items() if k.startswith(arch)]
        if not arch_logits:
            continue
        # Average across seeds
        avg_logits = soft_vote_ensemble(arch_logits)
        m = compute_metrics(avg_logits, labels)
        m["ece"] = ece_score(sigmoid(avg_logits), labels)
        all_results[f"{arch} (avg seeds)"] = m

    # Simple ensemble (all 9 models)
    all_logits_list = list(all_logits.values())
    simple_ens_logits = soft_vote_ensemble(all_logits_list)
    m = compute_metrics(simple_ens_logits, labels)
    m["ece"] = ece_score(sigmoid(simple_ens_logits), labels)
    all_results["Simple Ensemble (9)"] = m

    # Weighted ensemble (optimise on labels — note: only valid if these are val labels)
    weighted_logits, weights = weighted_ensemble(all_logits_list, labels)
    m = compute_metrics(weighted_logits, labels)
    m["ece"] = ece_score(sigmoid(weighted_logits), labels)
    all_results["Weighted Ensemble"] = m
    print(f"Optimal weights: {[f'{w:.1f}' for w in weights]}")

    # Uncertainty
    mean_logits, variance = deep_ensemble_uncertainty(all_logits)
    low_conf_mask = flag_low_confidence(variance, 90.0)
    print(f"\nLow-confidence cases (top 10% variance): {low_conf_mask.sum()} / {len(low_conf_mask)}")

    # DeLong's test between architectures
    arch_logits_by_name = {}
    for arch in archs:
        arch_logits_list = [v for k, v in all_logits.items() if k.startswith(arch)]
        if arch_logits_list:
            arch_logits_by_name[arch] = soft_vote_ensemble(arch_logits_list)

    if len(arch_logits_by_name) >= 2:
        arch_names = list(arch_logits_by_name.keys())
        print(f"\nDeLong's test: {arch_names[0]} vs {arch_names[1]}")
        delong = delong_test(
            arch_logits_by_name[arch_names[0]],
            arch_logits_by_name[arch_names[1]],
            labels,
        )
        print(json.dumps(delong, indent=2))

    # Print table
    print("\n" + format_results_table(all_results))

    # Save
    with open(result_dir / "ensemble_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEnsemble results saved to {result_dir / 'ensemble_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="TN5000 Thyroid Model Training")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b3", "swin_tiny"],
                        help="Architecture to train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds to train (for deep ensemble)")
    parser.add_argument("--search", action="store_true",
                        help="Run Optuna hyperparameter search before training")
    parser.add_argument("--n_trials", type=int, default=40,
                        help="Number of Optuna trials")
    parser.add_argument("--ensemble", action="store_true",
                        help="Run ensemble evaluation on saved logits")
    parser.add_argument("--all_archs", action="store_true",
                        help="Train all 3 architectures × 3 seeds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu / cuda:0)")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.ensemble:
        run_ensemble_evaluation()
        return

    archs_to_train = ["resnet50", "efficientnet_b3", "swin_tiny"] if args.all_archs else [args.arch]

    for arch in archs_to_train:
        for seed in args.seeds:
            run_training(
                arch=arch,
                seed=seed,
                device=device,
                search_first=args.search and seed == args.seeds[0],  # search only once
                n_trials=args.n_trials,
            )

    if args.all_archs or len(archs_to_train) > 1:
        run_ensemble_evaluation()


if __name__ == "__main__":
    main()
