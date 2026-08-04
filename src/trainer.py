"""
Training engine for TN5000.
Handles: optimizer, scheduler, mixed precision (GPU) / plain (CPU),
grad clipping, early stopping, staged freeze schedule, live progress bar.
"""

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# ── AMP: only use when CUDA is available ──────────────────────────────────────
_USE_AMP = torch.cuda.is_available()

if _USE_AMP:
    from torch.amp import GradScaler, autocast
    _SCALER_DEVICE = "cuda"
else:
    GradScaler = None          # won't be instantiated
    from contextlib import nullcontext as autocast   # no-op on CPU
    _SCALER_DEVICE = "cpu"

# ── tqdm progress bar ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _progress(iterable, **kwargs):
    """Wrap iterable with tqdm if available, else plain iteration."""
    if _HAS_TQDM:
        return tqdm(iterable, **kwargs)
    return iterable


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    pos_weight: float,
    pos_weight_scale: float = 1.0,
    label_smooth_eps: float = 0.05,
    grad_clip_norm: float = 1.0,
    epoch: int = 0,
    run_name: str = "",
) -> dict:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    pw = torch.tensor([pos_weight * pos_weight_scale], device=device)

    bar = _progress(
        loader,
        desc=f"  [{run_name}] Epoch {epoch:03d} TRAIN",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )

    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if _USE_AMP:
            with autocast(_SCALER_DEVICE):
                logits = model(images).squeeze(1)
                targets_smooth = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_smooth, pos_weight=pw
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            targets_smooth = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets_smooth, pos_weight=pw
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_logits.extend(logits.detach().cpu().float().tolist())
        all_labels.extend(labels.cpu().tolist())

        if _HAS_TQDM:
            bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_logits)
    return {"loss": avg_loss, "auc": auc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    pos_weight_scale: float = 1.0,
    label_smooth_eps: float = 0.05,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    pw = torch.tensor([pos_weight * pos_weight_scale], device=device)

    bar = _progress(loader, desc="  [eval]", leave=False, unit="batch", dynamic_ncols=True)

    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if _USE_AMP:
            with autocast(_SCALER_DEVICE):
                logits = model(images).squeeze(1)
                targets_smooth = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_smooth, pos_weight=pw
                )
        else:
            logits = model(images).squeeze(1)
            targets_smooth = labels * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets_smooth, pos_weight=pw
            )

        total_loss += loss.item() * images.size(0)
        all_logits.extend(logits.cpu().float().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_logits)
    return {"loss": avg_loss, "auc": auc, "logits": np.array(all_logits), "labels": np.array(all_labels)}


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_optimizer_and_scheduler(
    model, lr_head, lr_backbone, weight_decay, T_0, T_mult, last_epoch=-1
):
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr_head, lr_backbone)
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_head}]
    
    # Crucial fix: when last_epoch > -1, PyTorch schedulers expect 'initial_lr' to be set
    if last_epoch != -1:
        for group in param_groups:
            group.setdefault("initial_lr", group["lr"])

    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, last_epoch=last_epoch
    )
    return optimizer, scheduler

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    checkpoint_dir: Optional[str] = None,
    run_name: str = "run",
    device: Optional[torch.device] = None,
) -> dict:
    """
    Full training loop with:
    - AdamW + CosineAnnealingWarmRestarts
    - Mixed precision on GPU, plain float32 on CPU
    - Live per-batch progress bar (tqdm)
    - Staged freeze schedule + discriminative LRs
    - Early stopping on val AUC
    - Gradient clipping
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lr_head        = config.get("lr_head", 3e-4)
    lr_backbone    = lr_head * 0.1
    weight_decay   = config.get("weight_decay", 1e-4)
    pos_weight     = config.get("pos_weight", 0.4)
    pos_weight_sc  = config.get("pos_weight_scale", 1.0)
    max_epochs     = config.get("max_epochs", 60)
    patience       = config.get("patience", 10)
    min_delta      = config.get("min_delta", 0.001)
    T_0            = config.get("T_0", 10)
    T_mult         = config.get("T_mult", 2)
    smooth_eps     = config.get("label_smooth_eps", 0.05)
    grad_clip      = config.get("grad_clip_norm", 1.0)

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(_SCALER_DEVICE) if _USE_AMP else None
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    print(f"\n  Mode: {'GPU + AMP (float16)' if _USE_AMP else 'CPU (float32)'}")
    print(f"  Max epochs={max_epochs} | patience={patience} | batch={config.get('batch_size','?')}\n")

    # ── Build optimizer & scheduler ONCE before the loop ──────────────────────
    # Rebuilding AdamW every epoch discards all accumulated momentum/variance
    # and wastes CPU cycles. We only rebuild when staged unfreezing adds params.
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr_head, lr_backbone, weight_decay, T_0, T_mult, last_epoch=-1
    )
    _prev_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    for epoch in range(1, max_epochs + 1):
        if hasattr(model, "freeze_epoch"):
            model.freeze_epoch(epoch)
            # Rebuild optimizer only when staged unfreezing adds new parameters
            _curr_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            if _curr_trainable != _prev_trainable:
                print(f"  [unfreeze] Trainable params {_prev_trainable}\u2192{_curr_trainable}. Rebuilding optimizer.")
                optimizer, scheduler = build_optimizer_and_scheduler(
                    model, lr_head, lr_backbone, weight_decay, T_0, T_mult, last_epoch=epoch - 1
                )
                _prev_trainable = _curr_trainable

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            pos_weight, pos_weight_sc, smooth_eps, grad_clip,
            epoch=epoch, run_name=run_name,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, pos_weight, pos_weight_sc, smooth_eps)
        elapsed = time.time() - t0

        history["train_loss"].append(train_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        star = " ★" if early_stopping.counter == 0 or epoch == 1 else ""
        print(
            f"[{run_name}] Epoch {epoch:03d}/{max_epochs} | "
            f"Train AUC={train_metrics['auc']:.4f}  Loss={train_metrics['loss']:.4f} | "
            f"Val AUC={val_metrics['auc']:.4f}  Loss={val_metrics['loss']:.4f} | "
            f"{elapsed/60:.1f}min{star}"
        )

        should_stop = early_stopping(val_metrics["auc"], model)

        if checkpoint_dir and early_stopping.counter == 0:
            ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_metrics["auc"],
                    "config": config,
                },
                ckpt_path,
            )

        if should_stop:
            print(f"[{run_name}] Early stopping triggered at epoch {epoch}. Best val AUC = {early_stopping.best_score:.4f}")
            break

    early_stopping.restore_best(model)
    history["best_val_auc"] = early_stopping.best_score
    return history
