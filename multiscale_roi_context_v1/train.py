"""
Training script for multiscale_roi_context_v1.
Implements a custom training loop to handle the (full, tight, expanded) three-view tuple.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiscale_roi_context_v1.model import MultiScaleROIModel
from multiscale_roi_context_v1.dataset import (
    ROI_TN5000Dataset, ROI_AUITDDataset, ROI_TN5000TestDataset
)
from multiscale_roi_context_v1.transforms import (
    make_full_train_transform, make_roi_train_transform, make_tta_transforms
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model, loader, optimizer, scaler, device, pos_weight, epoch
):
    model.train()
    total_loss = 0.0
    pw = torch.tensor([pos_weight], device=device)
    
    for (full, tight, expanded), labels in loader:
        full = full.to(device, non_blocking=True)
        tight = tight.to(device, non_blocking=True)
        expanded = expanded.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(full, tight, expanded).squeeze(1)
                # Label smoothing: 0.05
                targets_smooth = labels * (1.0 - 0.05) + 0.5 * 0.05
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_smooth, pos_weight=pw
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(full, tight, expanded).squeeze(1)
            targets_smooth = labels * (1.0 - 0.05) + 0.5 * 0.05
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets_smooth, pos_weight=pw
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        total_loss += loss.item()
        
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_tta(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    for (full_tta, tight_tta, expanded_tta), labels, _ in loader:
        full_tta = full_tta.to(device)         # (B, 5, 3, 224, 224)
        tight_tta = tight_tta.to(device)
        expanded_tta = expanded_tta.to(device)
        
        B, n_tta, C, H, W = full_tta.shape
        full_tta = full_tta.view(-1, C, H, W)
        tight_tta = tight_tta.view(-1, C, H, W)
        expanded_tta = expanded_tta.view(-1, C, H, W)
        
        logits = model(full_tta, tight_tta, expanded_tta).squeeze(1) # (B*5)
        logits = logits.view(B, n_tta).mean(dim=1)                   # (B)
        
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_logits)
    return auc, all_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tn5000_root", default="data_raw/TN5000_forReview")
    parser.add_argument("--auitd_root", default="data_raw/auitd_dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16) # Consider 8 if OOM
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("multiscale_roi_context_v1/results") / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting multiscale_roi_context_v1 | Seed {args.seed} ---")
    
    # 1. Datasets
    full_tfm = make_full_train_transform()
    roi_tfm = make_roi_train_transform()
    tta_tfm = make_tta_transforms()
    
    tn5000_train = ROI_TN5000Dataset(
        args.tn5000_root, 
        split_file=os.path.join(args.tn5000_root, "ImageSets/Main/train.txt"),
        train_transform=full_tfm,
        roi_transform=roi_tfm
    )
    auitd_train = ROI_AUITDDataset(
        args.auitd_root,
        train_transform=full_tfm,
        roi_transform=roi_tfm
    )
    train_ds = ConcatDataset([tn5000_train, auitd_train])
    
    # Pos weight calculation
    labels = np.concatenate([tn5000_train.get_labels(), auitd_train.get_labels()])
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    pos_weight = float(n_neg / n_pos)
    print(f"Train samples: {len(train_ds)}, pos_weight: {pos_weight:.4f}")
    
    val_ds = ROI_TN5000TestDataset(args.tn5000_root, tta_transforms=tta_tfm)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )
    # Val loader batch size MUST be 1 because each item returns (B, 5, 3, 224, 224)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # 2. Model & Optimiser
    model = MultiScaleROIModel(dropout=0.3).to(device)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Groups will be updated in freeze_epoch loop
    model.freeze_epoch(1)
    optimizer = AdamW(model.get_param_groups(3e-4, 3e-5), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_auc = 0.0
    patience_cnt = 0
    patience_limit = 10
    
    # 3. Training Loop
    start_time = time.time()
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.freeze_epoch(epoch)
        # Update optimiser param groups for new freeze state
        optimizer.param_groups = model.get_param_groups(3e-4, 3e-5)
        
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, pos_weight, epoch)
        scheduler.step()
        
        val_auc, _ = evaluate_tta(model, val_loader, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Val AUC (TN5000 test): {val_auc:.4f}")
        history.append({"epoch": epoch, "loss": loss, "val_auc": val_auc})
        
        if val_auc > best_auc + 0.001:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "config": "multiscale_roi_context_v1"
            }, out_dir / "best.pt")
            print("  -> Saved new best model")
        else:
            patience_cnt += 1
            if patience_cnt >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"Training complete. Best Val AUC: {best_auc:.4f} in {(time.time()-start_time)/60:.1f} mins.")


if __name__ == "__main__":
    main()
