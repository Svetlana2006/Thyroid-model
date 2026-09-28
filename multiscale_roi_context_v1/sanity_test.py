"""
Sanity test suite for multiscale_roi_context_v1.

Tests (per §18 of spec):
 1.  Import / initialization
 2.  Model forward pass (synthetic batch)
 3.  Output shape
 4.  Parameter count
 5.  Shared-backbone assertion (only ONE backbone instance)
 6.  Gradients exist for fusion/head layers
 7.  TN5000 bounding-box parsing
 8.  ROI crop generation (tight + expanded)
 9.  Tight ROI dimensions / content
10.  Expanded ROI dimensions / content
11.  A4 preprocessing (train transform smoke test)
12.  TTA tensor shapes
13.  Dataset lengths > 0
14.  Patient-level Thyroid Pretraining grouping (exactly 2 per patient)
15.  Checkpoint save / load round-trip

Run:
    python -m multiscale_roi_context_v1.sanity_test \
        --tn5000_root  data_raw/TN5000_forReview \
        --auitd_root   data_raw/auitd_dataset \
        [--diveshzz_root   <path>] \
        [--pretraining_root <path>]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── helpers ───────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name):
        self.passed.append(name)
        print(f"  [PASS] {name}")

    def fail(self, name, reason=""):
        self.failed.append(name)
        print(f"  [FAIL] {name}" + (f": {reason}" if reason else ""))

    def check(self, condition, name, reason=""):
        if condition:
            self.ok(name)
        else:
            self.fail(name, reason)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"SANITY TEST RESULTS: {len(self.passed)}/{total} passed")
        if self.failed:
            print("FAILED:")
            for f in self.failed:
                print(f"  • {f}")
        print("="*60)
        return len(self.failed) == 0


def _synthetic_batch(B=2, device="cpu"):
    """Return (full, tight, expanded) batch of (B,3,224,224) tensors."""
    return (
        torch.randn(B, 3, 224, 224, device=device),
        torch.randn(B, 3, 224, 224, device=device),
        torch.randn(B, 3, 224, 224, device=device),
    )


# ── test functions ─────────────────────────────────────────────────────────────

def test_import(R: _Result):
    try:
        from multiscale_roi_context_v1.model import MultiScaleROIModel, FUSED_DIM
        from multiscale_roi_context_v1.roi import (
            compute_tight_roi, compute_expanded_roi, crop_and_pad, AutoROIExtractor,
        )
        from multiscale_roi_context_v1.transforms import (
            make_full_train_transform, make_roi_train_transform,
            make_val_transform, make_tta_transforms,
        )
        from multiscale_roi_context_v1.dataset import (
            ROI_TN5000Dataset, ROI_AUITDDataset,
        )
        R.ok("1. imports")
    except Exception as e:
        R.fail("1. imports", str(e))


def test_model_init(R: _Result):
    try:
        from multiscale_roi_context_v1.model import MultiScaleROIModel
        model = MultiScaleROIModel(dropout=0.3)
        R.ok("2. model initialization")
        return model
    except Exception as e:
        R.fail("2. model initialization", str(e))
        return None


def test_forward(R: _Result, model):
    if model is None:
        R.fail("3. forward pass", "model not initialized")
        return
    try:
        model.eval()
        full, tight, expanded = _synthetic_batch(B=2)
        with torch.no_grad():
            out = model(full, tight, expanded)
        R.check(out.shape == (2, 1), "3. forward pass output shape (2,1)",
                f"got {out.shape}")
    except Exception as e:
        R.fail("3. forward pass", str(e))


def test_output_shape_single(R: _Result, model):
    if model is None:
        R.fail("4. output shape B=1", "model not initialized")
        return
    try:
        model.eval()
        full, tight, expanded = _synthetic_batch(B=1)
        with torch.no_grad():
            out = model(full, tight, expanded)
        R.check(out.shape == (1, 1), "4. output shape B=1", f"got {out.shape}")
    except Exception as e:
        R.fail("4. output shape B=1", str(e))


def test_param_count(R: _Result, model):
    if model is None:
        R.fail("5. parameter count", "model not initialized")
        return
    n_total = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params:     {n_total:,}")
    print(f"       Backbone params:  {n_backbone:,}")
    print(f"       Trainable (ep1):  {n_trainable:,}")
    R.ok("5. parameter count")


def test_shared_backbone(R: _Result, model):
    """Verify only one backbone instance exists inside the model by identity."""
    if model is None:
        R.fail("6. shared backbone", "model not initialized")
        return
    # Walk all nn.Module children and collect unique ids of SwinTransformer top-level instances.
    # We check that every reference to a SwinTransformer module is the SAME object (same id),
    # meaning the backbone is truly shared and not duplicated.
    swin_ids = set()
    for name, mod in model.named_modules():
        # timm names the top-level Swin class "SwinTransformer"
        if type(mod).__name__ == "SwinTransformer":
            swin_ids.add(id(mod))
    R.check(
        len(swin_ids) == 1,
        "6. shared backbone (exactly 1 unique SwinTransformer object by id)",
        f"found {len(swin_ids)} unique SwinTransformer objects",
    )


def test_gradients(R: _Result, model):
    if model is None:
        R.fail("7. gradients through head", "model not initialized")
        return
    model.train()
    model.freeze_epoch(1)   # backbone frozen, head trainable
    full, tight, expanded = _synthetic_batch(B=2)
    out = model(full, tight, expanded)
    loss = out.mean()
    loss.backward()
    # Check that classifier + gate got gradients
    has_grad = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.classifier.parameters()
    )
    R.check(has_grad, "7. gradients flow through classifier + gate")
    model.zero_grad()


def test_bbox_parsing(R: _Result, tn5000_root: str):
    from multiscale_roi_context_v1.dataset import _parse_voc_annotation
    ann_dir = Path(tn5000_root) / "Annotations"
    if not ann_dir.exists():
        R.fail("8. TN5000 bbox parsing", f"Annotations dir not found: {ann_dir}")
        return
    xmls = list(ann_dir.glob("*.xml"))
    if not xmls:
        R.fail("8. TN5000 bbox parsing", "no XML files found")
        return
    try:
        label, (x0, y0, x1, y1) = _parse_voc_annotation(xmls[0])
        valid = x1 > x0 and y1 > y0 and label in (0, 1)
        R.check(valid, "8. TN5000 bbox parsing",
                f"label={label} box=({x0},{y0},{x1},{y1})")
    except Exception as e:
        R.fail("8. TN5000 bbox parsing", str(e))


def test_roi_crops(R: _Result, tn5000_root: str):
    import cv2
    from multiscale_roi_context_v1.dataset import _parse_voc_annotation
    from multiscale_roi_context_v1.roi import (
        compute_tight_roi, compute_expanded_roi, crop_and_pad, is_valid_crop,
    )
    ann_dir = Path(tn5000_root) / "Annotations"
    img_dir = Path(tn5000_root) / "JPEGImages"
    xmls = sorted(ann_dir.glob("*.xml"))[:5]

    all_tight_ok = True
    all_expanded_ok = True
    for ann_path in xmls:
        img_id = ann_path.stem
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        _, (xmin, ymin, xmax, ymax) = _parse_voc_annotation(ann_path)
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        tx0, ty0, tx1, ty1 = compute_tight_roi(xmin, ymin, xmax, ymax, W, H)
        tight_ok = is_valid_crop(tx0, ty0, tx1, ty1)
        if not tight_ok:
            all_tight_ok = False

        ex0, ey0, ex1, ey1 = compute_expanded_roi(xmin, ymin, xmax, ymax, W, H)
        # Expanded may be clamped to full image — that's OK
        expanded_ok = is_valid_crop(ex0, ey0, ex1, ey1)
        if not expanded_ok:
            all_expanded_ok = False

        # Expanded should always be >= tight
        expanded_area  = (ex1 - ex0) * (ey1 - ey0)
        tight_area     = (tx1 - tx0) * (ty1 - ty0)
        if expanded_area < tight_area:
            all_expanded_ok = False

    R.check(all_tight_ok,    "9.  tight ROI valid for sample images")
    R.check(all_expanded_ok, "10. expanded ROI valid and >= tight for sample images")


def test_transforms(R: _Result):
    from multiscale_roi_context_v1.transforms import (
        make_full_train_transform, make_roi_train_transform,
    )
    try:
        dummy = (np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        full_t  = make_full_train_transform()
        roi_t   = make_roi_train_transform()
        out_full = full_t(image=dummy)["image"]
        out_roi  = roi_t(image=dummy)["image"]
        ok = (out_full.shape == (3, 224, 224) and out_roi.shape == (3, 224, 224))
        R.check(ok, "11. A4 train + ROI transforms produce (3,224,224)",
                f"full={out_full.shape} roi={out_roi.shape}")
    except Exception as e:
        R.fail("11. A4 train + ROI transforms", str(e))


def test_tta_shapes(R: _Result):
    from multiscale_roi_context_v1.transforms import make_tta_transforms, TTA_SCALES
    try:
        tta = make_tta_transforms()
        dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        tensors = [t(image=dummy)["image"] for t in tta]
        ok = (len(tensors) == len(TTA_SCALES) and
              all(t.shape == (3, 224, 224) for t in tensors))
        R.check(ok, f"12. TTA produces {len(TTA_SCALES)} x (3,224,224) tensors",
                f"got {len(tensors)} tensors")
    except Exception as e:
        R.fail("12. TTA shapes", str(e))


def test_dataset_lengths(R: _Result, tn5000_root: str, auitd_root: str):
    from multiscale_roi_context_v1.dataset import ROI_TN5000Dataset, ROI_AUITDDataset
    split_file = Path(tn5000_root) / "ImageSets" / "Main" / "train.txt"
    try:
        tn = ROI_TN5000Dataset(tn5000_root, str(split_file))
        R.check(len(tn) > 0, f"13a. TN5000 train dataset length > 0 (got {len(tn)})")
    except Exception as e:
        R.fail("13a. TN5000 dataset", str(e))
    try:
        au = ROI_AUITDDataset(auitd_root)
        R.check(len(au) > 0, f"13b. AUITD dataset length > 0 (got {len(au)})")
    except Exception as e:
        R.fail("13b. AUITD dataset", str(e))


def test_patient_grouping(R: _Result, pretraining_root: str):
    if not pretraining_root:
        print("  [SKIP] 14. patient grouping (no pretraining_root provided)")
        return
    from multiscale_roi_context_v1.dataset import ROI_ThyroidPretrainingDataset
    from multiscale_roi_context_v1.transforms import make_tta_transforms
    try:
        ds = ROI_ThyroidPretrainingDataset(pretraining_root, make_tta_transforms())
        ok = len(ds) > 0
        if ok:
            # Spot-check first patient
            images, label, pid, raw = ds[0]
            ok = (images.shape[0] == 2)
        R.check(ok, "14. patient-level grouping: 2 images per patient")
    except Exception as e:
        R.fail("14. patient-level grouping", str(e))


def test_checkpoint(R: _Result, model):
    if model is None:
        R.fail("15. checkpoint save/load", "model not initialized")
        return
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "config": {"experiment": "multiscale_roi_context_v1"},
        }
        torch.save(ckpt, tmp_path)
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        os.unlink(tmp_path)
        ok = (
            "model_state_dict" in loaded
            and loaded["epoch"] == 1
            and set(loaded["model_state_dict"].keys()) == set(model.state_dict().keys())
        )
        R.check(ok, "15. checkpoint save/load round-trip")
    except Exception as e:
        R.fail("15. checkpoint save/load", str(e))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sanity tests for multiscale_roi_context_v1")
    parser.add_argument("--tn5000_root",      default="data_raw/TN5000_forReview")
    parser.add_argument("--auitd_root",       default="data_raw/auitd_dataset")
    parser.add_argument("--diveshzz_root",    default="")
    parser.add_argument("--pretraining_root", default="")
    args = parser.parse_args()

    print("="*60)
    print("multiscale_roi_context_v1 — SANITY TESTS")
    print("="*60)

    R = _Result()

    test_import(R)
    model = test_model_init(R)
    test_forward(R, model)
    test_output_shape_single(R, model)
    test_param_count(R, model)
    test_shared_backbone(R, model)
    test_gradients(R, model)
    test_bbox_parsing(R, args.tn5000_root)
    test_roi_crops(R, args.tn5000_root)
    test_transforms(R)
    test_tta_shapes(R)
    test_dataset_lengths(R, args.tn5000_root, args.auitd_root)
    test_patient_grouping(R, args.pretraining_root)
    test_checkpoint(R, model)

    if model is not None:
        print(f"\n{model.report()}")

    ok = R.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
