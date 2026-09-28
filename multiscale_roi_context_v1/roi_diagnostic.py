"""
ROI crop diagnostic for multiscale_roi_context_v1.

Saves side-by-side visualisations of:
  original image | tight ROI | expanded ROI

for a sample of TN5000 training images to verify correctness of the
bounding-box transforms BEFORE training.

Usage:
    python -m multiscale_roi_context_v1.roi_diagnostic \
        --data_root data_raw/TN5000_forReview \
        --n_samples 12 \
        --out_dir multiscale_roi_context_v1/diagnostics/roi_check
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiscale_roi_context_v1.roi import (
    compute_tight_roi,
    compute_expanded_roi,
    crop_and_pad,
    is_valid_crop,
    TIGHT_PAD_RATIO,
    EXPANDED_SCALE,
)


def _parse_voc(ann_path: Path):
    tree = ET.parse(ann_path)
    root = tree.getroot()
    obj = root.find("object")
    label = int(obj.find("name").text)
    b = obj.find("bndbox")
    return label, (
        int(float(b.find("xmin").text)),
        int(float(b.find("ymin").text)),
        int(float(b.find("xmax").text)),
        int(float(b.find("ymax").text)),
    )


def _draw_bbox(img: np.ndarray, x0, y0, x1, y1, color, label="") -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    if label:
        cv2.putText(out, label, (x0, max(y0 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    ratio = h / img.shape[0]
    new_w = max(1, int(img.shape[1] * ratio))
    return cv2.resize(img, (new_w, h))


def run_diagnostic(data_root: str, n_samples: int, out_dir: str):
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = data_root / "Annotations"
    img_dir = data_root / "JPEGImages"
    split_file = data_root / "ImageSets" / "Main" / "train.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]

    import random
    random.seed(42)
    sample_ids = random.sample(ids, min(n_samples, len(ids)))

    issues = []
    for img_id in sample_ids:
        ann_path = ann_dir / f"{img_id}.xml"
        img_path = img_dir / f"{img_id}.jpg"

        label, (xmin, ymin, xmax, ymax) = _parse_voc(ann_path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            issues.append(f"{img_id}: could not read image")
            continue
        H, W = img_bgr.shape[:2]

        # Draw GT box on full image
        full_vis = _draw_bbox(img_bgr, xmin, ymin, xmax, ymax,
                               (0, 255, 0), f"GT {'mal' if label else 'ben'}")

        # Tight ROI
        tx0, ty0, tx1, ty1 = compute_tight_roi(xmin, ymin, xmax, ymax, W, H)
        if not is_valid_crop(tx0, ty0, tx1, ty1):
            issues.append(f"{img_id}: degenerate TIGHT ROI {tx0},{ty0},{tx1},{ty1}")
        tight_crop = crop_and_pad(img_bgr, tx0, ty0, tx1, ty1)
        tight_resized = cv2.resize(tight_crop, (224, 224))

        # Expanded ROI
        ex0, ey0, ex1, ey1 = compute_expanded_roi(xmin, ymin, xmax, ymax, W, H)
        if not is_valid_crop(ex0, ey0, ex1, ey1):
            issues.append(f"{img_id}: degenerate EXPANDED ROI {ex0},{ey0},{ex1},{ey1}")
        expanded_crop = crop_and_pad(img_bgr, ex0, ey0, ex1, ey1)
        expanded_resized = cv2.resize(expanded_crop, (224, 224))

        # Draw labels on crops
        tight_vis = tight_resized.copy()
        cv2.putText(tight_vis, f"TIGHT (pad={TIGHT_PAD_RATIO})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        expanded_vis = expanded_resized.copy()
        cv2.putText(expanded_vis, f"EXPANDED (x{EXPANDED_SCALE})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

        # Side by side: full | tight | expanded (all resized to 224px height)
        PANEL_H = 224
        full_panel    = _resize_to_height(full_vis, PANEL_H)
        tight_panel   = tight_vis
        expanded_panel = expanded_vis

        strip = np.concatenate([full_panel, tight_panel, expanded_panel], axis=1)

        # Add header bar
        header = np.zeros((28, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(header,
                    f"{img_id}  |  Full (GT box)  |  Tight ROI  |  Expanded ROI",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        panel = np.concatenate([header, strip], axis=0)

        out_path = out_dir / f"{img_id}_roi_check.jpg"
        cv2.imwrite(str(out_path), panel)

    print(f"\n[ROI Diagnostic] Saved {len(sample_ids)} panels to {out_dir}/")
    print(f"  tight_pad_ratio = {TIGHT_PAD_RATIO}")
    print(f"  expanded_scale  = {EXPANDED_SCALE}")

    if issues:
        print(f"\n[WARNING] {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n[!] DO NOT TRAIN until these issues are resolved.")
    else:
        print("\n[OK] All crops look valid. No degenerate ROIs detected.")

    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROI crop diagnostic")
    parser.add_argument("--data_root", default="data_raw/TN5000_forReview")
    parser.add_argument("--n_samples", type=int, default=12)
    parser.add_argument("--out_dir",
                        default="multiscale_roi_context_v1/diagnostics/roi_check")
    args = parser.parse_args()
    issues = run_diagnostic(args.data_root, args.n_samples, args.out_dir)
    sys.exit(1 if issues else 0)
