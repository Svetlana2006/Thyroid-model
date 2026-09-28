"""
Dataset classes for multiscale_roi_context_v1.

Each dataset returns three views per sample:
  (full_tensor, tight_tensor, expanded_tensor), label

Bounding-box availability:
  TN5000:                  ✅ GT VOC XML boxes
  AUITD:                   ❌ no boxes — full image used for all three views
                              (at inference, AutoROIExtractor fills in ROIs)
  Diveshzz (eval only):    ❌ no boxes
  Thyroid Pretraining:     ❌ no boxes; patient-level aggregation preserved

NOTE: for AUITD training samples (no GT box), we fall back to using the
      full image for all three views. This means the model receives three
      identical views for AUITD. This is intentional: it preserves gradient
      flow through all branches during training without fabricating ROIs.
      At inference AutoROIExtractor generates automatic ROIs for all datasets.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from multiscale_roi_context_v1.roi import (
    compute_tight_roi,
    compute_expanded_roi,
    crop_and_pad,
    TIGHT_PAD_RATIO,
    EXPANDED_SCALE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image_bgr2rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_voc_annotation(ann_path: Path) -> Tuple[int, Tuple[int, int, int, int]]:
    """Parse TN5000 VOC XML annotation. Returns (label, (xmin,ymin,xmax,ymax))."""
    tree = ET.parse(ann_path)
    root = tree.getroot()
    obj = root.find("object")
    label = int(obj.find("name").text)
    bbox_elem = obj.find("bndbox")
    xmin = int(float(bbox_elem.find("xmin").text))
    ymin = int(float(bbox_elem.find("ymin").text))
    xmax = int(float(bbox_elem.find("xmax").text))
    ymax = int(float(bbox_elem.find("ymax").text))
    return label, (xmin, ymin, xmax, ymax)


def _apply_transform(image: np.ndarray, transform: Optional[Callable]) -> torch.Tensor:
    if transform is None:
        return torch.from_numpy(image.transpose(2, 0, 1)).float()
    return transform(image=image)["image"]


def _make_three_views(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    train_transform: Optional[Callable],
    roi_transform: Optional[Callable],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a full HWC image and optional GT bbox, produce:
      full:     transformed full image
      tight:    tight ROI crop + transform
      expanded: expanded ROI crop + transform

    If bbox is None (no GT box available), all three views are the full image.
    The roi_transform is applied identically to both ROI views.
    """
    full_t = _apply_transform(image, train_transform)

    if bbox is None:
        # No GT box: use full image for all views (AUITD case)
        tight_t = _apply_transform(image.copy(), roi_transform)
        expanded_t = _apply_transform(image.copy(), roi_transform)
    else:
        xmin, ymin, xmax, ymax = bbox
        H, W = image.shape[:2]

        tx0, ty0, tx1, ty1 = compute_tight_roi(xmin, ymin, xmax, ymax, W, H)
        tight_crop = crop_and_pad(image, tx0, ty0, tx1, ty1)
        tight_t = _apply_transform(tight_crop, roi_transform)

        ex0, ey0, ex1, ey1 = compute_expanded_roi(xmin, ymin, xmax, ymax, W, H)
        expanded_crop = crop_and_pad(image, ex0, ey0, ex1, ey1)
        expanded_t = _apply_transform(expanded_crop, roi_transform)

    return full_t, tight_t, expanded_t


# ── TN5000 (training + internal validation) ───────────────────────────────────

class ROI_TN5000Dataset(Dataset):
    """
    TN5000 dataset returning three views per sample.

    Annotations: VOC XML with GT bounding boxes (xmin,ymin,xmax,ymax).
    All three views use GT boxes — no automatic ROI needed here.
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        train_transform: Optional[Callable] = None,
        roi_transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.img_dir = self.data_root / "JPEGImages"
        self.ann_dir = self.data_root / "Annotations"
        self.train_transform = train_transform
        self.roi_transform = roi_transform

        with open(split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for img_id in ids:
            ann_path = self.ann_dir / f"{img_id}.xml"
            img_path = self.img_dir / f"{img_id}.jpg"
            label, bbox = _parse_voc_annotation(ann_path)
            self.samples.append({
                "id": img_id,
                "img_path": str(img_path),
                "label": label,
                "bbox": bbox,   # (xmin, ymin, xmax, ymax) — GT
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = _load_image_bgr2rgb(s["img_path"])
        full_t, tight_t, expanded_t = _make_three_views(
            image, s["bbox"], self.train_transform, self.roi_transform
        )
        label = torch.tensor(s["label"], dtype=torch.float32)
        return (full_t, tight_t, expanded_t), label

    def get_labels(self) -> np.ndarray:
        return np.array([s["label"] for s in self.samples])


# ── AUITD (training, no GT boxes) ─────────────────────────────────────────────

class ROI_AUITDDataset(Dataset):
    """
    AUITD dataset returning three views per sample.

    No bounding boxes available. All three views are the full image during
    training. At inference, AutoROIExtractor generates automatic ROIs.
    """

    def __init__(
        self,
        data_root: str,
        train_transform: Optional[Callable] = None,
        roi_transform: Optional[Callable] = None,
    ):
        self.train_transform = train_transform
        self.roi_transform = roi_transform
        self.samples: List[dict] = []

        dataset_dir = Path(data_root) / "dataset thyroid"
        for split in ["train", "test"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            for class_name in os.listdir(split_dir):
                class_dir = split_dir / class_name
                if not class_dir.is_dir():
                    continue
                label = {"benign": 0, "malignant": 1}.get(class_name.lower())
                if label is None:
                    continue
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.samples.append({
                                "img_path": os.path.join(root, file),
                                "label": label,
                            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = _load_image_bgr2rgb(s["img_path"])
        # bbox=None: all three views will be the full image
        full_t, tight_t, expanded_t = _make_three_views(
            image, None, self.train_transform, self.roi_transform
        )
        label = torch.tensor(s["label"], dtype=torch.float32)
        return (full_t, tight_t, expanded_t), label

    def get_labels(self) -> np.ndarray:
        return np.array([s["label"] for s in self.samples])


# ── TN5000 test (with TTA, three views per TTA scale) ─────────────────────────

class ROI_TN5000TestDataset(Dataset):
    """
    TN5000 test split: returns TTA tensors for all three views.

    Returns:
        (full_tta, tight_tta, expanded_tta): each (n_tta, 3, 224, 224)
        label: scalar
        img_id: str
    """

    def __init__(self, data_root: str, tta_transforms: List[Callable]):
        self.data_root = Path(data_root)
        self.tta_transforms = tta_transforms
        split_file = self.data_root / "ImageSets" / "Main" / "test.txt"
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for img_id in ids:
            ann_path = self.data_root / "Annotations" / f"{img_id}.xml"
            img_path = self.data_root / "JPEGImages" / f"{img_id}.jpg"
            label, bbox = _parse_voc_annotation(ann_path)
            self.samples.append({
                "id": img_id, "img_path": str(img_path),
                "label": label, "bbox": bbox,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = _load_image_bgr2rgb(s["img_path"])
        xmin, ymin, xmax, ymax = s["bbox"]
        H, W = image.shape[:2]

        tx0, ty0, tx1, ty1 = compute_tight_roi(xmin, ymin, xmax, ymax, W, H)
        tight_crop = crop_and_pad(image, tx0, ty0, tx1, ty1)

        ex0, ey0, ex1, ey1 = compute_expanded_roi(xmin, ymin, xmax, ymax, W, H)
        expanded_crop = crop_and_pad(image, ex0, ey0, ex1, ey1)

        full_tta    = torch.stack([t(image=image)["image"]         for t in self.tta_transforms])
        tight_tta   = torch.stack([t(image=tight_crop)["image"]    for t in self.tta_transforms])
        expanded_tta = torch.stack([t(image=expanded_crop)["image"] for t in self.tta_transforms])

        label = torch.tensor(s["label"], dtype=torch.float32)
        return (full_tta, tight_tta, expanded_tta), label, s["id"]


# ── Diveshzz (eval only, no GT boxes) ─────────────────────────────────────────

class ROI_DiveshzzDataset(Dataset):
    """
    Diveshzz evaluation dataset. No GT boxes.
    At inference the model uses AutoROIExtractor to generate ROIs,
    but for the dataset's __getitem__ we return TTA tensors of the full image
    only; the ROI extraction happens inside the evaluation loop.
    """

    def __init__(self, data_root: str, tta_transforms: List[Callable]):
        self.tta_transforms = tta_transforms
        self.samples: List[dict] = []
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        for cls, label in [("0", 0), ("1", 1)]:
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.exists(cls_dir):
                for img_file in glob.glob(os.path.join(cls_dir, "*.*")):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append({"img_path": img_file, "label": label})
        for i, s in enumerate(self.samples):
            s["id"] = f"{i:04d}_{os.path.basename(s['img_path'])}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = _load_image_bgr2rgb(s["img_path"])
        # Store raw image for AutoROI; return TTA of full image
        # ROI views are generated in the eval loop via AutoROIExtractor
        full_tta = torch.stack([t(image=image)["image"] for t in self.tta_transforms])
        label = torch.tensor(s["label"], dtype=torch.float32)
        return full_tta, label, s["id"], image   # image returned for auto ROI


# ── Thyroid for Pretraining (eval only, patient-level, no GT boxes) ───────────

class ROI_ThyroidPretrainingDataset(Dataset):
    """
    Thyroid for Pretraining evaluation dataset.
    Exactly 2 images per patient; consistent labels within patient.
    No GT boxes. AutoROIExtractor used at inference.
    Returns per-patient TTA tensors (2 images x n_tta).
    """

    def __init__(self, data_root: str, tta_transforms: List[Callable]):
        self.tta_transforms = tta_transforms
        self.patients: dict = defaultdict(list)
        dataset_dir = Path(data_root)
        for cls, label in [("0", 0), ("1", 1)]:
            class_dir = dataset_dir / "classifiy" / "augtrain" / str(cls)
            if not class_dir.exists():
                continue
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_path = os.path.join(root, file)
                        pid = Path(img_path).stem.split("_")[0]
                        self.patients[pid].append({"img_path": img_path, "label": label})
        self.patient_ids = list(self.patients.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        items = self.patients[pid]
        assert len(items) == 2, f"Patient {pid} has {len(items)} images, expected 2"
        labels_set = {it["label"] for it in items}
        assert len(labels_set) == 1, f"Patient {pid} inconsistent labels: {labels_set}"

        images = []
        raw_images = []
        for item in items:
            img = _load_image_bgr2rgb(item["img_path"])
            raw_images.append(img)
            full_tta = torch.stack([t(image=img)["image"] for t in self.tta_transforms])
            images.append(full_tta)

        label = torch.tensor(items[0]["label"], dtype=torch.float32)
        # raw_images returned for AutoROI generation in eval loop
        return torch.stack(images), label, pid, raw_images
