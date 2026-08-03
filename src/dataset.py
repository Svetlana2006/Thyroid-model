"""
TN5000 Thyroid Nodule Dataset Loader
VOC-format: each image has an XML annotation with <object><name> = 0 (benign) or 1 (malignant)
and a bounding box. We use the bounding box to crop the nodule region for training.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TN5000Dataset(Dataset):
    """
    TN5000 thyroid nodule classification dataset.

    Args:
        data_root: Root directory containing Annotations/, JPEGImages/, ImageSets/
        split_file: Path to .txt file listing image IDs (one per line)
        transform: Albumentation or torchvision transforms to apply
        crop_nodule: If True, crop to bounding box (with padding); else use full image
        bbox_pad_ratio: Fraction of bbox size to pad around bounding box crop
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        transform: Optional[Callable] = None,
        crop_nodule: bool = False,
        bbox_pad_ratio: float = 0.2,
    ):
        self.data_root = Path(data_root)
        self.img_dir = self.data_root / "JPEGImages"
        self.ann_dir = self.data_root / "Annotations"
        self.transform = transform
        self.crop_nodule = crop_nodule
        self.bbox_pad_ratio = bbox_pad_ratio

        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for img_id in self.ids:
            ann_path = self.ann_dir / f"{img_id}.xml"
            img_path = self.img_dir / f"{img_id}.jpg"
            label, bbox = self._parse_annotation(ann_path)
            self.samples.append(
                {
                    "id": img_id,
                    "img_path": str(img_path),
                    "label": label,
                    "bbox": bbox,  # (xmin, ymin, xmax, ymax) in pixel coords
                }
            )

    def _parse_annotation(self, ann_path):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        label = sample["label"]

        if self.crop_nodule:
            xmin, ymin, xmax, ymax = sample["bbox"]
            w, h = image.size
            bw = xmax - xmin
            bh = ymax - ymin
            pad_x = int(bw * self.bbox_pad_ratio)
            pad_y = int(bh * self.bbox_pad_ratio)
            crop_xmin = max(0, xmin - pad_x)
            crop_ymin = max(0, ymin - pad_y)
            crop_xmax = min(w, xmax + pad_x)
            crop_ymax = min(h, ymax + pad_y)
            image = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        image = np.array(image)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.float32)

    def get_labels(self):
        """Return all labels as numpy array (for class weight computation)."""
        return np.array([s["label"] for s in self.samples])

    def get_class_weights(self):
        """Compute pos_weight = n_benign / n_malignant for BCEWithLogitsLoss."""
        labels = self.get_labels()
        n_benign = (labels == 0).sum()
        n_malignant = (labels == 1).sum()
        pos_weight = n_benign / n_malignant
        return pos_weight
