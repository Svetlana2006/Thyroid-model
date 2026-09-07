import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix
from torch.utils.data import Dataset
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJ_DIM = 128
FUSION_DIM = 256
THRESHOLD = 0.5912

# TTA Scale Definitions
TTA_SCALES_DICT = {
    "S1V1": [0.85, 1.00, 1.15],
    "S1V2": [0.70, 0.85, 1.00, 1.15, 1.30],
    "S1V3": [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45],
    
    "S2V1": [0.80, 1.00, 1.20],
    "S2V2": [0.60, 0.80, 1.00, 1.20, 1.40],
    "S2V3": [0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60],
    
    "S3V1": [0.75, 1.00, 1.25],
    "S3V2": [0.50, 0.75, 1.00, 1.25, 1.50],
    "S3V3": [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75],
    
    "S4V1": [0.75, 1.00, 1.15],
    "S4V2": [0.60, 0.75, 1.00, 1.15, 1.40],
    "S4V3": [0.35, 0.60, 0.75, 1.00, 1.15, 1.40, 1.55],
    
    "S5V1": [0.85, 1.00, 1.25],
    "S5V2": [0.60, 0.85, 1.00, 1.25, 1.40],
    "S5V3": [0.45, 0.60, 0.85, 1.00, 1.25, 1.40, 1.65],
}

def get_transform(ar_type: str, scale: float):
    if ar_type == "A1":
        # Longest side 256 -> center crop 224 (current baseline)
        max_size = round(256 * scale)
        return A.Compose([
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max(max_size, 224), min_width=max(max_size, 224), border_mode=0),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    elif ar_type == "A2":
        # Longest side 224 -> pad to 224
        max_size = round(224 * scale)
        return A.Compose([
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max(max_size, 224), min_width=max(max_size, 224), border_mode=0),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    elif ar_type == "A3":
        # Longest side 256 -> pad to 256 -> resize 256->224
        max_size = round(256 * scale)
        return A.Compose([
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max(max_size, 224), min_width=max(max_size, 224), border_mode=0),
            A.Resize(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    elif ar_type == "A4":
        # Longest side 256 -> pad 256 -> center crop 224
        max_size = round(256 * scale)
        return A.Compose([
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown AR type: {ar_type}")

class MultiLevelSwin(nn.Module):
    STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        n_stages = len(self.STAGE_CHANNELS)
        self.stage_norms = nn.ModuleDict()
        self.stage_projs = nn.ModuleDict()
        for name, ch in self.STAGE_CHANNELS.items():
            key = name.replace(".", "_")
            self.stage_norms[key] = nn.LayerNorm(ch)
            self.stage_projs[key] = nn.Linear(ch, PROJ_DIM, bias=False)
        self.fusion_head = nn.Sequential(
            nn.Linear(PROJ_DIM * n_stages, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(FUSION_DIM, 1),
        )
        self._stage_feats = {}
        self._hooks = []

    def _register_hooks(self):
        for name in self.STAGE_CHANNELS:
            module = dict(self.backbone.named_modules())[name]
            handle = module.register_forward_hook(lambda mod, inp, out, n=name: self._stage_feats.update({n: out}))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._stage_feats.clear()
        self._register_hooks()
        _ = self.backbone(x)
        self._remove_hooks()
        pooled = []
        for name in self.STAGE_CHANNELS:
            key = name.replace(".", "_")
            feat = self._stage_feats[name].mean(dim=(1, 2))
            feat = self.stage_norms[key](feat)
            feat = self.stage_projs[key](feat)
            pooled.append(feat)
        fused = torch.cat(pooled, dim=-1)
        return self.fusion_head(fused)

class BaseDataset(Dataset):
    def __init__(self, ar_type: str, scales: list):
        self.transforms = [get_transform(ar_type, s) for s in scales]
        self.samples = []

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors = torch.stack([t(image=img)["image"] for t in self.transforms])
        return tensors, s["label"], s["id"]

class TN5000TestDataset(BaseDataset):
    def __init__(self, data_root: str, ar_type: str, scales: list):
        super().__init__(ar_type, scales)
        import xml.etree.ElementTree as ET
        img_dir = os.path.join(data_root, "JPEGImages")
        ann_dir = os.path.join(data_root, "Annotations")
        split_file = os.path.join(data_root, "ImageSets", "Main", "test.txt")
        
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]

        for img_id in ids:
            ann_path = os.path.join(ann_dir, f"{img_id}.xml")
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            tree = ET.parse(ann_path)
            root = tree.getroot()
            obj = root.find("object")
            label = int(obj.find("name").text)
            self.samples.append({"id": f"{img_id}.jpg", "img_path": img_path, "label": label})

class DiveshDataset(BaseDataset):
    def __init__(self, data_root: str, ar_type: str, scales: list):
        super().__init__(ar_type, scales)
        import glob
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        class_0_dir = os.path.join(dataset_dir, "0")
        if os.path.exists(class_0_dir):
            for img_file in glob.glob(os.path.join(class_0_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({"img_path": img_file, "label": 0})
        class_1_dir = os.path.join(dataset_dir, "1")
        if os.path.exists(class_1_dir):
            for img_file in glob.glob(os.path.join(class_1_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({"img_path": img_file, "label": 1})
                    
        for i, s in enumerate(self.samples):
            s["id"] = f"{i:04d}_{os.path.basename(s['img_path'])}"

class KaggleThyroidDataset(BaseDataset):
    def __init__(self, data_root: str, ar_type: str, scales: list):
        super().__init__(ar_type, scales)
        for root, dirs, files in os.walk(data_root):
            dirname = os.path.basename(root).lower().strip()
            label = 0 if dirname in ["benign", "0", "normal"] else (1 if dirname in ["malignant", "1"] else None)
            if label is not None:
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        patient_id = f.split('_')[0]
                        self.samples.append({"id": patient_id, "img_path": os.path.join(root, f), "label": label})

def calculate_metrics(y_true, y_pred_prob):
    y_pred_class = (y_pred_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    
    auc = roc_auc_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    sens = recall_score(y_true, y_pred_class)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "AUC": auc,
        "Accuracy": acc,
        "F1": f1,
        "Sensitivity": sens,
        "Specificity": spec
    }
