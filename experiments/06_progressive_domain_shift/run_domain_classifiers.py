"""
Experiments 6, 7, 8: Rigorous Domain Shift Classification

Evaluates whether the TN5000 vs AUITD domain gap is trivial (metadata/UI) or
fundamental to the image representation, using progressive standardization.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import TN5000Dataset, AUITDDataset
from src.models import build_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = Path("experiments/06_progressive_domain_shift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("data_raw/TN5000_forReview")

def run_cv_classification(X, y, classifier_name, clf):
    """Run rigorous 5-fold Stratified CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    fold_accs = []
    fold_f1s = []
    cms = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        fold_accs.append(accuracy_score(y_test, preds))
        fold_f1s.append(f1_score(y_test, preds, average="macro"))
        cms.append(confusion_matrix(y_test, preds))
        
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_f1 = float(np.mean(fold_f1s))
    
    # Aggregate CM
    total_cm = np.sum(cms, axis=0)
    
    return {
        "classifier": classifier_name,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_lower": mean_acc - 1.96 * (std_acc / np.sqrt(5)),
        "ci_upper": mean_acc + 1.96 * (std_acc / np.sqrt(5)),
        "mean_macro_f1": mean_f1,
        "confusion_matrix": total_cm.tolist()
    }

class DomainDataset(Dataset):
    def __init__(self, tn5000_paths, auitd_paths, transform=None, crop_borders=False):
        self.samples = []
        for p in tn5000_paths:
            self.samples.append({"path": p, "domain": 0})
        for p in auitd_paths:
            self.samples.append({"path": p, "domain": 1})
        self.transform = transform
        self.crop_borders = crop_borders
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(PILImage.open(s["path"]).convert("RGB"))
        
        if self.crop_borders:
            # Crop outer 15% to aggressively remove UI/text/logos
            h, w = img.shape[:2]
            crop_y, crop_x = int(h * 0.15), int(w * 0.15)
            img = img[crop_y:h-crop_y, crop_x:w-crop_x]
            
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, s["domain"]

def extract_deep_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, doms in dataloader:
            imgs = imgs.to(device)
            # Forward through backbone only
            feats = model.backbone(imgs)
            if feats.dim() > 2:
                feats = feats.mean(dim=[2, 3]) # Global average pool
            features.append(feats.cpu().numpy())
            labels.append(doms.numpy())
    return np.concatenate(features), np.concatenate(labels)

def get_conditions():
    conds = {}
    
    # Condition 1: Raw (Squashed to 224, no norm, no border crop)
    conds["1_Raw_Squashed"] = (
        A.Compose([A.Resize(224, 224), ToTensorV2()]),
        False
    )
    
    # Condition 2: AR-preserving (no norm, no border crop)
    conds["2_AR_Preserving"] = (
        A.Compose([
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
            ToTensorV2()
        ]),
        False
    )
    
    # Condition 3: AR-preserving + Normalization
    conds["3_AR_Normalized"] = (
        A.Compose([
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        False
    )
    
    # Condition 4: Border crop (UI removal) + AR-preserving + Normalization
    conds["4_UI_Removed_Normalized"] = (
        A.Compose([
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        True
    )
    return conds

def main():
    print("=" * 60)
    print("Rigorous Domain Shift Classification")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get 1000 random samples from each to balance and speed up
    np.random.seed(42)
    tn5000_all = [str(p) for p in (DATA_ROOT / "JPEGImages").glob("*.jpg")]
    auitd_all = []
    for root, _, files in os.walk("data_raw/auitd_dataset"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                auitd_all.append(os.path.join(root, f))
                
    tn5000_sub = np.random.choice(tn5000_all, 1000, replace=False).tolist()
    auitd_sub = np.random.choice(auitd_all, 1000, replace=False).tolist()
    
    print(f"Sampled 1000 TN5000 and 1000 AUITD images.")
    
    results = {}
    
    # Task 6 & 8: Progressive Standardization using ResNet50
    print("\n--- Running Progressive Standardization (ResNet50 Deep Features) ---")
    
    # Load pretrained resnet50 (ImageNet weights are fine for extracting generic deep features,
    # but we can also use our trained checkpoints. Let's use a trained checkpoint to see what it learned).
    ckpt_path = "outputs/checkpoints/resnet50_seed0_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model("resnet50").to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded trained ResNet50.")
    else:
        print("Checkpoint not found, using raw untrained ResNet50.")
        model = build_model("resnet50").to(device)
        
    conditions = get_conditions()
    
    for cond_name, (transform, crop_borders) in conditions.items():
        print(f"  Evaluating Condition: {cond_name}")
        ds = DomainDataset(tn5000_sub, auitd_sub, transform, crop_borders)
        # Convert images to float explicitly to avoid ByteTensor normalization issues if Normalize is absent
        def collate_fn(batch):
            imgs = torch.stack([b[0].float() if b[0].dtype == torch.uint8 else b[0] for b in batch])
            labels = torch.tensor([b[1] for b in batch])
            return imgs, labels
            
        dl = DataLoader(ds, batch_size=32, num_workers=0, collate_fn=collate_fn)
        
        X, y = extract_deep_features(model, dl, device)
        
        lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        res = run_cv_classification(X, y, f"ResNet50_{cond_name}", lr)
        results[cond_name] = res
        print(f"    Accuracy: {res['mean_accuracy']:.4f} (95% CI: {res['ci_lower']:.4f}-{res['ci_upper']:.4f})")
        print(f"    Macro F1: {res['mean_macro_f1']:.4f}")

    with open(OUTPUT_DIR / "progressive_domain_shift.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {OUTPUT_DIR}/progressive_domain_shift.json")

if __name__ == "__main__":
    main()
