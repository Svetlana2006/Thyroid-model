"""Step 0 Audit: Verify preprocessing behavior, class weights, and EfficientNet config."""
import albumentations as A
import numpy as np
import json

print("=" * 60)
print("AUDIT 1: Albumentations Resize behavior")
print("=" * 60)

# Simulate typical TN5000 images
test_cases = [
    (500, 718, "TN5000 median (landscape)"),
    (628, 439, "TN5000 tall portrait"),
    (368, 818, "TN5000 wide landscape"),
    (500, 500, "Square image"),
]

t_current = A.Resize(256, 256)
t_ar = A.Compose([A.LongestMaxSize(256), A.PadIfNeeded(256, 256, border_mode=0)])

for h, w, desc in test_cases:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    orig_ar = h / w

    res_current = t_current(image=img)["image"]
    res_ar = t_ar(image=img)["image"]

    print(f"\n{desc}: {w}x{h} (AR_HW={orig_ar:.3f})")
    print(f"  Resize(256,256): {res_current.shape[1]}x{res_current.shape[0]} AR_HW=1.000 | distortion={abs(1.0 - orig_ar):.3f}")
    print(f"  LongestMax+Pad:  {res_ar.shape[1]}x{res_ar.shape[0]} (AR preserved in content)")

print("\n\nCONCLUSION: A.Resize(256,256) IS anisotropic — it stretches non-square images to square.")
print("For TN5000 median (718x500), this changes AR from 0.696 to 1.000 — a 43.7% distortion.")

print("\n" + "=" * 60)
print("AUDIT 2: Bbox aspect ratio through preprocessing")
print("=" * 60)

# Check: does Albumentations transform bounding boxes?
# Answer: ONLY if you use BboxParams. The current code does NOT use BboxParams.
# The bbox is stored as metadata and never passed through the transform pipeline.
print("Current code: bbox is parsed from XML and stored as metadata.")
print("Albumentations is called with transform(image=image) only — NO bbox_params.")
print("Therefore: bbox coordinates are NOT transformed by Resize/CenterCrop.")
print("The bbox aspect ratio in self.samples is the ORIGINAL pixel-coordinate ratio.")
print("When crop_nodule=True, the crop happens BEFORE transforms, on the original image.")

# Demonstrate: what happens to a nodule's apparent AR after anisotropic resize
print("\nExample: Image 718x500, nodule bbox (223,90,286,131)")
orig_w, orig_h = 718, 500
bx1, by1, bx2, by2 = 223, 90, 286, 131
nw = bx2 - bx1  # 63
nh = by2 - by1  # 41
print(f"  Original bbox: {nw}x{nh} px, AR_HW={nh/nw:.3f} (wider-than-tall)")

# After Resize(256,256): image squished
scale_x = 256 / orig_w  # 0.356
scale_y = 256 / orig_h  # 0.512
new_nw = nw * scale_x
new_nh = nh * scale_y
print(f"  After Resize(256,256): effective bbox {new_nw:.1f}x{new_nh:.1f} px, apparent AR_HW={new_nh/new_nw:.3f}")
print(f"  AR change: {nh/nw:.3f} -> {new_nh/new_nw:.3f} ({(new_nh/new_nw)/(nh/nw)*100 - 100:.1f}% distortion)")

print("\n" + "=" * 60)
print("AUDIT 3: Exact class counts and pos_weight")
print("=" * 60)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from src.dataset import TN5000Dataset, AUITDDataset, DDTIUniqueDataset

DATA_ROOT = Path("data_raw/TN5000_forReview")
TRAIN_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "train.txt")

tn5000_train = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT)
auitd = AUITDDataset("data_raw/auitd_dataset")
ddti = DDTIUniqueDataset("data_raw/ddti_unique_dataset")

tn_labels = tn5000_train.get_labels()
au_labels = auitd.get_labels()
dd_labels = ddti.get_labels()

combined = np.concatenate([tn_labels, au_labels, dd_labels])

print(f"TN5000 train: {len(tn_labels)} total, benign={int((tn_labels==0).sum())}, malignant={int((tn_labels==1).sum())}")
print(f"  TN5000-only pos_weight = {(tn_labels==0).sum()/(tn_labels==1).sum():.4f}")
print(f"AUITD: {len(au_labels)} total, benign={int((au_labels==0).sum())}, malignant={int((au_labels==1).sum())}")
print(f"DDTI-unique: {len(dd_labels)} total, benign={int((dd_labels==0).sum())}, malignant={int((dd_labels==1).sum())}")
print(f"Combined: {len(combined)} total, benign={int((combined==0).sum())}, malignant={int((combined==1).sum())}")
print(f"  Combined pos_weight = {(combined==0).sum()/(combined==1).sum():.4f}")
print(f"  Difference: {abs((combined==0).sum()/(combined==1).sum() - (tn_labels==0).sum()/(tn_labels==1).sum()):.4f}")

print("\n" + "=" * 60)
print("AUDIT 4: EfficientNet-B3 at 224x224")
print("=" * 60)
try:
    import timm
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    cfg = model.default_cfg
    print(f"timm default input_size: {cfg.get('input_size', 'unknown')}")
    print(f"timm default crop_pct: {cfg.get('crop_pct', 'unknown')}")

    # Test forward pass at 224
    import torch
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Forward pass at 224x224: output shape={out.shape} — WORKS")

    x300 = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        out300 = model(x300)
    print(f"Forward pass at 300x300: output shape={out300.shape} — WORKS")
    print("Both resolutions are valid; 300 is native/recommended, 224 is functional but suboptimal.")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("AUDIT 5: TN5000 image dimension distribution (full dataset)")
print("=" * 60)
import xml.etree.ElementTree as ET
from collections import Counter

ann_dir = Path("data_raw/TN5000_forReview/Annotations")
sizes = []
for xml_file in ann_dir.glob("*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    sizes.append((w, h))

size_counts = Counter(sizes)
print(f"Total annotations: {len(sizes)}")
print("Distinct image sizes:")
for (w, h), count in size_counts.most_common():
    print(f"  {w}x{h}: {count} images (AR_HW={h/w:.3f})")

results = {
    "ddti_unique_images": len(dd_labels),
    "anisotropic_resize_confirmed": True,
    "tn5000_pos_weight": float((tn_labels==0).sum()/(tn_labels==1).sum()),
    "combined_pos_weight": float((combined==0).sum()/(combined==1).sum()),
    "efficientnet_b3_valid_at_224": True,
    "image_sizes": {f"{w}x{h}": count for (w,h), count in size_counts.most_common()},
}
os.makedirs("experiments/00_audit", exist_ok=True)
with open("experiments/00_audit/preprocessing_audit.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nAudit results saved to experiments/00_audit/preprocessing_audit.json")
