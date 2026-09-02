"""
Script to quantify geometric distortion and verify bounding box alignment
under different preprocessing pipelines.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import cv2
import albumentations as A

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import TN5000Dataset

OUTPUT_DIR = Path("experiments/02_preprocessing_ablation/geometry")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("data_raw/TN5000_forReview")
TRAIN_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "train.txt")

def get_pipelines():
    """Define the validation pipelines with BboxParams enabled."""
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])
    
    # A. Current preprocessing
    pipe_a = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224)
    ], bbox_params=bbox_params)
    
    # B. AR-preserving with CenterCrop (could crop context)
    pipe_b = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.CenterCrop(224, 224)
    ], bbox_params=bbox_params)
    
    # E. AR-preserving full image with explicit letterboxing to 224
    pipe_e = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0)
    ], bbox_params=bbox_params)
    
    return {"A_Current": pipe_a, "B_AR_Preserving": pipe_b, "E_Letterbox_224": pipe_e}

def draw_bbox(image, bbox, color=(255, 0, 0), thickness=2):
    """Draw a bounding box on a numpy image."""
    img_copy = image.copy()
    xmin, ymin, xmax, ymax = map(int, bbox)
    cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, thickness)
    return img_copy

def main():
    print("=" * 60)
    print("Quantifying Geometric Distortion in Preprocessing")
    print("=" * 60)
    
    dataset = TN5000Dataset(str(DATA_ROOT), TRAIN_TXT, transform=None)
    pipelines = get_pipelines()
    
    stats = {name: {"abs_distortion": [], "rel_distortion": [], "original_ar": [], "transformed_ar": []} 
             for name in pipelines.keys()}
    
    valid_bboxes = 0
    discarded_bboxes = {name: 0 for name in pipelines.keys()}
    
    # Track 20 diagnostic images
    diagnostic_count = 0
    os.makedirs(OUTPUT_DIR / "diagnostics", exist_ok=True)
    
    for i, sample in enumerate(dataset.samples):
        img = np.array(PILImage.open(sample["img_path"]).convert("RGB"))
        bbox = sample["bbox"] # [xmin, ymin, xmax, ymax]
        
        orig_w = bbox[2] - bbox[0]
        orig_h = bbox[3] - bbox[1]
        
        if orig_w <= 0 or orig_h <= 0:
            continue
            
        orig_ar = orig_h / orig_w
        
        # Save diagnostics for the first 20 valid images
        save_diagnostic = (diagnostic_count < 20)
        diag_images = {"Original": draw_bbox(img, bbox, color=(0, 255, 0), thickness=4)}
        
        all_pipelines_valid = True
        
        for name, pipe in pipelines.items():
            try:
                transformed = pipe(image=img, bboxes=[bbox], labels=[1])
                
                if not transformed["bboxes"]:
                    # Bbox was cropped completely out
                    discarded_bboxes[name] += 1
                    all_pipelines_valid = False
                    continue
                    
                t_bbox = transformed["bboxes"][0]
                t_w = t_bbox[2] - t_bbox[0]
                t_h = t_bbox[3] - t_bbox[1]
                
                if t_w <= 0 or t_h <= 0:
                    discarded_bboxes[name] += 1
                    all_pipelines_valid = False
                    continue
                    
                t_ar = t_h / t_w
                
                abs_dist = abs(t_ar - orig_ar)
                rel_dist = abs_dist / orig_ar
                
                stats[name]["abs_distortion"].append(abs_dist)
                stats[name]["rel_distortion"].append(rel_dist)
                stats[name]["original_ar"].append(orig_ar)
                stats[name]["transformed_ar"].append(t_ar)
                
                if save_diagnostic:
                    diag_images[name] = draw_bbox(transformed["image"], t_bbox, color=(255, 0, 0))
                    
            except Exception as e:
                discarded_bboxes[name] += 1
                all_pipelines_valid = False
                
        if all_pipelines_valid:
            valid_bboxes += 1
            if save_diagnostic:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(diag_images["Original"])
                axes[0].set_title(f"Original\nAR: {orig_ar:.3f}")
                
                axes[1].imshow(diag_images["A_Current"])
                axes[1].set_title(f"A_Current\nAR: {stats['A_Current']['transformed_ar'][-1]:.3f}")
                
                axes[2].imshow(diag_images["B_AR_Preserving"])
                axes[2].set_title(f"B_AR_Preserving\nAR: {stats['B_AR_Preserving']['transformed_ar'][-1]:.3f}")
                
                axes[3].imshow(diag_images["E_Letterbox_224"])
                axes[3].set_title(f"E_Letterbox_224\nAR: {stats['E_Letterbox_224']['transformed_ar'][-1]:.3f}")
                
                for ax in axes:
                    ax.axis("off")
                
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "diagnostics" / f"diagnostic_{diagnostic_count:02d}.png", dpi=100)
                plt.close(fig)
                diagnostic_count += 1
                
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(dataset.samples)} images...")
            
    print(f"\nProcessed {valid_bboxes} valid bounding boxes.")
    print("Bboxes dropped due to cropping out:")
    for name, dropped in discarded_bboxes.items():
        print(f"  {name}: {dropped} bboxes")
        
    # Generate report and histograms
    report = {}
    
    fig, axes = plt.subplots(1, len(pipelines), figsize=(18, 5))
    
    for idx, (name, s) in enumerate(stats.items()):
        rel_dist = np.array(s["rel_distortion"])
        report[name] = {
            "median_rel_distortion": float(np.median(rel_dist)),
            "mean_rel_distortion": float(np.mean(rel_dist)),
            "max_rel_distortion": float(np.max(rel_dist)),
            "iqr_rel_distortion": float(np.percentile(rel_dist, 75) - np.percentile(rel_dist, 25)),
            "zero_distortion_fraction": float(np.mean(rel_dist < 1e-4))
        }
        
        print(f"\n{name} Distortion:")
        print(f"  Median: {report[name]['median_rel_distortion']:.4f}")
        print(f"  Mean:   {report[name]['mean_rel_distortion']:.4f}")
        print(f"  Max:    {report[name]['max_rel_distortion']:.4f}")
        print(f"  IQR:    {report[name]['iqr_rel_distortion']:.4f}")
        
        # Plot histogram of relative distortion
        ax = axes[idx]
        ax.hist(rel_dist, bins=50, alpha=0.7, color='steelblue')
        ax.set_title(f"{name}\nRelative Distortion")
        ax.set_xlabel("|New AR - Orig AR| / Orig AR")
        ax.set_ylabel("Count")
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distortion_histogram.png", dpi=150)
    plt.close()
    
    with open(OUTPUT_DIR / "distortion_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nResults and diagnostics saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
