"""
Experiment 05: Occlusion / perturbation testing.

Systematically perturbs TN5000 test images and measures how each
perturbation affects model predictions. This identifies what image
regions/information each architecture depends on.

IMPORTANT: Positive ΔAUC means the perturbation REDUCED model performance
(the model depended on the removed information). This is supporting evidence,
not proof of what specific clinical features the model uses.

Usage:
    python experiments/05_occlusion/run_occlusion.py
    python experiments/05_occlusion/run_occlusion.py --arch swin_tiny --seed 0
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from scipy import ndimage
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset import TN5000Dataset
from src.models import build_model
from src.transforms import get_val_transforms

OUTPUT_DIR = Path("experiments/05_occlusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("data_raw/TN5000_forReview")
TEST_TXT = str(DATA_ROOT / "ImageSets" / "Main" / "test.txt")


def create_perturbations(image_np: np.ndarray, bbox: tuple,
                         img_w: int, img_h: int) -> dict:
    """
    Create perturbation variants including decomposed scale/context and boundary tests.
    """
    xmin, ymin, xmax, ymax = bbox
    h, w = image_np.shape[:2]
    perturbations = {}
    
    # Base/Shared stats
    mean_val = image_np.mean(axis=(0, 1)).astype(np.uint8)
    nodule_w, nodule_h = max(1, xmax - xmin), max(1, ymax - ymin)
    bbox_diameter = np.sqrt(nodule_w**2 + nodule_h**2)

    # -------------------------------------------------------------
    # Task 9: Decomposed scale / context experiment
    # -------------------------------------------------------------
    # A. Original full image
    perturbations["9A_original"] = image_np.copy()

    # B. Same full image, mask everything outside fixed bbox + 20% padding
    pad_x, pad_y = int(nodule_w * 0.2), int(nodule_h * 0.2)
    crop_xmin, crop_ymin = max(0, xmin - pad_x), max(0, ymin - pad_y)
    crop_xmax, crop_ymax = min(w, xmax + pad_x), min(h, ymax + pad_y)
    
    img_mask_outside = np.full_like(image_np, fill_value=mean_val)
    img_mask_outside[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = \
        image_np[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    perturbations["9B_mask_outside"] = img_mask_outside
    
    # C. Nodule crop resized to preserve original apparent scale (Letterboxed on mean background)
    img_preserve_scale = np.full_like(image_np, fill_value=mean_val)
    img_preserve_scale[ymin:ymax, xmin:xmax] = image_np[ymin:ymax, xmin:xmax]
    perturbations["9C_preserve_scale"] = img_preserve_scale
    
    # D. Nodule crop resized to standardized scale (Tight crop resized to full image)
    nodule_crop = image_np[ymin:ymax, xmin:xmax].copy()
    crop_pil = PILImage.fromarray(nodule_crop).resize((w, h), PILImage.BILINEAR)
    perturbations["9D_standardized_scale"] = np.array(crop_pil)
    
    # E. Same nodule region but preserve surrounding context (this is effectively the original image 
    # if you just crop it normally in the pipeline, but we will pass it as a tight crop padded with context)
    # Actually, E just means keeping the context intact, which is 9A. The user wants to separate scale from context.
    # We will provide a 50% padded crop resized to full frame to see if local context helps.
    pad_x50, pad_y50 = int(nodule_w * 0.5), int(nodule_h * 0.5)
    cx50_min, cy50_min = max(0, xmin - pad_x50), max(0, ymin - pad_y50)
    cx50_max, cy50_max = min(w, xmax + pad_x50), min(h, ymax + pad_y50)
    context_crop = image_np[cy50_min:cy50_max, cx50_min:cx50_max].copy()
    context_pil = PILImage.fromarray(context_crop).resize((w, h), PILImage.BILINEAR)
    perturbations["9E_local_context"] = np.array(context_pil)

    # -------------------------------------------------------------
    # Task 10: Improved Boundary Experiment
    # -------------------------------------------------------------
    # Helper for annular blurring/masking
    def apply_boundary_effect(img, effect_type, width_frac):
        margin = int(bbox_diameter * width_frac)
        margin = max(3, margin)
        
        result = img.copy()
        
        # Region to process
        r_ymin, r_ymax = max(0, ymin - margin), min(h, ymax + margin)
        r_xmin, r_xmax = max(0, xmin - margin), min(w, xmax + margin)
        region = img[r_ymin:r_ymax, r_xmin:r_xmax].copy()
        
        if effect_type == "blur":
            processed = np.zeros_like(region)
            for c in range(3):
                processed[:,:,c] = ndimage.gaussian_filter(region[:,:,c], sigma=5)
        elif effect_type == "mask_mean":
            processed = np.full_like(region, fill_value=mean_val)
            
        # Create annular mask (True for boundary band, False for interior and far exterior)
        mask = np.ones((r_ymax-r_ymin, r_xmax-r_xmin), dtype=bool)
        
        inner_ymin, inner_ymax = margin, (r_ymax - r_ymin) - margin
        inner_xmin, inner_xmax = margin, (r_xmax - r_xmin) - margin
        if inner_ymax > inner_ymin and inner_xmax > inner_xmin:
            mask[inner_ymin:inner_ymax, inner_xmin:inner_xmax] = False
            
        # Apply
        for c in range(3):
            result[r_ymin:r_ymax, r_xmin:r_xmax, c] = np.where(mask, processed[:,:,c], region[:,:,c])
        return result

    # B1: Very narrow boundary blur (5%)
    perturbations["10B1_narrow_blur"] = apply_boundary_effect(image_np, "blur", 0.05)
    
    # B2: Moderate-width annular boundary blur (15%)
    perturbations["10B2_moderate_blur"] = apply_boundary_effect(image_np, "blur", 0.15)
    
    # B3: Mask boundary (mean replacement) (15%)
    perturbations["10B3_mask_boundary"] = apply_boundary_effect(image_np, "mask_mean", 0.15)
    
    # B4: Remove/blur the entire peripheral nodule band (25%)
    perturbations["10B4_wide_blur"] = apply_boundary_effect(image_np, "blur", 0.25)
    
    # Original mask nodule for reference
    img_masked_nodule = image_np.copy()
    img_masked_nodule[ymin:ymax, xmin:xmax] = mean_val
    perturbations["ref_mask_nodule"] = img_masked_nodule

    return perturbations


def evaluate_perturbation(model, perturbed_images, labels, transform, device,
                          batch_size=32):
    """Evaluate model on a set of perturbed images."""
    model.eval()
    all_logits = []

    for i in range(0, len(perturbed_images), batch_size):
        batch_imgs = perturbed_images[i:i + batch_size]
        tensors = []
        for img in batch_imgs:
            t = transform(image=img)["image"]
            tensors.append(t)
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            logits = model(batch).squeeze(-1)
            all_logits.extend(logits.cpu().numpy().tolist())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(labels)

    try:
        auc = roc_auc_score(labels_arr, logits_arr)
    except ValueError:
        auc = float("nan")

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -500, 500)))
    return {
        "auc": auc,
        "logits": logits_arr,
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default=None,
                        help="Single architecture to test (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit test images for speed")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 7: Occlusion / Perturbation Testing")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = get_val_transforms()

    # Load test dataset with bboxes
    dataset = TN5000Dataset(str(DATA_ROOT), TEST_TXT)

    if args.max_images:
        dataset.samples = dataset.samples[:args.max_images]
    print(f"Test images: {len(dataset.samples)}")

    # Prepare images and perturbations
    print("Preparing perturbation variants...")
    perturbation_names = [
        "9A_original", "9B_mask_outside", "9C_preserve_scale", 
        "9D_standardized_scale", "9E_local_context",
        "10B1_narrow_blur", "10B2_moderate_blur", 
        "10B3_mask_boundary", "10B4_wide_blur",
        "ref_mask_nodule"
    ]

    # Store per-perturbation images and labels
    perturbed_sets = {name: [] for name in perturbation_names}
    labels = []

    for i, sample in enumerate(dataset.samples):
        img = np.array(PILImage.open(sample["img_path"]).convert("RGB"))
        bbox = sample["bbox"]
        label = sample["label"]
        labels.append(label)

        perts = create_perturbations(img, bbox, img.shape[1], img.shape[0])
        for name in perturbation_names:
            perturbed_sets[name].append(perts[name])

        if (i + 1) % 200 == 0:
            print(f"  Prepared {i+1}/{len(dataset.samples)}")

    # Evaluate each architecture
    archs = [args.arch] if args.arch else ["resnet50", "efficientnet_b3", "swin_tiny"]
    all_results = {}

    for arch in archs:
        seed = args.seed
        ckpt_path = f"outputs/checkpoints/{arch}_seed{seed}_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"Skipping {arch} seed{seed} (no checkpoint)")
            continue

        print(f"\nEvaluating {arch} seed{seed}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        model = build_model(arch, dropout=config.get("dropout", 0.3)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        arch_results = {}
        baseline_probs = None

        for pert_name in perturbation_names:
            print(f"  Perturbation: {pert_name}...", end=" ")
            result = evaluate_perturbation(
                model, perturbed_sets[pert_name], labels, transform, device)

            if pert_name == "9A_original":
                baseline_probs = result["probs"]
                baseline_auc = result["auc"]

            # Compare to baseline
            if baseline_probs is not None and pert_name != "9A_original":
                delta_auc = baseline_auc - result["auc"]
                prob_diff = np.abs(result["probs"] - baseline_probs)
                mean_abs_prob_change = float(prob_diff.mean())

                # Prediction flips (using 0.5 threshold)
                orig_preds = (baseline_probs >= 0.5).astype(int)
                pert_preds = (result["probs"] >= 0.5).astype(int)
                flip_rate = float((orig_preds != pert_preds).mean())

                # Rank correlation
                from scipy.stats import spearmanr
                rho, _ = spearmanr(baseline_probs, result["probs"])

                arch_results[pert_name] = {
                    "auc": float(result["auc"]),
                    "delta_auc": float(delta_auc),
                    "mean_abs_prob_change": mean_abs_prob_change,
                    "prediction_flip_rate": flip_rate,
                    "rank_correlation": float(rho),
                }
                print(f"AUC={result['auc']:.4f}, dAUC={delta_auc:+.4f}, "
                      f"flip={flip_rate:.3f}")
            else:
                arch_results[pert_name] = {
                    "auc": float(result["auc"]),
                }
                print(f"AUC={result['auc']:.4f} (baseline)")

        all_results[f"{arch}_seed{seed}"] = arch_results

    # Save results
    with open(OUTPUT_DIR / "occlusion_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print("OCCLUSION RESULTS SUMMARY")
    print(f"{'='*80}")
    header = f"{'Perturbation':<25} "
    for arch_key in all_results:
        header += f"{'dAUC(' + arch_key.split('_seed')[0] + ')':>18} "
    print(header)
    print("-" * 80)

    for pert in perturbation_names[1:]:
        row = f"{pert:<25} "
        for arch_key in all_results:
            if pert in all_results[arch_key]:
                da = all_results[arch_key][pert]["delta_auc"]
                row += f"{da:>+18.4f} "
            else:
                row += f"{'N/A':>18} "
        print(row)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
