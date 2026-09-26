"""
Experiment 12: Image Frequency Analysis
Calculates low, mid, and high-frequency energy for datasets to test if they differ
systematically in spatial-frequency characteristics.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from scipy import stats as sp_stats

OUTPUT_DIR = Path("experiments/12_frequency_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_radial_profile(data, center):
    """Compute radial profile of 2D image."""
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def analyze_frequencies(img_path):
    """Extract LF, MF, and HF energies via 2D FFT."""
    try:
        img = PILImage.open(img_path).convert("L")
        img = img.resize((256, 256), PILImage.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        
        # 2D FFT
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) ** 2
        
        # Radial profile
        center = (128, 128)
        rad_profile = compute_radial_profile(magnitude_spectrum, center)
        
        # Define bands based on radius (max radius ~ 128)
        # Total bins ~ 181 (sqrt(128^2 + 128^2))
        low_band = rad_profile[1:20]   # Skip DC (index 0)
        mid_band = rad_profile[20:60]
        high_band = rad_profile[60:]
        
        total_energy = np.sum(rad_profile[1:])
        if total_energy == 0:
            return None
            
        return {
            "lf_energy_frac": float(np.sum(low_band) / total_energy),
            "mf_energy_frac": float(np.sum(mid_band) / total_energy),
            "hf_energy_frac": float(np.sum(high_band) / total_energy),
        }
    except Exception:
        return None

def main():
    print("=" * 60)
    print("Image Spatial Frequency Analysis")
    print("=" * 60)
    
    datasets = {
        "TN5000": [str(p) for p in Path("data_raw/TN5000_forReview/JPEGImages").glob("*.jpg")],
        "AUITD": []
    }
    
    for root, _, files in os.walk("data_raw/auitd_dataset"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                datasets["AUITD"].append(os.path.join(root, f))
                
    try:
        import kagglehub
        divesh_path = kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")
        divesh_files = []
        for root, _, files in os.walk(divesh_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    divesh_files.append(os.path.join(root, f))
        datasets["Divesh"] = divesh_files
    except Exception as e:
        print(f"Warning: Could not load Divesh: {e}")

    np.random.seed(42)
    stats = {}
    
    for ds_name, paths in datasets.items():
        if len(paths) > 1000:
            paths = np.random.choice(paths, 1000, replace=False).tolist()
            
        print(f"Analyzing {len(paths)} images from {ds_name}...")
        ds_stats = {"lf": [], "mf": [], "hf": []}
        
        for p in paths:
            res = analyze_frequencies(p)
            if res:
                ds_stats["lf"].append(res["lf_energy_frac"])
                ds_stats["mf"].append(res["mf_energy_frac"])
                ds_stats["hf"].append(res["hf_energy_frac"])
                
        stats[ds_name] = ds_stats
        print(f"  LF: {np.mean(ds_stats['lf']):.4f} ± {np.std(ds_stats['lf']):.4f}")
        print(f"  MF: {np.mean(ds_stats['mf']):.4f} ± {np.std(ds_stats['mf']):.4f}")
        print(f"  HF: {np.mean(ds_stats['hf']):.4f} ± {np.std(ds_stats['hf']):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bands = [("lf", "Low Frequency"), ("mf", "Mid Frequency"), ("hf", "High Frequency")]
    
    for i, (band_key, band_name) in enumerate(bands):
        data_to_plot = [stats[ds][band_key] for ds in datasets.keys()]
        axes[i].boxplot(data_to_plot, labels=list(datasets.keys()))
        axes[i].set_title(band_name + " Energy Fraction")
        axes[i].set_ylabel("Fraction of Total Energy")
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "frequency_comparison.png", dpi=150)
    plt.close()
    
    # Statistical tests (Mann-Whitney U) vs TN5000
    report = {}
    for ds_name in datasets.keys():
        if ds_name == "TN5000":
            continue
        report[f"TN5000_vs_{ds_name}"] = {}
        for band_key, _ in bands:
            u_stat, p_val = sp_stats.mannwhitneyu(stats["TN5000"][band_key], stats[ds_name][band_key])
            n1, n2 = len(stats["TN5000"][band_key]), len(stats[ds_name][band_key])
            r = 1 - (2 * u_stat) / (n1 * n2)
            report[f"TN5000_vs_{ds_name}"][band_key] = {
                "p_value": float(p_val),
                "effect_size_r": float(r)
            }
            
    with open(OUTPUT_DIR / "frequency_stats.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
