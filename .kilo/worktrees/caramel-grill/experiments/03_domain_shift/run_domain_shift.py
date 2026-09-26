"""
Experiment 03: Domain shift analysis.

Computes image-level statistics across all datasets and performs
statistical comparisons to quantify domain differences.

Usage:
    python experiments/03_domain_shift/run_domain_shift.py
    python experiments/03_domain_shift/run_domain_shift.py --include-kaggle
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUTPUT_DIR = Path("experiments/03_domain_shift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


def compute_image_stats(img_path: str) -> dict:
    """Compute image-level statistics from a single image."""
    try:
        img = Image.open(img_path).convert("L")  # grayscale for intensity stats
        w, h = img.size
        arr = np.array(img, dtype=np.float64)

        # First-order intensity statistics
        mean_val = arr.mean()
        std_val = arr.std()
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])

        # Entropy
        hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))

        # Edge density (Sobel-like via numpy)
        gy = np.diff(arr, axis=0)
        gx = np.diff(arr, axis=1)
        # Use min of the two shapes
        min_h = min(gy.shape[0], gx.shape[0])
        min_w = min(gy.shape[1], gx.shape[1])
        grad_mag = np.sqrt(gy[:min_h, :min_w]**2 + gx[:min_h, :min_w]**2)
        edge_density = (grad_mag > 30).mean()  # fraction of strong edges

        # High-frequency energy (variance of Laplacian approximation)
        # Using simple second-order difference
        laplacian = (arr[2:, 1:-1] + arr[:-2, 1:-1] + arr[1:-1, 2:] + arr[1:-1, :-2]
                     - 4 * arr[1:-1, 1:-1])
        hf_energy = np.var(laplacian)

        return {
            "width": w,
            "height": h,
            "aspect_ratio_hw": h / w,
            "mean_intensity": mean_val,
            "std_intensity": std_val,
            "q25": q25,
            "median": q50,
            "q75": q75,
            "entropy": entropy,
            "edge_density": edge_density,
            "hf_energy": hf_energy,
        }
    except Exception as e:
        return None


def collect_dataset_stats(root: str, dataset_name: str, max_n: int = None) -> list:
    """Collect stats for all images in a dataset directory."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(dirpath, fn))

    if max_n and len(paths) > max_n:
        np.random.shuffle(paths)
        paths = paths[:max_n]

    print(f"  Computing stats for {len(paths)} images from {dataset_name}...")
    results = []
    for i, p in enumerate(paths):
        s = compute_image_stats(p)
        if s is not None:
            s["dataset"] = dataset_name
            s["path"] = p
            results.append(s)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(paths)} done")

    return results


def compare_distributions(stats_a: list, stats_b: list, feature: str):
    """Compare a feature between two datasets using Mann-Whitney U and effect size."""
    vals_a = np.array([s[feature] for s in stats_a])
    vals_b = np.array([s[feature] for s in stats_b])

    # Mann-Whitney U test
    u_stat, p_val = sp_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")

    # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
    n1, n2 = len(vals_a), len(vals_b)
    r_effect = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "mean_a": float(vals_a.mean()),
        "mean_b": float(vals_b.mean()),
        "std_a": float(vals_a.std()),
        "std_b": float(vals_b.std()),
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "effect_size_r": float(r_effect),
    }


def plot_distributions(all_stats: list, features: list, output_dir: Path):
    """Create distribution comparison plots."""
    datasets = sorted(set(s["dataset"] for s in all_stats))
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    for feat in features:
        fig, ax = plt.subplots(figsize=(8, 4))
        for ds, color in zip(datasets, colors):
            vals = [s[feat] for s in all_stats if s["dataset"] == ds]
            ax.hist(vals, bins=50, alpha=0.5, label=f"{ds} (n={len(vals)})",
                    color=color, density=True)
        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {feat}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"dist_{feat}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-kaggle", action="store_true")
    parser.add_argument("--max-per-dataset", type=int, default=None,
                        help="Max images to sample per dataset (for speed)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 5: Domain Shift Analysis")
    print("=" * 60)

    all_stats = []

    print("\nTN5000:")
    all_stats.extend(collect_dataset_stats(
        "data_raw/TN5000_forReview/JPEGImages", "TN5000", args.max_per_dataset))

    print("AUITD:")
    all_stats.extend(collect_dataset_stats(
        "data_raw/auitd_dataset", "AUITD", args.max_per_dataset))

    if args.include_kaggle:
        try:
            import kagglehub
            print("Divesh (downloading):")
            divesh_path = kagglehub.dataset_download(
                "diveshzz/thyroid-cancer-classification-ultrasound-dataset")
            all_stats.extend(collect_dataset_stats(
                divesh_path, "Divesh", args.max_per_dataset))

            print("DDTI full (downloading):")
            ddti_path = kagglehub.dataset_download(
                "dasmehdixtr/ddti-thyroid-ultrasound-images")
            all_stats.extend(collect_dataset_stats(
                ddti_path, "DDTI_full", args.max_per_dataset))
        except Exception as e:
            print(f"Kaggle download failed: {e}")

    print(f"\nTotal images analyzed: {len(all_stats)}")

    # Save raw stats
    with open(OUTPUT_DIR / "image_stats.csv", "w", newline="") as f:
        fieldnames = list(all_stats[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    # Statistical comparisons
    datasets = sorted(set(s["dataset"] for s in all_stats))
    features = ["mean_intensity", "std_intensity", "entropy", "edge_density",
                 "hf_energy", "aspect_ratio_hw", "width", "height"]

    comparisons = {}
    for i, ds_a in enumerate(datasets):
        for ds_b in datasets[i + 1:]:
            stats_a = [s for s in all_stats if s["dataset"] == ds_a]
            stats_b = [s for s in all_stats if s["dataset"] == ds_b]
            pair_key = f"{ds_a}_vs_{ds_b}"
            comparisons[pair_key] = {}
            for feat in features:
                comparisons[pair_key][feat] = compare_distributions(stats_a, stats_b, feat)

    with open(OUTPUT_DIR / "statistical_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("DOMAIN SHIFT SUMMARY")
    print(f"{'='*60}")
    for pair, feats in comparisons.items():
        print(f"\n{pair}:")
        for feat, vals in feats.items():
            sig = "***" if vals["p_value"] < 0.001 else "**" if vals["p_value"] < 0.01 else "*" if vals["p_value"] < 0.05 else "ns"
            print(f"  {feat:20s}: mean={vals['mean_a']:.2f} vs {vals['mean_b']:.2f}, "
                  f"|r|={abs(vals['effect_size_r']):.3f}, p={vals['p_value']:.4f} {sig}")

    # Plots
    print("\nGenerating distribution plots...")
    plot_distributions(all_stats, features, OUTPUT_DIR)

    # Per-dataset summary
    summary = {}
    for ds in datasets:
        ds_stats = [s for s in all_stats if s["dataset"] == ds]
        summary[ds] = {
            "n_images": len(ds_stats),
        }
        for feat in features:
            vals = np.array([s[feat] for s in ds_stats])
            summary[ds][f"{feat}_mean"] = float(vals.mean())
            summary[ds][f"{feat}_std"] = float(vals.std())

    with open(OUTPUT_DIR / "per_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
