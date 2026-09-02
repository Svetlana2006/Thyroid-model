"""
Experiment 02: Cross-dataset duplicate audit.

Computes perceptual hashes (pHash) for all images across TN5000, AUITD, and
the local DDTI-unique dataset. Reports exact duplicates and near-duplicates.

NOTE: The Divesh and full DDTI datasets require kagglehub download.
      This script works with locally available data first.
      Pass --include-kaggle to also download and check external datasets.

Usage:
    python experiments/02_duplicate_audit/run_duplicate_audit.py
    python experiments/02_duplicate_audit/run_duplicate_audit.py --include-kaggle
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUTPUT_DIR = Path("experiments/02_duplicate_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def md5_hash(filepath: str) -> str:
    """Exact file hash."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: Image.Image, hash_size: int = 16) -> int:
    """Difference hash — fast perceptual hash without imagehash dependency."""
    img = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return int(np.packbits(diff.flatten()).tobytes().hex(), 16)


def hamming_distance(h1: int, h2: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def collect_images(root: str, dataset_name: str) -> list:
    """Collect all image paths from a directory tree."""
    samples = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append({
                    "path": os.path.join(dirpath, fn),
                    "dataset": dataset_name,
                    "filename": fn,
                })
    return samples


def hash_all(samples: list) -> list:
    """Compute MD5 and dHash for all samples."""
    results = []
    for i, s in enumerate(samples):
        try:
            img = Image.open(s["path"]).convert("RGB")
            w, h = img.size
            s["md5"] = md5_hash(s["path"])
            s["dhash"] = dhash(img)
            s["width"] = w
            s["height"] = h
            results.append(s)
        except Exception as e:
            print(f"  SKIP {s['path']}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  Hashed {i+1}/{len(samples)}...")
    return results


def find_duplicates(all_samples: list, max_hamming: int = 10):
    """Find exact (MD5) and near (dHash) duplicates between datasets."""
    # Group by dataset
    by_dataset = defaultdict(list)
    for s in all_samples:
        by_dataset[s["dataset"]].append(s)

    datasets = sorted(by_dataset.keys())
    exact_dupes = []
    near_dupes = []

    # Cross-dataset comparisons
    for i, ds_a in enumerate(datasets):
        for ds_b in datasets[i + 1:]:
            print(f"\nComparing {ds_a} ({len(by_dataset[ds_a])}) vs {ds_b} ({len(by_dataset[ds_b])})...")

            # Exact MD5 matches
            md5_b = {s["md5"]: s for s in by_dataset[ds_b]}
            for s_a in by_dataset[ds_a]:
                if s_a["md5"] in md5_b:
                    exact_dupes.append({
                        "source": s_a["path"],
                        "source_dataset": ds_a,
                        "target": md5_b[s_a["md5"]]["path"],
                        "target_dataset": ds_b,
                        "match_type": "exact_md5",
                        "distance": 0,
                    })

            # Near-duplicate dHash matches
            for s_a in by_dataset[ds_a]:
                for s_b in by_dataset[ds_b]:
                    d = hamming_distance(s_a["dhash"], s_b["dhash"])
                    if d <= max_hamming:
                        near_dupes.append({
                            "source": s_a["path"],
                            "source_dataset": ds_a,
                            "target": s_b["path"],
                            "target_dataset": ds_b,
                            "match_type": f"dhash_dist_{d}",
                            "distance": d,
                        })

            n_exact = sum(1 for d in exact_dupes if d["source_dataset"] == ds_a and d["target_dataset"] == ds_b)
            n_near = sum(1 for d in near_dupes if d["source_dataset"] == ds_a and d["target_dataset"] == ds_b)
            print(f"  Exact MD5 matches: {n_exact}")
            print(f"  Near dHash matches (dist <= {max_hamming}): {n_near}")

    return exact_dupes, near_dupes


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset duplicate audit")
    parser.add_argument("--include-kaggle", action="store_true",
                        help="Download and check Divesh/DDTI from Kaggle")
    parser.add_argument("--max-hamming", type=int, default=10,
                        help="Maximum Hamming distance for near-duplicate detection")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 2: Cross-Dataset Duplicate Audit")
    print("=" * 60)

    # Collect local datasets
    all_samples = []

    print("\nCollecting TN5000...")
    tn5000 = collect_images("data_raw/TN5000_forReview/JPEGImages", "TN5000")
    print(f"  {len(tn5000)} images")
    all_samples.extend(tn5000)

    print("Collecting AUITD...")
    auitd = collect_images("data_raw/auitd_dataset", "AUITD")
    print(f"  {len(auitd)} images")
    all_samples.extend(auitd)

    print("Collecting DDTI-unique...")
    ddti = collect_images("data_raw/ddti_unique_dataset", "DDTI_unique")
    print(f"  {len(ddti)} images")
    all_samples.extend(ddti)

    if args.include_kaggle:
        try:
            import kagglehub
            print("Downloading Divesh dataset...")
            divesh_path = kagglehub.dataset_download(
                "diveshzz/thyroid-cancer-classification-ultrasound-dataset"
            )
            divesh = collect_images(divesh_path, "Divesh")
            print(f"  {len(divesh)} images")
            all_samples.extend(divesh)

            print("Downloading full DDTI dataset...")
            ddti_full_path = kagglehub.dataset_download(
                "dasmehdixtr/ddti-thyroid-ultrasound-images"
            )
            ddti_full = collect_images(ddti_full_path, "DDTI_full")
            print(f"  {len(ddti_full)} images")
            all_samples.extend(ddti_full)
        except Exception as e:
            print(f"Kaggle download failed: {e}")

    # Hash everything
    print(f"\nHashing {len(all_samples)} images...")
    all_samples = hash_all(all_samples)
    print(f"Successfully hashed {len(all_samples)} images")

    # Find duplicates
    exact_dupes, near_dupes = find_duplicates(all_samples, args.max_hamming)

    # Save results
    summary = {
        "datasets": {s["dataset"]: 0 for s in all_samples},
        "max_hamming_distance": args.max_hamming,
        "exact_duplicates": len(exact_dupes),
        "near_duplicates": len(near_dupes),
    }
    for s in all_samples:
        summary["datasets"][s["dataset"]] += 1

    with open(OUTPUT_DIR / "duplicate_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results as CSV
    if exact_dupes or near_dupes:
        all_dupes = exact_dupes + near_dupes
        with open(OUTPUT_DIR / "duplicate_matches.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source", "source_dataset", "target", "target_dataset",
                "match_type", "distance"
            ])
            writer.writeheader()
            writer.writerows(all_dupes)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for ds, count in summary["datasets"].items():
        print(f"  {ds}: {count} images")
    print(f"  Exact MD5 duplicates: {len(exact_dupes)}")
    print(f"  Near dHash duplicates (dist <= {args.max_hamming}): {len(near_dupes)}")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
