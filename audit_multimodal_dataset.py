"""
Dataset Audit and Leakage Check for the Multimodal Machine Learning Model Dataset

This script verifies data leakage / exact hash overlap of the newly extracted
multimodal dataset against all existing internal and external datasets:
1. Internal TN5000
2. Internal AUITD
3. External Diveshzz
4. External Thyroid for Pretraining
"""

import os
import hashlib
import glob
from collections import defaultdict
import kagglehub

def get_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def hash_dataset(paths):
    hashes = {}
    for p in paths:
        h = get_hash(p)
        if h:
            hashes[h] = p
    return hashes

def get_image_paths(root_dir):
    paths = []
    if os.path.exists(root_dir):
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    paths.append(os.path.join(root, f))
    return paths

def check_overlaps():
    multimodal_root = r"data_raw/multimodal_dataset"
    tn5000_root = r"data_raw/TN5000_forReview/JPEGImages"
    auitd_root = r"data_raw/auitd_dataset/dataset thyroid"
    
    print("--- DOWNLOADING / LOCATING EXISTING DATASETS ---")
    print("Downloading/Locating Diveshzz dataset...")
    divesh_root = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
    
    print("Downloading/Locating Thyroid for Pretraining dataset...")
    kaggle_pretrain_root = kagglehub.dataset_download('tingzen/thyroid-for-pretraining')
    
    print("\n--- NEW MULTIMODAL DATASET ANALYSIS ---")
    multimodal_paths = get_image_paths(multimodal_root)
    print(f"Discovered {len(multimodal_paths)} images in the newly extracted multimodal dataset.")
    
    if len(multimodal_paths) == 0:
        print("WARNING: No images found. Has the ZIP finished extracting, or is the directory structure different?")
        return

    print("Calculating hashes for the Multimodal dataset...")
    multimodal_hashes = hash_dataset(multimodal_paths)
    print(f"Total unique hashes in target: {len(multimodal_hashes)}")
    
    # 1. TN5000
    print("\n[1] Checking TN5000...")
    tn5000_paths = get_image_paths(tn5000_root)
    tn5000_hashes = hash_dataset(tn5000_paths)
    print(f"Loaded {len(tn5000_paths)} TN5000 files, unique hashes: {len(tn5000_hashes)}")
    overlap_tn5000 = set(multimodal_hashes.keys()).intersection(tn5000_hashes.keys())
    print(f"Exact Hash Overlap with TN5000: {len(overlap_tn5000)}")
    
    # 2. AUITD
    print("\n[2] Checking AUITD...")
    auitd_paths = get_image_paths(auitd_root)
    auitd_hashes = hash_dataset(auitd_paths)
    print(f"Loaded {len(auitd_paths)} AUITD files, unique hashes: {len(auitd_hashes)}")
    overlap_auitd = set(multimodal_hashes.keys()).intersection(auitd_hashes.keys())
    print(f"Exact Hash Overlap with AUITD: {len(overlap_auitd)}")

    # 3. Diveshzz
    print("\n[3] Checking Diveshzz...")
    divesh_paths = get_image_paths(divesh_root)
    divesh_hashes = hash_dataset(divesh_paths)
    print(f"Loaded {len(divesh_paths)} Diveshzz files, unique hashes: {len(divesh_hashes)}")
    overlap_divesh = set(multimodal_hashes.keys()).intersection(divesh_hashes.keys())
    print(f"Exact Hash Overlap with Diveshzz: {len(overlap_divesh)}")

    # 4. Thyroid for Pretraining (Kaggle)
    print("\n[4] Checking Thyroid for Pretraining (Kaggle)...")
    pretrain_paths = get_image_paths(kaggle_pretrain_root)
    pretrain_hashes = hash_dataset(pretrain_paths)
    print(f"Loaded {len(pretrain_paths)} Kaggle Pretrain files, unique hashes: {len(pretrain_hashes)}")
    overlap_pretrain = set(multimodal_hashes.keys()).intersection(pretrain_hashes.keys())
    print(f"Exact Hash Overlap with Thyroid for Pretraining: {len(overlap_pretrain)}")

    print("\n" + "="*50)
    total_overlaps = len(overlap_tn5000) + len(overlap_auitd) + len(overlap_divesh) + len(overlap_pretrain)
    if total_overlaps == 0:
        print("VERDICT: DATASET IS SAFE TO USE AS EXTERNAL VALIDATION.")
        print("No exact hash overlaps found with ANY of your training or prior validation datasets.")
    else:
        print(f"WARNING: TOTAL OVERLAPS DETECTED: {total_overlaps}! DATASET IS COMPROMISED.")
        if len(overlap_tn5000) > 0: print(f"- Collisions with TN5000: {len(overlap_tn5000)}")
        if len(overlap_auitd) > 0: print(f"- Collisions with AUITD: {len(overlap_auitd)}")
        if len(overlap_divesh) > 0: print(f"- Collisions with Diveshzz: {len(overlap_divesh)}")
        if len(overlap_pretrain) > 0: print(f"- Collisions with Kaggle Pretrain: {len(overlap_pretrain)}")
    print("="*50)

if __name__ == "__main__":
    check_overlaps()
