import imagehash
from PIL import Image
import glob, os
import kagglehub

def hash_all_images(folder_glob_pattern):
    hashes = {}
    for path in glob.glob(folder_glob_pattern, recursive=True):
        try:
            h = imagehash.phash(Image.open(path).convert("RGB"))
            hashes[path] = h
        except Exception:
            pass
    return hashes

print("Resolving dataset paths...")
ddti_path = kagglehub.dataset_download('dasmehdixtr/ddti-thyroid-ultrasound-images')
diveshzz_path = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')

print("Hashing DDTI...")
ddti_hashes  = hash_all_images(os.path.join(ddti_path, "**", "*.jpg"))
print("Hashing Diveshzz...")
diveshzz_hashes = hash_all_images(os.path.join(diveshzz_path, "**", "*.jpg"))
print("Hashing AU-ITD...")
auitd_hashes = hash_all_images(os.path.join("data_raw", "auitd_dataset", "**", "*.jpg"))
print("Hashing TN5000...")
tn5000_hashes = hash_all_images(os.path.join("data_raw", "TN5000_forReview", "JPEGImages", "**", "*.jpg"))

def find_near_duplicates(set_a, set_b, max_distance=5):
    matches = []
    for path_a, hash_a in set_a.items():
        for path_b, hash_b in set_b.items():
            if hash_a - hash_b <= max_distance:   # Hamming distance
                matches.append((path_a, path_b, hash_a - hash_b))
    return matches

print("Finding near-duplicates...")
ddti_vs_auitd = find_near_duplicates(ddti_hashes, auitd_hashes)
ddti_vs_tn5000 = find_near_duplicates(ddti_hashes, tn5000_hashes)

diveshzz_vs_auitd = find_near_duplicates(diveshzz_hashes, auitd_hashes)
diveshzz_vs_tn5000 = find_near_duplicates(diveshzz_hashes, tn5000_hashes)

print("\n--- DDTI Analysis ---")
print(f"DDTI images: {len(ddti_hashes)}")
print(f"DDTI vs AU-ITD near-duplicates: {len(ddti_vs_auitd)}")
print(f"DDTI vs TN5000 near-duplicates: {len(ddti_vs_tn5000)}")

overlap_set = {m[0] for m in ddti_vs_auitd} | {m[0] for m in ddti_vs_tn5000}
if len(ddti_hashes) > 0:
    overlap_pct = (len(overlap_set) / len(ddti_hashes)) * 100
else:
    overlap_pct = 0.0
print(f"% of DDTI overlapping with training pool: {overlap_pct:.2f}%")

print("\n--- Diveshzz Analysis ---")
print(f"Diveshzz images: {len(diveshzz_hashes)}")
print(f"Diveshzz vs AU-ITD near-duplicates: {len(diveshzz_vs_auitd)}")
print(f"Diveshzz vs TN5000 near-duplicates: {len(diveshzz_vs_tn5000)}")

overlap_set_divesh = {m[0] for m in diveshzz_vs_auitd} | {m[0] for m in diveshzz_vs_tn5000}
if len(diveshzz_hashes) > 0:
    overlap_pct_divesh = (len(overlap_set_divesh) / len(diveshzz_hashes)) * 100
else:
    overlap_pct_divesh = 0.0
print(f"% of Diveshzz overlapping with training pool: {overlap_pct_divesh:.2f}%")
