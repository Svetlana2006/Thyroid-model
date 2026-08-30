import imagehash
from PIL import Image
import glob, os, shutil
import xml.etree.ElementTree as ET
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

print("Hashing AU-ITD to check for duplicates...")
auitd_hashes = hash_all_images(os.path.join("data_raw", "auitd_dataset", "**", "*.jpg"))

print("Resolving DDTI path...")
ddti_path = kagglehub.dataset_download('dasmehdixtr/ddti-thyroid-ultrasound-images')

out_dir = os.path.join("data_raw", "ddti_unique_dataset")
os.makedirs(os.path.join(out_dir, "benign"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "malignant"), exist_ok=True)

xml_files = glob.glob(os.path.join(ddti_path, "**", "*.xml"), recursive=True)

copied_count = 0
skipped_count = 0

print("Processing DDTI images for deduplication...")
for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tirads_elem = root.find("tirads")
        if tirads_elem is None or tirads_elem.text is None:
            continue
        tirads = tirads_elem.text.strip().lower()
        
        # Strict logic for Training Dataset
        if tirads in ['2', '3']:
            label_name = "benign"
        elif tirads in ['4c', '5']:
            label_name = "malignant"
        else:
            continue
            
        case_dir = os.path.dirname(xml_file)
        case_name = os.path.splitext(os.path.basename(xml_file))[0]
        img_files = glob.glob(os.path.join(case_dir, f"{case_name}_*.jpg"))
        
        for img_file in img_files:
            try:
                img_hash = imagehash.phash(Image.open(img_file).convert("RGB"))
                is_duplicate = False
                for au_hash in auitd_hashes.values():
                    if img_hash - au_hash <= 5:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    skipped_count += 1
                else:
                    dest = os.path.join(out_dir, label_name, os.path.basename(img_file))
                    shutil.copy2(img_file, dest)
                    copied_count += 1
            except Exception:
                pass
    except Exception:
        pass

print(f"Finished! Copied {copied_count} unique images to {out_dir}. Skipped {skipped_count} duplicates/invalid.")
