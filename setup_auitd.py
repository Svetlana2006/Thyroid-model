import kagglehub
import os
import shutil

print("Downloading AUITD dataset...")
path = kagglehub.dataset_download("azouzmaroua/algeria-ultrasound-images-thyroid-dataset-auitd")
print("Downloaded to:", path)

target_dir = os.path.join("data_raw", "auitd_dataset")
if not os.path.exists(target_dir):
    print(f"Copying files to {target_dir}...")
    shutil.copytree(path, target_dir)
    print("Done!")
else:
    print(f"{target_dir} already exists.")
