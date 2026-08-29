import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("azouzmaroua/algeria-ultrasound-images-thyroid-dataset-auitd")
print("Path to dataset files:", path)

for root, dirs, files in os.walk(path):
    print(f"\nDirectory: {root}")
    print(f"Files: {files[:10]} (total {len(files)} files)")
    if len(dirs) > 0:
        print(f"Dirs: {dirs}")
