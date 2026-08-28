import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "ddti.csv" # guessing? Let's just try to download it without adapter first to see files.

path = kagglehub.dataset_download("dasmehdixtr/ddti-thyroid-ultrasound-images")
print("Path to dataset files:", path)

import os
print(os.listdir(path))
