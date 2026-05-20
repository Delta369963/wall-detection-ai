import kagglehub
import shutil
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download(
    "aprilsan/mit-scene-parsing-train-and-val"
)

print(f"\nDownloaded to: {path}")

# Destination folder
destination = Path("datasets/ade20k_raw")

# Copy contents
source = Path(path)

for item in source.iterdir():
    target = destination / item.name

    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item, target)

print("\nDataset copied successfully.")