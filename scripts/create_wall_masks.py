from pathlib import Path
import cv2
import numpy as np

# -----------------------------------
# Paths
# -----------------------------------

INPUT_MASK_DIR = Path("datasets/indoor_filtered/masks")

OUTPUT_MASK_DIR = Path(
    "datasets/indoor_filtered/wall_masks"
)

OUTPUT_MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# -----------------------------------
# ADE20K wall class ID
# -----------------------------------

WALL_CLASS_ID = 1

# -----------------------------------
# Process masks
# -----------------------------------

mask_files = list(INPUT_MASK_DIR.glob("*.png"))

count = 0

for mask_path in mask_files:

    # Read segmentation mask
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        continue

    # Create binary wall mask
    wall_mask = np.where(
        mask == WALL_CLASS_ID,
        255,
        0
    ).astype(np.uint8)

    # Save binary mask
    output_path = OUTPUT_MASK_DIR / mask_path.name

    cv2.imwrite(
        str(output_path),
        wall_mask
    )

    count += 1

    if count % 100 == 0:
        print(f"Processed {count} masks...")

print(f"\nDone.")
print(f"Total wall masks created: {count}")