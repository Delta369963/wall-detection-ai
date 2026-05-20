from pathlib import Path
import cv2
import random

# -----------------------------------
# Paths
# -----------------------------------

IMAGE_DIR = Path(
    "datasets/indoor_filtered/images"
)

MASK_DIR = Path(
    "datasets/indoor_filtered/wall_masks"
)

OUTPUT_DIR = Path(
    "outputs/overlays"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# -----------------------------------
# Get image list
# -----------------------------------

image_files = list(
    IMAGE_DIR.glob("*.jpg")
)

# Randomly select samples
samples = random.sample(image_files, 10)

# -----------------------------------
# Create overlays
# -----------------------------------

for image_path in samples:

    mask_path = MASK_DIR / (
        image_path.stem + ".png"
    )

    image = cv2.imread(str(image_path))
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        continue

    # Resize mask if needed
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0])
    )

    # Create coloured overlay
    overlay = image.copy()

    overlay[mask == 255] = [0, 255, 0]

    # Blend image + overlay
    blended = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    output_path = OUTPUT_DIR / (
        image_path.stem + "_overlay.jpg"
    )

    cv2.imwrite(
        str(output_path),
        blended
    )

print("\nOverlay visualisations saved.")