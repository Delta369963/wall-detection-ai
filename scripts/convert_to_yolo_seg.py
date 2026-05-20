from pathlib import Path
import cv2
import shutil
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

YOLO_IMAGE_TRAIN = Path(
    "datasets/yolo_seg/images/train"
)

YOLO_IMAGE_VAL = Path(
    "datasets/yolo_seg/images/val"
)

YOLO_LABEL_TRAIN = Path(
    "datasets/yolo_seg/labels/train"
)

YOLO_LABEL_VAL = Path(
    "datasets/yolo_seg/labels/val"
)

# Create directories
for path in [
    YOLO_IMAGE_TRAIN,
    YOLO_IMAGE_VAL,
    YOLO_LABEL_TRAIN,
    YOLO_LABEL_VAL
]:
    path.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Settings
# -----------------------------------

TRAIN_SPLIT = 0.9

# -----------------------------------
# Collect images
# -----------------------------------

image_files = list(
    IMAGE_DIR.glob("*.jpg")
)

random.shuffle(image_files)

split_index = int(
    len(image_files) * TRAIN_SPLIT
)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

# -----------------------------------
# Conversion function
# -----------------------------------

def convert_sample(
    image_path,
    image_output_dir,
    label_output_dir
):

    mask_path = MASK_DIR / (
        image_path.stem + ".png"
    )

    image = cv2.imread(str(image_path))
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        return

    height, width = mask.shape

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    label_lines = []

    for contour in contours:

        # Ignore tiny contours
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        contour = contour.squeeze()

        if len(contour.shape) != 2:
            continue

        polygon_points = []

        for point in contour:

            x, y = point

            x_norm = x / width
            y_norm = y / height

            polygon_points.append(
                f"{x_norm:.6f}"
            )

            polygon_points.append(
                f"{y_norm:.6f}"
            )

        if len(polygon_points) < 6:
            continue

        line = "0 " + " ".join(
            polygon_points
        )

        label_lines.append(line)

    # Save label file
    label_path = (
        label_output_dir /
        f"{image_path.stem}.txt"
    )

    with open(label_path, "w") as file:
        file.write("\n".join(label_lines))

    # Copy image
    shutil.copy2(
        image_path,
        image_output_dir / image_path.name
    )

# -----------------------------------
# Process training set
# -----------------------------------

print("\nProcessing training set...")

for i, image_path in enumerate(train_files):

    convert_sample(
        image_path,
        YOLO_IMAGE_TRAIN,
        YOLO_LABEL_TRAIN
    )

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1} train samples")

# -----------------------------------
# Process validation set
# -----------------------------------

print("\nProcessing validation set...")

for i, image_path in enumerate(val_files):

    convert_sample(
        image_path,
        YOLO_IMAGE_VAL,
        YOLO_LABEL_VAL
    )

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1} val samples")

print("\nDone.")
print(f"Train samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")