from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# -----------------------------------
# Load trained model
# -----------------------------------

model = YOLO(
    "runs/segment/training/logs/wall_segmentation/weights/best.pt"
)

# -----------------------------------
# Input image
# -----------------------------------

IMAGE_PATH = "test_img2.jpeg"

# -----------------------------------
# Read image
# -----------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Image not found.")

overlay = image.copy()

# -----------------------------------
# Run prediction
# -----------------------------------

results = model.predict(
    IMAGE_PATH,
    conf=0.5,
    retina_masks=True
)

# -----------------------------------
# Process masks
# -----------------------------------

for result in results:

    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()

    for mask in masks:

        # Convert mask to binary
        mask = (mask > 0.5).astype(np.uint8)

        # Resize to original image size
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0])
        )

        # -----------------------------------
        # CLEAN MASK
        # -----------------------------------

        kernel = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # -----------------------------------
        # Find contours
        # -----------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore tiny noisy regions
            if area < 5000:
                continue

            # Smooth polygon
            epsilon = 0.005 * cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

            # Draw ONLY wall borders
            cv2.polylines(
                overlay,
                [approx],
                True,
                (0, 0, 0),
                4
            )

# -----------------------------------
# Blend overlay
# -----------------------------------

final = overlay

# -----------------------------------
# Save output
# -----------------------------------

Path("outputs").mkdir(exist_ok=True)

cv2.imwrite(
    "outputs/final_wall_segmentation.jpg",
    final
)

print("\nDone.")