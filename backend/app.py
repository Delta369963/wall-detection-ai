from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import cv2
import numpy as np
from pathlib import Path

# -----------------------------------
# App
# -----------------------------------

app = FastAPI()

# -----------------------------------
# CORS
# -----------------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# -----------------------------------
# Load model
# -----------------------------------

model = YOLO("models/best.pt")

# -----------------------------------
# Folders
# -----------------------------------

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("processed")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------
# Wall Detection Endpoint
# -----------------------------------

@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    # -----------------------------------
    # Save uploaded image
    # -----------------------------------

    image_path = UPLOAD_DIR / file.filename

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # -----------------------------------
    # Read image
    # -----------------------------------

    image = cv2.imread(str(image_path))

    overlay = image.copy()

    # -----------------------------------
    # Run inference
    # -----------------------------------

    results = model.predict(
        str(image_path),
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

            mask = (mask > 0.5).astype(np.uint8)

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0])
            )

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

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < 5000:
                    continue

                epsilon = (
                    0.005 *
                    cv2.arcLength(contour, True)
                )

                approx = cv2.approxPolyDP(
                    contour,
                    epsilon,
                    True
                )

                # Draw ONLY borders
                cv2.polylines(
                    overlay,
                    [approx],
                    True,
                    (255, 0, 0),
                    4
                )

    # -----------------------------------
    # Save output
    # -----------------------------------

    output_path = OUTPUT_DIR / file.filename

    cv2.imwrite(
        str(output_path),
        overlay
    )

    return FileResponse(
        path=str(output_path),
        media_type="image/jpeg",
        filename="prediction.jpg"
    )