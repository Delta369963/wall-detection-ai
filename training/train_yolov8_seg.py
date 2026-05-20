from ultralytics import YOLO

# -----------------------------------
# Load pretrained segmentation model
# -----------------------------------

model = YOLO("yolov8s-seg.pt")

# -----------------------------------
# Train model
# -----------------------------------

model.train(
    data="training/configs/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    workers=2,

    # Optimisation
    optimizer="AdamW",
    lr0=0.001,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    # Save settings
    project="training/logs",
    name="wall_segmentation",

    pretrained=True
)