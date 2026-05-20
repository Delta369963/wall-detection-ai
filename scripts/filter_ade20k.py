from pathlib import Path
import shutil

# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path("datasets/ade20k_raw/ADEChallengeData2016")

IMAGE_DIR = BASE_DIR / "images" / "training"
ANNOTATION_DIR = BASE_DIR / "annotations" / "training"

SCENE_FILE = BASE_DIR / "sceneCategories.txt"

OUTPUT_IMAGE_DIR = Path("datasets/indoor_filtered/images")
OUTPUT_MASK_DIR = Path("datasets/indoor_filtered/masks")

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Indoor scene whitelist
# -----------------------------------

INDOOR_SCENES = {
    "bedroom",
    "living_room",
    "bathroom",
    "kitchen",
    "office",
    "conference_room",
    "corridor",
    "dining_room",
    "home_office",
    "hotel_room",
    "apartment",
    "closet",
    "library",
    "classroom",
    "childs_room",
    "staircase",
    "television_room",
    "waiting_room",
    "art_gallery",
    "airport_terminal"
}

# -----------------------------------
# Process scenes
# -----------------------------------

count = 0

with open(SCENE_FILE, "r") as file:

    for line in file:

        parts = line.strip().split()

        if len(parts) != 2:
            continue

        image_name, scene_type = parts

        if scene_type not in INDOOR_SCENES:
            continue

        image_file = IMAGE_DIR / f"{image_name}.jpg"
        mask_file = ANNOTATION_DIR / f"{image_name}.png"

        if not image_file.exists():
            continue

        if not mask_file.exists():
            continue

        shutil.copy2(
            image_file,
            OUTPUT_IMAGE_DIR / image_file.name
        )

        shutil.copy2(
            mask_file,
            OUTPUT_MASK_DIR / mask_file.name
        )

        count += 1

        if count % 100 == 0:
            print(f"Copied {count} indoor samples...")

print(f"\nDone.")
print(f"Total indoor samples copied: {count}")