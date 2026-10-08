from ultralytics import YOLO
import os
import shutil
import random
from shapely.geometry import Polygon, box
from pathlib import Path
import torch

def fix_label_filenames(image_dir, label_dir):
    image_paths = list(Path(image_dir).glob("*.[jp][pn]g"))  # Matches .jpg or .png
    renamed = 0

    for img_path in image_paths:
        true_stem = img_path.stem
        matches = list(Path(label_dir).glob(f"*-{true_stem}.txt"))

        if len(matches) == 1:
            label_file = matches[0]
            new_label_path = Path(label_dir) / (true_stem + ".txt")
            if label_file != new_label_path:
                try:
                    os.rename(label_file, new_label_path)
                    print(f"🔧 Renamed {label_file.name} → {new_label_path.name}")
                    renamed += 1
                except FileExistsError:
                    print(f"Skipped: {new_label_path.name} already exists")
        elif len(matches) > 1:
            print(f"Multiple matches found for {true_stem}, skipping.")
        else:
            print(f"No label found for image: {true_stem}")

    print(f"\nRenamed {renamed} label files.")


base_path = Path("datasets/city_issues")
images_dir = base_path / "images"
labels_dir = base_path / "labels"

# Fix label filenames first to match image files
fix_label_filenames(images_dir, labels_dir)

# Get all image files
image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
random.shuffle(image_files)

# Train/val split
train_ratio = 0.8
split_idx = int(len(image_files) * train_ratio)
train_imgs = image_files[:split_idx]
val_imgs = image_files[split_idx:]

# Create target folders
for split in ['train', 'val']:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

def move_files(img_list, split):
    for img_path in img_list:
        label_path = labels_dir / img_path.with_suffix(".txt").name

        # Move image
        shutil.move(str(img_path), str(images_dir / split / img_path.name))

        # Move label (if it exists)
        if label_path.exists():
            shutil.move(str(label_path), str(labels_dir / split / label_path.name))
        else:
            print(f"No label for image: {img_path.name}")

# Perform move
move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print("Done! Images and labels split into train/val.")

# STEP 1: Define custom class names and dataset path
CLASS_NAMES = [
    'pedestrian', 'crosswalk', 'litter', 'graffiti', 'bike_lane_marking',
    'bicycle', 'wheelchair', 'scooter', 'vehicle', 'construction_sign',
    'traffic_signal', 'cone'
]

DATA_YAML = r"""
train: C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\images\train
val: C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\images\val

nc: 12
names: [pedestrian, crosswalk, litter, graffiti, bike_lane_marking, bicycle, wheelchair, scooter, vehicle, construction_sign, traffic_signal, cone]
"""

with open(r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml", "w") as f:
    f.write(DATA_YAML)

# STEP 2: Load YOLOv8 model and train
model = YOLO('yolov8n.pt')  # use yolov8s.pt for more accuracy, yolov8n.pt for speed
model.train(
    data=r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name='city_issues_detector'
)

# STEP 3: Run inference on a test image
val_img = list((images_dir / 'val').glob('*.jpg'))[0]
print(f"Using test image: {val_img}")
results = model(str(val_img), conf=0.3)

# STEP 4: Detect higher-level issues from raw predictions

def is_inside_zone(bbox, zone_poly):
    return Polygon(zone_poly).intersects(box(*bbox))

def analyze_behavior(results, bike_lane_polygon, crosswalk_polygon):
    bad_behavior_detected = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()

            if cls_name == 'vehicle':
                if is_inside_zone(xyxy, bike_lane_polygon):
                    print("Vehicle parked in bike lane")
                    bad_behavior_detected = True
                if is_inside_zone(xyxy, crosswalk_polygon):
                    print("Vehicle parked on crosswalk")
                    bad_behavior_detected = True
            elif cls_name == 'litter':
                print("Littering Issue Detected")
                bad_behavior_detected = True
            elif cls_name == 'graffiti':
                print("Graffiti Detected")
                bad_behavior_detected = True
    if not bad_behavior_detected:
        print("No issues detected")

# Define example zones (replace with real image-relative coordinates)
bike_lane_polygon = [(100, 500), (600, 500), (600, 550), (100, 550)]
crosswalk_polygon = [(200, 600), (550, 600), (550, 650), (200, 650)]

analyze_behavior(results, bike_lane_polygon, crosswalk_polygon)

# STEP 5: Export for Jetson deployment
if torch.cuda.is_available():
    print("GPU available: exporting to TensorRT for Jetson...")
    model.export(format='engine')
else:
    print("Skipping TensorRT export — no GPU available (CPU-only machine)")

