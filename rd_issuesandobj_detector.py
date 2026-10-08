import os
import random
import shutil
import cv2
from ultralytics import YOLO
import torch
from pathlib import Path
import torch


def split_dataset(image_dir='datasets/city_issues/images', label_dir='datasets/city_issues/labels',
                  output_dir='datasets/city_issues', train_ratio=0.8):
    """Split images and corresponding label files into train/val sets."""

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # Create output folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Collect image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(image_dir)
                   if any(f.lower().endswith(ext) for ext in image_extensions)]

    if not image_files:
        print(f"No image files found in '{image_dir}'!")
        return

    # Shuffle and split
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Total images: {len(image_files)}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # Copy images + matching labels
    for files, split in [(train_files, train_dir), (val_files, val_dir)]:
        for img_file in files:
            src_img_path = os.path.join(image_dir, img_file)
            dst_img_path = os.path.join(split, 'images', img_file)
            shutil.copy(src_img_path, dst_img_path)

            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            src_label_path = os.path.join(label_dir, label_file)
            dst_label_path = os.path.join(split, 'labels', label_file)

            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)

    print("Dataset split complete!")


def detect_onroad(folder):
    """Detect vehicles in all images from a folder and save annotated outputs."""

    print(f"\nVehicles in: {folder}")
    print("=" * 60)

    if not os.path.exists(folder):
        print(f"❌ Error: '{folder}' folder not found!")
        return

    output_folder = os.path.join(folder, 'detections')
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(folder)
                   if any(f.lower().endswith(ext) for ext in image_extensions)]

    if not image_files:
        print(f"No image files found in '{folder}'!")
        return

    print(f"Found {len(image_files)} image(s)")

    print("\nLoading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("YOLO model loaded!")

    onroad_types = ['car', 'truck', 'bus', 'motorcycle']
    total_onroad_found = 0

    for i, file in enumerate(image_files, 1):
        image_path = os.path.join(folder, file)
        print(f"\n{i}/{len(image_files)}: {file}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {file}")
            continue

        results = model(image)
        result = results[0]

        onroad_found = 0
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name.lower() in onroad_types:
                onroad_found += 1
                total_onroad_found += 1
                detections.append((box, class_name, confidence))
                print(f"  ✔ {class_name}: {confidence:.2f}")

        if not detections:
            print("  ✖ No vehicles detected")
            continue

        # Draw bounding boxes
        for box, class_name, confidence in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = {
                'car': (0, 255, 0),         # Green
                'truck': (0, 0, 255),       # Red
                'motorcycle': (203, 192, 255),  # Pinkish
                'bus': (255, 0, 0)          # Blue
            }.get(class_name, (255, 255, 0))  # Yellow default

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        output_path = os.path.join(output_folder, f"detected_{file}")
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

    print(f"\nDetection finished — total vehicles detected: {total_onroad_found}")
    print(f"Annotated images saved to: {output_folder}")
    print("\nLegend:")
    print("green - car | red - truck | blue - bus | pink - motorcycle")


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
                    print("🚨 Vehicle parked in bike lane - Illegal Parking Detected")
                    bad_behavior_detected = True
                if is_inside_zone(xyxy, crosswalk_polygon):
                    print("🚨 Vehicle parked on crosswalk - Illegal Parking Detected")
                    bad_behavior_detected = True
            elif cls_name == 'litter':
                print("🗑️ Littering Issue Detected")
                bad_behavior_detected = True
            elif cls_name == 'graffiti':
                print("🎨 Graffiti Detected")
                bad_behavior_detected = True
    if not bad_behavior_detected:
        print("✅ No issues detected - Good street behavior")

# Define example zones (replace with real image-relative coordinates)
bike_lane_polygon = [(100, 500), (600, 500), (600, 550), (100, 550)]
crosswalk_polygon = [(200, 600), (550, 600), (550, 650), (200, 650)]

analyze_behavior(results, bike_lane_polygon, crosswalk_polygon)


if __name__ == "__main__":
    split_dataset()

    # Detect vehicles in both train and val sets
    detect_onroad('datasets/city_issues/train/images')
    detect_onroad('datasets/city_issues/val/images')

    # Optional export for Jetson (only if GPU available)
    if torch.cuda.is_available():
        print("\nGPU available! Exporting YOLOv8 model to TensorRT...")
        model = YOLO('yolov8n.pt')
        model.export(format='engine')
    else:
        print("\nSkipping TensorRT export — no GPU found (CPU-only environment)")