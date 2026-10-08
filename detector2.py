import random
import shutil
from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
from shapely.geometry import Polygon, box

base_path = Path("datasets/city_issues")
images_dir = base_path / "images"
labels_dir = base_path / "labels"

CLASS_NAMES = [
    'pedestrian', 'crosswalk', 'litter', 'graffiti', 'bike_lane_marking',
    'bicycle', 'vehicle', 'traffic_signal', 'cone'
]

def convert_labels(folder, image_folder):
    for txt_file in Path(folder).glob("*.txt"):
        stem = txt_file.stem
        image_path = next(p for p in Path(image_folder).glob(f"{stem}.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'])
        if not image_path or not image_path.exists():
            print(f"Image not found for label: {txt_file.name}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            continue

        h, w = img.shape[:2]

        new_lines = []
        with open(txt_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, box_w, box_h = parts
                cls = int(cls)
                x = float(x)
                y = float(y)
                box_w = float(box_w)
                box_h = float(box_h)

                # If coords >1, assume absolute pixels, normalize
                if max(x, y, box_w, box_h) > 1:
                    x /= w
                    y /= h
                    box_w /= w
                    box_h /= h

                new_lines.append(f"{cls} {x:.6f} {y:.6f} {box_w:.6f} {box_h:.6f}\n")

        with open(txt_file, "w") as file:
            file.writelines(new_lines)

    print("✨ Labels successfully normalized and converted! Slay")


convert_labels(str(labels_dir), str(images_dir))

image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.webp"))
random.shuffle(image_files)

train_ratio = 0.8
split_idx = int(len(image_files) * train_ratio)
train_imgs = image_files[:split_idx]
val_imgs = image_files[split_idx:]

for split in ['train', 'val']:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

def move_files(img_list, split):
    for img_path in img_list:
        label_path = labels_dir / img_path.with_suffix(".txt").name


        shutil.move(str(img_path), str(images_dir / split / img_path.name))

        if label_path.exists():
            shutil.move(str(label_path), str(labels_dir / split / label_path.name))
        else:
            print(f"No label for image: {img_path.name}")

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')
print("Done! Images and labels split into train/val.")

DATA_YAML = r"""
train: C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\images\train
val: C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\images\val

nc: 9
names: [pedestrian, crosswalk, litter, graffiti, bike_lane_marking, bicycle, vehicle, traffic_signal, cone]
"""

with open(r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml", "w") as f:
    f.write(DATA_YAML)

model = YOLO('yolov8n.pt') 
model.train(
    data=r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=0.001,
    name='city_issues_detector'
)

metrics = model.val(
    data=r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml",
    split='val',
    imgsz=640,
    batch=32,
    conf=0.25
)

results_dict = metrics.results_dict
print(f"\nValidation Results:")
print(f"Mean Precision: {results_dict['metrics/precision(B)']:.4f}")
print(f"Mean Recall: {results_dict['metrics/recall(B)']:.4f}")
print(f"mAP@0.5: {results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {results_dict['metrics/mAP50-95(B)']:.4f}")


train_imgs = list((images_dir / 'train').glob('*.jpg')) + list((images_dir / 'train').glob('*.png'))
if not train_imgs:
    raise FileNotFoundError("No training images found!")


for img_path in train_imgs:
    results = model(str(img_path), conf=0.3)
    image = cv2.imread(str(img_path))
    class_colors = {
        'pedestrian': (255, 255, 255),  
        'crosswalk': (153, 255, 204),   
        'litter': (102, 102, 255),      
        'graffiti': (204, 102, 255),    
        'bike_lane_marking': (0, 204, 153), 
        'bicycle': (0, 255, 255), 
        'traffic_signal': (255, 0, 255), 
        'vehicle': (255, 0, 0),
        'cone': (255, 255, 0), 
    }


    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            print(f"Detected {cls_name} at {xyxy}")  
            

            # Draw rectangle
            color = class_colors.get(cls_name, (0,255,0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    output_path = f"runs/detect/train_results/{img_path.stem}_result.jpg"
    cv2.imwrite(output_path, image) 



def is_inside_zone(bbox, zone_poly):
    bbox_poly = box(*bbox)
    zone = Polygon(zone_poly)
    return bbox_poly.intersects(zone)

def analyze_behavior(results, bike_lane_polygon, crosswalk_polygon):
    bad_behavior_detected = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()

            if cls_name == 'vehicle':
                if is_inside_zone(xyxy, bike_lane_polygon):
                    print("Vehicle parked in bike lane - Illegal Parking Detected")
                    bad_behavior_detected = True
                if is_inside_zone(xyxy, crosswalk_polygon):
                    print("Vehicle parked on crosswalk - Illegal Parking Detected")
                    bad_behavior_detected = True
            elif cls_name == 'litter':
                print("Littering Issue Detected")
                bad_behavior_detected = True
            elif cls_name == 'graffiti':
                print("Graffiti Detected")
                bad_behavior_detected = True
    if not bad_behavior_detected:
        print("No issues detected in image")

bike_lane_polygon = [(100, 500), (600, 500), (600, 550), (100, 550)]
crosswalk_polygon = [(200, 600), (550, 600), (550, 650), (200, 650)]

analyze_behavior(results, bike_lane_polygon, crosswalk_polygon)

if torch.cuda.is_available():
    print("GPU available: exporting to TensorRT for Jetson...")
    model.export(format='engine')
else:
    print("Skipping TensorRT export — no GPU available (CPU-only machine)")

