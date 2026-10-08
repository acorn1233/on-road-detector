from ultralytics import YOLO
import torch
from pathlib import Path
import cv2

base_path = Path("datasets/city_issues")
images_dir = base_path / "images"
labels_dir = base_path / "labels"

CLASS_NAMES = [
    'bicycle', 'bike_lane_marking', 'traffic_signal', 'crosswalk', 'graffiti', 'litter', 
    'pedestrian', 'cone', 'vehicle'
]

train_imgs = images_dir / "train"
val_imgs = images_dir / "val"

model = YOLO('yolov8n.pt') 
model.train(
    data=r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    lr0=0.001,
    name='city_issues_detector'
)


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

output_dir = Path("annotated_images")
output_dir.mkdir(parents=True, exist_ok=True)

def analyze_behavior(results):
    bad_behavior_detected = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            bad_behavior_detected = True
            if cls_name == 'litter':
                print("🗑️ Littering Issue Detected")
                bad_behavior_detected = True
            elif cls_name == 'graffiti':
                print("🎨 Graffiti Detected")
                bad_behavior_detected = True

    if not bad_behavior_detected:
        print("✅ No issues detected - Good street behavior")

# Run inference on training images
for img_path in train_imgs.glob("*.*"):  # Loop through images in the train folder
    results = model(str(img_path), conf=0.3)
    image = cv2.imread(str(img_path))

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            print(f"Detected {cls_name} at {xyxy}")

            # Draw detection box
            color = class_colors.get(cls_name, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save annotated image
    output_path = output_dir / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), image)

    # Analyze behavior
    analyze_behavior(results)

results = model.val(data=r"C:\Users\joahu\Downloads\on_road_detector\datasets\city_issues\data.yaml", imgsz=640)
print(f"Validation Results: {results}")
with open("validation_results.txt", "w") as f:
    f.write(str(results))

# Check if GPU is available and export accordingly
if torch.cuda.is_available():
    print("GPU available: exporting to TensorRT for Jetson...")
    model.export(format='engine')
else:
    print("Skipping TensorRT export — no GPU available (CPU-only machine)")