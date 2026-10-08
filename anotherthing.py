from pathlib import Path

label_dir = Path("datasets/city_issues/labels/train")
image_dir = Path("datasets/city_issues/images/train")

# Check label count and file sizes
labels = list(label_dir.glob("*.txt"))
print(f"Found {len(labels)} label files")

empty = [f for f in labels if f.read_text().strip() == ""]
print(f"Empty label files: {len(empty)}")

# Check if label filenames match image filenames
img_names = set(f.stem for f in image_dir.glob("*.png"))
label_names = set(f.stem for f in labels)

missing = img_names - label_names
print(f"Images with no matching label: {len(missing)}")
