from pathlib import Path

label_dir = Path("datasets/city_issues/labels/train")
label_files = list(label_dir.glob("*.txt"))

empty_labels = []
for label_file in label_files:
    content = label_file.read_text().strip()
    if not content:
        empty_labels.append(label_file.name)

print(f"Found {len(label_files)} label files")
print(f"{len(empty_labels)} of them are empty.")