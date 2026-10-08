from pathlib import Path

label_dir = Path("datasets/city_issues/labels")
invalid_labels = []

for split in ['train', 'val']:
    for file in (label_dir / split).glob("*.txt"):
        with open(file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id > 8:
                    print(f"Invalid class {cls_id} in {file.name}, line {i+1}")
                    invalid_labels.append(file)
            except ValueError:
                print(f"Corrupted line in {file.name}, line {i+1}: {line.strip()}")

print(f"\nScan complete. Found {len(invalid_labels)} problematic files.")
