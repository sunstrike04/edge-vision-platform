from pathlib import Path
from collections import Counter
import yaml

DATASET_ROOT = Path("data/warehouse")

with open(DATASET_ROOT / "data.yaml") as f:
    config = yaml.safe_load(f)

names = config["names"]

for split in ["train", "valid", "test"]:
    labels_dir = DATASET_ROOT / split / "labels"
    label_files = list(labels_dir.glob("*.txt"))

    instance_counts = Counter()
    image_counts = Counter()
    empty_files = 0

    for label_file in label_files:
        lines = [line for line in label_file.read_text().splitlines() if line.strip()]

        if not lines:
            empty_files += 1
            continue

        classes_in_image = set()

        for line in lines:
            class_id = int(line.split()[0])

            instance_counts[class_id] += 1
            classes_in_image.add(class_id)

        for class_id in classes_in_image:
            image_counts[class_id] += 1

    print(f"\n--- {split.upper()} ---")
    print(f"Label files: {len(label_files)}")
    print(f"Empty label files: {empty_files}")

    for class_id, class_name in enumerate(names):
        print(
            f"{class_name:12} "
            f"instances={instance_counts[class_id]:6}  "
            f"images={image_counts[class_id]:6}"
        )