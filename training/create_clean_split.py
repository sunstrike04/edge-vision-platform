from pathlib import Path
from collections import defaultdict
import random
import shutil

SOURCE = Path("data/warehouse_raw")
OUTPUT = Path("data/warehouse")

random.seed(42)

# Group every image/label pair by its original Roboflow image ID.
# Example:
# 1014_jpg.rf.abc.jpg
# 1014_jpg.rf.xyz.jpg
# both belong to "1014_jpg".
groups = defaultdict(list)

for split in ["train", "valid", "test"]:
    images_dir = SOURCE / split / "images"
    labels_dir = SOURCE / split / "labels"

    for image_path in images_dir.iterdir():
        original_id = image_path.name.split(".rf.")[0]

        label_path = labels_dir / (
            image_path.stem + ".txt"
        )

        groups[original_id].append((image_path, label_path))


# Shuffle ORIGINAL images, not individual augmented images.
original_ids = list(groups.keys())
random.shuffle(original_ids)

n = len(original_ids)

# Make a clean 80 / 10 / 10 split.
train_end = int(n * 0.80)
valid_end = int(n * 0.90)

split_ids = {
    "train": original_ids[:train_end],
    "valid": original_ids[train_end:valid_end],
    "test": original_ids[valid_end:]
}


# Create the new dataset.
for split in ["train", "valid", "test"]:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

    for original_id in split_ids[split]:
        for image_path, label_path in groups[original_id]:

            shutil.copy2(
                image_path,
                OUTPUT / split / "images" / image_path.name
            )

            shutil.copy2(
                label_path,
                OUTPUT / split / "labels" / label_path.name
            )


# Verify that original IDs cannot occur in multiple splits.
sets = {
    split: set(ids)
    for split, ids in split_ids.items()
}

print(f"Unique original images: {n}")

for split in ["train", "valid", "test"]:
    num_images = sum(len(groups[x]) for x in split_ids[split])

    print(
        f"{split}: "
        f"{len(split_ids[split])} originals, "
        f"{num_images} total images"
    )

print("\nCross-split overlap:")
print("train / valid:", len(sets["train"] & sets["valid"]))
print("train / test: ", len(sets["train"] & sets["test"]))
print("valid / test: ", len(sets["valid"] & sets["test"]))