import os
import shutil
import random

def split_and_copy(source_dir, class_name, base_output, split_ratio=0.8):
    # Create output folders
    train_class_dir = os.path.join(base_output, "train", class_name)
    test_class_dir = os.path.join(base_output, "test", class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Get image files
    all_files = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
    ]
    random.shuffle(all_files)

    # Split into train/test
    split_index = int(len(all_files) * split_ratio)
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    # Copy to train
    for file in train_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(train_class_dir, file)
        shutil.copy(src, dst)

    # Copy to test
    for file in test_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(test_class_dir, file)
        shutil.copy(src, dst)

    print(f"✅ {class_name}: {len(train_files)} train, {len(test_files)} test")

# Base path
base = "resized/SIZE_120_rescaled_max_area_1024"

# Split for each class
split_and_copy(os.path.join(base, "120_blunt", "data"), "120_blunt", base)
split_and_copy(os.path.join(base, "120_gun", "data"), "120_gun", base)
