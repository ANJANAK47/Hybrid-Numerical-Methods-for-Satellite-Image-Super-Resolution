import os
import shutil
import random

input_folder = "output_EuroSAT_RGB"
train_folder = "train_EuroSAT"
test_folder = "test_EuroSAT"

# Ensure train and test directories exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Iterate through each category (Forest, Industrial, etc.)
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    if not os.path.isdir(category_path):
        continue

    # Find all BiCubic images inside subfolders
    bicubic_images = []
    for root, _, files in os.walk(category_path):
        for file in files:
            if file == "BiCubic.png":
                bicubic_images.append(os.path.join(root, file))

    if not bicubic_images:
        print(f"⚠️ No BiCubic images found in {category}!")
        continue

    # Shuffle and split images into 80% train and 20% test
    random.shuffle(bicubic_images)
    split_idx = int(0.8 * len(bicubic_images))
    train_images = bicubic_images[:split_idx]
    test_images = bicubic_images[split_idx:]

    # Copy images to train and test folders with unique names
    for idx, img_path in enumerate(train_images):
        dest_dir = os.path.join(train_folder, category)
        os.makedirs(dest_dir, exist_ok=True)
        new_filename = f"{category}_{idx}.png"  # Unique name
        shutil.copy(img_path, os.path.join(dest_dir, new_filename))

    for idx, img_path in enumerate(test_images):
        dest_dir = os.path.join(test_folder, category)
        os.makedirs(dest_dir, exist_ok=True)
        new_filename = f"{category}_{idx}.png"  # Unique name
        shutil.copy(img_path, os.path.join(dest_dir, new_filename))

    print(f"✅ Processed {category}: {len(train_images)} train, {len(test_images)} test")

print("\n🎯 Data Splitting Completed Successfully!")
