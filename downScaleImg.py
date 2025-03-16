import os
import cv2

# Define input and output directories
INPUT_DIR = "dataset"
OUTPUT_DIR = "highScaledDataSet"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Downscale factor (25% of original size)
DOWNSCALE_FACTOR = 4

# Process each image in the input dataset
for filename in os.listdir(INPUT_DIR):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(INPUT_DIR, filename)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {filename}, unable to read image.")
            continue

        # Create sub-folder with image name (without extension)
        image_name = os.path.splitext(filename)[0]  # Get image name without extension
        image_folder = os.path.join(OUTPUT_DIR, image_name)
        os.makedirs(image_folder, exist_ok=True)

        # Get new dimensions (25% of original)
        new_width = img.shape[1] // DOWNSCALE_FACTOR
        new_height = img.shape[0] // DOWNSCALE_FACTOR

        # Resize image
        downscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Save images in the new sub-folder
        input_path = os.path.join(image_folder, "downScaled.png")
        output_path = os.path.join(image_folder, "output.png")

        cv2.imwrite(input_path, downscaled_img)  # Save downscaled image as input.png
        cv2.imwrite(output_path, img)           # Save original image as output.png

        print(f"Processed {filename} → {image_folder}")

print("✅ All images processed successfully!")

