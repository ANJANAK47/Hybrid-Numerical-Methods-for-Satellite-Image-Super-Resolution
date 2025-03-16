import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.transform import resize

# Define directories
INPUT_FOLDER = "highScaledDataSet"
OUTPUT_FOLDER = "srcnn_results"

# Create output directory if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to load only BiCubic interpolated images as input
def load_bicubic_images(folder):
    images = []
    filenames = []
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            bicubic_path = os.path.join(subdir_path, "downScaledBiCubic.png")
            gt_path = os.path.join(subdir_path, "output.png")  # Original high-res image

            if os.path.exists(bicubic_path) and os.path.exists(gt_path):
                # Read and convert to RGB
                bicubic_img = cv2.cvtColor(cv2.imread(bicubic_path), cv2.COLOR_BGR2RGB) / 255.0
                gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.0

                images.append((bicubic_img, gt_img))
                filenames.append(subdir)
    return images, filenames

# Load dataset (BiCubic only)
dataset, filenames = load_bicubic_images(INPUT_FOLDER)
print(f"Loaded {len(dataset)} images.")

# Prepare data
X_train = np.array([cv2.resize(img[0], (64, 64)) for img in dataset])  # Resize for training
Y_train = np.array([cv2.resize(img[1], (64, 64)) for img in dataset])  # Resize ground truth
X_train = np.array(X_train)  # Shape: (num_images, 64, 64, 3)
Y_train = np.array(Y_train)  # Shape: (num_images, 64, 64, 3)

# SRCNN Model
def build_srcnn():
    model = Sequential([
        Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(64, 64, 3)),
        Conv2D(32, (5, 5), activation='relu', padding='same'),
        Conv2D(3, (5, 5), activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train SRCNN
srcnn = build_srcnn()
srcnn.fit(X_train, Y_train, epochs=50, batch_size=1)

# Save Model
srcnn.save("srcnn_model.h5")

# Test SRCNN on dataset
for i, (bicubic_img, gt_img) in enumerate(dataset):
    input_img = cv2.resize(bicubic_img, (64, 64)).reshape(1, 64, 64, 3)
    predicted = srcnn.predict(input_img)[0]

    # Save output
    subfolder = os.path.join(OUTPUT_FOLDER, filenames[i])
    os.makedirs(subfolder, exist_ok=True)

    output_path = os.path.join(subfolder, "super_resolved.png")
    plt.imsave(output_path, np.clip(predicted, 0, 1))

    # Compute PSNR and SSIM
    predicted_resized = resize(predicted, gt_img.shape, anti_aliasing=True)

    # Compute PSNR
    psnr_value = psnr(gt_img, predicted_resized, data_range=1.0)
    predicted_resized = resize(predicted, gt_img.shape, anti_aliasing=True)

    # Compute SSIM
    ssim_value = ssim(gt_img, predicted_resized, channel_axis=2, data_range=1.0)
    print(f"Image {filenames[i]} → PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")
