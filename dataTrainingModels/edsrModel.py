import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add, Input
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Define directories
INPUT_FOLDER = "highScaledDataSet"
OUTPUT_FOLDER = "edsr_results"

# Create output directory if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to load only BiCubic interpolated images
def load_bicubic_images(folder):
    images, filenames = [], []
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            bicubic_path = os.path.join(subdir_path, "downScaledBiCubic.png")
            gt_path = os.path.join(subdir_path, "output.png")

            if os.path.exists(bicubic_path) and os.path.exists(gt_path):
                bicubic_img = cv2.cvtColor(cv2.imread(bicubic_path), cv2.COLOR_BGR2RGB) / 255.0
                gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.0

                images.append((bicubic_img, gt_img))
                filenames.append(subdir)
    return images, filenames

# Load dataset
dataset, filenames = load_bicubic_images(INPUT_FOLDER)
print(f"Loaded {len(dataset)} images.")

# Prepare data
X_train = np.array([cv2.resize(img[0], (64, 64)) for img in dataset])
Y_train = np.array([cv2.resize(img[1], (64, 64)) for img in dataset])
X_train = np.expand_dims(X_train, axis=0)
Y_train = np.expand_dims(Y_train, axis=0)

# EDSR Model
def residual_block(x):
    res = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    res = Conv2D(64, (3, 3), padding='same')(res)
    return Add()([x, res])  # Residual connection

def build_edsr():
    inputs = Input(shape=(64, 64, 3))
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

    # 16 Residual Blocks
    for _ in range(16):
        x = residual_block(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Add()([inputs, x])  # Global residual connection
    outputs = Conv2D(3, (3, 3), padding='same')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train EDSR
edsr = build_edsr()
edsr.fit(X_train, Y_train, epochs=50, batch_size=1)

# Save Model
edsr.save("edsr_model.h5")

# Test EDSR on dataset
for i, (bicubic_img, gt_img) in enumerate(dataset):
    input_img = cv2.resize(bicubic_img, (64, 64)).reshape(1, 64, 64, 3)
    predicted = edsr.predict(input_img)[0]

    # Save output
    subfolder = os.path.join(OUTPUT_FOLDER, filenames[i])
    os.makedirs(subfolder, exist_ok=True)

    output_path = os.path.join(subfolder, "super_resolved.png")
    plt.imsave(output_path, np.clip(predicted, 0, 1))

    # Compute PSNR and SSIM
    psnr_value = psnr(gt_img, predicted, data_range=1.0)
    ssim_value = ssim(gt_img, predicted, channel_axis=2, data_range=1.0)

    print(f"Image {filenames[i]} → PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")
