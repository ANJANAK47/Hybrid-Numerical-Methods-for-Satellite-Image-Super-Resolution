import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.interpolate import RectBivariateSpline

# Define OpenCV interpolation methods
INTERPOLATION_METHODS = {
    "BiCubic": cv2.INTER_CUBIC,
    "BiLinear": cv2.INTER_LINEAR,
    "NearestNeighbor": cv2.INTER_NEAREST
}

# Paths to dataset
input_folder = "highScaledDataSet"
psnr_results = {method: [] for method in INTERPOLATION_METHODS.keys()}
ssim_results = {method: [] for method in INTERPOLATION_METHODS.keys()}
psnr_results["Spline"] = []  # Adding Spline results
ssim_results["Spline"] = []

# Function to apply Spline interpolation
def spline_interpolation(image, target_shape):
    h, w = image.shape[:2]
    target_h, target_w = target_shape

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)

    new_x = np.linspace(0, w - 1, target_w)
    new_y = np.linspace(0, h - 1, target_h)

    interpolated_channels = []
    for i in range(image.shape[2]):  # Loop through color channels
        spline = RectBivariateSpline(y, x, image[:, :, i])
        interpolated_channel = spline(new_y, new_x)
        interpolated_channels.append(interpolated_channel)

    return np.stack(interpolated_channels, axis=2).astype(np.uint8)

# Process each satellite image
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    if not os.path.isdir(folder_path):
        continue

    downscaled_path = os.path.join(folder_path, "downScaled.png")
    original_path = os.path.join(folder_path, "output.png")

    if not os.path.exists(downscaled_path) or not os.path.exists(original_path):
        continue

    downscaled_img = cv2.imread(downscaled_path)
    original_img = cv2.imread(original_path)

    original_h, original_w = original_img.shape[:2]

    print(f"\n📌 Processing {folder_name}...")

    for method_name, method in INTERPOLATION_METHODS.items():
        # Upscale using OpenCV interpolation
        upscaled_img = cv2.resize(downscaled_img, (original_w, original_h), interpolation=method)

        # Compute PSNR
        psnr_value = psnr(original_img, upscaled_img, data_range=255)
        psnr_results[method_name].append(psnr_value)

        # Compute SSIM (Convert to grayscale as SSIM works better in single channel)
        ssim_value = ssim(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY),
                          data_range=255)
        ssim_results[method_name].append(ssim_value)

        print(f"🔹 {method_name}: PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}")

    # Apply Spline interpolation
    upscaled_spline = spline_interpolation(downscaled_img, (original_h, original_w))
    psnr_spline = psnr(original_img, upscaled_spline, data_range=255)
    ssim_spline = ssim(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(upscaled_spline, cv2.COLOR_BGR2GRAY),
                       data_range=255)

    psnr_results["Spline"].append(psnr_spline)
    ssim_results["Spline"].append(ssim_spline)

    print(f"🔹 Spline: PSNR = {psnr_spline:.2f}, SSIM = {ssim_spline:.4f}")

# Compute and print average PSNR & SSIM for each method
print("\n📊 **Average PSNR & SSIM for Each Method:**")
best_psnr_method = ""
best_ssim_method = ""
best_psnr_value = -1
best_ssim_value = -1

for method_name in psnr_results.keys():
    avg_psnr = np.mean(psnr_results[method_name]) if psnr_results[method_name] else 0
    avg_ssim = np.mean(ssim_results[method_name]) if ssim_results[method_name] else 0

    print(f"🔸 {method_name}: PSNR = {avg_psnr:.2f}, SSIM = {avg_ssim:.4f}")

    if avg_psnr > best_psnr_value:
        best_psnr_value = avg_psnr
        best_psnr_method = method_name

    if avg_ssim > best_ssim_value:
        best_ssim_value = avg_ssim
        best_ssim_method = method_name

print("\n🏆 **Best Interpolation Methods Based on Metrics:**")
print(f"✅ Best PSNR: {best_psnr_method} with PSNR = {best_psnr_value:.2f}")
print(f"✅ Best SSIM: {best_ssim_method} with SSIM = {best_ssim_value:.4f}")






import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.interpolate import RectBivariateSpline

# Define OpenCV interpolation methods
INTERPOLATION_METHODS = {
    "BiCubic": cv2.INTER_CUBIC,
    "BiLinear": cv2.INTER_LINEAR,
    "NearestNeighbor": cv2.INTER_NEAREST
}

# Paths to dataset
input_folder = "EuroSAT_RGB"
output_folder = "output_EuroSAT_RGB"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to apply Spline interpolation
def spline_interpolation(image, target_shape):
    h, w = image.shape[:2]
    target_h, target_w = target_shape

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)

    new_x = np.linspace(0, w - 1, target_w)
    new_y = np.linspace(0, h - 1, target_h)

    interpolated_channels = []
    for i in range(image.shape[2]):  # Loop through color channels
        spline = RectBivariateSpline(y, x, image[:, :, i])
        interpolated_channel = spline(new_y, new_x)
        interpolated_channels.append(interpolated_channel)

    return np.stack(interpolated_channels, axis=2).astype(np.uint8)

# Process each satellite image
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    if not os.path.isdir(category_path):
        continue

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        # Read original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            continue

        original_h, original_w = original_img.shape[:2]
        output_img_dir = os.path.join(output_folder, category, os.path.splitext(image_name)[0])
        os.makedirs(output_img_dir, exist_ok=True)

        print(f"\n📌 Processing {category}/{image_name}...")

        for method_name, method in INTERPOLATION_METHODS.items():
            upscaled_img = cv2.resize(original_img, (original_w * 2, original_h * 2), interpolation=method)
            output_path = os.path.join(output_img_dir, f"{method_name}.png")
            cv2.imwrite(output_path, upscaled_img)

        # Apply Spline interpolation
        upscaled_spline = spline_interpolation(original_img, (original_h * 2, original_w * 2))
        spline_output_path = os.path.join(output_img_dir, "Spline.png")
        cv2.imwrite(spline_output_path, upscaled_spline)

