import os
import cv2
import csv
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from Script.DWT import (
    load_images,
    preprocess_images,
    embed_watermark,
    extract_watermark,
    calculate_metrics
)

# === CONFIG ===
COVER_PATH = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\images\NatureLandscape-1080p-2.jpg"
WATERMARK_PATH = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\watermark.jpg"
CSV_OUTPUT = "attack_metrics.csv"
WAVELET = 'sym4'
LEVEL = 1
STRENGTH = 2.0
KEY = 1234
THRESHOLD_FACTOR = 1.5

# === LOAD IMAGES ===
cover_image, watermark = load_images(COVER_PATH, WATERMARK_PATH)
if cover_image is None or watermark is None:
    raise FileNotFoundError("Check paths to cover or watermark image.")

cover_pre, watermark_bin = preprocess_images(cover_image, watermark)

# === EMBED WATERMARK ===
watermarked_image = embed_watermark(
    cover_image=cover_pre,
    watermark=watermark_bin,
    wavelet=WAVELET,
    level=LEVEL,
    strength=STRENGTH,
    key=KEY
)

# === ATTACK FUNCTIONS ===
def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 1)

def salt_and_pepper(img, amount=0.01):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * img.size * 0.5).astype(int)

    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def jpeg_compression(img, quality=25):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 0)

def resize_attack(img):
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    return cv2.resize(small, (img.shape[1], img.shape[0]))

def rotate_attack(img, angle=15):
    h, w = img.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

# === ATTACKS ===
attacks = {
    "original": watermarked_image,
    "gaussian_blur": gaussian_blur(watermarked_image),
    "salt_and_pepper": salt_and_pepper(watermarked_image),
    "jpeg_compression": jpeg_compression(watermarked_image),
    "resize": resize_attack(watermarked_image),
    "rotate": rotate_attack(watermarked_image),
}

# === WRITE TO CSV ===
header = [
    "attack", "psnr", "ssim", "mse",
    "correlation_coefficient", "normalized_correlation",
    "bit_error_rate", "bit_errors", "total_bits"
]

with open(CSV_OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for name, attacked in attacks.items():
        recovered = extract_watermark(
            watermarked_image=attacked,
            watermark_size=watermark_bin.shape,
            wavelet=WAVELET,
            level=LEVEL,
            key=KEY,
            threshold_factor=THRESHOLD_FACTOR
        )

        metrics = calculate_metrics(
            original_image=cover_image,
            watermarked_image=attacked,
            original_watermark=watermark_bin,
            recovered_watermark=recovered
        )
        metrics["mse"] = float(mse(cover_image, attacked))
        metrics["ssim"] = float(ssim(cover_image, attacked))

        writer.writerow([
            name,
            metrics["psnr"],
            metrics["ssim"],
            metrics["mse"],
            metrics["correlation_coefficient"],
            metrics["normalized_correlation"],
            metrics["bit_error_rate"],
            metrics["bit_errors"],
            metrics["total_bits"]
        ])

print(f"[DONE] Results written to {CSV_OUTPUT}")
