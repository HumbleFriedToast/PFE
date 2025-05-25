import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse

# === LSB Embedding ===
def lsb_embed(cover, watermark):
    if len(cover.shape) == 3:
        cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    if len(watermark.shape) == 3:
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)

    watermark_bin = (watermark > 128).astype(np.uint8)
    wm_resized = cv2.resize(watermark_bin, (cover.shape[1], cover.shape[0]))
    cleared = cover & 0xFE
    embedded = cleared | wm_resized
    return embedded

# === LSB Extraction ===
def lsb_extract(watermarked, shape):
    if len(watermarked.shape) == 3:
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    bits = watermarked & 1
    resized = cv2.resize(bits.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (resized * 255).astype(np.uint8)

# === BER Calculation ===
def compute_ber(original, extracted):
    original_bin = (original > 128).astype(np.uint8)
    extracted_bin = (extracted > 128).astype(np.uint8)
    return np.sum(original_bin != extracted_bin) / original_bin.size

# === Attacks ===
def salt_and_pepper(img, amount=0.01):
    noisy = img.copy()
    num_salt = int(np.ceil(amount * img.size * 0.5))
    num_pepper = int(np.ceil(amount * img.size * 0.5))
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def jpeg_compression(img, quality=30):
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, 0)

def resize_attack(img):
    small = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return cv2.resize(small, (img.shape[1], img.shape[0]))

def rotate_attack(img, angle=15):
    h, w = img.shape
    center = (w//2, h//2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h))

