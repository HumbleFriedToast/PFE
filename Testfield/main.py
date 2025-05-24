import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from DCT import embed_watermark, extract_watermark, preprocess_watermark, compare_watermarks

# === Attack Functions ===
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

# === Config ===
cover_path = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\Testfield\cover.jpg"
watermark_path = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\Testfield\watermark.jpg"
alpha = 10  # strength of DCT embedding

# === Load Images ===
cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
if cover is None or watermark is None:
    raise FileNotFoundError("Check your cover or watermark path.")

# === Embed Watermark ===
watermarked = embed_watermark(cover, watermark, cover.shape, alpha=alpha)
wm_bin = preprocess_watermark(watermark, cover.shape)

# === Define Attacks ===
attacks = {
    "original": watermarked,
    "gaussian_blur": cv2.GaussianBlur(watermarked, (5, 5), 1),
    "salt_and_pepper": salt_and_pepper(watermarked),
    "jpeg_compression": jpeg_compression(watermarked),
    "resize": resize_attack(watermarked),
    "rotate": rotate_attack(watermarked)
}

# === Print Results ===
print(f"{'Attack':<20} {'PSNR':>6} {'SSIM':>6} {'BER':>6}")
print("-" * 40)

for name, attacked_img in attacks.items():
    extracted_bin, _ = extract_watermark(attacked_img, cover.shape, alpha=alpha)
    
    # Compute metrics
    psnr_val = psnr(cover, attacked_img)
    ssim_val = ssim(cover, attacked_img)
    ber_val = 1 - (compare_watermarks(wm_bin, extracted_bin) / 100)

    print(f"{name:<20} {psnr_val:6.2f} {ssim_val:6.3f} {ber_val:6.3f}")
