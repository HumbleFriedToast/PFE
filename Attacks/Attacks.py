import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from DWT import load_images, preprocess_images, extract_watermark, calculate_metrics

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
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR)

# === Config ===
cover_path = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\Testfield\cover.jpg"
watermark_path = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\Testfield\watermark.png"
watermarked_image_path = r"C:\Users\dalil_vgwbs8i\Desktop\PFE\GUI\Testfield\watermarked.png"

wavelet = 'sym4'
level = 1
key = 1234
threshold_factor = 1.5

# === Load Images ===
cover, watermark = load_images(cover_path, watermark_path)
watermarked = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
_, watermark_bin = preprocess_images(cover, watermark)

if watermarked is None:
    raise FileNotFoundError("Watermarked image could not be loaded.")

# === Define Attacks ===
attacks = {
    "original": watermarked,
    "gaussian_blur": cv2.GaussianBlur(watermarked, (5, 5), 1),
    "salt_and_pepper": salt_and_pepper(watermarked),
    "jpeg_compression": jpeg_compression(watermarked),
    "resize": resize_attack(watermarked),
    "rotate": rotate_attack(watermarked)
}

# === Evaluate Attacks ===
print(f"{'Attack':<20} {'PSNR':>6} {'SSIM':>6} {'BER':>6}")
print("-" * 40)

for name, attacked in attacks.items():
    # Extract watermark
    extracted = extract_watermark(
        watermarked_image=attacked,
        watermark_size=watermark_bin.shape,
        wavelet=wavelet,
        level=level,
        key=key,
        threshold_factor=threshold_factor
    )

    # Calculate metrics
    psnr_val = psnr(cover, attacked)
    ssim_val = ssim(cover, attacked)
    ber_val = np.sum(watermark_bin != extracted) / watermark_bin.size

    print(f"{name:<20} {psnr_val:6.2f} {ssim_val:6.3f} {ber_val:6.3f}")