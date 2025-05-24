import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from typing import Tuple, Optional
# Parameters
wavelet_type = 'haar'     # Wavelet type
level = 1                 # Decomposition level
embedding_strength = 2.0  # Embedding strength
secret_key = 1000         # Secret key (seed)
threshold_factor = 1.5    # Threshold determination factor
max_watermark_size = 5000 # Maximum watermark size (pixels)

def load_images(cover_path: str, watermark_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cover and watermark images with error handling
    """
    try:
        # Check if files exist
        if not os.path.exists(cover_path):
            print(f"Error: Cover image not found at {cover_path}")
            return None, None

        if not os.path.exists(watermark_path):
            print(f"Error: Watermark image not found at {watermark_path}")
            return None, None

        # Load images
        cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        if cover_image is None:
            raise ValueError("Failed to load cover image")

        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        if watermark is None:
            raise ValueError("Failed to load watermark image")

        return cover_image, watermark

    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return None, None

def preprocess_images(cover_image: np.ndarray, watermark: np.ndarray, max_watermark_size=None) -> Tuple[np.ndarray, np.ndarray]:

    # Convert cover image to float for processing
    cover_image_float = cover_image.astype(np.float64)
    # Resize watermark if needed (optimization)
    if max_watermark_size is not None:
        h, w = watermark.shape
        if h * w > max_watermark_size:
            # Calculate new dimensions while maintaining aspect ratio
            ratio = np.sqrt(max_watermark_size / (h * w))
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            watermark = cv2.resize(watermark, (new_w, new_h))
            print(f"Resized watermark from {h}x{w} to {new_h}x{new_w}")

    # Convert watermark to binary
    binary_watermark = (watermark > 128).astype(np.uint8)

    return cover_image_float, binary_watermark

def embed_watermark(cover_image: np.ndarray, watermark: np.ndarray,
                    wavelet: str = 'haar', level: int = 1,
                    strength: float = 2.0, key: int = 1000) -> np.ndarray:
    """
    Embed watermark into cover image using DWT
    """
    # Set random seed if provided
    if key is not None:
        np.random.seed(key)

    # Get dimensions
    Mc, Nc = cover_image.shape
    Mwm, Nwm = watermark.shape

    # Calculate size reduction factor
    L = 2**level

    # Reshape watermark to vector
    wm_vector = watermark.reshape(-1)

    # Perform DWT on cover image
    coeffs = pywt.wavedec2(cover_image, wavelet, level=level)
    cA, detail_coeffs = coeffs[0], list(coeffs[1:])

    # Extract detail coefficients at target level
    cH, cV, cD = detail_coeffs[0]

    # Embed each bit of the watermark
    for i in range(len(wm_vector)):
        # Generate pseudorandom noise sequence
        pn_sequence = np.round(2 * (np.random.rand(Mc//L, Nc//L) - 0.5))
        # Embed watermark bit by adding/subtracting PRN sequence
        if wm_vector[i] == 0:
            cD = cD + strength * pn_sequence

    # Update detail coefficients
    detail_coeffs[0] = (cH, cV, cD)

    # Reconstruct watermarked image
    new_coeffs = [cA] + detail_coeffs
    watermarked_image = pywt.waverec2(new_coeffs, wavelet)

    # Clip values to valid range and convert to uint8
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    return watermarked_image

def extract_watermark(watermarked_image: np.ndarray, watermark_size: Tuple[int, int],
                     wavelet: str = 'haar', level: int = 1, key: int = 1000,
                     threshold_factor: float = 1.5) -> np.ndarray:
    """
    Extract watermark from watermarked image
    """
    # Set random seed if provided
    if key is not None:
        np.random.seed(key)

    # Get dimensions
    Mw, Nw = watermarked_image.shape
    Mwm, Nwm = watermark_size
    wm_length = Mwm * Nwm

    # Calculate size reduction factor
    L = 2**level

    # Perform DWT on watermarked image
    coeffs = pywt.wavedec2(watermarked_image, wavelet, level=level)
    _, detail_coeffs = coeffs[0], list(coeffs[1:])

    # Extract detail coefficients at target level
    _, _, cD = detail_coeffs[0]
    cD_flat = cD.flatten()

    # Calculate correlations for each bit
    correlations = np.zeros(wm_length)

    for i in range(wm_length):
        # Generate same PRN sequence as during embedding
        pn_sequence = np.round(2 * (np.random.rand(Mw//L, Nw//L) - 0.5))
        pn_flat = pn_sequence.flatten()

        # Calculate correlation
        correlations[i] = np.corrcoef(cD_flat, pn_flat)[0, 1]

    # Determine threshold for watermark extraction
    threshold = np.mean(correlations) * threshold_factor

    # Extract watermark bits based on correlation
    recovered_vector = np.ones(wm_length)
    recovered_vector[correlations > threshold] = 0

    # Reshape vector to original watermark size
    recovered_watermark = recovered_vector.reshape(Mwm, Nwm)

    return recovered_watermark

def calculate_metrics(original_image: np.ndarray, watermarked_image: np.ndarray,
                      original_watermark: np.ndarray, recovered_watermark: np.ndarray) -> dict:
    """
    Calculate quality metrics
    """
    # Calculate PSNR between original and watermarked images
    psnr_value = psnr(original_image, watermarked_image)

    # Calculate correlation coefficient between original and recovered watermarks
    cc = np.corrcoef(original_watermark.flatten(), recovered_watermark.flatten())[0, 1]

    # Calculate normalized correlation
    ncc = np.sum(original_watermark * recovered_watermark) / (np.sqrt(np.sum(original_watermark**2)) * np.sqrt(np.sum(recovered_watermark**2)))

    # Calculate bit error rate
    bit_errors = np.sum(original_watermark != recovered_watermark)
    total_bits = original_watermark.size
    ber = bit_errors / total_bits

    return {
        "psnr": psnr_value,
        "correlation_coefficient": cc,
        "normalized_correlation": ncc,
        "bit_error_rate": ber,
        "bit_errors": bit_errors,
        "total_bits": total_bits
    }

def plot_results(original_image: np.ndarray, watermarked_image: np.ndarray,
                original_watermark: np.ndarray, recovered_watermark: np.ndarray,
                save_path=None):
    """
    Plot comparison of images and watermarks
    """
    # Plot original vs watermarked image
    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Cover Image')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(watermarked_image, cmap='gray')
    plt.title('Watermarked Image')
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(original_watermark, cmap='gray')
    plt.title('Original Watermark')
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(recovered_watermark, cmap='gray')
    plt.title('Recovered Watermark')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'watermark_comparison.png'))
    else:
        plt.show()

def DWT_FULL(cover_image,watermark_original):

    # Preprocess images
    cover_image_float, binary_watermark = preprocess_images(
        cover_image,
        watermark_original,
        max_watermark_size=max_watermark_size
    )

    # Embed watermark
    watermarked_image = embed_watermark(
        cover_image_float,
        binary_watermark,
        wavelet=wavelet_type,
        level=level,
        strength=embedding_strength,
        key=secret_key
    )
    return watermarked_image

def EXTRACT_FULL(watermarked_image,binary_watermark):
    # Extract watermark from watermarked image
    recovered_watermark = extract_watermark(
        watermarked_image,
        binary_watermark.shape,
        wavelet=wavelet_type,
        level=level,
        key=secret_key,
        threshold_factor=threshold_factor
    )
def evaluate_Dwt(cover_image,watermarked_image,binary_watermark,recovered_watermark):
    # Calculate quality metrics
    print("\n--- Watermarked Image Metrics ---")
    metrics = calculate_metrics(cover_image, watermarked_image, binary_watermark, recovered_watermark)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Plot results
    print("\nGenerating visualizations...")
    plot_results(cover_image, watermarked_image, binary_watermark, recovered_watermark)

    print("\nDone!")
