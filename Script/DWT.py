import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_images(cover_path: str, watermark_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    logger = logging.getLogger(__name__) # Get logger instance for this function
    """
    Load cover and watermark images with error handling
    """
    try:
        if not os.path.exists(cover_path):
            logger.error(f"Cover image not found at {cover_path}")
            return None, None
        
        if not os.path.exists(watermark_path):
            logger.error(f"Watermark image not found at {watermark_path}")
            return None, None
        
        cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        if cover_image is None:
            logger.error(f"Failed to load cover image from {cover_path}")
            # raise ValueError("Failed to load cover image") # Or just return None
            return None, None # Corrected to match original behavior
            
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        if watermark is None:
            logger.error(f"Failed to load watermark image from {watermark_path}")
            # raise ValueError("Failed to load watermark image") # Or just return None
            return None, None # Corrected to match original behavior
            
        return cover_image, watermark
        
    except Exception as e:
        logger.error(f"Error loading images: {str(e)}", exc_info=True)
        return None, None

def preprocess_images(cover_image: np.ndarray, watermark: np.ndarray, max_watermark_size=None) -> Tuple[np.ndarray, np.ndarray]:
    logger = logging.getLogger(__name__)
    """
    Preprocess images for watermarking with optional watermark resizing
    """
    cover_image_float = cover_image.astype(np.float64)
    
    if max_watermark_size is not None:
        h, w = watermark.shape
        if h * w > max_watermark_size:
            ratio = np.sqrt(max_watermark_size / (h * w))
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            watermark = cv2.resize(watermark, (new_w, new_h))
            logger.info(f"Resized watermark from {h}x{w} to {new_h}x{new_w}")
    
    binary_watermark = (watermark > 128).astype(np.uint8)
    return cover_image_float, binary_watermark

def embed_watermark(cover_image: np.ndarray, watermark: np.ndarray, 
                    wavelet: str = 'sym4', level: int = 1, 
                    strength: float = 2.0, key: int = 40) -> np.ndarray:
    """
    Embed watermark into cover image using DWT
    """
    if key is not None:
        np.random.seed(key)
    
    wm_vector = watermark.reshape(-1)
    # wm_length = len(wm_vector) # Not needed if generating PN per bit
    
    coeffs = pywt.wavedec2(cover_image, wavelet, level=level)
    cA, detail_coeffs = coeffs[0], list(coeffs[1:])
    cH, cV, cD = detail_coeffs[0]

    # Generate PN sequence per bit inside the loop to save memory
    for i in range(len(wm_vector)):
        # Generate pseudorandom noise sequence with the correct shape of cD
        pn_sequence = np.round(2 * (np.random.rand(*cD.shape) - 0.5))
        if wm_vector[i] == 0:
            cD = cD + strength * pn_sequence
    
    detail_coeffs[0] = (cH, cV, cD)
    new_coeffs = [cA] + detail_coeffs
    watermarked_image = pywt.waverec2(new_coeffs, wavelet)
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark(watermarked_image: np.ndarray, watermark_size: Tuple[int, int],
                     wavelet: str = 'sym4', level: int = 1, key: int = 40,
                     threshold_factor: float = 1.5) -> np.ndarray:
    """
    Extracts the embedded watermark from a watermarked image using DWT and PN sequence correlation.

    The process involves:    
    1. Performing DWT on the watermarked image to obtain coefficients.
    2. Regenerating the same pseudo-random (PN) sequences used during embedding. This requires 
       the same secret key (if used) and the same DWT parameters (wavelet, level) to ensure 
       the PN sequences match the shape of the DWT coefficients (`cD_extracted`).
    3. For each bit of the original watermark:
        a. Generate the corresponding PN sequence.
        b. Flatten the `cD_extracted` coefficients and the PN sequence.
        c. Calculate the Pearson correlation coefficient between the flattened `cD_extracted` 
           and the flattened PN sequence.
    4. Determine a threshold based on the mean of all calculated correlations and a threshold factor.
    5. Reconstruct the watermark: if a correlation is above the threshold, the bit is considered 0,
       otherwise 1 (this matches the embedding logic where PN sequence is added for bit 0).

    Args:
        watermarked_image (np.ndarray): The image from which to extract the watermark.
        watermark_size (Tuple[int, int]): The dimensions (rows, cols) of the original watermark, 
                                         used to reshape the extracted bit vector.
        wavelet (str, optional): The type of wavelet to use for DWT. Defaults to 'sym4'.
                                 Must match the wavelet used during embedding.
        level (int, optional): The decomposition level for DWT. Defaults to 1.
                               Must match the level used during embedding.
        key (int, optional): The secret key (seed for random number generator) used during 
                             embedding. If None, no seed is set. Defaults to None.
        threshold_factor (float, optional): A factor to multiply with the mean of correlations 
                                           to determine the extraction threshold. Defaults to 1.5.

    Returns:
        np.ndarray: The extracted binary watermark, reshaped to `watermark_size`.
    """
    # Step 1: Initialize random seed if a key is provided for PN sequence generation
    if key is not None:
        np.random.seed(key)
    
    # Get original watermark dimensions for reshaping the final extracted bit vector
    Mwm, Nwm = watermark_size
    wm_length = Mwm * Nwm # Total number of bits in the watermark
    
    # Step 2: Perform DWT on the watermarked image
    # This should use the same wavelet and level as during embedding
    coeffs = pywt.wavedec2(watermarked_image, wavelet, level=level)
    # We are interested in the detail coefficients, specifically cD (diagonal)
    _, detail_coeffs = coeffs[0], list(coeffs[1:])
    # cD_extracted are the diagonal detail coefficients from the watermarked image
    _, _, cD_extracted = detail_coeffs[0] 
    # Flatten cD_extracted for correlation calculation. It contains the watermark signal + noise.
    cD_flat = cD_extracted.flatten()
    
    # Initialize an array to store correlation values for each watermark bit
    correlations = np.zeros(wm_length)
    
    # Step 3: Pre-calculate statistics for cD_flat to speed up correlation in the loop
    # These are constant for all PN sequence comparisons against this cD_flat
    mean_cD_flat = np.mean(cD_flat)
    std_cD_flat = np.std(cD_flat)
    # Center cD_flat (subtract mean) for Pearson correlation formula
    cD_flat_centered = cD_flat - mean_cD_flat
    N = len(cD_flat) # Number of elements in the flattened cD coefficients

    # Handle cases where cD_flat has no variance (e.g., if it's a constant array)
    if std_cD_flat == 0: 
        correlations.fill(0.0) # All correlations will be 0 or undefined (NaN)
    else:
        # Step 4: Iterate through each bit of the expected watermark
        for i in range(wm_length):
            # Step 4a: Generate the same PN sequence as used for this bit during embedding.
            # Crucially, this uses `*cD_extracted.shape` to ensure the PN sequence has the
            # same dimensions as the DWT coefficient array it was (notionally) added to.
            # The np.random.seed(key) at the start ensures this sequence matches embedding.
            pn_sequence_2D = np.round(2 * (np.random.rand(*cD_extracted.shape) - 0.5))
            # Flatten the 2D PN sequence to a 1D vector for correlation
            pn_flat = pn_sequence_2D.flatten()
            
            # Step 4b: Calculate statistics for the current pn_flat
            mean_pn_flat = np.mean(pn_flat)
            std_pn_flat = np.std(pn_flat)

            # Handle cases where pn_flat has no variance (e.g., if it's all zeros or all ones)
            if std_pn_flat == 0:
                correlations[i] = 0.0 # Correlation is 0 or undefined
            else:
                # Center pn_flat for Pearson correlation
                pn_flat_centered = pn_flat - mean_pn_flat
                
                # Step 4c: Calculate Pearson correlation coefficient manually
                # r = cov(X, Y) / (std(X) * std(Y))
                # cov(X,Y) = sum((X - mean(X)) * (Y - mean(Y))) / N
                # Here, X is cD_flat_centered, Y is pn_flat_centered
                # np.dot(cD_flat_centered, pn_flat_centered) calculates sum((X-mean(X))*(Y-mean(Y)))
                cov_sum_prod = np.dot(cD_flat_centered, pn_flat_centered)
                cov = cov_sum_prod / N # Covariance
                correlations[i] = cov / (std_cD_flat * std_pn_flat) # Pearson correlation
    
    # Step 5: Determine the threshold for deciding watermark bits
    # A simple adaptive threshold based on the mean of calculated correlations and a factor
    threshold = np.mean(correlations) * threshold_factor
    
    # Step 6: Reconstruct the watermark vector
    # Initialize recovered vector with all 1s.
    recovered_vector = np.ones(wm_length)
    # If correlation for a bit is greater than the threshold, it implies the PN sequence was present
    # (strong positive correlation), which corresponds to an embedded bit '0' in this scheme.
    recovered_vector[correlations > threshold] = 0
    
    # Reshape the 1D recovered vector back to the original 2D watermark dimensions
    recovered_watermark = recovered_vector.reshape(Mwm, Nwm)
    
    return recovered_watermark

def calculate_metrics(original_image: np.ndarray, watermarked_image: np.ndarray,
                       original_watermark: np.ndarray, recovered_watermark: np.ndarray) -> dict:
    """
    Calculate quality metrics
    """
    # Calculate PSNR between original and watermarked images
    psnr_value = float(psnr(original_image, watermarked_image))
    
    # Calculate correlation coefficient between original and recovered watermarks
    cc_val = np.corrcoef(original_watermark.flatten(), recovered_watermark.flatten())[0, 1]
    cc = float(cc_val) if not np.isnan(cc_val) else 0.0
    
    # Calculate normalized correlation
    # Ensure denominators are not zero to avoid division by zero warning / NaN
    sum_orig_sq = np.sum(original_watermark**2)
    sum_rec_sq = np.sum(recovered_watermark**2)
    if sum_orig_sq == 0 or sum_rec_sq == 0:
        ncc = 0.0
    else:
        ncc_val = np.sum(original_watermark * recovered_watermark) / (np.sqrt(sum_orig_sq) * np.sqrt(sum_rec_sq))
        ncc = float(ncc_val) if not np.isnan(ncc_val) else 0.0
    
    # Calculate bit error rate
    bit_errors = int(np.sum(original_watermark != recovered_watermark))
    total_bits = int(original_watermark.size)
    ber = float(bit_errors / total_bits) if total_bits > 0 else 0.0
    
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
    plt.figure(figsize=(20, 16))
    
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
        # Ensure the directory of save_path exists, if save_path includes a directory
        save_dir = os.path.dirname(save_path)
        if save_dir: # If save_dir is not an empty string (i.e., path includes a directory)
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path) # Use save_path directly
        plt.close() # Close the figure to free memory
    else:
        plt.show()


def salt_and_pepper(img, amount=0.01):
    out = img.copy()
    num_salt = int(np.ceil(amount * img.size * 0.5))
    num_pepper = int(np.ceil(amount * img.size * 0.5))

    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0
    return out

