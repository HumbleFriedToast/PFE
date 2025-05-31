import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from typing import Tuple, Optional, Dict, List, Any
import logging
import pandas as pd
# from tqdm import tqdm # tqdm might not be needed anymore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Generic Attack Functions (Copied from apply_attacks_and_metrics.py) ---
# All attack functions will be removed.
# --- End of Attack Functions ---

# --- Common Helper Functions ---
def load_images_combined(cover_path: str, watermark_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cover and watermark images with error handling.
    (Adapted from dwt.py for its logging)
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
            return None, None
            
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE) # Load watermark as grayscale initially
        if watermark is None:
            logger.error(f"Failed to load watermark image from {watermark_path}")
            return None, None
            
        return cover_image, watermark
    except Exception as e:
        logger.error(f"Error loading images: {str(e)}", exc_info=True)
        return None, None

def preprocess_images_svd(cover_image: np.ndarray, watermark: np.ndarray, max_watermark_size=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess images for DWT-SVD watermarking.
    Cover image to float, watermark normalized to [0,1] after optional resize.
    (Adapted from dwt_svd.py)
    """
    logger.info("Preprocessing images for DWT-SVD...")
    cover_image_float = cover_image.astype(np.float64)
    
    processed_watermark = watermark.copy() # Work on a copy
    if max_watermark_size is not None:
        h, w = processed_watermark.shape
        if h * w > max_watermark_size:
            ratio = np.sqrt(max_watermark_size / (h * w))
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            processed_watermark = cv2.resize(processed_watermark, (new_w, new_h))
            logger.info(f"SVD: Resized watermark from {h}x{w} to {new_h}x{new_w}")
    
    normalized_watermark = processed_watermark.astype(np.float64) / 255.0
    return cover_image_float, normalized_watermark

def preprocess_images_ss(cover_image: np.ndarray, watermark: np.ndarray, max_watermark_size=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess images for DWT-SS watermarking.
    Cover image to float, watermark binarized after optional resize.
    (Adapted from dwt.py)
    """
    logger.info("Preprocessing images for DWT-SS...")
    cover_image_float = cover_image.astype(np.float64)

    processed_watermark = watermark.copy() # Work on a copy
    if max_watermark_size is not None:
        h, w = processed_watermark.shape
        if h * w > max_watermark_size:
            ratio = np.sqrt(max_watermark_size / (h * w))
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            processed_watermark = cv2.resize(processed_watermark, (new_w, new_h))
            logger.info(f"SS: Resized watermark from {h}x{w} to {new_h}x{new_w}")
    
    # Binarize the watermark (0 or 1)
    binary_watermark = (processed_watermark > 128).astype(np.uint8) 
    return cover_image_float, binary_watermark

# DWT-SVD functions (Adapted from dwt_svd.py)
def embed_watermark_dwt_svd(cover_image: np.ndarray, watermark: np.ndarray, 
                           wavelet: str = 'sym4', level: int = 1, 
                           strength: float = 0.05, subband: str = 'LL') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Embed watermark into cover image using DWT-SVD hybrid technique.
    Watermark should be preprocessed (normalized to [0,1]).
    Cover image should be preprocessed (astype float).
    """
    logger.info(f"Embedding with DWT-SVD: wavelet={wavelet}, level={level}, strength={strength}, subband={subband}")
    Mc, Nc = cover_image.shape
    # Mwm, Nwm = watermark.shape # Watermark is already resized if needed during preprocessing
    
    coeffs = pywt.wavedec2(cover_image, wavelet, level=level) # level is n
    
    if subband == 'LL':
        target_coeff_block = coeffs[0] # This is cA_n (LL subband at the coarsest level n)
    elif subband in ['LH', 'HL', 'HH']:
        if level == 0: 
            raise ValueError("DWT level must be >= 1 for LH, HL, HH subbands.")
        if len(coeffs) < 2: 
            logger.error(f"Unexpected coeffs structure for DWT level {level}. Coeffs length: {len(coeffs)}")
            raise ValueError(f"Cannot extract detail subbands for DWT level {level}.")

        detail_coeffs_at_coarsest_level = coeffs[1] # This is (cH_n, cV_n, cD_n)
        
        if subband == 'LH':
            target_coeff_block = detail_coeffs_at_coarsest_level[0] # cH_n
        elif subband == 'HL':
            target_coeff_block = detail_coeffs_at_coarsest_level[1] # cV_n
        elif subband == 'HH': # HH
            target_coeff_block = detail_coeffs_at_coarsest_level[2] # cD_n
    else:
        raise ValueError(f"Invalid subband specified: {subband}. Must be 'LL', 'LH', 'HL', or 'HH'.")
    
    target_h, target_w = target_coeff_block.shape
    watermark_resized_to_target: np.ndarray
    if watermark.shape != target_coeff_block.shape:
        logger.info(f"SVD Embed: Resizing watermark from {watermark.shape} to {target_coeff_block.shape} to match target DWT block.")
        watermark_resized_to_target = cv2.resize(watermark, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        watermark_resized_to_target = watermark.copy()
    
    U_cover, S_cover, Vt_cover = np.linalg.svd(target_coeff_block, full_matrices=False)
    U_wm, S_wm, Vt_wm = np.linalg.svd(watermark_resized_to_target, full_matrices=False)
    
    # Ensure S_wm is not longer than S_cover for embedding
    len_s_cover = len(S_cover)
    s_wm_for_embedding = S_wm[:len_s_cover] # Truncate S_wm if longer
    s_cover_for_embedding = S_cover[:len(s_wm_for_embedding)] # Match length

    S_watermarked_segment = s_cover_for_embedding + strength * s_wm_for_embedding
    
    # Reconstruct S_watermarked correctly
    S_watermarked = S_cover.copy() # Start with original S_cover
    S_watermarked[:len(S_watermarked_segment)] = S_watermarked_segment # Replace the segment

    # Reconstruct the watermarked coefficient matrix
    # Ensure the diagonal matrix for S_watermarked matches dimensions of U_cover and Vt_cover
    diag_S_watermarked = np.zeros((U_cover.shape[1], Vt_cover.shape[0]))
    min_dim_s = min(len(S_watermarked), diag_S_watermarked.shape[0], diag_S_watermarked.shape[1])
    diag_S_watermarked[:min_dim_s, :min_dim_s] = np.diag(S_watermarked[:min_dim_s])

    target_coeff_watermarked = U_cover @ diag_S_watermarked @ Vt_cover
    
    coeffs_watermarked_list = list(coeffs) # Make it mutable
    
    if subband == 'LL':
        coeffs_watermarked_list[0] = target_coeff_watermarked
    elif subband in ['LH', 'HL', 'HH']:
        if level == 0: # Should have been caught by earlier checks
            raise ValueError("DWT level must be >= 1 to update detail subbands.")
        if len(coeffs_watermarked_list) < 2:
             logger.error(f"Unexpected coeffs_watermarked_list structure for DWT level {level}. Length: {len(coeffs_watermarked_list)}")
             raise ValueError(f"Cannot update detail subbands for DWT level {level}.")

        current_details_tuple = list(coeffs_watermarked_list[1]) # (cH_n, cV_n, cD_n)
        if subband == 'LH':
            current_details_tuple[0] = target_coeff_watermarked
        elif subband == 'HL':
            current_details_tuple[1] = target_coeff_watermarked
        elif subband == 'HH': # HH
            current_details_tuple[2] = target_coeff_watermarked
        coeffs_watermarked_list[1] = tuple(current_details_tuple)
    # No else needed, invalid subband caught earlier
    
    watermarked_image = pywt.waverec2(coeffs_watermarked_list, wavelet)
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    
    embedding_info = {
        'U_cover': U_cover,
        'Vt_cover': Vt_cover,
        'S_cover': S_cover, # Original singular values of the cover block
        'U_wm_original': U_wm, 
        'Vt_wm_original': Vt_wm,
        'original_watermark_shape': watermark.shape, # Shape of watermark *before* resizing to target DWT block
        'target_coeff_block_shape': target_coeff_block.shape
    }
    return watermarked_image, embedding_info

def extract_watermark_dwt_svd(watermarked_image: np.ndarray, embedding_info: Dict[str, Any],
                             wavelet: str = 'sym4', level: int = 1, 
                             strength: float = 0.05, subband: str = 'LL') -> np.ndarray:
    """
    Extract watermark from watermarked image using DWT-SVD.
    """
    logger.info(f"Extracting with DWT-SVD: wavelet={wavelet}, level={level}, strength={strength}, subband={subband}")
    U_cover = embedding_info['U_cover']
    Vt_cover = embedding_info['Vt_cover']
    S_cover_original_block = embedding_info['S_cover'] # S_cover of the block used for embedding
    U_wm_original = embedding_info['U_wm_original']
    Vt_wm_original = embedding_info['Vt_wm_original']
    original_watermark_shape = embedding_info['original_watermark_shape'] # Watermark shape before it was resized to target DWT block
    # target_coeff_block_shape = embedding_info['target_coeff_block_shape']

    coeffs_attacked = pywt.wavedec2(watermarked_image.astype(np.float64), wavelet, level=level)
    
    target_coeff_attacked: np.ndarray
    if subband == 'LL':
        target_coeff_attacked = coeffs_attacked[0] # cA_n
    elif subband in ['LH', 'HL', 'HH']:
        if level == 0:
            raise ValueError("DWT level must be >= 1 for LH, HL, HH subbands for extraction.")
        if len(coeffs_attacked) < 2:
            logger.error(f"Unexpected coeffs_attacked structure for DWT level {level} during extraction. Length: {len(coeffs_attacked)}")
            raise ValueError(f"Cannot extract detail subbands for DWT level {level} during extraction.")
        
        detail_coeffs_at_coarsest_level_attacked = coeffs_attacked[1] # (cH_n, cV_n, cD_n) from attacked image
        
        if subband == 'LH':
            target_coeff_attacked = detail_coeffs_at_coarsest_level_attacked[0] # cH_n
        elif subband == 'HL':
            target_coeff_attacked = detail_coeffs_at_coarsest_level_attacked[1] # cV_n
        elif subband == 'HH': # HH
            target_coeff_attacked = detail_coeffs_at_coarsest_level_attacked[2] # cD_n
    else:
        raise ValueError(f"Invalid subband specified for extraction: {subband}. Must be 'LL', 'LH', 'HL', or 'HH'.")

    # SVD of the attacked DWT block
    _U_attacked, S_attacked, _Vt_attacked = np.linalg.svd(target_coeff_attacked, full_matrices=False)
    
    # Extract watermark singular values
    # Ensure S_cover_original_block is not longer than S_attacked for subtraction
    len_s_attacked = len(S_attacked)
    s_cover_for_extraction = S_cover_original_block[:len_s_attacked]
    s_attacked_for_extraction = S_attacked[:len(s_cover_for_extraction)]

    if strength == 0: strength = 1e-9 # Avoid division by zero
    S_extracted_segment = (s_attacked_for_extraction - s_cover_for_extraction) / strength
    
    # The S_extracted_segment should be used to form a diagonal matrix that is compatible with U_wm_original and Vt_wm_original
    # When full_matrices=False for A (M,N) -> U (M,K), S (K,), Vt (K,N) where K = min(M,N)
    # U_wm_original.shape[1] is K (number of singular values)
    num_singular_values_wm = U_wm_original.shape[1] 
    S_diag_extracted = np.zeros((num_singular_values_wm, num_singular_values_wm))
    
    len_S_extracted_segment = len(S_extracted_segment)
    # Fill the diagonal with the extracted singular values, up to the available number of singular values
    diag_fill_len = min(len_S_extracted_segment, num_singular_values_wm)
    S_diag_extracted[:diag_fill_len, :diag_fill_len] = np.diag(S_extracted_segment[:diag_fill_len])
    
    # Reconstruct the watermark (this will be of shape target_coeff_block_shape)
    # U_wm_original (M_w_rt, K_w) @ S_diag_extracted (K_w, K_w) @ Vt_wm_original (K_w, N_w_rt)
    watermark_extracted_resized = U_wm_original @ S_diag_extracted @ Vt_wm_original
    
    # Normalize extracted watermark to [0, 1] range first
    min_val = np.min(watermark_extracted_resized)
    max_val = np.max(watermark_extracted_resized)
    if max_val - min_val > 1e-9:
        watermark_extracted_normalized = (watermark_extracted_resized - min_val) / (max_val - min_val)
    else:
        watermark_extracted_normalized = np.zeros_like(watermark_extracted_resized)

    # Resize back to the original watermark's actual dimensions before it was normalized/resized for embedding by SVD process itself
    # This 'original_watermark_shape' is the shape of the watermark that was initially passed to embed_watermark_dwt_svd (after initial preprocessing)
    if watermark_extracted_normalized.shape != original_watermark_shape:
        logger.info(f"SVD Extract: Resizing extracted watermark from {watermark_extracted_normalized.shape} to original input shape {original_watermark_shape}.")
        watermark_extracted_final = cv2.resize(watermark_extracted_normalized, 
                                             (original_watermark_shape[1], original_watermark_shape[0]), 
                                             interpolation=cv2.INTER_LINEAR)
    else:
        watermark_extracted_final = watermark_extracted_normalized
    
    watermark_extracted_grayscale_uint8 = (watermark_extracted_final * 255).astype(np.uint8)
    
    # Binarize the extracted SVD watermark to be strictly black and white
    binary_threshold_value = 128  # Common threshold for 0-255 range
    # Resulting image will have values 0 (black) or 255 (white)
    watermark_extracted_binary = (watermark_extracted_grayscale_uint8 > binary_threshold_value).astype(np.uint8) * 255
    
    logger.info(f"SVD Extract: Binarized extracted watermark using threshold {binary_threshold_value}.")
    return watermark_extracted_binary

# DWT-SS functions (Adapted from dwt.py)
def embed_watermark_ss(cover_image: np.ndarray, watermark: np.ndarray, 
                    wavelet: str = 'sym4', level: int = 2, 
                    strength: float = 2.0, key: Optional[int] = None) -> np.ndarray:
    """
    Embed watermark into cover image using DWT Spread Spectrum.
    Cover image should be preprocessed (astype float).
    Watermark should be preprocessed (binary 0 or 1).
    """
    logger.info(f"Embedding with DWT-SS: wavelet={wavelet}, level={level}, strength={strength}, key={key}")
    if key is not None:
        np.random.seed(key)
    
    # Ensure watermark is 1D vector of 0s and 1s
    wm_vector = watermark.reshape(-1)
    if not np.all(np.isin(wm_vector, [0, 1])):
        logger.warning("SS Embed: Watermark for SS method should be binary (0s and 1s). Binarizing with >0.5 threshold.")
        wm_vector = (wm_vector > 0.5).astype(np.uint8) # Ensure it's 0 or 1

    coeffs = pywt.wavedec2(cover_image, wavelet, level=level)
    # Target the cD (diagonal) coefficients of the first level of detail for SS
    # This is a common choice for DWT spread spectrum. 
    # pywt.wavedec2 returns coeffs as [cA_level, (cH_level, cV_level, cD_level), ..., (cH_1, cV_1, cD_1)]
    # So, coeffs[level] would be (cH_1, cV_1, cD_1) if level index is used directly based on common understanding.
    # Or, if always targeting highest frequency band details (cD1), then it's coeffs[-1][2]
    # Based on dwt.py's original embed_watermark, it seems to target cD of the *first* detail level (highest freq)
    # The structure is cA, (cH_level, cV_level, cD_level), (cH_level-1, cV_level-1, cD_level-1)... (cH_1, cV_1, cD_1)
    # So, if level=1, coeffs are [cA1, (cH1,cV1,cD1)]. If level=2, [cA2, (cH2,cV2,cD2), (cH1,cV1,cD1)]
    # The original dwt.py used `coeffs[1:]` then `detail_coeffs[0]` which is (cH_level, cV_level, cD_level)
    # Let's stick to modifying cD coefficients of the specified `level`'s decomposition (e.g., cD_level from (cH_level, cV_level, cD_level)) 
    # No, the original dwt.py used `coeffs[0], list(coeffs[1:])` and then `detail_coeffs[0]` which implies it was using the highest level coeffs
    # cA is coeffs[0]. coeffs[1] is a tuple of (cH, cV, cD) for that level if level=1.
    # For multi-level, coeffs[0] is cA_L. coeffs[1] is (cH_L, cV_L, cD_L), ... coeffs[L] are details for level 1.
    # The original code: cA, detail_coeffs = coeffs[0], list(coeffs[1:]), cH, cV, cD = detail_coeffs[0]
    # This means it was modifying the (cH_level, cV_level, cD_level) if level=level. This seems correct for typical DWT. 

    coeffs_list = list(coeffs) # Make mutable
    # We need to find the correct index for the (cH, cV, cD) tuple for the specified `level`.
    # pywt.wavedec2 returns [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    # The level parameter in wavedec2 corresponds to n. So cD_n is coeffs[1][2].
    # This means we are modifying the LOWEST frequency detail coefficients for the given decomposition.
    # If you want to modify highest frequency (level 1 details regardless of total levels), it's coeffs[level][2]
    # Let's assume the 'level' param means the decomposition depth, and we modify the cD from THAT depth's details.
    # i.e., cD_level. This is coeffs[1][2].

    try:
        target_detail_coeffs_tuple = list(coeffs_list[1]) # (cH_level, cV_level, cD_level)
        cD_target_block = target_detail_coeffs_tuple[2] # cD_level
    except IndexError:
        logger.error(f"SS Embed: Could not access target cD coefficients for level {level}. Coefficients structure: {len(coeffs_list)} elements.")
        # Fallback or re-raise. For now, try to use the highest frequency cD (cD1) if level > 0
        if level > 0 and len(coeffs_list) > level:
             logger.warning(f"SS Embed: Falling back to cD of level 1 (highest frequency details).")
             target_detail_coeffs_tuple_idx = level # Index for (cH_1, cV_1, cD_1) is L if decomp level is L
             target_detail_coeffs_tuple = list(coeffs_list[target_detail_coeffs_tuple_idx])
             cD_target_block = target_detail_coeffs_tuple[2]
        else:
            logger.error("SS Embed: Cannot determine target cD block. Embedding will likely fail or be incorrect.")
            return cover_image.astype(np.uint8) # Return original if we can't proceed

    cD_modified = cD_target_block.copy()
    for i in range(len(wm_vector)):
        pn_sequence = np.round(2 * (np.random.rand(*cD_target_block.shape) - 0.5)) # PN sequence per bit
        if wm_vector[i] == 0: # As per original dwt.py logic
            cD_modified = cD_modified + strength * pn_sequence
    
    target_detail_coeffs_tuple[2] = cD_modified
    coeffs_list[1] = tuple(target_detail_coeffs_tuple)
    
    watermarked_image = pywt.waverec2(coeffs_list, wavelet)
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark_ss(watermarked_image: np.ndarray, watermark_size: Tuple[int, int],
                     wavelet: str = 'sym4', level: int = 2, key: Optional[int] = None,
                     threshold_factor: float = 1.5) -> np.ndarray:
    """
    Extracts the embedded watermark from a watermarked image using DWT-SS (Spread Spectrum).
    Watermark size is (rows, cols) of the original binary watermark.
    """
    logger.info(f"Extracting with DWT-SS: wavelet={wavelet}, level={level}, key={key}, threshold_factor={threshold_factor}")
    if key is not None:
        np.random.seed(key)
    
    Mwm, Nwm = watermark_size
    wm_length = Mwm * Nwm
    
    coeffs_attacked = pywt.wavedec2(watermarked_image.astype(np.float64), wavelet, level=level)
    
    # Determine the target cD block based on the same logic as embedding
    cD_extracted_block: Optional[np.ndarray] = None
    try:
        target_detail_coeffs_tuple_attacked = list(coeffs_attacked[1])
        cD_extracted_block = target_detail_coeffs_tuple_attacked[2]
    except IndexError:
        logger.error(f"SS Extract: Could not access target cD coefficients for level {level}. Coefficients structure: {len(coeffs_attacked)} elements.")
        if level > 0 and len(coeffs_attacked) > level:
             logger.warning(f"SS Extract: Falling back to cD of level 1 (highest frequency details).")
             target_detail_coeffs_tuple_idx = level
             target_detail_coeffs_tuple_attacked = list(coeffs_attacked[target_detail_coeffs_tuple_idx])
             cD_extracted_block = target_detail_coeffs_tuple_attacked[2]

    if cD_extracted_block is None:
        logger.error("SS Extract: Failed to get cD block. Returning empty watermark.")
        return np.zeros(watermark_size, dtype=np.uint8)

    cD_flat = cD_extracted_block.flatten()
    correlations = np.zeros(wm_length)
    
    mean_cD_flat = np.mean(cD_flat)
    std_cD_flat = np.std(cD_flat)
    cD_flat_centered = cD_flat - mean_cD_flat
    N_elements = len(cD_flat)

    if std_cD_flat == 0:
        correlations.fill(0.0)
    else:
        for i in range(wm_length):
            pn_sequence_2D = np.round(2 * (np.random.rand(*cD_extracted_block.shape) - 0.5))
            pn_flat = pn_sequence_2D.flatten()
            mean_pn_flat = np.mean(pn_flat)
            std_pn_flat = np.std(pn_flat)
            if std_pn_flat == 0:
                correlations[i] = 0.0
            else:
                pn_flat_centered = pn_flat - mean_pn_flat
                cov_sum_prod = np.dot(cD_flat_centered, pn_flat_centered)
                # cov = cov_sum_prod / N_elements # Definition of covariance
                # correlations[i] = cov / (std_cD_flat * std_pn_flat) # Pearson correlation
                # Simplified Pearson for 0-mean PN: sum(cD_i * pn_i) / (N * std_cD * std_pn)
                # Or, more directly, the formula used in the original dwt.py:
                correlations[i] = np.sum(cD_flat_centered * pn_flat_centered) / (N_elements * std_cD_flat * std_pn_flat) 
                # The original dwt.py used: cov_sum_prod = np.dot(cD_flat_centered, pn_flat_centered); cov = cov_sum_prod / N; correlations[i] = cov / (std_cD_flat * std_pn_flat)
                # This is equivalent to np.corrcoef(cD_flat, pn_flat)[0,1] if done carefully. Let's stick to original structure.
                # The original dwt.py did: cov_sum_prod = np.dot(cD_flat_centered, pn_flat_centered); cov = cov_sum_prod / N; correlations[i] = cov / (std_cD_flat * std_pn_flat)
                # This is the same as my previous: np.dot(cD_flat_centered, pn_flat_centered) / (N_elements * std_cD_flat * std_pn_flat)

    threshold = np.mean(correlations) * threshold_factor
    recovered_vector = np.ones(wm_length, dtype=np.uint8)
    recovered_vector[correlations > threshold] = 0 # If correlation high, bit is 0 (PN sequence was added)
    
    recovered_watermark = recovered_vector.reshape(Mwm, Nwm)
    return recovered_watermark
    
def calculate_metrics_combined(original_image: np.ndarray, watermarked_image: np.ndarray,
                               original_watermark_raw: np.ndarray, # This is the unprocessed, original watermark (e.g. grayscale 0-255)
                               recovered_watermark: np.ndarray, # This is the output from extraction (SVD: 0-255 float-like, SS: binary 0/1)
                               method: str) -> Optional[Dict[str, float]]:
    """
    Calculate quality metrics for SVD or SS methods.
    original_watermark_raw is the ground truth (e.g., uint8 grayscale).
    recovered_watermark is what the extraction method returned.
    """
    logger.info(f"Calculating metrics for {method} method...")
    metrics = {}

    # PSNR (Cover vs Watermarked)
    try:
        metrics['psnr_cover_vs_watermarked'] = float(psnr(original_image, watermarked_image))
    except Exception as e:
        logger.error(f"Error calculating PSNR: {e}")
        metrics['psnr_cover_vs_watermarked'] = 0.0

    # --- Watermark-specific metrics --- 
    # For SVD, recovered_watermark is typically [0,255] uint8 (after final conversion). 
    # original_watermark_raw is also [0,255] uint8.
    # For SS, recovered_watermark is binary [0,1] uint8. 
    # original_watermark_raw needs to be binarized for SS comparison.

    original_wm_for_comparison: np.ndarray
    recovered_wm_for_comparison: np.ndarray

    if method == "SVD":
        original_wm_for_comparison = original_watermark_raw.astype(np.float64)
        recovered_wm_for_comparison = recovered_watermark.astype(np.float64)
        # Ensure same shape for SVD metrics (recovered might have been resized during extraction)
        if original_wm_for_comparison.shape != recovered_wm_for_comparison.shape:
            logger.info(f"Metrics (SVD): Resizing recovered watermark from {recovered_wm_for_comparison.shape} to {original_wm_for_comparison.shape}")
            recovered_wm_for_comparison = cv2.resize(recovered_wm_for_comparison, 
                                                   (original_wm_for_comparison.shape[1], original_wm_for_comparison.shape[0]),
                                                   interpolation=cv2.INTER_LINEAR)
    elif method == "SS":
        # Binarize the raw original watermark for comparison with SS output
        original_wm_for_comparison = (original_watermark_raw > 128).astype(np.uint8)
        recovered_wm_for_comparison = recovered_watermark.astype(np.uint8) # Already binary from SS extraction
        # Ensure same shape for SS metrics
        if original_wm_for_comparison.shape != recovered_wm_for_comparison.shape:
            logger.info(f"Metrics (SS): Resizing recovered binary watermark from {recovered_wm_for_comparison.shape} to {original_wm_for_comparison.shape}")
            recovered_wm_for_comparison = cv2.resize(recovered_wm_for_comparison, 
                                                   (original_wm_for_comparison.shape[1], original_wm_for_comparison.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
    else:
        logger.error(f"Unknown method {method} for metrics calculation.")
        return None

    # NCC (Normalized Cross-Correlation)
    try:
        orig_flat = original_wm_for_comparison.flatten()
        rec_flat = recovered_wm_for_comparison.flatten()
        
        # Robust NCC: subtract mean, divide by std dev * N
        mean_orig = np.mean(orig_flat)
        mean_rec = np.mean(rec_flat)
        std_orig = np.std(orig_flat)
        std_rec = np.std(rec_flat)

        if std_orig == 0 or std_rec == 0 or len(orig_flat) == 0:
            ncc_val = 0.0
        else:
            # Pearson correlation coefficient formula
            num = np.sum((orig_flat - mean_orig) * (rec_flat - mean_rec))
            den = len(orig_flat) * std_orig * std_rec # Or (N-1)*std_orig*std_rec for sample std
            # Or use np.sqrt(np.sum((orig_flat - mean_orig)**2) * np.sum((rec_flat - mean_rec)**2))
            # Let's use a common np.corrcoef way if possible, or the direct formula carefully
            # Using: sum ( (X-meanX) * (Y-meanY) ) / sqrt ( sum(X-meanX)^2 * sum(Y-meanY)^2 )
            numerator = np.sum((orig_flat - mean_orig) * (rec_flat - mean_rec))
            denominator = np.sqrt(np.sum((orig_flat - mean_orig)**2) * np.sum((rec_flat - mean_rec)**2))
            if denominator == 0:
                ncc_val = 1.0 if numerator == 0 else 0.0 # If both are constant and equal, or if one is constant
            else:
                ncc_val = numerator / denominator
        
        metrics['ncc_watermark'] = float(ncc_val) if not np.isnan(ncc_val) else 0.0
    except Exception as e:
        logger.error(f"Error calculating NCC: {e}")
        metrics['ncc_watermark'] = 0.0

    # MSE (Mean Squared Error) for watermarks
    try:
        # For SS, this will be MSE between binary images. For SVD, between grayscale.
        metrics['mse_watermark'] = float(np.mean((original_wm_for_comparison - recovered_wm_for_comparison)**2))
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}")
        metrics['mse_watermark'] = float('inf')

    # BER (Bit Error Rate) and Accuracy - primarily for binary or binarized data
    # For SVD, we'll binarize both original_raw and recovered for BER/Accuracy
    try:
        if method == "SVD":
            # Binarize SVD results for BER/Accuracy
            original_binary_for_ber = (original_watermark_raw > 128).astype(np.uint8)
            recovered_binary_for_ber = (recovered_watermark > 128).astype(np.uint8)
            if original_binary_for_ber.shape != recovered_binary_for_ber.shape:
                recovered_binary_for_ber = cv2.resize(recovered_binary_for_ber,
                                                      (original_binary_for_ber.shape[1], original_binary_for_ber.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)
        else: # SS method already has binary inputs for comparison here
            original_binary_for_ber = original_wm_for_comparison.astype(np.uint8)
            recovered_binary_for_ber = recovered_wm_for_comparison.astype(np.uint8)
            # Shapes should match from above resizing, but double check
            if original_binary_for_ber.shape != recovered_binary_for_ber.shape:
                 recovered_binary_for_ber = cv2.resize(recovered_binary_for_ber, 
                                                       (original_binary_for_ber.shape[1], original_binary_for_ber.shape[0]), 
                                                       interpolation=cv2.INTER_NEAREST)

        bit_errors = int(np.sum(original_binary_for_ber != recovered_binary_for_ber))
        total_bits = int(original_binary_for_ber.size)
        ber_val = float(bit_errors / total_bits) if total_bits > 0 else 0.0
        metrics['ber_watermark'] = ber_val
        metrics['accuracy_watermark'] = 1.0 - ber_val
        metrics['bit_errors'] = float(bit_errors)
        metrics['total_bits'] = float(total_bits)
    except Exception as e:
        logger.error(f"Error calculating BER/Accuracy: {e}")
        metrics['ber_watermark'] = 1.0
        metrics['accuracy_watermark'] = 0.0
        metrics['bit_errors'] = float(original_watermark_raw.size if method == "SS" else original_watermark_raw.size)
        metrics['total_bits'] = float(original_watermark_raw.size if method == "SS" else original_watermark_raw.size)

    # SSIM (Structural Similarity Index) - Meaningful for grayscale/continuous watermarks (like SVD)
    if method == "SVD":
        try:
            # Using skimage.metrics.structural_similarity (requires skimage installation, already imported as psnr)
            # For SSIM, images should be in range [0,1] or data_range specified.
            # Our recovered_wm_for_comparison (SVD) is [0,255] float, original_wm_for_comparison (SVD) is [0,255] float.
            
            # To avoid dependency on full skimage for SSIM, use a simplified mean/variance/covariance approach for now
            # This was in dwt_svd.py, adapted for [0,255] range directly or normalized [0,1]
            # Let's normalize them to [0,1] for a more standard SSIM-like calculation for consistency
            
            orig_norm_ssim = original_wm_for_comparison / 255.0
            rec_norm_ssim = recovered_wm_for_comparison / 255.0

            mean_orig = np.mean(orig_norm_ssim)
            mean_rec = np.mean(rec_norm_ssim)
            var_orig = np.var(orig_norm_ssim)
            var_rec = np.var(rec_norm_ssim)
            cov = np.mean((orig_norm_ssim - mean_orig) * (rec_norm_ssim - mean_rec))
            
            K1, K2, L = 0.01, 0.03, 1.0 # L is max value of images (1.0 after normalization)
            C1 = (K1 * L)**2
            C2 = (K2 * L)**2
            
            numerator_ssim = (2 * mean_orig * mean_rec + C1) * (2 * cov + C2)
            denominator_ssim = (mean_orig**2 + mean_rec**2 + C1) * (var_orig + var_rec + C2)
            
            ssim_val = numerator_ssim / denominator_ssim if denominator_ssim != 0 else 0.0
            metrics['ssim_watermark'] = float(ssim_val) if not np.isnan(ssim_val) else 0.0
        except Exception as e:
            logger.error(f"Error calculating SSIM for SVD: {e}")
            metrics['ssim_watermark'] = 0.0
    else:
        metrics['ssim_watermark'] = 0.0 # Not typically used for binary SS watermarks directly

    return metrics

def plot_results_combined(original_image: np.ndarray, watermarked_image: np.ndarray,
                          original_watermark_raw: np.ndarray, # uint8 grayscale usually
                          recovered_watermark: np.ndarray, # SVD: uint8 grayscale, SS: uint8 binary
                          method: str, save_path: Optional[str] = None):
    """
    Plot original, watermarked, original watermark, and recovered watermark.
    Also plots the difference between watermarks, adapting to method.
    """
    logger.info(f"Plotting results for {method} method...")
    
    original_wm_display = original_watermark_raw.copy()
    recovered_wm_display = recovered_watermark.copy()

    # For SS, the recovered watermark is already binary. For display, original_raw should also be binarized.
    if method == "SS":
        original_wm_display = (original_watermark_raw > 128).astype(np.uint8) * 255 # Display as black/white
        # recovered_wm_display is already 0/1 from SS, scale to 0/255 for display
        recovered_wm_display = recovered_wm_display.astype(np.uint8) * 255

    # Ensure shapes match for difference calculation and display, especially for SVD
    if original_wm_display.shape != recovered_wm_display.shape:
        logger.info(f"Plot ({method}): Resizing recovered watermark from {recovered_wm_display.shape} to {original_wm_display.shape} for display consistency.")
        recovered_wm_display = cv2.resize(recovered_wm_display, 
                                         (original_wm_display.shape[1], original_wm_display.shape[0]),
                                         interpolation=cv2.INTER_NEAREST if method == "SS" else cv2.INTER_LINEAR)
    
    # Calculate difference image for cover vs watermarked
    diff_cover_vs_watermarked = np.abs(original_image.astype(np.float64) - watermarked_image.astype(np.float64))
    diff_cover_vs_watermarked_display = np.clip(diff_cover_vs_watermarked, 0, 255).astype(np.uint8)

    # Calculate difference between original and (potentially resized/binarized) recovered watermark
    wm_diff_float = np.abs(original_wm_display.astype(np.float64) - recovered_wm_display.astype(np.float64))
    wm_diff_display = np.clip(wm_diff_float, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Watermarking Results - Method: {method}', fontsize=16)
    
    axes[0, 0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Cover Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(watermarked_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Watermarked Cover Image')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(diff_cover_vs_watermarked_display, cmap='hot', vmin=0, vmax=255)
    axes[0, 2].set_title('Diff (Cover vs Watermarked)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(original_wm_display, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Original Watermark (Processed for Method)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(recovered_wm_display, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Recovered Watermark')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(wm_diff_display, cmap='hot', vmin=0, vmax=255)
    axes[1, 2].set_title('Diff (Original WM vs Recovered WM)')
    axes[1, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()

# --- Main Application Logic ---
def main():
    # CHOOSE WATERMARKING METHOD: "SVD" or "SS"
    WATERMARKING_METHOD = "SVD" # or "SS" 
    
    logger.info(f"Starting combined watermarking script with method: {WATERMARKING_METHOD}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- General Parameters ---
    cover_image_path = os.path.join(current_dir, 'cover12.jpg') # Replace with your cover image
    watermark_path = os.path.join(current_dir, 'watermark2.jpg') # Replace with your watermark
    
    output_dir_base_root = os.path.join(current_dir, 'combined_attack_results')
    output_dir_base = os.path.join(output_dir_base_root, WATERMARKING_METHOD)
    os.makedirs(output_dir_base, exist_ok=True)
    logger.info(f"Results will be saved in: {output_dir_base}")

    # --- Method-Specific Parameters ---
    params_svd = {
        'wavelet': 'db8', 'level': 3, 'embedding_strength': 70.0, 
        'target_subband': 'LL', 'max_watermark_size': 200000
    }
    params_ss = {
        'wavelet': 'sym4', 'level': 2, 'embedding_strength': 20.0, # typical strength for SS might be different
        'key': 42, 'max_watermark_size': None, 'extraction_threshold_factor': 1.5
    }

    if WATERMARKING_METHOD == "SVD":
        method_params = params_svd
    elif WATERMARKING_METHOD == "SS":
        method_params = params_ss
    else:
        logger.error(f"Unknown watermarking method: {WATERMARKING_METHOD}")
        return

    logger.info(f"Using parameters: {method_params}")

    # Load images
    logger.info("Loading images...")
    cover_image_original, watermark_original_raw = load_images_combined(cover_image_path, watermark_path)
    if cover_image_original is None or watermark_original_raw is None:
        logger.error("Failed to load images. Exiting.")
        return
    logger.info(f"Cover image size: {cover_image_original.shape}")
    logger.info(f"Original watermark raw size: {watermark_original_raw.shape}")

    # Preprocess images
    cover_image_processed: np.ndarray
    watermark_for_embedding: np.ndarray

    if WATERMARKING_METHOD == "SVD":
        cover_image_processed, watermark_for_embedding = preprocess_images_svd(
            cover_image_original, watermark_original_raw, method_params.get('max_watermark_size')
        )
    elif WATERMARKING_METHOD == "SS":
        cover_image_processed, watermark_for_embedding = preprocess_images_ss(
            cover_image_original, watermark_original_raw, method_params.get('max_watermark_size')
        )
    logger.info(f"Watermark size for embedding: {watermark_for_embedding.shape}")


    # Embed watermark (No Attack)
    watermarked_image_no_attack: Optional[np.ndarray] = None
    embedding_info_svd: Optional[Dict[str, Any]] = None # Specific to SVD

    logger.info(f"Embedding watermark (No Attack) using {WATERMARKING_METHOD}...")
    if WATERMARKING_METHOD == "SVD":
        watermarked_image_no_attack, embedding_info_svd = embed_watermark_dwt_svd(
            cover_image_processed, watermark_for_embedding,
            wavelet=method_params['wavelet'], level=method_params['level'],
            strength=method_params['embedding_strength'], subband=method_params['target_subband']
        )
    elif WATERMARKING_METHOD == "SS":
        watermarked_image_no_attack = embed_watermark_ss(
            cover_image_processed, watermark_for_embedding, # SS expects cover_image_float, binary_watermark
            wavelet=method_params['wavelet'], level=method_params['level'],
            strength=method_params['embedding_strength'], key=method_params.get('key')
        )
    
    if watermarked_image_no_attack is None:
        logger.error("Watermark embedding failed for no-attack scenario.")
        return
        
    no_attack_img_path = os.path.join(output_dir_base, f'watermarked_image_no_attack_{WATERMARKING_METHOD}.png')
    cv2.imwrite(no_attack_img_path, watermarked_image_no_attack)
    logger.info(f"Watermarked image (no attack) saved to {no_attack_img_path}")

    # --- No Attack Scenario ---
    logger.info("Evaluating no-attack scenario...")
    output_dir_no_attack = os.path.join(output_dir_base, 'no_attack')
    os.makedirs(output_dir_no_attack, exist_ok=True)

    recovered_watermark_no_attack: Optional[np.ndarray] = None
    if WATERMARKING_METHOD == "SVD":
        if embedding_info_svd:
            recovered_watermark_no_attack = extract_watermark_dwt_svd(
                watermarked_image_no_attack, embedding_info_svd,
                wavelet=method_params['wavelet'], level=method_params['level'],
                strength=method_params['embedding_strength'], subband=method_params['target_subband']
            )
    elif WATERMARKING_METHOD == "SS":
        recovered_watermark_no_attack = extract_watermark_ss(
            watermarked_image_no_attack, watermark_original_raw.shape, # Use original raw shape for SS extraction target size
            wavelet=method_params['wavelet'], level=method_params['level'],
            key=method_params.get('key'), threshold_factor=method_params['extraction_threshold_factor']
        )

    if recovered_watermark_no_attack is None:
        logger.error("Watermark extraction failed for no-attack scenario.")
        recovered_watermark_no_attack = np.zeros_like(watermark_original_raw) # Dummy for metrics
        
    metrics_no_attack = calculate_metrics_combined(
        cover_image_original, watermarked_image_no_attack,
        watermark_original_raw, recovered_watermark_no_attack, WATERMARKING_METHOD
    )
    if metrics_no_attack: # calculate_metrics might return None if it fails
        metrics_no_attack['attack_name'] = 'No Attack'
        metrics_no_attack['attack_param'] = 'N/A'
        logger.info(f"No Attack Metrics ({WATERMARKING_METHOD}): {metrics_no_attack}")
        all_metrics_list = [metrics_no_attack]
    else:
        all_metrics_list = []
        logger.error("Failed to calculate metrics for no-attack scenario.")

    plot_results_combined(
        cover_image_original, watermarked_image_no_attack,
        watermark_original_raw, recovered_watermark_no_attack, WATERMARKING_METHOD,
        save_path=os.path.join(output_dir_no_attack, f'results_no_attack_{WATERMARKING_METHOD}.png')
    )
    


    if all_metrics_list: # Should contain only no-attack metrics now
        metrics_df = pd.DataFrame(all_metrics_list)
        csv_path = os.path.join(output_dir_base, f'metrics_no_attack_{WATERMARKING_METHOD}.csv') # Changed CSV name
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Metrics for no-attack scenario saved to {csv_path}")
    else:
        logger.warning("No metrics were generated for the no-attack scenario.")
        
    logger.info(f"Combined watermarking script (No Attacks) ({WATERMARKING_METHOD}) finished.")

if __name__ == '__main__':
    main() 