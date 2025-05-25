import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import random

# --- Helper Function ---
def read_grayscale(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return img

# --- DCT Utilities ---
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# --- Frequency Region & Block_Size Indexing ---
def get_freq_position(block_size, region='mid'):
    if region == 'low':
        return (1, 1)
    elif region == 'mid':
        return (block_size // 2, block_size // 2)
    elif region == 'high':
        return (block_size - 2, block_size - 2)
    else:
        raise ValueError("Region must be 'low', 'mid', or 'high'")

def preprocess_watermark_bits(watermark_bits, cover_shape, block_size):
    h_blocks = cover_shape[0] // block_size
    w_blocks = cover_shape[1] // block_size
    expected_shape = (h_blocks, w_blocks)

    if watermark_bits.shape != expected_shape:
        watermark_bits = cv2.resize(
            watermark_bits.astype(np.uint8),
            (w_blocks, h_blocks),  # (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        _, watermark_bits = cv2.threshold(watermark_bits, 0, 1, cv2.THRESH_BINARY)

    return watermark_bits.astype(np.uint8)
def embed_watermark(cover, watermark_bits, block_size=8, alpha=10, region='mid'):
    if cover.ndim == 3:
        cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    watermark_binary = preprocess_watermark_bits(watermark_bits, cover.shape, block_size)

    h_blocks = cover.shape[0] // block_size
    w_blocks = cover.shape[1] // block_size
    watermarked = np.zeros_like(cover, dtype=np.float32)
    x, y = get_freq_position(block_size, region)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = cover[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = apply_dct(block)

            if watermark_binary[i, j] == 1:
                dct_block[x, y] += alpha
            else:
                dct_block[x, y] -= alpha

            idct_block = apply_idct(dct_block)
            watermarked[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = idct_block

    return np.clip(watermarked, 0, 255).astype(np.uint8), watermark_binary

# --- Basic DCT Extraction ---
def extract_watermark(watermarked, original_shape, block_size=8, alpha=10, region='mid'):
    if watermarked.ndim == 3:
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    h_blocks = original_shape[0] // block_size
    w_blocks = original_shape[1] // block_size
    extracted = np.zeros((h_blocks, w_blocks), dtype=np.uint8)
    x, y = get_freq_position(block_size, region)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = watermarked[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = apply_dct(block)
            extracted[i, j] = 1 if dct_block[x, y] > 0 else 0

    extracted_visual = cv2.resize(extracted * 255, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return extracted, extracted_visual


# --- Mid-band QIM Robust Embedding/Extraction ---

# Generate mid-band mask for 8x8 block (exclude DC and high/low extremes)
def mid_band_mask(block_size=8):
    return [(u, v) for u in range(block_size) for v in range(block_size)
            if 2 <= u+v <= 6 and not (u == 0 and v == 0)]

# QIM quantizer function
def qim_quantize(value, delta, bit):
    # bit=0: even multiple of delta, bit=1: odd multiple
    q = np.round(value / delta)
    if bit == 0:
        return delta * (2 * np.round(q/2))
    else:
        return delta * (2 * np.round((q-1)/2) + 1)

# Robust QIM Embedding
def embed_watermark_robust(cover, watermark_bits, delta=20, k=2, key=12345):
    """
    cover: grayscale or color image as numpy array
    watermark_bits: 2D numpy array of 0s and 1s
    delta: quantization step size
    k: number of coefficients per block to embed
    key: secret seed for pseudo-random selection
    """
    if cover.ndim == 3:
        cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)

    blk = 8
    expected_shape = (cover.shape[0] // blk, cover.shape[1] // blk)
    if watermark_bits.shape != expected_shape:
        watermark_bits = cv2.resize(watermark_bits.astype(np.uint8), expected_shape, interpolation=cv2.INTER_NEAREST)
        _, watermark_bits = cv2.threshold(watermark_bits, 0, 1, cv2.THRESH_BINARY)

    mask = mid_band_mask(blk)
    watermarked = np.copy(cover).astype(np.float32)
    bindex = 0

    for i in range(expected_shape[0]):
        for j in range(expected_shape[1]):
            block = cover[i*blk:(i+1)*blk, j*blk:(j+1)*blk]
            D = apply_dct(block)

            random.seed(key + bindex)
            sel = mask.copy()
            random.shuffle(sel)

            bit = int(watermark_bits[i, j])
            for idx in range(k):
                u, v = sel[idx]
                D[u, v] = qim_quantize(D[u, v], delta, bit)

            watermarked[i*blk:(i+1)*blk, j*blk:(j+1)*blk] = apply_idct(D)
            bindex += 1

    return np.clip(watermarked, 0, 255).astype(np.uint8), watermark_bits

# Robust QIM Extraction
def extract_watermark_robust(watermarked, shape, delta=20, k=2, key=12345):
    if watermarked.ndim == 3:
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    blk = 8
    mask = mid_band_mask(blk)
    wm_est = np.zeros((shape[0]//blk, shape[1]//blk), dtype=np.uint8)
    bindex = 0
    for i in range(shape[0] // blk):
        for j in range(shape[1] // blk):
            block = watermarked[i*blk:(i+1)*blk, j*blk:(j+1)*blk]
            D = apply_dct(block)
            random.seed(key + bindex)
            sel = mask.copy()
            random.shuffle(sel)
            # majority vote across k coefficients
            votes = []
            for idx in range(k):
                u, v = sel[idx]
                q = D[u, v] / delta
                votes.append(int(np.round(q) % 2))
            wm_est[i, j] = 1 if sum(votes) >= (k/2) else 0
            bindex += 1

    visual = cv2.resize(wm_est * 255, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return wm_est, visual

# --- Comparison ---
def compare_watermarks(original, extracted):
    total_bits = original.size
    matching_bits = np.sum(original == extracted)
    return (matching_bits / total_bits) * 100
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
