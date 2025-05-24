import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def apply_dct(block):
    """Apply 2D DCT to an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    """Apply 2D inverse DCT to an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def preprocess_watermark(watermark, shape):
    """Read and resize the watermark to match 8x8 block layout of the cover image."""
    watermark = cv2.resize(watermark, (shape[1] // 8, shape[0] // 8))
    _, binary = cv2.threshold(watermark, 128, 1, cv2.THRESH_BINARY) 
    return binary

def embed_watermark(cover, watermark_path, shape, alpha=10):
    """Embed a binary watermark into the cover image using DCT."""
    """Read and preprocess the watermark."""
    watermark_binary = preprocess_watermark(watermark_path, shape)

    """Embed the binary watermark into the DCT coefficients of the cover image."""
    h_blocks = cover.shape[0] // 8
    w_blocks = cover.shape[1] // 8
    watermarked = np.zeros_like(cover, dtype=np.float32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = cover[i*8:(i+1)*8, j*8:(j+1)*8]
            dct_block = apply_dct(block)

            # Embed watermark bit at mid-frequency position
            if watermark_binary[i, j] == 1:
                dct_block[4, 4] += alpha
            else:
                dct_block[4, 4] -= alpha

            idct_block = apply_idct(dct_block)
            watermarked[i*8:(i+1)*8, j*8:(j+1)*8] = idct_block

    return np.clip(watermarked, 0, 255).astype(np.uint8)

def show_images(title1, image1, title2, image2):
    """Display two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show first image
    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')

    # Show second image
    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.show()

def extract_watermark(watermarked, original_shape, alpha=10):
    h_blocks = original_shape[0] // 8
    w_blocks = original_shape[1] // 8
    extracted = np.zeros((h_blocks, w_blocks), dtype=np.uint8)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = watermarked[i*8:(i+1)*8, j*8:(j+1)*8]

            # Ensure the block is grayscale
            if len(block.shape) == 3:  # Color block
                block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)

            dct_block = apply_dct(block)

            # Compare scalar value at mid-frequency
            if dct_block[4, 4] > 0:
                extracted[i, j] = 1
            else:
                extracted[i, j] = 0

    extracted_visual = cv2.resize(extracted * 255, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return extracted, extracted_visual


def compare_watermarks(original, extracted):
    """Compare the original and extracted watermarks and calculate accuracy."""
    total_bits = original.size
    matching_bits = np.sum(original == extracted)
    accuracy = (matching_bits / total_bits) * 100
    return accuracy

def show_comparison_with_accuracy(original_bin, extracted_bin, accuracy):
    """Display the original and extracted watermark with accuracy info."""
    original_visual = cv2.resize(original_bin * 255, (original_bin.shape[1]*8, original_bin.shape[0]*8), interpolation=cv2.INTER_NEAREST)
    extracted_visual = cv2.resize(extracted_bin * 255, (extracted_bin.shape[1]*8, extracted_bin.shape[0]*8), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_visual, cmap='gray')
    axes[0].set_title(f"Original Watermark                             (Accuracy: {accuracy:.2f}%)")
    axes[0].axis('off')

    axes[1].imshow(extracted_visual, cmap='gray')
    axes[1].set_title("Extracted Watermark")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

