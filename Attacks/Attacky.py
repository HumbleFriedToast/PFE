import numpy as np
import cv2
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import argparse
import logging
from typing import List, Dict, Tuple
import glob
from pathlib import Path

# Import functions from the existing dwt.py module
from Script.DWT import (
    load_images, preprocess_images, embed_watermark, 
    extract_watermark, calculate_metrics
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_image_files(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a folder
    
    Args:
        folder_path: Path to the folder containing images
        extensions: List of image extensions to look for
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern_upper = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=False))
    
    return sorted(image_files)

def calculate_additional_metrics(original_image: np.ndarray, watermarked_image: np.ndarray) -> Dict:
    """
    Calculate additional image quality metrics
    
    Args:
        original_image: Original cover image
        watermarked_image: Watermarked image
    
    Returns:
        Dictionary containing MSE and SSIM values
    """
    # Ensure images are in the same data type and range for accurate comparison
    original_float = original_image.astype(np.float64)
    watermarked_float = watermarked_image.astype(np.float64)
    
    # Calculate MSE
    mse_value = float(mse(original_float, watermarked_float))
    
    # Calculate SSIM
    ssim_value = float(ssim(original_float, watermarked_float, data_range=255.0))
    
    return {
        'mse': mse_value,
        'ssim': ssim_value
    }

def calculate_watermark_correlation(original_wm: np.ndarray, recovered_wm: np.ndarray) -> Dict:
    """
    Calculate various correlation metrics between original and recovered watermarks
    
    Args:
        original_wm: Original watermark
        recovered_wm: Recovered watermark
    
    Returns:
        Dictionary containing correlation metrics
    """
    # Flatten watermarks for correlation calculation
    orig_flat = original_wm.flatten()
    rec_flat = recovered_wm.flatten()
    
    # Pearson correlation coefficient
    corr_matrix = np.corrcoef(orig_flat, rec_flat)
    pearson_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    
    # Normalized Cross Correlation (NCC)
    sum_orig_sq = np.sum(orig_flat**2)
    sum_rec_sq = np.sum(rec_flat**2)
    
    if sum_orig_sq == 0 or sum_rec_sq == 0:
        ncc = 0.0
    else:
        ncc_val = np.sum(orig_flat * rec_flat) / (np.sqrt(sum_orig_sq) * np.sqrt(sum_rec_sq))
        ncc = float(ncc_val) if not np.isnan(ncc_val) else 0.0
    
    # Bit Error Rate (BER)
    bit_errors = int(np.sum(orig_flat != rec_flat))
    total_bits = int(len(orig_flat))
    ber = float(bit_errors / total_bits) if total_bits > 0 else 0.0
    
    # Accuracy (1 - BER)
    accuracy = 1.0 - ber
    
    return {
        'watermark_pearson_correlation': pearson_corr,
        'watermark_ncc': ncc,
        'watermark_ber': ber,
        'watermark_accuracy': accuracy,
        'watermark_bit_errors': bit_errors,
        'watermark_total_bits': total_bits
    }

def process_single_image(cover_path: str, watermark_path: str, 
                        wavelet: str = 'sym4', level: int = 1, 
                        strength: float = 2.0, key: int = 42,
                        threshold_factor: float = 1.5, 
                        max_watermark_size: int = None) -> Dict:
    """
    Process a single image and calculate all metrics
    
    Args:
        cover_path: Path to cover image
        watermark_path: Path to watermark image
        wavelet: Wavelet type for DWT
        level: Decomposition level
        strength: Watermarking strength
        key: Random seed for embedding
        threshold_factor: Threshold factor for extraction
        max_watermark_size: Maximum watermark size (width*height), None for no limit
    
    Returns:
        Dictionary containing all calculated metrics
    """
    try:
        # Load images
        cover_image, watermark = load_images(cover_path, watermark_path)
        if cover_image is None or watermark is None:
            logger.error(f"Failed to load images: {cover_path}, {watermark_path}")
            return None
        
        # Preprocess images
        cover_float, binary_watermark = preprocess_images(cover_image, watermark, max_watermark_size)
        
        # Embed watermark
        watermarked_image = embed_watermark(
            cover_float, binary_watermark, 
            wavelet=wavelet, level=level, 
            strength=strength, key=key
        )
        
        # Extract watermark
        recovered_watermark = extract_watermark(
            watermarked_image, binary_watermark.shape,
            wavelet=wavelet, level=level, key=key,
            threshold_factor=threshold_factor
        )
        
        # Calculate image quality metrics
        psnr_value = float(psnr(cover_image, watermarked_image))
        additional_metrics = calculate_additional_metrics(cover_image, watermarked_image)
        
        # Calculate watermark correlation metrics
        watermark_metrics = calculate_watermark_correlation(binary_watermark, recovered_watermark)
        
        # Combine all metrics
        results = {
            'cover_image': os.path.basename(cover_path),
            'watermark_image': os.path.basename(watermark_path),
            'cover_shape': f"{cover_image.shape[0]}x{cover_image.shape[1]}",
            'watermark_shape': f"{binary_watermark.shape[0]}x{binary_watermark.shape[1]}",
            'psnr': psnr_value,
            'mse': additional_metrics['mse'],
            'ssim': additional_metrics['ssim'],
            **watermark_metrics,
            'wavelet': wavelet,
            'level': level,
            'strength': strength,
            'key': key,
            'threshold_factor': threshold_factor,
            'max_watermark_size': max_watermark_size,
            'status': 'success'
        }
        
        logger.info(f"Successfully processed {os.path.basename(cover_path)} - "
                   f"PSNR: {psnr_value:.2f}, SSIM: {additional_metrics['ssim']:.4f}, "
                   f"Watermark Correlation: {watermark_metrics['watermark_pearson_correlation']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {cover_path}: {str(e)}", exc_info=True)
        return {
            'cover_image': os.path.basename(cover_path),
            'watermark_image': os.path.basename(watermark_path) if watermark_path else 'N/A',
            'status': 'error',
            'error_message': str(e)
        }

def process_image_folder(cover_folder: str, watermark_path: str, output_csv: str,
                        wavelet: str = 'sym4', level: int = 1, 
                        strength: float = 2.0, key: int = 42,
                        threshold_factor: float = 1.5, max_watermark_size: int = None):
    """
    Process all images in a folder and save results to CSV
    
    Args:
        cover_folder: Path to folder containing cover images
        watermark_path: Path to single watermark image
        output_csv: Path to output CSV file
        wavelet: Wavelet type for DWT
        level: Decomposition level
        strength: Watermarking strength
        key: Random seed for embedding
        threshold_factor: Threshold factor for extraction
        max_watermark_size: Maximum watermark size (width*height), None for no limit
    """
    logger.info(f"Starting batch processing of images in: {cover_folder}")
    logger.info(f"Using watermark: {watermark_path}")
    
    # Get all image files
    image_files = get_image_files(cover_folder)
    
    if not image_files:
        logger.error(f"No image files found in {cover_folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, cover_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(cover_path)}")
        
        result = process_single_image(
            cover_path, watermark_path,
            wavelet=wavelet, level=level, 
            strength=strength, key=key,
            threshold_factor=threshold_factor,
            max_watermark_size=max_watermark_size
        )
        
        if result:
            results.append(result)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to: {output_csv}")
        
        # Print summary statistics
        successful_results = df[df['status'] == 'success']
        if len(successful_results) > 0:
            logger.info("\n" + "="*50)
            logger.info("SUMMARY STATISTICS")
            logger.info("="*50)
            logger.info(f"Total images processed: {len(results)}")
            logger.info(f"Successful: {len(successful_results)}")
            logger.info(f"Failed: {len(results) - len(successful_results)}")
            
            if len(successful_results) > 1:
                logger.info(f"\nImage Quality Metrics:")
                logger.info(f"  PSNR - Mean: {successful_results['psnr'].mean():.2f}, "
                           f"Std: {successful_results['psnr'].std():.2f}")
                logger.info(f"  MSE  - Mean: {successful_results['mse'].mean():.2f}, "
                           f"Std: {successful_results['mse'].std():.2f}")
                logger.info(f"  SSIM - Mean: {successful_results['ssim'].mean():.4f}, "
                           f"Std: {successful_results['ssim'].std():.4f}")
                
                logger.info(f"\nWatermark Quality Metrics:")
                logger.info(f"  Correlation - Mean: {successful_results['watermark_pearson_correlation'].mean():.4f}, "
                           f"Std: {successful_results['watermark_pearson_correlation'].std():.4f}")
                logger.info(f"  NCC - Mean: {successful_results['watermark_ncc'].mean():.4f}, "
                           f"Std: {successful_results['watermark_ncc'].std():.4f}")
                logger.info(f"  BER - Mean: {successful_results['watermark_ber'].mean():.4f}, "
                           f"Std: {successful_results['watermark_ber'].std():.4f}")
                logger.info(f"  Accuracy - Mean: {successful_results['watermark_accuracy'].mean():.4f}, "
                           f"Std: {successful_results['watermark_accuracy'].std():.4f}")
    else:
        logger.error("No results to save")

def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(description='Batch process images for watermarking metrics')
    
    parser.add_argument('--cover_folder', '-c', required=True,
                       help='Path to folder containing cover images')
    parser.add_argument('--watermark', '-w', required=True,
                       help='Path to watermark image')
    parser.add_argument('--output', '-o', default='watermark_metrics.csv',
                       help='Output CSV file path (default: watermark_metrics.csv)')
    parser.add_argument('--wavelet', default='sym4',
                       help='Wavelet type for DWT (default: sym4)')
    parser.add_argument('--level', type=int, default=1,
                       help='DWT decomposition level (default: 1)')
    parser.add_argument('--strength', type=float, default=2.0,
                       help='Watermarking strength (default: 2.0)')
    parser.add_argument('--key', type=int, default=42,
                       help='Random seed for embedding (default: 42)')
    parser.add_argument('--threshold_factor', type=float, default=1.5,
                       help='Threshold factor for extraction (default: 1.5)')
    parser.add_argument('--max_watermark_size', type=int, default=None,
                       help='Maximum watermark size (width*height), None for no limit (default: None)')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.cover_folder):
        logger.error(f"Cover folder does not exist: {args.cover_folder}")
        return
    
    if not os.path.exists(args.watermark):
        logger.error(f"Watermark image does not exist: {args.watermark}")
        return
    
    # Process images
    process_image_folder(
        cover_folder=args.cover_folder,
        watermark_path=args.watermark,
        output_csv=args.output,
        wavelet=args.wavelet,
        level=args.level,
        strength=args.strength,
        key=args.key,
        threshold_factor=args.threshold_factor,
        max_watermark_size=args.max_watermark_size
    )

if __name__ == "__main__":
    main()