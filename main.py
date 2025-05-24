import argparse
import json
import logging
from pathlib import Path
import os # For os.path.abspath if needed, though Path is preferred
import cv2 # For saving image, though dwt.py handles it

# Import functions from our modules
import dwt # Assuming dwt.py is in the same directory or accessible via PYTHONPATH
import attacks # Assuming attacks.py is in the same directory

# Set up logging (this will be the main logger for the application)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Added %(name)s
)
logger = logging.getLogger(__name__) # Logger for the main script

# Default configuration (can be overridden by a config file)
DEFAULT_CONFIG = {
    "wavelet_type": "sym4",
    "decomposition_level": 2, # User changed this to 2 in dwt.py, let's sync or make it configurable
    "embedding_strength": 2.0,
    "secret_key": 1000,
    "threshold_factor": 1.5,
    "max_watermark_size": 500000,
    "jpeg_quality": 100
    # "batch_size" was in old config, might not be relevant to current dwt.py logic
}

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from a JSON file.
    If config_path is None or file not found, returns DEFAULT_CONFIG.
    Values from the file override defaults.
    """
    config_to_use = DEFAULT_CONFIG.copy()
    if config_path:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    config_to_use.update(user_config) # User config overrides defaults
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config file '{config_path}': {e}. Using default/merged configuration.")
        else:
            logger.warning(f"Config file '{config_path}' not found. Using default/merged configuration.")
    else:
        logger.info("No config file specified. Using default configuration.")
    return config_to_use

def main():
    parser = argparse.ArgumentParser(description='DWT Digital Watermarking Tool')
    parser.add_argument('--cover', type=str, required=True, help='Path to the cover image.')
    parser.add_argument('--watermark', type=str, required=True, help='Path to the watermark image.')
    parser.add_argument('--output', type=str, default='output', help='Directory to save results (default: output).')
    parser.add_argument('--config', type=str, help='Path to a JSON configuration file.')
    parser.add_argument('--attack', type=str, default='none',
                        choices=['gaussian_noise', 'gaussian_blur', 'median_blur', 'compression', 'crop',
                                     'bilateral_blur', 'salt_pepper', 'rotation', 'resize',
                                     'contrast', 'motion_blur', 'sharpening'],
                        help='Type of attack to apply (default: none).')

    args = parser.parse_args()

    # Load configuration from file or use defaults
    config = load_config(args.config)
    logger.info(f"Using configuration: {config}")

    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir.resolve()}")

    # --- Main Watermarking Pipeline ---
    logger.info("Starting watermarking process...")

    # 1. Load images
    logger.info(f"Loading cover image: '{args.cover}', watermark: '{args.watermark}'")
    cover_image, watermark_original = dwt.load_images(args.cover, args.watermark)
    if cover_image is None or watermark_original is None:
        # load_images already logs errors, so we just exit
        return 
    logger.info(f"Cover image loaded: {cover_image.shape}, Watermark loaded: {watermark_original.shape}")

    # 2. Preprocess images
    # Note: config values are accessed directly from the config dict
    cover_image_float, binary_watermark = dwt.preprocess_images(
        cover_image, watermark_original, config.get('max_watermark_size')
    )
    logger.info(f"Binary watermark size after preprocessing: {binary_watermark.shape}")

    # 3. Embed watermark
    logger.info("Embedding watermark...")
    watermarked_image = dwt.embed_watermark(
        cover_image_float,
        binary_watermark,
        wavelet=config.get('wavelet_type', DEFAULT_CONFIG['wavelet_type']),
        level=config.get('decomposition_level', DEFAULT_CONFIG['decomposition_level']),
        strength=config.get('embedding_strength', DEFAULT_CONFIG['embedding_strength']),
        key=config.get('secret_key') # Can be None if not in config
    )
    logger.info("Watermark embedding complete.")

    # 4. Apply attack (if any)
    # The apply_attack function is now imported from the attacks module
    watermarked_image_attacked = attacks.apply_attack(watermarked_image, args.attack)

    # 5. Save watermarked (and potentially attacked) image
    watermarked_image_path = output_dir / "dwt_watermarked_attacked.jpg"
    cv2.imwrite(str(watermarked_image_path), watermarked_image_attacked, 
                [int(cv2.IMWRITE_JPEG_QUALITY), config.get('jpeg_quality', DEFAULT_CONFIG['jpeg_quality'])])
    logger.info(f"Watermarked image (attack: {args.attack}) saved to: {watermarked_image_path}")

    # 6. Extract watermark
    logger.info("Extracting watermark...")
    recovered_watermark = dwt.extract_watermark(
        watermarked_image_attacked,
        binary_watermark.shape, # Original binary watermark shape for reshaping
        wavelet=config.get('wavelet_type', DEFAULT_CONFIG['wavelet_type']),
        level=config.get('decomposition_level', DEFAULT_CONFIG['decomposition_level']),
        key=config.get('secret_key'),
        threshold_factor=config.get('threshold_factor', DEFAULT_CONFIG['threshold_factor'])
    )
    logger.info(f"Recovered watermark size: {recovered_watermark.shape}")

    # 7. Calculate and save metrics
    logger.info("Calculating metrics...")
    metrics = dwt.calculate_metrics(
        cover_image, # Original cover for PSNR against attacked watermarked image
        watermarked_image_attacked,
        binary_watermark, # Original binary watermark for comparison
        recovered_watermark
    )
    metrics_path = output_dir / "metrics.json"
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
    except TypeError as e:
        logger.error(f"Error serializing metrics to JSON: {e}. Metrics data: {metrics}")

    logger.info("--- Metrics ---   ")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name.replace('_', ' ').title()}: {metric_value}")

    # 8. Plot results
    logger.info("Generating and saving result plots...")
    plot_path = output_dir / "watermark_comparison_plot.png"
    dwt.plot_results(
        cover_image,
        watermarked_image_attacked,
        binary_watermark,
        recovered_watermark,
        save_path=str(plot_path)
    )
    logger.info(f"Comparison plot saved to {plot_path}")

    logger.info("Watermarking process finished.")

if __name__ == '__main__':
    main() 