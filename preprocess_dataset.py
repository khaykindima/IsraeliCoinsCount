# preprocess_dataset.py
import argparse
import logging
from pathlib import Path
import shutil
import cv2 # For saving images

# Project-specific modules
import config # To read INPUTS_DIR and output structure
from utils import (
    setup_logging,
    discover_and_pair_image_labels,
    convert_to_3channel_grayscale 
)

def main():
    # Convert string paths from config to Path objects for the rest of the script
    config.INPUTS_DIR = Path(config.INPUTS_DIR)
    
    parser = argparse.ArgumentParser(description="Preprocess dataset images. Optionally converts to 3-channel grayscale if enabled in config.")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None, 
        help="Base directory where the processed dataset will be saved. If not provided, it defaults based on config.INPUTS_DIR and grayscale setting."
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="If set, overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    preliminary_logger = logging.getLogger('preprocessor_prelim')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    preliminary_logger.addHandler(stream_handler)
    preliminary_logger.setLevel(logging.INFO)

    # Determine output_base_path
    output_suffix = "_grayscale" if config.ENABLE_GRAYSCALE_PREPROCESSING else "_processed_color" # Suffix based on config
    
    if args.output_base_dir:
        output_base_path = Path(args.output_base_dir)
    else:
        if not hasattr(config, 'INPUTS_DIR') or not isinstance(config.INPUTS_DIR, Path) or not config.INPUTS_DIR.name:
            preliminary_logger.error("config.INPUTS_DIR is not properly defined. Cannot determine default output directory.")
            return
        # Append suffix to the original dataset name
        output_base_path = config.INPUTS_DIR.parent / (config.INPUTS_DIR.name + output_suffix)
        preliminary_logger.info(f"No --output_base_dir provided. Defaulting to: {output_base_path}")

    script_log_dir = Path("logs") 
    script_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = script_log_dir / "preprocess_dataset.log"
    logger = setup_logging(log_file_path, logger_name='dataset_preprocessor_logger')
    
    logger.info(f"--- Starting Dataset Preprocessing (Grayscale enabled: {config.ENABLE_GRAYSCALE_PREPROCESSING}) ---")
    logger.info(f"Original dataset configured in INPUTS_DIR: {config.INPUTS_DIR}")
    logger.info(f"Processed dataset will be saved to: {output_base_path}")

    if output_base_path.exists() and args.force_overwrite:
        logger.warning(f"Output directory {output_base_path} exists and --force_overwrite is set. Removing existing directory.")
        try:
            shutil.rmtree(output_base_path)
        except OSError as e:
            logger.error(f"Error removing directory {output_base_path}: {e}")
            return
    elif output_base_path.exists() and not args.force_overwrite:
        logger.error(f"Output directory {output_base_path} already exists. Use --force_overwrite or remove manually. Exiting.")
        return

    try:
        output_base_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directory {output_base_path}: {e}")
        return

    image_label_pairs, _ = discover_and_pair_image_labels(
        config.INPUTS_DIR,
        config.IMAGE_SUBDIR_BASENAME,
        config.LABEL_SUBDIR_BASENAME,
        logger
    )

    if not image_label_pairs:
        logger.error(f"No image-label pairs found in the original dataset: {config.INPUTS_DIR}.")
        return

    logger.info(f"Found {len(image_label_pairs)} image-label pairs to process.")
    processed_count = 0
    failed_count = 0

    for img_path, lbl_path in image_label_pairs:
        try:
            logger.debug(f"Processing image: {img_path}")
            
            # The convert_to_3channel_grayscale function now handles the config flag internally
            processed_img_np = convert_to_3channel_grayscale(img_path, logger_instance=logger)

            if processed_img_np is None:
                logger.warning(f"Failed to process image {img_path}. Skipping.")
                failed_count += 1
                continue

            relative_img_path = img_path.relative_to(config.INPUTS_DIR)
            relative_lbl_path = lbl_path.relative_to(config.INPUTS_DIR)

            new_img_path = output_base_path / relative_img_path
            new_lbl_path = output_base_path / relative_lbl_path

            new_img_path.parent.mkdir(parents=True, exist_ok=True)
            new_lbl_path.parent.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(new_img_path), processed_img_np):
                 logger.warning(f"Failed to save processed image to {new_img_path}. Skipping.")
                 failed_count +=1
                 continue
            
            shutil.copy2(lbl_path, new_lbl_path)
            logger.debug(f"Saved processed image to {new_img_path} and copied label to {new_lbl_path}")
            processed_count += 1

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {img_path}: {e}", exc_info=True)
            failed_count += 1

    logger.info(f"--- Dataset Preprocessing Complete (Grayscale enabled: {config.ENABLE_GRAYSCALE_PREPROCESSING}) ---")
    logger.info(f"Successfully processed and saved: {processed_count} images.")
    logger.info(f"Failed to process: {failed_count} images.")
    logger.info(f"Processed dataset is available at: {output_base_path}")
    if config.ENABLE_GRAYSCALE_PREPROCESSING:
        logger.info(f"Remember to update INPUTS_DIR in config.py to '{output_base_path}' before training with grayscale images.")
    else:
        logger.info(f"Remember to update INPUTS_DIR in config.py to '{output_base_path}' if you intend to train with these processed color images (though they should be identical to originals if grayscale was off).")


if __name__ == '__main__':
    main()
