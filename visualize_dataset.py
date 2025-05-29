import argparse
import logging
import random
from pathlib import Path
import cv2
import numpy as np

# Project-specific modules
import config
from utils import (
    setup_logging,
    load_class_names_from_yaml,
    discover_and_pair_image_labels,
    parse_yolo_annotations,
    draw_ground_truth_boxes
)

def main():
    """
    Main function to run the ground truth visualization process.
    """
    parser = argparse.ArgumentParser(description="Visualize ground truth labels on images from the dataset.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=None, # MODIFIED: Default is now None to signal 'all images'
        help="Number of random images to generate. If not specified, all images in the input folder will be processed."
    )
    args = parser.parse_args()

    # Create a dedicated output directory for these visualizations
    output_dir = config.OUTPUT_DIR / "ground_truth_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file_path = output_dir / "visualization_log.log"
    logger = setup_logging(log_file_path, logger_name='gt_visualizer_logger')
    logger.info("--- Starting Ground Truth Visualization ---")
    logger.info(f"Output will be saved to: {output_dir}")

    # --- Step 1: Load Class Names ---
    class_names_yaml_path = config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME
    names_from_yaml = load_class_names_from_yaml(class_names_yaml_path, logger)
    if names_from_yaml is None:
        logger.error(f"CRITICAL: Could not load class names from '{class_names_yaml_path}'. Exiting.")
        return
    class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}
    logger.info(f"Loaded {len(class_names_map)} class names: {class_names_map}")

    # --- Step 2: Discover all image-label pairs ---
    image_label_pairs, _ = discover_and_pair_image_labels(
        config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
    )
    if not image_label_pairs:
        logger.error("No image-label pairs were found. Please check your INPUTS_DIR configuration.")
        return
    logger.info(f"Found a total of {len(image_label_pairs)} image-label pairs.")

    # --- Step 3: Select images to process (all by default) ---
    if args.num_images is None:
        # If no number is specified, process all images
        pairs_to_process = image_label_pairs
        logger.info(f"Processing all {len(pairs_to_process)} images...")
    else:
        # If a number is specified, process a random sample
        num_to_sample = min(args.num_images, len(image_label_pairs))
        logger.info(f"Randomly sampling {num_to_sample} images for visualization...")
        pairs_to_process = random.sample(image_label_pairs, num_to_sample)
    
    processed_count = 0
    for img_path, lbl_path in pairs_to_process:
        logger.debug(f"Processing: {img_path.name}")
        
        # Read the image
        image_np = cv2.imread(str(img_path))
        if image_np is None:
            logger.warning(f"Could not read image {img_path}, skipping.")
            continue
        
        h, w, _ = image_np.shape
        
        # Parse the corresponding YOLO annotation file
        annotations = parse_yolo_annotations(lbl_path, logger)
        if not annotations:
            logger.warning(f"No annotations found in {lbl_path}. A blank image will be saved.")
        
        # Convert YOLO format to the format needed by our drawing function
        ground_truths = []
        for cid, cx, cy, cw, ch in annotations:
            # Denormalize coordinates to get pixel values
            x1 = (cx - cw / 2) * w
            y1 = (cy - ch / 2) * h
            x2 = (cx + cw / 2) * w
            y2 = (cy + ch / 2) * h
            ground_truths.append({'cls': cid, 'xyxy': [x1, y1, x2, y2]})

        # Draw the ground truth boxes on the image
        annotated_image = draw_ground_truth_boxes(image_np, ground_truths, class_names_map, config)

        # Save the final annotated image
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), annotated_image)
        processed_count += 1

    logger.info(f"--- Visualization complete. {processed_count} images saved in {output_dir} ---")

if __name__ == '__main__':
    main()