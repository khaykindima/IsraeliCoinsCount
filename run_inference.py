import argparse # For command-line arguments
from pathlib import Path
import cv2
import logging

try:
    import config
    # --- REFACTORED: Import the new factory function ---
    from utils import (
        setup_logging, load_class_names_from_yaml, 
        discover_and_pair_image_labels, create_detector_from_config
    )
except ImportError as e:
    print(f"ImportError: {e}. Make sure config.py, utils.py, and detector.py are in the same directory or PYTHONPATH")
    exit()

# --- Main Inference Script ---
def main_inference(input_source_path=None): # Modified to accept a path
    """
    Runs inference on a single image or all images in a folder,
    or all images in config.INPUTS_DIR if no path is provided.
    Args:
        input_source_path (str, optional): Path to an image file or a folder containing images.
                                         If None, uses config.INPUTS_DIR.
    """
    # --- Setup ---
    # Create a simple logger for this script
    log_file = Path(config.OUTPUT_DIR) / f"{config.LOG_FILE_BASE_NAME}_inference.log"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file, logger_name='yolo_inference_logger')

    logger.info("--- Starting Inference Script ---")
    
    # --- Determine images to process ---
    images_to_process = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png'] # Common image extensions

    if input_source_path:
        provided_path = Path(input_source_path)
        if provided_path.is_file():
            # Check if it's a supported image type by extension
            if provided_path.suffix.lower() in [ext.replace('*','') for ext in image_extensions]:
                images_to_process.append(provided_path)
                logger.info(f"Processing single image file: {provided_path}")
            else:
                logger.error(f"Provided file is not a supported image type (jpg, jpeg, png): {provided_path}")
                return
        elif provided_path.is_dir():
            logger.info(f"Processing images from folder: {provided_path}")
            for ext in image_extensions:
                images_to_process.extend(list(provided_path.glob(ext)))
            if not images_to_process:
                logger.warning(f"No images found in the provided folder: {provided_path}")
                return
        else:
            logger.error(f"Provided input path is not a valid file or folder: {provided_path}")
            return
    else:
        logger.info(f"No input path provided. Defaulting to config.INPUTS_DIR: {config.INPUTS_DIR}")
        if not config.INPUTS_DIR.exists():
            logger.error(f"Default INPUTS_DIR does not exist: {config.INPUTS_DIR}")
            return
        # Use discover_and_pair_image_labels to get images from the structured dataset
        image_label_pairs, _ = discover_and_pair_image_labels(
            config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
        )
        if not image_label_pairs:
            logger.warning(f"No images found in the default INPUTS_DIR structure: {config.INPUTS_DIR}")
            # As a further fallback, try a simple glob in INPUTS_DIR if the structured search fails
            logger.info(f"Attempting simple recursive image search in {config.INPUTS_DIR} as fallback.")
            for ext in image_extensions:
                images_to_process.extend(list(config.INPUTS_DIR.rglob(ext))) # Recursive glob
            if not images_to_process:
                 logger.warning(f"Still no images found after recursive search in {config.INPUTS_DIR}")
                 return
        else:
            images_to_process = [img_path for img_path, _ in image_label_pairs]


    if not images_to_process:
        logger.info("No images to process.")
        return
    
    logger.info(f"Found {len(images_to_process)} image(s) to process.")

    # --- Load Class Names ---
    # Class names MUST now come from the specified YAML file for inference.
    class_names_yaml_path = config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME
    names_from_yaml = load_class_names_from_yaml(class_names_yaml_path, logger)
    
    if names_from_yaml is None:
        logger.error(f"CRITICAL: Could not load class names from '{class_names_yaml_path}'. "
                     f"This file is required for the CoinDetector to map class IDs to names. Exiting.")
        return # Exit if class names cannot be loaded
    else:
        class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}
        logger.info(f"Successfully loaded {len(class_names_map)} class names from {config.ORIGINAL_DATA_YAML_NAME}.")
    
    if not class_names_map and not (names_from_yaml is not None and len(names_from_yaml) == 0): # Allow explicitly empty names list if nc=0
        logger.error("Class names map is empty (and not explicitly defined as empty in YAML). "
                     "This can happen if 'names' list in YAML is missing or malformed. Cannot proceed if classes are expected.")
        return


    # --- Initialize Detector ---
    logger.info(f"Loading model for inference: {config.MODEL_PATH_FOR_PREDICTION}")
    if not config.MODEL_PATH_FOR_PREDICTION.exists():
        logger.error(f"Model file not found: {config.MODEL_PATH_FOR_PREDICTION}. Please check config.py.")
        return
    try:
        # --- REFACTORED: Use the factory to create the detector ---
        coin_detector = create_detector_from_config(
            config.MODEL_PATH_FOR_PREDICTION, class_names_map, config, logger
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CoinDetector with model {config.MODEL_PATH_FOR_PREDICTION}")
        return

    # --- Create output directory for inference results ---
    output_image_dir = config.OUTPUT_DIR / "inference_outputs" / Path(input_source_path).stem if input_source_path and Path(input_source_path).is_dir() else config.OUTPUT_DIR / "inference_outputs"
    if input_source_path and Path(input_source_path).is_file(): # Handle single file case for output dir naming
         output_image_dir = config.OUTPUT_DIR / "inference_outputs" / Path(input_source_path).stem
    
    output_image_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Annotated images will be saved to: {output_image_dir}")

    # --- Perform Prediction on each image ---
    for image_path in images_to_process:
        logger.info(f"Performing prediction on: {image_path}")
        try:
            image_np = cv2.imread(str(image_path))
            if image_np is None:
                logger.error(f"Failed to read image: {image_path}")
                continue

            final_predictions = coin_detector.predict(image_np) # Pass NumPy array

            logger.info(f"Predictions for {image_path.name}:")
            if final_predictions:
                for pred in final_predictions:
                    logger.info(f"  - Class: {pred['class_name']}, Confidence: {pred['conf']:.4f}, Box: {pred['xyxy']}")
            else:
                logger.info("  No objects detected.")

            # --- Visualize (Optional) ---
            image_with_predictions = coin_detector.draw_predictions_on_image(image_np, final_predictions)
            
            output_image_path = output_image_dir / f"{image_path.stem}_inferred{image_path.suffix}"
            cv2.imwrite(str(output_image_path), image_with_predictions)
            logger.info(f"Saved annotated image to: {output_image_path}")
            
        except Exception as e:
            logger.exception(f"An error occurred during inference on {image_path}:")
    
    logger.info("--- Inference Script Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model.")
    parser.add_argument(
        "input_path", 
        type=str, 
        nargs='?', 
        default=None, 
        help="Path to an image file or a folder containing images. If not provided, uses INPUTS_DIR from config.py."
    )
    args = parser.parse_args()

    if not config.MODEL_PATH_FOR_PREDICTION.exists():
        print(f"ERROR: MODEL_PATH_FOR_PREDICTION ('{config.MODEL_PATH_FOR_PREDICTION}') in config.py does not exist. Please update it.")
    else:
        main_inference(args.input_path)