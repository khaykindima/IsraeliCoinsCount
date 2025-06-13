import argparse
from pathlib import Path
import cv2
import logging
import pandas as pd
import shutil # For copying log file if needed, though now logger writes directly
from collections import Counter

try:
    import config
    from utils import (
        setup_logging, load_class_names_from_yaml, 
        create_detector_from_config,
        validate_config_and_paths,
        save_config_to_run_dir 
    )
except ImportError as e:
    print(f"ImportError: {e}. Make sure config.py, utils.py, and detector.py are in the same directory or PYTHONPATH")
    exit()

# Ideally, this function would be in utils.py
def _create_unique_inference_run_dir(base_output_dir, model_name_prefix, logger_for_util):
    """
    Creates a unique directory for an inference run.
    Format: base_output_dir/inference_runs/MODEL_NAME_PREFIX_inference_runXX/
    """
    inference_runs_base = base_output_dir / "inference_runs"
    inference_runs_base.mkdir(parents=True, exist_ok=True)
    
    run_idx = 1
    while True:
        # Sanitize model_name_prefix by replacing characters not suitable for directory names
        safe_model_name_prefix = model_name_prefix.replace('.', '_')
        run_dir_name = f"{safe_model_name_prefix}_inference_run{run_idx}"
        current_run_dir = inference_runs_base / run_dir_name
        if not current_run_dir.exists():
            try:
                current_run_dir.mkdir(parents=True)
                if logger_for_util: # Check if logger is available
                    logger_for_util.info(f"Created unique inference run directory: {current_run_dir}")
                else: # Fallback print if logger not ready
                    print(f"Created unique inference run directory: {current_run_dir}")
                return current_run_dir
            except OSError as e:
                if logger_for_util:
                    logger_for_util.error(f"Failed to create directory {current_run_dir}: {e}")
                else:
                    print(f"ERROR: Failed to create directory {current_run_dir}: {e}")
                # Fallback: use a generic name in inference_runs_base to avoid crash
                fallback_dir = inference_runs_base / f"inference_run_fallback_{run_idx}"
                fallback_dir.mkdir(parents=True, exist_ok=True) # Should succeed
                return fallback_dir

        run_idx += 1
        if run_idx > 1000: # Safety break
            if logger_for_util:
                logger_for_util.error("Exceeded 1000 attempts to create a unique run directory. Using fallback.")
            else:
                print("ERROR: Exceeded 1000 attempts to create a unique run directory. Using fallback.")
            # Create a fallback directory with a timestamp to ensure uniqueness
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fallback_dir = inference_runs_base / f"{safe_model_name_prefix}_inference_run_fallback_{timestamp}"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir


class InferenceRunner:
    """
    Encapsulates the logic for running inference using a CoinDetector.
    """
    def __init__(self, detector, logger, current_run_dir, config_module=config):
        self.detector = detector
        self.logger = logger
        self.config = config_module
        self.current_run_dir = current_run_dir

    def _get_images_to_process(self, input_source_path):
        """Determines the list of image file paths to process."""
        images_to_process = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png']

        if not input_source_path:
            self.logger.info(f"No input path provided. Defaulting to config.INPUTS_DIR: {self.config.INPUTS_DIR}")
            input_path = self.config.INPUTS_DIR
        else:
            input_path = Path(input_source_path)

        if not input_path.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            return []

        if input_path.is_file():
            if input_path.suffix.lower() in [ext.replace('*','') for ext in image_extensions]:
                images_to_process.append(input_path)
        elif input_path.is_dir():
            for ext in image_extensions:
                images_to_process.extend(list(input_path.rglob(ext))) # rglob for recursive search

        return images_to_process

    def _calculate_and_log_summary(self, predictions, image_name):
        """Calculates and logs the coin counts and total sum for an image."""
        if not predictions:
            self.logger.info(f"Summary for {image_name}: No coins detected.")
            return

        # Count occurrences of each coin type
        coin_counts = Counter(p['class_name'].lower().strip() for p in predictions)

        # Calculate the total monetary value and prepare strings for logging
        total_sum = 0
        count_strings = []
        
        # Sort the detected coins by their monetary value in ascending order
        # The key for sorting is the value from the COIN_VALUES map in config.py
        # x[0] is the coin_name (e.g., 'one') from the (key, value) tuple in coin_counts.items()
        sorted_coin_counts = sorted(coin_counts.items(), key=lambda x: self.config.COIN_VALUES.get(x[0], 0))

        for coin_name, count in sorted_coin_counts:
            coin_value = self.config.COIN_VALUES.get(coin_name, 0)
            total_sum += count * coin_value
            count_strings.append(f"{count}x {coin_name.capitalize()}")
        
        detection_summary = ", ".join(count_strings)

        self.logger.info(f"--- Summary for {image_name} ---")
        self.logger.info(f"Detections: {detection_summary}")
        self.logger.info(f"Total Sum: {total_sum} Shekels")
        self.logger.info("-" * (20 + len(image_name)))

    def run_on_source(self, input_path=None, save_annotated_images=True):
        """
        Runs inference on a source, saves annotated images, and returns prediction data.
        
        Args:
            input_path (str, optional): Path to an image or a directory of images.
            save_annotated_images (bool): If True, saves images with predictions drawn on them.
            
        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  prediction data for a single detected object.
        """
		
        self.logger.info("--- Starting Inference Run ---")
		
        images_to_process = self._get_images_to_process(input_path)
        if not images_to_process:
            self.logger.warning("No images found to process.")
            return []

        self.logger.info(f"Found {len(images_to_process)} image(s) to process.")
        
        # This list will store all prediction data for the Excel export
        all_predictions_data = []

        # Setup output directory for annotated images
        annotated_output_dir = self.current_run_dir / "inference_annotated_images"
        if save_annotated_images:
            annotated_output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Annotated images will be saved to: {annotated_output_dir}")

        for image_path in images_to_process:
            self.logger.info(f"---> Processing image: {image_path.name}")
            image_np = cv2.imread(str(image_path))
            if image_np is None:
                self.logger.warning(f"Could not read image {image_path}, skipping.")
                continue

            # Get predictions
            predictions = self.detector.predict(image_np, return_raw=False)

            # Calculate and log the summary of counts and total value
            self._calculate_and_log_summary(predictions, image_path.name)

            # Collect data for optional Excel export
            if not predictions:
                all_predictions_data.append({
                    'image_name': image_path.name,
                    'class_name': 'No Detections',
                    'probability': None,
                    'bbox_xyxy': None
                })
            else:
                for pred in predictions:
                    all_predictions_data.append({
                        'image_name': image_path.name,
                        'class_name': pred.get('class_name', 'N/A'),
                        'probability': pred.get('conf', 0.0),
                        'bbox_xyxy': str(pred.get('xyxy', 'N/A'))
                    })
            
            # Save the annotated image if requested
            if save_annotated_images:
                annotated_image = self.detector.draw_predictions_on_image(image_np, predictions)
                save_path = annotated_output_dir / image_path.name
                cv2.imwrite(str(save_path), annotated_image)
        
        return all_predictions_data


def setup_inference():
    """
    Sets up a unique run directory, logger, and the detector.
    Returns:
        tuple: (logger, detector, current_run_dir) or (None, None, None) on failure.
    """
    # Initial basic logger for directory creation, if needed before full logger setup
    temp_logger = logging.getLogger('inference_setup')
    temp_logger.addHandler(logging.StreamHandler()) # Log to console initially
    temp_logger.setLevel(logging.INFO)

    # Determine model name prefix for the run directory
    # Ensure MODEL_PATH_FOR_PREDICTION is a Path object
    model_path = Path(config.MODEL_PATH_FOR_PREDICTION)
    model_name_prefix = model_path.stem # e.g., "yolov8n_best_direct"

    # Create the unique run directory for this inference session
    # Pass temp_logger for initial messages from _create_unique_inference_run_dir
    current_run_dir = _create_unique_inference_run_dir(config.OUTPUT_DIR, model_name_prefix, temp_logger)
    if not current_run_dir: # Should not happen with fallback in _create_unique_inference_run_dir
        temp_logger.error("CRITICAL: Failed to create a unique run directory. Exiting.")
        return None, None, None

    # Now, set up the main logger to write into the unique run directory
    log_file_name = getattr(config, 'LOG_FILE_BASE_NAME', 'yolo_log') + "_inference.log"
    log_file_path = current_run_dir / log_file_name
    logger = setup_logging(log_file_path, logger_name='yolo_inference_logger')
    
    # Replace temp_logger usage with the fully configured logger
    logger.info(f"Switched to main logger. Log file: {log_file_path}")
	
    save_config_to_run_dir(current_run_dir, logger)


    if not validate_config_and_paths(config, 'inference', logger):
        return logger, None, current_run_dir # Return dir even on validation fail for logs

    logger.info(f"--- Setting up the detector for inference in run directory: {current_run_dir} ---")
    class_names_yaml_path = config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME
    names_from_yaml = load_class_names_from_yaml(class_names_yaml_path, logger)
    if names_from_yaml is None:
        logger.error(f"CRITICAL: Could not load class names from '{class_names_yaml_path}'.")
        return logger, None, current_run_dir
    class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}

    try:
        detector = create_detector_from_config(
            model_path, class_names_map, config, logger # Use model_path (Path object)
        )
        return logger, detector, current_run_dir
    except Exception as e:
        logger.exception(f"A critical error occurred during detector setup: {e}")
        return logger, None, current_run_dir

def main():
    """
    Main function to set up and run the inference process.
    """
    # Convert string paths from config to absolute Path objects for the rest of the script
    config.INPUTS_DIR = Path(config.INPUTS_DIR).resolve()
    config.OUTPUT_DIR = Path(config.OUTPUT_DIR).resolve()
    
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model.")
    parser.add_argument(
        "input_path", 
        type=str, 
        nargs='?', 
        default=None, 
        help="Path to an image file or a folder of images. If not provided, uses INPUTS_DIR from config.py."
    )
    # Added argument for Excel export 
    parser.add_argument(
        "--export_excel",
        action="store_true", # This makes it a flag, e.g., `python run_inference.py --export_excel`
        help="If set, exports the detailed prediction results to an Excel file."
    )
    parser.add_argument(
        "--no_save_annotated",
        action="store_false", # Default is True (save), this flag makes it False
        dest="save_annotated_images",
        help="If set, disables saving of annotated images."
    )
    parser.set_defaults(save_annotated_images=True)

    args = parser.parse_args()

    logger, detector, current_run_dir = setup_inference()

    if not detector:
        if logger: 
            logger.error(f"Exiting due to setup failure. Check logs in {current_run_dir if current_run_dir else config.OUTPUT_DIR}.")
        else: # Should not happen if setup_inference returns a logger
            print(f"CRITICAL ERROR: Detector setup failed. No logger available. Check console for earlier messages.")
        return

    runner = InferenceRunner(detector, logger, current_run_dir)
    # Pass the flag for saving annotated images
    collected_predictions = runner.run_on_source(
        input_path=args.input_path,
        save_annotated_images=args.save_annotated_images,
    )

    if args.export_excel and collected_predictions:
        try:
            # The predictions are already collected by the runner
            df = pd.DataFrame(collected_predictions)
            excel_path = current_run_dir / config.PREDICTIONS_CSV_NAME.replace('.csv', '.xlsx')
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logger.info(f"Successfully exported inference results to {excel_path}")
        except Exception as e:
            logger.error(f"Failed to export results to Excel: {e}")

    logger.info(f"--- Inference run finished. Outputs are in: {current_run_dir} ---")


if __name__ == '__main__':
    main()
