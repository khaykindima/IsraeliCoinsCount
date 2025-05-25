import argparse
from pathlib import Path
import cv2
import logging

try:
    import config
    from utils import (
        setup_logging, load_class_names_from_yaml, 
        create_detector_from_config,
        validate_config_and_paths
    )
except ImportError as e:
    print(f"ImportError: {e}. Make sure config.py, utils.py, and detector.py are in the same directory or PYTHONPATH")
    exit()

# --- A dedicated class for handling the inference process ---
class InferenceRunner:
    """
    Encapsulates the logic for running inference using a CoinDetector.
    """
    def __init__(self, detector, logger, config_module=config):
        """
        Initializes the InferenceRunner.
        Args:
            detector (CoinDetector): A pre-configured instance of CoinDetector.
            logger: A logger instance.
            config_module: The configuration module.
        """
        self.detector = detector
        self.logger = logger
        self.config = config_module

    def _get_images_to_process(self, input_source_path):
        """Determines the list of image file paths to process."""
        images_to_process = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png']

        if not input_source_path:
            self.logger.info(f"No input path provided. Defaulting to config.INPUTS_DIR: {self.config.INPUTS_DIR}")
            input_source_path = self.config.INPUTS_DIR

        provided_path = Path(input_source_path)

        if not provided_path.exists():
            self.logger.error(f"Provided input path does not exist: {provided_path}")
            return []

        if provided_path.is_file():
            if provided_path.suffix.lower() in [ext.replace('*', '') for ext in image_extensions]:
                images_to_process.append(provided_path)
            else:
                self.logger.error(f"Provided file is not a supported image type: {provided_path}")
        elif provided_path.is_dir():
            self.logger.info(f"Recursively searching for images in: {provided_path}")
            for ext in image_extensions:
                # --- FIXED: Changed glob to rglob for recursive search ---
                images_to_process.extend(list(provided_path.rglob(ext)))
        
        if not images_to_process:
            self.logger.warning(f"No images found for path: {provided_path}")
            
        return images_to_process

    def run(self, input_source_path=None):
        """
        Runs inference on a source and saves the annotated outputs.
        Args:
            input_source_path (str, optional): Path to a file or directory.
        """
        self.logger.info("--- Starting Inference Run ---")
        images_to_process = self._get_images_to_process(input_source_path)

        if not images_to_process:
            self.logger.info("No images to process. Exiting run.")
            return

        self.logger.info(f"Found {len(images_to_process)} image(s) to process.")
        
        output_image_dir = self.config.OUTPUT_DIR / "inference_outputs"
        output_image_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Annotated images will be saved to: {output_image_dir}")

        for image_path in images_to_process:
            self.logger.info(f"-> Processing: {image_path.name}")
            try:
                image_np = cv2.imread(str(image_path))
                if image_np is None:
                    self.logger.error(f"Failed to read image: {image_path}")
                    continue

                final_predictions = self.detector.predict(image_np)
                self.logger.info(f"  Detected {len(final_predictions)} objects.")

                image_with_predictions = self.detector.draw_predictions_on_image(image_np, final_predictions)
                
                output_image_path = output_image_dir / f"{image_path.stem}_inferred{image_path.suffix}"
                cv2.imwrite(str(output_image_path), image_with_predictions)
                self.logger.info(f"  Saved annotated image to: {output_image_path}")
                
            except Exception as e:
                self.logger.exception(f"An error occurred during inference on {image_path}:")
        
        self.logger.info("--- Inference Run Finished ---")


def main():
    """
    Main function to set up and run the inference process from the command line.
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model.")
    parser.add_argument(
        "input_path", 
        type=str, 
        nargs='?', 
        default=None, 
        help="Path to an image file or a folder of images. If not provided, uses INPUTS_DIR from config.py."
    )
    args = parser.parse_args()

    # --- Setup ---
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_inference.log"
    logger = setup_logging(log_file, logger_name='yolo_inference_logger')

    if not validate_config_and_paths(config, 'inference', logger):
        return

    # --- Load Class Names ---
    class_names_yaml_path = config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME
    names_from_yaml = load_class_names_from_yaml(class_names_yaml_path, logger)
    if names_from_yaml is None:
        logger.error(f"CRITICAL: Could not load class names from '{class_names_yaml_path}'. Exiting.")
        return
    class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}

    # --- Create Detector and Runner Instances ---
    try:
        detector = create_detector_from_config(
            config.MODEL_PATH_FOR_PREDICTION, class_names_map, config, logger
        )
        runner = InferenceRunner(detector, logger)
        runner.run(args.input_path)
    except Exception as e:
        logger.exception(f"A critical error occurred during setup or execution: {e}")

if __name__ == '__main__':
    main()