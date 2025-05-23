import os
import glob
import yaml
import random
import shutil # For copying files
import logging # For logging to file and console
import cv2      # To save images with drawn boxes
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import numpy as np # Required for image manipulation

# --- Configuration ---
# TODO: Update this path if your main dataset directory is different
INPUTS_DIR = "/mnt/c/Work/Repos/MyProjects/DeepLearning/CoinsUltralytics/Data/CoinCount.v34i.yolov5pytorch"  # Base directory containing variant subfolders (e.g., hard, easy) and data.yaml
IMAGE_SUBDIR_BASENAME = "images"  # Basename of the image subdirectory within each variant folder
LABEL_SUBDIR_BASENAME = "labels"  # Basename of the label subdirectory within each variant folder
ORIGINAL_DATA_YAML_NAME = "data.yaml" # Name of your existing data.yaml with class names, located in INPUTS_DIR

OUTPUT_DIR = "yolo_experiment_output"      # Directory to save YAML, results, etc.
# Name for the YAML file this script will generate for YOLO training
DATASET_YAML_NAME = "custom_dataset_for_training.yaml" 
INCORRECT_PREDICTIONS_SUBDIR = "incorrect_predictions" # Subfolder for saving incorrect predictions
LOG_FILE_NAME = "script_run.log" # Name of the log file

# MODEL_NAME = "yolov8n.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
MODEL_NAME = "best.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
EPOCHS = 0 # Number of training epochs. If 0, training is skipped, and MODEL_NAME is loaded for prediction.
IMG_SIZE = 640 # Image size for training

# --- Train/Validation/Test Split Ratios ---
# Ensure these sum to 1.0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Augmentation Configuration ---
# You can adjust these values or add more augmentation parameters as needed.
# Refer to Ultralytics documentation for a full list:
# https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
AUGMENTATION_PARAMS = {
    'hsv_h': 0.015,  # Hue augmentation (fraction)
    'hsv_s': 0.7,    # Saturation augmentation (fraction)
    'hsv_v': 0.4,    # Value augmentation (fraction)
    'degrees': 10.0,   # Image rotation (+/- deg)
    'translate': 0.1, # Image translation (+/- fraction)
    'scale': 0.5,    # Image scale (+/- gain)
    'shear': 0.0,    # Image shear (+/- deg) # Often kept low or 0 for general object detection
    'perspective': 0.0, # Image perspective distortion (+/- fraction), range 0.0-0.001
    'flipud': 0.0,   # Image flip up-down (probability)
    'fliplr': 0.0,   # Image flip left-right (probability)
    'mosaic': 1.0,   # Mosaic augmentation (probability)
    'mixup': 0.1,    # Mixup augmentation (probability)
    # 'copy_paste': 0.0 # Copy-paste augmentation (probability), typically for segmentation
}

# --- Logger Setup ---
# Get a logger instance (this will be the root logger if name is not specified,
# or a specific logger if a name is provided)
logger = logging.getLogger(__name__)
LOG_FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path_obj):
    """Configures logging to both console and a file."""
    if logger.handlers: # Clear existing handlers to avoid duplicate logs if called multiple times
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(logging.INFO) # Set the logging level for the logger instance

    # Create a file handler to write logs to a file
    try:
        file_handler = logging.FileHandler(log_file_path_obj, mode='w') # mode='w' to overwrite log each run
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to console if file handler fails
        # Using print here because logger might not be fully set up for console yet if this fails early
        print(f"CRITICAL ERROR: Failed to set up file logging to {log_file_path_obj}: {e}. Attempting console logging only.")


    # Create a stream handler to print logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO) # Set level for this handler
    stream_handler.setFormatter(LOG_FORMATTER) # You can use a simpler formatter for console if preferred
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized. Console output active. Initial file output to: {log_file_path_obj}")

def copy_log_to_run_directory(initial_log_path, run_dir_path, target_log_filename):
    """
    Copies the initial log file to the specified run directory.
    Args:
        initial_log_path (Path): Path to the initial log file (e.g., in OUTPUT_DIR).
        run_dir_path (Path): Path to the specific training run directory.
        target_log_filename (str): The desired name for the log file in the run directory.
    """
    if run_dir_path and initial_log_path.exists():
        try:
            destination_log_path = run_dir_path / target_log_filename
            # shutil.copy2 will overwrite if destination_log_path already exists
            shutil.copy2(initial_log_path, destination_log_path) 
            logger.info(f"Log file copied to: {destination_log_path}")
        except Exception as e:
            logger.error(f"Error copying log file from {initial_log_path} to {run_dir_path / target_log_filename}: {e}")
    elif not run_dir_path:
        logger.warning("Cannot copy log file: Run directory not specified or not created.")
    elif not initial_log_path.exists():
        logger.warning(f"Cannot copy log file: Initial log file {initial_log_path} does not exist.")


def discover_and_pair_image_labels(inputs_dir_pathobj, image_subdir_basename, label_subdir_basename):
    """
    Scans subdirectories of inputs_dir_pathobj for image and label folders,
    and creates pairs of (image_path, label_path).
    Args:
        inputs_dir_pathobj (Path): The root directory containing variant subfolders.
        image_subdir_basename (str): The name of the image subfolder (e.g., "images").
        label_subdir_basename (str): The name of the label subfolder (e.g., "labels").
    Returns:
        list: A list of tuples, where each tuple is (absolute_image_path, absolute_label_path).
        list: A list of absolute paths to valid label directories found (for class scanning).
    """
    image_label_pairs = []
    valid_label_dirs_for_class_scan = [] # For scanning class IDs later

    if not inputs_dir_pathobj.is_dir():
        logger.error(f"Root dataset directory '{inputs_dir_pathobj}' not found.")
        return image_label_pairs, valid_label_dirs_for_class_scan

    logger.info(f"Scanning for dataset variants in: {inputs_dir_pathobj}")
    for variant_dir in inputs_dir_pathobj.iterdir():
        if variant_dir.is_dir(): # e.g., "hard", "easy"
            image_dir = variant_dir / image_subdir_basename
            label_dir = variant_dir / label_subdir_basename

            if image_dir.is_dir() and label_dir.is_dir():
                logger.info(f"  Processing variant: '{variant_dir.name}'")
                valid_label_dirs_for_class_scan.append(label_dir.resolve())
                image_extensions = ['*.jpg', '*.jpeg', '*.png']
                for ext in image_extensions:
                    for img_path in image_dir.glob(ext):
                        # Construct corresponding label path
                        label_filename = img_path.stem + ".txt"
                        label_path = label_dir / label_filename
                        if label_path.is_file():
                            image_label_pairs.append((img_path.resolve(), label_path.resolve()))
                        else:
                            logger.warning(f"    Label file not found for image '{img_path.name}' in '{label_dir}'. Skipping this image.")
            # else:
                # logger.info(f"  Skipping '{variant_dir.name}': Missing '{image_subdir_basename}' or '{label_subdir_basename}' directory.")
                
    if not image_label_pairs:
        logger.warning(f"No valid image-label pairs found across all variants in '{inputs_dir_pathobj}'.")
        
    return image_label_pairs, valid_label_dirs_for_class_scan

def split_data(image_label_pairs, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Splits a list of (image_path, label_path) pairs into train, validation, and test sets.
    Args:
        image_label_pairs (list): List of (image_path, label_path) tuples.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        seed (int): Random seed for shuffling.
    Returns:
        tuple: (train_pairs, val_pairs, test_pairs)
    """
    if not (0.999 < train_ratio + val_ratio + test_ratio < 1.001): # Check sum with tolerance
        logger.error("Train, validation, and test ratios must sum to approximately 1.0.")
        raise ValueError("Train, validation, and test ratios must sum to approximately 1.0.")

    random.seed(seed) # for reproducible shuffles
    random.shuffle(image_label_pairs)

    total_samples = len(image_label_pairs)
    train_end_idx = int(total_samples * train_ratio)
    val_end_idx = train_end_idx + int(total_samples * val_ratio)

    train_pairs = image_label_pairs[:train_end_idx]
    val_pairs = image_label_pairs[train_end_idx:val_end_idx]
    test_pairs = image_label_pairs[val_end_idx:]
    
    logger.info(f"Data split: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test samples.")
    return train_pairs, val_pairs, test_pairs


def get_unique_class_ids(list_of_label_dir_paths): # Takes a list of Path objects
    """
    Scans label files in multiple directories to find all unique class IDs.
    Args:
        list_of_label_dir_paths (list): A list of Path objects, each pointing to a label directory.
    Returns:
        list: A sorted list of unique integer class IDs.
    """
    unique_ids = set()
    if not list_of_label_dir_paths:
        logger.warning("(get_unique_class_ids): No label directories provided for scanning.")
        return []

    for label_dir_path in list_of_label_dir_paths:
        if not label_dir_path.is_dir(): # Check if the provided path is a directory
            logger.warning(f"Provided label path '{label_dir_path}' is not a directory. Skipping.")
            continue

        for label_file in label_dir_path.glob('*.txt'): # Use glob on Path object
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts: # Ensure line is not empty
                            unique_ids.add(int(parts[0])) # Add the class ID (first element)
            except Exception as e:
                logger.error(f"Error reading label file {label_file}: {e}")
    
    if not unique_ids:
        logger.warning("No class IDs found across all scanned label directories. Check your label files.")
    return sorted(list(unique_ids))

def load_class_names_from_original_yaml(original_yaml_path_obj):
    """
    Loads the 'names' list from a given YAML file (e.g., data.yaml).
    Args:
        original_yaml_path_obj (Path): Path object to the data.yaml file.
    Returns:
        list: A list of class names, or None if file not found or 'names' key is missing/invalid.
    """
    if not original_yaml_path_obj.is_file():
        logger.info(f"Original data YAML file not found at '{original_yaml_path_obj}'.")
        return None
    try:
        with open(original_yaml_path_obj, 'r') as f:
            data = yaml.safe_load(f) # Load YAML content
        # Check if 'names' key exists and is a list
        if data and 'names' in data and isinstance(data['names'], list):
            return data['names'] # Return the list of names
        else:
            logger.warning(f"'names' key not found or is not a list in '{original_yaml_path_obj}'.")
            return None
    except Exception as e:
        logger.error(f"Error reading original data YAML file {original_yaml_path_obj}: {e}")
        return None

def create_dataset_yaml(dataset_root_abs_path_str, 
                        train_rel_img_dir_paths, 
                        val_rel_img_dir_paths,   
                        test_rel_img_dir_paths,  
                        class_names_dict, 
                        num_classes_val, 
                        output_yaml_path_obj):
    """
    Creates the dataset.yaml file required by YOLO for training.
    'train', 'val', and 'test' will be lists of relative image DIRECTORY paths.
    Args:
        dataset_root_abs_path_str (str): Absolute path to the dataset root directory (INPUTS_DIR).
        train_rel_img_dir_paths (list): List of strings, relative paths for training image directories.
        val_rel_img_dir_paths (list): List of strings, relative paths for validation image directories.
        test_rel_img_dir_paths (list): List of strings, relative paths for test image directories.
        class_names_dict (dict): Dictionary mapping class ID (int) to class name (str).
        num_classes_val (int): The total number of classes (nc).
        output_yaml_path_obj (Path): Path object to save the generated .yaml file.
    """
    # Initialize a list for names, ordered by class ID, as expected by YOLO YAML format
    names_list_for_yaml = [""] * num_classes_val 
    for class_id, name in class_names_dict.items():
        if 0 <= class_id < num_classes_val:
            names_list_for_yaml[class_id] = str(name) # Ensure name is a string
        else:
            # This warning indicates a mismatch that might cause issues during training
            logger.warning(f"(create_dataset_yaml): Class ID {class_id} ('{name}') is out of range for nc={num_classes_val}.")
    
    # Fill any gaps if class_names_dict didn't cover all IDs up to num_classes_val
    # (e.g., if num_classes was determined by max_id from labels and some intermediate IDs had no names)
    for i in range(num_classes_val):
        if not names_list_for_yaml[i]: # If the name is still an empty string
             names_list_for_yaml[i] = f"class_{i}" # Fallback to generic name

    data = {
        'path': dataset_root_abs_path_str,  # Absolute path to dataset root (INPUTS_DIR)
        'train': [str(p) for p in train_rel_img_dir_paths], # List of relative paths to train image DIRS
        'val': [str(p) for p in val_rel_img_dir_paths],   # List of relative paths to val image DIRS
        'test': [str(p) for p in test_rel_img_dir_paths],  # List of relative paths to test image DIRS
        'nc': num_classes_val,
        'names': names_list_for_yaml 
    }

    try:
        with open(output_yaml_path_obj, 'w') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=None) # Use default_flow_style for better readability
        logger.info(f"Successfully created dataset YAML: {output_yaml_path_obj}")
        # Log YAML content as a multi-line string for better readability in logs
        yaml_content_str = yaml.dump(data, sort_keys=False, default_flow_style=None)
        logger.info(f"Generated YAML content:\n{yaml_content_str}") # Using \n for multi-line log
        logger.info("Important: Please verify the 'path', 'train', 'val', and 'test' entries in the generated YAML file.") # Removed extra \n
        logger.info(f"YOLO will look for labels in directories relative to the image paths listed.")
        # Provide an example of how labels are expected to be found
        example_img_dir_str = train_rel_img_dir_paths[0] if train_rel_img_dir_paths else f"variant_folder/{IMAGE_SUBDIR_BASENAME}"
        logger.info(f"(e.g., if a train/val/test entry in YAML is '{example_img_dir_str}',")
        logger.info(f" Ultralytics will glob for images there. Labels are expected in a parallel '{LABEL_SUBDIR_BASENAME}' structure relative to the variant folder, e.g., '{Path(example_img_dir_str).parent / LABEL_SUBDIR_BASENAME}')")


    except Exception as e:
        logger.error(f"Error writing YAML file {output_yaml_path_obj}: {e}")

# --- HELPER FUNCTIONS FOR DRAWING ---
def parse_yolo_annotations(label_file_path):
    """
    Parses a YOLO format label file and returns detailed annotations.
    Args:
        label_file_path (Path): Path to the .txt label file.
    Returns:
        list: A list of tuples, where each tuple is (class_id, x_center, y_center, width, height).
              Returns an empty list if the file cannot be parsed.
    """
    annotations = []
    if not label_file_path.is_file():
        logger.warning(f"(parse_yolo_annotations): Label file not found at {label_file_path}")
        return annotations
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
    except Exception as e:
        logger.error(f"Error parsing YOLO annotation file {label_file_path}: {e}")
    return annotations

def draw_all_annotations(image_np, pred_boxes, gt_annotations, class_names_map):
    """
    Draws predicted and ground truth bounding boxes and labels on an image.
    Args:
        image_np (np.ndarray): The image to draw on.
        pred_boxes (ultralytics.engine.results.Boxes): The Boxes object from prediction results.
                                                      Can be None if no predictions.
        gt_annotations (list): A list of tuples from parse_yolo_annotations.
                               Can be empty if no ground truth.
        class_names_map (dict): Dictionary mapping class ID (int) to class name (str).
    Returns:
        np.ndarray: The image with all annotations drawn.
    """
    img_h, img_w = image_np.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Define Color Map for Predicted AND Ground Truth Boxes ---
    # Colors are in BGR format
    BOX_COLOR_MAP = {
        "one": (0, 255, 255),   # Yellow
        "two": (128, 0, 128),   # Purple
        "five": (0, 0, 255),    # Red
        "ten": (255, 255, 0),     # Cyan/Teal (Updated from Dark Green)
        # Add more class names and colors as needed
    }
    DEFAULT_BOX_COLOR = (255, 0, 0) # Blue for any other classes not in map

    # --- Draw Predicted Boxes ---
    if pred_boxes is not None:
        pred_font_scale = 1.5 # Font scale for prediction text
        pred_text_thickness = 4 # Thickness for text, must be an integer
        pred_box_thickness = 2 # Thickness for box lines, must be an integer

        for i in range(len(pred_boxes.xyxy)):
            x1, y1, x2, y2 = map(int, pred_boxes.xyxy[i])
            class_id = int(pred_boxes.cls[i])
            confidence = float(pred_boxes.conf[i])

            class_name = class_names_map.get(class_id, f"ID_{class_id}")
            label = f"{class_name} {confidence:.2f}"
            
            # Get color for prediction using the BOX_COLOR_MAP
            # Convert class_name to lowercase for robust matching with BOX_COLOR_MAP keys
            color = BOX_COLOR_MAP.get(class_name.lower(), DEFAULT_BOX_COLOR)

            # Draw the rectangle for the prediction
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, pred_box_thickness)

            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label, font, pred_font_scale, pred_text_thickness)

            # Position the label at the top-left of the box
            label_x_pos = x1
            label_y_pos = y1 - baseline - 3 
            if label_y_pos < text_h : # Adjust if off-screen
                label_y_pos = y1 + text_h + baseline + 3 # Place below if no space above

            cv2.rectangle(image_np, (label_x_pos, label_y_pos - text_h - baseline), (label_x_pos + text_w, label_y_pos + baseline), color, -1)
            cv2.putText(image_np, label, (label_x_pos, label_y_pos), font, pred_font_scale, (0,0,0), pred_text_thickness, cv2.LINE_AA) # Black text

    # --- Draw Ground Truth Boxes ---
    if gt_annotations:
        gt_font_scale = 1.4 # Font scale for ground truth text
        gt_text_thickness = 3 # Thickness for text, must be integer
        gt_box_thickness = 2 # Thickness for box lines, must be integer


        for ann in gt_annotations:
            class_id, x_center, y_center, width, height = ann
            abs_x_center, abs_y_center = x_center * img_w, y_center * img_h
            abs_width, abs_height = width * img_w, height * img_h
            x1, y1 = int(abs_x_center - abs_width / 2), int(abs_y_center - abs_height / 2)
            x2, y2 = int(x1 + abs_width), int(y1 + abs_height)

            class_name = class_names_map.get(class_id, f"ID_{class_id}")
            # Get color for ground truth using the BOX_COLOR_MAP
            gt_color = BOX_COLOR_MAP.get(class_name.lower(), DEFAULT_BOX_COLOR) # Fallback to default if GT class not in map

            cv2.rectangle(image_np, (x1, y1), (x2, y2), gt_color, gt_box_thickness)

            label = f"GT: {class_name}"
            (text_w, text_h), baseline = cv2.getTextSize(label, font, gt_font_scale, gt_text_thickness)
            
            text_margin = 3
            label_x_pos = x2 - text_w - text_margin 
            label_y_pos = y1 + text_h + text_margin 

            cv2.rectangle(image_np, (label_x_pos, label_y_pos - text_h - baseline), (label_x_pos + text_w, label_y_pos + baseline), gt_color, -1)
            cv2.putText(image_np, label, (label_x_pos, label_y_pos), font, gt_font_scale, (0,0,0), gt_text_thickness, cv2.LINE_AA) # Black text

    # --- Draw Legend ---
    legend_font_scale = 0.5
    # Position legend at the bottom of the image
    cv2.putText(image_np, "Predictions (Custom Colors, Top-Left)", (10, img_h - 40), font, legend_font_scale, (220, 220, 220), 1, cv2.LINE_AA) # Light gray for better visibility
    cv2.putText(image_np, "Ground Truth (Custom Colors, Inside Top-Right)", (10, img_h - 20), font, legend_font_scale, (220,220,220), 1, cv2.LINE_AA) # Also light gray for consistency

    return image_np
# --- HELPER FUNCTIONS END ---


def main():
    """
    Main function to run the YOLO training and evaluation pipeline.
    """
    # --- Basic Path Setup ---
    output_path = Path(OUTPUT_DIR) # Original variable name from OD_ultralytics.py context
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Setup Logging to an initial file in OUTPUT_DIR ---
    initial_log_file_path = output_path / LOG_FILE_NAME
    setup_logging(initial_log_file_path) 


    inputs_dir_pathobj = Path(INPUTS_DIR).resolve() # Resolve INPUTS_DIR to absolute path
    # Path to the original data.yaml which might contain class names
    original_data_yaml_abs_path = (inputs_dir_pathobj / ORIGINAL_DATA_YAML_NAME).resolve()

    logger.info(f"Using dataset base directory: {inputs_dir_pathobj}") 
    logger.info(f"Image subdirectory basename: '{IMAGE_SUBDIR_BASENAME}'")
    logger.info(f"Label subdirectory basename: '{LABEL_SUBDIR_BASENAME}'")
    logger.info(f"Attempting to load class names from: {original_data_yaml_abs_path}")

    # --- Step 0: Discover all image-label pairs and label directories ---
    logger.info("--- Step 0: Discovering and Pairing Image-Label Data ---") 
    # Get lists of (image_path, label_path) pairs and all valid label directories
    all_image_label_pairs, all_label_dirs_abs_for_class_scan = discover_and_pair_image_labels(
        inputs_dir_pathobj, IMAGE_SUBDIR_BASENAME, LABEL_SUBDIR_BASENAME
    )

    if not all_image_label_pairs:
        logger.error("No image-label pairs found. Cannot proceed.")
        return
    
    # --- Step 1: Determine Class Names and Number of Classes ---
    logger.info("--- Step 1: Determining Class Names and Number ---") 
    class_names_map = {}  # This will be the dict {id: name}
    num_classes = 0       # This will be nc (number of classes)

    # Try to load class names from the user-provided data.yaml
    names_list_from_file = load_class_names_from_original_yaml(original_data_yaml_abs_path)

    if names_list_from_file is not None: # Successfully loaded names from data.yaml (could be an empty list)
        logger.info(f"Successfully loaded {len(names_list_from_file)} class names from '{original_data_yaml_abs_path}'.")
        class_names_map = {i: str(name) for i, name in enumerate(names_list_from_file)}
        num_classes = len(names_list_from_file)

        # Optional: Validate against actual labels found, only if label dirs were discovered
        if all_label_dirs_abs_for_class_scan: 
            class_ids_in_labels = get_unique_class_ids(all_label_dirs_abs_for_class_scan) 
            if class_ids_in_labels: # If any class IDs were actually found in labels
                max_id_in_labels = max(class_ids_in_labels)
                if num_classes > 0 and max_id_in_labels >= num_classes: # Only warn if num_classes is defined
                    logger.warning(f"Max class ID in labels ({max_id_in_labels}) is >= number of classes defined in '{original_data_yaml_abs_path}' ({num_classes}).")
                    logger.warning("         This might lead to errors or misinterpretations during training if labels use IDs outside the defined names.")
                elif num_classes == 0 and max_id_in_labels >= 0 : # If data.yaml has nc:0 but labels exist
                     logger.warning(f"Data YAML '{original_data_yaml_abs_path}' defines 0 classes, but labels contain class IDs (max: {max_id_in_labels}).")
                     logger.warning("         Consider updating data.yaml or generating names from labels.")

    else: # Fallback: data.yaml not found or 'names' key missing/invalid
        logger.warning(f"Could not load 'names' list from '{original_data_yaml_abs_path}'.")
        if not all_label_dirs_abs_for_class_scan: # If no label dirs found, and data.yaml failed, cannot proceed
            logger.error("No label directories found to fall back on for class discovery. Cannot proceed.")
            return
            
        logger.info("         Attempting to generate generic class names based on IDs found in label files across all variants.")
        class_ids_in_labels = get_unique_class_ids(all_label_dirs_abs_for_class_scan) 
        
        if not class_ids_in_labels: # No labels found, and no data.yaml names
            logger.error("No class IDs found in any label files, and could not load names from data.yaml. Cannot proceed.")
            return # Exit
        
        max_id_in_labels = max(class_ids_in_labels)
        num_classes = max_id_in_labels + 1 # nc is count, so max_id + 1 for 0-indexed
        class_names_map = {i: f'class_{i}' for i in range(num_classes)} # Create names for all IDs up to max_id
        logger.info(f"         Generated {num_classes} generic class names based on labels (up to max ID {max_id_in_labels}).")

    logger.info(f"Final effective number of classes (nc): {num_classes}")
    logger.info(f"Final class names map: {class_names_map}")

    # Critical check: if nc is 0 (and it wasn't explicitly from an empty 'names: []' in data.yaml), then error.
    # An empty 'names: []' list in data.yaml is a valid scenario for some use cases (e.g. if you only want to detect 'something' without classes).
    if num_classes == 0 and not (names_list_from_file is not None and len(names_list_from_file) == 0):
        logger.error("Number of classes is zero, and it wasn't explicitly set via an empty 'names' list in data.yaml. Cannot proceed.")
        logger.error("Please check your data.yaml or label files.")
        return

    # --- Step 1.5: Split Data into Train, Validation, Test ---
    logger.info("--- Step 1.5: Splitting Data ---") 
    # Split all discovered image-label pairs
    train_pairs, val_pairs, test_pairs = split_data(
        all_image_label_pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Create sets of absolute image paths for quick lookup of which set an image belongs to
    train_image_paths_set = {img_path for img_path, _ in train_pairs}
    val_image_paths_set = {img_path for img_path, _ in val_pairs}
    test_image_paths_set = {img_path for img_path, _ in test_pairs}
    
    # Get unique relative DIRECTORY paths for images in each set for the YAML
    # These directories contain the actual image files.
    train_rel_img_dirs = sorted(list(set(img_path.parent.relative_to(inputs_dir_pathobj) for img_path, _ in train_pairs if train_pairs)))
    val_rel_img_dirs = sorted(list(set(img_path.parent.relative_to(inputs_dir_pathobj) for img_path, _ in val_pairs if val_pairs)))
    test_rel_img_dirs = sorted(list(set(img_path.parent.relative_to(inputs_dir_pathobj) for img_path, _ in test_pairs if test_pairs)))

    # Ensure these are strings for YAML, even if lists are empty
    train_rel_img_dirs_str = [str(d) for d in train_rel_img_dirs]
    val_rel_img_dirs_str = [str(d) for d in val_rel_img_dirs]
    test_rel_img_dirs_str = [str(d) for d in test_rel_img_dirs]


    # --- Step 2: Create dataset.yaml for Training/Evaluation ---
    logger.info("--- Step 2: Creating Dataset YAML for Training/Evaluation ---")
    # DATASET_YAML_NAME is for the generated YAML
    generated_dataset_yaml_path = output_path / DATASET_YAML_NAME 
    
    create_dataset_yaml(
        dataset_root_abs_path_str=str(inputs_dir_pathobj), # This is INPUTS_DIR (absolute)
        train_rel_img_dir_paths=train_rel_img_dirs_str,    # Pass list of relative directory paths
        val_rel_img_dir_paths=val_rel_img_dirs_str,      # Pass list of relative directory paths
        test_rel_img_dir_paths=test_rel_img_dirs_str,    # Pass list of relative directory paths
        class_names_dict=class_names_map,
        num_classes_val=num_classes,
        output_yaml_path_obj=generated_dataset_yaml_path
    )
    
    if not generated_dataset_yaml_path.exists():
        logger.error(f"Failed to create {generated_dataset_yaml_path}. Exiting.") 
        return

    # --- Step 3: Train the model (or load for prediction) ---
    logger.info(f"--- Step 3: Model Training/Loading ---")
    trained_model_path = None # Path to best.pt or specified model
    model_object_for_ops = None # To store the model object after training or loading
    current_run_dir = None # To store the actual path of the current training or prediction run

    # --- Conditional Training Block ---
    # If EPOCHS > 0, the script will train the model.
    # If EPOCHS == 0, it will skip training and load the specified MODEL_NAME for prediction.
    if EPOCHS > 0:
        logger.info(f"EPOCHS set to {EPOCHS}. Proceeding with model training.")
        try:
            # Initialize model for training. MODEL_NAME can be a base model like 'yolov8n.pt' or a path to resume training.
            model = YOLO(MODEL_NAME)
            model_object_for_ops = model # Store initial for fallback if training fails early
            logger.info(f"Starting training with model: {MODEL_NAME}, epochs: {EPOCHS}, img_size: {IMG_SIZE}")
            project_base_dir = output_path / "training_runs"
            # Use stem of MODEL_NAME if it's a path, or MODEL_NAME directly if it's like 'yolov8n.pt'
            base_model_name_for_run = Path(MODEL_NAME).stem if Path(MODEL_NAME).suffix else MODEL_NAME
            training_run_name = f"{base_model_name_for_run}"

            results = model.train(
                data=str(generated_dataset_yaml_path),
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                project=str(project_base_dir),
                name=training_run_name,
                exist_ok=False, # Set to True if you want to overwrite existing runs with the same name, False to create new (e.g., run_name2)
                **AUGMENTATION_PARAMS
            )
            logger.info("Training completed.")
            model_object_for_ops = model # The model object is updated in-place by .train()
            current_run_dir = Path(results.save_dir) # Get the actual save directory of this run
            logger.info(f"Training run artifacts saved to: {current_run_dir}")

            # Determine the path to the best trained model
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and model.trainer.best and Path(model.trainer.best).exists():
                trained_model_path = str(Path(model.trainer.best).resolve())
                logger.info(f"Best model saved at: {trained_model_path}")
            elif current_run_dir and (current_run_dir / "weights" / "best.pt").exists():
                trained_model_path = str((current_run_dir / "weights" / "best.pt").resolve())
                logger.info(f"Found best model at expected path: {trained_model_path}")
            else:
                logger.warning("Could not determine exact path to best.pt.")
        except Exception as e:
            logger.exception(f"An error occurred during training:")
            if 'results' in locals() and hasattr(results, 'save_dir') and Path(results.save_dir).is_dir():
                current_run_dir = Path(results.save_dir) # Try to get run dir even if error occurred late in training
            # The finally block in main() will attempt to copy the log using whatever current_run_dir is.
            return # Exit script
    else: # EPOCHS == 0, skip training and load a pre-trained model
        logger.info(f"EPOCHS is set to 0. Skipping training.")
        model_path_to_load = Path(MODEL_NAME)
        if not model_path_to_load.exists() or not model_path_to_load.is_file():
            logger.error(f"Model not found at '{model_path_to_load}'. When EPOCHS is 0, MODEL_NAME must be a valid path to a .pt model file.")
            return
        try:
            model_object_for_ops = YOLO(model_path_to_load) # Load the specified model
            trained_model_path = str(model_path_to_load.resolve()) # Store the absolute path to the loaded model
            logger.info(f"Successfully loaded model from: {trained_model_path}")

            # --- Create a unique directory for this prediction-only run ---
            base_prediction_run_name = f"{model_path_to_load.stem}_prediction_only_run"
            prediction_project_dir = output_path / "prediction_runs"
            
            current_run_dir_candidate = prediction_project_dir / base_prediction_run_name
            counter = 1
            # Loop to find a unique directory name, appending a number if needed
            while current_run_dir_candidate.exists():
                counter += 1
                current_run_dir_candidate = prediction_project_dir / f"{base_prediction_run_name}{counter}"
            
            current_run_dir = current_run_dir_candidate
            current_run_dir.mkdir(parents=True, exist_ok=False) # exist_ok=False as we've ensured uniqueness
            logger.info(f"Outputs for this prediction-only run will be saved in: {current_run_dir}")

        except Exception as e:
            logger.exception(f"An error occurred while loading the model '{model_path_to_load}':")
            return # --- End of EPOCHS == 0 block ---

    # --- Setup for incorrect predictions directory ---
    incorrect_preds_train_dir = incorrect_preds_val_dir = incorrect_preds_test_dir = None
    if current_run_dir:
        incorrect_preds_base_dir = current_run_dir / INCORRECT_PREDICTIONS_SUBDIR
        incorrect_preds_train_dir = incorrect_preds_base_dir / "train"
        incorrect_preds_val_dir = incorrect_preds_base_dir / "validation"
        incorrect_preds_test_dir = incorrect_preds_base_dir / "test"
        logger.info(f"Saving incorrect predictions to subfolders within: {incorrect_preds_base_dir}")
        incorrect_preds_base_dir.mkdir(parents=True, exist_ok=True)
        incorrect_preds_train_dir.mkdir(exist_ok=True)
        incorrect_preds_val_dir.mkdir(exist_ok=True)
        incorrect_preds_test_dir.mkdir(exist_ok=True)
    else:
        logger.warning("Run directory not identified. Incorrect predictions will not be saved.")

    # --- Step 4: Evaluate the model ---
    logger.info("--- Step 4: Evaluating Model ---")
    eval_model_instance = model_object_for_ops or (YOLO(trained_model_path) if trained_model_path else None)
    if not eval_model_instance:
        logger.error("No model available for evaluation.")
    else:
        try:
            if val_rel_img_dirs_str:
                logger.info("--- Evaluating on Validation Set ---")
                eval_model_instance.val(data=str(generated_dataset_yaml_path), split='val', project=str(current_run_dir), name="eval_val_results")
            if test_rel_img_dirs_str:
                logger.info("--- Evaluating on Test Set ---")
                eval_model_instance.val(data=str(generated_dataset_yaml_path), split='test', project=str(current_run_dir), name="eval_test_results")
        except Exception as e:
            logger.exception("An error occurred during evaluation:")

    # --- Step 5: Predict on ALL images, count objects, and save incorrect predictions ---
    logger.info("--- Step 5: Predicting and Comparing on ALL Images ---")
    predict_model_instance = model_object_for_ops or (YOLO(trained_model_path) if trained_model_path else None)
    if not predict_model_instance:
        logger.error("No model available for prediction.")
        copy_log_to_run_directory(initial_log_file_path, current_run_dir, LOG_FILE_NAME)
        return

    logger.info(f"Found {len(all_image_label_pairs)} total image-label pairs for comparison.")
    incorrect_prediction_count = 0

    try:
        # --- MODIFIED PREDICTION LOOP START ---
        for image_abs_path, label_abs_path in all_image_label_pairs:
            logger.info(f"  Processing: {image_abs_path.relative_to(inputs_dir_pathobj)}")
            
            # Load the original image using OpenCV
            image_to_draw_on = cv2.imread(str(image_abs_path))
            if image_to_draw_on is None:
                logger.error(f"    Failed to read image {image_abs_path}. Skipping.")
                continue

            # Perform prediction
            pred_results_list = predict_model_instance.predict(source=image_to_draw_on.copy(), save=False, verbose=False) # Pass a copy to predict
            
            predicted_class_counts = Counter()
            current_pred_boxes = None # To store ultralytics.engine.results.Boxes object
            
            if pred_results_list:
                r = pred_results_list[0]
                # Manually draw predicted boxes onto our loaded image
                if r.boxes:
                    current_pred_boxes = r.boxes 
                    predicted_class_counts = Counter(int(cls_id) for cls_id in r.boxes.cls.tolist())

            ground_truth_annotations = parse_yolo_annotations(label_abs_path)
            ground_truth_class_counts = Counter(ann[0] for ann in ground_truth_annotations)

            if predicted_class_counts != ground_truth_class_counts:
                incorrect_prediction_count += 1
                logger.info(f"    INCORRECT prediction for {image_abs_path.name}.")
                logger.info(f"      GT counts: {dict(ground_truth_class_counts)}")
                logger.info(f"      Pred counts: {dict(predicted_class_counts)}")

                # Draw all annotations (predicted then ground truth)
                image_with_all_annotations = draw_all_annotations(
                    image_to_draw_on.copy(), # Pass a fresh copy of the original image
                    current_pred_boxes, 
                    ground_truth_annotations, 
                    class_names_map
                )

                # Determine which folder to save the incorrect prediction in
                target_dir = None
                if image_abs_path in train_image_paths_set: target_dir = incorrect_preds_train_dir
                elif image_abs_path in val_image_paths_set: target_dir = incorrect_preds_val_dir
                elif image_abs_path in test_image_paths_set: target_dir = incorrect_preds_test_dir

                if target_dir:
                    try:
                        destination_path = target_dir / image_abs_path.name
                        cv2.imwrite(str(destination_path), image_with_all_annotations)
                        logger.info(f"      Saved comparison image to {destination_path}")
                    except Exception as e:
                        logger.error(f"      Failed to save comparison image for {image_abs_path.name}: {e}")
        # --- MODIFIED PREDICTION LOOP END ---

        logger.info(f"Total incorrect predictions (by class counts): {incorrect_prediction_count} out of {len(all_image_label_pairs)} images.")

    except Exception as e:
        logger.exception("An error occurred during prediction and comparison loop:")

    finally:
        copy_log_to_run_directory(initial_log_file_path, current_run_dir, LOG_FILE_NAME)


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Before running, ensure:
    # 1. INPUTS_DIR points to your main dataset directory.
    # 2. This directory contains ORIGINAL_DATA_YAML_NAME (e.g., data.yaml) if you want to use its class names.
    # 3. Variant subfolders (e.g., "hard", "easy") are inside INPUTS_DIR.
    # 4. Each variant folder has IMAGE_SUBDIR_BASENAME (e.g., "images") and LABEL_SUBDIR_BASENAME (e.g., "labels").
    # 5. If EPOCHS = 0, MODEL_NAME must be a valid path to a .pt model file.
    #    If EPOCHS > 0, MODEL_NAME can be a base model (e.g., 'yolov8n.pt') or a .pt file to resume from.
    #
    # Example expected structure for INPUTS_DIR:
    # INPUTS_DIR/
    #  ├── data.yaml (ORIGINAL_DATA_YAML_NAME)
    #  ├── variant1/
    #  │   ├── images/
    #  │   │   └── img1.jpg
    #  │   └── labels/
    #  │       └── img1.txt
    #  ├── variant2/
    #  │   ├── images/
    #  │   │   └── img2.jpg
    #  │   └── labels/
    #  │       └── img2.txt
    #  └── ...
    main()
