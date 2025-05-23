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
import csv # For exporting data to CSV

# --- Configuration ---
# TODO: Update this path if your main dataset directory is different
INPUTS_DIR = "/mnt/c/Work/Repos/MyProjects/DeepLearning/CoinsUltralytics/Data/CoinCount.v37i.yolov5pytorch"  # Base directory containing variant subfolders (e.g., hard, easy) and data.yaml
IMAGE_SUBDIR_BASENAME = "images"  # Basename of the image subdirectory within each variant folder
LABEL_SUBDIR_BASENAME = "labels"  # Basename of the label subdirectory within each variant folder
ORIGINAL_DATA_YAML_NAME = "data.yaml" # Name of your existing data.yaml with class names, located in INPUTS_DIR

OUTPUT_DIR = "yolo_experiment_output"      # Directory to save YAML, results, etc.
# Name for the YAML file this script will generate for YOLO training
DATASET_YAML_NAME = "custom_dataset_for_training.yaml" 
INCORRECT_PREDICTIONS_SUBDIR = "incorrect_predictions" # Subfolder for saving incorrect predictions
LOG_FILE_NAME = "script_run.log" # Name of the log file
PREDICTIONS_CSV_NAME = "predictions_summary.csv" # Name for the CSV output file
# MODEL_NAME = "yolov8n.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
MODEL_NAME = "yolov8n_best.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
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

# --- Prediction Settings ---
IOU_SUPPRESSION_THRESHOLD = 0.4 # Threshold for custom inter-class suppression
BOX_MATCHING_IOU_THRESHOLD = 0.5 # IoU threshold to consider a predicted box as "correct" if class also matches GT
DEFAULT_CONF_THRESHOLD = 0.25 # Default confidence if not specified per class
PER_CLASS_CONF_THRESHOLDS = {
    "one": 0.35,    # Example: Higher threshold for "one"
    "two": 0.45,    # Example: Lower threshold for "two"
    "five": 0.35,   # Example: Higher threshold for "five"
    "ten": 0.8,    # Example: Moderate threshold for "ten"
    # Add other class names (lowercase) and their desired thresholds here
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

def calculate_iou(box1_xyxy, box2_xyxy):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1_xyxy (list or tuple): [x1, y1, x2, y2] for the first box.
        box2_xyxy (list or tuple): [x1, y1, x2, y2] for the second box.
    Returns:
        float: The IoU score.
    """
    # Determine the coordinates of the intersection rectangle
    x1_i = max(box1_xyxy[0], box2_xyxy[0])
    y1_i = max(box1_xyxy[1], box2_xyxy[1])
    x2_i = min(box1_xyxy[2], box2_xyxy[2])
    y2_i = min(box1_xyxy[3], box2_xyxy[3])

    # Calculate the area of intersection rectangle
    intersection_width = max(0, x2_i - x1_i)
    intersection_height = max(0, y2_i - y1_i)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of both bounding boxes
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def draw_all_annotations(image_np, filtered_pred_data_list, gt_annotations, class_names_map):
    """
    Draws filtered predicted and ground truth bounding boxes and labels on an image.
    Args:
        image_np (np.ndarray): The image to draw on.
        filtered_pred_data_list (list): List of dictionaries, each containing {'xyxy', 'conf', 'cls'}
                                       and optionally 'correctness_status' (though not used for drawing label text here).
                                       Can be None or empty.
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
        "ten": (255, 255, 0),     # Cyan/Teal
    }
    DEFAULT_BOX_COLOR = (255, 0, 0) # Blue for any other classes not in map


    # --- Draw Filtered Predicted Boxes ---
    if filtered_pred_data_list:
        pred_font_scale = 1.5 # Font scale for prediction text
        pred_text_thickness = 4 # Thickness for text, must be an integer
        pred_box_thickness = 2 # Thickness for box lines, must be an integer

        for pred_data in filtered_pred_data_list:
            x1, y1, x2, y2 = map(int, pred_data['xyxy'])
            class_id = int(pred_data['cls'])
            confidence = float(pred_data['conf'])

            class_name = class_names_map.get(class_id, f"ID_{class_id}")
            label = f"{class_name} {confidence:.2f}"
            
            # Get color for prediction using the BOX_COLOR_MAP
            # Convert class_name to lowercase for robust matching with BOX_COLOR_MAP keys
            color = BOX_COLOR_MAP.get(class_name.lower().strip(), DEFAULT_BOX_COLOR)
            text_color_on_bg = (0,0,0) # Black text by default

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
            cv2.putText(image_np, label, (label_x_pos, label_y_pos), font, pred_font_scale, text_color_on_bg, pred_text_thickness, cv2.LINE_AA)

    # --- Draw Ground Truth Boxes ---
    if gt_annotations:
        gt_font_scale = 1.4 # Font scale for ground truth text
        gt_text_thickness = 3 # Thickness for text, must be integer
        gt_box_thickness = 2 # Thickness for box lines, must be integer


        for ann in gt_annotations:
            class_id, x_center, y_center, width, height = ann
            abs_x_center, abs_y_center = x_center * img_w, y_center * img_h
            abs_width, abs_height = width * img_w, height * img_h
            x1_gt, y1_gt = int(abs_x_center - abs_width / 2), int(abs_y_center - abs_height / 2)
            x2_gt, y2_gt = int(x1_gt + abs_width), int(y1_gt + abs_height)

            class_name = class_names_map.get(class_id, f"ID_{class_id}")
            # Get color for ground truth using the BOX_COLOR_MAP
            gt_color = BOX_COLOR_MAP.get(class_name.lower().strip(), DEFAULT_BOX_COLOR) # Added .strip() here for safety too

            cv2.rectangle(image_np, (x1_gt, y1_gt), (x2_gt, y2_gt), gt_color, gt_box_thickness)

            label = f"GT: {class_name}"
            (text_w, text_h), baseline = cv2.getTextSize(label, font, gt_font_scale, gt_text_thickness)
            
            text_margin = 3
            label_x_pos = x2_gt - text_w - text_margin 
            label_y_pos = y1_gt + text_h + text_margin 

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

    # --- Step 0: Discover all image-label pairs and label directories ---
    # --- Step 0 & 1: Data Preparation & Class Names ---
    # Get lists of (image_path, label_path) pairs and all valid label directories
    all_image_label_pairs, all_label_dirs_abs_for_class_scan = discover_and_pair_image_labels(
        inputs_dir_pathobj, IMAGE_SUBDIR_BASENAME, LABEL_SUBDIR_BASENAME
    )

    if not all_image_label_pairs:
        logger.error("No image-label pairs found. Cannot proceed.")
        return
    
    class_names_map = {}  # This will be the dict {id: name}
    num_classes = 0       # This will be nc (number of classes)

    # Try to load class names from the user-provided data.yaml
    names_list_from_file = load_class_names_from_original_yaml(original_data_yaml_abs_path)

    if names_list_from_file is not None: # Successfully loaded names from data.yaml (could be an empty list)
        logger.info(f"Successfully loaded {len(names_list_from_file)} class names from '{original_data_yaml_abs_path}'.")
        class_names_map = {i: str(name).strip() for i, name in enumerate(names_list_from_file)}
        num_classes = len(names_list_from_file)
    else: # Fallback: data.yaml not found or 'names' key missing/invalid
        logger.warning(f"Could not load 'names' list from '{original_data_yaml_abs_path}'.")
        if not all_label_dirs_abs_for_class_scan: # If no label dirs found, and data.yaml failed, cannot proceed
            logger.error("No label directories found for class discovery fallback. Cannot proceed.")
            return
            
        class_ids_in_labels = get_unique_class_ids(all_label_dirs_abs_for_class_scan) 
        
        if not class_ids_in_labels: # No labels found, and no data.yaml names
            logger.error("No class IDs found in any label files, and no data.yaml names. Cannot proceed.")
            return # Exit
        
        max_id_in_labels = max(class_ids_in_labels)
        num_classes = max_id_in_labels + 1 # nc is count, so max_id + 1 for 0-indexed
        class_names_map = {i: f'class_{i}' for i in range(num_classes)} # Create names for all IDs up to max_id
        logger.info(f"Generated {num_classes} generic class names based on labels.")

    logger.info(f"Final effective number of classes (nc): {num_classes}")
    logger.info(f"Final class names map: {class_names_map}")

    # Critical check: if nc is 0 (and it wasn't explicitly from an empty 'names: []' in data.yaml), then error.
    # An empty 'names: []' list in data.yaml is a valid scenario for some use cases (e.g. if you only want to detect 'something' without classes).
    if num_classes == 0 and not (names_list_from_file is not None and len(names_list_from_file) == 0):
        logger.error("Number of classes is zero. Cannot proceed.")
        return

    # Split all discovered image-label pairs
    train_pairs, val_pairs, test_pairs = split_data(all_image_label_pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Create sets of absolute image paths for quick lookup of which set an image belongs to
    train_image_paths_set = {img_path for img_path, _ in train_pairs}
    val_image_paths_set = {img_path for img_path, _ in val_pairs}
    test_image_paths_set = {img_path for img_path, _ in test_pairs}
    
    # Get unique relative DIRECTORY paths for images in each set for the YAML
    # These directories contain the actual image files.
    train_rel_img_dirs_str = [str(p.parent.relative_to(inputs_dir_pathobj)) for p, _ in train_pairs]
    val_rel_img_dirs_str = [str(p.parent.relative_to(inputs_dir_pathobj)) for p, _ in val_pairs]
    test_rel_img_dirs_str = [str(p.parent.relative_to(inputs_dir_pathobj)) for p, _ in test_pairs]

    # Ensure these are strings for YAML, even if lists are empty
    train_rel_img_dirs_str = sorted(list(set(train_rel_img_dirs_str)))
    val_rel_img_dirs_str = sorted(list(set(val_rel_img_dirs_str)))
    test_rel_img_dirs_str = sorted(list(set(test_rel_img_dirs_str)))


    # --- Step 2: Create dataset.yaml ---
    generated_dataset_yaml_path = output_path / DATASET_YAML_NAME
    create_dataset_yaml(str(inputs_dir_pathobj), train_rel_img_dirs_str, val_rel_img_dirs_str, test_rel_img_dirs_str, class_names_map, num_classes, generated_dataset_yaml_path)
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
                optimizer='Adam',
                lr0=0.0001, # Initial learning rate
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
            logger.info(f"Training completed. Run directory: {current_run_dir}. Best model: {trained_model_path}")
        except Exception as e:
            logger.exception("Error during training.")
            if 'results' in locals() and hasattr(results, 'save_dir'): current_run_dir = Path(results.save_dir)
            return # Exit script
    else: # EPOCHS == 0, skip training and load a pre-trained model
        logger.info(f"EPOCHS is set to 0. Skipping training.")
        model_path_to_load = Path(MODEL_NAME)
        if not model_path_to_load.is_file():
            logger.error(f"Model not found at '{model_path_to_load}' for prediction mode.")
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
            logger.info(f"Loaded model {trained_model_path}. Prediction outputs to: {current_run_dir}")

        except Exception as e:
            logger.exception(f"Error loading model '{model_path_to_load}'.")
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
    eval_model_instance = model_object_for_ops or (YOLO(trained_model_path) if trained_model_path and Path(trained_model_path).is_file() else None)
    if eval_model_instance:
        logger.info("--- Step 4: Evaluating Model ---")
        try:
            if val_rel_img_dirs_str:
                logger.info("--- Evaluating on Validation Set ---")
                eval_model_instance.val(data=str(generated_dataset_yaml_path), split='val', project=str(current_run_dir), name="eval_val_results")
            if test_rel_img_dirs_str:
                logger.info("--- Evaluating on Test Set ---")
                eval_model_instance.val(data=str(generated_dataset_yaml_path), split='test', project=str(current_run_dir), name="eval_test_results")
        except Exception as e:
            logger.exception("An error occurred during evaluation:")
    else:
        logger.error("No model available for evaluation.")

    # --- Step 5: Predict on ALL images, count objects, and save incorrect predictions ---
    logger.info("--- Step 5: Predicting and Comparing on ALL Images ---")
    predict_model_instance = model_object_for_ops or (YOLO(trained_model_path) if trained_model_path and Path(trained_model_path).is_file() else None)
    if not predict_model_instance:
        logger.error("No model available for prediction.")
    else:
        images_with_errors_count = 0 # Counter for images saved to incorrect_predictions
    
    # --- NEW: Initialize list to store data for CSV export ---
        prediction_data_for_csv = []

        try:
        # --- PREDICTION LOOP START ---
            for image_abs_path, label_abs_path in all_image_label_pairs:
                logger.info(f"--- Processing image for prediction: {image_abs_path.name} ---")
                image_to_draw_on = cv2.imread(str(image_abs_path))
                if image_to_draw_on is None:
                    logger.error(f"    Failed to read image {image_abs_path}. Skipping.")
                    continue

                pred_results_list = predict_model_instance.predict(source=image_to_draw_on.copy(), save=False, verbose=False, conf=DEFAULT_CONF_THRESHOLD)
                
                raw_predictions_data = [] # Store as list of dicts: {'xyxy', 'conf', 'cls'}
                if pred_results_list and pred_results_list[0].boxes:
                    r_boxes = pred_results_list[0].boxes
                    for i in range(len(r_boxes)):
                        raw_predictions_data.append({
                            'xyxy': r_boxes.xyxy[i].cpu().tolist(),
                            'conf': float(r_boxes.conf[i]),
                            'cls': int(r_boxes.cls[i])
                        })
                
                # --- Apply Per-Class Confidence Thresholds ---
                thresholded_predictions = []
                for pred_item in raw_predictions_data:
                    class_name = class_names_map.get(pred_item['cls'], "").lower().strip()
                    conf_thresh = PER_CLASS_CONF_THRESHOLDS.get(class_name, DEFAULT_CONF_THRESHOLD)
                    if pred_item['conf'] >= conf_thresh:
                        thresholded_predictions.append(pred_item)
                # --- End Per-Class Confidence Thresholding ---

                # --- Custom Inter-class Suppression (based on IoU and confidence) ---
                suppressed_flags = [False] * len(thresholded_predictions)
                for i in range(len(thresholded_predictions)):
                    if suppressed_flags[i]:
                        continue
                    for j in range(i + 1, len(thresholded_predictions)):
                        if suppressed_flags[j]:
                            continue
                        iou = calculate_iou(thresholded_predictions[i]['xyxy'], thresholded_predictions[j]['xyxy'])
                        if iou > IOU_SUPPRESSION_THRESHOLD: # If IoU is high, suppress the one with lower confidence
                            if thresholded_predictions[i]['conf'] >= thresholded_predictions[j]['conf']:
                                suppressed_flags[j] = True
                            else:
                                suppressed_flags[i] = True
                                break # Box i is suppressed, no need for it to suppress others

                filtered_predictions = []
                for idx, p_data in enumerate(thresholded_predictions):
                    if not suppressed_flags[idx]:
                        filtered_predictions.append(p_data) # Keep the original dict structure
                
                ground_truth_annotations = parse_yolo_annotations(label_abs_path)
                
                # --- Match predictions to GT to determine TP, FP, FN for this image ---
                num_gt_boxes = len(ground_truth_annotations)
                gt_boxes_for_matching = []
                img_h_for_coords, img_w_for_coords = image_to_draw_on.shape[:2] # Get image dimensions for coord conversion
                for gt_ann in ground_truth_annotations:
                    gt_class_id, x_c, y_c, w, h = gt_ann
                    x1_gt = (x_c - w / 2) * img_w_for_coords
                    y1_gt = (y_c - h / 2) * img_h_for_coords
                    x2_gt = (x_c + w / 2) * img_w_for_coords
                    y2_gt = (y_c + h / 2) * img_h_for_coords
                    gt_boxes_for_matching.append({'cls': gt_class_id, 'xyxy': [x1_gt, y1_gt, x2_gt, y2_gt], 'matched': False})

                num_true_positives_for_image = 0
                
                # Add correctness_status to each prediction in filtered_predictions
                # This list will be used for both logging/CSV and drawing
                final_predictions_with_status = []

                sorted_indices_preds = sorted(range(len(filtered_predictions)), key=lambda k: filtered_predictions[k]['conf'], reverse=True)

                for pred_idx_in_sorted_list in sorted_indices_preds:
                    pred_item = filtered_predictions[pred_idx_in_sorted_list] # Get the actual prediction item
                    pred_class_id = pred_item['cls']
                    pred_box_coords = pred_item['xyxy']
                    
                    current_pred_status = "Incorrect (FP)" # Default to False Positive
                    best_iou_for_this_pred = 0.0
                    best_gt_match_idx = -1

                    for gt_idx, gt_data in enumerate(gt_boxes_for_matching):
                        if not gt_data['matched'] and pred_class_id == gt_data['cls']: # If GT not matched and class is same
                            iou = calculate_iou(pred_box_coords, gt_data['xyxy'])
                            if iou > best_iou_for_this_pred:
                                best_iou_for_this_pred = iou
                                best_gt_match_idx = gt_idx
                    
                    if best_iou_for_this_pred > BOX_MATCHING_IOU_THRESHOLD:
                        if not gt_boxes_for_matching[best_gt_match_idx]['matched']: # Check again, just in case (though sorting preds helps)
                            current_pred_status = "Correct (TP)"
                            gt_boxes_for_matching[best_gt_match_idx]['matched'] = True
                            num_true_positives_for_image += 1
                    final_predictions_with_status.append({**pred_item, 'correctness_status': current_pred_status})
                
                # Sort back to original order if necessary, or just use final_predictions_with_status for logging/CSV
                # For simplicity, we'll log based on the order in final_predictions_with_status (which is confidence sorted)
                logger.info(f"  Detected objects in {image_abs_path.name} (after all filters & matching):")
                if final_predictions_with_status:
                    for pred_item_with_status in final_predictions_with_status:
                        predicted_class_name = class_names_map.get(pred_item_with_status['cls'], f"ID_{pred_item_with_status['cls']}")
                        probability = pred_item_with_status['conf']
                        box_correctness_status_for_csv = pred_item_with_status['correctness_status']
                        
                        logger.info(f"    - Class: {predicted_class_name}, Probability: {probability:.4f}, Box Correctness: {box_correctness_status_for_csv}")
                        prediction_data_for_csv.append([
                            image_abs_path.name, 
                            predicted_class_name, 
                            f"{probability:.4f}",
                            box_correctness_status_for_csv
                        ])
                else:
                    logger.info("    No objects detected after all filters.")
                
                # --- Determine if image is incorrect for saving ---
                num_false_positives = len(filtered_predictions) - num_true_positives_for_image
                num_false_negatives = num_gt_boxes - num_true_positives_for_image
                
                image_has_errors_for_saving = (num_false_positives > 0) or (num_false_negatives > 0)

                if image_has_errors_for_saving:
                    images_with_errors_count += 1
                    logger.info(f"    IMAGE FLAGGED: {image_abs_path.name} (TP: {num_true_positives_for_image}, FP: {num_false_positives}, FN: {num_false_negatives}).")

                    image_with_all_annotations = draw_all_annotations(
                        image_to_draw_on.copy(),
                        final_predictions_with_status, # Pass predictions with status for drawing
                        ground_truth_annotations,
                        class_names_map
                    )

                # Determine which folder to save the incorrect prediction in
                    target_dir = None
                    if image_abs_path in train_image_paths_set: target_dir = incorrect_preds_train_dir
                    elif image_abs_path in val_image_paths_set: target_dir = incorrect_preds_val_dir
                    elif image_abs_path in test_image_paths_set: target_dir = incorrect_preds_test_dir

                    if target_dir:
                        cv2.imwrite(str(target_dir / image_abs_path.name), image_with_all_annotations)
                        logger.info(f"      Saved comparison image to {target_dir / image_abs_path.name}")
            
            logger.info(f"Total images flagged with errors for saving: {images_with_errors_count}")

        # --- Write collected prediction data to CSV ---
            if current_run_dir and prediction_data_for_csv:
                csv_file_path = current_run_dir / PREDICTIONS_CSV_NAME
                try:
                    with open(csv_file_path, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(['Image Filename', 'Predicted Class', 'Probability', 'Box Correctness (Class & IoU)'])
                        csv_writer.writerows(prediction_data_for_csv)
                    logger.info(f"Prediction summary saved to CSV: {csv_file_path}")
                except Exception as e:
                    logger.error(f"Error writing prediction summary to CSV {csv_file_path}: {e}")
            elif not prediction_data_for_csv:
                 logger.info("No prediction data collected to write to CSV for this run yet.")
        # --- END CSV WRITING ---


        except Exception as e:
            logger.exception("Error during prediction loop.")

    # --- Final Log Copy ---
    try:
        pass  # No additional code needed here, but try is required for finally to work
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
