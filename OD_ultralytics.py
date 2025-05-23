import os
import glob
import yaml
import random
import shutil # For copying files
import logging # For logging to file and console
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

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

MODEL_NAME = "yolov8n.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
MODEL_NAME = "yolo_experiment_output/training_runs/yolov8n_custom_training3/weights/best.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
EPOCHS = 0 # Number of training epochs
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
        logger.warning("Cannot copy log file: Training run directory not specified or not created.")
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

def parse_label_file(label_file_path):
    """
    Parses a YOLO format label file and returns a Counter of class IDs.
    Args:
        label_file_path (Path): Path to the .txt label file.
    Returns:
        Counter: A Counter object with class_id: count.
    """
    class_counts = Counter()
    if not label_file_path.is_file():
        logger.warning(f"(parse_label_file): Label file not found at {label_file_path}")
        return class_counts
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    except Exception as e:
        logger.error(f"Error parsing label file {label_file_path}: {e}")
    return class_counts

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
        logger.error("No image-label pairs found. Cannot proceed with training.") 
        return
    
    # --- Step 1: Determine Class Names and Number of Classes ---
    logger.info("--- Step 1: Determining Class Names and Number ---") 
    class_names_map = {}  # This will be the dict {id: name}
    num_classes = 0       # This will be nc (number of classes)

    # Try to load class names from the user-provided data.yaml
    names_list_from_file = load_class_names_from_original_yaml(original_data_yaml_abs_path)

    if names_list_from_file: # Successfully loaded names from data.yaml
        logger.info(f"Successfully loaded {len(names_list_from_file)} class names from '{original_data_yaml_abs_path}'.")
        class_names_map = {i: str(name) for i, name in enumerate(names_list_from_file)}
        num_classes = len(names_list_from_file)

        # Optional: Validate against actual labels found, only if label dirs were discovered
        if all_label_dirs_abs_for_class_scan: 
            class_ids_in_labels = get_unique_class_ids(all_label_dirs_abs_for_class_scan) 
            if class_ids_in_labels: # If any class IDs were actually found in labels
                max_id_in_labels = max(class_ids_in_labels)
                if max_id_in_labels >= num_classes:
                    logger.warning(f"Max class ID in labels ({max_id_in_labels}) is >= number of classes defined in '{original_data_yaml_abs_path}' ({num_classes}).")
                    logger.warning("         This might lead to errors or misinterpretations during training if labels use IDs outside the defined names.")
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
    # Allow nc=0 if explicitly from an empty 'names' list in data.yaml, otherwise error.
    if num_classes == 0 and not (names_list_from_file is not None and len(names_list_from_file) == 0) :
        logger.error("Number of classes is zero. Cannot train. Please check your data.yaml or label files.")
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


    # --- Step 2: Create dataset.yaml for Training ---
    logger.info("--- Step 2: Creating Dataset YAML for Training ---") 
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

    # --- Step 3: Train the model ---
    # Using original variable names trained_model_path and model (for the loaded/trained object)
    logger.info(f"--- Step 3: Checking for Training Requirement ---")
    trained_model_path = None # Path to best.pt
    model_object_after_training = None # To store the model object after training
    current_training_run_actual_dir = None # To store the actual path of the current training run

    # --- Conditional Training Block ---
    # If EPOCHS > 0, the script will train the model.
    # If EPOCHS == 0, it will skip training and load the specified MODEL_NAME for prediction.
    if EPOCHS > 0:
        logger.info(f"EPOCHS set to {EPOCHS}. Proceeding with model training.")
        try:
            model = YOLO(MODEL_NAME)  # Load a pretrained model
            model_object_after_training = model # Store initial for fallback if training fails early
            logger.info(f"Training with model: {MODEL_NAME}, epochs: {EPOCHS}, img_size: {IMG_SIZE}")
            logger.info(f"Dataset YAML: {generated_dataset_yaml_path}")
            logger.info(f"Using Augmentations: {AUGMENTATION_PARAMS}")

            project_base_dir = output_path / "training_runs"
            training_run_name = f"{Path(MODEL_NAME).stem}_custom_training"

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
            model_object_after_training = model
            current_training_run_actual_dir = Path(results.save_dir) # Get the actual save directory
            logger.info(f"Training run artifacts saved to: {current_training_run_actual_dir}")

            if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and model.trainer.best and Path(model.trainer.best).exists():
                trained_model_path = str(Path(model.trainer.best).resolve())
                logger.info(f"Best model saved at (from model.trainer.best): {trained_model_path}")
            elif current_training_run_actual_dir and (current_training_run_actual_dir / "weights" / "best.pt").exists():
                trained_model_path = str((current_training_run_actual_dir / "weights" / "best.pt").resolve())
                logger.info(f"Found best model at expected path: {trained_model_path}")
            elif hasattr(model, 'ckpt_path') and model.ckpt_path and Path(model.ckpt_path).exists():
                 trained_model_path = str(Path(model.ckpt_path).resolve())
                 logger.info(f"Using last checkpoint as trained model path: {trained_model_path}")
            else:
                logger.info("Could not determine exact path to best.pt. Will use model object in memory if available.")

        except Exception as e:
            logger.exception(f"An error occurred during training:")
            logger.error("Please ensure your dataset YAML is correctly configured and points to valid image/label paths.")
            logger.error("Also, check if the model name (MODEL_NAME) is correct and downloadable by Ultralytics, or if it exists at the specified path.")
            # current_training_run_actual_dir might not be set if model.train() itself fails early
            # However, if 'results' object exists (meaning train started enough to return it)
            if 'results' in locals() and hasattr(results, 'save_dir') and Path(results.save_dir).is_dir():
                current_training_run_actual_dir = Path(results.save_dir)
            # The finally block will attempt to copy the log using whatever current_training_run_actual_dir is.
            return
    else: # New block for skipping training when EPOCHS == 0
        logger.info(f"EPOCHS is set to 0. Skipping training.")
        logger.info(f"Attempting to load model '{MODEL_NAME}' for prediction and evaluation.")
        try:
            model_path_obj = Path(MODEL_NAME)
            if not model_path_obj.exists():
                logger.error(f"Model not found at '{MODEL_NAME}'. When EPOCHS is 0, MODEL_NAME must be a valid path to a trained model file (e.g., 'best.pt').")
                # The finally block will run, but with current_training_run_actual_dir as None, so no log copy will happen.
                return # Exit the script

            # Load the specified model
            model_object_after_training = YOLO(model_path_obj)
            trained_model_path = str(model_path_obj.resolve()) # Store the absolute path to the model
            logger.info(f"Successfully loaded model from: {trained_model_path}")

            # Create a directory for this prediction-only run to store logs and incorrect predictions, maintaining script structure.
            prediction_run_name = f"{model_path_obj.stem}_prediction_run"
            # Place it in a 'prediction_runs' subfolder to distinguish from training runs
            current_training_run_actual_dir = output_path / "prediction_runs" / prediction_run_name
            current_training_run_actual_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Outputs for this prediction-only run will be saved in: {current_training_run_actual_dir}")

        except Exception as e:
            logger.exception("An error occurred while loading the model for prediction:")
            # The finally block will run, but with current_training_run_actual_dir as None, so no log copy will happen.
            return


    # --- Setup for incorrect predictions directory *after* training, inside the current run's folder ---
    incorrect_preds_train_dir = None
    incorrect_preds_val_dir = None
    incorrect_preds_test_dir = None
    if current_training_run_actual_dir: # Proceed only if training run directory is known
        incorrect_preds_base_dir = current_training_run_actual_dir / INCORRECT_PREDICTIONS_SUBDIR
        incorrect_preds_train_dir = incorrect_preds_base_dir / "train"
        incorrect_preds_val_dir = incorrect_preds_base_dir / "validation"
        incorrect_preds_test_dir = incorrect_preds_base_dir / "test"

        logger.info(f"Saving incorrect predictions to subfolders within: {incorrect_preds_base_dir}")
        incorrect_preds_train_dir.mkdir(parents=True, exist_ok=True)
        incorrect_preds_val_dir.mkdir(parents=True, exist_ok=True)
        incorrect_preds_test_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning("Training run directory not identified. Incorrect predictions will not be saved to a run-specific folder.")


    # --- Step 4: Evaluate the model on Validation and Test Sets ---
    logger.info("--- Step 4: Evaluating Model ---") 
    eval_model_instance = None 
    if trained_model_path and Path(trained_model_path).exists(): 
        logger.info(f"Loading best model from {trained_model_path} for evaluation.")
        eval_model_instance = YOLO(trained_model_path)
    elif model_object_after_training: # Use the model object from training if path not found
        logger.info("Evaluating with the model object from training (in memory).")
        eval_model_instance = model_object_after_training # Use the stored model object
    else: # Fallback if neither path nor object is available
        logger.error("No trained model available for evaluation (path or object).")
        # Log copy handled by finally block
        return 
        
    try:
        logger.info("--- Evaluating on Validation Set ---") 
        # Use the same generated YAML for validation, Ultralytics will use the 'val' key from it
        val_metrics = eval_model_instance.val(data=str(generated_dataset_yaml_path), split='val') 
        logger.info("Validation metrics:")
        if hasattr(val_metrics, 'box') and hasattr(val_metrics.box, 'map'): # Accessing metrics correctly
            logger.info(f"  mAP50-95 (val): {val_metrics.box.map:.4f}")
            logger.info(f"  mAP50 (val): {val_metrics.box.map50:.4f}")
            logger.info(f"  mAP75 (val): {val_metrics.box.map75:.4f}")
        else:
            logger.info("  Could not retrieve detailed mAP scores for validation set directly.")

        # Evaluate on the test set if it's not empty
        if test_rel_img_dirs_str: # Check if the list of test directories is not empty
            logger.info("--- Evaluating on Test Set ---") 
            # Ultralytics will use the 'test' key from the YAML
            test_metrics = eval_model_instance.val(data=str(generated_dataset_yaml_path), split='test') 
            logger.info("Test metrics:")
            if hasattr(test_metrics, 'box') and hasattr(test_metrics.box, 'map'): 
                logger.info(f"  mAP50-95 (test): {test_metrics.box.map:.4f}")
                logger.info(f"  mAP50 (test): {test_metrics.box.map50:.4f}")
                logger.info(f"  mAP75 (test): {test_metrics.box.map75:.4f}")
            else:
                logger.info("  Could not retrieve detailed mAP scores for test set directly.")
        else:
            logger.info("Skipping evaluation on test set as it is empty (no test directories found/specified).") 

    except Exception as e:
        logger.exception("An error occurred during evaluation:") 


    # --- Step 5: Predict on ALL images, count objects, and save incorrect predictions ---
    logger.info("--- Step 5: Predicting on ALL Images, Counting, and Saving Incorrect ---") 
    predict_model_instance = None # Variable to hold the model instance for prediction
    if trained_model_path and Path(trained_model_path).exists():
        logger.info(f"Loading best model from {trained_model_path} for prediction.")
        predict_model_instance = YOLO(trained_model_path)
    elif model_object_after_training:
        logger.info("Predicting with the model object from training (in memory).")
        predict_model_instance = model_object_after_training # Use the stored model object
    else:
        logger.error("No trained model available for prediction (path or object).")
        # Log copy handled by finally block
        return

    if not all_image_label_pairs: # Should have been caught earlier, but good check
        logger.warning("No images found in any of the discovered image subdirectories for prediction.") 
        # Log copy handled by finally block
        return

    logger.info(f"Found {len(all_image_label_pairs)} total image-label pairs for prediction across all sets.")
    
    total_class_counts_all_images = Counter()
    incorrect_prediction_count = 0
    
    try:
        for image_abs_path, label_abs_path in all_image_label_pairs: # Iterate over all pairs
            # Determine which set this image belongs to for saving incorrect predictions
            current_set_incorrect_target_dir = None # Renamed for clarity
            if incorrect_preds_train_dir and incorrect_preds_val_dir and incorrect_preds_test_dir : 
                if image_abs_path in train_image_paths_set:
                    current_set_incorrect_target_dir = incorrect_preds_train_dir
                elif image_abs_path in val_image_paths_set:
                    current_set_incorrect_target_dir = incorrect_preds_val_dir
                elif image_abs_path in test_image_paths_set:
                    current_set_incorrect_target_dir = incorrect_preds_test_dir
                
            # Show relative path from INPUTS_DIR for cleaner logging
            logger.info(f"  Predicting on: {image_abs_path.relative_to(inputs_dir_pathobj)}") 
            pred_results = predict_model_instance.predict(source=str(image_abs_path), save=False, verbose=False) 
            
            predicted_class_counts = Counter()
            if pred_results and len(pred_results) > 0:
                r = pred_results[0] 
                detected_class_ids_float = r.boxes.cls.tolist() 
                
                if detected_class_ids_float:
                    predicted_class_counts = Counter(int(cls_id) for cls_id in detected_class_ids_float)
                    # Update overall counts for all images
                    total_class_counts_all_images.update(predicted_class_counts) 
                # else: No objects predicted
            # else: No results object from prediction
            
            # Load ground truth labels for this image
            ground_truth_class_counts = parse_label_file(label_abs_path)

            # Compare predicted counts with ground truth counts
            if predicted_class_counts != ground_truth_class_counts:
                incorrect_prediction_count += 1
                logger.info(f"    INCORRECT prediction for {image_abs_path.name}.")
                logger.info(f"      GT counts: {dict(ground_truth_class_counts)}")
                logger.info(f"      Pred counts: {dict(predicted_class_counts)}")
                if current_set_incorrect_target_dir: 
                    try:
                        destination_path = current_set_incorrect_target_dir / image_abs_path.name
                        shutil.copyfile(image_abs_path, destination_path) 
                        logger.info(f"      Copied to {destination_path}")
                    except PermissionError as perm_e:
                        # If a PermissionError occurs, check if the file was still created (common in WSL)
                        if destination_path.exists() and destination_path.stat().st_size > 0:
                            logger.warning(f"      PermissionError occurred while copying {image_abs_path.name}, but file appears to exist at {destination_path}. Assuming data copied successfully despite metadata/permission error: {perm_e}")
                        else:
                            logger.error(f"      Failed to copy incorrect image {image_abs_path.name} due to PermissionError, and file does not exist or is empty: {perm_e}")
                    except Exception as copy_e: # Catch other potential errors during copy
                        logger.error(f"      Error copying incorrect image {image_abs_path.name} (non-PermissionError): {copy_e}")
            
        logger.info(f"Total incorrect predictions (by class counts): {incorrect_prediction_count} out of {len(all_image_label_pairs)} images.") 
        logger.info("--- Final Object Counts Across ALL IMAGES (Train, Val, Test) ---") 
        if total_class_counts_all_images:
            # Iterate sorted by class ID for consistent output
            for class_id_int, count in sorted(total_class_counts_all_images.items()): 
                # Use the class_names_map established in Step 1
                class_name_str = class_names_map.get(class_id_int, f"unknown_id_{class_id_int}")
                logger.info(f"  Class '{class_name_str}' (ID: {class_id_int}): {count} occurrences") 
        else:
            logger.info("No objects were detected in any of the images across all sets.") 

    except Exception as e:
        logger.exception("An error occurred during prediction and counting:") 
    
    # --- Final step: Copy the initial log file to the run directory if it exists ---
    finally: # Ensure this runs even if there are exceptions in prediction step or earlier returns
        copy_log_to_run_directory(initial_log_file_path, current_training_run_actual_dir, LOG_FILE_NAME)


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Before running, ensure INPUTS_DIR is the top-level directory containing
    # variant subfolders (e.g., "hard", "easy") AND your main data.yaml.
    # Ensure IMAGE_SUBDIR_BASENAME (e.g., "images") and LABEL_SUBDIR_BASENAME (e.g., "labels")
    # correctly name the subfolders within each variant.
    # Set TRAIN_RATIO, VAL_RATIO, TEST_RATIO at the top.
    #
    # Example expected structure:
    # INPUTS_DIR/  (e.g., /path/to/CoinCount.v30i.yolov5pytorch)
    #  ├── data.yaml (ORIGINAL_DATA_YAML_NAME)
    #  ├── hard/  
    #  │   ├── images/  (IMAGE_SUBDIR_BASENAME)
    #  │   │   └── img1.jpg
    #  │   └── labels/  (LABEL_SUBDIR_BASENAME)
    #  │       └── img1.txt
    #  ├── easy/
    #  │   ├── images/
    #  │   │   └── img2.jpg
    #  │   └── labels/
    #  │       └── img2.txt
    #  └── ... (other variant folders with similar images/labels structure)
    #
    # The script will generate a `custom_dataset_for_training.yaml` in OUTPUT_DIR
    # with distinct 'train', 'val', and 'test' lists of image DIRECTORY paths.
    main()
