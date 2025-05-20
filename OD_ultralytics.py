import os
import glob
import yaml
import random
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# --- Configuration ---
# TODO: Update this path if your main dataset directory is different
INPUTS_DIR = "/mnt/c/Work/Repos/MyProjects/DeepLearning/CoinsUltralytics/Data/CoinCount.v30i.yolov5pytorch"  # Base directory containing variant subfolders (e.g., hard, easy) and data.yaml
IMAGE_SUBDIR_BASENAME = "images"  # Basename of the image subdirectory within each variant folder
LABEL_SUBDIR_BASENAME = "labels"  # Basename of the label subdirectory within each variant folder
ORIGINAL_DATA_YAML_NAME = "data.yaml" # Name of your existing data.yaml with class names, located in INPUTS_DIR

OUTPUT_DIR = "yolo_experiment_output"      # Directory to save YAML, results, etc.
# Name for the YAML file this script will generate for YOLO training
DATASET_YAML_NAME = "custom_dataset_for_training.yaml" 

MODEL_NAME = "yolov8n.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.
EPOCHS = 50 # Number of training epochs
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
    'fliplr': 0.0,   # Image flip left-right (probability) # Kept original value from user's file
    'mosaic': 1.0,   # Mosaic augmentation (probability)
    'mixup': 0.1,    # Mixup augmentation (probability)
    # 'copy_paste': 0.0 # Copy-paste augmentation (probability), typically for segmentation
}

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
        print(f"Error: Root dataset directory '{inputs_dir_pathobj}' not found.")
        return image_label_pairs, valid_label_dirs_for_class_scan

    print(f"Scanning for dataset variants in: {inputs_dir_pathobj}")
    for variant_dir in inputs_dir_pathobj.iterdir():
        if variant_dir.is_dir(): # e.g., "hard", "easy"
            image_dir = variant_dir / image_subdir_basename
            label_dir = variant_dir / label_subdir_basename

            if image_dir.is_dir() and label_dir.is_dir():
                print(f"  Processing variant: '{variant_dir.name}'")
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
                            print(f"    Warning: Label file not found for image '{img_path.name}' in '{label_dir}'. Skipping this image.")
            # else:
                # print(f"  Skipping '{variant_dir.name}': Missing '{image_subdir_basename}' or '{label_subdir_basename}' directory.")
                
    if not image_label_pairs:
        print(f"Warning: No valid image-label pairs found across all variants in '{inputs_dir_pathobj}'.")
        
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
        raise ValueError("Train, validation, and test ratios must sum to approximately 1.0.")

    random.seed(seed) # for reproducible shuffles
    random.shuffle(image_label_pairs)

    total_samples = len(image_label_pairs)
    train_end_idx = int(total_samples * train_ratio)
    val_end_idx = train_end_idx + int(total_samples * val_ratio)

    train_pairs = image_label_pairs[:train_end_idx]
    val_pairs = image_label_pairs[train_end_idx:val_end_idx]
    test_pairs = image_label_pairs[val_end_idx:]
    
    print(f"Data split: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test samples.")
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
        print("Warning (get_unique_class_ids): No label directories provided for scanning.")
        return []

    for label_dir_path in list_of_label_dir_paths:
        if not label_dir_path.is_dir(): # Check if the provided path is a directory
            print(f"Warning: Provided label path '{label_dir_path}' is not a directory. Skipping.")
            continue

        for label_file in label_dir_path.glob('*.txt'): # Use glob on Path object
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts: # Ensure line is not empty
                            unique_ids.add(int(parts[0])) # Add the class ID (first element)
            except Exception as e:
                print(f"Error reading label file {label_file}: {e}")
    
    if not unique_ids:
        print(f"Warning: No class IDs found across all scanned label directories. Check your label files.")
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
        print(f"Info: Original data YAML file not found at '{original_yaml_path_obj}'.")
        return None
    try:
        with open(original_yaml_path_obj, 'r') as f:
            data = yaml.safe_load(f) # Load YAML content
        # Check if 'names' key exists and is a list
        if data and 'names' in data and isinstance(data['names'], list):
            return data['names'] # Return the list of names
        else:
            print(f"Warning: 'names' key not found or is not a list in '{original_yaml_path_obj}'.")
            return None
    except Exception as e:
        print(f"Error reading original data YAML file {original_yaml_path_obj}: {e}")
        return None

def create_dataset_yaml(dataset_root_abs_path_str, 
                        train_rel_img_paths, 
                        val_rel_img_paths,
                        test_rel_img_paths,
                        class_names_dict, 
                        num_classes_val, 
                        output_yaml_path_obj):
    """
    Creates the dataset.yaml file required by YOLO for training.
    'train', 'val', and 'test' will be lists of relative image directory paths.
    Args:
        dataset_root_abs_path_str (str): Absolute path to the dataset root directory (INPUTS_DIR).
        train_rel_img_paths (list): List of strings, relative paths for training images.
        val_rel_img_paths (list): List of strings, relative paths for validation images.
        test_rel_img_paths (list): List of strings, relative paths for test images.
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
            print(f"Warning (create_dataset_yaml): Class ID {class_id} ('{name}') is out of range for nc={num_classes_val}.")
    
    # Fill any gaps if class_names_dict didn't cover all IDs up to num_classes_val
    # (e.g., if num_classes was determined by max_id from labels and some intermediate IDs had no names)
    for i in range(num_classes_val):
        if not names_list_for_yaml[i]: # If the name is still an empty string
             names_list_for_yaml[i] = f"class_{i}" # Fallback to generic name

    data = {
        'path': dataset_root_abs_path_str,
        'train': [str(p) for p in train_rel_img_paths], 
        'val': [str(p) for p in val_rel_img_paths],   
        'test': [str(p) for p in test_rel_img_paths], 
        'nc': num_classes_val,
        'names': names_list_for_yaml 
    }

    try:
        with open(output_yaml_path_obj, 'w') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=None) # Use default_flow_style for better readability
        print(f"Successfully created dataset YAML: {output_yaml_path_obj}")
        print("YAML content:")
        print(yaml.dump(data, sort_keys=False, default_flow_style=None))
        print("\nImportant: Please verify the 'path', 'train', 'val', and 'test' entries in the generated YAML file.")
        print(f"YOLO will look for labels in directories relative to the image paths listed.")
        print(f"(e.g., if an image path in YAML is 'variant_folder/{IMAGE_SUBDIR_BASENAME}/image.jpg',")
        print(f" labels are expected in 'variant_folder/{LABEL_SUBDIR_BASENAME}/image.txt' relative to 'path')")

    except Exception as e:
        print(f"Error writing YAML file {output_yaml_path_obj}: {e}")


def main():
    """
    Main function to run the YOLO training and evaluation pipeline.
    """
    # --- Basic Path Setup ---
    output_path = Path(OUTPUT_DIR) # Original variable name from OD_ultralytics.py context
    output_path.mkdir(parents=True, exist_ok=True)

    inputs_dir_pathobj = Path(INPUTS_DIR).resolve() # Resolve INPUTS_DIR to absolute path
    # Path to the original data.yaml which might contain class names
    original_data_yaml_abs_path = (inputs_dir_pathobj / ORIGINAL_DATA_YAML_NAME).resolve()

    print(f"\nUsing dataset base directory: {inputs_dir_pathobj}")
    print(f"Image subdirectory basename: '{IMAGE_SUBDIR_BASENAME}'")
    print(f"Label subdirectory basename: '{LABEL_SUBDIR_BASENAME}'")
    print(f"Attempting to load class names from: {original_data_yaml_abs_path}")

    # --- Step 0: Discover all image-label pairs and label directories ---
    print("\n--- Step 0: Discovering and Pairing Image-Label Data ---")
    all_image_label_pairs, all_label_dirs_abs_for_class_scan = discover_and_pair_image_labels(
        inputs_dir_pathobj, IMAGE_SUBDIR_BASENAME, LABEL_SUBDIR_BASENAME
    )

    if not all_image_label_pairs:
        print(f"Error: No image-label pairs found. Cannot proceed with training.")
        return
    
    # --- Step 1: Determine Class Names and Number of Classes ---
    print("\n--- Step 1: Determining Class Names and Number ---")
    class_names_map = {}  # This will be the dict {id: name}
    num_classes = 0       # This will be nc (number of classes)

    # Try to load class names from the user-provided data.yaml
    names_list_from_file = load_class_names_from_original_yaml(original_data_yaml_abs_path)

    if names_list_from_file: # Successfully loaded names from data.yaml
        print(f"Successfully loaded {len(names_list_from_file)} class names from '{original_data_yaml_abs_path}'.")
        class_names_map = {i: str(name) for i, name in enumerate(names_list_from_file)}
        num_classes = len(names_list_from_file)

        # Optional: Validate against actual labels found, only if label dirs were discovered
        if all_label_dirs_abs_for_class_scan: 
            class_ids_in_labels = get_unique_class_ids(all_label_dirs_abs_for_class_scan) 
            if class_ids_in_labels: # If any class IDs were actually found in labels
                max_id_in_labels = max(class_ids_in_labels)
                if max_id_in_labels >= num_classes:
                    print(f"Warning: Max class ID in labels ({max_id_in_labels}) is >= number of classes defined in '{original_data_yaml_abs_path}' ({num_classes}).")
                    print("         This might lead to errors or misinterpretations during training if labels use IDs outside the defined names.")
    else: # Fallback: data.yaml not found or 'names' key missing/invalid
        print(f"Warning: Could not load 'names' list from '{original_data_yaml_abs_path}'.")
        if not all_label_dirs_abs_for_class_scan: 
            print("Error: No label directories found to fall back on for class discovery. Cannot proceed.")
            return
            
        print("         Attempting to generate generic class names based on IDs found in label files across all variants.")
        class_ids_in_labels = get_unique_class_ids(all_label_dirs_abs_for_class_scan) 
        
        if not class_ids_in_labels: # No labels found, and no data.yaml names
            print("Error: No class IDs found in any label files, and could not load names from data.yaml. Cannot proceed.")
            return # Exit
        
        max_id_in_labels = max(class_ids_in_labels)
        num_classes = max_id_in_labels + 1 # nc is count, so max_id + 1 for 0-indexed
        class_names_map = {i: f'class_{i}' for i in range(num_classes)} # Create names for all IDs up to max_id
        print(f"         Generated {num_classes} generic class names based on labels (up to max ID {max_id_in_labels}).")

    print(f"Final effective number of classes (nc): {num_classes}")
    print(f"Final class names map: {class_names_map}")
    # Allow nc=0 if explicitly from an empty 'names' list in data.yaml, otherwise error.
    if num_classes == 0 and not (names_list_from_file is not None and len(names_list_from_file) == 0) :
        print("Error: Number of classes is zero. Cannot train. Please check your data.yaml or label files.")
        return

    # --- Step 1.5: Split Data into Train, Validation, Test ---
    print("\n--- Step 1.5: Splitting Data ---")
    train_pairs, val_pairs, test_pairs = split_data(
        all_image_label_pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Extract just the image paths for the YAML file, making them relative to INPUTS_DIR
    train_rel_img_paths = [img_path.relative_to(inputs_dir_pathobj) for img_path, _ in train_pairs]
    val_rel_img_paths = [img_path.relative_to(inputs_dir_pathobj) for img_path, _ in val_pairs]
    test_rel_img_paths = [img_path.relative_to(inputs_dir_pathobj) for img_path, _ in test_pairs]

    # --- Step 2: Create dataset.yaml for Training ---
    print("\n--- Step 2: Creating Dataset YAML for Training ---")
    generated_dataset_yaml_path = output_path / DATASET_YAML_NAME 
    
    create_dataset_yaml(
        dataset_root_abs_path_str=str(inputs_dir_pathobj), 
        train_rel_img_paths=train_rel_img_paths,
        val_rel_img_paths=val_rel_img_paths,
        test_rel_img_paths=test_rel_img_paths,
        class_names_dict=class_names_map,
        num_classes_val=num_classes,
        output_yaml_path_obj=generated_dataset_yaml_path
    )
    
    if not generated_dataset_yaml_path.exists():
        print(f"Failed to create {generated_dataset_yaml_path}. Exiting.")
        return

    # --- Step 3: Train the model ---
    # Using original variable names trained_model_path and model (for the loaded/trained object)
    print(f"\n--- Step 3: Training Model ({MODEL_NAME}) ---") 
    trained_model_path = None # Path to best.pt
    model_object_after_training = None # To store the model object after training

    try:
        model = YOLO(MODEL_NAME)  # Load a pretrained model
        model_object_after_training = model # Store initial for fallback if training fails early
        print(f"Training with model: {MODEL_NAME}, epochs: {EPOCHS}, img_size: {IMG_SIZE}")
        print(f"Dataset YAML: {generated_dataset_yaml_path}")
        print(f"Using Augmentations: {AUGMENTATION_PARAMS}")
        
        # Using Path(MODEL_NAME).stem to get 'yolov8n' from 'yolov8n.pt' for the run name
        training_run_name = f"{Path(MODEL_NAME).stem}_custom_training" # Kept naming convention
        results = model.train(
            data=str(generated_dataset_yaml_path),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            project=str(output_path / "training_runs"), # Save runs here
            name=training_run_name, 
            **AUGMENTATION_PARAMS # Unpack augmentation parameters
        )
        print("Training completed.")
        model_object_after_training = model # Update with trained model object
        
        # Determine the path to the best model (logic from previous correct versions)
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and model.trainer.best and Path(model.trainer.best).exists():
            trained_model_path = str(Path(model.trainer.best).resolve())
            print(f"Best model saved at (from model.trainer.best): {trained_model_path}")
        else:
            # Fallback: Construct path based on expected output structure
            potential_best_path = output_path / "training_runs" / training_run_name / "weights" / "best.pt"
            if potential_best_path.exists():
                trained_model_path = str(potential_best_path.resolve())
                print(f"Found best model at expected path: {trained_model_path}")
            elif hasattr(model, 'ckpt_path') and model.ckpt_path and Path(model.ckpt_path).exists(): # Check last checkpoint
                 trained_model_path = str(Path(model.ckpt_path).resolve())
                 print(f"Using last checkpoint as trained model path: {trained_model_path}")
            else:
                print("Could not determine exact path to best.pt. Will use model object in memory if available.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure your dataset YAML is correctly configured and points to valid image/label paths.")
        print("Also, check if the model name (MODEL_NAME) is correct and downloadable by Ultralytics, or if it exists at the specified path.")
        import traceback # For more detailed error for debugging
        traceback.print_exc()
        return

    # --- Step 4: Evaluate the model ---
    print("\n--- Step 4: Evaluating Model ---")
    eval_model_instance = None # Variable to hold the model instance for evaluation
    if trained_model_path and Path(trained_model_path).exists(): 
        print(f"Loading best model from {trained_model_path} for evaluation.")
        eval_model_instance = YOLO(trained_model_path)
    elif model_object_after_training: # Use the model object from training if path not found
        print("Evaluating with the model object from training (in memory).")
        eval_model_instance = model_object_after_training # Use the stored model object
    else: # Fallback if neither path nor object is available
        print("Error: No trained model available for evaluation (path or object).")
        return 
        
    try:
        print("\n--- Evaluating on Validation Set ---")
        val_metrics = eval_model_instance.val(data=str(generated_dataset_yaml_path), split='val') 
        print("Validation metrics:")
        if hasattr(val_metrics, 'box') and hasattr(val_metrics.box, 'map'): 
            print(f"  mAP50-95 (val): {val_metrics.box.map:.4f}")
            print(f"  mAP50 (val): {val_metrics.box.map50:.4f}")
            print(f"  mAP75 (val): {val_metrics.box.map75:.4f}")
        else:
            print("  Could not retrieve detailed mAP scores for validation set directly.")

        if test_rel_img_paths: # Only evaluate on test set if it's not empty
            print("\n--- Evaluating on Test Set ---")
            test_metrics = eval_model_instance.val(data=str(generated_dataset_yaml_path), split='test') 
            print("Test metrics:")
            if hasattr(test_metrics, 'box') and hasattr(test_metrics.box, 'map'): 
                print(f"  mAP50-95 (test): {test_metrics.box.map:.4f}")
                print(f"  mAP50 (test): {test_metrics.box.map50:.4f}")
                print(f"  mAP75 (test): {test_metrics.box.map75:.4f}")
            else:
                print("  Could not retrieve detailed mAP scores for test set directly.")
        else:
            print("\nSkipping evaluation on test set as it is empty.")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback 
        traceback.print_exc()

    # --- Step 5: Predict on images and count objects ---
    print("\n--- Step 5: Predicting and Counting Objects on TEST SET ---")
    predict_model_instance = None # Variable to hold the model instance for prediction
    if trained_model_path and Path(trained_model_path).exists():
        print(f"Loading best model from {trained_model_path} for prediction.")
        predict_model_instance = YOLO(trained_model_path)
    elif model_object_after_training:
        print("Predicting with the model object from training (in memory).")
        predict_model_instance = model_object_after_training # Use the stored model object
    else:
        print("Error: No trained model available for prediction (path or object).")
        return

    # Collect all images from all discovered image directories
    test_image_files_to_predict = [img_path for img_path, _ in test_pairs]

    if not test_image_files_to_predict:
        print(f"No images found in the generated test set for prediction.")
        return

    print(f"Found {len(test_image_files_to_predict)} images in the test set for prediction.")
    
    total_class_counts_test_set = Counter()
    
    try:
        for image_file_pathobj in test_image_files_to_predict: 
            # Show relative path from INPUTS_DIR for cleaner logging
            print(f"  Predicting on (test image): {image_file_pathobj.relative_to(inputs_dir_pathobj)}") 
            # Set verbose=False for predict to reduce console spam for many images
            pred_results = predict_model_instance.predict(source=str(image_file_pathobj), save=False, verbose=False) 
            
            if pred_results and len(pred_results) > 0:
                r = pred_results[0] # Get the Results object for the first (and only) image
                detected_class_ids_float = r.boxes.cls.tolist() # List of float class IDs
                
                if detected_class_ids_float:
                    # Convert float IDs to int for Counter and dictionary lookup
                    current_image_counts_int_ids = Counter(int(cls_id) for cls_id in detected_class_ids_float)
                    
                    # Prepare human-readable counts for printing using class_names_map
                    readable_counts = {
                        class_names_map.get(int_cls_id, f"unknown_id_{int_cls_id}"): count 
                        for int_cls_id, count in current_image_counts_int_ids.items()
                    }
                    print(f"    Detections in {image_file_pathobj.name}: {readable_counts}")
                    total_class_counts_test_set.update(current_image_counts_int_ids) 
                else:
                    print(f"    No objects detected in {image_file_pathobj.name}")
            else:
                print(f"    No results from prediction for {image_file_pathobj.name}")

        print("\n--- Final Object Counts on TEST SET ---")
        if total_class_counts_test_set:
            # Iterate sorted by class ID for consistent output
            for class_id_int, count in sorted(total_class_counts_test_set.items()): 
                # Use the class_names_map established in Step 1
                class_name_str = class_names_map.get(class_id_int, f"unknown_id_{class_id_int}")
                print(f"  Class '{class_name_str}' (ID: {class_id_int}): {count} occurrences")
        else:
            print("No objects were detected in any of the test set images.")
    except Exception as e:
        print(f"An error occurred during prediction and counting on the test set: {e}")
        import traceback 
        traceback.print_exc()

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Before running, ensure INPUTS_DIR is the top-level directory containing
    # variant subfolders (e.g., "hard", "easy") AND your main data.yaml.
    # Ensure IMAGE_SUBDIR_BASENAME (e.g., "images") and LABEL_SUBDIR_BASENAME (e.g., "labels")
    # correctly name the subfolders within each variant.
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
    # The script will generate a `custom_dataset_for_training.yaml` in OUTPUT_DIR.
    main()
