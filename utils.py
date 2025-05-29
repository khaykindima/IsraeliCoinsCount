import os
import yaml
import random
import shutil
import logging
from pathlib import Path
import cv2 # For drawing
import numpy as np # For drawing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from detector import CoinDetector


def draw_ground_truth_boxes(image_np, ground_truths_list, class_names_map, config_module):
    """
    Draws ground truth bounding boxes on an image for visualization.

    Args:
        image_np (np.ndarray): The image to draw on.
        ground_truths_list (list): A list of ground truth dictionaries.
                                   Each dict: {'cls': class_id, 'xyxy': [x1, y1, x2, y2]}
        class_names_map (dict): A map from class ID to class name.
        config_module (module): The configuration module for styling.
    """
    img_to_draw_on = image_np.copy()
    font = config_module.FONT_FACE
    box_color_map = config_module.BOX_COLOR_MAP
    default_box_color = config_module.DEFAULT_BOX_COLOR
    
    for gt_data in ground_truths_list:
        x1, y1, x2, y2 = map(int, gt_data['xyxy'])
        class_id = gt_data['cls']
        class_name = class_names_map.get(class_id, f"ID_{class_id}")
        label = f"GT: {class_name}"
        color = box_color_map.get(class_name.lower().strip(), default_box_color)

        # Use drawing parameters from config
        cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, config_module.BOX_THICKNESS)
        (text_w, text_h), _ = cv2.getTextSize(label, font, config_module.INFERENCE_FONT_SCALE, config_module.TEXT_THICKNESS)
        cv2.rectangle(img_to_draw_on, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(img_to_draw_on, label, (x1, y1 - 5), font, config_module.INFERENCE_FONT_SCALE, (0,0,0), config_module.TEXT_THICKNESS)
        
    return img_to_draw_on

def save_config_to_run_dir(run_dir_path, logger):
    """Copies config.py to the specified run directory for reproducibility."""
    try:
        # Assuming config.py is in the project root directory where the script is run
        config_source_path = Path("config.py")
        if not config_source_path.exists():
             logger.warning("config.py not found. Skipping save to run directory.")
             return

        destination_path = Path(run_dir_path) / "config.py"
        shutil.copy2(config_source_path, destination_path)
        logger.info(f"Saved a copy of the configuration to {destination_path}")
    except Exception as e:
        logger.error(f"Could not save config.py to run directory: {e}")


# Factory function for creating a detector 
def create_detector_from_config(model_path, class_map, config_module, logger):
    """
    Creates a fully configured CoinDetector instance from config.
    Args:
        model_path (str or Path): Path to the model file. Will be converted to Path if str.
        class_map (dict): Mapping of class IDs to class names.
        config_module (module): The configuration module.
        logger (logging.Logger): Logger instance.
    Returns:
        CoinDetector: Configured instance of the detector.
    """
    # Ensure model_path is a Path object before passing to CoinDetector
    model_path_obj = Path(model_path)
    logger.info(f"Creating detector instance with model: {model_path_obj}")
    
    detector = CoinDetector(
        model_path=model_path_obj, # Pass the Path object
        class_names_map=class_map,
        per_class_conf_thresholds=config_module.PER_CLASS_CONF_THRESHOLDS,
        default_conf_thresh=config_module.DEFAULT_CONF_THRESHOLD,
        iou_suppression_threshold=config_module.IOU_SUPPRESSION_THRESHOLD,
        box_color_map=config_module.BOX_COLOR_MAP,
        default_box_color=config_module.DEFAULT_BOX_COLOR,
        box_thickness=config_module.BOX_THICKNESS,
        text_thickness=config_module.TEXT_THICKNESS,
        font_face=config_module.FONT_FACE,
        font_scale=config_module.INFERENCE_FONT_SCALE,
        enable_aspect_ratio_filter=config_module.ENABLE_ASPECT_RATIO_FILTER,
        aspect_ratio_filter_threshold=config_module.ASPECT_RATIO_FILTER_THRESHOLD,
        enable_per_class_confidence=config_module.ENABLE_PER_CLASS_CONFIDENCE,
        enable_custom_nms=config_module.ENABLE_CUSTOM_NMS
    )
    return detector

# --- Logger Setup ---
LOG_FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

def setup_logging(log_file_path_obj, logger_name='yolo_script_logger'): # Changed default logger name
    """Configures logging to both console and a file for a given logger."""
    logger = logging.getLogger(logger_name)
    
    if logger.handlers: 
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path_obj, mode='w')
    file_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(stream_handler)
    
    logger.info(f"Logging for '{logger_name}' initialized. File output to: {log_file_path_obj}")
    return logger

def copy_log_to_run_directory(initial_log_path, run_dir_path, target_log_filename, logger_instance=None):
    """Copies the log file to the specified run directory."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    if run_dir_path and initial_log_path.exists():
        destination_log_path = Path(run_dir_path) / target_log_filename
        shutil.copy2(initial_log_path, destination_log_path) 
        log.info(f"Log file copied to: {destination_log_path}")

# --- Centralized function for creating unique run directories ---
def create_unique_run_dir(base_dir_pathobj, run_name_prefix):
    """Creates a unique run directory by appending a counter."""
    candidate_dir = base_dir_pathobj / run_name_prefix
    counter = 1
    while candidate_dir.exists():
        candidate_dir = base_dir_pathobj / f"{run_name_prefix}{counter}"
        counter += 1
    candidate_dir.mkdir(parents=True, exist_ok=False)
    return candidate_dir

def validate_config_and_paths(config_module, mode, logger):
    """Validates key settings from the config module based on the run mode."""
    is_valid = True
    logger.info("--- Validating Configuration ---")

    # Validate INPUTS_DIR
    if not config_module.INPUTS_DIR.exists():
        logger.error(f"Config Error: INPUTS_DIR does not exist at '{config_module.INPUTS_DIR}'.")
        is_valid = False

    # Validate data split ratios
    if not (0.999 < config_module.TRAIN_RATIO + config_module.VAL_RATIO + config_module.TEST_RATIO < 1.001):
        logger.error("Config Error: Data split ratios must sum to 1.0.")
        is_valid = False

    # Validate paths/settings specific to the run mode
    if mode in ['evaluate', 'inference', 'train_direct_eval']:
        model_path_str = config_module.MODEL_PATH_FOR_PREDICTION
        model_path_obj = Path(model_path_str) # Convert to Path for validation
        if not model_path_obj.exists():
            logger.error(f"Config Error: MODEL_PATH_FOR_PREDICTION does not exist at '{model_path_str}'.")
            is_valid = False

    if mode == 'train' and config_module.EPOCHS <= 0:
        logger.error(f"Config Error: EPOCHS must be > 0 for training, but is {config_module.EPOCHS}.")
        is_valid = False

    return is_valid

def discover_and_pair_image_labels(inputs_dir_pathobj, image_subdir_basename, label_subdir_basename, logger_instance=None):
    """Scans subdirectories for image and label folders, and creates pairs."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    image_label_pairs = []
    valid_label_dirs_for_class_scan = []
    for variant_dir in inputs_dir_pathobj.iterdir():
        if variant_dir.is_dir():
            image_dir = variant_dir / image_subdir_basename
            label_dir = variant_dir / label_subdir_basename
            if image_dir.is_dir() and label_dir.is_dir():
                valid_label_dirs_for_class_scan.append(label_dir.resolve())
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in image_dir.glob(ext):
                        label_path = label_dir / (img_path.stem + ".txt")
                        if label_path.is_file():
                            image_label_pairs.append((img_path.resolve(), label_path.resolve()))
    return image_label_pairs, valid_label_dirs_for_class_scan

def split_data(image_label_pairs, train_ratio, val_ratio, test_ratio, seed=42, logger_instance=None):
    """Splits image-label pairs into train, validation, and test sets."""
    random.seed(seed)
    random.shuffle(image_label_pairs)
    total = len(image_label_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return image_label_pairs[:train_end], image_label_pairs[train_end:val_end], image_label_pairs[val_end:]

def get_unique_class_ids(list_of_label_dir_paths, logger_instance=None):
    """Scans label files to find all unique class IDs."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    unique_ids = set()
    for label_dir_path in list_of_label_dir_paths:
        for label_file in label_dir_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if parts := line.strip().split():
                        unique_ids.add(int(parts[0]))
    return sorted(list(unique_ids))

def load_class_names_from_yaml(yaml_path_obj, logger_instance=None): # Renamed for clarity
    """Loads the 'names' list from a YAML file."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    if not yaml_path_obj.is_file(): return None
    try:
        with open(yaml_path_obj, 'r') as f: data = yaml.safe_load(f)
        return data.get('names') if isinstance(data.get('names'), list) else None
    except Exception:
        return None

def create_yolo_dataset_yaml(dataset_root_abs_path_str, train_rel_img_dir_paths, val_rel_img_dir_paths,
                        test_rel_img_dir_paths, class_names_map, num_classes_val,
                        output_yaml_path_obj, image_subdir_basename, label_subdir_basename, logger_instance=None): # Renamed for clarity
    """Creates the dataset.yaml file for YOLO training."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    names = [class_names_map.get(i, f"class_{i}") for i in range(num_classes_val)]
    data = {
        'path': dataset_root_abs_path_str,
        'train': [str(p) for p in train_rel_img_dir_paths],
        'val': [str(p) for p in val_rel_img_dir_paths],
        'test': [str(p) for p in test_rel_img_dir_paths],
        'nc': num_classes_val,
        'names': names
    }
    output_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path_obj, 'w') as f: yaml.dump(data, f, sort_keys=False)
    log.info(f"Successfully created dataset YAML: {output_yaml_path_obj}")

def parse_yolo_annotations(label_file_path, logger_instance=None):
    """Parses a YOLO format label file for detailed annotations."""
    annotations = []
    if Path(label_file_path).is_file():
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append((int(parts[0]), *map(float, parts[1:])))
    return annotations

def plot_readable_confusion_matrix(matrix_data, class_names, output_path, title='Confusion Matrix'):
    """
    Generates and saves a readable confusion matrix plot using Seaborn.
    Handles the case where the matrix includes an extra 'background' class.
    """
    logger = logging.getLogger('yolo_script_logger')
    if matrix_data is None:
        logger.warning("Confusion matrix data is None. Skipping plot generation.")
        return

    matrix_data_int = matrix_data.astype(int)
    # Make a mutable copy of the provided class names
    plot_labels = list(class_names)

    # Check if the matrix dimensions are larger than the known class names
    if matrix_data_int.shape[0] == len(plot_labels) + 1:
        plot_labels.append('background')
    
    # Final check to prevent crash if dimensions are still mismatched
    if matrix_data_int.shape[0] != len(plot_labels):
        logger.error(
            f"Shape mismatch for confusion matrix plot. Matrix is {matrix_data_int.shape}, "
            f"but there are {len(plot_labels)} labels. Skipping plot."
        )
        return

    df_cm = pd.DataFrame(matrix_data_int, index=plot_labels, columns=plot_labels)
    
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12})
                          
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    
    plt.ylabel('Predicted', fontsize=14)
    plt.xlabel('True', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        logger.info(f"Saved readable confusion matrix plot to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix plot to {output_path}: {e}")
    finally:
        plt.close()

def draw_error_annotations(image_np, fp_predictions_to_draw, fn_gt_to_draw, class_names_map, config_module):
    """Draws specified False Positive and False Negative boxes for error analysis."""
    font = config_module.FONT_FACE
    box_color_map = config_module.BOX_COLOR_MAP
    default_box_color = config_module.DEFAULT_BOX_COLOR

    # --- Draw False Positive Predicted Boxes ---
    if fp_predictions_to_draw:
        for pred_data in fp_predictions_to_draw:
            x1, y1, x2, y2 = map(int, pred_data['xyxy'])
            class_name = class_names_map.get(int(pred_data['cls']), f"ID_{int(pred_data['cls'])}")
            label = f"FP: {class_name} {pred_data['conf']:.2f}"
            color = box_color_map.get(class_name.lower().strip(), default_box_color)
            
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, config_module.BOX_THICKNESS)
            (text_w, text_h), _ = cv2.getTextSize(label, font, config_module.ERROR_FP_FONT_SCALE, config_module.TEXT_THICKNESS)
            cv2.rectangle(image_np, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(image_np, label, (x1, y1 - 5), font, config_module.ERROR_FP_FONT_SCALE, (0,0,0), config_module.TEXT_THICKNESS)

    # --- Draw Missed Ground Truth Boxes (False Negatives) ---
    if fn_gt_to_draw:
        for gt_data in fn_gt_to_draw:
            x1, y1, x2, y2 = map(int, gt_data['xyxy'])
            class_name = class_names_map.get(gt_data['cls'], f"ID_{gt_data['cls']}")
            label = f"GT: {class_name}"
            color = box_color_map.get(class_name.lower().strip(), default_box_color)

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, config_module.BOX_THICKNESS)
            (text_w, text_h), _ = cv2.getTextSize(label, font, config_module.ERROR_FN_FONT_SCALE, config_module.TEXT_THICKNESS)
            cv2.rectangle(image_np, (x1, y2), (x1 + text_w, y2 + text_h + 5), color, -1)
            cv2.putText(image_np, label, (x1, y2 + text_h + 5), font, config_module.ERROR_FN_FONT_SCALE, (0,0,0), config_module.TEXT_THICKNESS)

    return image_np

def calculate_prf1(tp, fp, fn):
    """
    Calculates precision, recall, and F1-score from TP, FP, and FN counts.
    This is a centralized utility function to avoid code duplication.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

def _get_relative_path_for_yolo_yaml(pairs, inputs_dir):
    """
    Gets unique parent directories of images from pairs, relative to a base directory.
    This is a helper for creating dataset.yaml files.
    """
    if not pairs:
        return []
    # Using a set to handle duplicates automatically
    # The path is converted to a string for use in the YAML file
    relative_dirs = {str(p[0].parent.relative_to(inputs_dir)) for p in pairs}
    return sorted(list(relative_dirs))