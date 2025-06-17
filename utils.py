import os
import yaml
import random
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
from types import ModuleType

from bbox_utils import calculate_aspect_ratio

# Forward reference for CoinDetector to avoid circular import
if 'CoinDetector' not in globals():
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from detector import CoinDetector


def check_image_blur(image_np: np.ndarray, threshold: float, logger: logging.Logger, image_name: str) -> None:
    """
    Checks if an image is blurry by calculating the variance of its Laplacian.
    If the variance is below a given threshold, a warning is logged.
    """
    if image_np is None:
        return

    # Convert to grayscale to compute Laplacian
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # Calculate the variance of the Laplacian
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < threshold:
        logger.warning(
            f"Image '{image_name}' may be blurry (Laplacian variance: {variance:.2f}). "
            f"Detection results may be inaccurate."
        )

def check_image_darkness(image_np: np.ndarray, threshold: float, logger: logging.Logger, image_name: str) -> None:
    """
    Checks if an image is too dark by calculating the mean of its grayscale pixel values.
    If the mean brightness is below a given threshold, a warning is logged.
    """
    if image_np is None:
        return

    # Convert to grayscale to calculate average brightness
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # Calculate the mean of all pixel values
    brightness = gray.mean()

    if brightness < threshold:
        logger.warning(
            f"Image '{image_name}' may be too dark (average brightness: {brightness:.2f}). "
            f"Detection results may be inaccurate."
        )

def check_sharp_angle(
    predictions_data: List[Dict[str, Any]], 
    ar_threshold: float, 
    min_percentage_threshold: float, 
    logger: logging.Logger, 
    image_name: str
) -> None:
    """
    Checks if an image was potentially taken from a sharp angle by analyzing
    the aspect ratio of detected bounding boxes.

    Args:
        predictions_data (list): A list of raw prediction dictionaries.
        ar_threshold (float): The aspect ratio above which a box is suspicious.
        min_percentage_threshold (float): The percentage of suspicious boxes required
                                          to trigger a warning for the image.
        logger (logging.Logger): Logger instance.
        image_name (str): The name of the image for the log message.
    """
    if not predictions_data:
        return  # Cannot check if no objects were detected

    suspicious_boxes_count = 0
    total_boxes = len(predictions_data)

    for pred in predictions_data:
        aspect_ratio = calculate_aspect_ratio(pred['xyxy'])
        if aspect_ratio > ar_threshold:
            suspicious_boxes_count += 1

    if total_boxes > 0:
        percentage_suspicious = (suspicious_boxes_count / total_boxes) * 100
        if percentage_suspicious >= min_percentage_threshold:
            logger.warning(
                f"Image '{image_name}' may have been taken from a sharp angle. "
                f"{suspicious_boxes_count}/{total_boxes} ({percentage_suspicious:.1f}%) of detected objects "
                f"have an aspect ratio > {ar_threshold}."
            )


def get_adaptive_drawing_params(image_width: int, config_module: ModuleType) -> Dict[str, Union[int, float]]:
    """
    Calculates drawing parameters scaled to the image size, based on a reference width.
    """
    # If adaptive drawing is disabled, return the base values from config
    if not getattr(config_module, 'ADAPTIVE_DRAWING_ENABLED', False):
        return {
            "box_thickness": config_module.BOX_THICKNESS,
            "text_thickness": config_module.TEXT_THICKNESS,
            "inference_font_scale": config_module.INFERENCE_FONT_SCALE,
            "error_fp_font_scale": config_module.ERROR_FP_FONT_SCALE,
            "error_fn_font_scale": config_module.ERROR_FN_FONT_SCALE,
        }
    
    # Calculate the scaling factor based on image width
    reference_width: int = getattr(config_module, 'REFERENCE_IMAGE_WIDTH', 4000)
    scaling_factor = image_width / reference_width

    # Clip the scaling factor to prevent parameters from becoming too small on tiny images
    scaling_factor = max(0.1, scaling_factor)

    # Scale the parameters, ensuring thickness is at least 1
    box_thickness = max(1, int(round(config_module.BOX_THICKNESS * scaling_factor)))
    text_thickness = max(1, int(round(config_module.TEXT_THICKNESS * scaling_factor)))
    inference_font_scale = config_module.INFERENCE_FONT_SCALE * scaling_factor
    error_fp_font_scale = config_module.ERROR_FP_FONT_SCALE * scaling_factor
    error_fn_font_scale = config_module.ERROR_FN_FONT_SCALE * scaling_factor

    return {
        "box_thickness": box_thickness,
        "text_thickness": text_thickness,
        "inference_font_scale": inference_font_scale,
        "error_fp_font_scale": error_fp_font_scale,
        "error_fn_font_scale": error_fn_font_scale,
    }


# Factory function for creating a detector 
def create_detector_from_config(model_path: Union[str, Path], class_map: Dict[int, str], config_module: ModuleType, logger: logging.Logger) -> 'CoinDetector':
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
	
    from detector import CoinDetector
	
    # Ensure model_path is a Path object before passing to CoinDetector
    model_path_obj = Path(model_path)
    logger.info(f"Creating detector instance with model: {model_path_obj}")
    
    detector = CoinDetector(
        model_path=model_path_obj,
        class_names_map=class_map,
        config_module=config_module,  # Pass the entire config module
        logger=logger,
        per_class_conf_thresholds=config_module.PER_CLASS_CONF_THRESHOLDS,
        default_conf_thresh=config_module.DEFAULT_CONF_THRESHOLD,
        iou_suppression_threshold=config_module.IOU_SUPPRESSION_THRESHOLD,
        enable_aspect_ratio_filter=config_module.ENABLE_ASPECT_RATIO_FILTER,
        aspect_ratio_filter_threshold=config_module.ASPECT_RATIO_FILTER_THRESHOLD,
        enable_per_class_confidence=config_module.ENABLE_PER_CLASS_CONFIDENCE,
        enable_custom_nms=config_module.ENABLE_CUSTOM_NMS,
        enable_grayscale_preprocessing_from_config=config_module.ENABLE_GRAYSCALE_PREPROCESSING
    )
    return detector

def convert_to_3channel_grayscale(image_np_or_path: Union[np.ndarray, str, Path], logger_instance: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """
    Loads an image if a path is given, converts it to grayscale,
    and then converts the grayscale image to a 3-channel BGR format.
    """
    log = logger_instance if logger_instance else logging.getLogger(__name__)

    log.debug(f"Starting image preprocessing for input: {type(image_np_or_path)}")
    image_np: Optional[np.ndarray] = None
    if isinstance(image_np_or_path, (str, Path)):
        if not Path(image_np_or_path).exists():
            log.error(f"Image path does not exist: {image_np_or_path}")
            return None
        image_np = cv2.imread(str(image_np_or_path))
        if image_np is None:
            log.error(f"Failed to load image from path: {image_np_or_path}")
            return None
        log.debug(f"Successfully loaded image from path: {image_np_or_path}")
    elif isinstance(image_np_or_path, np.ndarray):
        image_np = image_np_or_path.copy() 
        log.debug("Processing image from NumPy array.")
    else:
        log.error(f"Invalid image input type: {type(image_np_or_path)}")
        return None

    if image_np.size == 0:
        log.error("Input image array is empty.")
        return None

    if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):
        log.debug("Image appears to be already grayscale or has only one channel.")
        gray_image_np = image_np if image_np.ndim == 2 else image_np[:,:,0]
    elif image_np.ndim == 3 and image_np.shape[2] == 3: 
        gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        log.debug("Converted BGR image to grayscale.")
    elif image_np.ndim == 3 and image_np.shape[2] == 4: 
        log.debug("Input image has 4 channels (e.g., BGRA). Converting to BGR first.")
        bgr_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        gray_image_np = cv2.cvtColor(bgr_image_np, cv2.COLOR_BGR2GRAY)
        log.debug("Converted 4-channel image to grayscale.")
    else:
        log.error(f"Unsupported image format/dimensions: {image_np.shape}")
        return None
            
    image_for_model = cv2.cvtColor(gray_image_np, cv2.COLOR_GRAY2BGR)
    log.debug("Converted grayscale image to 3-channel BGR for model input. Preprocessing finished.")
    return image_for_model

def draw_ground_truth_boxes(image_np: np.ndarray, ground_truths_list: List[Dict[str, Any]], class_names_map: Dict[int, str], config_module: ModuleType) -> np.ndarray:
    """
    Draws ground truth bounding boxes on an image using adaptive settings.

    Args:
        image_np (np.ndarray): The image to draw on.
        ground_truths_list (list): A list of ground truth dictionaries.
                                   Each dict: {'cls': class_id, 'xyxy': [x1, y1, x2, y2]}
        class_names_map (dict): A map from class ID to class name.
        config_module (module): The configuration module for styling.
    """
    img_to_draw_on = image_np.copy()
    h, w, _ = img_to_draw_on.shape
    
    params = get_adaptive_drawing_params(w, config_module)
    box_thickness = params['box_thickness']
    text_thickness = params['text_thickness']
    font_scale = params['inference_font_scale']
    font = config_module.FONT_FACE
    box_color_map = config_module.BOX_COLOR_MAP
    default_box_color = config_module.DEFAULT_BOX_COLOR
    
    for gt_data in ground_truths_list:
        x1, y1, x2, y2 = map(int, gt_data['xyxy'])
        class_id = gt_data['cls']
        class_name = class_names_map.get(class_id, f"ID_{class_id}")
        label = f"GT: {class_name}"
        color = box_color_map.get(class_name.lower().strip(), default_box_color)

        cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, int(box_thickness))
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, int(text_thickness))
        cv2.rectangle(img_to_draw_on, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(img_to_draw_on, label, (x1, y1 - 5), font, font_scale, (0,0,0), int(text_thickness))
        
    return img_to_draw_on

def save_config_to_run_dir(run_dir_path: Path, logger: logging.Logger) -> None:
    """Copies config.py to the specified run directory for reproducibility."""
    try:
        # Build path relative to this utils.py file, not the current working directory
        project_root = Path(__file__).parent
        config_source_path = project_root / "config.py"

        if not config_source_path.exists():
             logger.warning(f"config.py not found at expected path {config_source_path}. Skipping save.")
             return

        destination_path = Path(run_dir_path) / "config.py"
        shutil.copy2(config_source_path, destination_path)
        logger.info(f"Saved a copy of the configuration to {destination_path}")
    except Exception as e:
        logger.error(f"Could not save config.py to run directory: {e}")


# --- Logger Setup ---
LOG_FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

def setup_logging(log_file_path_obj: Path, logger_name: str = 'yolo_script_logger') -> logging.Logger:
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


# --- Centralized function for creating unique run directories ---
def create_unique_run_dir(base_dir_pathobj: Path, run_name_prefix: str) -> Path:
    """Creates a unique run directory by appending a counter."""
    candidate_dir = base_dir_pathobj / run_name_prefix
    counter = 1
    candidate_dir = base_dir_pathobj / f"{run_name_prefix}"
    # Initial check for the non-suffixed directory name
    if not candidate_dir.exists():
        candidate_dir.mkdir(parents=True, exist_ok=False)
        return candidate_dir
    
    # If it exists, start appending counters
    while candidate_dir.exists():
        candidate_dir = base_dir_pathobj / f"{run_name_prefix}{counter}"
        counter += 1
    candidate_dir.mkdir(parents=True, exist_ok=False)
    return candidate_dir

def validate_config_and_paths(config_module: ModuleType, mode: str, logger: logging.Logger) -> bool:
    """Validates key settings from the config module based on the run mode."""
    is_valid = True
    logger.info("--- Validating Configuration ---")

    # Validate INPUTS_DIR
    if not config_module.INPUTS_DIR.exists():
        logger.error(f"Config Error: INPUTS_DIR does not exist at '{config_module.INPUTS_DIR}'.")
        is_valid = False

    # Validate data split ratios
    if not config_module.USE_PREDEFINED_SPLITS and not (0.999 < config_module.TRAIN_RATIO + config_module.VAL_RATIO + config_module.TEST_RATIO < 1.001):
        logger.error("Config Error: Data split ratios must sum to 1.0 when not using pre-defined splits.")
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

def discover_and_pair_image_labels(
    inputs_dir_pathobj: Path, 
    image_subdir_basename: str, 
    label_subdir_basename: str, 
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """
    Recursively scans for image and label folders and creates pairs.
    It looks for sibling folders with the specified base names (e.g., 'images' and 'labels').
    """
    log = logger_instance if logger_instance else logging.getLogger(__name__)
    image_label_pairs: List[Tuple[Path, Path]] = []
    valid_label_dirs_for_class_scan: List[Path] = []
    
    log.info(f"Recursively searching for '{image_subdir_basename}' folders within {inputs_dir_pathobj}...")

    # Use rglob to find all directories named `image_subdir_basename` at any depth
    for image_dir in inputs_dir_pathobj.rglob(image_subdir_basename):
        if not image_dir.is_dir():
            continue

        # The corresponding label directory should be a sibling to the image directory
        label_dir = image_dir.parent / label_subdir_basename
        
        if label_dir.is_dir():
            log.info(f"Found matching pair of data folders: '{image_dir.parent.name}'")
            valid_label_dirs_for_class_scan.append(label_dir.resolve())
            
            # The rest of the logic for pairing files within the folders is the same
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in image_dir.glob(ext):
                    label_path = label_dir / (img_path.stem + ".txt")
                    if label_path.is_file():
                        image_label_pairs.append((img_path.resolve(), label_path.resolve()))
        else:
            log.warning(f"Found an '{image_dir}' but no corresponding sibling directory named '{label_subdir_basename}'. Skipping.")

    if not image_label_pairs:
        log.warning(f"No image-label pairs were found after a recursive search in {inputs_dir_pathobj}.")
        
    return image_label_pairs, valid_label_dirs_for_class_scan

ImageLabelPair = Tuple[Path, Path]

def split_data(
    image_label_pairs: List[ImageLabelPair], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float, 
    seed: int = 42, 
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[List[ImageLabelPair], List[ImageLabelPair], List[ImageLabelPair]]:
    """Splits image-label pairs into train, validation, and test sets."""
    random.seed(seed)
    random.shuffle(image_label_pairs)
    total = len(image_label_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return image_label_pairs[:train_end], image_label_pairs[train_end:val_end], image_label_pairs[val_end:]


def load_class_names_from_yaml(yaml_path_obj: Path, logger_instance: Optional[logging.Logger] = None) -> Optional[List[str]]:
    """Loads the 'names' list from a YAML file."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    if not yaml_path_obj.is_file(): return None
    try:
        with open(yaml_path_obj, 'r') as f: data = yaml.safe_load(f)
        names = data.get('names')
        return names if isinstance(names, list) else None
    except Exception:
        return None

def get_class_map_from_yaml(config_module: ModuleType, logger: logging.Logger) -> Optional[Dict[int, str]]:
    """
    Loads the class names from the project's central YAML file and returns
    it as a dictionary map.
    
    Returns:
        dict: A map of class IDs to class names, or None if loading fails.
    """
    class_names_yaml_path = Path(config_module.CLASS_NAMES_YAML)
    names_from_yaml = load_class_names_from_yaml(class_names_yaml_path, logger)
    if names_from_yaml is None:
        logger.error(f"CRITICAL: Could not load class names from '{class_names_yaml_path}'.")
        return None
        
    class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}
    logger.info(f"Loaded {len(class_names_map)} class names: {class_names_map}")
    return class_names_map

def create_yolo_dataset_yaml(
    dataset_root_abs_path_str: str, 
    train_rel_img_dir_paths: List[str], 
    val_rel_img_dir_paths: List[str],
    test_rel_img_dir_paths: List[str], 
    class_names_map: Dict[int, str], 
    num_classes_val: int,
    output_yaml_path_obj: Path, 
    image_subdir_basename: str, 
    label_subdir_basename: str, 
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """Creates the dataset.yaml file for YOLO training."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    names = [class_names_map.get(i, f"class_{i}") for i in range(num_classes_val)]
    data = {
        'path': dataset_root_abs_path_str,
        'train': train_rel_img_dir_paths,
        'val': val_rel_img_dir_paths,
        'test': test_rel_img_dir_paths,
        'nc': num_classes_val,
        'names': names
    }
    output_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path_obj, 'w') as f: yaml.dump(data, f, sort_keys=False)
    log.info(f"Successfully created dataset YAML: {output_yaml_path_obj}")

def parse_yolo_annotations(label_file_path: Union[str, Path], logger_instance: Optional[logging.Logger] = None) -> List[Tuple[int, float, float, float, float]]:
    """Parses a YOLO format label file for detailed annotations."""
    annotations: List[Tuple[int, float, float, float, float]] = []
    if Path(label_file_path).is_file():
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append((int(parts[0]), *map(float, parts[1:])))
    return annotations

def plot_readable_confusion_matrix(matrix_data: np.ndarray, class_names: List[str], output_path: Path, title: str = 'Confusion Matrix') -> None:
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

def draw_error_annotations(
    image_np: np.ndarray, 
    fp_predictions_to_draw: List[Dict[str, Any]], 
    fn_gt_to_draw: List[Dict[str, Any]], 
    class_names_map: Dict[int, str], 
    config_module: ModuleType
) -> np.ndarray:
    """Draws specified False Positive and False Negative boxes for error analysis."""
    h, w, _ = image_np.shape
    params = get_adaptive_drawing_params(w, config_module)
    box_thickness = int(params['box_thickness'])
    text_thickness = int(params['text_thickness'])
    fp_font_scale = params['error_fp_font_scale']
    fn_font_scale = params['error_fn_font_scale']
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
            
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, box_thickness)
            (text_w, text_h), _ = cv2.getTextSize(label, font, fp_font_scale, text_thickness)
            cv2.rectangle(image_np, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(image_np, label, (x1, y1 - 5), font, fp_font_scale, (0,0,0), text_thickness)

    # --- Draw Missed Ground Truth Boxes (False Negatives) ---
    if fn_gt_to_draw:
        for gt_data in fn_gt_to_draw:
            x1, y1, x2, y2 = map(int, gt_data['xyxy'])
            class_name = class_names_map.get(gt_data['cls'], f"ID_{gt_data['cls']}")
            label = f"GT: {class_name}"
            color = box_color_map.get(class_name.lower().strip(), default_box_color)

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, box_thickness)
            (text_w, text_h), _ = cv2.getTextSize(label, font, fn_font_scale, text_thickness)
            cv2.rectangle(image_np, (x1, y2), (x1 + text_w, y2 + text_h + 5), color, -1)
            cv2.putText(image_np, label, (x1, y2 + text_h + 5), font, fn_font_scale, (0,0,0), text_thickness)

    return image_np

def calculate_prf1(tp: float, fp: float, fn: float) -> Dict[str, float]:
    """
    Calculates precision, recall, and F1-score from TP, FP, and FN counts.
    This is a centralized utility function to avoid code duplication.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

def _get_relative_path_for_yolo_yaml(pairs: List[Tuple[Path, Any]], inputs_dir: Path) -> List[str]:
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