import os
import glob
import yaml
import random
import shutil
import logging
from pathlib import Path
from collections import Counter
import cv2 # For drawing
import numpy as np # For drawing

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

    try:
        file_handler = logging.FileHandler(log_file_path_obj, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to set up file logging to {log_file_path_obj}: {e}. Attempting console logging only.")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(stream_handler)
    
    logger.info(f"Logging for '{logger_name}' initialized. File output to: {log_file_path_obj}")
    return logger

def copy_log_to_run_directory(initial_log_path, run_dir_path, target_log_filename, logger_instance=None):
    """Copies the log file to the specified run directory."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    if run_dir_path and initial_log_path.exists():
        try:
            destination_log_path = Path(run_dir_path) / target_log_filename
            shutil.copy2(initial_log_path, destination_log_path) 
            log.info(f"Log file copied to: {destination_log_path}")
        except Exception as e:
            log.error(f"Error copying log file from {initial_log_path} to {destination_log_path}: {e}")
    elif not run_dir_path:
        log.warning("Cannot copy log file: Run directory not specified or not created.")
    elif not initial_log_path.exists():
        log.warning(f"Cannot copy log file: Initial log file {initial_log_path} does not exist.")

def discover_and_pair_image_labels(inputs_dir_pathobj, image_subdir_basename, label_subdir_basename, logger_instance=None):
    """Scans subdirectories for image and label folders, and creates pairs."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    image_label_pairs = []
    valid_label_dirs_for_class_scan = []

    if not inputs_dir_pathobj.is_dir():
        log.error(f"Root dataset directory '{inputs_dir_pathobj}' not found.")
        return image_label_pairs, valid_label_dirs_for_class_scan

    log.info(f"Scanning for dataset variants in: {inputs_dir_pathobj}")
    for variant_dir in inputs_dir_pathobj.iterdir():
        if variant_dir.is_dir():
            image_dir = variant_dir / image_subdir_basename
            label_dir = variant_dir / label_subdir_basename
            if image_dir.is_dir() and label_dir.is_dir():
                log.info(f"  Processing variant: '{variant_dir.name}'")
                valid_label_dirs_for_class_scan.append(label_dir.resolve())
                image_extensions = ['*.jpg', '*.jpeg', '*.png']
                for ext in image_extensions:
                    for img_path in image_dir.glob(ext):
                        label_filename = img_path.stem + ".txt"
                        label_path = label_dir / label_filename
                        if label_path.is_file():
                            image_label_pairs.append((img_path.resolve(), label_path.resolve()))
                        else:
                            log.warning(f"    Label file not found for image '{img_path.name}' in '{label_dir}'. Skipping.")
    if not image_label_pairs:
        log.warning(f"No valid image-label pairs found in '{inputs_dir_pathobj}'.")
    return image_label_pairs, valid_label_dirs_for_class_scan

def split_data(image_label_pairs, train_ratio, val_ratio, test_ratio, seed=42, logger_instance=None):
    """Splits image-label pairs into train, validation, and test sets."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    if not (0.999 < train_ratio + val_ratio + test_ratio < 1.001):
        log.error("Train, validation, and test ratios must sum to approximately 1.0.")
        raise ValueError("Ratios must sum to 1.0.")
    random.seed(seed)
    random.shuffle(image_label_pairs)
    total_samples = len(image_label_pairs)
    train_end_idx = int(total_samples * train_ratio)
    val_end_idx = train_end_idx + int(total_samples * val_ratio)
    train_pairs = image_label_pairs[:train_end_idx]
    val_pairs = image_label_pairs[train_end_idx:val_end_idx]
    test_pairs = image_label_pairs[val_end_idx:]
    log.info(f"Data split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test.")
    return train_pairs, val_pairs, test_pairs

def get_unique_class_ids(list_of_label_dir_paths, logger_instance=None):
    """Scans label files to find all unique class IDs."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    unique_ids = set()
    if not list_of_label_dir_paths:
        log.warning("(get_unique_class_ids): No label directories provided.")
        return []
    for label_dir_path in list_of_label_dir_paths:
        if not label_dir_path.is_dir():
            log.warning(f"Label path '{label_dir_path}' is not a directory. Skipping.")
            continue
        for label_file in label_dir_path.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts: unique_ids.add(int(parts[0]))
            except Exception as e:
                log.error(f"Error reading label file {label_file}: {e}")
    if not unique_ids:
        log.warning("No class IDs found in label directories.")
    return sorted(list(unique_ids))

def load_class_names_from_yaml(yaml_path_obj, logger_instance=None): # Renamed for clarity
    """Loads the 'names' list from a YAML file."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    if not yaml_path_obj.is_file():
        log.info(f"Data YAML file not found: '{yaml_path_obj}'.")
        return None
    try:
        with open(yaml_path_obj, 'r') as f: data = yaml.safe_load(f)
        if data and 'names' in data and isinstance(data['names'], list):
            return data['names']
        else:
            log.warning(f"'names' key not found or invalid in '{yaml_path_obj}'.")
            return None
    except Exception as e:
        log.error(f"Error reading YAML {yaml_path_obj}: {e}")
        return None

def create_yolo_dataset_yaml(dataset_root_abs_path_str, train_rel_img_dir_paths, val_rel_img_dir_paths,
                        test_rel_img_dir_paths, class_names_map, num_classes_val,
                        output_yaml_path_obj, image_subdir_basename, label_subdir_basename, logger_instance=None): # Renamed for clarity
    """Creates the dataset.yaml file for YOLO training."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    names_list_for_yaml = [""] * num_classes_val 
    for class_id, name in class_names_map.items():
        if 0 <= class_id < num_classes_val:
            names_list_for_yaml[class_id] = str(name)
        else:
            log.warning(f"Class ID {class_id} ('{name}') out of range for nc={num_classes_val}.")
    for i in range(num_classes_val):
        if not names_list_for_yaml[i]: names_list_for_yaml[i] = f"class_{i}"

    data = {
        'path': dataset_root_abs_path_str,
        'train': [str(p) for p in train_rel_img_dir_paths],
        'val': [str(p) for p in val_rel_img_dir_paths],
        'test': [str(p) for p in test_rel_img_dir_paths],
        'nc': num_classes_val,
        'names': names_list_for_yaml
    }
    try:
        output_yaml_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        with open(output_yaml_path_obj, 'w') as f: yaml.dump(data, f, sort_keys=False, default_flow_style=None)
        log.info(f"Successfully created dataset YAML: {output_yaml_path_obj}")
        yaml_content_str = yaml.dump(data, sort_keys=False, default_flow_style=None)
        log.info(f"Generated YAML content:\n{yaml_content_str}")
        log.info(f"YOLO expects labels in a '{label_subdir_basename}' folder relative to the image folders listed in train/val/test.")
    except Exception as e:
        log.error(f"Error writing YAML {output_yaml_path_obj}: {e}")

def parse_yolo_annotations(label_file_path, logger_instance=None):
    """Parses a YOLO format label file for detailed annotations."""
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    annotations = []
    if not Path(label_file_path).is_file(): # Ensure Path object for is_file()
        log.warning(f"Annotation file not found: {label_file_path}")
        return annotations
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    except Exception as e:
        log.error(f"Error parsing annotation file {label_file_path}: {e}")
    return annotations

def calculate_iou(box1_xyxy, box2_xyxy):
    """Calculates IoU of two bounding boxes [x1, y1, x2, y2]."""
    x1_i, y1_i = max(box1_xyxy[0], box2_xyxy[0]), max(box1_xyxy[1], box2_xyxy[1])
    x2_i, y2_i = min(box1_xyxy[2], box2_xyxy[2]), min(box1_xyxy[3], box2_xyxy[3])
    intersection_width = max(0, x2_i - x1_i)
    intersection_height = max(0, y2_i - y1_i)
    intersection_area = intersection_width * intersection_height
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def draw_error_annotations(image_np, fp_predictions_to_draw, fn_gt_to_draw, class_names_map, box_color_map, default_box_color, logger_instance=None):
    """
    Draws specified False Positive predictions and False Negative Ground Truth boxes for error analysis.
    Args:
        image_np (np.ndarray): The image to draw on.
        fp_predictions_to_draw (list): List of prediction dicts {'xyxy', 'conf', 'cls'} that are FPs.
        fn_gt_to_draw (list): List of GT dicts {'cls', 'xyxy'} that are FNs (missed GTs).
        class_names_map (dict): Dictionary mapping class ID (int) to class name (str).
        box_color_map (dict): Dictionary mapping class name (str) to BGR color tuple.
        default_box_color (tuple): Default BGR color if class not in box_color_map.
    Returns:
        np.ndarray: The image with specified error annotations drawn.
    """
    log = logger_instance if logger_instance else logging.getLogger('yolo_script_logger')
    img_h, img_w = image_np.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Draw False Positive Predicted Boxes ---
    if fp_predictions_to_draw:
        pred_font_scale = 1.5 # Font scale for prediction text
        pred_text_thickness = 4 # Thickness for text, must be an integer
        pred_box_thickness = 2 # Thickness for box lines, must be an integer

        for pred_data in fp_predictions_to_draw:
            x1, y1, x2, y2 = map(int, pred_data['xyxy'])
            class_id = int(pred_data['cls'])
            confidence = float(pred_data['conf'])
            
            class_name = class_names_map.get(class_id, f"ID_{class_id}")
            label = f"FP: {class_name} {confidence:.2f}"
            
            color = box_color_map.get(class_name.lower().strip(), default_box_color)
            text_color_on_bg = (0,0,0) 

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, pred_box_thickness)
            (text_w, text_h), baseline = cv2.getTextSize(label, font, pred_font_scale, pred_text_thickness)

            label_x_pos = x1
            label_y_pos = y1 - baseline - 3 
            if label_y_pos < text_h :
                label_y_pos = y1 + text_h + baseline + 3

            cv2.rectangle(image_np, (label_x_pos, label_y_pos - text_h - baseline), (label_x_pos + text_w, label_y_pos + baseline), color, -1)
            cv2.putText(image_np, label, (label_x_pos, label_y_pos), font, pred_font_scale, text_color_on_bg, pred_text_thickness, cv2.LINE_AA)

    # --- Draw Missed Ground Truth Boxes (False Negatives) ---
    if fn_gt_to_draw:
        gt_font_scale = 1.4 # Font scale for ground truth text
        gt_text_thickness = 3 # Thickness for text, must be integer
        gt_box_thickness = 2 # Thickness for box lines, must be integer

        for gt_data in fn_gt_to_draw:
            x1_gt, y1_gt, x2_gt, y2_gt = map(int, gt_data['xyxy'])
            class_id = gt_data['cls']
            class_name = class_names_map.get(class_id, f"ID_{class_id}")
            gt_color = box_color_map.get(class_name.lower().strip(), default_box_color)

            cv2.rectangle(image_np, (x1_gt, y1_gt), (x2_gt, y2_gt), gt_color, gt_box_thickness)
            label = f"GT: {class_name}" # Label as GT, context implies it was missed
            (text_w, text_h), baseline = cv2.getTextSize(label, font, gt_font_scale, gt_text_thickness)
            
            text_margin = 3
            label_x_pos = x2_gt - text_w - text_margin 
            label_y_pos = y1_gt + text_h + text_margin 

            cv2.rectangle(image_np, (label_x_pos, label_y_pos - text_h - baseline), (label_x_pos + text_w, label_y_pos + baseline), gt_color, -1)
            cv2.putText(image_np, label, (label_x_pos, label_y_pos), font, gt_font_scale, (0,0,0), gt_text_thickness, cv2.LINE_AA)

    # --- Draw Legend ---
    legend_font_scale = 0.5
    cv2.putText(image_np, "FP Preds (Custom Colors, Top-Left)", (10, img_h - 40), font, legend_font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image_np, "Missed GT (FN) (Custom Colors, Inside Top-Right)", (10, img_h - 20), font, legend_font_scale, (220,220,220), 1, cv2.LINE_AA)

    return image_np
