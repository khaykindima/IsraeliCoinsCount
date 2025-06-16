"""
config.py

Central configuration file for the entire project.

This file contains all the tunable parameters, paths, and settings used by the various
scripts. Modifying this file is the primary way to control the behavior of training,
evaluation, and inference without changing the core logic.
"""
import cv2 # Import for font selection

# --- Base Paths ---
INPUTS_DIR = "Data/CoinCountv54_plus640size"

# Example of a flexible, nested structure now supported:
# The script will recursively find all sibling 'images' and 'labels' folders.
# INPUTS_DIR/
#  ├── session_1_daylight/
#  │   ├── images/
#  │   │   └── img1.jpg
#  │   └── labels/
#  │       └── img1.txt
#  ├── session_2_indoor/
#  │   ├── setup_A/
#  │   │   ├── images/
#  │   │   │   └── img2.jpg
#  │   │   └── labels/
#  │   │       └── img2.txt
#  │   └── setup_B/
#  │       └── ...
#  └── ...
OUTPUT_DIR = "experiment_results"     # Base directory for all outputs

# --- Data Configuration ---
IMAGE_SUBDIR_BASENAME = "images"
LABEL_SUBDIR_BASENAME = "labels"
CLASS_NAMES_YAML = "classes_names.yaml" # Centralized file for class names
# If True, the script will look for 'train/', 'valid/', 'test/' subfolders in INPUTS_DIR.
# If False, it will split all data according to the ratios below.
USE_PREDEFINED_SPLITS = False

# --- Script-Generated Files ---
DATASET_YAML_NAME = "custom_dataset_for_training.yaml" # For YOLO training
INCORRECT_PREDICTIONS_SUBDIR = "incorrect_predictions"
LOG_FILE_BASE_NAME = "run_log" # Will be appended with script type, e.g., train_run_log.log
PREDICTIONS_CSV_NAME = "predictions_summary.csv"
METRICS_JSON_NAME = "final_evaluation_metrics.json" # Name for the metrics JSON file
CONFUSION_MATRIX_PLOT_NAME = "confusion_matrix.png" # Filename for the saved plot

# --- Model Configuration ---
MODEL_NAME_FOR_TRAINING = "BestModels/yolov8n_v6.pt"
MODEL_PATH_FOR_PREDICTION = "BestModels/yolov8n_v6.pt"

# --- Training Parameters ---
EPOCHS = 0 # Set to >0 for training, 0 for prediction/evaluation only using prediction model
IMG_SIZE = 640
TRAINING_OPTIMIZER = 'Adam' # Default is 'SGD', or 'AdamW'
TRAINING_LR0 = 0.0001 # Initial learning rate
TRAINING_LRF = 0.01 # Final OneCycleLR learning rate (lr0 * lrf)

# --- Data Split Ratios (only used if USE_PREDEFINED_SPLITS is False) ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# --- Image Preprocessing ---
ENABLE_GRAYSCALE_PREPROCESSING = False # Set to False to use original color images


# --- Image Quality Checks ---
# Blur Detection
ENABLE_BLUR_DETECTION = False
# Images with a Laplacian variance below this threshold will be flagged as potentially blurry.
# This value may need tuning based on image resolution and content.
BLUR_DETECTION_THRESHOLD = 100.0

# Darkness Detection
ENABLE_DARKNESS_DETECTION = True
# Images with an average pixel intensity below this threshold will be flagged as too dark.
# The value ranges from 0 (completely black) to 255 (completely white).
DARKNESS_DETECTION_THRESHOLD = 50.0

# Sharp Angle Detection (based on object aspect ratios)
ENABLE_SHARP_ANGLE_DETECTION = True
# Threshold for an individual box's aspect ratio (longer side / shorter side) to be considered suspicious.
SHARP_ANGLE_AR_THRESHOLD = 2.5
# If this percentage of detected boxes exceed the AR threshold, a warning for the whole image is logged.
SHARP_ANGLE_MIN_PERCENTAGE = 50.0

# Cut-off Coin Detection
ENABLE_CUT_OFF_CHECK = True
# Defines how close a box edge needs to be to the image edge to be flagged (in pixels).
CUT_OFF_TOLERANCE = 2


# --- Augmentation Parameters ---
AUGMENTATION_PARAMS = {
    'hsv_h': 0.015,     # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,       # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,       # image HSV-Value augmentation (fraction)
    'degrees': 10.0,    # image rotation (+/- deg)
    'translate': 0.1,   # image translation (+/- fraction)
    'scale': 0.5,       # image scale (+/- gain)
    'shear': 0.0,       # image shear (+/- deg)
    'perspective': 0.0, # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,      # image flip up-down (probability)
    'fliplr': 0.0,      # image flip left-right (probability)
    'mosaic': 1.0,      # mosaic augmentation (probability)
    'mixup': 0.1,       # mixup augmentation (probability)
}

# --- Prediction & Evaluation Settings ---
# For matching predictions to GT in custom evaluation
BOX_MATCHING_IOU_THRESHOLD = 0.5

# For custom Non-Maximum Suppression (NMS)
ENABLE_CUSTOM_NMS = True
IOU_SUPPRESSION_THRESHOLD = 0.4

# Default confidence for initial raw predictions from the model
DEFAULT_CONF_THRESHOLD = 0.25
# Per-class confidence thresholds (class names should be lowercase)
ENABLE_PER_CLASS_CONFIDENCE = True  # Set to True to use per-class thresholds, False to use only DEFAULT_CONF_THRESHOLD for all classes
PER_CLASS_CONF_THRESHOLDS = {
    "one": 0.5,
    "two": 0.5,
    "five": 0.5,
    "ten": 0.8,
}
# --- Aspect Ratio Filtering ---
ENABLE_ASPECT_RATIO_FILTER = True  # Set to True to enable, False to disable
# Filters out boxes where the ratio of the longer side to the shorter side exceeds this threshold.
# For example, a threshold of 2.5 means boxes where one side is more than 2.5x the other will be removed.
# A perfect square has a ratio of 1.0.
ASPECT_RATIO_FILTER_THRESHOLD = 2.1


# --- Drawing Configuration ---
# Colors are in BGR format (Blue, Green, Red)
BOX_COLOR_MAP = {
    "one": (0, 255, 255),   # Yellow
    "two": (128, 0, 128),   # Purple
    "five": (0, 0, 255),    # Red
    "ten": (255, 255, 0),     # Cyan/Teal
}
DEFAULT_BOX_COLOR = (255, 0, 0) # Blue for any other classes not in map

# Base values for drawing parameters. If adaptive drawing is enabled,
# these are scaled relative to the reference image size.
BOX_THICKNESS = 2
TEXT_THICKNESS = 3
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
# Font scale for regular predictions (run_inference.py)
INFERENCE_FONT_SCALE = 1.2
# Font scales for error analysis images (evaluate_model.py)
ERROR_FP_FONT_SCALE = 1.5 # Font for False Positives
ERROR_FN_FONT_SCALE = 1.4 # Font for False Negatives

# --- Adaptive Drawing Configuration ---
ADAPTIVE_DRAWING_ENABLED = True
REFERENCE_IMAGE_WIDTH = 4000 # The width for which the base drawing parameters above are optimized.

# --- Coin Values ---
# Maps class names to their monetary value for calculating total sum
COIN_VALUES = {
    "one": 1,
    "two": 2,
    "five": 5,
    "ten": 10,
}