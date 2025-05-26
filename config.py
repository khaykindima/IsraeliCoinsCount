from pathlib import Path
import os
import cv2 # Import for font selection

# --- Base Paths ---
# Allow overriding the input directory via an environment variable for portability.
# Fallback to the original hardcoded path if the env var is not set.
INPUTS_DIR_ENV = os.getenv("YOLO_COINS_INPUT_DIR")
INPUTS_DIR = Path(INPUTS_DIR_ENV) if INPUTS_DIR_ENV else Path("/mnt/c/Work/Repos/MyProjects/DeepLearning/CoinsUltralytics/Data/CoinCount.v38i.yolov5pytorch")
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
OUTPUT_DIR = Path("yolo_experiment_output")      # Base directory for all outputs

# --- Data Configuration ---
IMAGE_SUBDIR_BASENAME = "images"
LABEL_SUBDIR_BASENAME = "labels"
ORIGINAL_DATA_YAML_NAME = "data.yaml" # Name of your existing data.yaml with class names

# --- Script-Generated Files ---
DATASET_YAML_NAME = "custom_dataset_for_training.yaml" # For YOLO training
INCORRECT_PREDICTIONS_SUBDIR = "incorrect_predictions"
LOG_FILE_BASE_NAME = "run_log" # Will be appended with script type, e.g., train_run_log.log
PREDICTIONS_CSV_NAME = "predictions_summary.csv"
METRICS_JSON_NAME = "final_evaluation_metrics.json" # Name for the metrics JSON file
CONFUSION_MATRIX_PLOT_NAME = "confusion_matrix.png" # Filename for the saved plot

# --- Model Configuration ---
# MODEL_NAME_FOR_TRAINING can be a base model like "yolov8n.pt" to start fresh,
# or a path to a .pt file to resume training or use as a base.
MODEL_NAME_FOR_TRAINING = "yolov8n_best.pt" # You can change this to other YOLOv11 variants like 'yolov11s.pt' etc.

# MODEL_PATH_FOR_PREDICTION should be the path to your trained model (e.g., best.pt from a training run)
# This will be used by evaluate_model.py and run_inference.py
# Example: MODEL_PATH_FOR_PREDICTION = OUTPUT_DIR / "training_runs" / "yolov8n_custom_training" / "weights" / "best.pt"
MODEL_PATH_FOR_PREDICTION = Path("yolov8n_best.pt") # Placeholder - update this to your actual best model path

# --- Training Parameters ---
EPOCHS = 0 # Set to >0 for training, 0 for prediction/evaluation only using MODEL_PATH_FOR_PREDICTION
IMG_SIZE = 640
TRAINING_OPTIMIZER = 'Adam' # Default is 'SGD', or 'AdamW'
TRAINING_LR0 = 0.0001 # Initial learning rate
TRAINING_LRF = 1.00 # Final OneCycleLR learning rate (lr0 * lrf)

# --- Data Split Ratios ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Augmentation Parameters ---
AUGMENTATION_PARAMS = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.1,
}

# --- Prediction & Evaluation Settings ---
# For custom NMS in detector.py and evaluate_model.py
IOU_SUPPRESSION_THRESHOLD = 0.4
# For matching predictions to GT to determine TP/FP in evaluate_model.py
BOX_MATCHING_IOU_THRESHOLD = 0.5
# Default confidence threshold for classes not in PER_CLASS_CONF_THRESHOLDS and For initial model.predict() call to get raw boxes before per-class filtering
DEFAULT_CONF_THRESHOLD = 0.25
# Per-class confidence thresholds (class names should be lowercase)
PER_CLASS_CONF_THRESHOLDS = {
    "one": 0.35,
    "two": 0.45,
    "five": 0.35,
    "ten": 0.8,
}

# --- Metrics Calculation ---
# Specify which metrics to calculate and report. 
# Options: 'precision', 'recall', 'f1_score', 'confusion_matrix'.
# If empty or None, all available core metrics (P, R, F1, CM) will be calculated.
REQUESTED_METRICS = ['precision', 'recall', 'f1_score', 'confusion_matrix'] 

# --- Drawing Configuration ---
# Colors are in BGR format
BOX_COLOR_MAP = {
    "one": (0, 255, 255),   # Yellow
    "two": (128, 0, 128),   # Purple
    "five": (0, 0, 255),    # Red
    "ten": (255, 255, 0),     # Cyan/Teal
}
DEFAULT_BOX_COLOR = (255, 0, 0) # Blue for any other classes not in map
# ADDED: Configurable thickness and font settings
BOX_THICKNESS = 2
TEXT_THICKNESS = 3
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
# Font scale for regular predictions (run_inference.py)
INFERENCE_FONT_SCALE = 1.2
# Font scales for error analysis images (evaluate_model.py)
ERROR_FP_FONT_SCALE = 1.5 # False Positives
ERROR_FN_FONT_SCALE = 1.4 # False Negatives (missed GT)