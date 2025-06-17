import os
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from collections import Counter

# Import from your existing project files
import config
from utils import create_detector_from_config, get_class_map_from_yaml
from detector import CoinDetector

# --- Application Setup ---
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = Path('static/uploads/')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Loading (IMPORTANT: Load model only once at startup) ---
# This section is run only when the application starts.
print("Loading model and detector... This may take a moment.")

# Convert string paths from config to Path objects
config.INPUTS_DIR = Path(config.INPUTS_DIR).resolve()
config.OUTPUT_DIR = Path(config.OUTPUT_DIR).resolve()
model_path = Path(config.MODEL_PATH_FOR_PREDICTION)

# Load class names from the YAML file
class_names_map = get_class_map_from_yaml(config, app.logger)
if not class_names_map:
    raise RuntimeError("Could not load class names. Check classes_names.yaml")

# Create the detector instance using your utility function
detector = create_detector_from_config(model_path, class_names_map, config, app.logger)
if not detector:
    raise RuntimeError("Failed to create the detector.")

print("Model loaded successfully.")

def calculate_summary(predictions: list) -> tuple[int, str]:
    """
    Calculates coin counts and total sum.
    Adapted from run_inference.py
    """
    if not predictions:
        return 0, "No coins detected."

    coin_counts = Counter(p['class_name'].lower().strip() for p in predictions) #
    total_sum = 0
    count_strings = []
    
    # Sort by coin value for consistent display
    sorted_coin_counts = sorted(coin_counts.items(), key=lambda x: config.COIN_VALUES.get(x[0], 0))

    for coin_name, count in sorted_coin_counts:
        coin_value = config.COIN_VALUES.get(coin_name, 0) #
        total_sum += count * coin_value #
        count_strings.append(f"{count}x {coin_name.capitalize()}")
    
    detection_summary = ", ".join(count_strings) #
    return total_sum, detection_summary

# --- Web Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Secure the filename and read the image into memory
        filename = secure_filename(file.filename)
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run prediction using the pre-loaded detector
        predictions = detector.predict(image_np, image_name=filename)

        # Process results
        total_sum, summary_text = calculate_summary(predictions)

        # Draw predictions on the image for display
        annotated_image = detector.draw_predictions_on_image(image_np, predictions, show_confidence=False)

        # Save the annotated image to the static folder to be displayed
        annotated_filename = f"annotated_{filename}"
        save_path = app.config['UPLOAD_FOLDER'] / annotated_filename
        cv2.imwrite(str(save_path), annotated_image)
        
        # Pass results to the result page
        return render_template('result.html',
                               total_sum=total_sum,
                               summary_text=summary_text,
                               image_filename=annotated_filename)

    return redirect(request.url)