# Israeli Coins Detection and Counting

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/YOLO-v8-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive computer vision project to detect and classify Israeli coins 

<img src="images/10Ag.jpg" height="40"> **10 Agorot**

<img src="images/50Ag.jpg" height="40"> **50 Agorot**

<img src="images/1Sh.jpg" height="40"> **1 Shekel**

<img src="images/2Sh.jpg" height="40"> **2 Shekels**

<img src="images/5Sh.jpg" height="40"> **5 Shekels**

<img src="images/10Sh.jpeg" height="40"> **10 Shekels**

using YOLOv8.

## Table of Contents
- [Demo](#demo)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Scripts](#project-scripts)
- [Pre-trained Models](#pre-trained-models)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [Image Acquisition Guidelines](#image-acquisition-guidelines)
- [Automated Kaggle Workflow](#automated-kaggle-workflow)
- [Contributing](#contributing)
- [License](#license)

## Demo
Here is an example of the model detecting and counting coins in an image:

<p align="center">
  <!-- For a more dynamic demo, consider creating a GIF of the inference script running on multiple images and embedding it here. -->
  <img src="https://github.com/khaykindima/IsraeliCoinsCount/blob/main/ReadmeImages/demo_image.jpg?raw=true" alt="Demo Image" width="750">
  <br>
  <em>A composite image showing the annotated detection<br>and the corresponding summary printed to the console.</em>
</p>

## Key Features

* **⚙️ Modular Configuration**: Key parameters are centralized in `config.py`, and class names are defined separately in `classes_names.yaml`, decoupling the project logic from the dataset structure.
* **📂 Flexible Data Handling**: Supports automatic data splitting by ratio or using pre-defined `train/valid/test` folders. The script can also recursively discover `images`/`labels` folders in nested subdirectories.
* **🧐 Advanced Image Quality Checks**:
    * **Blur Detection**: Automatically analyzes input images for blurriness using Laplacian variance and logs a warning for low-quality images.
    * **Darkness Detection**: Checks for underexposed images by calculating the average pixel intensity and warns if an image is too dark.
    * **Sharp Angle Warning**: Warns if an image was likely taken from a sharp angle by analyzing the aspect ratio of a significant percentage of detected objects.
    * **Cut-off Coin Detection**: Logs a warning for detected coins that touch the edge of the image, as their detection may be unreliable.
* **🔬 Advanced Post-Processing**: Includes a customizable pipeline to improve model accuracy by filtering predictions based on:
    * Per-class confidence thresholds.
    * Bounding box aspect ratio.
    * Optimized Non-Maximum Suppression (NMS).
* **📊 In-Depth Evaluation & Comparison**: The evaluation script generates a multi-sheet Excel report comparing model performance before and after the post-processing pipeline. It can also evaluate multiple models in a single run, generating a side-by-side summary report for easy comparison.
* **🖼️ Error Analysis**: Automatically saves images of incorrect predictions (False Positives and False Negatives) for visual inspection and debugging.
* **☁️ Automated Cloud Workflow**: Includes a Kaggle notebook for automated setup, training, evaluation, and results packaging on cloud GPUs.
* **📦 Reproducible Environments**: Provides dedicated environment files (`ultralytics_wsl_env.yml` and `ultralytics_win_env.yml`) for reproducible setups on both Linux/WSL and native Windows.

## Dataset

The dataset used for this project contains images of Israeli coins and is publicly available on Kaggle.

* **Dataset Link**: [Israeli Coins Dataset on Kaggle](https://www.kaggle.com/datasets/dimakhaykin/israelicoins)

## Project Scripts

This project uses a modular script-based workflow. Here is a summary of the main scripts:

| File | Purpose |
| :--- | :--- |
| `config.py` | **Central configuration file.** All paths, model names, and hyperparameters are set here. |
| `train.py` | **Main script for training and evaluation.** Trains a new model or evaluates existing ones based on the `EPOCHS` setting in `config.py`. |
| `run_inference.py` | **Runs the trained model on new images.** Takes an image or folder path and outputs annotated images and a summary of the coin values. |
| `detector.py` | **Core detection logic.** Wraps the YOLO model and applies custom post-processing like NMS and confidence filtering. |
| `evaluate_model.py` | **Performs detailed model evaluation.** Generates comprehensive reports, including Excel summaries and images of incorrect predictions. |
| `visualize_dataset.py` | **Utility script to verify annotations.** Draws ground-truth bounding boxes on images to ensure labels are correct before training. |
| `preprocess_dataset.py` | **Utility script for image preprocessing.** Can be used to apply transformations like grayscale conversion to the entire dataset. |
| `bbox_utils.py` | **Bounding box utilities.** Provides functions for IoU calculation and matching predictions to ground truths. |
| `metrics_calculator.py`| **Calculates performance metrics.** Derives Precision, Recall, and F1-score from the model's confusion matrix. |
| `utils.py` | **General helper functions.** Contains shared utilities for logging, data handling, and file operations to support the main scripts. |
| `israelicoinscount.ipynb` | **Kaggle Notebook** for automating the entire workflow (setup, train/eval, and results packaging) on the cloud. |

## Pre-trained Models

The current recommended model for direct evaluation and inference is `deployed.pt`, located in the main project folder.

## Project Structure

The project will recursively find all sibling `images` and `labels` folders within the `INPUTS_DIR`. The project's data handling logic is flexible to accommodate nested structures.

IsraeliCoinsCount/
├── Data/                     # Root folder for input datasets
│   └── IsraeliCoinsV66/
│       ├── data.yaml
│       ├── session_1_daylight/
│       │   ├── images/
│       │   │   └── img1.jpg
│       │   └── labels/
│       │       └── img1.txt
│       └── ...
├── experiment_results/       # Root folder for all outputs (not tracked by Git)
│   ├── direct_evaluation_runs/
│   ├── ground_truth_visualizations/
│   ├── inference_runs/
│   └── training_runs/
├── .gitignore
├── README.md
├── classes_names.yaml        # Defines the class names and their order
├── config.py                 # Central configuration for all scripts
├── deployed.pt               # The recommended, pre-trained model
├── train.py                  # Main script for training and evaluation
├── run_inference.py          # Script to run inference on new images
├── ultralytics_wsl_env.yml   # Conda environment for Linux/WSL
├── ultralytics_win_env.yml   # Conda environment for Windows
└── ... (other project scripts)

This project can be set up on either WSL (Linux) or native Windows. Please follow the instructions for your specific operating system.

### For WSL (Linux) Users (Recommended)

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/khaykindima/IsraeliCoinsCount.git](https://github.com/khaykindima/IsraeliCoinsCount.git)
    cd IsraeliCoinsCount
    ```
2.  **Create and Activate Conda Environment**
    The environment file will install all necessary dependencies, including PyTorch with CUDA support.
    ```bash
    # Create the environment from the WSL file
    conda env create -f ultralytics_wsl_env.yml

    # Activate the new environment
    conda activate ultralytics_wsl_env
    ```

### For Native Windows Users

1.  **Clone the Repository**
    ```powershell
    git clone [https://github.com/khaykindima/IsraeliCoinsCount.git](https://github.com/khaykindima/IsraeliCoinsCount.git)
    cd IsraeliCoinsCount
    ```
2.  **Create and Activate Conda Environment**
    The `ultralytics_win_env.yml` file will attempt to install all packages, including PyTorch, via pip.
    ```powershell
    # Create the environment from the Windows file
    conda env create -f ultralytics_win_env.yml

    # Activate the new environment
    conda activate ultralytics_win_env
    ```
3.  **Troubleshooting / Manual Installation**
    
    If the automatic installation of PyTorch fails (which can happen if the pre-configured CUDA version does not match your hardware), follow these manual steps in your activated `ultralytics_win_env` terminal:

    * **Install PyTorch Manually**: Run **one** of the following commands based on your hardware.
        * **If you have an NVIDIA GPU:**
            ```powershell
            pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
            ```
        * **If you do NOT have an NVIDIA GPU (CPU only):**
            ```powershell
            pip install torch torchvision torchaudio
            ```
    * **Install ultralytics-thop Manually**:
        ```powershell
        python -m pip install ultralytics-thop>=2.0.0
        ```

## How to Use

Once your environment is set up and activated, all workflows are controlled by `config.py`.

### 1. Configuration (`config.py`)

* `INPUTS_DIR`: Set the path to your dataset folder.
* `OUTPUT_DIR`: Set the path where all outputs (logs, models, reports) will be saved (default is `experiment_results`).
* `MODEL_PATH_FOR_PREDICTION`: Point this to the model file (`.pt`) you want to evaluate or use for inference.
* `EPOCHS`: This is the master switch.
    * Set to `> 0` to run the **training** workflow.
    * Set to `0` to run the **evaluation** workflow.

### 2. Training a New Model

1.  In `config.py`, set `EPOCHS` to a number greater than 0 (e.g., `EPOCHS = 50`).
2.  Specify the base model to use for training in `MODEL_NAME_FOR_TRAINING`.
3.  Run the training script:
    ```bash
    python train.py
    ```
    All outputs, including the trained models and evaluation reports, will be saved in a unique folder inside `experiment_results/training_runs/`.

### 3. Evaluating an Existing Model

1.  In `config.py`, set `EPOCHS = 0`.
2.  Set `MODEL_PATH_FOR_PREDICTION` to the path of your trained model (e.g., `deployed.pt`). You can also point it to a directory containing multiple `.pt` files to evaluate them all sequentially.
3.  Run the script:
    ```bash
    python train.py
    ```
    A unique evaluation folder will be created in `experiment_results/direct_evaluation_runs/`. If you evaluated a directory of models, this folder will also contain a `multi-model_evaluation_summary.xlsx` file comparing their performance.

### 4. Running Inference on New Images

1.  Run the inference script from the command line, providing a path to an image or a folder of images.
    ```bash
    # Run on a single image
    python run_inference.py /path/to/your/image.jpg

    # Run on a folder of images and export results to Excel
    python run_inference.py /path/to/your/folder/ --export_excel
    ```
    Annotated images and the optional Excel report will be saved in a unique folder inside `experiment_results/inference_runs/`.

2.  For each image, a summary of the detected coins and their total monetary value will be printed to the console.

    **Example Output:**
    ```
    INFO: --- Summary for your_image_name.jpg ---
    INFO: Detections: 2x One, 3x Two, 1x Five, 6x Ten
    INFO: Total Sum: 73 Shekels
    INFO: -----------------------------------------
    ```
    *Note: If the model detects coins belonging to the 'Other' class, they will be identified, but their value (0) will not be added to the total sum.*

### 5. Utility Scripts

* **Verify Annotations**: Check your ground truth labels by running `visualize_dataset.py`. This will save annotated images to `experiment_results/ground_truth_visualizations/`.
    ```bash
    # Visualize a random sample of 10 images
    python visualize_dataset.py --num_images 10
    ```
* **Preprocess Dataset**: To convert your entire dataset to grayscale (if enabled in `config.py`), run the `preprocess_dataset.py` script. This will create a new, processed version of your dataset.
    ```bash
    python preprocess_dataset.py
    ```

## Image Acquisition Guidelines

To ensure the most accurate results, please follow these image acquisition guidelines.

| Good Example                                                                                                     | Bad Example                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/khaykindima/IsraeliCoinsCount/blob/main/ReadmeImages/good_example_image.jpg?raw=true" width="350" height="350"> | <img src="https://github.com/khaykindima/IsraeliCoinsCount/blob/main/ReadmeImages/bad_example_image.jpg?raw=true" width="350" height="350"> |
| *Even lighting, top-down view, coins separated, plain background.* | *Low lighting, blurred image, cut coins, busy background.* |

* **Lighting**: Use bright, diffuse, and even lighting. The photo should be well-lit and not dark. Avoid direct, harsh light (like a flashlight or direct sun) that creates strong shadows or reflective glare on the coins.
* **Background**: Use a simple, non-reflective, contrasting background.
* **Camera Angle**: Shoot from directly above (top-down view). Avoid sharp angles.
* **Separation**: Spread coins out so they are not touching or are only slightly overlapping.
* **Focus**: Ensure the image is sharp and in focus.
* **Framing**: Make sure all coins are fully inside the picture frame and not cut off.

## Automated Kaggle Workflow

The `israelicoinscount.ipynb` notebook automates the setup, execution, and results retrieval process on the Kaggle platform, which is ideal for leveraging their free GPU sessions.

1.  **Setup on Kaggle**:
    * Upload the `israelicoinscount.ipynb` notebook to a new Kaggle project.
    * Add the [Israeli Coins Dataset](https://www.kaggle.com/datasets/dimakhaykin/israelicoins) as input data.
    * If you need to pull the latest code from your private GitHub repo, add your GitHub Personal Access Token (PAT) as a Kaggle Secret with the exact label `GITHUB_PAT_ISRAELICOINS`.

2.  **Execution**:
    * The notebook is pre-configured to run in **evaluation mode** (`EPOCHS = 0`). You can edit the cell to set `EPOCHS` to a value greater than 0 to run training instead.
    * The notebook will automatically:
        * Clone the repository from GitHub.
        * Copy the dataset to a writable location.
        * Install all necessary dependencies.
        * Dynamically modify `config.py` for the Kaggle environment (setting paths, epochs, etc.).
        * Run the main `train.py` script to start either evaluation or training.
        * Zip the entire output folder and provide a downloadable link for your results.

## Contributing
Contributions are welcome! If you have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
