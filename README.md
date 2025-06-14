# Israeli Coins Detection and Counting

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive computer vision project to detect and classify Israeli coins (One, Two, Five, and Ten Shekels) using YOLOv8. The project includes a full pipeline from data preprocessing and training to in-depth model evaluation and inference.

## Table of Contents
- [Demo](#demo)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Pre-trained Models](#pre-trained-models)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [Automated Kaggle Workflow](#automated-kaggle-workflow)
- [Contributing](#contributing)
- [License](#license)

## Demo
Here is an example of the model detecting and counting coins in an image:

<p align="center">
  <img src="https://github.com/khaykindima/IsraeliCoinsCount/blob/main/demo_image.jpg?raw=true" alt="Demo Image" width="750">
  <br>
  <em>A composite image showing the annotated detection<br>and the corresponding summary printed to the console.</em>
</p>

## Key Features

* **Config-Driven Workflows**: Almost all parameters are centralized in `config.py` for easy management of experiments.
* **Flexible Data Handling**: Supports automatic data splitting by ratio or using pre-defined `train/valid/test` folders. The script can also recursively discover `images`/`labels` folders in nested subdirectories.
* **Advanced Post-Processing**: Includes a customizable pipeline to improve model accuracy by filtering predictions based on:
    * Per-class confidence thresholds.
    * Bounding box aspect ratio.
    * Optimized Non-Maximum Suppression (NMS).
* **In-Depth Evaluation**: The evaluation script generates a multi-sheet Excel report comparing model performance before and after the post-processing pipeline, providing deep insights into the model's behavior.
* **Error Analysis**: Automatically saves images of incorrect predictions (False Positives and False Negatives) for visual inspection and debugging.
* **Automated Cloud Workflow**: Includes a Kaggle notebook for automated setup, training, evaluation, and results packaging on cloud GPUs.
* **Reproducible Environments**: Provides dedicated environment files (`ultralytics_wsl_env.yml` and `ultralytics_win_env.yml`) for reproducible setups on both Linux/WSL and native Windows.

## Dataset

The dataset used for this project contains images of Israeli coins (1, 2, 5, and 10 Shekels) and is publicly available on Kaggle.

* **Dataset Link**: [Israeli Coins Dataset on Kaggle](https://www.kaggle.com/datasets/dimakhaykin/israelicoins)

## Pre-trained Models

This repository includes a `BestModels/` directory containing several well-performing model weights.

The best model to date, **`yolov8n_v5.pt`**, is recommended for direct evaluation and inference. It achieves an F1-score of **0.9982** on the test set.

## Project Structure

The project will recursively find all sibling `images` and `labels` folders within the `INPUTS_DIR`.

```
IsraeliCoinsCount/
├── BestModels/
│   └── yolov8n_v5.pt
├── Data/
│   └── CoinCount.v54/
│       ├── data.yaml
│       ├── session_1_daylight/
│       │   ├── images/
│       │   │   └── img1.jpg
│       │   └── labels/
│       │       └── img1.txt
│       └── session_2_indoor/
│           └── setup_A/
│               ├── images/
│               │   └── img2.jpg
│               └── labels/
│                   └── img2.txt
├── README.md
├── .gitignore
├── config.py
├── train.py
├── utils.py
└── ... (other project files)
```

## Setup and Installation

This project can be set up on either WSL (Linux) or native Windows. Please follow the instructions for your specific operating system.

### For WSL (Linux) Users (Recommended)

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/khaykindima/IsraeliCoinsCount.git](https://github.com/khaykindima/IsraeliCoinsCount.git)
    cd IsraeliCoinsCount
    ```
2.  **Create and Activate Conda Environment**
    ```bash
    # Create the environment from the WSL file
    conda env create -f ultralytics_wsl_env.yml

    # Activate the new environment
    conda activate ultralytics_wsl_env
    ```

### For Native Windows Users

The setup for native Windows requires a few extra steps after creating the base environment.

1.  **Clone the Repository**
    ```powershell
    git clone [https://github.com/khaykindima/IsraeliCoinsCount.git](https://github.com/khaykindima/IsraeliCoinsCount.git)
    cd IsraeliCoinsCount
    ```
2.  **Create the Base Conda Environment**
    ```powershell
    # Create the environment from the Windows file
    conda env create -f ultralytics_win_env.yml
    ```
3.  **Activate the Environment**
    ```powershell
    conda activate ultralytics_win_env
    ```
4.  **Install PyTorch Manually**
    
    PyTorch must be installed separately to ensure the correct version for your hardware is used. Run **one** of the following commands in your activated terminal.

    * **If you have an NVIDIA GPU:**
        ```powershell
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```
    * **If you do NOT have an NVIDIA GPU (CPU only):**
        ```powershell
        pip install torch torchvision torchaudio
        ```
5.  **Install ultralytics-thop Manually**
    
    This package is also required and must be installed separately.
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
2.  Set `MODEL_PATH_FOR_PREDICTION` to the path of your trained model (e.g., `BestModels/yolov8n_v5.pt`). You can also point it to a directory containing multiple `.pt` files to evaluate them all sequentially.
3.  Run the script:
    ```bash
    python train.py
    ```
    A unique evaluation folder will be created in `experiment_results/direct_evaluation_runs/`, containing detailed reports and error images.

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

### 5. Utility Scripts

* **Verify Annotations**: Check your ground truth labels by running `visualize_dataset.py`. This will save annotated images to `experiment_results/ground_truth_visualizations/`.
    ```bash
    # Visualize a random sample of 10 images
    python visualize_dataset.py --num_images 10
    ```
* **Preprocess Dataset**: To convert your entire dataset to grayscale (if needed), enable it in `config.py` and run:
    ```bash
    python preprocess_dataset.py
    ```

## Automated Kaggle Workflow

The `israelicoinscount.ipynb` notebook automates the **training and evaluation processes** on the Kaggle platform, making it easy to leverage their free GPU sessions.

1.  **Upload**: Upload the notebook to Kaggle.
2.  **Add Data**: Attach the coin dataset to the notebook.
3.  **Add Secret**: Add your GitHub Personal Access Token (PAT) as a Kaggle Secret with the label `GITHUB_PAT_ISRAELICOINS`.
4.  **Run All**: The notebook will automatically:
    * Clone the repository from GitHub.
    * Install dependencies.
    * Dynamically configure `config.py` for the Kaggle environment. You can edit the cell to set the number of `EPOCHS` for either training (`>0`) or evaluation (`0`).
    * Run the main `train.py` script.
    * Zip the output folder and provide a download link.

## Contributing
Contributions are welcome! If you have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.