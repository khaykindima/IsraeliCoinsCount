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
  <em>A composite image showing the annotated detection and the corresponding summary printed to the console.</em>
</p>

## Key Features

* **Config-Driven Workflows**: Almost all parameters are centralized in `config.py` for easy management of experiments.
* **Flexible Data Handling**: Supports automatic data splitting by ratio or using pre-defined `train/valid/test` folders.
* **Advanced Post-Processing**: Includes a customizable pipeline to improve model accuracy by filtering predictions based on:
    * Per-class confidence thresholds.
    * Bounding box aspect ratio.
    * Optimized Non-Maximum Suppression (NMS).
* **In-Depth Evaluation**: The evaluation script generates a multi-sheet Excel report comparing model performance before and after the post-processing pipeline, providing deep insights into the model's behavior.
* **Error Analysis**: Automatically saves images of incorrect predictions (False Positives and False Negatives) for visual inspection and debugging.
* **Automated Cloud Workflow**: Includes a Kaggle notebook for automated setup, training, evaluation, and results packaging on cloud GPUs.
* **Reproducible Environments**: Comes with a `ultralytics_env.yml` file to ensure a consistent Conda environment for development and execution.

## Dataset

The dataset used for this project contains images of Israeli coins (1, 2, 5, and 10 Shekels) and is publicly available on Kaggle.

* **Dataset Link**: [Israeli Coins Dataset on Kaggle](https://www.kaggle.com/datasets/dimakhaykin/israelicoins)

## Pre-trained Models

This repository includes a `BestModels/` directory containing several well-performing model weights.

The best model to date, **`yolov8n_v5.pt`**, is recommended for direct evaluation and inference. It achieves an F1-score of **0.9982** on the test set.

## Project Structure

```
IsraeliCoinsCount/
├── BestModels/
│   └── yolov8n_v5.pt
├── Data/
│   └── CoinCount.v54/
│       ├── data.yaml
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   └── ...
│       └── test/
│           └── ...
├── README.md
├── .gitignore
├── bbox_utils.py
├── config.py
├── detector.py
├── evaluate_model.py
├── israelicoinscount.ipynb
├── metrics_calculator.py
├── preprocess_dataset.py
├── run_inference.py
├── train.py
├── ultralytics_env.yml
├── utils.py
└── visualize_dataset.py
```

## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone [https://github.com/khaykindima/IsraeliCoinsCount.git](https://github.com/khaykindima/IsraeliCoinsCount.git)
cd IsraeliCoinsCount
```

### Step 2: Set Up the Environment

It is highly recommended to use the provided Conda environment file for a consistent setup.

**Option A: Using Conda (Recommended)**

1.  Ensure you have Miniconda or Anaconda installed.
2.  Create the environment from the `ultralytics_env.yml` file:
    ```bash
    conda env create -f ultralytics_env.yml
    ```
3.  Activate the new environment:
    ```bash
    conda activate ultralytics_env
    ```

**Option B: Using pip**

If you are not using Conda, you can install the required packages using pip. Create a `requirements.txt` file with the core dependencies and install from it.

```
# requirements.txt
ultralytics==8.3.143
torch
torchvision
pandas
openpyxl
opencv-python
seaborn
matplotlib
```

Then install the packages:

```bash
pip install -r requirements.txt
```

## How to Use

All workflows are controlled by settings in the `config.py` file. Before running any script, review and adjust the configuration as needed.

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