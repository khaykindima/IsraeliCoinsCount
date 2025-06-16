import logging
import time
import datetime
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import shutil

# Project-specific modules
import config
from utils import (
    setup_logging,
    discover_and_pair_image_labels, split_data,
    create_yolo_dataset_yaml, validate_config_and_paths,
    create_unique_run_dir, create_detector_from_config,
    _get_relative_path_for_yolo_yaml,
    save_config_to_run_dir,
    get_class_map_from_yaml
)
from evaluate_model import YoloEvaluator

def main_train():
    """Main entry point for the training and evaluation script."""
    start_time = time.time()
    
    # Convert string paths from config to absolute Path objects for the rest of the script
    config.INPUTS_DIR = Path(config.INPUTS_DIR).resolve()
    config.OUTPUT_DIR = Path(config.OUTPUT_DIR).resolve()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_log_file = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval.log"
    logger = setup_logging(main_log_file, logger_name='yolo_main_script_logger')
    
    is_training_mode = config.EPOCHS > 0
    validation_mode = 'train' if is_training_mode else 'train_direct_eval'
    if not validate_config_and_paths(config, validation_mode, logger):
        return

    logger.info("--- Step 0 & 1: Data Preparation & Class Names ---")
    image_label_pairs, _ = discover_and_pair_image_labels(
        config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
    )
    if not image_label_pairs:
        raise FileNotFoundError("No image-label pairs found. Check INPUTS_DIR and its structure.")

    class_names_map, num_classes = load_or_derive_class_names(logger)
    if num_classes == 0:
        raise ValueError("Number of classes is zero. Cannot proceed.")

    logger.info(f"Final number of classes: {num_classes}")
    logger.info(f"Class names map: {class_names_map}")

    run_dir = None
    final_log_name = "final_run_log.log"

    if is_training_mode:
        final_log_name = f"{config.LOG_FILE_BASE_NAME}_train_final.log"
        run_dir = run_training_workflow(image_label_pairs, class_names_map, num_classes, logger)
    else:
        final_log_name = f"{config.LOG_FILE_BASE_NAME}_direct_eval_final.log"
        run_dir = run_direct_evaluation_workflow(image_label_pairs, class_names_map, logger)

    end_time = time.time()
    duration_seconds = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration_seconds))
    logger.info(f"--- Total execution time for train/evaluation run: {formatted_duration} ---")

    logging.shutdown()

    if run_dir and main_log_file.exists():
        try:
            destination_path = run_dir / final_log_name
            shutil.move(str(main_log_file), str(destination_path))
            print(f"Log file successfully moved to: {destination_path}")
        except Exception as e:
            print(f"ERROR: Failed to move log file: {e}")


def load_or_derive_class_names(logger):
    """Loads class names from the required YAML file or raises an error if not found."""
    class_names_map = get_class_map_from_yaml(config, logger)
    
    if class_names_map:
        return class_names_map, len(class_names_map)
    else:
        raise FileNotFoundError(
            f"CRITICAL: The class names file '{config.CLASS_NAMES_YAML}' was not found or is invalid. "
            f"This file is required for all operations."
        )

def run_training_workflow(pairs, class_names_map, num_classes, logger):
    """Orchestrates the model training process."""
    logger.info("--- Starting Training Workflow ---")

    if config.USE_PREDEFINED_SPLITS:
        logger.info("Configuration set to use pre-defined splits.")
        train_dir = config.INPUTS_DIR / 'train'
        val_dir = config.INPUTS_DIR / 'valid'
        test_dir = config.INPUTS_DIR / 'test'
        
        # Validate that the required directories exist if the flag is set
        if not (train_dir.is_dir() and val_dir.is_dir()):
            raise FileNotFoundError(
                f"USE_PREDEFINED_SPLITS is True, but 'train' and 'valid' directories "
                f"were not found in {config.INPUTS_DIR}."
            )
            
        logger.info("Found pre-defined 'train' and 'valid' directories.")
        # The paths for the YAML file are now fixed relative to INPUTS_DIR
        # Assumes structure like .../train/images, .../train/labels
        train_rel_img_dirs = [str(Path('train') / config.IMAGE_SUBDIR_BASENAME)]
        val_rel_img_dirs = [str(Path('valid') / config.IMAGE_SUBDIR_BASENAME)]
        
        if test_dir.is_dir():
            logger.info("Found optional 'test' directory.")
            test_rel_img_dirs = [str(Path('test') / config.IMAGE_SUBDIR_BASENAME)]
        else:
            logger.info("Optional 'test' directory not found.")
            test_rel_img_dirs = []
    else:
        logger.info("Configuration set to split all discovered data based on ratios.")
        # --- Data Splitting and YAML Creation (Original Logic) ---
        train_pairs, val_pairs, test_pairs = split_data(
            image_label_pairs=pairs, # `pairs` is the full list passed to this function
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            logger_instance=logger
        )
        train_rel_img_dirs = _get_relative_path_for_yolo_yaml(train_pairs, config.INPUTS_DIR)
        val_rel_img_dirs = _get_relative_path_for_yolo_yaml(val_pairs, config.INPUTS_DIR)
        test_rel_img_dirs = _get_relative_path_for_yolo_yaml(test_pairs, config.INPUTS_DIR)

    dataset_yaml_path = config.OUTPUT_DIR / config.DATASET_YAML_NAME

    create_yolo_dataset_yaml(
        str(config.INPUTS_DIR.resolve()),
        train_rel_img_dirs,
        val_rel_img_dirs,
        test_rel_img_dirs,
        class_names_map,
        num_classes,
        dataset_yaml_path,
        config.IMAGE_SUBDIR_BASENAME,
        config.LABEL_SUBDIR_BASENAME,
        logger
    )
    
    model = YOLO(config.MODEL_NAME_FOR_TRAINING)
    
    training_runs_base_dir = config.OUTPUT_DIR / "training_runs"
    run_name_prefix = f"{Path(config.MODEL_NAME_FOR_TRAINING).stem}_train_run"
    unique_run_dir = create_unique_run_dir(training_runs_base_dir, run_name_prefix)
    logger.info(f"Standardized training run directory created at: {unique_run_dir}")

    results = model.train(
        data=str(dataset_yaml_path), 
		epochs=config.EPOCHS, 
		imgsz=config.IMG_SIZE,
        project=str(unique_run_dir.parent), 
		name=unique_run_dir.name,
        optimizer=config.TRAINING_OPTIMIZER, 
		lr0=config.TRAINING_LR0, 
		lrf=config.TRAINING_LRF,
        exist_ok=True, # Set to True to use our pre-created unique directory
        save_period=1,
		**config.AUGMENTATION_PARAMS)
        
    run_dir = Path(results.save_dir)
	
    save_config_to_run_dir(run_dir, logger) 

    # --- Post-Training Evaluation ---
    best_model_path = _find_best_model(model, run_dir, logger)


    if best_model_path:
        # --- OPTIMIZATION: Run standard validation on the test set post-training ---
        logger.info("--- Starting Standard Ultralytics Validation on Test Set ---")
        try:
            # Re-initialize model with the best weights for validation
            validation_model = YOLO(best_model_path)
            validation_results = validation_model.val(
                data=str(dataset_yaml_path),
                split='test', # Use the test split defined in the YAML
                project=str(run_dir),
                name="standard_test_validation",
                iou=config.BOX_MATCHING_IOU_THRESHOLD
            )
            logger.info(f"Standard validation results saved in: {validation_results.save_dir}")
        except Exception as e:
            logger.exception("An error occurred during standard post-training validation.")
        
        # Run our custom detailed evaluation on all data
        logger.info("--- Starting Custom Detailed Evaluation (Post-Training) on ALL data ---")
        _run_single_evaluation(best_model_path, class_names_map, pairs, run_dir, logger)

    return run_dir


def run_direct_evaluation_workflow(pairs, class_names_map, logger):
    """Orchestrates direct evaluation of one or more pre-trained models."""
    logger.info("--- Starting Direct Evaluation Workflow ---")
    model_path_str = config.MODEL_PATH_FOR_PREDICTION
    model_path_obj = Path(model_path_str)
    base_dir = config.OUTPUT_DIR / "direct_evaluation_runs"

    run_dir = None
    if model_path_obj.is_dir():
        logger.info(f"Detected folder path. Evaluating all models in: {model_path_obj}")
        run_dir = base_dir / model_path_obj.name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Multi-model evaluation reports will be saved in: {run_dir}")
        
        model_files = sorted(list(model_path_obj.glob("*.pt")))
        if not model_files:
            logger.error(f"No model files (.pt) found in directory: {model_path_obj}")
            return run_dir
            
        summary_results = []
        for model_file in model_files:
            logger.info(f"--- Evaluating model: {model_file.name} ---")
            # Create a specific sub-directory for this model's detailed reports
            model_specific_dir = run_dir / model_file.stem
            model_specific_dir.mkdir(parents=True, exist_ok=True)

            stats = _run_single_evaluation(model_file, class_names_map, pairs, model_specific_dir, logger)
            stats['model_name'] = model_file.name
            summary_results.append(stats)
        
        # Create and save the multi-model summary report in the shared folder
        if summary_results:
            logger.info("--- Generating Multi-Model Summary Report ---")
            summary_df = pd.DataFrame(summary_results)
            column_order = ['model_name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Images with Errors']
            summary_df = summary_df[column_order]
            # Save the summary Excel file in the shared directory
            summary_excel_path = run_dir / "multi_model_evaluation_summary.xlsx"
            summary_df.to_excel(summary_excel_path, index=False, engine='openpyxl')
            logger.info(f"Multi-model summary report saved to: {summary_excel_path}")
    else:
        logger.info("Detected single model file path. Running standard evaluation.")
        run_name_prefix = f"{model_path_obj.stem}_eval"
        run_dir = create_unique_run_dir(base_dir, run_name_prefix)
        _run_single_evaluation(model_path_obj, class_names_map, pairs, run_dir, logger)
    
    return run_dir


def _find_best_model(model, run_dir, logger):
    """Finds the path to the best trained model weights."""
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and Path(model.trainer.best).exists():
        path = Path(model.trainer.best).resolve()
        logger.info(f"Best model from trainer: {path}")
        return path
    fallback_path = run_dir / "weights" / "best.pt"
    if fallback_path.exists():
        logger.info(f"Best model found at fallback path: {fallback_path}")
        return fallback_path.resolve()
    logger.warning(f"No best model found in {run_dir}. Check training outputs.")
    return None

def _run_single_evaluation(model_path_obj, class_names_map, pairs, output_dir, logger):
    """
    Helper function to run a complete evaluation for a single model and save
    results to a specified directory.
    
    Args:
        model_path_obj (Path): Path to the model file (.pt).
        class_names_map (dict): Mapping of class IDs to names.
        pairs (list): List of image-label pairs for evaluation.
        output_dir (Path): The exact directory to save the evaluation results.
        logger (Logger): The logger instance.
        
    Returns:
        dict: Overall statistics for the model.
    """

    # The calling function now provides the exact directory.
    save_config_to_run_dir(output_dir, logger)

    detector = create_detector_from_config(model_path_obj, class_names_map, config, logger)
    evaluator = YoloEvaluator(detector, logger)

    # The detailed evaluation reports (Excel, incorrect predictions) will be saved
    # directly into the provided output_dir.
    overall_stats = evaluator.perform_detailed_evaluation(
        eval_output_dir=output_dir,
        all_image_label_pairs_eval=pairs
    )
    
    return overall_stats

if __name__ == '__main__':
    main_train()
