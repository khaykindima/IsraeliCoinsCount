import logging
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# Project-specific modules
import config
from utils import (
    setup_logging, copy_log_to_run_directory,
    discover_and_pair_image_labels, split_data,
    get_unique_class_ids, load_class_names_from_yaml,
    create_yolo_dataset_yaml, validate_config_and_paths,
    create_unique_run_dir, create_detector_from_config,
    _get_relative_path_for_yolo_yaml,
    save_config_to_run_dir
)
from evaluate_model import YoloEvaluator

def main_train():
    """Main entry point for the training and evaluation script."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_log_file = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval.log"
    logger = setup_logging(main_log_file, logger_name='yolo_main_script_logger')
    final_log_suffix = "_train_or_direct_eval_final.log"

    # --- Enhanced: Centralized configuration validation ---
    is_training_mode = config.EPOCHS > 0
    validation_mode = 'train' if is_training_mode else 'train_direct_eval'
    if not validate_config_and_paths(config, validation_mode, logger):
        return  # Exit if validation fails

    try:
        logger.info("--- Step 0 & 1: Data Preparation & Class Names ---")
        image_label_pairs, label_dirs = discover_and_pair_image_labels(
            config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
        )
        if not image_label_pairs:
            # Use a more specific exception to be caught by the enhanced handler
            raise FileNotFoundError("No image-label pairs found. Check INPUTS_DIR and its structure.")

        class_names_map, num_classes = load_or_derive_class_names(label_dirs, logger)
        if num_classes == 0:
            raise ValueError("Number of classes is zero. Cannot proceed.")

        logger.info(f"Final number of classes: {num_classes}")
        logger.info(f"Class names map: {class_names_map}")

        if is_training_mode:
            run_training_workflow(image_label_pairs, class_names_map, num_classes, main_log_file, logger)
        else:
            run_direct_evaluation_workflow(image_label_pairs, class_names_map, main_log_file, logger)

    # --- Enhanced: More specific error handling ---
    except (FileNotFoundError, ValueError) as e:
        # Catch configuration or data-related errors
        finalize_and_exit(logger, main_log_file, None, f"A pre-run error occurred: {e}", final_log_suffix)
    except RuntimeError as e:
        # Catch critical errors during model execution (e.g., from Ultralytics library)
        finalize_and_exit(logger, main_log_file, None, f"A critical runtime error occurred: {e}", final_log_suffix)
    except Exception as e:
        # General catch-all for any other unexpected errors
        logger.exception("An unexpected error occurred during the main workflow.")
        finalize_and_exit(logger, main_log_file, None, f"An unexpected error occurred: {e}", final_log_suffix)

def load_or_derive_class_names(label_dirs, logger):
    """Loads class names from YAML, falling back to deriving them from labels."""
    names_from_yaml = load_class_names_from_yaml(config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME, logger)
    if names_from_yaml is not None:
        return {i: str(name).strip() for i, name in enumerate(names_from_yaml)}, len(names_from_yaml)

    logger.warning("Falling back to deriving class names from labels.")
    unique_ids = get_unique_class_ids(label_dirs, logger)
    num_classes = max(unique_ids) + 1 if unique_ids else 0
    return {i: f"class_{i}" for i in range(num_classes)}, num_classes

def run_training_workflow(pairs, class_names_map, num_classes, main_log_file, logger):
    """Orchestrates the model training process."""
    logger.info("--- Starting Training Workflow ---")

    # --- Data Splitting and YAML Creation ---
    train_pairs, val_pairs, test_pairs = split_data(pairs, 
	config.TRAIN_RATIO, 
	config.VAL_RATIO, 
	config.TEST_RATIO, logger
	)
	
    dataset_yaml_path = config.OUTPUT_DIR / config.DATASET_YAML_NAME

    create_yolo_dataset_yaml(
        str(config.INPUTS_DIR.resolve()),
        _get_relative_path_for_yolo_yaml(train_pairs, config.INPUTS_DIR),
        _get_relative_path_for_yolo_yaml(val_pairs, config.INPUTS_DIR),
        _get_relative_path_for_yolo_yaml(test_pairs, config.INPUTS_DIR),
        class_names_map,
        num_classes,
        dataset_yaml_path,
        config.IMAGE_SUBDIR_BASENAME,
        config.LABEL_SUBDIR_BASENAME,
        logger
    )

    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML creation failed. Expected at: {dataset_yaml_path}")

    model = YOLO(config.MODEL_NAME_FOR_TRAINING)
    training_dir = config.OUTPUT_DIR / "training_runs"
    training_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(dataset_yaml_path), 
		epochs=config.EPOCHS, 
		imgsz=config.IMG_SIZE,
        project=str(training_dir), 
		name=f"{Path(config.MODEL_NAME_FOR_TRAINING).stem}_custom",
        optimizer=config.TRAINING_OPTIMIZER, 
		lr0=config.TRAINING_LR0, 
		lrf=config.TRAINING_LRF,
        exist_ok=False, 
        save_period=1, #-1 sets to default to save ony last.pt and best.pt
		**config.AUGMENTATION_PARAMS)
        
    run_dir = Path(results.save_dir)
	
    save_config_to_run_dir(run_dir, logger) 

    # --- Post-Training Evaluation ---
    best_model_path = _find_best_model(model, run_dir, logger)


    if best_model_path:
        # --- OPTIMIZATION: Run standard validation on the test set post-training ---
        logger.info("--- Starting Standard Ultralytics Validation on Test Set ---")
        try:
            YOLO(best_model_path).val(data=str(dataset_yaml_path), 
			split='test', project=str(run_dir), 
			name="standard_test_validation", 
			iou=config.BOX_MATCHING_IOU_THRESHOLD
			)
        except Exception as e:
            logger.exception("An error occurred during standard post-training validation.")
        
        # Run our custom detailed evaluation on all data
        logger.info("--- Starting Custom Detailed Evaluation (Post-Training) on ALL data ---")
        _run_single_evaluation(best_model_path, class_names_map, pairs, run_dir, logger)

    copy_log_to_run_directory(main_log_file, run_dir, f"{config.LOG_FILE_BASE_NAME}_train_final.log", logger)

def run_direct_evaluation_workflow(pairs, class_names_map, main_log_file, logger):
    """Orchestrates direct evaluation of one or more pre-trained models."""
    logger.info("--- Starting Direct Evaluation Workflow ---")
    model_path_str = config.MODEL_PATH_FOR_PREDICTION
    model_path_obj = Path(model_path_str)
    base_dir = config.OUTPUT_DIR / "direct_evaluation_runs"

    if model_path_obj.is_dir():
        logger.info(f"Detected folder path. Evaluating all models in: {model_path_obj}")
        # Create a single shared parent folder for this multi-model run
        shared_run_dir = base_dir / model_path_obj.name
        shared_run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Multi-model evaluation reports will be saved in: {shared_run_dir}")
        
        model_files = sorted(list(model_path_obj.glob("*.pt")))
        if not model_files:
            logger.error(f"No model files (.pt) found in directory: {model_path_obj}")
            return
            
        summary_results = []
        for model_file in model_files:
            logger.info(f"--- Evaluating model: {model_file.name} ---")
            # Create a specific sub-directory for this model's detailed reports
            model_specific_dir = shared_run_dir / model_file.stem
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
            summary_excel_path = shared_run_dir / "multi_model_evaluation_summary.xlsx"
            summary_df.to_excel(summary_excel_path, index=False, engine='openpyxl')
            logger.info(f"Multi-model summary report saved to: {summary_excel_path}")
    else:
        logger.info("Detected single model file path. Running standard evaluation.")
        run_name_prefix = f"{model_path_obj.stem}_eval"
        run_dir = create_unique_run_dir(base_dir, run_name_prefix)
        _run_single_evaluation(model_path_obj, class_names_map, pairs, run_dir, logger)
        copy_log_to_run_directory(main_log_file, run_dir, f"{config.LOG_FILE_BASE_NAME}_direct_eval_final.log", logger)

def finalize_and_exit(logger, main_log_file, run_dir, message, final_log_suffix):
    """Logs a final error message and copies the log file before exiting."""
    logger.error(message)
    if run_dir:
        copy_log_to_run_directory(main_log_file, run_dir, f"{config.LOG_FILE_BASE_NAME}{final_log_suffix}", logger)

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

    # The calling function now provides the exact output directory.
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
