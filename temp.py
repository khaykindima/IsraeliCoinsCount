import logging
from pathlib import Path
from ultralytics import YOLO
import sys # For redirecting stdout/stderr
import io  # For redirecting stdout/stderr

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
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_log_file = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval.log"
    logger = setup_logging(main_log_file, logger_name='yolo_main_script_logger')
    final_log_suffix = "_train_or_direct_eval_final.log"

    # --- Enhanced: Centralized configuration validation ---
    is_training_mode = config.EPOCHS > 0
    validation_mode = 'train' if is_training_mode else 'train_direct_eval'
    if not validate_config_and_paths(config, validation_mode, logger):
        # If validation fails, copy the main log to a distinct failure log and exit
        failure_log_dir = config.OUTPUT_DIR / "failed_runs"
        failure_log_dir.mkdir(parents=True, exist_ok=True)
        copy_log_to_run_directory(main_log_file, failure_log_dir, f"{config.LOG_FILE_BASE_NAME}_validation_failed.log", logger)
        logger.error("Configuration validation failed. Exiting.")
        return

    run_dir_for_logs = None # Initialize, will be set if training or direct eval starts a run
    try:
        logger.info("--- Step 0 & 1: Data Preparation & Class Names ---")
        image_label_pairs, label_dirs = discover_and_pair_image_labels(
            config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
        )
        if not image_label_pairs:
            raise FileNotFoundError("No image-label pairs found. Check INPUTS_DIR and its structure.")

        class_names_map, num_classes = load_or_derive_class_names(label_dirs, logger)
        if num_classes == 0:
            raise ValueError("Number of classes is zero. Cannot proceed.")

        logger.info(f"Final number of classes: {num_classes}")
        logger.info(f"Class names map: {class_names_map}")

        if is_training_mode:
            run_dir_for_logs = run_training_workflow(image_label_pairs, class_names_map, num_classes, logger)
            final_log_suffix = "_train_final.log" # Suffix for successful training run
        else:
            run_dir_for_logs = run_direct_evaluation_workflow(image_label_pairs, class_names_map, logger)
            final_log_suffix = "_direct_eval_final.log" # Suffix for direct eval run
        # If workflow completes, copy the main log to the specific run directory
        if run_dir_for_logs:
             copy_log_to_run_directory(main_log_file, run_dir_for_logs, f"{config.LOG_FILE_BASE_NAME}{final_log_suffix}", logger)

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"A pre-run or data-related error occurred: {e}", exc_info=True)
        # Copy log to a general errors directory if a specific run_dir wasn't established
        error_log_dir = run_dir_for_logs if run_dir_for_logs else config.OUTPUT_DIR / "error_runs"
        error_log_dir.mkdir(parents=True, exist_ok=True)
        copy_log_to_run_directory(main_log_file, error_log_dir, f"{config.LOG_FILE_BASE_NAME}_error_final.log", logger)
    except RuntimeError as e:
        logger.error(f"A critical runtime error occurred: {e}", exc_info=True)
        error_log_dir = run_dir_for_logs if run_dir_for_logs else config.OUTPUT_DIR / "error_runs"
        error_log_dir.mkdir(parents=True, exist_ok=True)
        copy_log_to_run_directory(main_log_file, error_log_dir, f"{config.LOG_FILE_BASE_NAME}_runtime_error_final.log", logger)
    except Exception as e:
        logger.exception("An unexpected error occurred during the main workflow.")
        error_log_dir = run_dir_for_logs if run_dir_for_logs else config.OUTPUT_DIR / "error_runs"
        error_log_dir.mkdir(parents=True, exist_ok=True)
        copy_log_to_run_directory(main_log_file, error_log_dir, f"{config.LOG_FILE_BASE_NAME}_unexpected_error_final.log", logger)

def load_or_derive_class_names(label_dirs, logger):
    names_from_yaml = load_class_names_from_yaml(config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME, logger)
    if names_from_yaml is not None:
        return {i: str(name).strip() for i, name in enumerate(names_from_yaml)}, len(names_from_yaml)

    logger.warning(f"Could not load class names from '{config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME}'. Falling back to deriving class names from labels.")
    if not label_dirs:
        logger.error("No label directories provided to derive class names from.")
        return {}, 0

    unique_ids = get_unique_class_ids(label_dirs, logger)
    if not unique_ids:
        logger.error("No unique class IDs found in label files.")
        return {}, 0

    # Ensure class IDs are contiguous from 0
    if min(unique_ids) != 0:
        logger.warning(f"Class IDs in labels do not start from 0 (min_id: {min(unique_ids)}). This might cause issues if not handled by model/training.")
    
    num_classes = max(unique_ids) + 1
    derived_map = {i: f"class_{i}" for i in range(num_classes)} # Create for all up to max
    logger.info(f"Derived {num_classes} classes. Map: {derived_map}")
    return derived_map, num_classes


def run_training_workflow(pairs, class_names_map, num_classes, logger):
    """
    Runs the full training workflow.
    Returns the path to the run directory for log copying.
    """
    logger.info("--- Starting Training Workflow ---")

    train_pairs, val_pairs, test_pairs = split_data(
        image_label_pairs=pairs, 
        train_ratio=config.TRAIN_RATIO, 
        val_ratio=config.VAL_RATIO, 
        test_ratio=config.TEST_RATIO, 
        logger_instance=logger
    )
    # Use a subdirectory within OUTPUT_DIR for generated dataset YAMLs to keep things tidy
    dataset_yaml_dir = config.OUTPUT_DIR / "dataset_yamls"
    dataset_yaml_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path = dataset_yaml_dir / config.DATASET_YAML_NAME

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
    training_base_dir = config.OUTPUT_DIR / "training_runs" # Base for all training runs
    # Ultralytics will create a subdirectory like 'yolov8n_custom' under training_base_dir/project_name
    # Let Ultralytics handle the unique run name under the 'project' and 'name' structure.
    # The 'project' arg to model.train() defines the parent directory for the run.
    # The 'name' arg to model.train() defines the specific run's subdirectory name.
    
    run_name = f"{Path(config.MODEL_NAME_FOR_TRAINING).stem}_custom_train_run" # Define a base name for the run
    
    logger.info(f"Starting model training. Output will be under: {training_base_dir}/{run_name}")

    # Store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    
    results = None
    actual_run_dir = None

    try:
        # Redirect stdout and stderr
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        results = model.train(
            data=str(dataset_yaml_path), 
            epochs=config.EPOCHS, 
            imgsz=config.IMG_SIZE,
            project=str(training_base_dir), # Parent directory for all runs from this project
            name=run_name,                  # Specific name for this run's subdirectory
            optimizer=config.TRAINING_OPTIMIZER, 
            lr0=config.TRAINING_LR0, 
            lrf=config.TRAINING_LRF,
            exist_ok=False, # Important: Fails if run_name directory already exists
            verbose=True,   # Ensure Ultralytics own logging is verbose
            **config.AUGMENTATION_PARAMS
        )
    except Exception as e:
        logger.error(f"Error during model.train: {e}", exc_info=True)
        # Log any captured output before re-raising or handling
        ultralytics_log_output_error = captured_stdout.getvalue() + captured_stderr.getvalue()
        if ultralytics_log_output_error.strip():
            logger.info(f"--- Ultralytics Model Train Log (Partial on Error) ---\n{ultralytics_log_output_error.strip()}")
        raise # Re-raise the exception to be caught by the main try-except block
    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    # Get the captured output and log it
    ultralytics_train_log_output = captured_stdout.getvalue() + captured_stderr.getvalue()
    if ultralytics_train_log_output.strip():
        # Use logger.info or logger.debug as appropriate for the verbosity
        logger.info(f"--- Ultralytics Model Train Log ---\n{ultralytics_train_log_output.strip()}")
        logger.info(f"--- End Ultralytics Model Train Log ---")

    if results and hasattr(results, 'save_dir') and results.save_dir:
        actual_run_dir = Path(results.save_dir)
        logger.info(f"Training complete. Results saved in: {actual_run_dir}")
        save_config_to_run_dir(actual_run_dir, logger) 
        best_model_path = _find_best_model(model, actual_run_dir, logger) # model object might not be needed if results.best is reliable

        if best_model_path:
            logger.info(f"--- Starting Standard Ultralytics Validation on Test Set using {best_model_path} ---")
            try:
                validation_model = YOLO(best_model_path)
                validation_results = validation_model.val(
                    data=str(dataset_yaml_path),
                    split='test', 
                    project=str(actual_run_dir), # Save validation within the same run directory
                    name="standard_test_validation_after_training",
                    iou=config.BOX_MATCHING_IOU_THRESHOLD,
                    verbose=True # Also capture validation logs if needed (similar stdout/stderr redirection)
                )
                logger.info(f"Standard validation results saved in: {validation_results.save_dir}")
            except Exception as e:
                logger.exception("An error occurred during standard post-training validation.")

            eval_output_dir = actual_run_dir / "custom_detailed_evaluation_on_all_data_after_training"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            detector = create_detector_from_config(best_model_path, class_names_map, config, logger)
            evaluator = YoloEvaluator(detector, logger, config_module=config) # Pass config
            logger.info(f"--- Starting Custom Detailed Evaluation (Post-Training) on ALL data using {best_model_path} ---")
            evaluator.perform_detailed_evaluation(
                eval_output_dir=eval_output_dir,
                all_image_label_pairs_eval=pairs 
            )
        else:
            logger.warning("No best model path found. Skipping post-training validation and evaluation.")
    else:
        logger.error("Training did not produce expected results or save directory. Cannot proceed with post-training steps.")
        # actual_run_dir might still be None here, handle log copying in main_train's finally
    
    return actual_run_dir # Return the specific run directory for log copying


def run_direct_evaluation_workflow(pairs, class_names_map, logger):
    """
    Runs the direct evaluation workflow.
    Returns the path to the run directory for log copying.
    """
    logger.info("--- Starting Direct Evaluation Workflow ---")
    model_path_str = config.MODEL_PATH_FOR_PREDICTION
    model_path = Path(model_path_str)
    if not model_path.exists():
        logger.error(f"Model for direct evaluation not found at: {model_path}. Exiting.")
        raise FileNotFoundError(f"Model for direct evaluation not found: {model_path}")


    base_dir = config.OUTPUT_DIR / "direct_evaluation_runs"
    run_name_prefix = f"{model_path.stem}_direct_eval_run"
    run_dir = create_unique_run_dir(base_dir, run_name_prefix) # This creates the unique directory
    logger.info(f"Direct evaluation run directory: {run_dir}")
    
    save_config_to_run_dir(run_dir, logger)
    
    detector = create_detector_from_config(model_path, class_names_map, config, logger)

    logger.info(f"--- Starting Custom Detailed Evaluation using {model_path} ---")
    custom_eval_output_dir = run_dir / "custom_detailed_evaluation" 
    custom_eval_output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = YoloEvaluator(detector, logger, config_module=config) # Pass config
    evaluator.perform_detailed_evaluation(
        eval_output_dir=custom_eval_output_dir,
        all_image_label_pairs_eval=pairs
    )
    return run_dir # Return the specific run directory for log copying


def _find_best_model(model_obj, specific_run_dir, logger):
    """
    Finds the best model path.
    Args:
        model_obj: The YOLO model object after training.
        specific_run_dir (Path): The specific directory for this training run (e.g., .../runs/detect/trainX).
        logger: Logger instance.
    Returns:
        Path or None: Path to the best.pt model, or None if not found.
    """
    # Ultralytics often stores the best model path in the results object or trainer
    if hasattr(model_obj, 'best') and model_obj.best and Path(model_obj.best).exists():
        best_path = Path(model_obj.best).resolve()
        logger.info(f"Best model path from model object: {best_path}")
        return best_path
    if hasattr(model_obj, 'trainer') and hasattr(model_obj.trainer, 'best') and model_obj.trainer.best and Path(model_obj.trainer.best).exists():
        best_path = Path(model_obj.trainer.best).resolve()
        logger.info(f"Best model path from trainer attribute: {best_path}")
        return best_path
    
    # Fallback: Check the standard location within the specific run directory
    # specific_run_dir is typically like 'yolo_experiment_output/training_runs/yolov8n_v2_custom'
    fallback_path = specific_run_dir / "weights" / "best.pt"
    if fallback_path.exists():
        logger.info(f"Best model found at fallback path: {fallback_path.resolve()}")
        return fallback_path.resolve()
        
    logger.warning(f"Could not find 'best.pt' in {specific_run_dir / 'weights'}. Check training outputs.")
    return None

if __name__ == '__main__':
    main_train()
