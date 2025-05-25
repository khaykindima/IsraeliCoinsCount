import logging
from pathlib import Path
from ultralytics import YOLO

# Project-specific modules
import config
from utils import (
    setup_logging, copy_log_to_run_directory,
    discover_and_pair_image_labels, split_data,
    get_unique_class_ids, load_class_names_from_yaml,
    create_yolo_dataset_yaml, validate_config_and_paths,
    create_unique_run_dir, create_detector_from_config
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
    names_from_yaml = load_class_names_from_yaml(config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME, logger)
    if names_from_yaml is not None:
        return {i: str(name).strip() for i, name in enumerate(names_from_yaml)}, len(names_from_yaml)

    logger.warning("Falling back to deriving class names from labels.")
    if not label_dirs:
        return {}, 0

    unique_ids = get_unique_class_ids(label_dirs, logger)
    if not unique_ids:
        return {}, 0

    num_classes = max(unique_ids) + 1
    return {i: f"class_{i}" for i in range(num_classes)}, num_classes

def finalize_and_exit(logger, main_log_file, run_dir, message, final_log_suffix):
    logger.error(message)
    copy_log_to_run_directory(main_log_file, run_dir, f"{config.LOG_FILE_BASE_NAME}{final_log_suffix}", logger)
    return

def run_training_workflow(pairs, class_names_map, num_classes, main_log_file, logger):
    logger.info("--- Starting Training ---")

    # --- FIXED: Used keyword arguments to prevent passing logger as seed ---
    train_pairs, val_pairs, test_pairs = split_data(
        image_label_pairs=pairs, 
        train_ratio=config.TRAIN_RATIO, 
        val_ratio=config.VAL_RATIO, 
        test_ratio=config.TEST_RATIO, 
        logger_instance=logger
    )
    dataset_yaml_path = config.OUTPUT_DIR / config.DATASET_YAML_NAME

    create_yolo_dataset_yaml(
        str(config.INPUTS_DIR),
        _rel_dirs(train_pairs),
        _rel_dirs(val_pairs),
        _rel_dirs(test_pairs),
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

    run_name = f"{Path(config.MODEL_NAME_FOR_TRAINING).stem}_custom_training"
    train_args = {
        'data': str(dataset_yaml_path),
        'epochs': config.EPOCHS,
        'imgsz': config.IMG_SIZE,
        'project': str(training_dir),
        'name': run_name,
        'optimizer': config.TRAINING_OPTIMIZER,
        'lr0': config.TRAINING_LR0,
        'lrf': config.TRAINING_LRF,
        'exist_ok': False, # Set to False to prevent overwriting existing runs
    }
    train_args.update(config.AUGMENTATION_PARAMS)

    results = model.train(**train_args)
    run_dir = Path(results.save_dir)
    best_model_path = _find_best_model(model, run_dir, logger)


    if best_model_path:
        eval_dir = run_dir / "post_train_detailed_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        

        detector = create_detector_from_config(best_model_path, class_names_map, config, logger)
        evaluator = YoloEvaluator(detector, logger)
        evaluator.perform_detailed_evaluation(
            eval_output_dir=eval_dir,
            all_image_label_pairs_eval=pairs
        )

    copy_log_to_run_directory(main_log_file, run_dir, f"{config.LOG_FILE_BASE_NAME}_train_final.log", logger)

def run_direct_evaluation_workflow(pairs, class_names_map, main_log_file, logger):
    logger.info("--- Starting Direct Evaluation ---")
    model_path = config.MODEL_PATH_FOR_PREDICTION

    base_dir = config.OUTPUT_DIR / "direct_evaluation_runs"
    run_name_prefix = f"{model_path.stem}_direct_eval_run"
    run_dir = create_unique_run_dir(base_dir, run_name_prefix)
    logger.info(f"Created unique run directory: {run_dir}")
    
    detector = create_detector_from_config(model_path, class_names_map, config, logger)
    evaluator = YoloEvaluator(detector, logger)
    evaluator.perform_detailed_evaluation(
        eval_output_dir=run_dir,
        all_image_label_pairs_eval=pairs
    )

    copy_log_to_run_directory(main_log_file, run_dir, f"{config.LOG_FILE_BASE_NAME}_direct_eval_final.log", logger)

def _rel_dirs(pairs):
    return sorted({p.parent.relative_to(config.INPUTS_DIR) for p, _ in pairs})

def _find_best_model(model, run_dir, logger):
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and Path(model.trainer.best).exists():
        path = Path(model.trainer.best).resolve()
        logger.info(f"Best model from trainer: {path}")
        return path
    fallback = run_dir / "weights" / "best.pt"
    if fallback.exists():
        logger.info(f"Best model found at fallback path: {fallback}")
        return fallback.resolve()
    logger.warning("No best model found.")
    return None

if __name__ == '__main__':
    main_train()