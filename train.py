import logging
from pathlib import Path
from ultralytics import YOLO

# Assuming config.py and utils.py are in the same directory or accessible in PYTHONPATH
try:
    import config
    from utils import (setup_logging, copy_log_to_run_directory, 
                       discover_and_pair_image_labels, split_data,
                       get_unique_class_ids, load_class_names_from_yaml, 
                       create_yolo_dataset_yaml)
    from evaluate_model import perform_detailed_evaluation # Import the reusable evaluation function
except ImportError as e:
    print(f"ImportError: {e}. Make sure config.py, utils.py, and evaluate_model.py are in the same directory or PYTHONPATH")
    exit()

# --- Main Training Script ---
def main_train(): 
    # --- Setup Run Directory and Logging ---
    # Create a general log file first, then copy to specific run dir
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_log_file = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval.log" # Log file for this script run
    logger = setup_logging(main_log_file, logger_name='yolo_main_script_logger') # General logger for this script

    # --- Data Preparation (Common for both training and direct evaluation) ---
    logger.info("--- Step 0 & 1: Data Preparation & Class Names ---")
    all_image_label_pairs, all_label_dirs = discover_and_pair_image_labels(
        config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
    )
    if not all_image_label_pairs:
        logger.error("No image-label pairs found. Cannot proceed.")
        # Attempt to save main log even if data prep fails early
        copy_log_to_run_directory(main_log_file, None, f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval_final.log", logger) 
        return

    class_names_map = {}
    num_classes = 0
    names_from_yaml = load_class_names_from_yaml(config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME, logger)

    if names_from_yaml is not None:
        logger.info(f"Loaded {len(names_from_yaml)} class names from {config.ORIGINAL_DATA_YAML_NAME}.")
        class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}
        num_classes = len(class_names_map) # Use length of map after potential stripping
    else:
        logger.warning(f"Could not load names from {config.ORIGINAL_DATA_YAML_NAME}. Attempting to derive from labels.")
        if not all_label_dirs:
            logger.error("No label directories found to derive class names. Cannot proceed.")
            copy_log_to_run_directory(main_log_file, None, f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval_final.log", logger)
            return
        unique_ids = get_unique_class_ids(all_label_dirs, logger)
        if not unique_ids:
            logger.error("No class IDs found in labels. Cannot proceed.")
            copy_log_to_run_directory(main_log_file, None, f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval_final.log", logger)
            return
        num_classes = max(unique_ids) + 1
        class_names_map = {i: f"class_{i}" for i in range(num_classes)}
        logger.info(f"Derived {num_classes} generic class names from labels.")

    logger.info(f"Final number of classes (nc): {num_classes}")
    logger.info(f"Class names map: {class_names_map}")
    if num_classes == 0 and not (names_from_yaml is not None and len(names_from_yaml) == 0) :
        logger.error("Number of classes is zero. Cannot proceed.")
        copy_log_to_run_directory(main_log_file, None, f"{config.LOG_FILE_BASE_NAME}_train_or_direct_eval_final.log", logger)
        return

    # --- Conditional Logic: Training or Direct Evaluation ---
    if config.EPOCHS > 0:
        logger.info("--- Starting YOLO Model Training Workflow ---")
        logger.info(f"EPOCHS set to {config.EPOCHS} in config.py. Proceeding with training.")

        train_pairs, val_pairs, test_pairs = split_data(
            all_image_label_pairs, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO, logger_instance=logger
        )
        
        train_rel_img_dirs = sorted(list(set(p.parent.relative_to(config.INPUTS_DIR) for p, _ in train_pairs if train_pairs)))
        val_rel_img_dirs = sorted(list(set(p.parent.relative_to(config.INPUTS_DIR) for p, _ in val_pairs if val_pairs)))
        test_rel_img_dirs = sorted(list(set(p.parent.relative_to(config.INPUTS_DIR) for p, _ in test_pairs if test_pairs)))

        logger.info("--- Step 2: Creating Dataset YAML for Training ---")
        dataset_yaml_path = config.OUTPUT_DIR / config.DATASET_YAML_NAME
        create_yolo_dataset_yaml(
            str(config.INPUTS_DIR),
            [str(d) for d in train_rel_img_dirs],
            [str(d) for d in val_rel_img_dirs],
            [str(d) for d in test_rel_img_dirs],
            class_names_map,
            num_classes,
            dataset_yaml_path,
            config.IMAGE_SUBDIR_BASENAME,
            config.LABEL_SUBDIR_BASENAME,
            logger
        )
        if not dataset_yaml_path.exists():
            logger.error(f"Failed to create {dataset_yaml_path}. Exiting training.")
            copy_log_to_run_directory(main_log_file, None, f"{config.LOG_FILE_BASE_NAME}_train_final.log", logger)
            return

    # --- Model Training ---
        logger.info("--- Step 3: Model Training ---")
        current_training_run_dir = None
        best_model_path = None
        try:
            model = YOLO(config.MODEL_NAME_FOR_TRAINING)
            
            project_base_dir = config.OUTPUT_DIR / "training_runs"
        	# Ensure project_base_dir exists
            project_base_dir.mkdir(parents=True, exist_ok=True)

            base_model_name_for_run = Path(config.MODEL_NAME_FOR_TRAINING).stem if Path(config.MODEL_NAME_FOR_TRAINING).suffix else config.MODEL_NAME_FOR_TRAINING
            training_run_name = f"{base_model_name_for_run}_custom_training" # Ultralytics will add number if it exists

            logger.info(f"Starting training with model: {config.MODEL_NAME_FOR_TRAINING}, epochs: {config.EPOCHS}, img_size: {config.IMG_SIZE}")
            logger.info(f"Dataset YAML: {dataset_yaml_path}")
            logger.info(f"Augmentations: {config.AUGMENTATION_PARAMS}")
            # Add optimizer and LR from config if they exist, otherwise use YOLO defaults
            train_args = {
                'data': str(dataset_yaml_path),
                'epochs': config.EPOCHS,
                'imgsz': config.IMG_SIZE,
                'project': str(project_base_dir),
                'name': training_run_name,
                'optimizer': config.TRAINING_OPTIMIZER,
                'lr0': config.TRAINING_LR0,
                'lrf': config.TRAINING_LRF,
                'exist_ok': False, # Important for creating new run folders (e.g., run_name2)
            }

            train_args.update(config.AUGMENTATION_PARAMS) # Add augmentation params

            results = model.train(**train_args)

            logger.info("Training completed.")
            current_training_run_dir = Path(results.save_dir)
            logger.info(f"Training run artifacts saved to: {current_training_run_dir}")

            best_model_path = None
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and model.trainer.best and Path(model.trainer.best).exists():
                best_model_path = Path(model.trainer.best).resolve()
                logger.info(f"Best model saved at (from model.trainer.best): {best_model_path}")
            elif current_training_run_dir and (current_training_run_dir / "weights" / "best.pt").exists():
                best_model_path = (current_training_run_dir / "weights" / "best.pt").resolve()
                logger.info(f"Found best model at expected path: {best_model_path}")
            else:
                logger.warning("Could not determine exact path to best.pt from training results.")

            if best_model_path:
                logger.info(f"To use this model for evaluation/prediction, update MODEL_PATH_FOR_PREDICTION in config.py to: '{str(best_model_path)}'")

        except Exception as e:
            logger.exception("An error occurred during training:")
            if 'results' in locals() and hasattr(results, 'save_dir') and Path(results.save_dir).is_dir():
                    current_training_run_dir = Path(results.save_dir) # Attempt to get run dir even on error
        finally:
            if current_training_run_dir:
                copy_log_to_run_directory(main_log_file, current_training_run_dir, f"{config.LOG_FILE_BASE_NAME}_train_final.log", logger)
            else:
                logger.warning("Training run directory not established. Final log not copied.")

        # --- Optional: Perform Detailed Evaluation on best.pt ---
        if config.EVALUATE_AFTER_TRAINING and best_model_path and current_training_run_dir:
            logger.info(f"--- Running Post-Training Detailed Evaluation on: {best_model_path} ---")
            post_eval_output_dir = current_training_run_dir / "post_train_detailed_evaluation"
            post_eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            perform_detailed_evaluation(
                model_path_to_eval=best_model_path,
                class_names_map_eval=class_names_map,
                eval_output_dir=post_eval_output_dir,
                config_module=config,
                parent_logger=logger,
                all_image_label_pairs_eval=all_image_label_pairs # Evaluate on all data
            )
            logger.info(f"Post-training detailed evaluation results saved in: {post_eval_output_dir}")
        elif config.EVALUATE_AFTER_TRAINING:
            logger.warning("Post-training evaluation was requested but 'best_model_path' or 'current_training_run_dir' is not available.")

    else: # config.EPOCHS <= 0: Run direct evaluation
        logger.info(f"EPOCHS set to {config.EPOCHS} in config.py. Skipping training and proceeding to direct evaluation.")
        
        model_to_evaluate = config.MODEL_PATH_FOR_PREDICTION
        if not model_to_evaluate.is_file():
            logger.error(f"Model for direct evaluation not found at: {model_to_evaluate} (from config.MODEL_PATH_FOR_PREDICTION). Exiting.")
            copy_log_to_run_directory(main_log_file, None, f"{config.LOG_FILE_BASE_NAME}_direct_eval_final.log", logger)
            return

        logger.info(f"--- Starting Direct Detailed Evaluation on Model: {model_to_evaluate} ---")
        
        # Create a unique directory for this direct evaluation run
        base_direct_eval_run_name = f"{model_to_evaluate.stem}_direct_eval_run"
        direct_eval_project_dir = config.OUTPUT_DIR / "direct_evaluation_runs"
        
        current_direct_eval_run_dir_candidate = direct_eval_project_dir / base_direct_eval_run_name
        counter = 1
        while current_direct_eval_run_dir_candidate.exists():
            counter += 1
            current_direct_eval_run_dir_candidate = direct_eval_project_dir / f"{base_direct_eval_run_name}{counter}"
        
        current_direct_eval_run_dir = current_direct_eval_run_dir_candidate
        current_direct_eval_run_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Direct evaluation outputs will be saved in: {current_direct_eval_run_dir}")

        perform_detailed_evaluation(
            model_path_to_eval=model_to_evaluate,
            class_names_map_eval=class_names_map,
            eval_output_dir=current_direct_eval_run_dir,
            config_module=config,
            parent_logger=logger,
            all_image_label_pairs_eval=all_image_label_pairs # Evaluate on all data
        )
        logger.info(f"Direct detailed evaluation results saved in: {current_direct_eval_run_dir}")
        copy_log_to_run_directory(main_log_file, current_direct_eval_run_dir, f"{config.LOG_FILE_BASE_NAME}_direct_eval_final.log", logger)

if __name__ == '__main__':

 main_train()
