import logging
from pathlib import Path
import cv2
from collections import Counter
import csv
import argparse # For command line arguments if needed for standalone evaluation

# Assuming config.py, utils.py, and detector.py are in the same directory or accessible in PYTHONPATH
try:
    import config
    from utils import (setup_logging, copy_log_to_run_directory,
                       discover_and_pair_image_labels, parse_yolo_annotations,
                       calculate_iou, draw_error_annotations, load_class_names_from_yaml, 
                       create_yolo_dataset_yaml) 
    from detector import CoinDetector 
except ImportError as e:
    print(f"ImportError: {e}. Make sure config.py, utils.py, and detector.py are in the same directory or PYTHONPATH.")
    exit()

def perform_detailed_evaluation(model_path_to_eval, class_names_map_eval, eval_output_dir, 
                                config_module, parent_logger, all_image_label_pairs_eval):
    """
    Performs detailed evaluation on a model: TP/FP/FN analysis, CSV, saves incorrect images.
    Args:
        model_path_to_eval (Path): Path to the model.pt file.
        class_names_map_eval (dict): Mapping of class IDs to names.
        eval_output_dir (Path): Directory to save evaluation outputs (CSV, incorrect images).
        config_module (module): The imported config module.
        parent_logger (logging.Logger): Logger instance to use.
        all_image_label_pairs_eval (list): List of (image_path, label_path) tuples for evaluation.
    """
    parent_logger.info(f"--- Starting Detailed Evaluation Logic for Model: {model_path_to_eval} ---")
    parent_logger.info(f"Detailed evaluation outputs will be saved in: {eval_output_dir}")

    if not class_names_map_eval:
        parent_logger.error("Class names map is empty for detailed evaluation. Cannot proceed.")
        return

    # --- Initialize Detector ---
    try:
        eval_detector = CoinDetector(
            model_path=model_path_to_eval,
            class_names_map=class_names_map_eval, 
            per_class_conf_thresholds=config_module.PER_CLASS_CONF_THRESHOLDS,
            default_conf_thresh=config_module.DEFAULT_CONF_THRESHOLD,
            iou_suppression_threshold=config_module.IOU_SUPPRESSION_THRESHOLD,
            box_color_map=config_module.BOX_COLOR_MAP,
            default_box_color=config_module.DEFAULT_BOX_COLOR
        )
    except Exception as e:
        parent_logger.exception(f"Failed to initialize CoinDetector for detailed evaluation with model {model_path_to_eval}")
        return

    incorrect_preds_base_dir = eval_output_dir / config_module.INCORRECT_PREDICTIONS_SUBDIR
    incorrect_preds_base_dir.mkdir(parents=True, exist_ok=True) # Ensure it's created here

    images_with_errors_count = 0
    prediction_data_for_csv = []

    try:
        for image_abs_path, label_abs_path in all_image_label_pairs_eval: 
            parent_logger.info(f"--- Processing image for detailed analysis: {image_abs_path.name} ---")
            image_for_processing = cv2.imread(str(image_abs_path))
            if image_for_processing is None:
                parent_logger.error(f"    Failed to read image {image_abs_path}. Skipping.")
                continue

            final_predictions = eval_detector.predict(image_for_processing.copy()) 

            ground_truth_annotations_raw = parse_yolo_annotations(label_abs_path, parent_logger)
            
            num_gt_boxes = len(ground_truth_annotations_raw)
            gt_boxes_for_matching_and_drawing = []
            img_h_for_coords, img_w_for_coords = image_for_processing.shape[:2]
            for gt_ann_raw in ground_truth_annotations_raw:
                gt_class_id, x_c, y_c, w, h = gt_ann_raw
                x1_gt = (x_c - w / 2) * img_w_for_coords
                y1_gt = (y_c - h / 2) * img_h_for_coords
                x2_gt = (x_c + w / 2) * img_w_for_coords
                y2_gt = (y_c + h / 2) * img_h_for_coords
                gt_boxes_for_matching_and_drawing.append({'cls': gt_class_id, 'xyxy': [x1_gt, y1_gt, x2_gt, y2_gt], 'matched_by_pred': False})

            num_true_positives_for_image = 0
            final_predictions_with_status = [] 
            
            sorted_pred_indices = sorted(range(len(final_predictions)), key=lambda k: final_predictions[k]['conf'], reverse=True)

            for i_sorted in sorted_pred_indices:
                pred_item = final_predictions[i_sorted]
                pred_class_id = pred_item['cls']
                pred_box_coords = pred_item['xyxy']
                
                current_pred_status = "Incorrect (FP)" 
                best_iou_for_this_pred = 0.0
                best_gt_match_idx = -1

                for gt_idx, gt_data in enumerate(gt_boxes_for_matching_and_drawing):
                    if not gt_data['matched_by_pred'] and pred_class_id == gt_data['cls']:
                        iou = calculate_iou(pred_box_coords, gt_data['xyxy'])
                        if iou > best_iou_for_this_pred: 
                            best_iou_for_this_pred = iou
                            best_gt_match_idx = gt_idx 
                
                if best_iou_for_this_pred > config_module.BOX_MATCHING_IOU_THRESHOLD:
                    if not gt_boxes_for_matching_and_drawing[best_gt_match_idx]['matched_by_pred']: 
                         current_pred_status = "Correct (TP)"
                         gt_boxes_for_matching_and_drawing[best_gt_match_idx]['matched_by_pred'] = True 
                         num_true_positives_for_image += 1
                
                final_predictions_with_status.append({**pred_item, 'correctness_status': current_pred_status})
            
            parent_logger.info(f"  Detected objects in {image_abs_path.name} (after all filters & matching):")
            if final_predictions_with_status:
                for pred_item_with_status in final_predictions_with_status:
                    predicted_class_name = class_names_map_eval.get(pred_item_with_status['cls'], f"ID_{pred_item_with_status['cls']}")
                    probability = pred_item_with_status['conf']
                    box_correctness_status_for_csv = pred_item_with_status['correctness_status']
                    
                    parent_logger.info(f"    - Class: {predicted_class_name}, Probability: {probability:.4f}, Box Correctness: {box_correctness_status_for_csv}")
                    prediction_data_for_csv.append([
                        image_abs_path.name, 
                        predicted_class_name, 
                        f"{probability:.4f}",
                        box_correctness_status_for_csv
                    ])
            else:
                parent_logger.info("    No objects detected after all filters.")
            
            num_false_positives = len(final_predictions_with_status) - num_true_positives_for_image
            num_false_negatives = num_gt_boxes - num_true_positives_for_image
            
            image_has_errors_for_saving = (num_false_positives > 0) or (num_false_negatives > 0)

            if image_has_errors_for_saving:
                images_with_errors_count += 1
                parent_logger.info(f"    IMAGE FLAGGED: {image_abs_path.name} (TP: {num_true_positives_for_image}, FP: {num_false_positives}, FN: {num_false_negatives}).")

                fp_predictions_to_draw = [p for p in final_predictions_with_status if p['correctness_status'] == "Incorrect (FP)"]
                fn_gt_to_draw = [gt for gt in gt_boxes_for_matching_and_drawing if not gt['matched_by_pred']]

                image_with_selective_annotations = draw_error_annotations( 
                    image_for_processing.copy(), 
                    fp_predictions_to_draw, 
                    fn_gt_to_draw, 
                    class_names_map_eval,
                    config_module.BOX_COLOR_MAP,
                    config_module.DEFAULT_BOX_COLOR,
                    parent_logger
                )
                
                relative_image_path = image_abs_path.relative_to(config_module.INPUTS_DIR)
                destination_path = incorrect_preds_base_dir / relative_image_path
                destination_path.parent.mkdir(parents=True, exist_ok=True) 

                try:
                    cv2.imwrite(str(destination_path), image_with_selective_annotations)
                    parent_logger.info(f"      Saved error analysis image to {destination_path}")
                except Exception as e:
                    parent_logger.error(f"      Failed to save error analysis image for {image_abs_path.name} to {destination_path}: {e}")
        
        parent_logger.info(f"Total images flagged with errors for saving: {images_with_errors_count}")

        if prediction_data_for_csv:
            csv_file_path = eval_output_dir / config_module.PREDICTIONS_CSV_NAME
            try:
                with open(csv_file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['Image Filename', 'Predicted Class', 'Probability', 'Box Correctness (Class & IoU)'])
                    csv_writer.writerows(prediction_data_for_csv)
                parent_logger.info(f"Prediction summary saved to CSV: {csv_file_path}")
            except Exception as e:
                parent_logger.error(f"Error writing prediction summary to CSV {csv_file_path}: {e}")
        else:
             parent_logger.info("No prediction data collected to write to CSV.")

    except Exception as e:
        parent_logger.exception("Error during detailed prediction analysis loop.")
    parent_logger.info(f"--- Detailed Evaluation Logic Finished for Model: {model_path_to_eval} ---")


def main_evaluate():
    # --- Setup ---
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    base_eval_run_name = f"{config.MODEL_PATH_FOR_PREDICTION.stem}_evaluation_run"
    eval_project_dir = config.OUTPUT_DIR / "evaluation_runs"
    
    current_run_dir_candidate = eval_project_dir / base_eval_run_name
    counter = 1
    while current_run_dir_candidate.exists():
        counter += 1
        current_run_dir_candidate = eval_project_dir / f"{base_eval_run_name}{counter}"
    current_run_dir = current_run_dir_candidate
    current_run_dir.mkdir(parents=True, exist_ok=False)

    main_log_file = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_evaluate.log" 
    logger = setup_logging(main_log_file, logger_name='yolo_main_evaluation_logger')
    logger.info(f"--- Starting Standalone Model Evaluation Script ---")
    logger.info(f"Evaluation outputs will be saved in: {current_run_dir}")
    logger.info(f"Using model for evaluation: {config.MODEL_PATH_FOR_PREDICTION}")

    names_from_yaml = load_class_names_from_yaml(config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME, logger)
    if names_from_yaml is None:
        logger.error(f"CRITICAL: Could not load class names from '{config.ORIGINAL_DATA_YAML_NAME}' in '{config.INPUTS_DIR}'. Exiting.")
        copy_log_to_run_directory(main_log_file, current_run_dir, f"{config.LOG_FILE_BASE_NAME}_evaluate_final.log", logger)
        return
    class_names_map = {i: str(name).strip() for i, name in enumerate(names_from_yaml)}
    if not class_names_map and not (names_from_yaml is not None and len(names_from_yaml) == 0):
        logger.error("Class names map is empty. Cannot proceed.")
        copy_log_to_run_directory(main_log_file, current_run_dir, f"{config.LOG_FILE_BASE_NAME}_evaluate_final.log", logger)
        return

    # Discover all images for evaluation (typically test set, or all if specified)
    all_image_label_pairs_for_eval, _ = discover_and_pair_image_labels(
        config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
    )
    if not all_image_label_pairs_for_eval:
        logger.warning("No image-label pairs found for evaluation.")
        copy_log_to_run_directory(main_log_file, current_run_dir, f"{config.LOG_FILE_BASE_NAME}_evaluate_final.log", logger)
        return
    
    # --- Standard YOLO Evaluation (using model.val()) ---
    logger.info("--- Performing Standard YOLO Evaluation (model.val()) ---")
    try:
        # For model.val(), we pass the CoinDetector's internal YOLO model instance
        detector_for_std_eval = CoinDetector(
            model_path=config.MODEL_PATH_FOR_PREDICTION, class_names_map=class_names_map,
            # Other params are not strictly needed for model.val() but are part of CoinDetector init
            initial_conf_thresh=config.INITIAL_PREDICT_CONF_THRESHOLD,
            per_class_conf_thresholds=config.PER_CLASS_CONF_THRESHOLDS,
            default_conf_thresh=config.DEFAULT_CONF_THRESHOLD,
            iou_suppression_threshold=config.IOU_SUPPRESSION_THRESHOLD
        )

        all_images_rel_dirs = sorted(list(set(p.parent.relative_to(config.INPUTS_DIR) for p, _ in all_image_label_pairs_for_eval)))
        num_classes_for_yaml = len(class_names_map)
        eval_dataset_yaml_path = current_run_dir / "eval_temp_dataset.yaml"
        create_yolo_dataset_yaml(
            str(config.INPUTS_DIR), [], [], [str(d) for d in all_images_rel_dirs], 
            class_names_map, num_classes_for_yaml, eval_dataset_yaml_path,
            config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger
        )
        if eval_dataset_yaml_path.exists():
            detector_for_std_eval.model.val(data=str(eval_dataset_yaml_path), split='test', 
                                          project=str(current_run_dir), name="standard_eval_results",
                                          iou=config.BOX_MATCHING_IOU_THRESHOLD)
        else:
            logger.warning("Could not create temp dataset YAML for standard evaluation.")
    except Exception as e:
        logger.exception("Error during standard model.val() evaluation.")

    # --- Perform Custom Detailed Evaluation ---
    perform_detailed_evaluation(
        model_path_to_eval=config.MODEL_PATH_FOR_PREDICTION,
        class_names_map_eval=class_names_map,
        eval_output_dir=current_run_dir, # Save in the unique eval run dir
        config_module=config,
        parent_logger=logger,
        all_image_label_pairs_eval=all_image_label_pairs_for_eval # Evaluate on all discovered images
    )
    
    copy_log_to_run_directory(main_log_file, current_run_dir, f"{config.LOG_FILE_BASE_NAME}_evaluate_final.log", logger)
    logger.info("--- Standalone Model Evaluation Script Finished ---")

if __name__ == '__main__':
    main_evaluate()
