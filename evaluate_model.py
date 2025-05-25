import logging
from pathlib import Path
import cv2
import csv

import config

from utils import (
    setup_logging, copy_log_to_run_directory,
    discover_and_pair_image_labels, parse_yolo_annotations,
    calculate_iou, draw_error_annotations, load_class_names_from_yaml,
    create_yolo_dataset_yaml, validate_config_and_paths,
    create_unique_run_dir, create_detector_from_config
)
from detector import CoinDetector

class YoloEvaluator:
    # --- REFACTORED: Accept a pre-built detector object ---
    def __init__(self, detector, logger, config_module=config):
        """
        Initializes the YoloEvaluator.
        Args:
            detector (CoinDetector): A pre-configured instance of CoinDetector.
            logger: A logger instance.
            config_module: The configuration module.
        """
        self.logger = logger
        self.config = config_module
        self.detector = detector

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        # Get model path and class map from the detector instance
        model_path_to_eval = self.detector.model_path
        class_names_map_eval = self.detector.class_names_map

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} ---")
        self.logger.info(f"Evaluation outputs will be saved in: {eval_output_dir}")

        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)

        csv_rows = []
        error_count = 0

        for img_path, lbl_path in all_image_label_pairs_eval:
            self.logger.info(f"Processing: {img_path.name}")
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Cannot read image: {img_path}")
                continue

            preds = self.detector.predict(img.copy())
            gts_raw = parse_yolo_annotations(lbl_path, self.logger)
            h, w = img.shape[:2]

            gts = [{
                'cls': gt[0],
                'xyxy': [(gt[1] - gt[3]/2) * w, (gt[2] - gt[4]/2) * h,
                         (gt[1] + gt[3]/2) * w, (gt[2] + gt[4]/2) * h],
                'matched': False
            } for gt in gts_raw]

            tp, fp = 0, 0
            final_preds = []

            for p in sorted(preds, key=lambda x: x['conf'], reverse=True):
                status = "Incorrect (FP)"
                for i, gt in enumerate(gts):
                    if not gt['matched'] and p['cls'] == gt['cls']:
                        iou = calculate_iou(p['xyxy'], gt['xyxy'])
                        if iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
                            gt['matched'] = True
                            tp += 1
                            status = "Correct (TP)"
                            break
                final_preds.append({**p, 'status': status})

            fn = len([g for g in gts if not g['matched']])
            fp = len(final_preds) - tp

            for p in final_preds:
                # This now correctly uses the class map from the instance
                class_name = class_names_map_eval.get(p['cls'], f"ID_{p['cls']}")
                csv_rows.append([img_path.name, class_name, f"{p['conf']:.4f}", p['status']])

            if fp > 0 or fn > 0:
                error_count += 1
                fp_preds = [p for p in final_preds if p['status'] == "Incorrect (FP)"]
                fn_gts = [g for g in gts if not g['matched']]

                annotated_img = draw_error_annotations(
                    img.copy(), fp_preds, fn_gts, class_names_map_eval,
                    self.config.BOX_COLOR_MAP, self.config.DEFAULT_BOX_COLOR,
                    self.logger
                )

                out_path = incorrect_dir / img_path.relative_to(self.config.INPUTS_DIR)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cv2.imwrite(str(out_path), annotated_img)
                    self.logger.info(f"Saved annotated error image to: {out_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save annotated image: {e}")

        self.logger.info(f"Total error images: {error_count}")
        if csv_rows:
            csv_path = eval_output_dir / self.config.PREDICTIONS_CSV_NAME
            try:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Image Filename', 'Predicted Class', 'Probability', 'Box Correctness'])
                    writer.writerows(csv_rows)
                self.logger.info(f"Saved prediction CSV to: {csv_path}")
            except Exception as e:
                self.logger.error(f"Failed to write CSV: {e}")

        self.logger.info(f"--- Evaluation Finished for Model: {model_path_to_eval} ---")

def main_evaluate():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_log = config.OUTPUT_DIR / f"{config.LOG_FILE_BASE_NAME}_evaluate.log"
    logger = setup_logging(main_log, logger_name='yolo_eval_logger')

    if not validate_config_and_paths(config, 'evaluate', logger):
        return

    run_name = f"{config.MODEL_PATH_FOR_PREDICTION.stem}_evaluation_run"
    eval_dir = config.OUTPUT_DIR / "evaluation_runs"
    run_dir = create_unique_run_dir(eval_dir, run_name)

    logger.info("--- Standalone Evaluation Start ---")
    logger.info(f"Saving to: {run_dir}")
    logger.info(f"Evaluating: {config.MODEL_PATH_FOR_PREDICTION}")

    class_names = load_class_names_from_yaml(config.INPUTS_DIR / config.ORIGINAL_DATA_YAML_NAME, logger)
    if class_names is None:
        return _exit_with_log(logger, main_log, run_dir, "Class names YAML missing.")

    class_map = {i: str(n).strip() for i, n in enumerate(class_names)}
    if not class_map:
        return _exit_with_log(logger, main_log, run_dir, "Class names map is empty.")

    pairs, _ = discover_and_pair_image_labels(config.INPUTS_DIR, config.IMAGE_SUBDIR_BASENAME,
                                              config.LABEL_SUBDIR_BASENAME, logger)
    if not pairs:
        return _exit_with_log(logger, main_log, run_dir, "No image-label pairs found.")

    # --- REFACTORED: Use the factory to create the detector ---
    detector = create_detector_from_config(
        config.MODEL_PATH_FOR_PREDICTION, class_map, config, logger
    )
    # --- REFACTORED: Pass the detector instance to the evaluator ---
    evaluator = YoloEvaluator(detector, logger)

    logger.info("--- YOLO Standard Evaluation (model.val) ---")
    try:
        rel_dirs = sorted({p.parent.relative_to(config.INPUTS_DIR) for p, _ in pairs})
        yaml_path = run_dir / "eval_temp_dataset.yaml"
        create_yolo_dataset_yaml(str(config.INPUTS_DIR), [], [], [str(d) for d in rel_dirs],
                                 class_map, len(class_map), yaml_path,
                                 config.IMAGE_SUBDIR_BASENAME, config.LABEL_SUBDIR_BASENAME, logger)

        if yaml_path.exists():
            # Use the model from the detector for the standard val method
            evaluator.detector.model.val(data=str(yaml_path), split='test',
                                         project=str(run_dir), name="standard_eval_results",
                                         iou=config.BOX_MATCHING_IOU_THRESHOLD)
        else:
            logger.warning("Failed to generate dataset YAML for val().")
    except Exception as e:
        logger.exception("model.val() failed.")

    evaluator.perform_detailed_evaluation(
        eval_output_dir=run_dir,
        all_image_label_pairs_eval=pairs
    )

    copy_log_to_run_directory(main_log, run_dir, f"{config.LOG_FILE_BASE_NAME}_evaluate_final.log", logger)
    logger.info("--- Evaluation Complete ---")

def _exit_with_log(logger, log_path, run_dir, msg):
    logger.error(msg)
    copy_log_to_run_directory(log_path, run_dir, f"{config.LOG_FILE_BASE_NAME}_evaluate_final.log", logger)

if __name__ == '__main__':
    main_evaluate()