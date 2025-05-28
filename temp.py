import logging
from pathlib import Path
import cv2
import csv
import json
from collections import defaultdict, Counter
import copy
import pandas as pd

import config
from bbox_utils import calculate_iou
from utils import (
    parse_yolo_annotations,
    draw_error_annotations,
    calculate_prf1 # Import the centralized function
)
from metrics_calculator import DetectionMetricsCalculator

class YoloEvaluator:
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

    def _calculate_stats_for_image(self, predictions, ground_truths, class_names_map):
        """
        Calculates TP, FP, FN for a single image for the purpose of detailed error analysis and visualization.
        This function does NOT accumulate totals; it identifies which specific boxes are TPs, FPs, or FNs.

        Returns:
            fp_predictions (list): List of prediction dicts that are FPs.
            fn_ground_truths (list): List of GT dicts that are FNs.
        """
        fp_predictions = []
        fn_ground_truths = []

        # Ensure GTs have a 'matched' flag for this run
        gts_processed = copy.deepcopy(ground_truths)
        for gt in gts_processed:
            gt['matched_to_pred'] = False

        predictions_processed = copy.deepcopy(predictions)

        # Iterate through predictions (sorted by confidence) to find TPs and FPs
        for pred in sorted(predictions_processed, key=lambda x: x['conf'], reverse=True):
            is_tp = False
            pred_class_id = pred['cls']
            pred['class_name'] = class_names_map.get(pred_class_id, f"ID_{pred_class_id}")

            best_match_gt_idx = -1
            max_iou = 0

            # Find the best possible GT match for the current prediction
            for gt_idx, gt in enumerate(gts_processed):
                if not gt['matched_to_pred'] and pred_class_id == gt['cls']:
                    iou = calculate_iou(pred['xyxy'], gt['xyxy'])
                    if iou > max_iou:
                        max_iou = iou
                        best_match_gt_idx = gt_idx

            # If the best match has IoU > threshold, mark it as a TP and "consume" the GT
            if max_iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
                if not gts_processed[best_match_gt_idx]['matched_to_pred']:
                    gts_processed[best_match_gt_idx]['matched_to_pred'] = True
                    is_tp = True

            if not is_tp:
                fp_predictions.append(pred)

        # Any ground truth not matched by the end is a False Negative
        for gt in gts_processed:
            if not gt['matched_to_pred']:
                gt['class_name'] = class_names_map.get(gt['cls'], f"ID_{gt['cls']}")
                fn_ground_truths.append(gt)

        return fp_predictions, fn_ground_truths

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        model_path_to_eval = self.detector.model_path
        class_names_map = self.detector.class_names_map
        total_images = len(all_image_label_pairs_eval)

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} on {total_images} images ---")
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        error_count = 0
        
        # Initialize the new stateless metrics calculator
        metrics_calc = DetectionMetricsCalculator(class_names_map, self.logger, self.config)

        for img_path, lbl_path in all_image_label_pairs_eval:
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                self.logger.warning(f"Could not read image {img_path}, skipping.")
                continue
            
            # Use return_raw=False as we only need the final predictions after NMS for evaluation
            final_predictions = self.detector.predict(original_img.copy(), return_raw=False)
            
            h, w = original_img.shape[:2]
            gts_raw_yolo_format = parse_yolo_annotations(lbl_path, self.logger)
            current_gts = [{'cls': cid, 'xyxy': [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]} for cid, cx, cy, cw, ch in gts_raw_yolo_format]
            
            # THE FIX: Feed raw predictions and GTs to the confusion matrix.
            # This is now the ONLY place where data for the final metrics is sent.
            metrics_calc.update_confusion_matrix(final_predictions, current_gts)

            # Use the helper function to get FP/FN lists for drawing error images
            fp_list, fn_list = self._calculate_stats_for_image(final_predictions, current_gts, class_names_map)

            if fp_list or fn_list:
                error_count += 1
                
                gt_class_names = [class_names_map.get(gt['cls']) for gt in current_gts]
                gt_counts = Counter(gt_class_names)
                gt_str = ", ".join([f"{count}x {name}" for name, count in sorted(gt_counts.items())]) if gt_counts else "None"

                pred_class_names = [pred['class_name'] for pred in final_predictions]
                pred_counts = Counter(pred_class_names)
                pred_str = ", ".join([f"{count}x {name}" for name, count in sorted(pred_counts.items())]) if pred_counts else "None"

                self.logger.warning(f"Incorrect Prediction in: {img_path.name}\n"
                                f"  - Ground Truth: [{gt_str}]\n"
                                f"  - Predicted   : [{pred_str}]")

                annotated_img = draw_error_annotations(original_img.copy(), fp_list, fn_list, class_names_map, self.config)
                out_path = incorrect_dir / img_path.name
                cv2.imwrite(str(out_path), annotated_img)

        # --- Generate Final Metrics and Reports ---
        self.logger.info("--- Final Corrected Metrics (Derived from Confusion Matrix) ---")
        final_metrics_results = metrics_calc.compute_metrics(
            requested_metrics=self.config.REQUESTED_METRICS,
            eval_output_dir=eval_output_dir 
        )

        self.logger.info(f"Total images with errors (FP or FN): {error_count} / {total_images}")

        # Log per-class metrics
        for class_name, metrics in final_metrics_results.get('per_class', {}).items():
            gt_count = metrics.get('TP', 0) + metrics.get('FN', 0)
            log_line = f"Class: {class_name}\n"
            log_line += f"  TP: {metrics.get('TP',0)}, FP: {metrics.get('FP',0)}, FN: {metrics.get('FN',0)} (GTs: {gt_count}), "
            log_line += f"Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}, F1-score: {metrics.get('f1_score', 0):.4f}"
            self.logger.info(log_line)

        # Log overall metrics
        overall = final_metrics_results.get('overall', {})
        if overall:
            total_gts = overall.get('TP', 0) + overall.get('FN', 0)
            log_line_overall = "--- Overall Metrics ---\n"
            log_line_overall += f"  Total TP: {overall.get('TP',0)}, Total FP: {overall.get('FP',0)}, Total FN: {overall.get('FN',0)} (Total GTs: {total_gts}), "
            log_line_overall += f"Precision (Micro): {overall.get('precision_micro',0):.4f}, Recall (Micro): {overall.get('recall_micro',0):.4f}, F1-score (Micro): {overall.get('f1_score_micro',0):.4f}"
            self.logger.info(log_line_overall)

        metrics_file_path = eval_output_dir / self.config.METRICS_JSON_NAME
        with open(metrics_file_path, 'w') as f:
            json.dump(final_metrics_results, f, indent=4)
        self.logger.info(f"Saved final metrics dictionary to: {metrics_file_path}")