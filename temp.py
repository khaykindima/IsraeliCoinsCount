import logging
from pathlib import Path
import cv2
import csv
import json 

import config # Assuming this is your project's config.py

from utils import (
    parse_yolo_annotations,
    calculate_iou, 
    draw_error_annotations
)
from metrics_calculator import DetectionMetricsCalculator

class YoloEvaluator:
    def __init__(self, detector, logger, config_module=config):
        self.logger = logger
        self.config = config_module 
        self.detector = detector

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        model_path_to_eval = self.detector.model_path
        class_names_map_eval = self.detector.class_names_map

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} ---")
        self.logger.info(f"Evaluation outputs will be saved in: {eval_output_dir}")

        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)

        csv_rows = []
        error_count = 0

        metrics_calc = DetectionMetricsCalculator(
            class_names_map_eval, 
            self.logger, 
            config_module=self.config
        )

        for img_path, lbl_path in all_image_label_pairs_eval:
            self.logger.info(f"Processing: {img_path.name}")
            original_img_for_drawing = cv2.imread(str(img_path))
            if original_img_for_drawing is None:
                self.logger.error(f"Cannot read image: {img_path}")
                continue
            
            img_for_prediction = original_img_for_drawing.copy()
            
            preds_from_detector = self.detector.predict(img_for_prediction) 
            gts_raw = parse_yolo_annotations(lbl_path, self.logger)
            h, w = original_img_for_drawing.shape[:2]

            current_gts_for_image_dicts = []
            for gt_class_id, gt_cx, gt_cy, gt_w, gt_h in gts_raw:
                gt_data = {
                    'cls': gt_class_id,
                    'xyxy': [(gt_cx - gt_w/2) * w, (gt_cy - gt_h/2) * h,
                             (gt_cx + gt_w/2) * w, (gt_cy + gt_h/2) * h],
                    'matched': False 
                }
                current_gts_for_image_dicts.append(gt_data)
                metrics_calc.update_stats(class_id=gt_class_id, gt_delta=1)
            
            # --- FIXED: Condition to call update_confusion_matrix ---
            # Check if confusion matrix processing is requested based on the new logic
            should_update_cm = False
            if self.config.REQUESTED_METRICS is None or not self.config.REQUESTED_METRICS: # If list is None or empty, CM is a default
                should_update_cm = True
            elif isinstance(self.config.REQUESTED_METRICS, list) and \
                 'confusion_matrix' in [m.lower() for m in self.config.REQUESTED_METRICS]:
                should_update_cm = True
            
            if should_update_cm:
                self.logger.debug(f"Updating confusion matrix for image: {img_path.name}")
                metrics_calc.update_confusion_matrix(preds_from_detector, current_gts_for_image_dicts)
            else:
                self.logger.debug(f"Skipping confusion matrix update for image: {img_path.name} as not requested.")
            # --- End of fix ---
            
            img_tp_count = 0
            img_fp_count_details = {} 
            final_preds_with_status = []

            for p_data in sorted(preds_from_detector, key=lambda x: x['conf'], reverse=True):
                is_tp = False
                pred_class_id = p_data['cls']
                for gt_idx, gt_data in enumerate(current_gts_for_image_dicts):
                    if not gt_data['matched'] and pred_class_id == gt_data['cls']:
                        iou = calculate_iou(p_data['xyxy'], gt_data['xyxy'])
                        if iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
                            current_gts_for_image_dicts[gt_idx]['matched'] = True
                            is_tp = True
                            break 
                
                if is_tp:
                    metrics_calc.update_stats(class_id=pred_class_id, tp_delta=1)
                    img_tp_count +=1
                    status = "Correct (TP)"
                else:
                    metrics_calc.update_stats(class_id=pred_class_id, fp_delta=1)
                    img_fp_count_details[pred_class_id] = img_fp_count_details.get(pred_class_id, 0) + 1
                    status = "Incorrect (FP)"
                final_preds_with_status.append({**p_data, 'status': status})

            img_fn_details_for_drawing = []
            for gt_data in current_gts_for_image_dicts:
                if not gt_data['matched']:
                    metrics_calc.update_stats(class_id=gt_data['cls'], fn_delta=1)
                    img_fn_details_for_drawing.append(gt_data)

            for p_info in final_preds_with_status:
                class_name = class_names_map_eval.get(p_info['cls'], f"ID_{p_info['cls']}")
                csv_rows.append([img_path.name, class_name, f"{p_info['conf']:.4f}", p_info['status']])

            num_img_fps = sum(img_fp_count_details.values())
            if num_img_fps > 0 or len(img_fn_details_for_drawing) > 0:
                error_count += 1
                fp_preds_to_draw = [p for p in final_preds_with_status if p['status'] == "Incorrect (FP)"]
                annotated_img = draw_error_annotations(
                    original_img_for_drawing.copy(), 
                    fp_preds_to_draw, img_fn_details_for_drawing, 
                    class_names_map_eval, self.config.BOX_COLOR_MAP, 
                    self.config.DEFAULT_BOX_COLOR, self.logger
                )
                out_path = incorrect_dir / img_path.name
                try:
                    cv2.imwrite(str(out_path), annotated_img)
                except Exception as e:
                    self.logger.error(f"Failed to save annotated error image {out_path}: {e}")

        self.logger.info(f"Total images with errors (FP or FN): {error_count}")
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

        self.logger.info("--- Final Metrics (based on CoinDetector's output) ---")
        final_metrics_results = metrics_calc.compute_metrics(
            requested_metrics=self.config.REQUESTED_METRICS,
            eval_output_dir=eval_output_dir 
        )

        for class_name, metrics in final_metrics_results.get('per_class', {}).items():
            self.logger.info(f"Class: {class_name}")
            log_line = f"  TP: {metrics.get('TP',0)}, FP: {metrics.get('FP',0)}, FN: {metrics.get('FN',0)} (GTs: {metrics.get('GT_count',0)})"
            if 'precision' in metrics: log_line += f", Precision: {metrics['precision']:.4f}"
            if 'recall' in metrics: log_line += f", Recall: {metrics['recall']:.4f}"
            if 'f1_score' in metrics: log_line += f", F1-score: {metrics['f1_score']:.4f}"
            self.logger.info(log_line)

        overall_metrics = final_metrics_results.get('overall', {})
        if overall_metrics:
            self.logger.info("--- Overall Metrics ---")
            log_line_overall = (
                f"  Total TP: {overall_metrics.get('TP',0)}, "
                f"Total FP: {overall_metrics.get('FP',0)}, "
                f"Total FN: {overall_metrics.get('FN',0)} "
                f"(Total GTs: {overall_metrics.get('GT_count',0)})"
            )
            if 'precision_micro' in overall_metrics: log_line_overall += f", Precision (Micro): {overall_metrics['precision_micro']:.4f}"
            if 'recall_micro' in overall_metrics: log_line_overall += f", Recall (Micro): {overall_metrics['recall_micro']:.4f}"
            if 'f1_score_micro' in overall_metrics: log_line_overall += f", F1-score (Micro): {overall_metrics['f1_score_micro']:.4f}"
            self.logger.info(log_line_overall)

        if final_metrics_results.get('confusion_matrix_data') is not None:
            self.logger.info(f"Confusion Matrix Data (raw array) also saved in JSON.")
        if final_metrics_results.get('confusion_matrix_plot_path_info'):
             self.logger.info(final_metrics_results['confusion_matrix_plot_path_info'])

        metrics_file_path = eval_output_dir / self.config.METRICS_JSON_NAME
        try:
            with open(metrics_file_path, 'w') as f:
                json.dump(final_metrics_results, f, indent=4)
            self.logger.info(f"Saved final metrics dictionary to: {metrics_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save final metrics JSON: {e}")

        self.logger.info(f"--- Detailed Evaluation Finished for Model: {model_path_to_eval} ---")