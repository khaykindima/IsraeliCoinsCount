import logging
from pathlib import Path
import cv2
import csv
import json # For saving metrics
from collections import defaultdict, Counter # Import Counter
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

    def _calculate_stats(self, predictions, ground_truths, class_names_map):
        """
        Calculates TP, FP, FN for the purpose of detailed error analysis.
        This function is crucial for identifying which specific boxes are TPs, FPs, or FNs,
        which is needed for generating error-annotated images and detailed reports.
        
        Returns:
            stats (dict): Dictionary of TP, FP, FN counts per class.
            tp_predictions_detailed (list): List of prediction dicts that are TPs.
            fp_predictions_detailed (list): List of prediction dicts that are FPs.
            fn_ground_truths_detailed (list): List of GT dicts that are FNs.
        """
        stats = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}
        
        # Lists to store detailed information for the new Excel report
        tp_predictions_detailed = []
        fp_predictions_detailed = []
        fn_ground_truths_detailed = []
        
        # Ensure GTs have a 'matched' flag for this run
        gts_processed = copy.deepcopy(ground_truths)
        for gt in gts_processed:
            gt['matched_to_pred'] = False # Renamed for clarity
            
        # Process predictions sorted by confidence
        # Make a copy of predictions to avoid modifying the original list if it's used elsewhere
        predictions_processed = copy.deepcopy(predictions)

        for pred in sorted(predictions_processed, key=lambda x: x['conf'], reverse=True):
            is_tp = False
            pred_class_id = pred['cls']
            pred_class_name = class_names_map.get(pred_class_id, f"ID_{pred_class_id}")
            # Ensure class_name is in pred dict for easier access later
            pred['class_name'] = pred_class_name 

            best_match_gt_idx = -1
            max_iou = 0

            for gt_idx, gt in enumerate(gts_processed):
                if not gt['matched_to_pred'] and pred_class_id == gt['cls']:
                    iou = calculate_iou(pred['xyxy'], gt['xyxy'])
                    if iou > max_iou: # Find the best GT match for this prediction
                        max_iou = iou
                        best_match_gt_idx = gt_idx
            
            if max_iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
                # This prediction is a TP for the best matching GT if that GT is not already matched
                if not gts_processed[best_match_gt_idx]['matched_to_pred']:
                    gts_processed[best_match_gt_idx]['matched_to_pred'] = True
                    is_tp = True

            # for gt in gts_processed:
            #     if not gt['matched_to_pred'] and pred_class_id == gt['cls']:
            #         iou = calculate_iou(pred['xyxy'], gt['xyxy'])
            #         if iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
            #             gt['matched_to_pred'] = True
            #             is_tp = True
            #             break
            
            if is_tp:
                stats[pred_class_name]['TP'] += 1
                tp_predictions_detailed.append(pred) # Add the original prediction dict
            else:
                stats[pred_class_name]['FP'] += 1
                fp_predictions_detailed.append(pred) # Add the original prediction dict

        # Collect unmatched ground truths as False Negatives
        for gt in gts_processed:
            if not gt['matched_to_pred']:
                gt_class_name = class_names_map.get(gt['cls'], f"ID_{gt['cls']}")
                stats[gt_class_name]['FN'] += 1
                # Ensure class_name is in gt dict for easier access later
                gt['class_name'] = gt_class_name 
                fn_ground_truths_detailed.append(gt)
                
        return stats, tp_predictions_detailed, fp_predictions_detailed, fn_ground_truths_detailed

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        # Get model path and class map from the detector instance
        model_path_to_eval = self.detector.model_path
        class_names_map = self.detector.class_names_map
        total_images = len(all_image_label_pairs_eval)

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} on {total_images} images ---")
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        stats_before_postprocessing = defaultdict(lambda: defaultdict(int))
        
        error_count_before = 0
        error_count_after = 0
        
        metrics_calc = DetectionMetricsCalculator(class_names_map, self.logger, self.config)
        
        # To store data for the new detailed Excel report
        all_detection_correctness_data = []

        for img_path, lbl_path in all_image_label_pairs_eval:
            # self.logger.info(f"Processing: {img_path.name}")
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                self.logger.warning(f"Could not read image {img_path}, skipping.")
                continue
            
            preds_after_postprocessing, preds_before_postprocessing = self.detector.predict(original_img.copy(), return_raw=True)
            
            h, w = original_img.shape[:2]
            gts_raw_yolo_format = parse_yolo_annotations(lbl_path, self.logger)
            current_gts_for_matching = [{'cls': cid, 'xyxy': [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]} for cid, cx, cy, cw, ch in gts_raw_yolo_format]
            
            # Calculate stats for predictions BEFORE post-processing
            img_stats_b, _, _, _ = self._calculate_stats(preds_before_postprocessing, current_gts_for_matching, class_names_map)
            
            # Calculate stats for predictions AFTER post-processing (these are the ones for detailed report and main metrics)
            # The returned fp_after_detailed and fn_after_detailed are used for drawing error annotations
            img_stats_a, tp_after_detailed, fp_after_detailed, fn_after_detailed = self._calculate_stats(preds_after_postprocessing, current_gts_for_matching, class_names_map)

            # --- Populate data for the new detailed Excel report ---
            for pred_tp in tp_after_detailed:
                all_detection_correctness_data.append({
                    'image_name': img_path.name,
                    'class_name': pred_tp['class_name'], # class_name added in _calculate_stats
                    'probability': pred_tp['conf'],
                    'box_correctness': 'TP'
                })
            for pred_fp in fp_after_detailed:
                 all_detection_correctness_data.append({
                    'image_name': img_path.name,
                    'class_name': pred_fp['class_name'], # class_name added in _calculate_stats
                    'probability': pred_fp['conf'],
                    'box_correctness': 'FP'
                })
            for gt_fn in fn_after_detailed:
                all_detection_correctness_data.append({
                    'image_name': img_path.name,
                    'class_name': gt_fn['class_name'], # class_name added in _calculate_stats
                    'probability': None, # FNs are missed GTs, no prediction probability
                    'box_correctness': 'FN'
                })
            # --- End Excel data population ---

            if any(fp_after_detailed) or any(fn_after_detailed): # Check if there are any FPs or FNs after post-processing
                error_count_after += 1
                # --- NEW DETAILED LOGGING FOR INCORRECT IMAGES ---
                gt_class_names = [class_names_map.get(gt['cls']) for gt in current_gts_for_matching]
                gt_counts = Counter(gt_class_names)
                gt_str = ", ".join([f"{count}x {name}" for name, count in sorted(gt_counts.items())]) if gt_counts else "None"

                pred_class_names = [pred['class_name'] for pred in preds_after_postprocessing]
                pred_counts = Counter(pred_class_names)
                pred_str = ", ".join([f"{count}x {name}" for name, count in sorted(pred_counts.items())]) if pred_counts else "None"

                self.logger.warning(f"Incorrect Prediction in: {img_path.name}\n"
                                f"  - Ground Truth: [{gt_str}]\n"
                                f"  - Predicted   : [{pred_str}]")
                # --- END OF NEW LOGGING ---

                annotated_img = draw_error_annotations(original_img.copy(), fp_after_detailed, fn_after_detailed, class_names_map, self.config)
                out_path = incorrect_dir / img_path.name
                cv2.imwrite(str(out_path), annotated_img)
            
            if any(val for stat_dict in img_stats_b.values() for key, val in stat_dict.items() if key in ['FP', 'FN'] and val > 0):
                 error_count_before +=1

            for class_name_val in class_names_map.values():
                for metric in ['TP', 'FP', 'FN']:
                    stats_before_postprocessing[class_name_val][metric] += img_stats_b.get(class_name_val, {}).get(metric, 0)
            
            # The metrics calculator is the single source of truth for final metrics
            metrics_calc.update_confusion_matrix(preds_after_postprocessing, current_gts_for_matching)

        # --- Generate Final Metrics and Reports ---

        self.logger.info("--- Final Metrics from MetricsCalculator (Post-Processed) ---")
        final_metrics_results = metrics_calc.compute_metrics(
            requested_metrics=self.config.REQUESTED_METRICS,
            eval_output_dir=eval_output_dir 
        )

        # The detailed stats for the 'After' case now come directly from the metrics calculator
        stats_after_postprocessing = {cn:v for cn, v in final_metrics_results.get('per_class', {}).items()}

        self._generate_aggregate_excel_summary(eval_output_dir, class_names_map, stats_before_postprocessing, stats_after_postprocessing, error_count_before, error_count_after)
        self._generate_detection_correctness_excel(eval_output_dir, all_detection_correctness_data)
        self._log_console_summaries(total_images, class_names_map, stats_before_postprocessing, stats_after_postprocessing, error_count_before, error_count_after)

        for class_name_key, metrics_val in final_metrics_results.get('per_class', {}).items():
             self.logger.info(f"Class: {class_name_key} | TP: {metrics_val.get('TP',0)}, FP: {metrics_val.get('FP',0)}, FN: {metrics_val.get('FN',0)} | P: {metrics_val.get('precision', 0):.4f}, R: {metrics_val.get('recall', 0):.4f}, F1: {metrics_val.get('f1_score', 0):.4f}")

        overall = final_metrics_results.get('overall', {})
        if overall:
             self.logger.info(f"--- Overall (Micro) | P: {overall.get('precision_micro',0):.4f}, R: {overall.get('recall_micro',0):.4f}, F1: {overall.get('f1_score_micro',0):.4f} ---")

        metrics_file_path = eval_output_dir / self.config.METRICS_JSON_NAME
        with open(metrics_file_path, 'w') as f:
            json.dump(final_metrics_results, f, indent=4)
        self.logger.info(f"Saved final metrics dictionary to: {metrics_file_path}")

    def _log_console_summaries(self, total_images, class_names_map, stats_before, stats_after, error_count_before, error_count_after):
        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing):  {error_count_after} / {total_images}")
        self.logger.info("-" * 50)
        self.logger.info("--- Post-Processing Impact Summary (TP, FP, FN) ---")
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            b = stats_before.get(class_name, {})
            a = stats_after.get(class_name, {}) # 'stats_after' now from metrics_calc results
            delta_tp = a.get('TP', 0) - b.get('TP', 0)
            delta_fp = a.get('FP', 0) - b.get('FP', 0)
            delta_fn = a.get('FN', 0) - b.get('FN', 0)
            self.logger.info(f"Class: {class_name}")
            self.logger.info(f"  - Before: TP={b.get('TP',0):<4} FP={b.get('FP',0):<4} FN={b.get('FN',0):<4}")
            self.logger.info(f"  - After:  TP={a.get('TP',0):<4} FP={a.get('FP',0):<4} FN={a.get('FN',0):<4}")
            self.logger.info(f"  - Change: TP={delta_tp:<+4} FP={delta_fp:<+4} FN={delta_fn:<+4}")
        self.logger.info("-" * 50)

    def _generate_aggregate_excel_summary(self, eval_output_dir, class_names_map, stats_before, stats_after, error_count_before, error_count_after):
        # Renamed from _generate_excel_summary to be more specific
        summary_data = []
        
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            b_stats = stats_before.get(class_name, {})
            a_stats = stats_after.get(class_name, {})
            b_prf1 = calculate_prf1(b_stats.get('TP',0), b_stats.get('FP',0), b_stats.get('FN',0))
            a_prf1 = calculate_prf1(a_stats.get('TP',0), a_stats.get('FP',0), a_stats.get('FN',0))
            summary_data.append({
                'Class': class_name,
                'TP (Before)': b_stats.get('TP',0), 'FP (Before)': b_stats.get('FP',0), 'FN (Before)': b_stats.get('FN',0),
                'Precision (Before)': b_prf1['precision'], 'Recall (Before)': b_prf1['recall'], 'F1-Score (Before)': b_prf1['f1_score'],
                'TP (After)': a_stats.get('TP',0), 'FP (After)': a_stats.get('FP',0), 'FN (After)': a_stats.get('FN',0),
                'Precision (After)': a_prf1['precision'], 'Recall (After)': a_prf1['recall'], 'F1-Score (After)': a_prf1['f1_score'],
            })
            
        b_tp_total = sum(s.get('TP',0) for s in stats_before.values())
        b_fp_total = sum(s.get('FP',0) for s in stats_before.values())
        b_fn_total = sum(s.get('FN',0) for s in stats_before.values())
        a_tp_total = sum(s.get('TP',0) for s in stats_after.values())
        a_fp_total = sum(s.get('FP',0) for s in stats_after.values())
        a_fn_total = sum(s.get('FN',0) for s in stats_after.values())
        
        b_prf1_total = calculate_prf1(b_tp_total, b_fp_total, b_fn_total)
        a_prf1_total = calculate_prf1(a_tp_total, a_fp_total, a_fn_total)
        
        summary_data.append({
            'Class': 'Overall (Micro)',
            'TP (Before)': b_tp_total, 'FP (Before)': b_fp_total, 'FN (Before)': b_fn_total,
            'Precision (Before)': b_prf1_total['precision'], 'Recall (Before)': b_prf1_total['recall'], 'F1-Score (Before)': b_prf1_total['f1_score'],
            'TP (After)': a_tp_total, 'FP (After)': a_fp_total, 'FN (After)': a_fn_total,
            'Precision (After)': a_prf1_total['precision'], 'Recall (After)': a_prf1_total['recall'], 'F1-Score (After)': a_prf1_total['f1_score'],
        })

        df_metrics = pd.DataFrame(summary_data)
        if not df_metrics.empty:
            df_metrics = df_metrics.set_index('Class')
            # Reconstruct column names carefully to avoid errors if a part is missing
            new_cols = []
            for c in df_metrics.columns:
                parts = c.split(' ')
                if len(parts) > 1 and parts[1].startswith('(') and parts[1].endswith(')'):
                    new_cols.append((parts[1][1:-1], parts[0])) # (State, Metric)
                else:
                    new_cols.append(("N/A", c)) # Fallback if format is unexpected
            df_metrics.columns = pd.MultiIndex.from_tuples(new_cols)


        df_images = pd.DataFrame({
            'State': ['Before Post-Processing', 'After Post-Processing'],
            'Images with Errors': [error_count_before, error_count_after]
        })

        excel_path = eval_output_dir / "evaluation_aggregate_summary.xlsx" # Renamed original Excel
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                if not df_metrics.empty:
                    df_metrics.to_excel(writer, sheet_name='Class Metrics Summary')
                df_images.to_excel(writer, sheet_name='Image Error Summary', index=False)
            self.logger.info(f"Successfully generated aggregate Excel summary: {excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate aggregate Excel summary: {e}. Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    def _generate_detection_correctness_excel(self, eval_output_dir, detection_data):
        """
        Generates an Excel file detailing each detection's correctness (TP, FP, FN).
        """
        if not detection_data:
            self.logger.warning("No detailed detection data to export to Excel.")
            return

        df_detections = pd.DataFrame(detection_data)
        # Ensure desired column order
        columns_ordered = ['image_name', 'class_name', 'probability', 'box_correctness']
        df_detections = df_detections[columns_ordered]
        
        # Sort for better readability
        df_detections = df_detections.sort_values(by=['image_name', 'box_correctness', 'probability'], ascending=[True, True, False])

        excel_path = eval_output_dir / "detection_correctness_details.xlsx"
        try:
            df_detections.to_excel(excel_path, index=False, engine='openpyxl')
            self.logger.info(f"Successfully generated detailed detection correctness Excel: {excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate detailed detection correctness Excel: {e}. Ensure 'openpyxl' is installed (`pip install openpyxl`).")

