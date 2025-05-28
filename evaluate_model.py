import logging
from pathlib import Path
import cv2
import json
from collections import Counter
import pandas as pd

import config
from bbox_utils import match_predictions, calculate_iou
from utils import (
    parse_yolo_annotations,
    draw_error_annotations,
    calculate_prf1
)
from metrics_calculator import DetectionMetricsCalculator

class YoloEvaluator:
    def __init__(self, detector, logger, config_module=config):
        self.logger = logger
        self.config = config_module
        self.detector = detector

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        model_path_to_eval = self.detector.model_path
        class_names_map = self.detector.class_names_map
        total_images = len(all_image_label_pairs_eval)

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} on {total_images} images ---")
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_calc = DetectionMetricsCalculator(class_names_map, self.logger, self.config)
        
        total_stats_before = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}
        all_detection_correctness_data = []
        error_count_before = 0
        error_count_after = 0
        
        for img_path, lbl_path in all_image_label_pairs_eval:
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                self.logger.warning(f"Could not read image {img_path}, skipping.")
                continue
            
            preds_after, preds_before = self.detector.predict(original_img.copy(), return_raw=True)
            
            h, w = original_img.shape[:2]
            gts_raw = parse_yolo_annotations(lbl_path, self.logger)
            ground_truths = [{'cls': cid, 'xyxy': [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]} for cid, cx, cy, cw, ch in gts_raw]
            
            # --- "Before" stats for comparison ---
            stats_b, _, fp_b, fn_b = match_predictions(preds_before, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
            if any(fp_b) or any(fn_b):
                error_count_before += 1
            for cn in class_names_map.values():
                for metric in ['TP', 'FP', 'FN']:
                    total_stats_before[cn][metric] += stats_b[cn][metric]

            # --- Use ONE matching function for ALL detailed "After" analysis ---
            _, tp_after, fp_after, fn_after = match_predictions(preds_after, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
            
            # --- Populate Excel data from the consistent results ---
            for pred_tp in tp_after: 
                all_detection_correctness_data.append({'image_name': img_path.name, 'class_name': pred_tp['class_name'], 'probability': pred_tp['conf'], 'box_correctness': 'TP'})
            for pred_fp in fp_after: 
                all_detection_correctness_data.append({'image_name': img_path.name, 'class_name': pred_fp['class_name'], 'probability': pred_fp['conf'], 'box_correctness': 'FP'})
            for gt_fn in fn_after: 
                all_detection_correctness_data.append({'image_name': img_path.name, 'class_name': gt_fn['class_name'], 'probability': None, 'box_correctness': 'FN'})

            metrics_calc.update_confusion_matrix(preds_after, ground_truths)
            
            if any(fp_after) or any(fn_after):
                error_count_after += 1
                gt_counts = Counter(g['cls'] for g in ground_truths)
                pred_counts = Counter(p['cls'] for p in preds_after)
                gt_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(gt_counts.items())])
                pred_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(pred_counts.items())])
                self.logger.warning(f"Incorrect Prediction in: {img_path.name}\n  - Ground Truth: [{gt_str}]\n  - Predicted   : [{pred_str}]")
                annotated_img = draw_error_annotations(original_img.copy(), fp_after, fn_after, class_names_map, self.config)
                cv2.imwrite(str(incorrect_dir / img_path.name), annotated_img)

        final_metrics = metrics_calc.compute_metrics(eval_output_dir=eval_output_dir)
        stats_after = final_metrics.get('per_class', {})

        # --- CONSOLIDATED REPORTING ---
        self._log_console_summaries(total_images, class_names_map, total_stats_before, stats_after, error_count_before, error_count_after)
        self._generate_consolidated_excel_report(
            eval_output_dir, class_names_map, total_stats_before, stats_after, 
            all_detection_correctness_data, error_count_before, error_count_after
        )

        overall = final_metrics.get('overall', {})
        self.logger.info(f"--- Overall (Micro) | Precision: {overall.get('precision_micro',0):.4f}, Recall: {overall.get('recall_micro',0):.4f}, F1-score: {overall.get('f1_score_micro',0):.4f} ---")
        metrics_file_path = eval_output_dir / "final_evaluation_metrics.json"
        with open(metrics_file_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        self.logger.info(f"Saved final metrics dictionary to: {metrics_file_path}")

    def _log_console_summaries(self, total_images, class_names_map, stats_before, stats_after, error_count_before, error_count_after):
        """Logs clear, well-formatted summaries to the console."""
        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing):  {error_count_after} / {total_images}")
        self.logger.info("-" * 70)
        
        self.logger.info("--- Post-Processing Impact and Final Metrics ---")
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            b = stats_before.get(class_name, {})
            a = stats_after.get(class_name, {})
            
            after_log_str = (
                f"TP: {a.get('TP', 0)}, FP: {a.get('FP', 0)}, FN: {a.get('FN', 0)} (GTs: {a.get('GT_count',0)}), "
                f"Precision: {a.get('precision', 0):.4f}, Recall: {a.get('recall', 0):.4f}, F1-score: {a.get('f1_score', 0):.4f}"
            )
            
            self.logger.info(f"Class: {class_name}")
            self.logger.info(f"  - Before: TP: {b.get('TP',0)}, FP: {b.get('FP',0)}, FN: {b.get('FN',0)}")
            self.logger.info(f"  - After:  {after_log_str}")
        self.logger.info("-" * 70)

    def _generate_consolidated_excel_report(self, eval_output_dir, class_names_map, stats_before, stats_after, detection_correctness_data, error_count_before, error_count_after):
        """
        Generates a single Excel file with a restructured summary and a detailed detection list.
        """
        excel_path = eval_output_dir / "evaluation_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Class Metrics Summary
                summary_data = []
                
                # Process per-class rows using the helper
                for class_id in sorted(class_names_map.keys()):
                    class_name = class_names_map[class_id]
                    row_data = {('Class', ''): class_name}
                    row_data.update(self._prepare_metrics_row(stats_before.get(class_name, {}), 'Before'))
                    row_data.update(self._prepare_metrics_row(stats_after.get(class_name, {}), 'After'))
                    summary_data.append(row_data)

                # Process Overall row using the helper
                total_stats_b = {metric: sum(s.get(metric, 0) for s in stats_before.values()) for metric in ['TP', 'FP', 'FN']}
                total_stats_a = {metric: sum(s.get(metric, 0) for s in stats_after.values()) for metric in ['TP', 'FP', 'FN']}

                overall_row_data = {('Class', ''): 'Overall (Micro)'}
                overall_row_data.update(self._prepare_metrics_row(total_stats_b, 'Before', error_count=error_count_before))
                overall_row_data.update(self._prepare_metrics_row(total_stats_a, 'After', error_count=error_count_after))
                summary_data.append(overall_row_data)
                
                df_metrics = pd.DataFrame(summary_data)
                df_metrics.columns = pd.MultiIndex.from_tuples(df_metrics.columns)
                df_metrics = df_metrics.set_index(('Class', ''))
                df_metrics.index.name = 'Class'

                # --- FIX: Define and apply the correct column order ---
                # Define the desired order of sub-columns for "Before" and "After"
                metric_order = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Images with Errors']
                
                # Get the top-level columns ('Before', 'After') in their original order
                prefixes = []
                if ('Before', 'TP') in df_metrics.columns: prefixes.append('Before')
                if ('After', 'TP') in df_metrics.columns: prefixes.append('After')

                # Build the final, ordered list of MultiIndex columns
                final_column_order = []
                for prefix in prefixes:
                    for metric in metric_order:
                        # Add the column to our final list only if it exists in the DataFrame
                        if (prefix, metric) in df_metrics.columns:
                            final_column_order.append((prefix, metric))
                
                # Reorder the DataFrame columns to match our desired order
                df_metrics = df_metrics[final_column_order]
                # --- End of fix ---

                df_metrics.to_excel(writer, sheet_name='Class Metrics Summary')

                # Sheet 2: Detailed Detections
                if detection_correctness_data:
                    df_detections = pd.DataFrame(detection_correctness_data)
                    df_detections = df_detections[['image_name', 'class_name', 'probability', 'box_correctness']].sort_values(by=['image_name', 'box_correctness', 'probability'], ascending=[True, True, False])
                    df_detections.to_excel(writer, sheet_name='Detailed Detections', index=False)

        except Exception as e:
            self.logger.error(f"Failed to generate consolidated Excel report: {e}. Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    def _prepare_metrics_row(self, stats, prefix, error_count=None):
        tp, fp, fn = stats.get('TP', 0), stats.get('FP', 0), stats.get('FN', 0)
        prf1 = calculate_prf1(tp, fp, fn)
        row_data = {
            (prefix, 'TP'): tp,
            (prefix, 'FP'): fp,
            (prefix, 'FN'): fn,
            (prefix, 'Precision'): prf1['precision'],
            (prefix, 'Recall'): prf1['recall'],
            (prefix, 'F1-Score'): prf1['f1_score'],
        }
        if error_count is not None:
            row_data[(prefix, 'Images with Errors')] = error_count
            
        return row_data