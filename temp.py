import logging
from pathlib import Path
import cv2
import json
from collections import Counter
import pandas as pd

import config
from bbox_utils import match_predictions, calculate_iou, calculate_aspect_ratio
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

    # <<< MODIFIED: This method is now much cleaner >>>
    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        """
        Performs a full evaluation loop, generating metrics and reports.
        """
        model_path_to_eval = self.detector.model_path
        class_names_map = self.detector.class_names_map
        total_images = len(all_image_label_pairs_eval)

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} on {total_images} images ---")
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_calc = DetectionMetricsCalculator(class_names_map, self.logger, self.config)
        
        # Initialize containers for aggregate stats
        agg_stats_before = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}
        agg_stats_after = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}
        all_correctness_data = []
        error_count_before, error_count_after = 0, 0
        
        for img_path, lbl_path in all_image_label_pairs_eval:
            # Process each image and get its stats
            results = self._process_single_image(img_path, lbl_path, class_names_map)
            if results is None:
                continue

            # Unpack results
            original_img, preds_after, ground_truths = results['image'], results['preds_after'], results['gts']
            stats_b_img, stats_a_img = results['stats_before'], results['stats_after']
            tp_a_img, fp_a_img, fn_a_img = results['tp_after'], results['fp_after'], results['fn_after']

            # Aggregate statistics
            for cn in class_names_map.values():
                for metric in ['TP', 'FP', 'FN']:
                    agg_stats_before[cn][metric] += stats_b_img[cn][metric]
                    agg_stats_after[cn][metric] += stats_a_img[cn][metric]
            
            # Update data for Excel report
            self._update_correctness_list(all_correctness_data, img_path.name, tp_a_img, fp_a_img, fn_a_img)
            
            # Update Ultralytics Confusion Matrix
            metrics_calc.update_confusion_matrix(preds_after, ground_truths)
            
            # Log errors and save annotated images
            error_count_before += 1 if (any(results['fp_before']) or any(results['fn_before'])) else 0
            if any(fp_a_img) or any(fn_a_img):
                error_count_after += 1
                self._log_and_save_errors(original_img, img_path, fp_a_img, fn_a_img, ground_truths, preds_after, incorrect_dir)

        # Compute final metrics and generate reports
        final_metrics_ucm = metrics_calc.compute_metrics(eval_output_dir=eval_output_dir)
        self._log_console_summaries(total_images, class_names_map, agg_stats_before, agg_stats_after, final_metrics_ucm, error_count_before, error_count_after)
        self._generate_consolidated_excel_report(eval_output_dir, class_names_map, agg_stats_before, agg_stats_after, final_metrics_ucm, all_correctness_data, error_count_before, error_count_after)
        
        self.logger.info(f"Saved final Ultralytics CM metrics dictionary to: {eval_output_dir / 'final_evaluation_metrics_ultralytics_cm.json'}")
        
        return self._prepare_summary_for_return(agg_stats_after, error_count_after)

    # <<< NEW HELPER METHOD >>>
    def _process_single_image(self, img_path, lbl_path, class_names_map):
        """Processes one image: reads it, gets predictions, and matches them to ground truths."""
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            self.logger.warning(f"Could not read image {img_path}, skipping.")
            return None
        
        preds_after, preds_before = self.detector.predict(original_img.copy(), return_raw=True, image_name=img_path.name)
        
        h, w = original_img.shape[:2]
        gts_raw = parse_yolo_annotations(lbl_path, self.logger)
        ground_truths = [{'cls': cid, 'xyxy': [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]} for cid, cx, cy, cw, ch in gts_raw]
        
        stats_b, _, fp_b, fn_b = match_predictions(preds_before, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
        stats_a, tp_a, fp_a, fn_a = match_predictions(preds_after, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
        
        return {
            'image': original_img, 'preds_after': preds_after, 'gts': ground_truths,
            'stats_before': stats_b, 'stats_after': stats_a,
            'fp_before': fp_b, 'fn_before': fn_b,
            'tp_after': tp_a, 'fp_after': fp_a, 'fn_after': fn_a
        }

    # <<< NEW HELPER METHOD >>>
    def _update_correctness_list(self, all_data, image_name, tps, fps, fns):
        """Updates the list used for the detailed Excel report."""
        for pred in tps: 
            all_data.append({'image_name': image_name, 'class_name': pred['class_name'], 'probability': pred['conf'], 'pred_aspect_ratio': calculate_aspect_ratio(pred['xyxy']),'gt_aspect_ratio': calculate_aspect_ratio(pred['matched_gt_xyxy']), 'box_correctness': 'TP'})
        for pred in fps: 
            all_data.append({'image_name': image_name, 'class_name': pred['class_name'], 'probability': pred['conf'], 'pred_aspect_ratio': calculate_aspect_ratio(pred['xyxy']), 'gt_aspect_ratio': None, 'box_correctness': 'FP'})
        for gt in fns: 
            all_data.append({'image_name': image_name, 'class_name': gt['class_name'], 'probability': None,'pred_aspect_ratio': None, 'gt_aspect_ratio': calculate_aspect_ratio(gt['xyxy']), 'box_correctness': 'FN'})

    # <<< NEW HELPER METHOD >>>
    def _log_and_save_errors(self, image, img_path, fps, fns, gts, preds, output_dir):
        """Logs details of an incorrect prediction and saves the annotated error image."""
        class_names_map = self.detector.class_names_map
        gt_counts = Counter(g['cls'] for g in gts)
        pred_counts = Counter(p['cls'] for p in preds)
        gt_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(gt_counts.items())])
        pred_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(pred_counts.items())])
        
        self.logger.warning(f"Incorrect Prediction in: {img_path.name}\n  - Ground Truth: [{gt_str}]\n  - Predicted (Post-Filter): [{pred_str}]")
        annotated_img = draw_error_annotations(image.copy(), fps, fns, class_names_map, self.config)
        cv2.imwrite(str(output_dir / img_path.name), annotated_img)

    def _log_console_summaries(self, total_images, class_names_map, 
                               stats_before, stats_after, 
                               final_metrics_ucm,
                               error_count_before, error_count_after):
        """Logs clear, well-formatted summaries to the console."""
        per_class_ucm = final_metrics_ucm.get('per_class', {})
        overall_ucm = final_metrics_ucm.get('overall', {})

        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing, Custom Match): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing, Custom Match):  {error_count_after} / {total_images}")
        self.logger.info("-" * 70)
        
        self.logger.info("--- Metrics Summary ---")
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            sbc = stats_before.get(class_name, {})
            sac = stats_after.get(class_name, {})
            sau_pc = per_class_ucm.get(class_name, {})
            
            prf1_sbc = calculate_prf1(sbc.get('TP',0), sbc.get('FP',0), sbc.get('FN',0))
            prf1_sac = calculate_prf1(sac.get('TP',0), sac.get('FP',0), sac.get('FN',0))
            
            self.logger.info(f"Class: {class_name}")
            self.logger.info(f"  - Before (Custom Match): TP: {sbc.get('TP',0)}, FP: {sbc.get('FP',0)}, FN: {sbc.get('FN',0)}, P: {prf1_sbc['precision']:.4f}, R: {prf1_sbc['recall']:.4f}, F1: {prf1_sbc['f1_score']:.4f}")
            self.logger.info(f"  - After (Custom Match):  TP: {sac.get('TP',0)}, FP: {sac.get('FP',0)}, FN: {sac.get('FN',0)}, P: {prf1_sac['precision']:.4f}, R: {prf1_sac['recall']:.4f}, F1: {prf1_sac['f1_score']:.4f}")
            self.logger.info(f"  - After (Ultralytics CM):TP: {sau_pc.get('TP',0)}, FP: {sau_pc.get('FP',0)}, FN: {sau_pc.get('FN',0)} (GTs: {sau_pc.get('GT_count',0)}), P: {sau_pc.get('precision',0):.4f}, R: {sau_pc.get('recall',0):.4f}, F1: {sau_pc.get('f1_score',0):.4f}")
        
        self.logger.info("-" * 70)
        self.logger.info("Overall (Ultralytics CM - Micro):") 
        self.logger.info(f"  TP: {overall_ucm.get('TP',0)}, FP: {overall_ucm.get('FP',0)}, FN: {overall_ucm.get('FN',0)}") 
        self.logger.info(f"  P: {overall_ucm.get('precision_micro',0):.4f}, R: {overall_ucm.get('recall_micro',0):.4f}, F1: {overall_ucm.get('f1_score_micro',0):.4f}") 
        self.logger.info("-" * 70)

    # <<< MODIFIED: This method is now much cleaner >>>
    def _generate_consolidated_excel_report(self, eval_output_dir, class_names_map, 
                                            stats_before_custom, stats_after_custom, 
                                            final_metrics_ucm,
                                            detection_correctness_data, error_count_before, error_count_after):
        """Generates a single Excel file with multiple sheets."""
        excel_path = eval_output_dir / "evaluation_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Class Metrics Summary
                self._build_summary_metrics_sheet(
                    writer, class_names_map, stats_before_custom, stats_after_custom, 
                    final_metrics_ucm, error_count_before, error_count_after
                )
                # Sheet 2: Detailed list of all detections
                self._build_detailed_detections_sheet(writer, detection_correctness_data)
        except Exception as e:
            self.logger.error(f"Failed to generate consolidated Excel report: {e}. Ensure 'openpyxl' is installed.")

    # <<< NEW HELPER METHOD >>>
    def _build_summary_metrics_sheet(self, writer, class_names_map, stats_before, stats_after, final_metrics_ucm, err_before, err_after):
        """Builds and writes the summary metrics DataFrame to an Excel sheet."""
        per_class_ucm = final_metrics_ucm.get('per_class', {})
        overall_ucm = final_metrics_ucm.get('overall', {})
        summary_data = []

        for cid in sorted(class_names_map.keys()):
            name = class_names_map[cid]
            row = {('Class', ''): name}
            row.update(self._prepare_metrics_row(stats_before.get(name, {}), 'Before Post-Process (Custom Match)'))
            row.update(self._prepare_metrics_row(stats_after.get(name, {}), 'After Post-Process (Custom Match)'))
            row.update(self._prepare_metrics_row(per_class_ucm.get(name, {}), 'After Post-Process (Ultralytics CM)'))
            summary_data.append(row)

        # Process Overall row
        overall_sbc = {m: sum(s.get(m, 0) for s in stats_before.values()) for m in ['TP', 'FP', 'FN']}
        overall_sac = {m: sum(s.get(m, 0) for s in stats_after.values()) for m in ['TP', 'FP', 'FN']}
        overall_sau_excel = {'TP': overall_ucm.get('TP', 0), 'FP': overall_ucm.get('FP', 0), 'FN': overall_ucm.get('FN', 0), 'precision': overall_ucm.get('precision_micro', 0), 'recall': overall_ucm.get('recall_micro', 0), 'f1_score': overall_ucm.get('f1_score_micro', 0)}
        
        overall_row = {('Class', ''): 'Overall (Micro)'}
        overall_row.update(self._prepare_metrics_row(overall_sbc, 'Before Post-Process (Custom Match)', err_before))
        overall_row.update(self._prepare_metrics_row(overall_sac, 'After Post-Process (Custom Match)', err_after))
        overall_row.update(self._prepare_metrics_row(overall_sau_excel, 'After Post-Process (Ultralytics CM)'))
        summary_data.append(overall_row)
        
        df_metrics = pd.DataFrame(summary_data)
        df_metrics.columns = pd.MultiIndex.from_tuples(df_metrics.columns)
        df_metrics = df_metrics.set_index(('Class', ''))
        df_metrics.index.name = 'Class'

        # Reorder columns for readability
        metric_order = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Images with Errors']
        prefixes = [p for p in ['Before Post-Process (Custom Match)', 'After Post-Process (Custom Match)', 'After Post-Process (Ultralytics CM)'] if (p, 'TP') in df_metrics.columns]
        final_column_order = [(prefix, metric) for prefix in prefixes for metric in metric_order if (prefix, metric) in df_metrics.columns]
        
        df_metrics[final_column_order].to_excel(writer, sheet_name='Class Metrics Summary')

    # <<< NEW HELPER METHOD >>>
    def _build_detailed_detections_sheet(self, writer, detection_data):
        """Builds and writes the detailed detections DataFrame to an Excel sheet."""
        if not detection_data:
            return
        df_detections = pd.DataFrame(detection_data)
        column_order = ['image_name', 'class_name', 'probability', 'pred_aspect_ratio', 'gt_aspect_ratio', 'box_correctness']
        df_detections = df_detections[column_order].sort_values(by=['image_name', 'box_correctness', 'probability'], ascending=[True, True, False])
        df_detections.to_excel(writer, sheet_name='Detailed Detections', index=False)

    def _prepare_metrics_row(self, stats, prefix, error_count=None): 
        """Helper to format a single row of metrics for the summary DataFrame."""
        tp, fp, fn = stats.get('TP', 0), stats.get('FP', 0), stats.get('FN', 0)
        
        if 'precision' in stats: # Metrics from UCM are pre-calculated
            p, r, f1 = stats['precision'], stats['recall'], stats['f1_score']
        else: # Calculate for custom match stats
            prf1 = calculate_prf1(tp, fp, fn)
            p, r, f1 = prf1['precision'], prf1['recall'], prf1['f1_score']

        row_data = {(prefix, 'TP'): tp, (prefix, 'FP'): fp, (prefix, 'FN'): fn, (prefix, 'Precision'): p, (prefix, 'Recall'): r, (prefix, 'F1-Score'): f1}
        if error_count is not None:
            row_data[(prefix, 'Images with Errors')] = error_count
            
        return row_data

    # <<< NEW HELPER METHOD >>>
    def _prepare_summary_for_return(self, stats_after, error_count_after):
        """Prepares the final dictionary to be returned for multi-model comparison."""
        overall_stats = {metric: sum(s.get(metric, 0) for s in stats_after.values()) for metric in ['TP', 'FP', 'FN']}
        prf1_overall = calculate_prf1(overall_stats['TP'], overall_stats['FP'], overall_stats['FN'])
        overall_stats['Precision'] = prf1_overall['precision']
        overall_stats['Recall'] = prf1_overall['recall']
        overall_stats['F1-Score'] = prf1_overall['f1_score']
        overall_stats['Images with Errors'] = error_count_after
        return overall_stats