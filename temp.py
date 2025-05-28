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
        # The logic for processing images remains the same
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
            
            stats_b, _, fp_b, fn_b = match_predictions(preds_before, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
            if any(fp_b) or any(fn_b):
                error_count_before += 1
            for cn in class_names_map.values():
                for metric in ['TP', 'FP', 'FN']:
                    total_stats_before[cn][metric] += stats_b[cn][metric]

            _, tp_after, fp_after, fn_after = match_predictions(preds_after, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
            
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

        self._log_console_summaries(total_images, class_names_map, total_stats_before, stats_after, error_count_before, error_count_after)
        self._generate_consolidated_excel_report(
            eval_output_dir, class_names_map, total_stats_before, stats_after, 
            all_detection_correctness_data, error_count_before, error_count_after
        )

        # --- RESTORED FINAL METRICS LOGGING ---
        self.logger.info("--- Final Class Metrics (After Post-Processing) ---")
        for class_name, metrics in stats_after.items():
            gt_count = metrics.get('GT_count', 0)
            log_str = (
                f"Class: {class_name} | TP: {metrics.get('TP', 0)}, FP: {metrics.get('FP', 0)}, FN: {metrics.get('FN', 0)} (GTs: {gt_count}), "
                f"Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}, F1-score: {metrics.get('f1_score', 0):.4f}"
            )
            self.logger.info(log_str)
        # --- END OF RESTORED BLOCK ---

        overall = final_metrics.get('overall', {})
        self.logger.info(f"--- Overall (Micro) | P: {overall.get('precision_micro',0):.4f}, R: {overall.get('recall_micro',0):.4f}, F1: {overall.get('f1_score_micro',0):.4f} ---")
        metrics_file_path = eval_output_dir / "final_evaluation_metrics.json"
        with open(metrics_file_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        self.logger.info(f"Saved final metrics dictionary to: {metrics_file_path}")

    def _log_console_summaries(self, total_images, class_names_map, stats_before, stats_after, error_count_before, error_count_after):
        """Logs the before-and-after summaries to the console."""
        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing):  {error_count_after} / {total_images}")
        self.logger.info("-" * 70)
        
        self.logger.info("--- Post-Processing Impact Summary (TP, FP, FN) ---")
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            b = stats_before.get(class_name, {})
            a = stats_after.get(class_name, {})
            self.logger.info(f"Class: {class_name} | Before: TP={b.get('TP',0):<4} FP={b.get('FP',0):<4} FN={b.get('FN',0):<4} | After: TP={a.get('TP',0):<4} FP={a.get('FP',0):<4} FN={a.get('FN',0):<4}")
        self.logger.info("-" * 70)

    def _generate_consolidated_excel_report(self, eval_output_dir, class_names_map, stats_before, stats_after, detection_correctness_data, error_count_before, error_count_after):
        # This function remains the same as the last version
        excel_path = eval_output_dir / "evaluation_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Class Metrics Summary
                summary_data = []
                for class_id in sorted(class_names_map.keys()):
                    class_name = class_names_map[class_id]
                    b_stats, a_stats = stats_before.get(class_name, {}), stats_after.get(class_name, {})
                    b_prf1, a_prf1 = calculate_prf1(b_stats.get('TP',0), b_stats.get('FP',0), b_stats.get('FN',0)), calculate_prf1(a_stats.get('TP',0), a_stats.get('FP',0), a_stats.get('FN',0))
                    summary_data.append({
                        ('Class', ''): class_name,
                        ('Before', 'TP'): b_stats.get('TP', 0), ('Before', 'FP'): b_stats.get('FP', 0), ('Before', 'FN'): b_stats.get('FN', 0),
                        ('Before', 'Precision'): b_prf1['precision'], ('Before', 'Recall'): b_prf1['recall'], ('Before', 'F1-Score'): b_prf1['f1_score'],
                        ('After', 'TP'): a_stats.get('TP', 0), ('After', 'FP'): a_stats.get('FP', 0), ('After', 'FN'): a_stats.get('FN', 0),
                        ('After', 'Precision'): a_prf1['precision'], ('After', 'Recall'): a_prf1['recall'], ('After', 'F1-Score'): a_prf1['f1_score'],
                    })
                
                b_tp_total, b_fp_total, b_fn_total = sum(s['TP'] for s in stats_before.values()), sum(s['FP'] for s in stats_before.values()), sum(s['FN'] for s in stats_before.values())
                a_tp_total, a_fp_total, a_fn_total = sum(s['TP'] for s in stats_after.values()), sum(s['FP'] for s in stats_after.values()), sum(s['FN'] for s in stats_after.values())
                b_prf1_total, a_prf1_total = calculate_prf1(b_tp_total, b_fp_total, b_fn_total), calculate_prf1(a_tp_total, a_fp_total, a_fn_total)
                
                summary_data.append({
                    ('Class', ''): 'Overall (Micro)',
                    ('Before', 'TP'): b_tp_total, ('Before', 'FP'): b_fp_total, ('Before', 'FN'): b_fn_total,
                    ('Before', 'Precision'): b_prf1_total['precision'], ('Before', 'Recall'): b_prf1_total['recall'], ('Before', 'F1-Score'): b_prf1_total['f1_score'],
                    ('Before', 'Images with Errors'): error_count_before,
                    ('After', 'TP'): a_tp_total, ('After', 'FP'): a_fp_total, ('After', 'FN'): a_fn_total,
                    ('After', 'Precision'): a_prf1_total['precision'], ('After', 'Recall'): a_prf1_total['recall'], ('After', 'F1-Score'): a_prf1_total['f1_score'],
                    ('After', 'Images with Errors'): error_count_after,
                })
                
                # Correct handling of MultiIndex creation
                df_metrics = pd.DataFrame(summary_data)
                df_metrics.columns = pd.MultiIndex.from_tuples(df_metrics.columns)
                df_metrics = df_metrics.set_index(('Class', ''))
                df_metrics.index.name = 'Class'
                df_metrics.to_excel(writer, sheet_name='Class Metrics Summary')

                # Sheet 2: Detailed Detections
                if detection_correctness_data:
                    df_detections = pd.DataFrame(detection_correctness_data)
                    df_detections = df_detections[['image_name', 'class_name', 'probability', 'box_correctness']].sort_values(by=['image_name', 'box_correctness', 'probability'], ascending=[True, True, False])
                    df_detections.to_excel(writer, sheet_name='Detailed Detections', index=False)
            
            self.logger.info(f"Successfully generated consolidated Excel report with 2 sheets: {excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate consolidated Excel report: {e}. Ensure 'openpyxl' is installed (`pip install openpyxl`).")