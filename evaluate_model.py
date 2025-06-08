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

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        model_path_to_eval = self.detector.model_path
        class_names_map = self.detector.class_names_map
        total_images = len(all_image_label_pairs_eval)

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} on {total_images} images ---")
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_calc = DetectionMetricsCalculator(class_names_map, self.logger, self.config)
        
        # Metrics Set 1: Before Post-Process (Custom Match)
        aggregate_stats_before_custom = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}
        
        # Metrics Set 2: After Post-Process (Custom Match)
        aggregate_stats_after_custom = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}

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
            
            # --- "Before" stats (Custom Match) ---
            stats_b_img, _, fp_b_img, fn_b_img = match_predictions(preds_before, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
            if any(fp_b_img) or any(fn_b_img):
                error_count_before += 1
            for cn in class_names_map.values():
                for metric in ['TP', 'FP', 'FN']:
                    aggregate_stats_before_custom[cn][metric] += stats_b_img[cn][metric]

            # --- "After" stats (Custom Match) ---
            # This is also used for error images and detailed TP/FP/FN list for Excel
            stats_a_custom_img, tp_after_img, fp_after_img, fn_after_img = match_predictions(
                preds_after, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map
            )
            for cn in class_names_map.values():
                for metric in ['TP', 'FP', 'FN']:
                    aggregate_stats_after_custom[cn][metric] += stats_a_custom_img[cn][metric]
            
            # --- Populate Excel data from the "After (Custom Match)" results ---
            for pred_tp in tp_after_img: 
                all_detection_correctness_data.append({
                    'image_name': img_path.name, 
                    'class_name': pred_tp['class_name'], 
                    'probability': pred_tp['conf'], 
                    'pred_aspect_ratio': calculate_aspect_ratio(pred_tp['xyxy']),
                    'gt_aspect_ratio': calculate_aspect_ratio(pred_tp['matched_gt_xyxy']) if 'matched_gt_xyxy' in pred_tp else None, 
                    'box_correctness': 'TP'
                })
            for pred_fp in fp_after_img: 
                all_detection_correctness_data.append({
                    'image_name': img_path.name, 
                    'class_name': pred_fp['class_name'], 
                    'probability': pred_fp['conf'], 
                    'pred_aspect_ratio': calculate_aspect_ratio(pred_fp['xyxy']), 
                    'gt_aspect_ratio': None, # No GT for FP
                    'box_correctness': 'FP'
                })
            for gt_fn in fn_after_img: 
                all_detection_correctness_data.append({
                    'image_name': img_path.name, 
                    'class_name': gt_fn['class_name'], 
                    'probability': None,
                    'pred_aspect_ratio': None, 
                    'gt_aspect_ratio': calculate_aspect_ratio(gt_fn['xyxy']), 
                    'box_correctness': 'FN'
                })

            # Update Ultralytics CM with "After" predictions for Metrics Set 3
            metrics_calc.update_confusion_matrix(preds_after, ground_truths)
            
            # --- Draw Error Annotations for After ---
            if any(fp_after_img) or any(fn_after_img): # error_count_after based on custom match
                error_count_after += 1
                gt_counts = Counter(g['cls'] for g in ground_truths)
                pred_counts = Counter(p['cls'] for p in preds_after)
                gt_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(gt_counts.items())])
                pred_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(pred_counts.items())])
                self.logger.warning(f"Incorrect Prediction in: {img_path.name}\n  - Ground Truth: [{gt_str}]\n  - Predicted (Post-Filter): [{pred_str}]")
                annotated_img = draw_error_annotations(original_img.copy(), fp_after_img, fn_after_img, class_names_map, self.config)
                cv2.imwrite(str(incorrect_dir / img_path.name), annotated_img)

        # --- Metrics Set 3: After Post-Process (Ultralytics CM) ---
        final_metrics_ultralytics_cm = metrics_calc.compute_metrics(eval_output_dir=eval_output_dir)
        per_class_stats_ultralytics_cm = final_metrics_ultralytics_cm.get('per_class', {})
        overall_stats_ultralytics_cm = final_metrics_ultralytics_cm.get('overall', {})

        # --- CONSOLIDATED REPORTING ---
        self._log_console_summaries(total_images, class_names_map, 
                                    aggregate_stats_before_custom, 
                                    aggregate_stats_after_custom, # For logging, can choose one "After" or log both
                                    per_class_stats_ultralytics_cm, 
                                    overall_stats_ultralytics_cm,
                                    error_count_before, error_count_after)
        
        self._generate_consolidated_excel_report(
            eval_output_dir, class_names_map, 
            aggregate_stats_before_custom, 
            aggregate_stats_after_custom,
            per_class_stats_ultralytics_cm, 
            overall_stats_ultralytics_cm, 
            all_detection_correctness_data, 
            error_count_before, error_count_after
        )

        self.logger.info(f"--- Overall (Micro - Ultralytics CM) | Precision: {overall_stats_ultralytics_cm.get('precision_micro',0):.4f}, Recall: {overall_stats_ultralytics_cm.get('recall_micro',0):.4f}, F1-score: {overall_stats_ultralytics_cm.get('f1_score_micro',0):.4f} ---")
        metrics_file_path = eval_output_dir / "final_evaluation_metrics_ultralytics_cm.json" # Clarify filename
        with open(metrics_file_path, 'w') as f:
            json.dump(final_metrics_ultralytics_cm, f, indent=4)
        self.logger.info(f"Saved final Ultralytics CM metrics dictionary to: {metrics_file_path}")

        # Prepare the dictionary to be returned for the multi-model summary
        overall_sac = {metric: sum(s.get(metric, 0) for s in aggregate_stats_after_custom.values()) for metric in ['TP', 'FP', 'FN']}
        prf1_sac_overall = calculate_prf1(overall_sac['TP'], overall_sac['FP'], overall_sac['FN'])
        overall_sac['Precision'] = prf1_sac_overall['precision']
        overall_sac['Recall'] = prf1_sac_overall['recall']
        overall_sac['F1-Score'] = prf1_sac_overall['f1_score']
        overall_sac['Images with Errors'] = error_count_after

        # Return the calculated overall stats for the "After Post-Process (Custom Match)"
        return overall_sac


    def _log_console_summaries(self, total_images, class_names_map, 
                               stats_before_custom, stats_after_custom, 
                               per_class_stats_ucm, overall_stats_ucm,
                               error_count_before, error_count_after):
							   
        """Logs clear, well-formatted summaries to the console."""
        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing, Custom Match): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing, Custom Match):  {error_count_after} / {total_images}")
        self.logger.info("-" * 70)
        
        self.logger.info("--- Metrics Summary ---")
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            sbc = stats_before_custom.get(class_name, {})
            sac = stats_after_custom.get(class_name, {})
            sau_pc = per_class_stats_ucm.get(class_name, {})
            
            # Recalculate P,R,F1 for console summary consistency if needed, or use pre-calculated ones
            prf1_sbc = calculate_prf1(sbc.get('TP',0), sbc.get('FP',0), sbc.get('FN',0))
            prf1_sac = calculate_prf1(sac.get('TP',0), sac.get('FP',0), sac.get('FN',0))
            # sau already contains P,R,F1 calculated by metrics_calculator
            
            self.logger.info(f"Class: {class_name}")
            self.logger.info(f"  - Before (Custom Match): TP: {sbc.get('TP',0)}, FP: {sbc.get('FP',0)}, FN: {sbc.get('FN',0)}, "
                             f"P: {prf1_sbc['precision']:.4f}, R: {prf1_sbc['recall']:.4f}, F1: {prf1_sbc['f1_score']:.4f}")
            self.logger.info(f"  - After (Custom Match):  TP: {sac.get('TP',0)}, FP: {sac.get('FP',0)}, FN: {sac.get('FN',0)}, "
                             f"P: {prf1_sac['precision']:.4f}, R: {prf1_sac['recall']:.4f}, F1: {prf1_sac['f1_score']:.4f}")
            self.logger.info(f"  - After (Ultralytics CM):TP: {sau_pc.get('TP',0)}, FP: {sau_pc.get('FP',0)}, FN: {sau_pc.get('FN',0)} (GTs: {sau_pc.get('GT_count',0)}), " 
                             f"P: {sau_pc.get('precision',0):.4f}, R: {sau_pc.get('recall',0):.4f}, F1: {sau_pc.get('f1_score',0):.4f}")
        self.logger.info("-" * 70) #
        self.logger.info("Overall (Ultralytics CM - Micro):") 
        self.logger.info(f"  TP: {overall_stats_ucm.get('TP',0)}, FP: {overall_stats_ucm.get('FP',0)}, FN: {overall_stats_ucm.get('FN',0)}") 
        self.logger.info(f"  P: {overall_stats_ucm.get('precision_micro',0):.4f}, R: {overall_stats_ucm.get('recall_micro',0):.4f}, F1: {overall_stats_ucm.get('f1_score_micro',0):.4f}") 
        self.logger.info("-" * 70) #

    def _generate_consolidated_excel_report(self, eval_output_dir, class_names_map, 
                                            stats_before_custom, stats_after_custom, 
                                            per_class_stats_ultralytics_cm, 
                                            overall_stats_ultralytics_cm,
                                            detection_correctness_data, error_count_before, error_count_after):
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
                    row_data.update(self._prepare_metrics_row(stats_before_custom.get(class_name, {}), 'Before Post-Process (Custom Match)'))
                    row_data.update(self._prepare_metrics_row(stats_after_custom.get(class_name, {}), 'After Post-Process (Custom Match)'))
                    row_data.update(self._prepare_metrics_row(per_class_stats_ultralytics_cm.get(class_name, {}), 'After Post-Process (Ultralytics CM)'))
                    summary_data.append(row_data)

                # Process Overall row using the helper
                overall_sbc = {metric: sum(s.get(metric, 0) for s in stats_before_custom.values()) for metric in ['TP', 'FP', 'FN']}
                overall_sac = {metric: sum(s.get(metric, 0) for s in stats_after_custom.values()) for metric in ['TP', 'FP', 'FN']}
                overall_sau_for_excel = { #
                    'TP': overall_stats_ultralytics_cm.get('TP', 0), 
                    'FP': overall_stats_ultralytics_cm.get('FP', 0), 
                    'FN': overall_stats_ultralytics_cm.get('FN', 0), 
                    'precision': overall_stats_ultralytics_cm.get('precision_micro', 0), 
                    'recall': overall_stats_ultralytics_cm.get('recall_micro', 0), 
                    'f1_score': overall_stats_ultralytics_cm.get('f1_score_micro', 0) 
                }

                overall_row_data = {('Class', ''): 'Overall (Micro)'}
                overall_row_data.update(self._prepare_metrics_row(overall_sbc, 'Before Post-Process (Custom Match)', error_count=error_count_before))
                overall_row_data.update(self._prepare_metrics_row(overall_sac, 'After Post-Process (Custom Match)', error_count=error_count_after))
                overall_row_data.update(self._prepare_metrics_row(overall_sau_for_excel, 'After Post-Process (Ultralytics CM)')) #
                summary_data.append(overall_row_data)
                
                df_metrics = pd.DataFrame(summary_data)
                df_metrics.columns = pd.MultiIndex.from_tuples(df_metrics.columns)
                df_metrics = df_metrics.set_index(('Class', ''))
                df_metrics.index.name = 'Class'

                metric_order = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Images with Errors'] 
                
                prefixes = [] 
                if ('Before Post-Process (Custom Match)', 'TP') in df_metrics.columns:  
                    prefixes.append('Before Post-Process (Custom Match)') 
                if ('After Post-Process (Custom Match)', 'TP') in df_metrics.columns:  
                    prefixes.append('After Post-Process (Custom Match)') 
                if ('After Post-Process (Ultralytics CM)', 'TP') in df_metrics.columns: 
                    prefixes.append('After Post-Process (Ultralytics CM)')


                final_column_order = []
                for prefix_name in prefixes:
                    for metric_name_suffix in metric_order: 
                        if (prefix_name, metric_name_suffix) in df_metrics.columns:
                            final_column_order.append((prefix_name, metric_name_suffix))
                
                df_metrics = df_metrics[final_column_order]
                df_metrics.to_excel(writer, sheet_name='Class Metrics Summary')


                if detection_correctness_data: 
                    df_detections = pd.DataFrame(detection_correctness_data) 
                    column_order = ['image_name', 'class_name', 'probability', 
                                    'pred_aspect_ratio', 'gt_aspect_ratio',  
                                    'box_correctness'] 
                    df_detections = df_detections[column_order].sort_values( 
                        by=['image_name', 'box_correctness', 'probability'],  
                        ascending=[True, True, False] 
                    )
                    df_detections.to_excel(writer, sheet_name='Detailed Detections', index=False) 


        except Exception as e:
            self.logger.error(f"Failed to generate consolidated Excel report: {e}. Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    def _prepare_metrics_row(self, stats, prefix, error_count=None): 
        tp = stats.get('TP', 0) 
        fp = stats.get('FP', 0) 
        fn = stats.get('FN', 0) 

        if 'precision' in stats and 'recall' in stats and 'f1_score' in stats: 
            precision = stats['precision'] 
            recall = stats['recall'] 
            f1_score = stats['f1_score'] 
        else:  #
            prf1_calc = calculate_prf1(tp, fp, fn) 
            precision = prf1_calc['precision'] 
            recall = prf1_calc['recall'] 
            f1_score = prf1_calc['f1_score'] #

        row_data = { #
            (prefix, 'TP'): tp, #
            (prefix, 'FP'): fp, #
            (prefix, 'FN'): fn, #
            (prefix, 'Precision'): precision, #
            (prefix, 'Recall'): recall, #
            (prefix, 'F1-Score'): f1_score, #
        }

        if error_count is not None:
            row_data[(prefix, 'Images with Errors')] = error_count
            
        return row_data