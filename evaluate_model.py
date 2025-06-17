"""
evaluate_model.py

This module defines the YoloEvaluator class, responsible for conducting a
detailed performance evaluation of a trained model. It compares predictions against
ground truths, calculates metrics both before and after custom post-processing,
and generates comprehensive reports including an Excel summary and images of
incorrect predictions.
"""
import logging
from pathlib import Path
import cv2
import json
from collections import Counter
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Any, Optional
from types import ModuleType

import config
from bbox_utils import match_predictions, calculate_iou, calculate_aspect_ratio
from utils import (
    parse_yolo_annotations,
    draw_error_annotations,
    calculate_prf1
)
from metrics_calculator import DetectionMetricsCalculator
from detector import CoinDetector

ImageLabelPair = Tuple[Path, Path]
StatsDict = Dict[str, Dict[str, int]]
Prediction = Dict[str, Any]
GroundTruth = Dict[str, Any]

class YoloEvaluator:
    """
    Orchestrates the detailed evaluation of a YOLO model's performance.
    """
    def __init__(self, detector: 'CoinDetector', logger: logging.Logger, config_module: ModuleType = config) -> None:
        """Initializes the YoloEvaluator."""
        self.logger = logger
        self.config = config_module
        self.detector = detector

    def perform_detailed_evaluation(self, eval_output_dir: Path, all_image_label_pairs_eval: List[ImageLabelPair]) -> Dict[str, Union[int, float]]:
        """
        Performs a full evaluation loop, generating metrics and reports.
        This is the main orchestration method for the evaluation process.
        """
        model_path = self.detector.model_path
        class_names = self.detector.class_names_map
        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path} on {len(all_image_label_pairs_eval)} images ---")
        
        # Setup output directory for incorrect prediction images
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics containers
        metrics_calc = DetectionMetricsCalculator(class_names, self.logger, self.config)
        # Metrics Set 1: Before Post-Process (Custom Match)
        agg_stats_before: StatsDict = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names.values()}
        
        # Metrics Set 2: After Post-Process (Custom Match)
        agg_stats_after: StatsDict = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names.values()}

        all_correctness_data: List[Dict[str, Any]] = []
        error_count_before, error_count_after = 0, 0
        
        # Main loop over all evaluation images
        for img_path, lbl_path in all_image_label_pairs_eval:
            # Process each image and get its stats
            results = self._process_single_image(img_path, lbl_path, class_names)
            if not results: continue

            # Aggregate statistics from the single image results
            for cn in class_names.values():
                for metric in ['TP', 'FP', 'FN']:
                    agg_stats_before[cn][metric] += results['stats_before'][cn][metric]
                    agg_stats_after[cn][metric] += results['stats_after'][cn][metric]
            
            # Update data for Excel report
            self._update_correctness_list(all_correctness_data, img_path.name, results['tp_after'], results['fp_after'], results['fn_after'])
            
            # Update Ultralytics Confusion Matrix
            metrics_calc.update_confusion_matrix(results['preds_after'], results['gts'])
            
            # Log and save error images
            error_count_before += 1 if (any(results['fp_before']) or any(results['fn_before'])) else 0
            if any(results['fp_after']) or any(results['fn_after']):
                error_count_after += 1
                self._log_and_save_errors(results['image'], img_path, results['fp_after'], results['fn_after'], results['gts'], results['preds_after'], incorrect_dir)

        # --- Final Reporting ---
        final_metrics_ucm = metrics_calc.compute_metrics(eval_output_dir=eval_output_dir)
        self._log_console_summaries(len(all_image_label_pairs_eval), class_names, agg_stats_before, agg_stats_after, final_metrics_ucm, error_count_before, error_count_after)
        self._generate_consolidated_excel_report(eval_output_dir, class_names, agg_stats_before, agg_stats_after, final_metrics_ucm, all_correctness_data, error_count_before, error_count_after)
        
        metrics_file = eval_output_dir / self.config.METRICS_JSON_NAME
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics_ucm, f, indent=4)
        self.logger.info(f"Saved final Ultralytics CM metrics dictionary to: {metrics_file}")
        
        return self._prepare_summary_for_return(agg_stats_after, error_count_after)

    def _process_single_image(self, img_path: Path, lbl_path: Path, class_names_map: Dict[int, str]) -> Optional[Dict[str, Any]]:
        """Helper to process one image: read, predict, and match to ground truths."""
        original_img: Optional[np.ndarray] = cv2.imread(str(img_path))
        if original_img is None:
            self.logger.warning(f"Could not read image {img_path}, skipping.")
            return None
        
        prediction_result = self.detector.predict(original_img.copy(), return_raw=True, image_name=img_path.name)
        preds_after: List[Prediction]
        preds_before: List[Prediction]
        preds_after, preds_before = prediction_result # type: ignore
        
        h, w = original_img.shape[:2]
        gts_raw = parse_yolo_annotations(lbl_path, self.logger)
        ground_truths: List[GroundTruth] = [{'cls': cid, 'xyxy': [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]} for cid, cx, cy, cw, ch in gts_raw]
        
        stats_b, _, fp_b, fn_b = match_predictions(preds_before, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
        stats_a, tp_a, fp_a, fn_a = match_predictions(preds_after, ground_truths, self.config.BOX_MATCHING_IOU_THRESHOLD, class_names_map)
        
        return {
            'image': original_img, 'preds_after': preds_after, 'preds_before': preds_before, 'gts': ground_truths,
            'stats_before': stats_b, 'stats_after': stats_a,
            'fp_before': fp_b, 'fn_before': fn_b,
            'tp_after': tp_a, 'fp_after': fp_a, 'fn_after': fn_a
        }

    def _update_correctness_list(self, all_data: List[Dict[str, Any]], image_name: str, tps: List[Prediction], fps: List[Prediction], fns: List[GroundTruth]) -> None:
        """Helper to populate the list used for the detailed Excel report."""
        for pred in tps: 
            all_data.append({'image_name': image_name, 'class_name': pred['class_name'], 'probability': pred['conf'], 'pred_aspect_ratio': calculate_aspect_ratio(pred['xyxy']),'gt_aspect_ratio': calculate_aspect_ratio(pred.get('matched_gt_xyxy', [])), 'box_correctness': 'TP'})
        for pred in fps: 
            all_data.append({'image_name': image_name, 'class_name': pred['class_name'], 'probability': pred['conf'], 'pred_aspect_ratio': calculate_aspect_ratio(pred['xyxy']), 'gt_aspect_ratio': None, 'box_correctness': 'FP'})
        for gt in fns: 
            all_data.append({'image_name': image_name, 'class_name': gt['class_name'], 'probability': None,'pred_aspect_ratio': None, 'gt_aspect_ratio': calculate_aspect_ratio(gt['xyxy']), 'box_correctness': 'FN'})

    def _log_and_save_errors(self, image: np.ndarray, img_path: Path, fps: List[Prediction], fns: List[GroundTruth], gts: List[GroundTruth], preds: List[Prediction], output_dir: Path) -> None:
        """Helper to log incorrect prediction details and save the error image."""
        class_names_map = self.detector.class_names_map
        gt_counts = Counter(g['cls'] for g in gts)
        pred_counts = Counter(p['cls'] for p in preds)
        gt_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(gt_counts.items())])
        pred_str = ", ".join([f"{count}x {class_names_map[name]}" for name, count in sorted(pred_counts.items())])
        
        self.logger.warning(f"Incorrect Prediction in: {img_path.name}\n  - Ground Truth: [{gt_str}]\n  - Predicted (Post-Filter): [{pred_str}]")
        annotated_img = draw_error_annotations(image.copy(), fps, fns, class_names_map, self.config)
        cv2.imwrite(str(output_dir / img_path.name), annotated_img)

    def _log_console_summaries(self, total_images: int, 
				class_names_map: Dict[int, str], 
				stats_before: StatsDict, 
				stats_after: StatsDict, 
				final_metrics_ucm: Dict[str, Any], 
				error_count_before: int, 
				error_count_after: int
				) -> None:
        """Helper to log clear, well-formatted summaries to the console."""
        per_class_ucm = final_metrics_ucm.get('per_class', {})
        overall_ucm = final_metrics_ucm.get('overall', {})

        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing):  {error_count_after} / {total_images}")
        self.logger.info("-" * 70)
        
        self.logger.info("--- Metrics Summary ---")
        for cid in sorted(class_names_map.keys()):
            name = class_names_map[cid]
            sbc, sac, sau_pc = stats_before.get(name, {}), stats_after.get(name, {}), per_class_ucm.get(name, {})
            prf1_sbc, prf1_sac = calculate_prf1(sbc.get('TP',0), sbc.get('FP',0), sbc.get('FN',0)), calculate_prf1(sac.get('TP',0), sac.get('FP',0), sac.get('FN',0))
            
            self.logger.info(f"Class: {name}")
            self.logger.info(f"  - Before (Custom Match): TP: {sbc.get('TP',0)}, FP: {sbc.get('FP',0)}, FN: {sbc.get('FN',0)}, P: {prf1_sbc['precision']:.4f}, R: {prf1_sbc['recall']:.4f}, F1: {prf1_sbc['f1_score']:.4f}")
            self.logger.info(f"  - After (Custom Match):  TP: {sac.get('TP',0)}, FP: {sac.get('FP',0)}, FN: {sac.get('FN',0)}, P: {prf1_sac['precision']:.4f}, R: {prf1_sac['recall']:.4f}, F1: {prf1_sac['f1_score']:.4f}")
            self.logger.info(f"  - After (Ultralytics CM):TP: {sau_pc.get('TP',0)}, FP: {sau_pc.get('FP',0)}, FN: {sau_pc.get('FN',0)}, P: {sau_pc.get('precision',0):.4f}, R: {sau_pc.get('recall',0):.4f}, F1: {sau_pc.get('f1_score',0):.4f}")
        
        self.logger.info("-" * 70)
        self.logger.info(f"Overall (Ultralytics CM - Micro): P: {overall_ucm.get('precision_micro',0):.4f}, R: {overall_ucm.get('recall_micro',0):.4f}, F1: {overall_ucm.get('f1_score_micro',0):.4f}")
        self.logger.info("-" * 70)

    def _generate_consolidated_excel_report(self, eval_output_dir: Path, class_names_map: Dict[int, str], stats_before: StatsDict, stats_after: StatsDict, final_metrics_ucm: Dict[str, Any], correctness_data: List[Dict[str, Any]], err_before: int, err_after: int) -> None:
        """Helper to generate the final multi-sheet Excel report."""
        excel_path = eval_output_dir / "evaluation_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Class Metrics Summary
                self._build_summary_metrics_sheet(writer, class_names_map, stats_before, stats_after, final_metrics_ucm, err_before, err_after)
                # Sheet 2: Detailed list of all detections
                self._build_detailed_detections_sheet(writer, correctness_data)
        except Exception as e:
            self.logger.error(f"Failed to generate consolidated Excel report: {e}. Ensure 'openpyxl' is installed.")

    def _build_summary_metrics_sheet(self, writer: pd.ExcelWriter, class_names_map: Dict[int, str], stats_before: StatsDict, stats_after: StatsDict, final_metrics_ucm: Dict[str, Any], err_before: int, err_after: int) -> None:
        """Builds and writes the summary metrics DataFrame to an Excel sheet."""
        per_class_ucm = final_metrics_ucm.get('per_class', {})
        overall_ucm = final_metrics_ucm.get('overall', {})
        summary_data: List[Dict[Tuple[str, str], Any]] = []

        for cid in sorted(class_names_map.keys()):
            name = class_names_map[cid]
            row: Dict[Tuple[str, str], Any] = {('Class', ''): name}
            row.update(self._prepare_metrics_row(stats_before.get(name, {}), 'Before (Custom)'))
            row.update(self._prepare_metrics_row(stats_after.get(name, {}), 'After (Custom)'))
            row.update(self._prepare_metrics_row(per_class_ucm.get(name, {}), 'After (Ultralytics)'))
            summary_data.append(row)

        # Process Overall row
        overall_sbc = {m: sum(s.get(m, 0) for s in stats_before.values()) for m in ['TP', 'FP', 'FN']}
        overall_sac = {m: sum(s.get(m, 0) for s in stats_after.values()) for m in ['TP', 'FP', 'FN']}
        overall_sau_excel = {'TP': overall_ucm.get('TP', 0), 'FP': overall_ucm.get('FP', 0), 'FN': overall_ucm.get('FN', 0), 'precision': overall_ucm.get('precision_micro', 0), 'recall': overall_ucm.get('recall_micro', 0), 'f1_score': overall_ucm.get('f1_score_micro', 0)}
        
        overall_row: Dict[Tuple[str, str], Any] = {('Class', ''): 'Overall (Micro)'}
        overall_row.update(self._prepare_metrics_row(overall_sbc, 'Before (Custom)', err_before))
        overall_row.update(self._prepare_metrics_row(overall_sac, 'After (Custom)', err_after))
        overall_row.update(self._prepare_metrics_row(overall_sau_excel, 'After (Ultralytics)'))
        summary_data.append(overall_row)
        
        df_metrics = pd.DataFrame(summary_data)
        df_metrics.columns = pd.MultiIndex.from_tuples(df_metrics.columns)
        df_metrics = df_metrics.set_index(('Class', ''))
        df_metrics.index.name = 'Class'

        # Reorder columns for readability
        metric_order = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Images with Errors']
        prefixes = [p for p in ['Before (Custom)', 'After (Custom)', 'After (Ultralytics)'] if (p, 'TP') in df_metrics.columns]
        final_column_order = [(prefix, metric) for prefix in prefixes for metric in metric_order if (prefix, metric) in df_metrics.columns]
        
        df_metrics[final_column_order].to_excel(writer, sheet_name='Class Metrics Summary')

    def _build_detailed_detections_sheet(self, writer: pd.ExcelWriter, detection_data: List[Dict[str, Any]]) -> None:
        """Builds and writes the detailed detections DataFrame to an Excel sheet."""
        if not detection_data: return
        df_detections = pd.DataFrame(detection_data)
        column_order = ['image_name', 'class_name', 'probability', 'pred_aspect_ratio', 'gt_aspect_ratio', 'box_correctness']
        df_detections = df_detections[column_order].sort_values(by=['image_name', 'box_correctness', 'probability'], ascending=[True, True, False])
        df_detections.to_excel(writer, sheet_name='Detailed Detections', index=False)

    def _prepare_metrics_row(self, stats: Dict[str, Any], prefix: str, error_count: Optional[int] = None) -> Dict[Tuple[str, str], Union[int, float]]: 
        """Helper to format a single row of metrics for the summary DataFrame."""
        tp, fp, fn = stats.get('TP', 0), stats.get('FP', 0), stats.get('FN', 0)
        
        if 'precision' in stats: # Use pre-calculated metrics if available
            p, r, f1 = stats['precision'], stats['recall'], stats['f1_score']
        else: # Calculate on-the-fly otherwise
            prf1 = calculate_prf1(float(tp), float(fp), float(fn))
            p, r, f1 = prf1['precision'], prf1['recall'], prf1['f1_score']

        row_data: Dict[Tuple[str, str], Union[int, float]] = {
            (prefix, 'TP'): tp, 
            (prefix, 'FP'): fp, 
            (prefix, 'FN'): fn, 
            (prefix, 'Precision'): p, 
            (prefix, 'Recall'): r, 
            (prefix, 'F1-Score'): f1
        }
        if error_count is not None:
            row_data[(prefix, 'Images with Errors')] = error_count
            
        return row_data

    def _prepare_summary_for_return(self, stats_after: StatsDict, error_count_after: int) -> Dict[str, Union[int, float]]:
        """Prepares the final summary dictionary to be returned for multi-model comparison."""
        overall_stats = {m: sum(s.get(m, 0) for s in stats_after.values()) for m in ['TP', 'FP', 'FN']}
        prf1_overall = calculate_prf1(float(overall_stats['TP']), float(overall_stats['FP']), float(overall_stats['FN']))
        
        summary_to_return: Dict[str, Union[int, float]] = {
            **overall_stats,
            'Precision': prf1_overall['precision'], 
            'Recall': prf1_overall['recall'], 
            'F1-Score': prf1_overall['f1_score'], 
            'Images with Errors': error_count_after
        }
        return summary_to_return