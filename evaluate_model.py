import logging
from pathlib import Path
import cv2
import csv
import json # For saving metrics
from collections import defaultdict
import copy

import config
from bbox_utils import calculate_iou
from utils import (
    parse_yolo_annotations,
    draw_error_annotations
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
        """Calculates TP, FP, FN and returns the lists of FP predictions and FN ground truths."""
        stats = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in class_names_map.values()}
        fp_predictions = []
        fn_ground_truths = []
        
        # Ensure GTs have a 'matched' flag for this run
        gts = copy.deepcopy(ground_truths)
        for gt in gts:
            gt['matched'] = False
            
        for pred in sorted(predictions, key=lambda x: x['conf'], reverse=True):
            is_tp = False
            pred_class_id = pred['cls']
            pred_class_name = class_names_map.get(pred_class_id, f"ID_{pred_class_id}")

            for gt in gts:
                if not gt['matched'] and pred_class_id == gt['cls']:
                    iou = calculate_iou(pred['xyxy'], gt['xyxy'])
                    if iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
                        gt['matched'] = True
                        is_tp = True
                        break
            
            if is_tp:
                stats[pred_class_name]['tp'] += 1
            else:
                stats[pred_class_name]['fp'] += 1
                fp_predictions.append(pred)

        for gt in gts:
            if not gt['matched']:
                gt_class_name = class_names_map.get(gt['cls'], f"ID_{gt['cls']}")
                stats[gt_class_name]['fn'] += 1
                fn_ground_truths.append(gt)
                
        return stats, fp_predictions, fn_ground_truths

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        # Get model path and class map from the detector instance
        model_path_to_eval = self.detector.model_path
        class_names_map = self.detector.class_names_map
        total_images = len(all_image_label_pairs_eval)

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} on {total_images} images ---")
        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        stats_before = defaultdict(lambda: defaultdict(int))
        stats_after = defaultdict(lambda: defaultdict(int))
        # Counters for images with any error
        error_count_before = 0
        error_count_after = 0
        
        metrics_calc = DetectionMetricsCalculator(class_names_map, self.logger, self.config)

        for img_path, lbl_path in all_image_label_pairs_eval:
            self.logger.info(f"Processing: {img_path.name}")
            original_img = cv2.imread(str(img_path))
            
            # Get both raw and post-processed predictions
            preds_after, preds_before = self.detector.predict(original_img.copy(), return_raw=True)
            
            h, w = original_img.shape[:2]
            gts_raw = parse_yolo_annotations(lbl_path, self.logger)
            current_gts = [{'cls': cid, 'xyxy': [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]} for cid, cx, cy, cw, ch in gts_raw]
            
            # --- Calculate stats for both sets of predictions ---
            img_stats_before, fp_before, fn_before = self._calculate_stats(preds_before, current_gts, class_names_map)
            # We care about the details (FPs, FNs) for the FINAL predictions
            img_stats_after, fp_after, fn_after = self._calculate_stats(preds_after, current_gts, class_names_map)

            # Check for errors and increment image error counters
            if fp_before or fn_before:
                error_count_before += 1
            if fp_after or fn_after:
                error_count_after += 1
                # Save the annotated image only if there are errors AFTER post-processing
                annotated_img = draw_error_annotations(
                    original_img.copy(), fp_after, fn_after,
                    class_names_map, self.config 
                )
                out_path = incorrect_dir / img_path.name
                cv2.imwrite(str(out_path), annotated_img)

            # Accumulate TP/FP/FN stats
            for class_name in class_names_map.values():
                for metric in ['tp', 'fp', 'fn']:
                    stats_before[class_name][metric] += img_stats_before[class_name][metric]
                    stats_after[class_name][metric] += img_stats_after[class_name][metric]

            metrics_calc.update_confusion_matrix(preds_after, current_gts)

        # --- Log the summary outputs ---
        self.logger.info("--- Image Error Summary ---")
        self.logger.info(f"Total images with errors (Before Post-Processing): {error_count_before} / {total_images}")
        self.logger.info(f"Total images with errors (After Post-Processing):  {error_count_after} / {total_images}")
        self.logger.info("-" * 50)
        self.logger.info("--- Post-Processing Impact Summary (TP, FP, FN) ---")
        for class_id in sorted(class_names_map.keys()):
            class_name = class_names_map[class_id]
            b = stats_before[class_name]
            a = stats_after[class_name]
            
            delta_tp = a['tp'] - b['tp']
            delta_fp = a['fp'] - b['fp']
            delta_fn = a['fn'] - b['fn']

            self.logger.info(f"Class: {class_name}")
            self.logger.info(f"  - Before: TP={b['tp']:<4} FP={b['fp']:<4} FN={b['fn']:<4}")
            self.logger.info(f"  - After:  TP={a['tp']:<4} FP={a['fp']:<4} FN={a['fn']:<4}")
            self.logger.info(f"  - Change: TP={delta_tp:<+4} FP={delta_fp:<+4} FN={delta_fn:<+4}")
        self.logger.info("-" * 50)

        # --- Compute and log the final, post-processed metrics ---
        self.logger.info("--- Final Metrics (based on Post-Processed Detections) ---")
        final_metrics_results = metrics_calc.compute_metrics(
            requested_metrics=self.config.REQUESTED_METRICS,
            eval_output_dir=eval_output_dir 
        )

        # Logging the final metrics remains the same...
        for class_name, metrics in final_metrics_results.get('per_class', {}).items():
             self.logger.info(f"Class: {class_name} | TP: {metrics.get('TP',0)}, FP: {metrics.get('FP',0)}, FN: {metrics.get('FN',0)} | P: {metrics.get('precision', 0):.4f}, R: {metrics.get('recall', 0):.4f}, F1: {metrics.get('f1_score', 0):.4f}")

        overall = final_metrics_results.get('overall', {})
        if overall:
             self.logger.info(f"--- Overall (Micro) | P: {overall.get('precision_micro',0):.4f}, R: {overall.get('recall_micro',0):.4f}, F1: {overall.get('f1_score_micro',0):.4f} ---")

        metrics_file_path = eval_output_dir / self.config.METRICS_JSON_NAME
        with open(metrics_file_path, 'w') as f:
            json.dump(final_metrics_results, f, indent=4)
        self.logger.info(f"Saved final metrics dictionary to: {metrics_file_path}")