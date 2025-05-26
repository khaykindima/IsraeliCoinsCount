import logging
import torch # For tensor conversions
import numpy as np
from ultralytics.utils import metrics as ultralytics_metrics # For ConfusionMatrix
from pathlib import Path # Added for eval_output_dir type hint if needed

class DetectionMetricsCalculator:
    """
    Calculates and stores detection metrics like Precision, Recall, F1-score,
    and handles confusion matrix generation.
    """
    def __init__(self, class_names_map, logger=None, config_module=None):
        """
        Initializes the DetectionMetricsCalculator.
        Args:
            class_names_map (dict): Maps class IDs (int) to class names (str).
            logger (logging.Logger, optional): Logger instance.
            config_module (module, optional): The project's config module for settings
                                             like CONFUSION_MATRIX_PLOT_NAME.
        """
        self.class_names_map = class_names_map
        self.num_classes = len(class_names_map)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config_module # Store config for plot name etc.
        
        self.stats_per_class = {
            class_id: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0}
            for class_id in self.class_names_map.keys()
        }
        
        # Initialize Ultralytics Confusion Matrix (it will only be used if requested)
        self.ultralytics_cm = ultralytics_metrics.ConfusionMatrix(nc=self.num_classes)
        self.logger.info(f"MetricsCalculator initialized for {self.num_classes} classes. Ultralytics CM ready.")

    def update_stats(self, class_id, tp_delta=0, fp_delta=0, fn_delta=0, gt_delta=0):
        """Updates the TP, FP, FN, and GT counts for Precision/Recall/F1."""
        if class_id not in self.stats_per_class:
            self.logger.warning(f"Class ID {class_id} not in initial map for P/R/F1 stats. Adding.")
            self.stats_per_class[class_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0}
            if class_id not in self.class_names_map: # Should ideally not happen
                 self.class_names_map[class_id] = f"Unknown_ID_{class_id}"
                 # If a new class ID appears, CM might need reinitialization or careful handling if nc was fixed.
                 # For now, Ultralytics CM is initialized with nc from the initial map.
                 # Unseen class IDs will cause issues with CM if its nc is fixed.
                 # Best to ensure class_names_map is exhaustive.
        
        self.stats_per_class[class_id]['tp'] += tp_delta
        self.stats_per_class[class_id]['fp'] += fp_delta
        self.stats_per_class[class_id]['fn'] += fn_delta
        self.stats_per_class[class_id]['gt_count'] += gt_delta

    def update_confusion_matrix(self, predictions_for_image, ground_truths_for_image):
        """
        Updates the internal confusion matrix with data from a single image.
        Args:
            predictions_for_image (list of dict): Detections from CoinDetector.
                                                 Each dict: {'xyxy', 'conf', 'cls'}
            ground_truths_for_image (list of dict): Ground truths for the image.
                                                  Each dict: {'cls', 'xyxy'}
        """
        if not self.ultralytics_cm: # Should always be initialized now
            self.logger.warning("Ultralytics CM not initialized, skipping CM update.")
            return

        # Prepare predictions tensor: (N, 6) [x1, y1, x2, y2, conf, cls_id]
        if predictions_for_image:
            preds_list = [[*p['xyxy'], p['conf'], float(p['cls'])] for p in predictions_for_image]
            preds_tensor = torch.tensor(preds_list, dtype=torch.float32)
        else:
            preds_tensor = torch.empty((0, 6), dtype=torch.float32)

        gt_bboxes_list = []
        gt_cls_list = []
        if ground_truths_for_image:
            for gt in ground_truths_for_image:
                gt_bboxes_list.append(gt['xyxy'])
                gt_cls_list.append(float(gt['cls']))
            
            gt_bboxes_tensor = torch.tensor(gt_bboxes_list, dtype=torch.float32)
            gt_cls_tensor = torch.tensor(gt_cls_list, dtype=torch.float32)
        else:
            gt_bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            gt_cls_tensor = torch.empty((0,), dtype=torch.float32)
        
        # Call process_batch with the signature (detections, gt_bboxes, gt_cls)
        self.ultralytics_cm.process_batch(preds_tensor, gt_bboxes_tensor, gt_cls_tensor)


    def _calculate_single_metric_set(self, tp, fp, fn, gt_count):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def compute_metrics(self, requested_metrics=None, eval_output_dir: Path = None):
        """
        Computes metrics and handles confusion matrix plotting.
        Args:
            requested_metrics (list of str, optional): Metrics to compute.
                Supported: 'precision', 'recall', 'f1_score', 'confusion_matrix'.
                If None or empty, defaults include P, R, F1, and CM (data & plot).
            eval_output_dir (Path, optional): Directory to save plots like confusion matrix.
        """
        # --- MODIFIED: Determine if CM calculation/plotting is needed ---
        should_process_confusion_matrix = False
        base_metrics_to_calculate = ['precision', 'recall', 'f1_score'] # Core metrics

        if requested_metrics is None or not requested_metrics:
            # Default behavior: calculate P, R, F1 and full CM processing
            metrics_to_calculate = base_metrics_to_calculate
            should_process_confusion_matrix = True 
        else:
            processed_requested_metrics = [m.lower() for m in requested_metrics]
            metrics_to_calculate = [m for m in processed_requested_metrics if m in base_metrics_to_calculate]
            if 'confusion_matrix' in processed_requested_metrics:
                should_process_confusion_matrix = True
            # If CM is requested, ensure we also have the core metrics for context in the output
            if should_process_confusion_matrix and not all(m in metrics_to_calculate for m in base_metrics_to_calculate):
                 metrics_to_calculate.extend(base_metrics_to_calculate)
                 metrics_to_calculate = sorted(list(set(metrics_to_calculate)))


        self.logger.info(f"Computing base metrics: {metrics_to_calculate}")
        if should_process_confusion_matrix: self.logger.info("Confusion matrix processing also requested (data and plot).")
        
        results = {'per_class': {}, 'overall': {}, 'confusion_matrix_data': None, 'confusion_matrix_plot_path_info': None}
        total_tp_all_classes, total_fp_all_classes, total_fn_all_classes, total_gt_all_classes = 0,0,0,0

        for class_id, stats in self.stats_per_class.items():
            class_name = self.class_names_map.get(class_id, f"Unknown_ID_{class_id}")
            tp, fp, fn, gt_count = stats['tp'], stats['fp'], stats['fn'], stats['gt_count']
            class_metrics_set = self._calculate_single_metric_set(tp, fp, fn, gt_count)
            
            results['per_class'][class_name] = {'TP': tp, 'FP': fp, 'FN': fn, 'GT_count': gt_count}
            for metric_name in metrics_to_calculate:
                if metric_name in class_metrics_set:
                    results['per_class'][class_name][metric_name] = class_metrics_set[metric_name]
            
            total_tp_all_classes += tp; total_fp_all_classes += fp; total_fn_all_classes += fn; total_gt_all_classes += gt_count

        overall_calculated_metrics = self._calculate_single_metric_set(
            total_tp_all_classes, total_fp_all_classes, total_fn_all_classes, total_gt_all_classes
        )
        results['overall'] = {
            'TP': total_tp_all_classes, 'FP': total_fp_all_classes, 
            'FN': total_fn_all_classes, 'GT_count': total_gt_all_classes
        }
        for metric_name in metrics_to_calculate:
            overall_metric_key = f"{metric_name}_micro" 
            if metric_name in overall_calculated_metrics:
                 results['overall'][overall_metric_key] = overall_calculated_metrics[metric_name]

        # --- MODIFIED: Confusion Matrix Processing based on refined logic ---
        if self.ultralytics_cm and should_process_confusion_matrix:
            self.logger.info("Processing confusion matrix data and plot...")

            # --- ADD THIS LOGGING ---
            if self.ultralytics_cm.matrix is not None:
                self.logger.info(f"Raw Confusion Matrix from Ultralytics CM before plotting:\n{self.ultralytics_cm.matrix}")
            else:
                self.logger.info("Ultralytics CM matrix is None before plotting.")
            # --- END OF ADDED LOGGING ---

            results['confusion_matrix_data'] = self.ultralytics_cm.matrix.tolist() # Always provide data if CM is processed
            
            if eval_output_dir and self.config and hasattr(self.config, 'CONFUSION_MATRIX_PLOT_NAME'):
                # The plot name from config is used by Ultralytics as a base, it might append e.g. "_normalized"
                # We'll log based on the config name for user reference.
                plot_base_name_from_config = Path(self.config.CONFUSION_MATRIX_PLOT_NAME).stem
                try:
                    sorted_class_names = [self.class_names_map[i] for i in sorted(self.class_names_map.keys()) if i < self.num_classes]
                    self.ultralytics_cm.plot(
                        save_dir=str(eval_output_dir), 
                        names=sorted_class_names,
                        normalize=False
                    )
                    # Ultralytics will save it as something like 'confusion_matrix.png' or 'confusion_matrix_normalized.png'
                    # if normalize=True. Since normalize=False, it should be closer to the base name.
                    # We inform the user it's saved in the directory.
                    self.logger.info(f"Confusion matrix plot saved in directory: {eval_output_dir} (default name e.g., confusion_matrix.png)")
                    results['confusion_matrix_plot_path_info'] = f"Plot saved in {eval_output_dir} (e.g., {plot_base_name_from_config}.png)"
                except Exception as e:
                    self.logger.error(f"Failed to plot/save confusion matrix: {e}")
                    results['confusion_matrix_plot_path_info'] = f"Failed to save plot: {e}"
            else:
                self.logger.warning("Cannot save confusion matrix plot: eval_output_dir or CONFUSION_MATRIX_PLOT_NAME not properly specified in config.")
                results['confusion_matrix_plot_path_info'] = "Plot not saved due to missing output path or filename config."
        
        self.logger.info("Metrics computation complete.")
        return results

# Example Usage (if you want to test this module directly)
if __name__ == '__main__':
    import json
    from pathlib import Path
    class MockConfig: 
        CONFUSION_MATRIX_PLOT_NAME = "test_cm_simplified.png" # Changed plot name for new test

    mock_config_instance = MockConfig()
    test_output_dir = Path("./test_metrics_output_cm_simplified_logic") 
    test_output_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger("MetricsCalculatorTest")

    sample_class_map = {0: 'cat', 1: 'dog', 2: 'bird'}
    calculator = DetectionMetricsCalculator(sample_class_map, logger=logger, config_module=mock_config_instance)

    # Image 1 Data
    preds_img1 = [{'xyxy': [10,10,50,50], 'conf': 0.9, 'cls': 0}, {'xyxy': [60,60,100,100], 'conf': 0.8, 'cls': 1}]
    gts_img1 = [{'xyxy': [12,12,48,48], 'cls': 0}, {'xyxy': [150,150,180,180], 'cls': 1}] 
    calculator.update_stats(class_id=0, tp_delta=1, gt_delta=1); calculator.update_stats(class_id=1, fp_delta=1); calculator.update_stats(class_id=1, fn_delta=1, gt_delta=1)
    calculator.update_confusion_matrix(preds_img1, gts_img1)

    # Image 2 Data
    preds_img2 = [{'xyxy': [20,20,60,60], 'conf': 0.7, 'cls': 0}, {'xyxy': [70,70,110,110], 'conf': 0.85, 'cls': 2}] 
    gts_img2 = [{'xyxy': [20,20,60,60], 'cls': 0}, {'xyxy': [75,75,105,105], 'cls': 1}] 
    calculator.update_stats(class_id=0, tp_delta=1, gt_delta=1); calculator.update_stats(class_id=2, fp_delta=1); calculator.update_stats(class_id=1, fn_delta=1, gt_delta=1)
    calculator.update_confusion_matrix(preds_img2, gts_img2)
    
    logger.info("\n--- Test 1: REQUESTED_METRICS = None (Default: P, R, F1 + CM) ---")
    default_metrics = calculator.compute_metrics(requested_metrics=None, eval_output_dir=test_output_dir)
    logger.info(json.dumps(default_metrics, indent=4))

    logger.info("\n--- Test 2: REQUESTED_METRICS = ['precision', 'recall', 'f1_score'] (No CM) ---")
    core_metrics_only = calculator.compute_metrics(requested_metrics=['precision', 'recall', 'f1_score'], eval_output_dir=test_output_dir)
    logger.info(json.dumps(core_metrics_only, indent=4))

    logger.info("\n--- Test 3: REQUESTED_METRICS = ['confusion_matrix'] (CM data & plot + core P,R,F1) ---")
    cm_only_metrics = calculator.compute_metrics(requested_metrics=['confusion_matrix'], eval_output_dir=test_output_dir)
    logger.info(json.dumps(cm_only_metrics, indent=4))
    
    logger.info("\n--- Test 4: REQUESTED_METRICS = ['precision', 'confusion_matrix'] ---")
    precision_and_cm = calculator.compute_metrics(requested_metrics=['precision', 'confusion_matrix'], eval_output_dir=test_output_dir)
    logger.info(json.dumps(precision_and_cm, indent=4))

    logger.info(f"\nCheck for plots in: {test_output_dir.resolve()}")