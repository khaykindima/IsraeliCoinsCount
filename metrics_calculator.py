import logging
import torch
from pathlib import Path
from ultralytics.utils import metrics as ultralytics_metrics
from utils import plot_readable_confusion_matrix # Import the new plotting utility

class DetectionMetricsCalculator:
    """
    Calculates detection metrics like Precision, Recall, F1-score by deriving
    them from a confusion matrix.
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
        
        # Initialize Ultralytics Confusion Matrix. It is now the single source of truth for metrics.
        self.ultralytics_cm = ultralytics_metrics.ConfusionMatrix(nc=self.num_classes)
        self.logger.info(f"MetricsCalculator initialized for {self.num_classes} classes. Ultralytics CM is ready.")

    def update_confusion_matrix(self, predictions_for_image, ground_truths_for_image):
        """
        Updates the internal confusion matrix with data from a single image.
        Args:
            predictions_for_image (list of dict): Detections from CoinDetector.
                                                 Each dict: {'xyxy', 'conf', 'cls'}
            ground_truths_for_image (list of dict): Ground truths for the image.
                                                  Each dict: {'cls', 'xyxy'}
        """
        if not self.ultralytics_cm:
            self.logger.warning("Ultralytics CM not initialized, skipping CM update.")
            return

        # Prepare predictions tensor: (N, 6) [x1, y1, x2, y2, conf, cls_id]
        if predictions_for_image:
            preds_list = [[*p['xyxy'], p['conf'], float(p['cls'])] for p in predictions_for_image]
            preds_tensor = torch.tensor(preds_list, dtype=torch.float32)
        else:
            preds_tensor = torch.empty((0, 6), dtype=torch.float32)

        if ground_truths_for_image:
            gt_bboxes_list = [gt['xyxy'] for gt in ground_truths_for_image]
            gt_cls_list = [float(gt['cls']) for gt in ground_truths_for_image]
            gt_bboxes_tensor = torch.tensor(gt_bboxes_list, dtype=torch.float32)
            gt_cls_tensor = torch.tensor(gt_cls_list, dtype=torch.float32)
        else:
            gt_bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            gt_cls_tensor = torch.empty((0,), dtype=torch.float32)
        
        self.ultralytics_cm.process_batch(preds_tensor, gt_bboxes_tensor, gt_cls_tensor)


    def _calculate_single_metric_set(self, tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def compute_metrics(self, requested_metrics=None, eval_output_dir: Path = None):
        """
        Computes metrics by deriving stats from the confusion matrix and handles plotting.
        Args:
            requested_metrics (list of str, optional): Metrics to compute.
            eval_output_dir (Path, optional): Directory to save plots.
        """
        results = {'per_class': {}, 'overall': {}, 'confusion_matrix_data': None, 'confusion_matrix_plot_path_info': None}
        
        matrix = self.ultralytics_cm.matrix
        if matrix is None:
            self.logger.error("Cannot compute metrics: Confusion Matrix has not been populated.")
            return results

        # Determine which metrics to compute
        base_metrics_to_calculate = ['precision', 'recall', 'f1_score']
        should_process_confusion_matrix = False
        if requested_metrics is None or not requested_metrics:
            metrics_to_calculate = base_metrics_to_calculate
            should_process_confusion_matrix = True
        else:
            processed_requested = [m.lower() for m in requested_metrics]
            metrics_to_calculate = [m for m in processed_requested if m in base_metrics_to_calculate]
            if 'confusion_matrix' in processed_requested:
                should_process_confusion_matrix = True

        self.logger.info(f"Computing base metrics: {metrics_to_calculate}")
        if should_process_confusion_matrix: self.logger.info("Confusion matrix processing also requested.")

        total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0

        # Derive TP, FP, FN for each class from the matrix
        for i in range(self.num_classes):
            class_name = self.class_names_map.get(i, f"Unknown_ID_{i}")
            
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            fn = matrix[i, :].sum() - tp
            gt = tp + fn

            class_metrics_set = self._calculate_single_metric_set(tp, fp, fn)
            results['per_class'][class_name] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'GT_count': int(gt)}
            for metric_name in metrics_to_calculate:
                if metric_name in class_metrics_set:
                    results['per_class'][class_name][metric_name] = class_metrics_set[metric_name]
            
            total_tp += tp; total_fp += fp; total_fn += fn; total_gt += gt

        overall_metrics = self._calculate_single_metric_set(total_tp, total_fp, total_fn)
        results['overall'] = {'TP': int(total_tp), 'FP': int(total_fp), 'FN': int(total_fn), 'GT_count': int(total_gt)}
        for name in metrics_to_calculate:
            if name in overall_metrics:
                 results['overall'][f"{name}_micro"] = overall_metrics[name]

        if should_process_confusion_matrix:
            self.logger.info(f"Raw Confusion Matrix data:\n{matrix}")
            results['confusion_matrix_data'] = matrix.tolist()
            
            if eval_output_dir and self.config and hasattr(self.config, 'CONFUSION_MATRIX_PLOT_NAME'):
                plot_path = eval_output_dir / self.config.CONFUSION_MATRIX_PLOT_NAME
                sorted_class_names = [self.class_names_map[i] for i in sorted(self.class_names_map.keys())]
                
                plot_readable_confusion_matrix(
                    matrix_data=matrix,
                    class_names=sorted_class_names,
                    output_path=plot_path,
                    title='Custom Evaluation Confusion Matrix'
                )
                results['confusion_matrix_plot_path_info'] = f"Plot saved to {plot_path}"
            else:
                results['confusion_matrix_plot_path_info'] = "Plot not saved due to missing output path or filename config."
        
        self.logger.info("Metrics computation complete.")
        return results