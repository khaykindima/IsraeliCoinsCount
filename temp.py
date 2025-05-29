# Modified DetectionMetricsCalculator in metrics_calculator.py

import logging
import torch
from pathlib import Path
from ultralytics.utils import metrics as ultralytics_metrics
from utils import plot_readable_confusion_matrix, calculate_prf1 #

class DetectionMetricsCalculator:
    """
    Calculates detection metrics like Precision, Recall, F1-score by deriving
    them from a confusion matrix.
    """
    def __init__(self, class_names_map, logger=None, config_module=None): #
        """
        Initializes the DetectionMetricsCalculator.
        Args:
            class_names_map (dict): Maps class IDs (int) to class names (str).
            logger (logging.Logger, optional): Logger instance.
            config_module (module, optional): The project's config module for settings
                                             like CONFUSION_MATRIX_PLOT_NAME and BOX_MATCHING_IOU_THRESHOLD.
        """
        self.class_names_map = class_names_map #
        self.num_classes = len(class_names_map) #
        self.logger = logger if logger else logging.getLogger(__name__) #
        self.config = config_module # Store config for plot name etc.
        
        # Get the IoU threshold from the config module for initializing the ConfusionMatrix
        # Default to a common value (e.g., 0.45, which is Ultralytics' default for CM) if not specified,
        # though your config.py already has BOX_MATCHING_IOU_THRESHOLD.
        iou_threshold_for_cm = 0.45 # Default fallback
        if self.config and hasattr(self.config, 'BOX_MATCHING_IOU_THRESHOLD'): #
            iou_threshold_for_cm = self.config.BOX_MATCHING_IOU_THRESHOLD #
            self.logger.info(f"Using BOX_MATCHING_IOU_THRESHOLD: {iou_threshold_for_cm} for Ultralytics ConfusionMatrix.") #
        else:
            self.logger.warning( #
                f"BOX_MATCHING_IOU_THRESHOLD not found in config. Using default IoU {iou_threshold_for_cm} for Ultralytics ConfusionMatrix." #
            )

        # Initialize Ultralytics Confusion Matrix with the specified IoU threshold.
        # The parameter name in Ultralytics is typically 'iou_thres'.
        self.ultralytics_cm = ultralytics_metrics.ConfusionMatrix(
            nc=self.num_classes, 
            iou_thres=iou_threshold_for_cm 
        )
        self.logger.info(f"MetricsCalculator initialized for {self.num_classes} classes. Ultralytics CM is ready with IoU threshold: {iou_threshold_for_cm}.") #

    def update_confusion_matrix(self, predictions_for_image, ground_truths_for_image): #
        # ... (rest of the method remains the same)
        if not self.ultralytics_cm: #
            self.logger.warning("Ultralytics CM not initialized, skipping CM update.") #
            return

        if predictions_for_image: #
            preds_list = [[*p['xyxy'], p['conf'], float(p['cls'])] for p in predictions_for_image] #
            preds_tensor = torch.tensor(preds_list, dtype=torch.float32) #
        else:
            preds_tensor = torch.empty((0, 6), dtype=torch.float32) #

        if ground_truths_for_image: #
            gt_bboxes_list = [gt['xyxy'] for gt in ground_truths_for_image] #
            gt_cls_list = [float(gt['cls']) for gt in ground_truths_for_image] #
            gt_bboxes_tensor = torch.tensor(gt_bboxes_list, dtype=torch.float32) #
            gt_cls_tensor = torch.tensor(gt_cls_list, dtype=torch.float32) #
        else:
            gt_bboxes_tensor = torch.empty((0, 4), dtype=torch.float32) #
            gt_cls_tensor = torch.empty((0,), dtype=torch.float32) #
        
        self.ultralytics_cm.process_batch(preds_tensor, gt_bboxes_tensor, gt_cls_tensor) #


    def compute_metrics(self, eval_output_dir: Path = None): #
        # ... (rest of the method remains the same)
        results = {'per_class': {}, 'overall': {}, 'confusion_matrix_data': None, 'confusion_matrix_plot_path_info': None} #
        
        matrix = self.ultralytics_cm.matrix #
        if matrix is None: #
            self.logger.error("Cannot compute metrics: Confusion Matrix has not been populated.") #
            return results #
        
        metrics_to_calculate = ['precision', 'recall', 'f1_score'] #
        self.logger.info(f"Computing base metrics: {metrics_to_calculate} and confusion matrix.") #

        total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0 #

        for i in range(self.num_classes): #
            class_name = self.class_names_map.get(i, f"Unknown_ID_{i}") #
            
            tp = matrix[i, i] #
            fp = matrix[i, :].sum() - tp #
            fn = matrix[:, i].sum() - tp #
            gt = tp + fn #

            class_metrics_set = calculate_prf1(tp, fp, fn) #
            results['per_class'][class_name] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'GT_count': int(gt)} #
            for metric_name in metrics_to_calculate: #
                if metric_name in class_metrics_set: #
                    results['per_class'][class_name][metric_name] = class_metrics_set[metric_name] #
            
            total_tp += tp; total_fp += fp; total_fn += fn; total_gt += gt #

        overall_metrics = calculate_prf1(total_tp, total_fp, total_fn) #
        results['overall'] = {'TP': int(total_tp), 'FP': int(total_fp), 'FN': int(total_fn), 'GT_count': int(total_gt)} #
        for name in metrics_to_calculate: #
            if name in overall_metrics: #
                 results['overall'][f"{name}_micro"] = overall_metrics[name] #

        self.logger.info(f"Raw Confusion Matrix data:\n{matrix}") #
        results['confusion_matrix_data'] = matrix.tolist() #
        
        if eval_output_dir and self.config and hasattr(self.config, 'CONFUSION_MATRIX_PLOT_NAME'): #
            plot_path = eval_output_dir / self.config.CONFUSION_MATRIX_PLOT_NAME #
            sorted_class_names = [self.class_names_map[i] for i in sorted(self.class_names_map.keys())] #
            
            plot_readable_confusion_matrix( #
                matrix_data=matrix, #
                class_names=sorted_class_names, #
                output_path=plot_path, #
                title='Custom Evaluation Confusion Matrix' #
            )
            results['confusion_matrix_plot_path_info'] = f"Plot saved to {plot_path}" #
        else:
            results['confusion_matrix_plot_path_info'] = "Plot not saved due to missing output path or filename config." #
        
        self.logger.info("Metrics computation complete.") #
        return results #