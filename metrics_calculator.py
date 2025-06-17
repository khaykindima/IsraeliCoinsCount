import logging
import torch
import numpy as np
from pathlib import Path
from ultralytics.utils import metrics as ultralytics_metrics
from typing import List, Dict, Any, Optional
from types import ModuleType

from utils import plot_readable_confusion_matrix, calculate_prf1

class DetectionMetricsCalculator:
    """
    Calculates detection metrics like Precision, Recall, F1-score by deriving
    them from a confusion matrix.
    """
    def __init__(self, class_names_map: Dict[int, str], logger: Optional[logging.Logger] = None, config_module: Optional[ModuleType] = None) -> None:
        """
        Initializes the DetectionMetricsCalculator.
        Args:
            class_names_map (dict): Maps class IDs (int) to class names (str).
            logger (logging.Logger, optional): Logger instance.
            config_module (module, optional): The project's config module for settings
                                             like CONFUSION_MATRIX_PLOT_NAME and BOX_MATCHING_IOU_THRESHOLD.
        """
        self.class_names_map = class_names_map
        self.num_classes = len(class_names_map)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config_module # Store config for plot name etc.
        
        # Get the IoU threshold from the config module for initializing the ConfusionMatrix
        # Default to a common value (e.g., 0.45, which is Ultralytics' default for CM) if not specified,
        # though your config.py already has BOX_MATCHING_IOU_THRESHOLD.
        iou_threshold_for_cm = 0.45 # Default fallback
        if self.config and hasattr(self.config, 'BOX_MATCHING_IOU_THRESHOLD'):
            iou_threshold_for_cm = self.config.BOX_MATCHING_IOU_THRESHOLD
            self.logger.info(f"Using BOX_MATCHING_IOU_THRESHOLD: {iou_threshold_for_cm} for Ultralytics ConfusionMatrix.")
        else:
            self.logger.warning( #
                f"BOX_MATCHING_IOU_THRESHOLD not found in config. Using default IoU {iou_threshold_for_cm} for Ultralytics ConfusionMatrix."
            )

        # Initialize Ultralytics Confusion Matrix. It is now the single source of truth for metrics.
        self.ultralytics_cm = ultralytics_metrics.ConfusionMatrix(
            nc=self.num_classes, 
            iou_thres=iou_threshold_for_cm 
        )
        self.logger.info(f"MetricsCalculator initialized for {self.num_classes} classes. Ultralytics CM is ready with IoU threshold: {iou_threshold_for_cm}.")

    def update_confusion_matrix(self, predictions_for_image: List[Dict[str, Any]], ground_truths_for_image: List[Dict[str, Any]]) -> None:
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


    def compute_metrics(self, eval_output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Computes all metrics by deriving stats from the confusion matrix and handles plotting.
        Args:
            eval_output_dir (Path, optional): Directory to save plots.
        """
        results: Dict[str, Any] = {'per_class': {}, 'overall': {}, 'confusion_matrix_data': None, 'confusion_matrix_plot_path_info': None}
        
        matrix: Optional[np.ndarray] = self.ultralytics_cm.matrix
        if matrix is None:
            self.logger.error("Cannot compute metrics: Confusion Matrix has not been populated.")
            return results
        
        total_tp, total_fp, total_fn, total_gt = 0.0, 0.0, 0.0, 0.0

        # Derive TP, FP, FN for each class from the matrix
        for i in range(self.num_classes):
            class_name = self.class_names_map.get(i, f"Unknown_ID_{i}")
            
            tp = matrix[i, i]
            fp = matrix[i, :].sum() - tp
            fn = matrix[:, i].sum() - tp
            gt = tp + fn

            # Use the centralized utility function
            class_metrics_set = calculate_prf1(tp, fp, fn)
            results['per_class'][class_name] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'GT_count': int(gt), **class_metrics_set}
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += gt

        overall_metrics = calculate_prf1(total_tp, total_fp, total_fn)
        results['overall'] = {
            'TP': int(total_tp), 'FP': int(total_fp), 'FN': int(total_fn), 'GT_count': int(total_gt),
            'precision_micro': overall_metrics['precision'],
            'recall_micro': overall_metrics['recall'],
            'f1_score_micro': overall_metrics['f1_score']
        }

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