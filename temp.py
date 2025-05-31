# metrics_calculator.py
import logging
import torch
from pathlib import Path
from ultralytics.utils import metrics as ultralytics_metrics
from utils import plot_readable_confusion_matrix, calculate_prf1
import numpy as np

class DetectionMetricsCalculator:
    def __init__(self, class_names_map, logger=None, config_module=None):
        self.class_names_map = class_names_map
        self.num_classes = len(class_names_map)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config_module
        
        iou_threshold_for_cm = 0.45 
        if self.config and hasattr(self.config, 'BOX_MATCHING_IOU_THRESHOLD'):
            iou_threshold_for_cm = self.config.BOX_MATCHING_IOU_THRESHOLD
            self.logger.info(f"Using BOX_MATCHING_IOU_THRESHOLD: {iou_threshold_for_cm} for Ultralytics ConfusionMatrix.")
        else:
            self.logger.warning(
                f"BOX_MATCHING_IOU_THRESHOLD not found in config. Using default IoU {iou_threshold_for_cm} for Ultralytics ConfusionMatrix."
            )

        self.ultralytics_cm = ultralytics_metrics.ConfusionMatrix(
            nc=self.num_classes, 
            iou_thres=iou_threshold_for_cm 
        )
        self.logger.info(f"MetricsCalculator initialized for {self.num_classes} classes. Ultralytics CM is ready with IoU threshold: {iou_threshold_for_cm}.")

    def update_confusion_matrix(self, predictions_for_image, ground_truths_for_image):
        if not self.ultralytics_cm:
            self.logger.warning("Ultralytics CM not initialized, skipping CM update.")
            return

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

    # MODIFIED: Added plot_suffix parameter with a default value
    def compute_metrics(self, eval_output_dir: Path = None, plot_suffix: str = ""):
        results = {'per_class': {}, 'overall': {}, 'confusion_matrix_data': None, 'confusion_matrix_plot_path_info': None}
        
        matrix = self.ultralytics_cm.matrix
        if matrix is None:
            self.logger.error("Cannot compute metrics: Confusion Matrix has not been populated.")
            return results
        
        # Using the suffix in the log message if provided
        log_suffix_msg = f" ({plot_suffix.replace('_', ' ').strip()})" if plot_suffix else ""
        self.logger.info(f"Computing metrics from Ultralytics CM matrix{log_suffix_msg}.")


        total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0

        for i in range(self.num_classes):
            class_name = self.class_names_map.get(i, f"Unknown_ID_{i}")
            
            tp = matrix[i, i]
            fp = matrix[i, :].sum() - tp 
            fn = matrix[:, i].sum() - tp
            gt_count = tp + fn

            class_metrics_set = calculate_prf1(tp, fp, fn)
            results['per_class'][class_name] = {
                'TP': int(tp), 
                'FP': int(fp), 
                'FN': int(fn), 
                'GT_count': int(gt_count),
                'precision': class_metrics_set['precision'],
                'recall': class_metrics_set['recall'],
                'f1_score': class_metrics_set['f1_score']
            }
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += gt_count

        overall_metrics = calculate_prf1(total_tp, total_fp, total_fn)
        results['overall'] = {
            'TP': int(total_tp), 
            'FP': int(total_fp), 
            'FN': int(total_fn), 
            'GT_count': int(total_gt),
            'precision_micro': overall_metrics['precision'],
            'recall_micro': overall_metrics['recall'],
            'f1_score_micro': overall_metrics['f1_score']
        }

        self.logger.info(f"Raw Confusion Matrix data{log_suffix_msg}:\n{matrix}")
        results['confusion_matrix_data'] = matrix.tolist()
        
        # MODIFIED: Plot saving logic to incorporate plot_suffix
        if eval_output_dir and self.config and hasattr(self.config, 'CONFUSION_MATRIX_PLOT_NAME'):
            base_plot_name = Path(self.config.CONFUSION_MATRIX_PLOT_NAME)
            # Ensure plot_suffix, if provided, results in a clean modification (e.g., _before_processing)
            # The suffix should typically start with an underscore if it's meant to be a suffix.
            final_plot_filename = f"{base_plot_name.stem}{plot_suffix}{base_plot_name.suffix}"
            plot_path = eval_output_dir / final_plot_filename
            
            sorted_class_names = [self.class_names_map[i] for i in sorted(self.class_names_map.keys())]
            
            title_suffix_str = plot_suffix.replace("_", " ").strip().title()
            plot_title = f'Confusion Matrix {title_suffix_str}'.strip()
            if not title_suffix_str: # Default title if suffix is empty
                plot_title = 'Confusion Matrix (After Post-Processing)'


            plot_readable_confusion_matrix(
                matrix_data=matrix,
                class_names=sorted_class_names,
                output_path=plot_path,
                title=plot_title
            )
            results['confusion_matrix_plot_path_info'] = f"Plot saved to {plot_path}"
        else:
            results['confusion_matrix_plot_path_info'] = "Plot not saved: Output dir or plot name config missing."
        
        self.logger.info(f"Metrics computation complete{log_suffix_msg}.")
        return results