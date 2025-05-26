import logging

class DetectionMetricsCalculator:
    """
    Calculates and stores detection metrics like Precision, Recall, and F1-score.
    It can compute these metrics per class and overall (micro-averaged).
    """
    def __init__(self, class_names_map, logger=None):
        """
        Initializes the DetectionMetricsCalculator.
        Args:
            class_names_map (dict): A dictionary mapping class IDs (int) to 
                                    class names (str). Example: {0: 'cat', 1: 'dog'}
            logger (logging.Logger, optional): Logger instance for messages.
        """
        self.class_names_map = class_names_map
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Initialize stats for each class
        self.stats_per_class = {
            class_id: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0}
            for class_id in self.class_names_map.keys()
        }
        self.logger.info(f"MetricsCalculator initialized for {len(self.class_names_map)} classes.")

    def update_stats(self, class_id, tp_delta=0, fp_delta=0, fn_delta=0, gt_delta=0):
        """
        Updates the TP, FP, FN, and GT counts for a given class ID.
        Args:
            class_id (int): The class ID for which to update stats.
            tp_delta (int): Number of True Positives to add.
            fp_delta (int): Number of False Positives to add.
            fn_delta (int): Number of False Negatives to add.
            gt_delta (int): Number of Ground Truths to add for this class.
        """
        if class_id not in self.stats_per_class:
            # This case should ideally not be hit if class_names_map is comprehensive
            # and all class IDs encountered are in the map.
            self.logger.warning(f"Class ID {class_id} not in initial map. Adding to stats.")
            self.stats_per_class[class_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0}
            if class_id not in self.class_names_map: # Also add to map if truly new
                 self.class_names_map[class_id] = f"Unknown_ID_{class_id}"


        self.stats_per_class[class_id]['tp'] += tp_delta
        self.stats_per_class[class_id]['fp'] += fp_delta
        self.stats_per_class[class_id]['fn'] += fn_delta
        self.stats_per_class[class_id]['gt_count'] += gt_delta

    def _calculate_single_metric_set(self, tp, fp, fn, gt_count):
        """Helper to calculate P, R, F1 for a single set of TP, FP, FN."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall can be based on (TP+FN) or directly on gt_count if FN is accurately tracked
        # Using (TP+FN) is generally robust as long as FN accumulation is correct.
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Sanity check: recall_alt = tp / gt_count if gt_count > 0 else 0.0
        # self.logger.debug(f"Recall check: TP+FN based: {recall}, GT_count based: {recall_alt}")

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def compute_metrics(self, requested_metrics=None):
        """
        Computes the specified metrics based on the accumulated TP, FP, FN counts.
        Args:
            requested_metrics (list of str, optional): A list of metric names to compute.
                Supported: 'precision', 'recall', 'f1_score'.
                If None or empty, all supported metrics are computed.
        Returns:
            dict: A dictionary containing per-class and overall metrics.
                  Example: 
                  {
                      'per_class': {
                          'cat': {'TP': 10, ..., 'precision': 0.8, ...}, ...
                      },
                      'overall': {'TP': 20, ..., 'precision_micro': 0.75, ...}
                  }
        """
        if requested_metrics is None or not requested_metrics:
            # Default to all core metrics if none are specified
            metrics_to_calculate = ['precision', 'recall', 'f1_score']
        else:
            metrics_to_calculate = [m.lower() for m in requested_metrics]

        self.logger.info(f"Computing metrics: {metrics_to_calculate}")
        
        results = {'per_class': {}, 'overall': {}}
        total_tp_all_classes = 0
        total_fp_all_classes = 0
        total_fn_all_classes = 0
        total_gt_all_classes = 0

        for class_id, stats in self.stats_per_class.items():
            class_name = self.class_names_map.get(class_id, f"Unknown_ID_{class_id}")
            tp, fp, fn, gt_count = stats['tp'], stats['fp'], stats['fn'], stats['gt_count']
            
            class_metrics = self._calculate_single_metric_set(tp, fp, fn, gt_count)
            
            # Store raw counts and requested metrics
            results['per_class'][class_name] = {'TP': tp, 'FP': fp, 'FN': fn, 'GT_count': gt_count}
            for metric_name in metrics_to_calculate:
                if metric_name in class_metrics:
                    results['per_class'][class_name][metric_name] = class_metrics[metric_name]
            
            total_tp_all_classes += tp
            total_fp_all_classes += fp
            total_fn_all_classes += fn
            total_gt_all_classes += gt_count

        # Overall (Micro-averaged) Metrics
        overall_calculated_metrics = self._calculate_single_metric_set(
            total_tp_all_classes, total_fp_all_classes, total_fn_all_classes, total_gt_all_classes
        )
        
        results['overall'] = {
            'TP': total_tp_all_classes, 
            'FP': total_fp_all_classes, 
            'FN': total_fn_all_classes, 
            'GT_count': total_gt_all_classes
        }
        for metric_name in metrics_to_calculate:
            # For overall, keys are typically like 'precision_micro'
            overall_metric_key = f"{metric_name}_micro" 
            if metric_name in overall_calculated_metrics:
                 results['overall'][overall_metric_key] = overall_calculated_metrics[metric_name]

        self.logger.info("Metrics computation complete.")
        return results

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MetricsCalculatorTest")

    sample_class_map = {0: 'cat', 1: 'dog', 2: 'bird'}
    calculator = DetectionMetricsCalculator(sample_class_map, logger=logger)

    # Simulate some updates
    calculator.update_stats(class_id=0, tp_delta=10, fp_delta=2, fn_delta=3, gt_delta=13) # Cat
    calculator.update_stats(class_id=0, tp_delta=5, fp_delta=1, fn_delta=1, gt_delta=6)  # Cat (more data)
    calculator.update_stats(class_id=1, tp_delta=20, fp_delta=5, fn_delta=2, gt_delta=22) # Dog
    # Bird: 0 TPs, 0 FPs, 5 FNs (e.g., 5 bird GTs were missed)
    calculator.update_stats(class_id=2, fn_delta=5, gt_delta=5) 
    # Simulate an FP for a class not initially in map (should be handled)
    # calculator.update_stats(class_id=3, fp_delta=1) 


    all_metrics = calculator.compute_metrics()
    logger.info("\n--- All Calculated Metrics ---")
    logger.info(json.dumps(all_metrics, indent=4))

    requested = ['precision', 'f1_score']
    partial_metrics = calculator.compute_metrics(requested_metrics=requested)
    logger.info(f"\n--- Requested Metrics Only ({requested}) ---")
    logger.info(json.dumps(partial_metrics, indent=4))

    # Test case: No detections for a class with GTs
    empty_class_map = {0: 'ghost'}
    empty_calculator = DetectionMetricsCalculator(empty_class_map, logger)
    empty_calculator.update_stats(class_id=0, fn_delta=3, gt_delta=3)
    empty_metrics = empty_calculator.compute_metrics()
    logger.info("\n--- Empty Class Metrics ---")
    logger.info(json.dumps(empty_metrics, indent=4))