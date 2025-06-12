import numpy as np
from collections import defaultdict

def calculate_iou(box1_xyxy, box2_xyxy):
    """Calculates IoU of two bounding boxes [x1, y1, x2, y2]."""
    x1_i, y1_i = max(box1_xyxy[0], box2_xyxy[0]), max(box1_xyxy[1], box2_xyxy[1])
    x2_i, y2_i = min(box1_xyxy[2], box2_xyxy[2]), min(box1_xyxy[3], box2_xyxy[3])
    intersection_width = max(0, x2_i - x1_i)
    intersection_height = max(0, y2_i - y1_i)
    intersection_area = intersection_width * intersection_height
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_aspect_ratio(xyxy):
    """
    Calculates the aspect ratio of a bounding box (longer side / shorter side).

    Args:
        xyxy (list or tuple): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        float: The aspect ratio, or 0.0 if width or height is zero.
    """
    x1, y1, x2, y2 = xyxy
    width = float(x2 - x1)
    height = float(y2 - y1)

    if width <= 0 or height <= 0:
        return 0.0  # Or None, or handle as an error
    
    return max(width / height, height / width)


def match_predictions(predictions, ground_truths, iou_threshold, class_names_map):
    """
    Matches predictions to ground truths to determine TP, FP, and FN.

    This function provides a clear, step-by-step implementation of the matching logic,
    serving as the single point of truth for detailed analysis (e.g., before/after summaries).

    Args:
        predictions (list): A list of prediction dictionaries.
        ground_truths (list): A list of ground truth dictionaries.
        iou_threshold (float): The IoU threshold for a match to be considered valid.
        class_names_map (dict): A map from class ID to class name.

    Returns:
        tuple: A tuple containing:
            - dict: Per-class stats {'TP': count, 'FP': count, 'FN': count}.
            - list: Detailed list of True Positive predictions.
            - list: Detailed list of False Positive predictions.
            - list: Detailed list of False Negative ground truths.
    """
    stats = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in class_names_map.values()}
    tp_details = []
    fp_details = []
    fn_details = []

    # A copy of ground truths with a flag to track matching
    gt_matched_flags = [False] * len(ground_truths)
    
    # OPTIMIZATION: Group ground truths by class for faster lookups
    gt_by_class = defaultdict(list)
    for i, gt in enumerate(ground_truths):
        gt_by_class[gt['cls']].append((i, gt)) # Store original index and the gt object

    # Sort predictions by confidence score in descending order
    sorted_preds = sorted(predictions, key=lambda x: x['conf'], reverse=True)

    # --- Step 1: Identify True Positives and False Positives ---
    for pred in sorted_preds:
        pred_class_id = pred['cls']
        class_name = class_names_map.get(pred_class_id, f"ID_{pred_class_id}")
        pred['class_name'] = class_name # Add class name to all preds for convenience

        best_match_gt_idx = -1
        max_iou = -1

        # Find the best possible GT match for the current prediction
        # by only searching in the list of GTs for the predicted class.
        if pred_class_id in gt_by_class:
            for i, gt in gt_by_class[pred_class_id]:
                # Only consider GTs that haven't been matched yet
                if not gt_matched_flags[i]:
                    iou = calculate_iou(pred['xyxy'], gt['xyxy'])
                    if iou > max_iou:
                        max_iou = iou
                        best_match_gt_idx = i
        
        # If a good enough match is found, it's a True Positive
        if max_iou >= iou_threshold:
            stats[class_name]['TP'] += 1
            gt_matched_flags[best_match_gt_idx] = True  # Mark this GT as used
            pred['matched_gt_xyxy'] = ground_truths[best_match_gt_idx]['xyxy'] 
            tp_details.append(pred)
        else:
            stats[class_name]['FP'] += 1
            fp_details.append(pred)

    # --- Step 2: Identify False Negatives ---
    # Any ground truth that was not matched by any prediction is a False Negative.
    for i, is_matched in enumerate(gt_matched_flags):
        if not is_matched:
            gt = ground_truths[i]
            class_name = class_names_map.get(gt['cls'])
            gt['class_name'] = class_name
            stats[class_name]['FN'] += 1
            fn_details.append(gt)

    return stats, tp_details, fp_details, fn_details