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