from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from pathlib import Path
# Assuming utils.py is in the same directory or accessible in PYTHONPATH
try:
    from .utils import calculate_iou # For package structure
except ImportError:
    from utils import calculate_iou # For running as standalone script in same dir

class CoinDetector:
    def __init__(self, model_path, class_names_map, 
                 per_class_conf_thresholds=None, 
                 default_conf_thresh=0.25,
                 iou_suppression_threshold=0.4,
                 box_color_map=None,
                 default_box_color=(255,0,0) # Blue
                 ):
        """
        Initializes the CoinDetector.
        Args:
            model_path (str or Path): Path to the YOLO model file (.pt).
            class_names_map (dict): Dictionary mapping class ID (int) to class name (str).
                                    Example: {0: 'one', 1: 'two'}
            per_class_conf_thresholds (dict): Dict mapping lowercase class names to their specific confidence thresholds.
            default_conf_thresh (float): Default confidence threshold if a class is not in per_class_conf_thresholds and Low confidence threshold for initial model.predict().
            iou_suppression_threshold (float): IoU threshold for custom inter-class NMS.
            box_color_map (dict): Dictionary mapping lowercase class names to BGR color tuples for drawing.
            default_box_color (tuple): Default BGR color for classes not in box_color_map.
        """
        self.model = YOLO(model_path)
        self.class_names_map = class_names_map 
        self.per_class_conf_thresholds = per_class_conf_thresholds if per_class_conf_thresholds else {}
        self.default_conf_thresh = default_conf_thresh
        self.iou_suppression_threshold = iou_suppression_threshold
        self.box_color_map = box_color_map if box_color_map else {}
        self.default_box_color = default_box_color


    def _apply_per_class_confidence(self, raw_predictions_data):
        """Applies per-class confidence thresholds to raw predictions."""
        thresholded_predictions = []
        for pred_item in raw_predictions_data:
            class_name = self.class_names_map.get(pred_item['cls'], "").lower().strip()
            conf_thresh = self.per_class_conf_thresholds.get(class_name, self.default_conf_thresh)
            if pred_item['conf'] >= conf_thresh:
                thresholded_predictions.append(pred_item)
        return thresholded_predictions

    def _apply_custom_nms(self, predictions_data):
        """Applies custom inter-class Non-Maximum Suppression."""
        if not predictions_data:
            return []
        
        num_preds = len(predictions_data)
        suppressed_flags = [False] * num_preds
        
        # Sort by confidence to give higher confidence boxes priority
        # This helps in a greedy NMS approach where a higher confidence box suppresses a lower one
        sorted_indices = sorted(range(num_preds), key=lambda k: predictions_data[k]['conf'], reverse=True)

        for i_idx_in_sorted in range(num_preds):
            i = sorted_indices[i_idx_in_sorted] # Actual index in predictions_data
            if suppressed_flags[i]:
                continue
            for j_idx_in_sorted in range(i_idx_in_sorted + 1, num_preds):
                j = sorted_indices[j_idx_in_sorted] # Actual index in predictions_data
                if suppressed_flags[j]:
                    continue
                
                iou = calculate_iou(predictions_data[i]['xyxy'], predictions_data[j]['xyxy']) 
                
                if iou > self.iou_suppression_threshold:
                    # Since predictions_data[i] has higher or equal confidence (due to sorting),
                    # suppress predictions_data[j]
                    suppressed_flags[j] = True 
            
        final_predictions = [p for idx, p in enumerate(predictions_data) if not suppressed_flags[idx]]
        return final_predictions

    def predict(self, image_np_or_path): # Can accept NumPy array or image path
        """
        Performs prediction on an image with custom post-processing.
        Args:
            image_np_or_path (np.ndarray or str or Path): Image as NumPy array or path to image file.
        Returns:
            list: List of final predictions. Each prediction is a dict:
                  {'xyxy', 'conf', 'cls', 'class_name'}
        """
        raw_predictions_data = []
        # The model.predict source can be a NumPy array, path, URL, etc.
        pred_results_list = self.model.predict(source=image_np_or_path, 
                                               save=False, 
                                               verbose=False, 
                                               conf=self.default_conf_thresh)

        if pred_results_list and pred_results_list[0].boxes:
            r_boxes = pred_results_list[0].boxes
            for i in range(len(r_boxes)):
                raw_predictions_data.append({
                    'xyxy': r_boxes.xyxy[i].cpu().tolist(), # [x1, y1, x2, y2]
                    'conf': float(r_boxes.conf[i]),
                    'cls': int(r_boxes.cls[i])
                })
        
        thresholded_predictions = self._apply_per_class_confidence(raw_predictions_data)
        final_predictions_after_nms = self._apply_custom_nms(thresholded_predictions)

        # Add class names to the final predictions
        for pred in final_predictions_after_nms:
            pred['class_name'] = self.class_names_map.get(pred['cls'], f"ID_{pred['cls']}")
        
        return final_predictions_after_nms

    def draw_predictions_on_image(self, image_np, predictions_list):
        """
        Draws final predictions on an image (for deployment/visualization).
        Args:
            image_np (np.ndarray): Image to draw on.
            predictions_list (list): List of prediction dicts from self.predict().
        Returns:
            np.ndarray: Image with predictions drawn.
        """
        img_to_draw_on = image_np.copy() # Work on a copy
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if predictions_list:
            pred_font_scale = 1.5 # Font scale for prediction text
            pred_text_thickness = 4 # Thickness for text, must be an integer
            pred_box_thickness = 2 # Thickness for box lines, must be an integer

            for pred_data in predictions_list:
                x1, y1, x2, y2 = map(int, pred_data['xyxy'])
                class_name = pred_data['class_name']
                confidence = pred_data['conf']
                label = f"{class_name} {confidence:.2f}"
                
                color = self.box_color_map.get(class_name.lower().strip(), self.default_box_color)
                text_color_on_bg = (0,0,0)

                cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, pred_box_thickness)
                (text_w, text_h), baseline = cv2.getTextSize(label, font, pred_font_scale, pred_text_thickness)

                label_x_pos = x1
                label_y_pos = y1 - baseline - 3 
                if label_y_pos < text_h :
                    label_y_pos = y1 + text_h + baseline + 3

                cv2.rectangle(img_to_draw_on, (label_x_pos, label_y_pos - text_h - baseline), (label_x_pos + text_w, label_y_pos + baseline), color, -1)
                cv2.putText(img_to_draw_on, label, (label_x_pos, label_y_pos), font, pred_font_scale, text_color_on_bg, pred_text_thickness, cv2.LINE_AA)
        return img_to_draw_on
