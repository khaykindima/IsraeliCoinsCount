from ultralytics import YOLO
import cv2
from pathlib import Path
from bbox_utils import calculate_iou

class CoinDetector:
    def __init__(self, model_path, class_names_map,
                 per_class_conf_thresholds=None,
                 default_conf_thresh=0.25,
                 iou_suppression_threshold=0.4,
                 box_color_map=None, default_box_color=(255,0,0),
                 # ADDED: Drawing parameters
                 box_thickness=2, text_thickness=2, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0
                 ):
        """Initializes the CoinDetector."""
        self.model_path = Path(model_path)
        self.model = YOLO(self.model_path)
        self.class_names_map = class_names_map
        self.per_class_conf_thresholds = per_class_conf_thresholds or {}
        self.default_conf_thresh = default_conf_thresh
        self.iou_suppression_threshold = iou_suppression_threshold
        # Store drawing parameters
        self.box_color_map = box_color_map or {}
        self.default_box_color = default_box_color
        self.box_thickness = box_thickness
        self.text_thickness = text_thickness
        self.font_face = font_face
        self.font_scale = font_scale


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

    def predict(self, image_np_or_path, return_raw=False):
        """
        Performs prediction on an image with custom post-processing.
        Args:
            image_np_or_path (np.ndarray or str or Path): Image as NumPy array or path to image file.
            return_raw (bool): If True, returns both final and raw predictions.
        Returns:
            list or tuple: A list of final predictions, or a tuple of 
                           (final_predictions, raw_predictions) if return_raw is True.
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
        
        if return_raw:
            return final_predictions_after_nms, raw_predictions_data
        else:
            return final_predictions_after_nms

    def draw_predictions_on_image(self, image_np, predictions_list):
        """Draws final predictions on an image using configured settings."""
        img_to_draw_on = image_np.copy()
        for pred_data in predictions_list:
            x1, y1, x2, y2 = map(int, pred_data['xyxy'])
            class_name = pred_data['class_name']
            label = f"{class_name} {pred_data['conf']:.2f}"
            color = self.box_color_map.get(class_name.lower().strip(), self.default_box_color)

            # MODIFIED: Use stored drawing parameters
            cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, self.box_thickness)
            (text_w, text_h), _ = cv2.getTextSize(label, self.font_face, self.font_scale, self.text_thickness)
            cv2.rectangle(img_to_draw_on, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img_to_draw_on, label, (x1, y1 - 5), self.font_face, self.font_scale, (0,0,0), self.text_thickness)
            
        return img_to_draw_on