from ultralytics import YOLO
import cv2
from pathlib import Path
from bbox_utils import calculate_iou, calculate_aspect_ratio 
import logging
from utils import convert_to_3channel_grayscale, get_adaptive_drawing_params
import torch
from torchvision.ops import nms
import numpy as np

class CoinDetector:
    def __init__(self, model_path, class_names_map, config_module,
                 per_class_conf_thresholds=None,
                 default_conf_thresh=0.25,
                 iou_suppression_threshold=0.4,
                 enable_aspect_ratio_filter=False, aspect_ratio_filter_threshold=2.5,
                 enable_per_class_confidence=True,
                 enable_custom_nms=True,
                 enable_grayscale_preprocessing_from_config=False # Default to True if not passed, or could raise error
                 ):
        """Initializes the CoinDetector."""
        self.model_path = Path(model_path)
        self.model = YOLO(self.model_path)
        self.class_names_map = class_names_map
        self.config = config_module # Store the entire config module
        self.per_class_conf_thresholds = per_class_conf_thresholds or {}
        self.default_conf_thresh = default_conf_thresh
        self.iou_suppression_threshold = iou_suppression_threshold
        self.enable_aspect_ratio_filter = enable_aspect_ratio_filter
        self.aspect_ratio_filter_threshold = aspect_ratio_filter_threshold
        self.enable_per_class_confidence = enable_per_class_confidence
        self.enable_custom_nms = enable_custom_nms
        self.logger = logging.getLogger(__name__) 
        self.enable_grayscale_preprocessing = enable_grayscale_preprocessing_from_config
        
        # Placeholders for current image dimensions
        self.current_image_height = None
        self.current_image_width = None

    def _apply_per_class_confidence(self, raw_predictions_data):
        """
        Applies per-class confidence thresholds if enabled.
        If enabled and a class is not in per_class_conf_thresholds, it raises an error.
        If disabled, this method is skipped, and initial model.predict thresholding is used.
        """
        if not self.enable_per_class_confidence:
            self.logger.info("Per-class confidence thresholding is disabled. Skipping this step.")
            return raw_predictions_data

        thresholded_predictions = []
        self.logger.info("Applying per-class confidence thresholds.")
        for pred_item in raw_predictions_data:
            class_name = self.class_names_map.get(pred_item['cls'], f"Unnamed_ID_{pred_item['cls']}").lower().strip()
            
            if class_name not in self.per_class_conf_thresholds:
                raise KeyError(
                    f"Confidence threshold not defined for class '{class_name}' (ID: {pred_item['cls']}) "
                    f"in PER_CLASS_CONF_THRESHOLDS (config.py), but per-class confidence is enabled."
                )
            conf_thresh_to_apply = self.per_class_conf_thresholds[class_name]
            
            if pred_item['conf'] >= conf_thresh_to_apply:
                thresholded_predictions.append(pred_item)
        
        original_count = len(raw_predictions_data)
        retained_count = len(thresholded_predictions)
        if original_count > 0:
            self.logger.info(f"Per-class confidence filtering: Retained {retained_count}/{original_count} predictions.")
        
        return thresholded_predictions
    
    def _check_for_cut_off_coins(self, predictions_data):
        """Logs a warning for any detected bounding box touching the image edges."""
        if not self.config.ENABLE_CUT_OFF_CHECK:
            return

        if self.current_image_height is None or self.current_image_width is None:
            self.logger.error("Image dimensions not set. Cannot check for cut-off coins.")
            return

        tolerance = self.config.CUT_OFF_TOLERANCE
        for pred in predictions_data:
            x1, y1, x2, y2 = pred['xyxy']
            
            is_cut_off = (
                x1 <= tolerance or
                y1 <= tolerance or
                x2 >= self.current_image_width - tolerance or
                y2 >= self.current_image_height - tolerance
            )
            
            if is_cut_off:
                class_name = self.class_names_map.get(pred['cls'], "Unknown")
                self.logger.warning(
                    f"Cut-off coin detected: A '{class_name.capitalize()}' coin is touching the image edge. "
                    f"Its detection may be unreliable."
                )

    def _apply_aspect_ratio_filter(self, predictions_data):
        """Filters predictions based on bounding box aspect ratio."""
        if not self.enable_aspect_ratio_filter:
            return predictions_data

        filtered_predictions = []
        for pred in predictions_data:
            aspect_ratio = calculate_aspect_ratio(pred['xyxy'])

            if aspect_ratio == 0.0:
                self.logger.debug(f"Skipping degenerate box (xyxy: {pred['xyxy']}) before aspect ratio check.")
                continue

            if aspect_ratio <= self.aspect_ratio_filter_threshold:
                filtered_predictions.append(pred)
            else:
                self.logger.debug(
                    f"Filtered out box due to aspect ratio: {aspect_ratio:.2f} "
                    f"(Threshold: {self.aspect_ratio_filter_threshold}). Box: {pred['xyxy']}"
                )
        
        original_count = len(predictions_data)
        filtered_count = len(filtered_predictions)
        if original_count > filtered_count:
             self.logger.info(f"Aspect ratio filter: Retained {filtered_count}/{original_count} predictions.")
        
        return filtered_predictions

    def _apply_custom_nms(self, predictions_data):
        """Applies custom inter-class Non-Maximum Suppression using torchvision.ops.nms."""
        if not self.enable_custom_nms:
            self.logger.info("Custom NMS is disabled. Skipping NMS.")
            return predictions_data
        if not predictions_data:
            return []

        # Extract boxes and scores into tensors
        boxes = torch.tensor([p['xyxy'] for p in predictions_data], dtype=torch.float32)
        scores = torch.tensor([p['conf'] for p in predictions_data], dtype=torch.float32)

        # Apply NMS
        indices_to_keep = nms(boxes, scores, self.iou_suppression_threshold)

        # Filter the original predictions list
        final_predictions = [predictions_data[i] for i in indices_to_keep]

        original_count = len(predictions_data)
        filtered_count = len(final_predictions)
        if original_count > filtered_count:
            self.logger.info(f"Custom NMS: Retained {filtered_count}/{original_count} predictions.")

        return final_predictions

    def _apply_postprocessing_pipeline(self, raw_predictions_data):
        """
        Applies the full custom post-processing pipeline to raw predictions.
        """
        self.logger.info("Starting custom post-processing pipeline...")
        
		# Step 1: Confidence Filtering
        processed_data = self._apply_per_class_confidence(raw_predictions_data)
        
        # Step 2: Check for Cut-off Coins (before filtering them out)
        self._check_for_cut_off_coins(processed_data)
        
        # Step 3: Aspect Ratio Filtering
        processed_data = self._apply_aspect_ratio_filter(processed_data)

        # Step 4: Non-Maximum Suppression
        processed_data = self._apply_custom_nms(processed_data)
        self.logger.info("Custom post-processing pipeline finished.")
		
        return processed_data

    def predict(self, image_np_or_path, return_raw=False):
        """
        Performs prediction on an image with custom pre and post-processing.
        Args:
            image_np_or_path (np.ndarray or str or Path): Image as NumPy array or path to image file.
            return_raw (bool): If True, returns both final and raw predictions.
        Returns:
            list or tuple: A list of final predictions, or a tuple of 
                           (final_predictions, raw_predictions) if return_raw is True.
        """
        # --- Image Preprocessing ---
        
        if self.enable_grayscale_preprocessing:
            self.logger.info("Grayscale preprocessing is ENABLED. Converting image to 3-channel grayscale.")
            # The convert_to_3channel_grayscale function from utils.py expects a NumPy array
            preprocessed_image_np = convert_to_3channel_grayscale(image_np_or_path, logger_instance=self.logger)
            if preprocessed_image_np is None:
                self.logger.error("Image preprocessing failed. Cannot proceed with prediction.")
                return ([], []) if return_raw else []
        else:
            preprocessed_image_np = image_np_or_path if isinstance(image_np_or_path, np.ndarray) else cv2.imread(str(image_np_or_path))

        # Store current image dimensions for post-processing checks
        self.current_image_height, self.current_image_width, _ = preprocessed_image_np.shape

        # Model Prediction
        pred_results_list = self.model.predict(source=preprocessed_image_np, 
                                            save=False, 
                                            verbose=False,
                                            conf=self.default_conf_thresh)
       
        raw_predictions_data = []
        if pred_results_list and pred_results_list[0].boxes:
            r_boxes = pred_results_list[0].boxes
            for i in range(len(r_boxes)):
                raw_predictions_data.append({
                    'xyxy': r_boxes.xyxy[i].cpu().tolist(), 
                    'conf': float(r_boxes.conf[i]),
                    'cls': int(r_boxes.cls[i])
                })
        
        # --- Custom Post-processing ---
        final_predictions = self._apply_postprocessing_pipeline(raw_predictions_data)
        
        # Add class names to the final predictions
        for pred in final_predictions:
            pred['class_name'] = self.class_names_map.get(pred['cls'], f"ID_{pred['cls']}")
        
        if return_raw:
            # 'raw_predictions_data' here refers to predictions after initial model.predict filtering
            # but before any of the CoinDetector's custom post-processing.
            return final_predictions, raw_predictions_data 
        else:
            return final_predictions 

    def draw_predictions_on_image(self, image_np, predictions_list):
        """Draws final predictions on an image using adaptive settings."""
        img_to_draw_on = image_np.copy()
        h, w, _ = img_to_draw_on.shape
        
        # Get adaptive parameters based on the image width
        params = get_adaptive_drawing_params(w, self.config)
        box_thickness = params['box_thickness']
        text_thickness = params['text_thickness']
        font_scale = params['inference_font_scale']
        font_face = self.config.FONT_FACE
        color_map = self.config.BOX_COLOR_MAP
        default_color = self.config.DEFAULT_BOX_COLOR

        for pred_data in predictions_list:
            x1, y1, x2, y2 = map(int, pred_data['xyxy'])
            class_name = pred_data['class_name']
            label = f"{class_name} {pred_data['conf']:.2f}"
            color = color_map.get(class_name.lower().strip(), default_color)

            cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, box_thickness)
            (text_w, text_h), _ = cv2.getTextSize(label, font_face, font_scale, text_thickness)
            cv2.rectangle(img_to_draw_on, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img_to_draw_on, label, (x1, y1 - 5), font_face, font_scale, (0,0,0), text_thickness)
            
        return img_to_draw_on