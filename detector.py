"""
detector.py

This module defines the `CoinDetector` class, which serves as the core wrapper
around the YOLO model. It handles making predictions and applies a custom
post-processing pipeline (e.g., per-class confidence, NMS, aspect ratio filtering)
and image quality checks.
"""
from ultralytics import YOLO
import cv2
from pathlib import Path
import logging
import torch
from torchvision.ops import nms
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from types import ModuleType

from bbox_utils import calculate_aspect_ratio 
from utils import (
    convert_to_3channel_grayscale, get_adaptive_drawing_params, 
    check_image_blur, check_image_darkness, check_sharp_angle
)

Prediction = Dict[str, Any]

class CoinDetector:
    """
    A wrapper for the YOLO model to provide a tailored prediction and
    post-processing pipeline for coin detection.
    """
    def __init__(self, 
                 model_path: Path, 
                 class_names_map: Dict[int, str], 
                 config_module: ModuleType,
                 logger: Optional[logging.Logger] = None,
                 per_class_conf_thresholds: Optional[Dict[str, float]] = None,
                 default_conf_thresh: float = 0.25,
                 iou_suppression_threshold: float = 0.4,
                 enable_aspect_ratio_filter: bool = False, 
                 aspect_ratio_filter_threshold: float = 2.5,
                 enable_per_class_confidence: bool = True,
                 enable_custom_nms: bool = True,
                 enable_grayscale_preprocessing_from_config: bool = False
                 ) -> None:
        """
        Initializes the CoinDetector.

        Args:
            model_path (Path): Path to the trained YOLO model (.pt file).
            class_names_map (dict): A map from class ID to class name.
            config_module (module): The project's config file.
            logger (logging.Logger, optional): Logger instance.
            ... (other parameters load from config_module).
        """
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.class_names_map = class_names_map
        self.config = config_module
        self.per_class_conf_thresholds = per_class_conf_thresholds or {}
        self.default_conf_thresh = default_conf_thresh
        self.iou_suppression_threshold = iou_suppression_threshold
        self.enable_aspect_ratio_filter = enable_aspect_ratio_filter
        self.aspect_ratio_filter_threshold = aspect_ratio_filter_threshold
        self.enable_per_class_confidence = enable_per_class_confidence
        self.enable_custom_nms = enable_custom_nms
        self.logger = logger if logger else logging.getLogger(__name__) 
        self.enable_grayscale_preprocessing = enable_grayscale_preprocessing_from_config
        
        self.current_image_height: Optional[int] = None
        self.current_image_width: Optional[int] = None

    def _apply_per_class_confidence(self, raw_predictions_data: List[Prediction]) -> List[Prediction]:
        """
        Filters predictions based on per-class confidence thresholds defined in config.
        """
        if not self.enable_per_class_confidence:
            # If not enabled, just return predictions filtered by the model's default confidence
            return [p for p in raw_predictions_data if p['conf'] >= self.default_conf_thresh]

        thresholded_predictions: List[Prediction] = []
        for pred_item in raw_predictions_data:
            class_name = self.class_names_map.get(pred_item['cls'], "").lower().strip()
            
            if class_name not in self.per_class_conf_thresholds:
                raise KeyError(
                    f"Confidence threshold not defined for class '{class_name}' in config.py"
                )
            conf_thresh_to_apply = self.per_class_conf_thresholds[class_name]
            
            if pred_item['conf'] >= conf_thresh_to_apply:
                thresholded_predictions.append(pred_item)
        

        original_count = len(raw_predictions_data)
        retained_count = len(thresholded_predictions)
        if original_count > retained_count:
            self.logger.info(f"Per-class confidence filtering: Retained {retained_count}/{original_count} predictions.")
    
        return thresholded_predictions
    
    def _check_for_cut_off_coins(self, predictions_data: List[Prediction]) -> None:
        """Logs a warning for any detected box touching the image edges."""
        if not self.config.ENABLE_CUT_OFF_CHECK:
            return

        if self.current_image_height is None or self.current_image_width is None:
            self.logger.error("Image dimensions not set. Cannot check for cut-off coins.")
            return

        tolerance = self.config.CUT_OFF_TOLERANCE
        for pred in predictions_data:
            x1, y1, x2, y2 = pred['xyxy']
            is_cut_off = (
                x1 <= tolerance or y1 <= tolerance or
                x2 >= self.current_image_width - tolerance or
                y2 >= self.current_image_height - tolerance
            )
            if is_cut_off:
                class_name = self.class_names_map.get(pred['cls'], "Unknown")
                self.logger.warning(f"Cut-off coin detected: A '{class_name}' is touching the image edge.")

    def _apply_aspect_ratio_filter(self, predictions_data: List[Prediction]) -> List[Prediction]:
        """Filters predictions based on bounding box aspect ratio."""
        if not self.enable_aspect_ratio_filter:
            return predictions_data

        filtered_predictions: List[Prediction] = []
        for pred in predictions_data:
            aspect_ratio = calculate_aspect_ratio(pred['xyxy'])
            if aspect_ratio == 0.0:
                continue
            if aspect_ratio <= self.aspect_ratio_filter_threshold:
                filtered_predictions.append(pred)
            else:
                self.logger.debug(f"Filtered out box due to high aspect ratio: {aspect_ratio:.2f}")
        
        original_count = len(predictions_data)
        retained_count = len(filtered_predictions)
        if original_count > retained_count:
             self.logger.info(f"Aspect ratio filter: Retained {retained_count}/{original_count} predictions.")
        
        return filtered_predictions

    def _apply_custom_nms(self, predictions_data: List[Prediction]) -> List[Prediction]:
        """Applies Non-Maximum Suppression (NMS) across all classes."""
        if not self.enable_custom_nms or not predictions_data:
            return predictions_data

        boxes = torch.tensor([p['xyxy'] for p in predictions_data], dtype=torch.float32)
        scores = torch.tensor([p['conf'] for p in predictions_data], dtype=torch.float32)
        
        indices_to_keep = nms(boxes, scores, self.iou_suppression_threshold)
        final_predictions = [predictions_data[i] for i in indices_to_keep]

        original_count = len(predictions_data)
        retained_count = len(final_predictions)
        if original_count > retained_count:
            self.logger.info(f"Custom NMS: Retained {retained_count}/{original_count} predictions.")

        return final_predictions

    def _run_all_quality_checks(self, image_np: np.ndarray, raw_predictions_data: List[Prediction], image_name: str) -> None:
        """Consolidated function to run all configured image quality checks."""
        if not image_name: return

        self.logger.debug(f"Running quality checks for {image_name}...")
        # Check for blurriness (requires image)
        if self.config.ENABLE_BLUR_DETECTION:
            check_image_blur(image_np, self.config.BLUR_DETECTION_THRESHOLD, self.logger, image_name)
        
        # Check for darkness (requires image)
        if self.config.ENABLE_DARKNESS_DETECTION:
            check_image_darkness(image_np, self.config.DARKNESS_DETECTION_THRESHOLD, self.logger, image_name)

        # Check for sharp angle (requires raw predictions)
        if self.config.ENABLE_SHARP_ANGLE_DETECTION:
            check_sharp_angle(
                raw_predictions_data,
                self.config.SHARP_ANGLE_AR_THRESHOLD,
                self.config.SHARP_ANGLE_MIN_PERCENTAGE,
                self.logger,
                image_name
            )
        # Check for cut-off coins (requires raw predictions)
        if self.config.ENABLE_CUT_OFF_CHECK:
            self._check_for_cut_off_coins(raw_predictions_data)

    def _apply_postprocessing_pipeline(self, raw_predictions_data: List[Prediction]) -> List[Prediction]:
        """
        Applies the full custom post-processing pipeline to raw predictions.
        """        
		# Step 1: Confidence Filtering
        processed_data = self._apply_per_class_confidence(raw_predictions_data)
        
        # Step 2: Aspect Ratio Filtering
        processed_data = self._apply_aspect_ratio_filter(processed_data)

        # Step 3: Non-Maximum Suppression
        processed_data = self._apply_custom_nms(processed_data)
		
        return processed_data

    def predict(self, 
                image_np_or_path: Union[np.ndarray, str, Path], 
                return_raw: bool = False, 
                image_name: str = ""
                ) -> Union[List[Prediction], Tuple[List[Prediction], List[Prediction]]]:
        """
        Performs prediction on an image with custom pre and post-processing.
        
        Args:
            image_np_or_path (np.ndarray or str or Path): Image to process.
            return_raw (bool): If True, returns predictions before and after post-processing.
            image_name (str, optional): The name of the image for logging purposes.
            
        Returns:
            list or tuple: A list of final predictions, or (final, raw) if return_raw is True.
        """
        # --- 1. Image Preprocessing ---
        preprocessed_image_np: Optional[np.ndarray] = None
        if self.enable_grayscale_preprocessing:
            # The convert_to_3channel_grayscale function from utils.py expects a NumPy array
            preprocessed_image_np = convert_to_3channel_grayscale(image_np_or_path, logger_instance=self.logger)
            if preprocessed_image_np is None:
                self.logger.error("Grayscale preprocessing failed.")
                return ([], []) if return_raw else []
        else:
            if isinstance(image_np_or_path, np.ndarray):
                preprocessed_image_np = image_np_or_path
            else:
                preprocessed_image_np = cv2.imread(str(image_np_or_path))

        if preprocessed_image_np is None:
            self.logger.error(f"Failed to load image for prediction: {image_name}")
            return ([], []) if return_raw else []

        self.current_image_height, self.current_image_width, _ = preprocessed_image_np.shape

        # --- 2. Model Prediction (Raw) ---
        pred_results = self.model.predict(source=preprocessed_image_np, 
                                            imgsz=self.config.IMG_SIZE,
                                            save=False, 
                                            verbose=False,
                                            conf=self.default_conf_thresh)
       
        raw_predictions_data: List[Prediction] = []
        if pred_results and pred_results[0].boxes:
            r_boxes = pred_results[0].boxes
            for i in range(len(r_boxes)):
                raw_predictions_data.append({
                    'xyxy': r_boxes.xyxy[i].cpu().tolist(), 
                    'conf': float(r_boxes.conf[i]),
                    'cls': int(r_boxes.cls[i])
                })
        
        # --- 3. Image Quality Checks ---
        self._run_all_quality_checks(preprocessed_image_np, raw_predictions_data, image_name)
        
        # --- 4. Custom Post-processing ---
        final_predictions = self._apply_postprocessing_pipeline(raw_predictions_data)
        
        # Add class names to the final predictions for convenience
        for pred in final_predictions:
            pred['class_name'] = self.class_names_map.get(pred['cls'], f"ID_{pred['cls']}")
        
        if return_raw:
            # 'raw_predictions_data' here refers to predictions after initial model.predict filtering
            # but before any of the CoinDetector's custom post-processing.
            return final_predictions, raw_predictions_data 
        else:
            return final_predictions 

    def draw_predictions_on_image(self, image_np: np.ndarray, predictions_list: list, show_confidence: bool = True) -> np.ndarray:
        """Draws final predictions on an image using adaptive settings."""
        img_to_draw_on = image_np.copy()
        h, w, _ = img_to_draw_on.shape
        params = get_adaptive_drawing_params(w, self.config)

        for pred_data in predictions_list:
            x1, y1, x2, y2 = map(int, pred_data['xyxy'])
            class_name = pred_data['class_name']

            # Use the new parameter to decide which label to create
            if show_confidence:
                label = f"{class_name} {pred_data['conf']:.2f}"
            else:
                label = class_name

            color = self.config.BOX_COLOR_MAP.get(class_name.lower().strip(), self.config.DEFAULT_BOX_COLOR)

            cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, int(params['box_thickness']))
            (text_w, text_h), _ = cv2.getTextSize(label, self.config.FONT_FACE, params['inference_font_scale'], int(params['text_thickness']))
            cv2.rectangle(img_to_draw_on, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img_to_draw_on, label, (x1, y1 - 5), self.config.FONT_FACE, params['inference_font_scale'], (0,0,0), int(params['text_thickness']))
            
        return img_to_draw_on