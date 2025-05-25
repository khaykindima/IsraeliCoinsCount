import logging
from pathlib import Path
import cv2
import csv

import config

from utils import (
    parse_yolo_annotations,
    calculate_iou, 
    draw_error_annotations
)
from detector import CoinDetector

class YoloEvaluator:
    def __init__(self, detector, logger, config_module=config):
        """
        Initializes the YoloEvaluator.
        Args:
            detector (CoinDetector): A pre-configured instance of CoinDetector.
            logger: A logger instance.
            config_module: The configuration module.
        """
        self.logger = logger
        self.config = config_module
        self.detector = detector

    def perform_detailed_evaluation(self, eval_output_dir, all_image_label_pairs_eval):
        # Get model path and class map from the detector instance
        model_path_to_eval = self.detector.model_path
        class_names_map_eval = self.detector.class_names_map

        self.logger.info(f"--- Starting Detailed Evaluation for Model: {model_path_to_eval} ---")
        self.logger.info(f"Evaluation outputs will be saved in: {eval_output_dir}")

        incorrect_dir = eval_output_dir / self.config.INCORRECT_PREDICTIONS_SUBDIR
        incorrect_dir.mkdir(parents=True, exist_ok=True)

        csv_rows = []
        error_count = 0

        for img_path, lbl_path in all_image_label_pairs_eval:
            self.logger.info(f"Processing: {img_path.name}")
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Cannot read image: {img_path}")
                continue

            preds = self.detector.predict(img.copy())
            gts_raw = parse_yolo_annotations(lbl_path, self.logger)
            h, w = img.shape[:2]

            gts = [{
                'cls': gt[0],
                'xyxy': [(gt[1] - gt[3]/2) * w, (gt[2] - gt[4]/2) * h,
                         (gt[1] + gt[3]/2) * w, (gt[2] + gt[4]/2) * h],
                'matched': False
            } for gt in gts_raw]

            tp, fp = 0, 0
            final_preds = []

            for p in sorted(preds, key=lambda x: x['conf'], reverse=True):
                status = "Incorrect (FP)"
                for i, gt in enumerate(gts):
                    if not gt['matched'] and p['cls'] == gt['cls']:
                        iou = calculate_iou(p['xyxy'], gt['xyxy'])
                        if iou > self.config.BOX_MATCHING_IOU_THRESHOLD:
                            gt['matched'] = True
                            tp += 1
                            status = "Correct (TP)"
                            break
                final_preds.append({**p, 'status': status})

            fn = len([g for g in gts if not g['matched']])
            fp = len(final_preds) - tp

            for p in final_preds:
                # This now correctly uses the class map from the instance
                class_name = class_names_map_eval.get(p['cls'], f"ID_{p['cls']}")
                csv_rows.append([img_path.name, class_name, f"{p['conf']:.4f}", p['status']])

            if fp > 0 or fn > 0:
                error_count += 1
                fp_preds = [p for p in final_preds if p['status'] == "Incorrect (FP)"]
                fn_gts = [g for g in gts if not g['matched']]

                annotated_img = draw_error_annotations(
                    img.copy(), fp_preds, fn_gts, class_names_map_eval,
                    self.config.BOX_COLOR_MAP, self.config.DEFAULT_BOX_COLOR,
                    self.logger
                )

                out_path = incorrect_dir / img_path.relative_to(self.config.INPUTS_DIR)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cv2.imwrite(str(out_path), annotated_img)
                    self.logger.info(f"Saved annotated error image to: {out_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save annotated image: {e}")

        self.logger.info(f"Total error images: {error_count}")
        if csv_rows:
            csv_path = eval_output_dir / self.config.PREDICTIONS_CSV_NAME
            try:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Image Filename', 'Predicted Class', 'Probability', 'Box Correctness'])
                    writer.writerows(csv_rows)
                self.logger.info(f"Saved prediction CSV to: {csv_path}")
            except Exception as e:
                self.logger.error(f"Failed to write CSV: {e}")

        self.logger.info(f"--- Evaluation Finished for Model: {model_path_to_eval} ---")
