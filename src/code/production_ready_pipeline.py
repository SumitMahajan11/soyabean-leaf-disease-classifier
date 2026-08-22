"""
PRODUCTION-READY SOYBEAN DISEASE DETECTION & CLASSIFICATION PIPELINE

Complete workflow implementation:
Input Image → YOLO Detection (96.95% mAP) → Crop Detection → 
Enhanced Preprocessing → Enhanced EfficientNet Classification (98.14% accuracy) → JSON Output
"""

import torch
from pathlib import Path
from typing import List, Dict, Any
import json
from PIL import Image

from ultralytics import YOLO

# Import the existing production classifier
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced


class ProductionSoybeanDiseasePipeline:
    """Production-ready pipeline: YOLO leaf detection + Enhanced EfficientNet disease classification."""
    
    def __init__(
        self,
        yolo_weights: str = "experiments/runs/detect/yolo_soybean_disease_training/weights/best.pt",
        model_dir: str = "models/CNN_trained_models",
        yolo_imgsz: int = 512,
        yolo_conf: float = 0.25,
        disease_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialize the production pipeline with YOLO detector and Enhanced EfficientNet classifier.
        
        Args:
            yolo_weights: Path to trained YOLO weights for leaf/disease detection
            model_dir: Directory containing the EnhancedEfficientNet model
            yolo_imgsz: Input image size for YOLO
            yolo_conf: Confidence threshold for YOLO detections
            disease_confidence_threshold: Confidence threshold for disease predictions
        """
        self.yolo_weights = Path(yolo_weights)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf

        # Choose device for YOLO
        if torch.cuda.is_available():
            self.yolo_device = "0"  # first CUDA GPU
        else:
            self.yolo_device = "cpu"

        if not self.yolo_weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.yolo_weights}. "
                "Ensure the YOLO model is properly trained and saved."
            )

        # Load YOLO detector (96.95% mAP model)
        self.detector = YOLO(str(self.yolo_weights))
        print(f"✓ YOLO Detector Loaded (96.95% mAP)")
        print(f"  Model: {self.yolo_weights}")
        print(f"  Device: {self.yolo_device}")

        # Load Enhanced EfficientNet classifier (98.14% accuracy)
        self.classifier = SoybeanDiseaseClassifierEnhanced(
            model_dir=model_dir,
            confidence_threshold=disease_confidence_threshold,
        )
        print(f"✓ Enhanced EfficientNet Classifier Loaded (98.14% accuracy)")

    def _run_yolo(self, image_path: Path) -> List[Dict[str, Any]]:
        """Run YOLO on an image and return detection results."""
        results = self.detector(
            str(image_path), 
            imgsz=self.yolo_imgsz, 
            conf=self.yolo_conf, 
            device=self.yolo_device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()  # [N, 4]
            confs = r.boxes.conf.cpu().numpy()  # [N]
            classes = r.boxes.cls.cpu().numpy()  # [N]

            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, classes):
                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score),
                        "class_id": int(cls_id),
                        "class_name": "soybean_disease",  # trained to detect soybean diseases
                    }
                )

        return detections

    def _classify_crops(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Crop each detected region and classify disease using Enhanced EfficientNet."""
        results: List[Dict[str, Any]] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Ensure valid integer pixel coordinates
            crop_box = (
                max(0, int(x1)),
                max(0, int(y1)),
                max(0, int(x2)),
                max(0, int(y2)),
            )

            # Crop the detected region
            crop = image.crop(crop_box)

            # Use the predict_from_pil method for in-memory crops
            disease_pred = self.classifier.predict_from_pil(crop)

            results.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detector_class": det["class_name"],
                    "detector_score": det["score"],
                    "disease_class": disease_pred["predicted_class"],
                    "disease_confidence": disease_pred["confidence"],
                    "disease_confidence_percentage": disease_pred["confidence_percentage"],
                    "disease_confidence_level": disease_pred["confidence_level"],
                    "top_predictions": disease_pred["top_predictions"],
                }
            )

        return results

    def run_on_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run full detection+classification pipeline on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing complete detection and classification results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load original image once for cropping
        image = Image.open(image_path).convert("RGB")

        # 1) YOLO detection (96.95% mAP)
        raw_detections = self._run_yolo(image_path)

        if not raw_detections:
            return {
                "image_path": str(image_path),
                "detections": [],
                "summary": {
                    "total_detected_regions": 0,
                    "disease_counts": {},
                    "note": "No disease regions detected by YOLO.",
                },
                "pipeline_metrics": {
                    "detector_accuracy": "96.95% mAP",
                    "classifier_accuracy": "98.14%",
                    "status": "No detections"
                }
            }

        # 2) Enhanced EfficientNet disease classification (98.14% accuracy) on each crop
        detection_results = self._classify_crops(image, raw_detections)

        # 3) Build summary
        disease_counts: Dict[str, int] = {}
        for r in detection_results:
            disease = r["disease_class"]
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

        summary = {
            "total_detected_regions": len(detection_results),
            "disease_counts": disease_counts,
        }

        return {
            "image_path": str(image_path),
            "detections": detection_results,
            "summary": summary,
            "pipeline_metrics": {
                "detector_accuracy": "96.95% mAP",
                "classifier_accuracy": "98.14%",
                "status": "Completed"
            }
        }

    def run_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run the pipeline on multiple images.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            List of results for each image
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.run_on_image(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": img_path,
                    "error": str(e),
                    "status": "failed"
                })
        return results

    def export_result_json(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Export result to JSON file.
        
        Args:
            result: Result dictionary from run_on_image
            output_path: Path to save JSON output
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✓ Results exported to: {output_path}")


def demo_pipeline():
    """Demo function to show how to use the production pipeline."""
    print("\n" + "=" * 80)
    print("PRODUCTION-READY SOYBEAN DISEASE DETECTION & CLASSIFICATION PIPELINE")
    print("Input Image → YOLO Detection (96.95% mAP) → Crop Detection →")
    print("Enhanced Preprocessing → Enhanced EfficientNet Classification (98.14% accuracy) → JSON Output")
    print("=" * 80)

    # Initialize pipeline with correct paths
    pipeline = ProductionSoybeanDiseasePipeline(
        yolo_weights="experiments/runs/detect/yolo_soybean_disease_training/weights/best.pt",
        model_dir="models/CNN_trained_models",
        yolo_imgsz=512,
        yolo_conf=0.25,
        disease_confidence_threshold=0.6,
    )

    print("\nPipeline initialized successfully!")
    print("\nTo run on an image:")
    print("  from production_ready_pipeline import ProductionSoybeanDiseasePipeline")
    print("  pipeline = ProductionSoybeanDiseasePipeline()")
    print("  result = pipeline.run_on_image('path/to/your_image.jpg')")
    print("  print(result)")
    print("\nTo export results as JSON:")
    print("  pipeline.export_result_json(result, 'output_results.json')")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    demo_pipeline()