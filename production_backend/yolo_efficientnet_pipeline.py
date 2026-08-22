"""
YOLO + V2 Ensemble (EffNet-B4 + ResNet152) detection and classification pipeline.

- YOLO (Ultralytics) is used ONLY for leaf/plant detection in full images.
- V2 Ensemble Classifier is used ONLY for disease classification
  on cropped leaf images, via SoybeanDiseaseClassifierEnhanced.
"""

import torch
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

from ultralytics import YOLO

# Import the existing production classifier
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced


class LeafDetectionAndClassificationPipeline:
    """End-to-end pipeline: YOLO leaf detection + EfficientNet disease classification."""

    def __init__(
        self,
        yolo_weights: str = "runs/detect/train/weights/best.pt",
        model_dir: str = "CNN_trained_models",
        yolo_imgsz: int = 512,
        yolo_conf: float = 0.25,
        disease_confidence_threshold: float = 0.6,
    ) -> None:
        """Initialize YOLO detector and V2 Ensemble classifier.

        Args:
            yolo_weights: Path to trained YOLO weights (.pt) for leaf detection.
            model_dir: Directory containing the model weights.
            yolo_imgsz: Input image size for YOLO (square, e.g. 512).
            yolo_conf: Confidence threshold for YOLO detections.
            disease_confidence_threshold: Confidence threshold for Ensemble predictions.
        """
        self.yolo_weights = Path(yolo_weights)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf

        # Choose device for YOLO (and let EfficientNet pick its own device)
        if torch.cuda.is_available():
            self.yolo_device = "0"  # first CUDA GPU
        else:
            self.yolo_device = "cpu"

        if not self.yolo_weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.yolo_weights}. "
                "Train the leaf detector first or update the path."
            )

        # Load YOLO detector
        self.detector = YOLO(str(self.yolo_weights))

        # Load V2 Ensemble classifier
        self.classifier = SoybeanDiseaseClassifierEnhanced(
            model_dir=model_dir,
            confidence_threshold=disease_confidence_threshold,
        )

    def _run_yolo(self, image_path: Path) -> List[Dict[str, Any]]:
        """Run YOLO on an image with enhancements (TTA, Agnostic NMS)."""
        results = self.detector(
            str(image_path), 
            imgsz=self.yolo_imgsz, 
            conf=self.yolo_conf, 
            device=self.yolo_device,
            augment=True,       # Enable Test-Time Augmentation (TTA)
            agnostic_nms=True   # Class-agnostic NMS
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
                        "class_name": "leaf",  # single-class detector
                    }
                )
        
        # Sort detections by score descending
        detections.sort(key=lambda x: x["score"], reverse=True)
        return detections

    def _classify_crops(
        self, image: Image.Image, detections: List[Dict[str, Any]], padding_ratio: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Crop each detected leaf with padding and classify disease using V2 Ensemble."""
        results: List[Dict[str, Any]] = []
        img_w, img_h = image.size

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            # Add padding to the bounding box for better classification
            bw = x2 - x1
            bh = y2 - y1
            pad_w = bw * padding_ratio
            pad_h = bh * padding_ratio
            
            # Ensure valid integer pixel coordinates with padding
            crop_box = (
                max(0, int(x1 - pad_w)),
                max(0, int(y1 - pad_h)),
                min(img_w, int(x2 + pad_w)),
                min(img_h, int(y2 + pad_h)),
            )

            # Crop the leaf region
            crop = image.crop(crop_box)

            # Use the new predict_from_pil method for in-memory crops
            disease_pred = self.classifier.predict_from_pil(crop)

            results.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "padded_bbox": list(crop_box),
                    "detector_class": det["class_name"],
                    "detector_score": det["score"],
                    "disease_class": disease_pred["predicted_class"],
                    "disease_confidence": disease_pred["confidence"],
                    "disease_confidence_percentage": disease_pred["confidence_percentage"],
                    "disease_confidence_level": disease_pred["confidence_level"],
                    "top_predictions": disease_pred.get("top_predictions", [])
                }
            )

        return results

    def run_on_image(self, image_path: str) -> Dict[str, Any]:
        """Run full detection+classification pipeline on a single image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load original image once for cropping
        image = Image.open(image_path).convert("RGB")

        # 1) YOLO detection
        raw_detections = self._run_yolo(image_path)

        if not raw_detections:
            return {
                "image_path": str(image_path),
                "detections": [],
                "summary": {
                    "total_leaves": 0,
                    "disease_counts": {},
                    "note": "No leaves detected by YOLO.",
                },
            }

        # 2) EfficientNet disease classification on each crop
        leaf_results = self._classify_crops(image, raw_detections)

        # 3) Build summary
        disease_counts: Dict[str, int] = {}
        for r in leaf_results:
            disease = r["disease_class"]
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

        summary = {
            "total_leaves": len(leaf_results),
            "disease_counts": disease_counts,
        }

        return {
            "image_path": str(image_path),
            "detections": leaf_results,
            "summary": summary,
        }


def demo_pipeline():
    """Simple demo that prints results for a single image path."""
    print("\n" + "=" * 70)
    print("YOLO + Enhanced EfficientNet-B3 Detection + Classification Pipeline")
    print("=" * 70)

    # Update yolo_weights to your trained model path if different
    pipeline = LeafDetectionAndClassificationPipeline(
        yolo_weights="runs/detect/train/weights/best.pt",
        model_dir="CNN_trained_models",
        yolo_imgsz=512,
        yolo_conf=0.25,
        disease_confidence_threshold=0.6,
    )

    print("\nPipeline initialized. To run on an image, use:")
    print("  from yolo_efficientnet_pipeline import demo_pipeline, LeafDetectionAndClassificationPipeline")
    print("  pipeline = LeafDetectionAndClassificationPipeline()")
    print("  result = pipeline.run_on_image('path/to/your_image.jpg')")
    print("  print(result)\n")


if __name__ == "__main__":
    demo_pipeline()
