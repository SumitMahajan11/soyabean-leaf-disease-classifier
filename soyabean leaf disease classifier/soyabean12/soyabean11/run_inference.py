"""
Soybean Disease Detection - End-to-End Inference Script
Input Image → YOLO Detection (96.95% mAP) → Crop Detection → 
Enhanced Preprocessing → Enhanced EfficientNet Classification (98.14% accuracy) → JSON Output
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings('ignore')

def setup_device():
    """Setup device for inference (GPU if available)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_models(device):
    """Load both YOLO and EfficientNet models"""
    project_root = Path(__file__).parent
    
    # YOLO model path
    yolo_path = project_root / "experiments" / "runs" / "detect" / "yolo_soybean_disease_training" / "weights" / "best.pt"
    if not yolo_path.exists():
        # Use the path from the current backend system
        yolo_path = project_root / "src" / "backend" / "../../../experiments/runs/detect/yolo_soybean_disease_training/weights/best.pt"
        if not yolo_path.exists():
            # Look in the standard location used by the backend
            yolo_path = project_root / "experiments" / "runs" / "detect" / "yolo_soybean_disease_training" / "weights" / "best.pt"
            if not yolo_path.exists():
                raise FileNotFoundError(f"YOLO model not found at any expected location")
    
    # EfficientNet model path
    efficientnet_path = project_root / "models" / "CNN_trained_models" / "EnhancedEfficientNet" / "best_model_checkpoint.pth"
    if not efficientnet_path.exists():
        raise FileNotFoundError(f"Enhanced EfficientNet model not found at {efficientnet_path}")
    
    print("Loading YOLO model...")
    yolo_model = YOLO(str(yolo_path))
    yolo_model.to(device)
    
    print("Loading Enhanced EfficientNet model...")
    # Define the model architecture (same as training)
    from torchvision.models import efficientnet_b3
    classifier = efficientnet_b3(weights=None)
    # Modify the final layer for 12 soybean disease classes
    classifier.classifier[1] = torch.nn.Linear(classifier.classifier[1].in_features, 12)
    
    # Load the checkpoint
    checkpoint = torch.load(str(efficientnet_path), map_location=device, weights_only=False)
    # Extract the model state dict if it's wrapped in additional metadata
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # If the entire checkpoint is the state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load the state dict
    classifier.load_state_dict(state_dict, strict=False)
    classifier.to(device)
    classifier.eval()
    
    # Disease classes
    disease_classes = [
        'Bacterial Blight', 'Brown Spot', 'Caterpillar Pest', 'Ferrugen',
        'Healthy', 'Mosaic Virus', 'Rust', 'Septoria', 
        'Southern Blight', 'Sudden Death Syndrome', 'Vein Necrosis', 'Yellow Mosaic'
    ]
    
    return yolo_model, classifier, disease_classes, device

def preprocess_for_efficientnet():
    """Return the exact preprocessing transforms used during training"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])

def run_detection(yolo_model, image_path, device):
    """Run YOLO detection on the image"""
    print(f"Running YOLO detection on {image_path}")
    results = yolo_model(str(image_path), imgsz=512, conf=0.25, device=device)
    
    detections = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
            scores = result.boxes.conf.cpu().numpy()  # [N]
            classes = result.boxes.cls.cpu().numpy()  # [N]
            
            for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'detector_score': float(score),
                    'class_id': int(cls_id)
                })
    
    print(f"Found {len(detections)} detections")
    return detections

def crop_and_classify_region(classifier, original_image, bbox, preprocess_fn, device, disease_classes):
    """Crop the detected region and classify it with EfficientNet"""
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(original_image.width, int(x2)), min(original_image.height, int(y2))
    
    # Crop the region
    cropped_image = original_image.crop((x1, y1, x2, y2))
    
    # Preprocess for EfficientNet
    input_tensor = preprocess_fn(cropped_image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    # Run classification
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
    
    disease_name = disease_classes[predicted_idx]
    
    return {
        'disease': disease_name,
        'disease_confidence': confidence
    }

def run_end_to_end_inference(image_path):
    """Complete end-to-end inference pipeline"""
    # Setup
    device = setup_device()
    
    # Load models
    yolo_model, classifier, disease_classes, device = load_models(device)
    
    # Load image
    print(f"Loading image: {image_path}")
    original_image = Image.open(image_path).convert("RGB")
    
    # Run YOLO detection
    detections = run_detection(yolo_model, image_path, device)
    
    # Process each detection with EfficientNet classification
    results = []
    preprocess_fn = preprocess_for_efficientnet()
    
    for detection in detections:
        classification_result = crop_and_classify_region(
            classifier, original_image, 
            detection['bbox'], 
            preprocess_fn, device, 
            disease_classes
        )
        
        # Combine detection and classification results
        result = {
            'bbox': detection['bbox'],
            'detector_score': detection['detector_score'],
            'disease': classification_result['disease'],
            'disease_confidence': classification_result['disease_confidence']
        }
        results.append(result)
    
    # Prepare final output
    output = {
        'image': os.path.basename(image_path),
        'detections': results,
        'summary': {
            'total_detections': len(results),
            'model_performance': {
                'detector_accuracy': '96.95% mAP',
                'classifier_accuracy': '98.14%'
            }
        }
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description='Soybean Disease Detection - End-to-End Inference')
    parser.add_argument('image_path', help='Path to input image')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        sys.exit(1)
    
    try:
        result = run_end_to_end_inference(args.image_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during inference: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()