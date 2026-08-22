"""
Test script for the production-ready soybean disease detection pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from production_ready_pipeline import ProductionSoybeanDiseasePipeline
import os

def test_pipeline():
    """Test the production pipeline with a sample image."""
    print("Initializing Production Pipeline...")
    
    # Initialize the pipeline with correct paths
    pipeline = ProductionSoybeanDiseasePipeline(
        yolo_weights="experiments/runs/detect/yolo_soybean_disease_training/weights/best.pt",
        model_dir="models/CNN_trained_models",
        yolo_imgsz=512,
        yolo_conf=0.25,
        disease_confidence_threshold=0.6,
    )
    
    print("\nPipeline initialized successfully!")
    print("Testing with a sample image from the dataset...")
    
    # Try to find a sample image from the dataset
    sample_paths = [
        "assets/data/yolo_dataset/images/val",  # YOLO format validation images
        "assets/data/final_dataset/val",        # Original format validation images
        "assets/data/final_dataset_enhanced/val", # Enhanced format validation images
    ]
    
    sample_image = None
    for path in sample_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                sample_image = os.path.join(path, files[0])
                break
    
    if sample_image:
        print(f"Testing with sample image: {sample_image}")
        try:
            result = pipeline.run_on_image(sample_image)
            print("\n✓ Pipeline executed successfully!")
            print(f"Image: {result['image_path']}")
            print(f"Total detections: {result['summary']['total_detected_regions']}")
            print(f"Disease counts: {result['summary']['disease_counts']}")
            print(f"Pipeline metrics: {result['pipeline_metrics']}")
            
            if result['detections']:
                first_detection = result['detections'][0]
                print(f"\nFirst detection details:")
                print(f"  Disease: {first_detection['disease_class']}")
                print(f"  Confidence: {first_detection['disease_confidence_percentage']}")
                print(f"  Bounding box: {first_detection['bbox']}")
                
        except Exception as e:
            print(f"✗ Error during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No sample images found in expected directories.")
        print("Please provide a path to a soybean disease image to test the pipeline.")
        print("\nExample usage:")
        print("result = pipeline.run_on_image('path/to/your_image.jpg')")
        print("print(result)")


if __name__ == "__main__":
    test_pipeline()