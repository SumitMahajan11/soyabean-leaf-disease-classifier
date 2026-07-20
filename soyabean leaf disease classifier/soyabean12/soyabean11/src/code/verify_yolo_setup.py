"""
Verify the YOLO dataset was created correctly and run a quick test training
"""
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch


def verify_dataset():
    """Verify that the YOLO dataset was created correctly"""
    dataset_path = Path("e:/soyabean11/soyabean11/data/yolo_dataset")
    
    print("Verifying YOLO dataset structure...")
    print(f"Dataset path: {dataset_path}")
    
    # Check if dataset.yaml exists
    yaml_path = dataset_path / "dataset.yaml"
    if not yaml_path.exists():
        print(f"✗ dataset.yaml not found at {yaml_path}")
        return False
    
    # Load and display dataset configuration
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Dataset config loaded from {yaml_path}")
    print(f"  Path: {config.get('path', 'N/A')}")
    print(f"  Train: {config.get('train', 'N/A')}")
    print(f"  Val: {config.get('val', 'N/A')}")
    print(f"  Test: {config.get('test', 'N/A')}")
    print(f"  Number of classes: {config.get('nc', 'N/A')}")
    print(f"  Class names: {config.get('names', 'N/A')}")
    
    # Check if image directories exist
    train_img_path = dataset_path / config.get('train', 'images/train')
    val_img_path = dataset_path / config.get('val', 'images/val')
    
    if not train_img_path.exists():
        print(f"✗ Train image directory not found: {train_img_path}")
        return False
    
    if not val_img_path.exists():
        print(f"✗ Validation image directory not found: {val_img_path}")
        return False
    
    # Count images in directories
    train_img_count = len(list(train_img_path.glob('*.[jJ][pP][gG]')) + 
                          list(train_img_path.glob('*.[pP][nN][gG]')) + 
                          list(train_img_path.glob('*.[jJ][pP][eE][gG]')))
    val_img_count = len(list(val_img_path.glob('*.[jJ][pP][gG]')) + 
                        list(val_img_path.glob('*.[pP][nN][gG]')) + 
                        list(val_img_path.glob('*.[jJ][pP][eE][gG]')))
    
    print(f"  Train images: {train_img_count}")
    print(f"  Validation images: {val_img_count}")
    
    return True


def test_model_loading():
    """Test if we can load the YOLO model"""
    print("\nTesting YOLO model loading...")
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Load a model
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        return None


def run_quick_training_test():
    """Run a quick training test with just a few epochs"""
    print("\nRunning quick training test (3 epochs)...")
    
    try:
        # Load the model
        model = YOLO('yolov8n.pt')
        
        # Start a short training run
        results = model.train(
            data="e:/soyabean11/soyabean11/data/yolo_dataset/dataset.yaml",
            epochs=3,  # Just 3 epochs for testing
            imgsz=320,  # Smaller image size for faster training
            batch=8,    # Smaller batch size
            device=0 if torch.cuda.is_available() else 'cpu',
            name='quick_test_training',
            plots=True,
            verbose=True
        )
        
        print("✓ Quick training test completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error during quick training test: {e}")
        return False


def main():
    print("="*60)
    print("VERIFYING YOLO DATASET AND TRAINING SETUP")
    print("="*60)
    
    # Verify dataset
    if not verify_dataset():
        print("✗ Dataset verification failed")
        return
    
    print("✓ Dataset verification passed")
    
    # Test model loading
    model = test_model_loading()
    if model is None:
        print("✗ Model loading test failed")
        return
    
    print("✓ Model loading test passed")
    
    # Run quick training test
    success = run_quick_training_test()
    if not success:
        print("✗ Quick training test failed")
        return
    
    print("✓ Quick training test passed")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The YOLO dataset and training setup are ready.")
    print("You can now run the full training with 100 epochs using:")
    print("  python code\\train_yolo_soybean.py")
    print("\nNote: Full training will take considerable time and resources.")


if __name__ == "__main__":
    main()