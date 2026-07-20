"""
YOLO Training Script for Soybean Disease Detection
Trains a YOLO model on the soybean disease dataset with GPU acceleration and at least 50 epochs.
"""
from ultralytics import YOLO
import torch
import os
from pathlib import Path


def train_yolo_model(data_yaml_path, epochs=100, imgsz=320, batch_size=4):
    """
    Train YOLO model on soybean disease dataset
    
    Args:
        data_yaml_path (str): Path to the dataset YAML configuration file
        epochs (int): Number of training epochs (default: 100, at least 50 as requested)
        imgsz (int): Image size for training (default: 512)
        batch_size (int): Batch size for training (default: 16)
    """
    print("="*60)
    print("STARTING YOLO TRAINING FOR SOYBEAN DISEASE DETECTION")
    print("="*60)
    
    # Check GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA not available. Training will use CPU (much slower).")
    
    # Use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Load a model (using YOLOv8n as a starting point)
    print("\nLoading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # You can change this to yolov8s.pt, yolov8m.pt, etc. for different model sizes
    
    print(f"\nStarting training with:")
    print(f"- Dataset: {data_yaml_path}")
    print(f"- Epochs: {epochs}")
    print(f"- Image size: {imgsz}")
    print(f"- Batch size: {batch_size} (reduced for 6GB GPU memory)")
    print(f"- Device: {device}")
    
    # Start training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        name='yolo_soybean_disease_training',
        patience=50,  # Number of epochs to wait for improvement before early stopping
        plots=True,   # Generate training plots
        verbose=True  # Show detailed training progress
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {results.save_dir}")
    
    # Validate the model
    print("\nValidating the trained model...")
    validation_results = model.val()
    print(f"Validation results: {validation_results}")
    
    # Save the best model path for reference
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_model_path.exists():
        print(f"\nBest model saved at: {best_model_path}")
        print("You can use this model for inference with:")
        print(f"  model = YOLO('{best_model_path}')")
    
    return model


def main():
    """Main function to run YOLO training"""
    # Path to the converted YOLO dataset
    data_yaml_path = "e:/soyabean12/soyabean11/data/yolo_dataset/dataset.yaml"
    
    # Check if the dataset exists
    if not Path(data_yaml_path).exists():
        print(f"ERROR: Dataset configuration not found at {data_yaml_path}")
        print("Please run convert_classification_to_yolo.py first to convert the dataset.")
        return
    
    # Train the YOLO model with 150 epochs as requested
    trained_model = train_yolo_model(
        data_yaml_path=data_yaml_path,
        epochs=150,  # Increased epochs for better accuracy
        imgsz=512,   # Improved resolution
        batch_size=16 # Standard batch size for modern GPU
    )
    
    print("\n" + "="*60)
    print("YOLO TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Next steps:")
    print("1. Check the training results in the 'runs/detect/yolo_soybean_disease_training' directory")
    print("2. Use the trained model for inference on new images")
    print("3. Evaluate the model performance on your test set")


if __name__ == "__main__":
    main()