"""
Complete YOLO Training Pipeline for Soybean Disease Detection
This script runs the complete pipeline: dataset conversion -> model training
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display its output"""
    print(f"\n{description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Run the complete YOLO training pipeline"""
    print("=" * 70)
    print("YOLO TRAINING PIPELINE FOR SOYBEAN DISEASE DETECTION")
    print("This will:")
    print("1. Convert classification dataset to YOLO format")
    print("2. Train YOLO model with GPU acceleration and 100 epochs")
    print("=" * 70)
    
    # Check if required packages are available
    try:
        import torch
        import ultralytics
        print("✓ Required packages are available")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return
    
    # Check if the final dataset exists
    final_dataset_path = Path("e:/soyabean11/soyabean11/data/final_dataset")
    if not final_dataset_path.exists():
        print(f"✗ Final dataset not found at {final_dataset_path}")
        return
    
    print(f"✓ Found dataset at {final_dataset_path}")
    print(f"  Classes: {len([d for d in final_dataset_path.iterdir() if d.is_dir()])} categories")
    
    # Step 1: Convert dataset to YOLO format
    print("\n" + "="*50)
    print("STEP 1: CONVERTING DATASET TO YOLO FORMAT")
    print("="*50)
    
    convert_script = "e:/soyabean11/soyabean11/code/convert_classification_to_yolo.py"
    success = run_command(f'python "{convert_script}"', "Converting classification dataset to YOLO format...")
    
    if not success:
        print("✗ Dataset conversion failed. Stopping pipeline.")
        return
    
    # Check if the converted dataset exists
    yolo_dataset_path = Path("e:/soyabean11/soyabean11/data/yolo_dataset/dataset.yaml")
    if not yolo_dataset_path.exists():
        print(f"✗ YOLO dataset not found at {yolo_dataset_path}")
        return
    
    print(f"✓ YOLO dataset created successfully at {yolo_dataset_path}")
    
    # Step 2: Train YOLO model
    print("\n" + "="*50)
    print("STEP 2: TRAINING YOLO MODEL")
    print("="*50)
    
    train_script = "e:/soyabean11/soyabean11/code/train_yolo_soybean.py"
    success = run_command(f'python "{train_script}"', "Training YOLO model...")
    
    if not success:
        print("✗ YOLO training failed.")
        return
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Training results are available in the 'runs/detect/yolo_soybean_disease_training' directory")
    print("The best model is saved as 'best.pt' in the weights subdirectory")
    print("\nYou can now use the trained model for inference on new images.")


if __name__ == "__main__":
    main()