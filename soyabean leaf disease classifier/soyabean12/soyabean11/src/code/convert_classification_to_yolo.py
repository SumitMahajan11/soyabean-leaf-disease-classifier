"""
Convert classification dataset to YOLO detection format.
This script converts the soybean disease classification dataset to YOLO detection format
by creating bounding boxes that encompass the entire image for each class.
"""
import os
import shutil
from pathlib import Path
import yaml
from PIL import Image
import numpy as np


def create_yolo_dataset_structure(base_path):
    """Create the directory structure required for YOLO training."""
    yolo_path = Path(base_path) / "yolo_dataset"
    
    # Create directory structure
    (yolo_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    return yolo_path


def convert_classification_to_yolo_format(classification_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Convert hierarchical classification dataset to YOLO detection format."""
    classification_path = Path(classification_path)
    output_path = Path(output_path)
    
    # Get all leaf directories containing images
    all_leaf_dirs = []
    for root, dirs, files in os.walk(classification_path):
        if not dirs:  # This is a leaf directory
            # Check if it has images
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) for f in files):
                all_leaf_dirs.append(Path(root))
    
    class_names = [d.name for d in all_leaf_dirs]
    class_names.sort() # Ensure consistent order
    
    # Create class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Create YAML configuration for YOLO
    yaml_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the YAML config
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path / 'dataset.yaml'}")
    print(f"Classes: {class_names}")
    
    # Collect all images with their class labels
    all_images = []
    for class_dir in all_leaf_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                all_images.append((img_file, class_idx, class_name))
    
    print(f"Total images found: {len(all_images)}")
    
    # Shuffle the images
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    # Split the dataset
    total_images = len(all_images)
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)
    
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")
    
    # Process each split
    splits = {
        'train': train_images,
        'val': val_images, 
        'test': test_images
    }
    
    for split_name, split_images in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Create directories for this split if they don't exist
        (output_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        
        for img_path, class_idx, class_name in split_images:
            # Copy image to appropriate directory
            img_output_path = output_path / "images" / split_name / img_path.name
            shutil.copy2(img_path, img_output_path)
            
            # Create YOLO label file
            # For classification-like images, we create a bounding box that encompasses the entire image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # YOLO format: class_id center_x center_y width height (normalized)
            center_x = 0.5  # Center of image
            center_y = 0.5  # Center of image
            width = 0.9     # Almost full width (leaving small margin)
            height = 0.9    # Almost full height (leaving small margin)
            
            label_content = f"{class_idx} {center_x} {center_y} {width} {height}\n"
            
            # Create corresponding label file
            label_file = output_path / "labels" / split_name / f"{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write(label_content)
    
    print(f"\nDataset conversion completed!")
    print(f"Output directory: {output_path}")
    print(f"Classes: {len(class_names)} - {class_names}")
    
    return output_path / 'dataset.yaml'


if __name__ == "__main__":
    # Convert the final dataset to YOLO format
    classification_dataset_path = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    output_path = "e:/soyabean12/soyabean11/data/yolo_dataset"
    
    yaml_path = convert_classification_to_yolo_format(
        classification_path=classification_dataset_path,
        output_path=output_path
    )
    
    print(f"\nYOLO dataset created successfully!")
    print(f"Dataset config file: {yaml_path}")
    print("You can now use this dataset to train a YOLO model.")