import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse

def conservative_shadow_removal_enhancement(image_path, output_path):
    """
    Conservative shadow removal that preserves image quality while minimally addressing shadows
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Method 1: Very gentle shadow detection and correction
    # Convert to HSV for brightness adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply very gentle CLAHE to value channel to normalize lighting minimally
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    v_enhanced = clahe.apply(v)
    
    # Blend original and enhanced value channels very gently
    # Only apply 10% of the enhancement to preserve original quality
    v_final = cv2.addWeighted(v, 0.9, v_enhanced, 0.1, 0)
    
    # Merge back and convert to BGR
    enhanced_hsv = cv2.merge([h, s, v_final])
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # Method 2: Minimal color space adjustments to preserve quality
    # Convert to LAB for subtle luminance adjustments
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate mean luminance
    mean_l = np.mean(l)
    
    # Apply very subtle luminance normalization
    # Only adjust if significantly different from optimal range
    if mean_l < 100:  # If image is too dark
        # Very gentle brightening
        l_enhanced = cv2.convertScaleAbs(l, alpha=1.05, beta=2)
        l_final = l_enhanced
    elif mean_l > 180:  # If image is too bright
        # Very gentle darkening
        l_enhanced = cv2.convertScaleAbs(l, alpha=0.95, beta=0)
        l_final = l_enhanced
    else:
        l_final = l  # Keep original if already in good range
    
    # Merge LAB channels back
    enhanced_lab = cv2.merge([l_final, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Method 3: Preserve original image quality by blending
    # Blend the enhanced image very minimally with the original
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Blend with 95% original and 5% enhanced to preserve quality
    output_img = cv2.addWeighted(img_rgb, 0.95, result_rgb, 0.05, 0)
    
    # Save the result
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), output_bgr)
    print(f"Processed: {image_path} -> {output_path}")

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each category directory
    for category_dir in input_path.iterdir():
        if category_dir.is_dir():
            category_output_dir = output_path / category_dir.name
            category_output_dir.mkdir(exist_ok=True)
            
            print(f"Processing category: {category_dir.name}")
            
            # Process each image in the category
            for img_file in category_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    output_file = category_output_dir / img_file.name
                    conservative_shadow_removal_enhancement(img_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conservative Shadow Removal and Image Enhancement for Soybean Disease Dataset")
    parser.add_argument("--input_dir", type=str, default="data/temporary_dataset", help="Input directory with images")
    parser.add_argument("--output_dir", type=str, default="data/temporary_dataset_enhanced_v3", help="Output directory for enhanced images")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
    print("Conservative shadow removal and enhancement completed!")