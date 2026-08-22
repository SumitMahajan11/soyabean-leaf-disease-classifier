import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse

def shadow_remove_enhancement(image_path, output_path):
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Method 1: Shadow removal using illumination correction
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel to normalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge the LAB channels back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Method 2: Shadow removal using RGB channels
    # Identify shadow areas by analyzing the RGB channels
    b, g, r = cv2.split(img)
    
    # Calculate shadow mask based on luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    shadow_mask = luminance < 80  # Threshold for shadows
    
    # Enhance the shadow areas
    enhanced_img = img.copy()
    enhanced_img[shadow_mask] = cv2.convertScaleAbs(enhanced_img[shadow_mask], alpha=1.2, beta=20)
    
    # Method 3: Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced_img, 9, 75, 75)
    
    # Method 4: Enhance contrast and brightness
    # Convert to HSV for better control over brightness
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply adaptive histogram equalization to value channel
    clahe_v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe_v.apply(v)
    
    # Merge back and convert to BGR
    enhanced_hsv = cv2.merge([h, s, v])
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # Apply Gaussian blur to background to reduce distractions
    # Create a mask for the leaf (brighter, more saturated areas)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[:,:,1]
    
    # Create a mask that identifies the leaf area (bright and saturated)
    leaf_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    saturation_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)[1]
    
    # Combine masks
    combined_mask = cv2.bitwise_and(leaf_mask, saturation_mask)
    
    # Apply Gaussian blur to the background (inverse of leaf area)
    background_blur = cv2.GaussianBlur(result, (21, 21), 0)
    
    # Create inverse mask
    inv_mask = cv2.bitwise_not(combined_mask)
    
    # Create background with blur
    background_blur_rgb = cv2.cvtColor(background_blur, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Use the original image for leaf areas and blurred image for background
    output_img = result_rgb.copy()
    output_img[inv_mask > 0] = background_blur_rgb[inv_mask > 0]
    
    # Convert back to BGR for saving
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    # Save the result
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
                    shadow_remove_enhancement(img_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow Removal and Image Enhancement for Soybean Disease Dataset")
    parser.add_argument("--input_dir", type=str, default="data/temporary_dataset", help="Input directory with images")
    parser.add_argument("--output_dir", type=str, default="data/temporary_dataset_enhanced", help="Output directory for enhanced images")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
    print("Shadow removal and enhancement completed!")