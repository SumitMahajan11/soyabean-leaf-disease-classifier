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
    
    # NEW: Improved background suppression while keeping the enhancement part intact
    # Create a leaf mask using edge detection and color segmentation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use a combination of techniques to create a better leaf mask
    # 1. Calculate saturation to identify colorful (leaf) regions
    hsv_for_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv_for_mask[:,:,1]
    
    # 2. Use edge detection to identify leaf boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Create initial mask based on saturation and luminance
    _, saturation_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
    _, luminance_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # 4. Combine masks to identify leaf areas
    combined_mask = cv2.bitwise_and(saturation_mask, luminance_mask)
    
    # 5. Use morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 6. Add edge information to refine the mask
    combined_mask = cv2.bitwise_or(combined_mask, edges)
    
    # 7. Apply Gaussian blur to mask edges for smoother transitions
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    combined_mask = combined_mask.astype(np.uint8)
    
    # 8. Create inverse mask for background
    inv_mask = 255 - combined_mask
    
    # 9. Create background with more aggressive blur
    background_blur = cv2.GaussianBlur(output_img, (15, 15), 0)
    
    # 10. Blend leaf area (original) with background (blurred) using the mask
    # Normalize the mask to range [0, 1] for blending
    mask_normalized = combined_mask.astype(np.float32) / 255.0
    
    # Apply the mask to blend original and blurred images
    output_img = (output_img * mask_normalized[:, :, np.newaxis] + 
                  background_blur * (1 - mask_normalized[:, :, np.newaxis])).astype(np.uint8)
    
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
    parser = argparse.ArgumentParser(description="Conservative Shadow Removal and Background Suppression for Full Soybean Disease Dataset")
    parser.add_argument("--input_dir", type=str, default="data/final_dataset", help="Input directory with images")
    parser.add_argument("--output_dir", type=str, default="data/final_dataset_enhanced", help="Output directory for enhanced images")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
    print("Full dataset conservative enhancement and background suppression completed!")