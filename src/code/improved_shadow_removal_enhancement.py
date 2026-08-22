import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse

def improved_shadow_removal_enhancement(image_path, output_path):
    """
    Improved shadow removal and enhancement with better balance to avoid over-enhancement
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Method 1: Gentle shadow removal using luminance adjustment
    # Convert to HSV for better control over brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply gentle CLAHE to value channel to normalize lighting
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    v_enhanced = clahe.apply(v)
    
    # Merge back and convert to BGR
    enhanced_hsv = cv2.merge([h, s, v_enhanced])
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # Method 2: Gentle contrast enhancement using alpha/beta adjustment
    # Calculate mean and standard deviation to adjust contrast gently
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate mean and std for luminance
    mean_l = np.mean(l)
    std_l = np.std(l)
    
    # Normalize luminance with gentle adjustments
    if std_l > 0:
        l_normalized = (l - mean_l) * (50.0 / std_l) + mean_l
        l_normalized = np.clip(l_normalized, 0, 255).astype(np.uint8)
    else:
        l_normalized = l
    
    # Merge LAB channels back
    enhanced_lab = cv2.merge([l_normalized, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Method 3: Identify and preserve leaf regions while reducing background noise
    # Create a leaf mask based on saturation and luminance
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    hsv_full = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    saturation = hsv_full[:,:,1]
    
    # Create a mask that identifies the leaf area (moderately saturated and bright)
    # Using adaptive thresholding for robustness
    leaf_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    
    # Refine the mask by considering saturation
    saturation_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Combine masks to identify leaf areas
    combined_mask = cv2.bitwise_and(leaf_mask, saturation_mask)
    
    # Apply slight Gaussian blur to background only (non-leaf areas)
    # Create inverse mask for background
    inv_mask = cv2.bitwise_not(combined_mask)
    
    # Apply gentle blur to background
    background_blur = cv2.GaussianBlur(result, (5, 5), 0)
    
    # Blend original and blurred based on mask
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    background_blur_rgb = cv2.cvtColor(background_blur, cv2.COLOR_BGR2RGB)
    
    # Use the original image for leaf areas and slightly blurred image for background
    output_img = result_rgb.copy()
    
    # Only apply background blur where the inverse mask is active
    for c in range(3):  # For each color channel
        output_img[:,:,c] = np.where(inv_mask > 0, background_blur_rgb[:,:,c], result_rgb[:,:,c])
    
    # Method 4: Final gentle color balance adjustment
    # Convert to float to prevent overflow
    output_img = output_img.astype(np.float32)
    
    # Calculate channel means for subtle color balance
    channel_means = [np.mean(output_img[:,:,i]) for i in range(3)]
    overall_mean = np.mean(channel_means)
    
    # Apply gentle color balance adjustment
    for i in range(3):
        adjustment = overall_mean / (channel_means[i] + 1e-6)  # Add small value to prevent division by zero
        # Limit adjustment to prevent over-correction
        adjustment = np.clip(adjustment, 0.9, 1.1)
        output_img[:,:,i] *= adjustment
    
    # Clip values to valid range and convert back to uint8
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    
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
                    improved_shadow_removal_enhancement(img_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Shadow Removal and Image Enhancement for Soybean Disease Dataset")
    parser.add_argument("--input_dir", type=str, default="data/temporary_dataset", help="Input directory with images")
    parser.add_argument("--output_dir", type=str, default="data/temporary_dataset_enhanced_v2", help="Output directory for enhanced images")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
    print("Improved shadow removal and enhancement completed!")