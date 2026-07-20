import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

def conservative_shadow_removal_enhancement(image_path, output_path):
    """
    Conservative shadow removal that preserves image quality while minimally addressing shadows
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
    
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
    return True

def process_hierarchical_dataset(input_dir, output_dir, progress_file="enhancement_progress.json"):
    """Process dataset with nested directory structure"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load progress if it exists
    progress_data = {}
    progress_file_path = Path(progress_file)
    if progress_file_path.exists():
        with open(progress_file_path, 'r') as f:
            progress_data = json.load(f)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    # Recursively process all directories
    def process_directory(current_input_dir, current_output_dir):
        nonlocal total_processed, total_skipped, total_failed
        
        # Create output directory
        current_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all items in current directory
        for item in current_input_dir.iterdir():
            if item.is_dir():
                # Recursively process subdirectory
                sub_output_dir = current_output_dir / item.name
                print(f"Processing directory: {item.relative_to(input_path)}")
                process_directory(item, sub_output_dir)
            elif item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                # Process image file
                output_file = current_output_dir / item.name
                
                # Check if already processed
                img_str = str(item.relative_to(input_path))
                if progress_data.get(img_str) == "completed":
                    total_skipped += 1
                    continue
                
                # Process the image
                success = conservative_shadow_removal_enhancement(item, output_file)
                
                if success:
                    progress_data[img_str] = "completed"
                    total_processed += 1
                    
                    # Save progress periodically
                    if total_processed % 50 == 0:
                        with open(progress_file_path, 'w') as f:
                            json.dump(progress_data, f)
                    
                    if total_processed % 10 == 0:  # Print every 10 images
                        print(f"Processed: {total_processed} | Skipped: {total_skipped} | Failed: {total_failed}")
                else:
                    total_failed += 1
                    print(f"Failed: {item.name}")
    
    # Start processing from root
    process_directory(input_path, output_path)
    
    # Save final progress
    with open(progress_file_path, 'w') as f:
        json.dump(progress_data, f)
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped (already done): {total_skipped}")
    print(f"Total images failed: {total_failed}")
    print(f"{'='*60}")
    
    return total_processed, total_skipped, total_failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Shadow Removal for Nested Dataset Structure")
    parser.add_argument("--input_dir", type=str, default="final_organized_dataset", help="Input directory with nested structure")
    parser.add_argument("--output_dir", type=str, default="final_dataset_enhanced_hierarchical", help="Output directory for enhanced images")
    parser.add_argument("--progress_file", type=str, default="hierarchical_enhancement_progress.json", help="Progress tracking file")
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"Starting hierarchical enhancement process at {start_time}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    process_hierarchical_dataset(args.input_dir, args.output_dir, args.progress_file)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nProcess completed at {end_time}")
    print(f"Total duration: {duration}")
