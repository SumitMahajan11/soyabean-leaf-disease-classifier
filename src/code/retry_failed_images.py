import cv2
import numpy as np
from pathlib import Path
import json

def find_missing_images(input_dir, output_dir, progress_file):
    """Find images that failed to process"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load progress file
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    failed_images = []
    processed_count = 0
    
    def check_directory(current_input_dir, current_output_dir):
        nonlocal processed_count
        for item in current_input_dir.iterdir():
            if item.is_dir():
                sub_output_dir = current_output_dir / item.name
                check_directory(item, sub_output_dir)
            elif item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                output_file = current_output_dir / item.name
                img_str = str(item.relative_to(input_path))
                
                # Check if marked as completed
                if progress_data.get(img_str) == "completed":
                    processed_count += 1
                    # Verify output file exists
                    if not output_file.exists():
                        failed_images.append((item, output_file, "marked_complete_but_missing"))
                else:
                    failed_images.append((item, output_file, "not_in_progress"))
    
    check_directory(input_path, output_path)
    
    print(f"Total images in progress file: {processed_count}")
    print(f"Failed/Missing images found: {len(failed_images)}")
    
    return failed_images

def conservative_shadow_removal_enhancement(image_path, output_path):
    """Conservative shadow removal that preserves image quality"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not read: {image_path}")
            return False
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Method 1: Gentle CLAHE
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        v_enhanced = clahe.apply(v)
        v_final = cv2.addWeighted(v, 0.9, v_enhanced, 0.1, 0)
        
        enhanced_hsv = cv2.merge([h, s, v_final])
        result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # Method 2: LAB adjustments
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        mean_l = np.mean(l)
        
        if mean_l < 100:
            l_enhanced = cv2.convertScaleAbs(l, alpha=1.05, beta=2)
            l_final = l_enhanced
        elif mean_l > 180:
            l_enhanced = cv2.convertScaleAbs(l, alpha=0.95, beta=0)
            l_final = l_enhanced
        else:
            l_final = l
        
        enhanced_lab = cv2.merge([l_final, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Method 3: Quality preservation
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        output_img = cv2.addWeighted(img_rgb, 0.95, result_rgb, 0.05, 0)
        
        # Background suppression
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_for_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv_for_mask[:,:,1]
        
        edges = cv2.Canny(gray, 50, 150)
        
        _, saturation_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
        _, luminance_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        combined_mask = cv2.bitwise_and(saturation_mask, luminance_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        combined_mask = cv2.bitwise_or(combined_mask, edges)
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        combined_mask = combined_mask.astype(np.uint8)
        
        background_blur = cv2.GaussianBlur(output_img, (15, 15), 0)
        mask_normalized = combined_mask.astype(np.float32) / 255.0
        
        output_img = (output_img * mask_normalized[:, :, np.newaxis] + 
                      background_blur * (1 - mask_normalized[:, :, np.newaxis])).astype(np.uint8)
        
        # Save result
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output_bgr)
        return True
    except Exception as e:
        print(f"❌ Error processing {image_path}: {str(e)}")
        return False

def retry_failed_images(input_dir, output_dir, progress_file):
    """Retry processing failed images"""
    print("Scanning for failed images...")
    failed_images = find_missing_images(input_dir, output_dir, progress_file)
    
    if not failed_images:
        print("✅ No failed images found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Retrying {len(failed_images)} failed images...")
    print(f"{'='*60}\n")
    
    # Load progress file
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    success_count = 0
    still_failed = []
    
    for idx, (input_file, output_file, reason) in enumerate(failed_images, 1):
        print(f"[{idx}/{len(failed_images)}] Retry: {input_file.name} ({reason})")
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Retry processing
        if conservative_shadow_removal_enhancement(input_file, output_file):
            img_str = str(input_file.relative_to(Path(input_dir)))
            progress_data[img_str] = "completed"
            success_count += 1
            print(f"  ✅ Success")
        else:
            still_failed.append((input_file, output_file, reason))
            print(f"  ❌ Still failed")
    
    # Save updated progress
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)
    
    print(f"\n{'='*60}")
    print(f"Retry Results:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Still failed: {len(still_failed)}")
    print(f"{'='*60}\n")
    
    if still_failed:
        print("Images still failing:")
        for input_file, _, reason in still_failed:
            print(f"  - {input_file}")

if __name__ == "__main__":
    input_dir = "e:/soyabean12/soyabean11/final_organized_dataset"
    output_dir = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    progress_file = "e:/soyabean12/soyabean11/hierarchical_enhancement_progress.json"
    
    retry_failed_images(input_dir, output_dir, progress_file)
